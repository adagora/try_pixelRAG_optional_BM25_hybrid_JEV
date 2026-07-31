#!/usr/bin/env python
"""Profile the query-encode path — where the retrieval latency actually is.

Three views, because each answers a different question and the cheap one alone
has misled this project before:

  --mode cprofile   Python-level cumulative time. Cheap, and enough to tell
                    tokenise/template overhead from the forward pass.
  --mode torch      Per-operator time (torch.profiler). This is the view that
                    names the bottleneck op rather than the Python frame that
                    happens to call it.
  --mode phases     Wall-clock split of template / tokenise / forward / pool /
                    transfer. The one to trust for "did my change help",
                    because it needs no profiler overhead to be accurate.

CUDA timings are synchronised before each boundary; without that, `forward`
reads as instant and the cost lands on whatever touches the tensor next.

    python scripts/profile_search.py --mode phases --device cuda
    python scripts/profile_search.py --mode torch --device cuda --top 20

--attn / --tokenise / --compile exist to compare encoder configurations without
touching the shipped encoder. Defaults reproduce what rag.py and encoder.py do,
so a run with no flags is still the baseline. Warmup call timings are printed
individually: for --compile that first number *is* the compile cost, and it is
the number that decides whether compiling is worth it for a sidecar.
"""

import argparse
import cProfile
import io
import pstats
import statistics
import time
from pathlib import Path

import yaml

import tokenise_cache

DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."
MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def load(device: str, dtype_name: str, attn: str = "sdpa", tokenise: str = "stock",
         compile_mode: str = "off", pad_to: int = 0):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[dtype_name]
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=dtype, attn_implementation=attn).eval().to(device)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print(f"loaded {MODEL} on {device}/{dtype_name} attn={attn} "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)

    if tokenise == "memo":
        tokenise_cache.memoise_strict_classes()
    elif tokenise == "fast":
        # Checked against an untouched processor rather than trusted: the fast
        # path is only worth having if it is provably the same tokenisation.
        stock = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        tokenise_cache.cache_merged_kwargs(processor)
        tokenise_cache.assert_identical(stock, processor, queries(4))

    if compile_mode == "cudagraphs":
        # The dynamo backend that needs no Triton, which matters because there is
        # no Triton for Windows in this env: every inductor path (mode=default,
        # reduce-overhead, max-autotune) dies with TritonMissing. This backend
        # only replays the launch sequence as a graph, which is the cost that
        # actually dominates here.
        #
        # dynamic=True is what a variable-length query needs and is also what
        # makes this pointless: dynamo then refuses to capture ("skipping
        # cudagraphs due to cpu device") and the 28s warmup buys nothing. With
        # --pad-to every forward is one shape, so compile static and get a graph.
        model.model = torch.compile(model.model, backend="cudagraphs",
                                    dynamic=not pad_to)
    elif compile_mode != "off":
        # The forward is what we want compiled; the pooling around it is three ops.
        # dynamic=True because query length varies per request — without it every
        # new sequence length is a fresh compile on the request path.
        model.model = torch.compile(model.model, mode=compile_mode, dynamic=True)
    return torch, model, processor


def queries(n: int) -> list[str]:
    qs = [q["q"] for q in yaml.safe_load(
        Path("eval/questions.yaml").read_text(encoding="utf-8"))]
    return [qs[i % len(qs)] for i in range(n)]


def pad_inputs(torch, processor, inputs: dict, pad_to: int) -> dict:
    """Right-pad to a fixed length so the forward always sees one shape.

    Only useful with CUDA graphs, which cannot capture a variable-shape forward:
    one length means one captured graph instead of a 6s capture per new query
    length. Attention is causal and the mask marks the padding as unattendable,
    so no real token can see a pad token and the pooled last-real-token vector is
    the same value in exact arithmetic. It is *not* the same computation — the
    kernels now run over a longer sequence, so reductions tile differently and
    the last bits move. parity_check.py is how that gets quantified, not assumed.
    """
    n = inputs["input_ids"].shape[1]
    if not pad_to or n > pad_to:
        return inputs
    pad_id = processor.tokenizer.pad_token_id
    # attention_mask 0 keeps the pad unattended; mm_token_type_ids 0 is "text",
    # which is what every token in a text-only query already is.
    fill = {"input_ids": pad_id, "attention_mask": 0, "mm_token_type_ids": 0}
    out = {}
    for k, v in inputs.items():
        if k in fill and hasattr(v, "shape") and v.ndim == 2:
            out[k] = torch.nn.functional.pad(v, (0, pad_to - n), value=fill[k])
        else:
            out[k] = v
    return out


def encode(torch, model, processor, text: str, device: str, pad_to: int = 0):
    """Byte-identical to pixelrag_serve._encode_queries — do not 'improve' this."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs = pad_inputs(torch, processor, inputs, pad_to)
    with torch.no_grad():
        out = model.model(**inputs)
    h = out.last_hidden_state
    idx = inputs["attention_mask"].sum(dim=1) - 1
    pooled = h[torch.arange(h.size(0), device=h.device), idx]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().float().numpy()


PHASES = ("template", "tokenise", "to_device", "forward", "pool", "to_host")


def time_phases(torch, model, processor, text: str, device: str,
                pad_to: int = 0) -> dict[str, float]:
    """One encode, split by phase in ms. The reference timing for this project.

    Kept separate from encode() so that anything comparing configurations
    (sweep_encoder.py, ab_encoder.py) measures with the same syncs rather than
    growing its own subtly different stopwatch.
    """
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    sync(); t = [time.perf_counter()]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    t.append(time.perf_counter())
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    t.append(time.perf_counter())
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs = pad_inputs(torch, processor, inputs, pad_to)
    sync(); t.append(time.perf_counter())
    with torch.no_grad():
        out = model.model(**inputs)
    sync(); t.append(time.perf_counter())
    h = out.last_hidden_state
    idx = inputs["attention_mask"].sum(dim=1) - 1
    pooled = h[torch.arange(h.size(0), device=h.device), idx]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    sync(); t.append(time.perf_counter())
    pooled.cpu().float().numpy()
    sync(); t.append(time.perf_counter())
    out = {name: (b - a) * 1000 for name, a, b in zip(PHASES, t[:-1], t[1:])}
    out["TOTAL"] = (t[-1] - t[0]) * 1000
    return out


def mode_phases(torch, model, processor, device: str, qs: list[str], pad_to: int = 0):
    acc: dict[str, list[float]] = {k: [] for k in PHASES + ("TOTAL",)}
    for text in qs:
        for k, v in time_phases(torch, model, processor, text, device, pad_to).items():
            acc[k].append(v)

    total = statistics.mean(acc["TOTAL"])
    print(f"\n{'phase':<12}{'mean ms':>10}{'median':>10}{'p95':>10}{'% total':>10}")
    print("-" * 52)
    for name in PHASES + ("TOTAL",):
        xs = sorted(acc[name])
        p95 = xs[min(len(xs) - 1, int(round(0.95 * len(xs))) - 1)]
        share = "" if name == "TOTAL" else f"{100*statistics.mean(xs)/total:9.1f}%"
        print(f"{name:<12}{statistics.mean(xs):>10.2f}{statistics.median(xs):>10.2f}"
              f"{p95:>10.2f}{share:>10}")


def mode_cprofile(torch, model, processor, device: str, qs: list[str], top: int):
    pr = cProfile.Profile()
    pr.enable()
    for text in qs:
        encode(torch, model, processor, text, device)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(top)
    print(s.getvalue())


def mode_torch(torch, model, processor, device: str, qs: list[str], top: int):
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if device == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=False) as prof:
        for text in qs:
            encode(torch, model, processor, text, device)
    key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["phases", "cprofile", "torch"], default="phases")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--attn", choices=["sdpa", "eager", "flash_attention_2"],
                    default="sdpa", help="sdpa is what the index was built with")
    ap.add_argument("--tokenise", choices=["stock", "memo", "fast"], default="stock",
                    help="see tokenise_cache.py; both variants are token-identical")
    ap.add_argument("--compile", dest="compile_mode", default="off",
                    choices=["off", "cudagraphs", "default", "reduce-overhead",
                             "max-autotune"])
    ap.add_argument("--pad-to", type=int, default=0,
                    help="right-pad every query to this token length (0 = off). "
                         "Only worth it with --compile cudagraphs")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    torch, model, processor = load(args.device, args.dtype, args.attn, args.tokenise,
                                  args.compile_mode, args.pad_to)
    for i, text in enumerate(queries(args.warmup)):
        t0 = time.perf_counter()
        encode(torch, model, processor, text, args.device, args.pad_to)
        print(f"warmup {i}: {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    qs = queries(args.n)
    print(f"\nmode={args.mode} device={args.device} dtype={args.dtype} "
          f"attn={args.attn} tokenise={args.tokenise} compile={args.compile_mode} "
          f"pad_to={args.pad_to} n={args.n}")
    if args.mode == "phases":
        mode_phases(torch, model, processor, args.device, qs, args.pad_to)
    elif args.mode == "cprofile":
        mode_cprofile(torch, model, processor, args.device, qs, args.top)
    else:
        mode_torch(torch, model, processor, args.device, qs, args.top)


if __name__ == "__main__":
    main()
