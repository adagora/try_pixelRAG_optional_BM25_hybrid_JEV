"""How the query encoder runs — the one copy of those decisions.

Both the in-process path (scripts/rag.py) and the sidecar (scripts/encoder.py)
load the same weights, and rag.py falls back to the former when the latter is
down. If the two disagreed about device, dtype, input shape or whether the
forward is graphed, that fallback would quietly change both the latency profile
and the last bits of every embedding, so the choices live here and neither file
gets its own copy.

Every knob here was measured rather than guessed, and all but the tokeniser cost
some precision against the fp32 reference; the numbers quoted below come from

    python scripts/sweep_encoder.py --repeats 4 --n 100 --warmup 12
    python scripts/parity_check.py --dtype fp16 --tokenise fast --pad-to 64 \
        --compile cudagraphs --verify-server
"""

import concurrent.futures
import os

# Accepts profile_search.py's short names as well as torch's own, because the
# env var is most often set by hand right after reading one of those outputs.
_DTYPES = {
    "fp16": "float16", "half": "float16", "float16": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp32": "float32", "float": "float32", "float32": "float32",
}


def resolve(torch):
    """(device, dtype) for the query encoder. cuda > mps > cpu, env-overridable.

    Takes the caller's torch rather than importing its own: both callers import
    it lazily so that a process which never encodes anything (the sidecar is
    up, or the reader only fetches tiles) never pays for it.
    """
    dev = os.environ.get("PIXELRAG_ENCODER_DEVICE", "").strip().lower()
    if not dev:
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

    name = os.environ.get("PIXELRAG_ENCODER_DTYPE", "").strip().lower()
    if not name:
        # fp16 pays off only where there are half-precision kernels to run it.
        # CPU has none for this model, so torch widens back to fp32 around every
        # op and the "cheaper" dtype comes out slower than plain fp32.
        name = "fp32" if dev.split(":")[0] == "cpu" else "fp16"
    if name not in _DTYPES:
        # Plain ASCII: this surfaces through a traceback on a cp1252 console.
        raise ValueError(f"PIXELRAG_ENCODER_DTYPE={name!r}, expected one of "
                         f"{sorted(_DTYPES)}")
    return dev, getattr(torch, _DTYPES[name])


# Round every query's token count up to a multiple of this. 64 because the
# longest question in eval/questions.yaml tokenises to 64, so the common case is
# one shape with nothing truncated. 0 disables padding, and with it graphing.
PAD_TO_MULTIPLE = int(os.environ.get("PIXELRAG_ENCODER_PAD", "64") or 0)

# Widths to capture before serving, derived from the multiple so that overriding
# one cannot leave a padding width no graph covers. Three of them: the first is
# the eval set, the other two are where a user's long question lands. ~6s each,
# and anything past the last runs un-graphed (see prepare_forward).
WARM_WIDTHS = tuple(PAD_TO_MULTIPLE * i for i in (1, 2, 3))

GRAPH_FORWARD = os.environ.get("PIXELRAG_ENCODER_COMPILE", "1") != "0"

# Strings to prove the fast tokeniser against. Raw queries, not templated
# prompts: the template is identical in both arms of the comparison, so it would
# only move the instruction constant into a third file for check_parity.py to
# police. Polish because the corpus is, and byte-level BPE is where a broken
# merge table shows up first; the long one crosses the padding buckets.
_PROBES = (
    "warmup",
    "Jaka jest cena netto drzwi zewnętrznych 90 cm w kolorze antracytowym?",
    "price list row and column reference " * 24,
)


def use_fast_tokenise(processor, model_id: str) -> None:
    """Stop `processor(text=...)` rebuilding a validation dataclass per call.

    Takes the tokenise phase from 4.9ms to 0.5ms, all of it plumbing:
    transformers synthesises a fresh TypedDict on every call so
    huggingface_hub's lru_cache over it never hits (see tokenise_cache.py).

    Checked against an untouched processor rather than trusted. It is the one
    change here with no numerical cost, which is exactly why it must be proved
    to have none — a wrong tokenisation is a wrong ranking with nothing raised.
    """
    from transformers import AutoProcessor

    import tokenise_cache

    stock = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenise_cache.cache_merged_kwargs(processor)
    tokenise_cache.assert_identical(stock, processor, list(_PROBES))


def prepare_forward(torch, model, processor, inputs: dict, width: int = 0):
    """Pad the batch to a captured shape; return (inputs, the module to run).

    One function because it is one decision: the graphed module is only safe for
    a shape warm-up already recorded, so what the input is padded to and which
    module runs it cannot be chosen separately. `width` forces a shape, which is
    how warm-up records them in the first place.

    On the padding — attention is causal and the mask marks the pad unattendable,
    so no real token can see a pad token and the pooled last-real-token vector is
    the same value in exact arithmetic. It is *not* the same computation: kernels
    run over a longer sequence, reductions tile differently, and the last bits
    move (7.9e-04 on an index score, cos 0.999994 to the fp32 reference, 24/24
    rankings unchanged). That is only worth paying to pin a shape for the graph,
    so with nothing graphed nothing is padded either.

    Past the last warmed width, run the module un-graphed and unpadded. Recording
    a graph on the request path is not the ~5s stall the sweep measured in a
    single-threaded harness: cuBLAS cannot create its per-thread handle inside a
    capture, so from a uvicorn worker thread the capture aborts with
    cudaErrorStreamCaptureInvalidated, and it poisons the context — measured
    here, every later request 500s too, and rag.search then falls back to 1673ms
    server-side encoding forever with nothing but a log line to say so. Today's
    latency on a rare long question is the right way to lose.
    """
    plain = getattr(model, "_ungraphed_model", None)
    if plain is None:
        return inputs, model.model
    n = inputs["input_ids"].shape[1]
    width = width or -(-n // PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE
    if width > WARM_WIDTHS[-1]:
        return inputs, plain
    if width <= n:
        return inputs, model.model
    pad_id = processor.tokenizer.pad_token_id
    # attention_mask 0 keeps the pad unattended; mm_token_type_ids 0 is "text",
    # which is what every token in a text-only query already is.
    fill = {"input_ids": pad_id, "attention_mask": 0, "mm_token_type_ids": 0}
    padded = {k: torch.nn.functional.pad(v, (0, width - n), value=fill[k])
              if k in fill and hasattr(v, "shape") and v.ndim == 2 else v
              for k, v in inputs.items()}
    return padded, model.model


def graph_forward(torch, model, device: str) -> bool:
    """Replay the decoder forward from a captured CUDA graph. True if it applies.

    This forward is launch-bound, not arithmetic-bound: 8.1ms of GPU kernel time
    inside a ~30ms wall clock, spread over ~700 launches. Graphing deletes the
    launches — encode p95 31.6ms -> 20.8ms — and, more usefully for a p95 target,
    deletes their jitter: run-to-run spread collapses from 33-63ms to 20-22ms,
    because a replay cannot be descheduled part way through a launch sequence.

    backend="cudagraphs" rather than any inductor mode because there is no Triton
    wheel for Windows in this env and every inductor path dies with
    TritonMissing. dynamic=False because with dynamic shapes dynamo declines to
    capture at all ("skipping cudagraphs due to cpu device") and 28s of warmup
    buys nothing; prepare_forward() is what makes a static shape true.

    Not numerically free: up to 3.7e-04 per component against the same config
    un-graphed, the same order as fp16 rounding itself, and no ranking moved.
    """
    if not (GRAPH_FORWARD and PAD_TO_MULTIPLE) or device.split(":")[0] != "cuda":
        return False
    # Straight into __dict__ because nn.Module.__setattr__ would register the
    # same module a second time under a new name, and these weights would then
    # show up twice in any state_dict or .apply() walk. prepare_forward runs this
    # copy for the lengths warm-up could not capture.
    model.__dict__["_ungraphed_model"] = model.model
    model.model = torch.compile(model.model, backend="cudagraphs", dynamic=False)
    return True


# One thread loads the weights, records the graphs and replays them; every encode
# is queued onto it. Two reasons, both found by breaking the sidecar:
#
#   - torch keeps its cudagraph tree per thread but its "this function is already
#     warmed up" set process-wide, so a second thread skips the eager warmup and
#     goes straight to recording. Recording needs a cuBLAS handle that a fresh
#     thread does not have, cublasCreate inside a capture returns
#     CUBLAS_STATUS_NOT_INITIALIZED, and the aborted capture poisons the CUDA
#     context: measured here, every subsequent request 500s, whatever its length.
#   - a replayed graph reads and writes the buffers it was captured with, so two
#     forwards in flight would corrupt each other and each hand back a
#     well-formed embedding for the wrong query.
#
# Neither caller can assume it is single-threaded — uvicorn runs the sidecar's
# endpoints in a threadpool and app.py serves the in-process path from one. The
# GPU serialises this work regardless; pinning it only decides where it is queued.
_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="pixelrag-encode")


def on_model_thread(fn, *args):
    """Run fn on the thread that owns the model. Never call this from inside it.

    The pool is one thread, so a nested submit would wait on the thread it is
    already running on. Callers that reach a second encode from within an encode
    (warm-up loops inside a lazy load) call their inner function directly.
    """
    return _worker.submit(fn, *args).result()


def warm_widths() -> tuple[int, ...]:
    """Widths to encode at load. Empty unless graphing, where capture is the cost.

    ~6s per width, which is why it belongs before the sidecar accepts and not on
    the first user. Un-graphed there is nothing shape-specific to warm and one
    ordinary warmup call already covers the CUDA autotune.
    """
    return WARM_WIDTHS if (GRAPH_FORWARD and PAD_TO_MULTIPLE) else ()
