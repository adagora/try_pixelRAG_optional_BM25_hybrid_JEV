#!/usr/bin/env python
"""Does a faster encoder still retrieve the same things?

This repo's whole latency story is moving query encoding out of `pixelrag serve`
and onto an accelerator in fp16. That is only sound if the vector it produces
lands in the same space as the indexed documents: the index was built with this
model, and a query encoded even slightly differently returns plausible-looking
wrong tiles with no error raised anywhere. A shortcut that fed raw query text
past the chat template scored 1/10 identical rankings — replicating the template
took it to 10/10 with a max score delta of 0.0008. This is that check, committed
so it can be re-run instead of remembered.

Reference is the same model on cpu/float32, the closest thing to ground truth
reachable without rebuilding the index. Candidate is whatever --device/--dtype
is being proposed.

Cosine alone would not settle it — the reader consumes a *ranking*, so both
vectors go through the real /search and the returned (article_id, tile_index,
chunk_index) orderings are compared. Three agreement numbers, because they
diverge and the gap between them is the finding: fp16 rounding reorders hits
that were already near-tied, which changes the order while retrieving exactly
the same tiles. Exact order is the strict signal, set equality is what decides
pass/fail, top-1 is what a reader that opens one tile actually sees.

    python scripts/check_parity.py                          # cuda/fp16 vs cpu/fp32
    python scripts/check_parity.py --device cpu --dtype bf16

Exit 0 only if every query's top-5 set matches and the worst-case cosine is
>= 0.999. Requires `pixelrag serve` on --port. Never imports faiss — that
segfault is the reason the encoder was split out in the first place.
"""

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path

# Before transformers is imported anywhere: the weights are already cached, and
# a hub round-trip per load is multi-second stall for no information.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("PIXELRAG_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."

N_DOCS = 5
COSINE_FLOOR = 0.999

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}

# Every file that keeps its own copy of the instruction string. Both arms of this
# script share one encode body, so a wrong template cancels out and reads as
# cosine 1.0 — the exact bug this check exists to catch would be invisible.
# Comparing the constants at source level closes that hole, and reading api.py
# as text rather than importing it keeps faiss out of this process.
INSTRUCTION_OWNERS = [
    "scripts/rag.py",
    "scripts/encoder.py",
    "scripts/profile_search.py",
    ".venv/Lib/site-packages/pixelrag_serve/api.py",
    ".venv/lib/python3*/site-packages/pixelrag_serve/api.py",
]
_INSTRUCTION_RE = re.compile(r'^DEFAULT_INSTRUCTION\s*=\s*"(.*)"\s*$', re.M)


def instruction_mismatches() -> list[str]:
    """Which files disagree with us about the instruction string."""
    bad = []
    for pattern in INSTRUCTION_OWNERS:
        for path in sorted(ROOT.glob(pattern)):
            m = _INSTRUCTION_RE.search(path.read_text(encoding="utf-8"))
            if m and m.group(1) != DEFAULT_INSTRUCTION:
                bad.append(f"{path.relative_to(ROOT).as_posix()}: {m.group(1)!r}")
    return bad


def questions(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def encode_all(texts: list[str], device: str, dtype_name: str) -> np.ndarray:
    """Load the model, encode every query, release it. Returns (n, dim) float32.

    The body below must stay step-for-step with
    pixelrag_serve._encode_queries (api.py:289) — same chat template, same
    instruction, same last-token pool over last_hidden_state, same L2
    normalise. One query per forward, because that is what the serving path
    does: batching pads, and where the last real token lands is exactly what
    the pool indexes.
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = getattr(torch, DTYPES[dtype_name])
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=dtype).eval().to(device)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print(f"  loaded on {device}/{dtype_name} in {time.perf_counter()-t0:.1f}s",
          flush=True)

    out = []
    for text in texts:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        inputs = processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model.model(**inputs).last_hidden_state
        idx = inputs["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        out.append(pooled.cpu().float().numpy()[0])

    del model, processor, inputs, hidden, pooled
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.stack(out)


def key(hit: dict) -> tuple[int, int, int]:
    return (hit["article_id"], hit["tile_index"], hit["chunk_index"])


def search(api: str, vector: np.ndarray) -> list[dict]:
    r = requests.post(f"{api}/search",
                      json={"queries": [{"embedding": vector.tolist()}],
                            "n_docs": N_DOCS},
                      timeout=120)
    r.raise_for_status()
    return r.json()["results"][0]["hits"]


def score_delta(ref_hits: list[dict], cand_hits: list[dict]) -> float:
    """Largest score disagreement, matched by hit identity rather than by rank.

    Comparing rank-for-rank would report the gap between two *different* tiles
    whenever a near-tie swaps, which is a property of the corpus, not of the
    encoder. Matching on (article, tile, chunk) isolates the rounding error.
    """
    ref = {key(h): h["score"] for h in ref_hits}
    cand = {key(h): h["score"] for h in cand_hits}
    shared = ref.keys() & cand.keys()
    return max((abs(ref[k] - cand[k]) for k in shared), default=float("nan"))


def _flag(ok: bool) -> str:
    return " ok  " if ok else "DIFF "


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare a candidate query encoder against cpu/float32.")
    ap.add_argument("--device", default="cuda",
                    help="Candidate device; the reference is always cpu.")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16",
                    help="Candidate dtype; the reference is always fp32.")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--questions", default="eval/questions.yaml")
    args = ap.parse_args()

    # The eval set is in Polish; a cp1252 console raises UnicodeEncodeError on
    # the first question and takes the verdict line down with it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    qpath = Path(args.questions)
    if not qpath.is_absolute() and not qpath.exists():
        qpath = ROOT / args.questions
    items = questions(qpath)
    texts = [q["q"] for q in items]

    api = f"http://127.0.0.1:{args.port}"
    # Check the server before spending a minute loading two 2B models.
    try:
        requests.get(f"{api}/health", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"No search API on {api} ({e}). Start pixelrag serve first; "
                 f"this check needs real rankings, not just cosine.")

    bad = instruction_mismatches()
    if bad:
        print("instruction string has drifted — every copy must match "
              f"{DEFAULT_INSTRUCTION!r}:")
        for line in bad:
            print(f"  {line}")
        print("\nPARITY FAIL: encoders disagree on the instruction string; "
              "ranking comparison would be meaningless.")
        return 1

    print(f"parity check  candidate={args.device}/{args.dtype}  "
          f"reference=cpu/fp32  queries={len(texts)}  n_docs={N_DOCS}\n")

    # Candidate first, then reference, and never both resident. from_pretrained
    # materialises the weights in host RAM before .to(cuda) copies them across,
    # so the candidate wants ~4.5GB of host RAM transiently — free now, an extra
    # 4.5GB on top of the reference's ~9GB if the reference were already loaded.
    # Order also decides how fast a mistake surfaces: a bad --device/--dtype
    # fails during the fast load rather than after paying for the slow one.
    print("candidate:")
    cand_vecs = encode_all(texts, args.device, args.dtype)
    print("reference:")
    ref_vecs = encode_all(texts, "cpu", "fp32")

    rows = []
    for i, (item, ref_v, cand_v) in enumerate(zip(items, ref_vecs, cand_vecs)):
        a, b = ref_v.astype(np.float64), cand_v.astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        ref_hits = search(api, ref_v)
        cand_hits = search(api, cand_v)
        ref_order = [key(h) for h in ref_hits]
        cand_order = [key(h) for h in cand_hits]
        rows.append({
            "n": i + 1, "q": item["q"], "kind": item.get("kind", "?"), "cos": cos,
            "order": ref_order == cand_order,
            "set": set(ref_order) == set(cand_order),
            "top1": ref_order[:1] == cand_order[:1],
            "dscore": score_delta(ref_hits, cand_hits),
            "ref_order": ref_order, "cand_order": cand_order,
        })

    cosines = [r["cos"] for r in rows]
    deltas = [r["dscore"] for r in rows if r["dscore"] == r["dscore"]]
    worst_cos = min(cosines)
    max_delta = max(deltas) if deltas else float("nan")
    n_order = sum(r["order"] for r in rows)
    n_set = sum(r["set"] for r in rows)
    n_top1 = sum(r["top1"] for r in rows)
    n = len(rows)

    print(f"\n{'#':>3} {'cosine':>10} {'dscore':>9}  order  set  top1  query")
    print("-" * 78)
    for r in rows:
        print(f"{r['n']:>3} {r['cos']:>10.6f} {r['dscore']:>9.6f}  "
              f"{_flag(r['order'])} {_flag(r['set'])}{_flag(r['top1'])} {r['q'][:38]}")

    print(f"\ncosine        worst {worst_cos:.6f}   mean "
          f"{sum(cosines)/n:.6f}   best {max(cosines):.6f}")
    print(f"top-5 order   {n_order}/{n} identical")
    print(f"top-5 set     {n_set}/{n} identical (order ignored)")
    print(f"top-1 hit     {n_top1}/{n} identical")
    print(f"score delta   max {max_delta:.6f}")

    reordered = [r for r in rows if r["set"] and not r["order"]]
    if reordered:
        print("\nSame tiles, different order — near-ties reordered by rounding:")
        for r in reordered:
            print(f"  #{r['n']} {r['q'][:60]}")
            print(f"      ref  {r['ref_order']}")
            print(f"      cand {r['cand_order']}")

    failures = [r for r in rows if not r["set"] or r["cos"] < COSINE_FLOOR]
    if failures:
        print("\nFAILING QUERIES")
        for r in failures:
            why = []
            if r["cos"] < COSINE_FLOOR:
                why.append(f"cosine {r['cos']:.6f} < {COSINE_FLOOR}")
            if not r["set"]:
                ref_set, cand_set = set(r["ref_order"]), set(r["cand_order"])
                why.append(
                    f"retrieved different tiles: only in ref "
                    f"{[k for k in r['ref_order'] if k not in cand_set]}, "
                    f"only in candidate "
                    f"{[k for k in r['cand_order'] if k not in ref_set]}")
            print(f"  #{r['n']} [{r['kind']}] {r['q']}")
            for w in why:
                print(f"      {w}")

    ok = not failures
    print(f"\n{'PARITY PASS' if ok else 'PARITY FAIL'}: {args.device}/{args.dtype} "
          f"vs cpu/fp32 — {n_set}/{n} top-5 sets, {n_order}/{n} exact order, "
          f"{n_top1}/{n} top-1, worst cosine {worst_cos:.6f}, max score delta "
          f"{max_delta:.6f} ({n} eval queries, n_docs={N_DOCS}).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
