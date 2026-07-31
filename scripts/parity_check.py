#!/usr/bin/env python
"""Does this encoder configuration still land in the index's vector space?

Overlaps scripts/check_parity.py on purpose and should probably be folded into
it: that one is the pass/fail gate for the shipped device/dtype and exits
non-zero, this one sweeps the tuning knobs check_parity.py does not expose
(--attn, --pad-to, --compile) and reports magnitudes rather than a verdict. If
those knobs land in check_parity.py, delete this file.

Latency work on the query encoder is only ever half a result. The index was
built with one particular forward pass, and a query encoded any other way lands
somewhere else in the 2048-d space — retrieval then degrades silently, with no
error anywhere. So every configuration gets scored on two axes, and the second
one vetoes the first.

The reference is CPU/fp32/SDPA, which is what `pixelrag serve` runs on this box
(pixelrag_serve/api.py:641 picks fp32 on CPU) and the closest thing available to
exact arithmetic. --verify-server checks that reference against the live service
so the reference itself is not taken on trust.

What the numbers mean:
  cos            cosine to the reference vector, per query. 1 - cos is the
                 error; anything that changes ranking shows up here first.
  score delta    the same error expressed where it matters: inner product
                 against the 36 indexed tiles, i.e. what faiss sorts on.
  top-k agree    did the returned tile order actually change.

A config with a visible cos gap and unchanged rankings is not "passing" — this
index has 36 vectors and generous score margins. Report the gap.

    python scripts/parity_check.py --dtype fp16 --tokenise fast
    python scripts/parity_check.py --dtype bf16 --attn eager --verify-server
"""

import argparse
import statistics
import tempfile
from pathlib import Path

import numpy as np
import requests

import profile_search as ps
from bench_search import query_set

SEARCH_API = "http://127.0.0.1:30001"
INDEX_EMB = Path("index/embeddings/shard_000.npz")


def embed_all(qs: list[str], device: str, dtype: str, attn: str, tokenise: str,
              compile_mode: str = "off", pad_to: int = 0) -> np.ndarray:
    torch, model, processor = ps.load(device, dtype, attn, tokenise, compile_mode,
                                      pad_to)
    for text in qs[:2]:
        ps.encode(torch, model, processor, text, device, pad_to)
    out = np.vstack([ps.encode(torch, model, processor, t, device, pad_to)
                     for t in qs])
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def index_matrix() -> np.ndarray:
    """The indexed tile vectors, as float32 — stored fp16, which is itself a
    reminder of how much precision this pipeline already throws away."""
    with np.load(INDEX_EMB) as z:
        key = "embeddings" if "embeddings" in z else z.files[0]
        return z[key].astype(np.float32)


def report(ref: np.ndarray, cand: np.ndarray, qs: list[str], k: int) -> dict:
    cos = np.einsum("ij,ij->i", ref, cand) / (
        np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1))
    maxabs = np.abs(ref - cand).max()

    docs = index_matrix()
    s_ref, s_cand = ref @ docs.T, cand @ docs.T
    top_ref, top_cand = np.argsort(-s_ref, axis=1)[:, :k], np.argsort(-s_cand, axis=1)[:, :k]
    agree_k = float(np.mean([set(a) == set(b) for a, b in zip(top_ref, top_cand)]))
    same_order = float(np.mean([list(a) == list(b) for a, b in zip(top_ref, top_cand)]))
    top1 = float(np.mean(top_ref[:, 0] == top_cand[:, 0]))
    score_delta = float(np.abs(s_ref - s_cand).max())

    worst = int(np.argmin(cos))
    print(f"\nn={len(qs)} queries")
    print(f"cos      min {cos.min():.6f}  mean {statistics.fmean(cos):.6f}  "
          f"max {cos.max():.6f}")
    print(f"         worst query: {qs[worst][:70]!r}")
    print(f"max |delta| per component   {maxabs:.3e}")
    print(f"max |delta| on index score  {score_delta:.3e}   (scores span "
          f"{s_ref.min():.3f}..{s_ref.max():.3f})")
    print(f"top-1 same {top1:.1%}   top-{k} same set {agree_k:.1%}   "
          f"same order {same_order:.1%}")
    return {"cos_min": float(cos.min()), "cos_mean": float(statistics.fmean(cos)),
            "maxabs": float(maxabs), "score_delta": score_delta,
            "top1": top1, "agree_k": agree_k, "same_order": same_order}


def verify_server(ref: np.ndarray, qs: list[str], k: int) -> None:
    """Confirm the reference matches what the live faiss service encodes itself.

    Same code and same dtype, but a different torch build (the .venv ships
    CPU-only torch), so the residual here is the floor: no configuration can be
    called identical to the service below this much difference.
    """
    try:
        requests.get(f"{SEARCH_API}/health", timeout=3).raise_for_status()
    except requests.RequestException as e:
        print(f"\n[verify-server] skipped: {e}")
        return
    worst_score, worst_order = 0.0, 0
    for i, q in enumerate(qs):
        a = requests.post(f"{SEARCH_API}/search",
                          json={"queries": [{"text": q}], "n_docs": k},
                          timeout=180).json()["results"][0]["hits"]
        b = requests.post(f"{SEARCH_API}/search",
                          json={"queries": [{"embedding": ref[i].tolist()}], "n_docs": k},
                          timeout=180).json()["results"][0]["hits"]
        ka = [(h["article_id"], h["tile_index"], h["chunk_index"]) for h in a]
        kb = [(h["article_id"], h["tile_index"], h["chunk_index"]) for h in b]
        worst_order += ka != kb
        worst_score = max(worst_score, max(abs(x["score"] - y["score"])
                                           for x, y in zip(a, b)))
    print(f"\n[verify-server] reference vs service-encoded: "
          f"max score delta {worst_score:.3e}, order differs on "
          f"{worst_order}/{len(qs)} queries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--attn", choices=["sdpa", "eager", "flash_attention_2"],
                    default="sdpa")
    ap.add_argument("--tokenise", choices=["stock", "memo", "fast"], default="stock")
    ap.add_argument("--compile", dest="compile_mode", default="off",
                    choices=["off", "cudagraphs", "default", "reduce-overhead",
                             "max-autotune"])
    ap.add_argument("--pad-to", type=int, default=0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--verify-server", action="store_true")
    ap.add_argument("--ref-cache",
                    default=str(Path(tempfile.gettempdir()) / "pixelrag_parity_ref.npz"),
                    help="reference vectors are ~11s of CPU fp32 per run; reuse them")
    args = ap.parse_args()

    qs = query_set(args.n)
    cache = Path(args.ref_cache)
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            if list(z["queries"]) == qs:
                ref = z["ref"]
            else:
                ref = None
    else:
        ref = None
    if ref is None:
        print("computing CPU/fp32/SDPA reference (slow, cached afterwards)", flush=True)
        ref = embed_all(qs, "cpu", "fp32", "sdpa", "stock")
        np.savez(cache, ref=ref, queries=np.array(qs, dtype=object))

    if args.verify_server:
        verify_server(ref, qs, args.k)

    print(f"\ncandidate: device={args.device} dtype={args.dtype} attn={args.attn} "
          f"tokenise={args.tokenise} compile={args.compile_mode} pad_to={args.pad_to}")
    cand = embed_all(qs, args.device, args.dtype, args.attn, args.tokenise,
                     args.compile_mode, args.pad_to)
    report(ref, cand, qs, args.k)


if __name__ == "__main__":
    main()
