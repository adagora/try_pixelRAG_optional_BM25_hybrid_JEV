#!/usr/bin/env python
"""Latency harness for the retrieval path — the thing the p95 target is about.

Reports the phase split, not just the total, because the total has never been
the interesting number here: encoding the query is the whole cost and faiss is
noise. Keeping them separate is what tells you whether an optimisation moved
the part that matters or just moved the queue.

    encode  query text -> 2048-d vector   (transformer forward)
    post    POST /search with that vector (faiss IVF + metadata join)
    total   what a caller actually waits for

Query set is eval/questions.yaml cycled to --n with length variants mixed in,
so the tail reflects long queries rather than one repeated string sitting in
every cache. Warmup requests are excluded — a p95 that includes model load is
measuring startup, not steady state.

    python scripts/bench_search.py --n 150
    python scripts/bench_search.py --n 150 --encode-only   # no server needed
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import rag
import requests
import yaml

# Suffixes that lengthen a query without changing what it asks for. Token count
# drives encoder cost, so a benchmark of only short queries understates the tail.
_PAD = [
    "",
    " — proszę podać cenę netto w PLN",
    " for a residential installation, standard hardware, no automation",
    (" I need the exact figure from the price list including which row and "
     "column of the matrix it came from, and whether VAT is included"),
]


def query_set(n: int, path: str = "eval/questions.yaml") -> list[str]:
    base = [q["q"] for q in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]
    out = []
    while len(out) < n:
        for i, q in enumerate(base):
            out.append(q + _PAD[(len(out) + i) % len(_PAD)])
            if len(out) == n:
                break
    return out


def pct(xs: list[float], p: float) -> float:
    """Nearest-rank percentile — no interpolation, so p95 is a real observation."""
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p / 100 * len(s))) - 1))]


def summarise(name: str, xs: list[float]) -> dict:
    return {
        "phase": name, "n": len(xs),
        "mean": statistics.mean(xs), "median": statistics.median(xs),
        "p95": pct(xs, 95), "p99": pct(xs, 99),
        "min": min(xs), "max": max(xs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--encode-only", action="store_true",
                    help="Skip faiss; measure only the encoder (no server needed).")
    ap.add_argument("--json-out", help="Write the raw per-request timings here.")
    ap.add_argument("--label", default="", help="Tag for the results file.")
    args = ap.parse_args()

    rag.SEARCH_API = f"http://127.0.0.1:{args.port}"
    queries = query_set(args.n)

    if not args.encode_only:
        try:
            requests.get(f"{rag.SEARCH_API}/health", timeout=5).raise_for_status()
        except Exception as e:
            sys.exit(f"No search API on {rag.SEARCH_API} ({e}). "
                     f"Start pixelrag serve, or pass --encode-only.")

    mode = ("sidecar" if rag._sidecar_available() else "in-process") if rag.LOCAL_ENCODE \
        else "server-side"
    print(f"encode: {mode}   queries: {args.n}   warmup: {args.warmup}\n", flush=True)

    for q in queries[:args.warmup]:
        if args.encode_only:
            rag.embed_query(q)
        else:
            rag.search(q, n_results=args.k)

    enc, post, tot = [], [], []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        # In server-side mode there is no local encode to time; the whole cost
        # lands in the POST, which is exactly the point of that comparison.
        emb = rag.embed_query(q) if rag.LOCAL_ENCODE else None
        t1 = time.perf_counter()
        if not args.encode_only:
            payload = {"embedding": emb} if emb is not None else {"text": q}
            r = requests.post(f"{rag.SEARCH_API}/search",
                              json={"queries": [payload], "n_docs": args.k},
                              timeout=300)
            r.raise_for_status()
            hits = r.json()["results"][0]["hits"]
            if not hits:
                print(f"  WARNING: no hits for {q[:50]!r}", flush=True)
        t2 = time.perf_counter()
        enc.append((t1 - t0) * 1000)
        post.append((t2 - t1) * 1000)
        tot.append((t2 - t0) * 1000)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{args.n}  total p95 so far {pct(tot, 95):7.1f}ms", flush=True)

    rows = [summarise("encode", enc)]
    if not args.encode_only:
        rows += [summarise("post(faiss)", post), summarise("TOTAL", tot)]

    print(f"\n{'phase':<14}{'mean':>9}{'median':>9}{'p95':>9}{'p99':>9}{'min':>9}{'max':>9}")
    print("-" * 68)
    for r in rows:
        print(f"{r['phase']:<14}{r['mean']:>9.1f}{r['median']:>9.1f}"
              f"{r['p95']:>9.1f}{r['p99']:>9.1f}{r['min']:>9.1f}{r['max']:>9.1f}")

    target = rows[-1]["p95"]
    verdict = "PASS" if target < 300 else "FAIL"
    print(f"\n{verdict}: {rows[-1]['phase']} p95 = {target:.1f}ms (target <300ms)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "label": args.label, "encode_mode": mode, "n": args.n,
            "summary": rows, "encode_ms": enc, "post_ms": post, "total_ms": tot,
        }, indent=2), encoding="utf-8")
        print(f"raw timings -> {args.json_out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
