#!/usr/bin/env python
"""Profile end-to-end search latency: embed sidecar + faiss /search.

Writes scratch/latency_latest.json and optionally cProfiles the in-process
encode path for hotspot shape (faiss process stays untouched).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("PIXELRAG_LOCAL_ENCODE", "1")

import requests  # noqa: E402
import rag  # noqa: E402

QUERIES = [
    "wkładka antywłamaniowa cena",
    "dimension callout hinge height",
    "exploded view parts table",
    "ile kosztuje zamek",
    "szerokość skrzydła drzwi",
    "montaż ościeżnicy schemat",
    "technical drawing elevation gate",
    "części zamienne lista",
    "odległość otworów zawias",
    "what is the price of the lock cylinder",
]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    i = int(round((p / 100) * (len(xs) - 1)))
    return xs[i]


def timed_search(q: str) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    t1 = time.perf_counter()
    emb = rag.embed_query(q)
    t_embed = time.perf_counter() - t1
    t2 = time.perf_counter()
    r = requests.post(
        f"{rag.SEARCH_API}/search",
        json={"queries": [{"embedding": emb}], "n_docs": 5},
        timeout=120,
    )
    r.raise_for_status()
    t_faiss = time.perf_counter() - t2
    return time.perf_counter() - t0, t_embed, t_faiss


def run_bench(n: int, out: Path) -> dict:
    for _ in range(3):
        rag.search(QUERIES[0], n_results=5)

    queries = (QUERIES * ((n + len(QUERIES) - 1) // len(QUERIES)))[:n]
    totals, embeds, faisses = [], [], []
    for q in queries:
        tot, emb, fai = timed_search(q)
        totals.append(tot * 1000)
        embeds.append(emb * 1000)
        faisses.append(fai * 1000)

    summary = {
        "n": len(totals),
        "total_ms": {
            "median": statistics.median(totals),
            "p95": pct(totals, 95),
            "p99": pct(totals, 99),
            "mean": statistics.mean(totals),
            "min": min(totals),
            "max": max(totals),
        },
        "embed_ms": {
            "median": statistics.median(embeds),
            "p95": pct(embeds, 95),
            "p99": pct(embeds, 99),
        },
        "faiss_ms": {
            "median": statistics.median(faisses),
            "p95": pct(faisses, 95),
            "p99": pct(faisses, 99),
        },
        "sidecar": rag._sidecar_available(),
        "local_encode": rag.LOCAL_ENCODE,
        "pass": pct(totals, 95) < 300.0,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(
            f"FAIL: p95 {summary['total_ms']['p95']:.1f}ms >= 300ms target"
        )
    return summary


def run_cprofile(out: Path, reps: int = 5) -> None:
    """Force in-process encode so cProfile sees torch, not HTTP."""
    rag._sidecar_ok = False
    rag._get_encoder()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(reps):
        rag.embed_query("profile probe dimension callout hinge")
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
    text = buf.getvalue()
    out.write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "scratch" / "latency_latest.json")
    args = ap.parse_args()
    run_bench(args.n, args.out)
    if args.profile:
        run_cprofile(ROOT / "scratch" / "encode_profile.txt")


if __name__ == "__main__":
    main()
