#!/usr/bin/env python
"""Stage-break sidecar embed vs server-side search. No LLM keys used."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import requests
import rag

QUERIES = [
    "wkładka antywłamaniowa cena",
    "dimension callout hinge height",
    "exploded view parts table",
    "ile kosztuje zamek",
    "szerokość skrzydła drzwi",
] * 10


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[int(round((p / 100) * (len(xs) - 1)))]


def main() -> None:
    for _ in range(3):
        requests.post("http://127.0.0.1:8001/embed", json={"text": QUERIES[0]}, timeout=60)

    http_ms, search_ms = [], []
    emb = None
    for q in QUERIES:
        t0 = time.perf_counter()
        r = requests.post("http://127.0.0.1:8001/embed", json={"text": q}, timeout=60)
        r.raise_for_status()
        emb = r.json()["embedding"]
        http_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        requests.post(
            f"{rag.SEARCH_API}/search",
            json={"queries": [{"embedding": emb}], "n_docs": 5},
            timeout=60,
        ).raise_for_status()
        search_ms.append((time.perf_counter() - t1) * 1000)

    out = {
        "sidecar_embed_http_ms": {
            "n": len(http_ms),
            "median": statistics.median(http_ms),
            "p95": pct(http_ms, 95),
        },
        "faiss_ms": {
            "median": statistics.median(search_ms),
            "p95": pct(search_ms, 95),
        },
        "emb_dim": len(emb) if emb else None,
        "emb_json_chars": len(json.dumps(emb)) if emb else None,
    }

    print("server-side encode n=10 (expected multi-second)...")
    ss = []
    for q in QUERIES[:10]:
        t0 = time.perf_counter()
        requests.post(
            f"{rag.SEARCH_API}/search",
            json={"queries": [{"text": q}], "n_docs": 5},
            timeout=180,
        ).raise_for_status()
        ss.append((time.perf_counter() - t0) * 1000)
    out["server_side_ms"] = {
        "median": statistics.median(ss),
        "p95": pct(ss, 95),
        "min": min(ss),
        "max": max(ss),
    }
    print(json.dumps(out, indent=2))
    Path(__file__).resolve().parents[1].joinpath("scratch/latency_stages.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
