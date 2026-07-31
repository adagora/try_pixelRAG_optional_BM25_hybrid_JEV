#!/usr/bin/env python
"""Measure retrieval quality against eval/questions.yaml.

Reports recall@k overall and broken down by question kind, because the
breakdown is what tells you what to fix. Visual embeddings are strong on
diagrams and layout and weak on exact lexical match — if part_number recall
lags while diagram recall is fine, that is your signal to add an OCR text
sidecar and go hybrid, not to tune the vision model.

Requires the search API to be running:
    .venv/bin/pixelrag serve --index-dir ./index --port 30001

Usage:
    .venv/bin/python scripts/evaluate.py [--k 5] [--port 30001]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import requests
import yaml

import layout
import rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="recall@k cutoff")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--questions",
                    default=str(layout.ROOT / "eval" / "questions.yaml"))
    args = ap.parse_args()

    api = f"http://127.0.0.1:{args.port}"
    questions = yaml.safe_load(Path(args.questions).read_text())

    # articles.json is a list; a document's article_id is its position in it.
    articles = json.loads(layout.DEFAULT.articles_json.read_text(encoding="utf-8"))
    stem_of = {i: a["title"] for i, a in enumerate(articles)}

    try:
        requests.get(f"{api}/health", timeout=5).raise_for_status()
    except Exception:
        sys.exit(
            f"No search API on {api}.\n"
            f"Start it with: .venv/bin/pixelrag serve --index-dir ./index --port {args.port}"
        )

    by_kind = defaultdict(lambda: [0, 0])  # kind -> [hits, total]
    misses = []

    for item in questions:
        # Go through rag.search so this measures the path the app actually
        # uses, including local/sidecar query encoding.
        hits = rag.search(item["q"], n_results=args.k)

        # For PDFs tile_index is the 0-based page, so page == tile_index + 1.
        got = {(stem_of.get(h["article_id"], "?"), h["tile_index"] + 1) for h in hits}
        want = {(item["pdf"], p) for p in item["pages"]}
        ok = bool(got & want)

        kind = item.get("kind", "unspecified")
        by_kind[kind][1] += 1
        if ok:
            by_kind[kind][0] += 1
        else:
            misses.append((item, sorted(got)[: args.k]))

        print(f"  {'PASS' if ok else 'FAIL'}  [{kind}] {item['q']}")

    total_hit = sum(v[0] for v in by_kind.values())
    total_all = sum(v[1] for v in by_kind.values())

    print(f"\n{'='*58}\nrecall@{args.k}: {total_hit}/{total_all} "
          f"({total_hit/total_all*100:.0f}%)\n{'='*58}")
    for kind in sorted(by_kind):
        hit, tot = by_kind[kind]
        print(f"  {kind:<12} {hit}/{tot}  ({hit/tot*100:3.0f}%)")

    if misses:
        print("\nMisses — what came back instead:")
        for item, got in misses:
            print(f"\n  Q: {item['q']}")
            print(f"     wanted {item['pdf']} p{item['pages']}")
            print(f"     got    {got}")


if __name__ == "__main__":
    main()
