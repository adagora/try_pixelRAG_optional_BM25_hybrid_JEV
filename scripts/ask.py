#!/usr/bin/env python
"""Ask a question; the agent searches and browses tiles until it can answer.

Retrieval, tools, and prompting live in rag.py, shared with the web UI
(scripts/app.py) so the two cannot drift.

Requires the search API and an Anthropic key:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/pixelrag serve \
      --index-dir ./index --tiles-dir ./index/tiles \
      --articles-json ./index/articles.json --port 30001 --device cpu
    export ANTHROPIC_API_KEY=...        # or: ant auth login

Usage:
    .venv/bin/python scripts/ask.py "Ile kosztuje brama Connect H 2400x3000?"
    .venv/bin/python scripts/ask.py "..." --retrieve-only   # no model call
"""

import argparse
import sys

import rag
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--k", type=int, default=5,
                    help="hits for --retrieve-only (the agent picks its own otherwise)")
    ap.add_argument("--retrieve-only", action="store_true",
                    help="show what search returns; skip the billed agent loop")
    ap.add_argument("--max-steps", type=int, default=rag.MAX_STEPS)
    ap.add_argument("--mode", choices=["oneshot", "agent"],
                    help="default: PIXELRAG_ASK or oneshot")
    ap.add_argument("--provider", choices=["gemini", "anthropic"],
                    help="default: whichever API key is set")
    ap.add_argument("--list-models", action="store_true",
                    help="list models the configured key can reach, then exit")
    args = ap.parse_args()

    if args.list_models:
        provider = args.provider or rag.detect_provider()
        print(f"# {provider}")
        for m in rag.list_models(provider):
            print(" ", m)
        return
    if not args.question:
        ap.error("a question is required (or use --list-models)")

    if args.retrieve_only:
        try:
            hits = rag.search(args.question, n_results=args.k)
        except requests.RequestException as e:
            sys.exit(f"Search API unreachable on {rag.SEARCH_API}: {e}")
        for h in hits:
            print(f"  {rag.doc_title(h['article_id'])} "
                  f"p{h['tile_index'] + 1} region {h['chunk_index']}  "
                  f"score {h['score']:.3f}")
        return

    def show(ev):
        if ev["type"] == "search":
            print(f"  ⌕ searched {ev['query']!r} — {len(ev['hits'])} hits", file=sys.stderr)
        elif ev["type"] == "tile":
            print(f"  ▣ read {ev['document']} p{ev['page']} region {ev['chunk_index']}",
                  file=sys.stderr)
        elif ev["type"] == "tile_error":
            print(f"  ! no region {ev['tile_index']}:{ev['chunk_index']}", file=sys.stderr)

    try:
        result = rag.run_agent(args.question, on_event=show,
                               max_steps=args.max_steps, provider=args.provider,
                               mode=args.mode)
    except requests.RequestException as e:
        sys.exit(f"Search API unreachable on {rag.SEARCH_API}: {e}")

    print()
    print(result["answer"])
    u = result["usage"]
    cost = f" — about ${u['cost_usd']:.3f}" if u["cost_usd"] is not None else ""
    thoughts = u.get("thoughts") or 0
    think = f" · {thoughts} think" if thoughts else ""
    print(f"\n[{result['provider']}/{result['model']} · {result['steps']} round trips · "
          f"{u['input']} in / {u['output']} out{think}{cost}]", file=sys.stderr)


if __name__ == "__main__":
    main()
