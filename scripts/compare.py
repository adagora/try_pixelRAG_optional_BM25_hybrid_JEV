#!/usr/bin/env python
"""Run one question through several retrieval modes and report what each did.

evaluate_pl.py answers "which mode is better across the question set", offline,
against labelled pages. This answers the other question — "what happened to THIS
question, just now" — which is the one you have when an answer cited the wrong
page and you need to know whether the right page was missing from the candidate
pool, present but ranked below the cut, or there and reranked away. Those are
three different bugs and the answer text cannot tell them apart.

NO READER IS CALLED. Nothing here spends an image token, so comparing four modes
costs four retrievals plus, in the Jev modes, two TypeSafe calls each. Answering
with a mode is still one click away per row in the UI, and that is where the
reader's money gets spent.

Modes run SEQUENTIALLY, in the order asked. They share an encoder, a faiss
service with a single search thread and one connection pool, so running them
concurrently would turn each mode's latency into a measurement of the others.

    .venv/bin/python scripts/compare.py "Ile kosztuje brama Connect?"
    .venv/bin/python scripts/compare.py --modes visual,jev --json "..."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import corpus
import rag

log = logging.getLogger("pixelrag")


def modes() -> list[dict]:
    """Every mode with the reason it cannot run here — for the UI's picker."""
    return rag.retrieval_modes()


def run(question: str, mode: str, n_pages: int | None = None) -> dict:
    """One mode's retrieval, as a row of the comparison table.

    A blocked mode returns a row that says so and no numbers. It would be easy
    to run it anyway — `jev` without a key degrades to visual retrieval quite
    happily — and that is exactly the trap: a visual result under a Jev label,
    in a table whose whole job is to attribute results to modes.

    A mode that fails outright is also a row, not an exception. One broken
    sidecar must not take the other three modes' measurements with it.
    """
    mode = rag.resolve_retrieval(mode)
    n_pages = n_pages or rag.ONESHOT_PAGES
    blocked = rag.retrieval_blocked(mode)
    if blocked:
        return _row(mode, error=blocked)
    started = time.perf_counter()
    try:
        ranked, debug, stats = rag.retrieve_for(question, mode, n_pages)
    except Exception as e:
        log.warning("retrieval mode %s failed", mode, exc_info=True)
        return _row(mode, error=f"{type(e).__name__}: {e}",
                    ms=round((time.perf_counter() - started) * 1000, 1))
    return _row(mode, stats=stats, variants=debug, ms=stats["ms"],
                pages=[{**rag._hit_row(p), "region": p.focus} for p in ranked])


def _row(mode: str, *, pages=None, stats=None, variants=None,
         error=None, ms=None) -> dict:
    return {"mode": mode, "ok": error is None, "error": error,
            "ms": ms, "pages": pages or [], "stats": stats,
            "variants": variants or []}


def compare(question: str, names: list[str] | None = None,
            n_pages: int | None = None):
    """Yield one row per mode, in order, as each finishes.

    A generator because each mode is seconds of wall clock and the UI streams
    them: four modes behind one JSON response is fifteen seconds of spinner.

    One throwaway search runs first. The query encoder loads lazily, and on a
    cold process that first query costs about 26 seconds — measured here,
    against this index. Charged to whichever mode happened to run first, that is
    a number about process startup wearing a retrieval mode's name, in the one
    table whose entire purpose is comparing modes by their cost. evaluate_pl.py
    warms up before its sweep for exactly this reason.
    """
    try:
        rag.searcher("visual")("rozgrzewka", 1)
    except Exception:
        # A failed warm-up is not a failed comparison: every mode reports its
        # own error, and the first one will simply include the encoder load.
        log.debug("warm-up search failed", exc_info=True)
    for name in names or [m["mode"] for m in modes()]:
        yield run(question, name, n_pages)


def agreement(rows: list[dict]) -> list[dict]:
    """Which pages the modes agreed on — the one number a table cannot show.

    Two modes that return four pages each and share three of them are a
    different situation from two that share none, and reading that off four
    lists by eye is precisely the work this is here to avoid.
    """
    found = {r["mode"]: {(p["article_id"], p["page"]) for p in r["pages"]}
             for r in rows if r["ok"]}
    out = []
    for key in sorted(set().union(*found.values()) if found else ()):
        out.append({"article_id": key[0], "page": key[1],
                    "modes": sorted(m for m, pages in found.items() if key in pages)})
    return sorted(out, key=lambda row: -len(row["modes"]))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cost(stats: dict) -> str:
    """Retrieval money, or the tokens when nobody has told us the rate."""
    jev = (stats or {}).get("jev")
    if not jev:
        return "$0"
    if jev["cost_usd"] is not None:
        return f"${jev['cost_usd']:.4f}"
    return f"{jev['input_tokens'] + jev['output_tokens']} tok"


def _print(rows: list[dict]) -> None:
    head = f"{'mode':<12}{'pages':>6}{'chunks':>8}{'cand.pg':>9}{'rerank':>8}{'ms':>8}{'cost':>10}"
    print(head)
    print("-" * len(head))
    for r in rows:
        s = r["stats"]
        if not r["ok"]:
            print(f"{r['mode']:<12}{'—':>6}  {r['error']}")
            continue
        print(f"{r['mode']:<12}{s['pages']:>6}{s['unique_chunks']:>8}"
              f"{s['candidate_pages']:>9}{s['reranker']:>8}{s['ms']:>8.0f}"
              f"{_cost(s):>10}")
        for note in s["notes"]:
            print(f"{'':<12}  ! {note}")
    print()
    for row in agreement(rows):
        title = corpus.doc_title(row["article_id"])
        print(f"  s.{row['page']:<4} {title[:40]:<42} {', '.join(row['modes'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--modes", default=None,
                    help=f"comma-separated subset of {', '.join(rag.RETRIEVAL_MODES)}")
    ap.add_argument("-k", "--pages", type=int, default=None,
                    help="pages per mode (default: PIXELRAG_ONESHOT_PAGES)")
    ap.add_argument("--json", action="store_true", help="full rows as JSON")
    ap.add_argument("--log", default="warning")
    args = ap.parse_args()

    rag.configure_logging(args.log)
    names = [m.strip() for m in args.modes.split(",")] if args.modes else None
    rows = list(compare(args.question, names, args.pages))
    if args.json:
        json.dump({"question": args.question, "rows": rows,
                   "agreement": agreement(rows)},
                  sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        _print(rows)


if __name__ == "__main__":
    main()
