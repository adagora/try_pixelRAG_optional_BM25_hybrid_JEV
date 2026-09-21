#!/usr/bin/env python
"""Re-sweep the ranking constants against the corpus that is actually indexed.

`retrieve.AGREEMENT` and `retrieve.LEX_WEIGHT` carry comments quoting sweeps
over "20 questions" — a question set built for a corpus (`Cennik - Bramy
garazowe`) this index no longer contains. The README says so and says nothing
reproduces them. `scripts/oracle.py` has since built a question set for the
corpus that IS here, so the sweeps can be re-run instead of disclaimed, and
that is all this file is.

WHY IT IS FAST, WHICH IS ALSO WHY IT IS EXACT. Neither constant touches
retrieval. `AGREEMENT` weights chunks that a search already returned;
`LEX_WEIGHT` weights a BM25 list that was already fetched. So every query is
encoded and searched ONCE, the hit lists are kept, and each parameter value
replays `aggregate`/`fuse`/`rrf` over the same hits. A 15-point sweep costs
one search per question, not fifteen — and no value is compared against a
different draw of the same randomness, because there is none.

Only `visual` and `hybrid` are swept, on purpose. They are the two modes
bench.py grades fairly (pixels against text, and string match), so a winner
read out of this table is not a model agreeing with itself. The Jev modes
would also cost a call per question per value.

    .venv/bin/python scripts/sweep_ranking.py           # both, ~13 searches
    .venv/bin/python scripts/sweep_ranking.py --json

Requires `pixelrag serve` on --port. No TypeSafe key, no reader, no money.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import bench
import pagehit
import rag
import retrieve

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Both grids straddle the shipped value so a flat line is visible as flat
# rather than as an edge of the range.
AGREEMENT_GRID = [0.0, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.50]
LEX_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


@dataclass
class Cached:
    """One question's retrieval, fetched once and replayed at every value."""

    q: str
    gold: set[tuple[int, int]]
    primary: tuple[int, int] | None
    hit_lists: list[list[dict]]
    lex_rows: list[dict]


def collect(questions, t2i, k: int, want_lexical: bool) -> list[Cached]:
    """Search each question once. This is the only part that costs anything."""
    lexical = rag._lexical_fn(required=True) if want_lexical else None
    if want_lexical and lexical is None:
        sys.exit("no index/text.json — build it with scripts/build_text_index.py")
    search = rag.searcher("visual")
    per_query = rag._per_query(k)

    out = []
    for q in questions:
        aid = t2i.get(q.doc)
        if aid is None:
            continue
        variants = retrieve.query_variants(q.q)
        out.append(Cached(
            q=q.q,
            gold={(aid, p) for p in q.pages},
            primary=(aid, q.primary) if q.primary is not None else None,
            hit_lists=[search(v, per_query) for v in variants],
            lex_rows=lexical(q.q, retrieve.FUSE_DEPTH) if lexical else [],
        ))
    return out


def rank(c: Cached, agreement: float, lex_weight: float | None,
         k: int) -> list[tuple[int, int]]:
    """Replay the ranking policy at one parameter setting. No IO."""
    before = retrieve.AGREEMENT
    try:
        retrieve.AGREEMENT = agreement
        visual = retrieve.fuse([retrieve.aggregate(h, rag._scale_of)
                                for h in c.hit_lists])
    finally:
        retrieve.AGREEMENT = before

    if lex_weight is None:
        pages = visual[:k]
    else:
        lex = [pagehit.from_lexical(h) for h in c.lex_rows]
        pages = retrieve.rrf([visual[:retrieve.FUSE_DEPTH], lex],
                             weights=[1.0, lex_weight])[:k]
    return [(p.article_id, p.page) for p in pages]


def score(cached: list[Cached], agreement: float, lex_weight: float | None,
          k: int) -> dict:
    """bench.py's grading, to the letter — top-1 on `primary`, recall on any gold."""
    top1 = recall = covered = ceiling = 0
    for c in cached:
        got = rank(c, agreement, lex_weight, k)
        hit = [g for g in got if g in c.gold]
        covered += len(hit)
        ceiling += min(len(c.gold), k)
        if hit:
            recall += 1
        if c.primary is not None and got and got[0] == c.primary:
            top1 += 1
    n = len(cached)
    return {"n": n, "top1": top1, "recall": recall,
            "cover": covered, "ceiling": ceiling,
            "top1_pct": round(100 * top1 / n, 1) if n else None,
            "recall_pct": round(100 * recall / n, 1) if n else None}


def table(title: str, shipped: float, rows: list[tuple[float, dict, dict]]) -> None:
    # Sizes come off the rows. They were hardcoded once, and stayed reading
    # "13" and "23" after the question set tripled — which is the exact failure
    # this whole file exists to have caught.
    n_ver = rows[0][1]["n"] if rows else 0
    n_ora = rows[0][2]["n"] if rows else 0
    print(f"\n{title}")
    print(f"  {'value':>7}  {'top-1':>12}  {'recall':>12}  "
          f"{'top-1':>12}  {'recall':>12}")
    print(f"  {'':>7}  {f'--- verified ({n_ver}) ---':^26}  "
          f"{f'--- oracle-labelled ({n_ora}) ---':^26}")
    for value, v, o in rows:
        mark = " <- shipped" if value == shipped else ""
        print(f"  {value:>7}  {v['top1']:>5}/{v['n']:<6}  "
              f"{v['recall']:>5}/{v['n']:<6}  "
              f"{o['top1']:>5}/{o['n']:<6}  {o['recall']:>5}/{o['n']:<6}{mark}")


def joint(cached: list[Cached], k: int) -> None:
    """Both constants at once, top-1 on the verified set.

    Sweeping one at the other's shipped value is only sound if they do not
    interact, and that is an assumption rather than a finding — a fused ranking
    mixes a visual list whose order AGREEMENT sets with a lexical list whose
    weight LEX_WEIGHT sets, so they plainly can. The grid is free: the hits are
    already in memory and every cell is a replay.
    """
    print(f"\nJOINT — `hybrid` top-1 of {len(cached)}, k={k}")
    print("            " + "".join(f"{w:>7}" for w in LEX_WEIGHT_GRID)
          + "   <- LEX_WEIGHT")
    best = (-1, None, None)
    for a in AGREEMENT_GRID:
        cells = [score(cached, a, w, k)["top1"] for w in LEX_WEIGHT_GRID]
        for w, c in zip(LEX_WEIGHT_GRID, cells):
            if c > best[0]:
                best = (c, a, w)
        print(f"  A={a:<7}" + "".join(f"{c:>7}" for c in cells))
    print(f"\n  best cell: AGREEMENT={best[1]}, LEX_WEIGHT={best[2]} "
          f"-> {best[0]}/{len(cached)} top-1")
    print(f"  shipped:   AGREEMENT={retrieve.AGREEMENT}, "
          f"LEX_WEIGHT={retrieve.LEX_WEIGHT} -> "
          f"{score(cached, retrieve.AGREEMENT, retrieve.LEX_WEIGHT, k)['top1']}"
          f"/{len(cached)} top-1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-k", "--pages", type=int, default=4)
    ap.add_argument("--joint", action="store_true",
                    help="2D grid over both constants — they can interact")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--log", default="warning")
    args = ap.parse_args()
    rag.configure_logging(args.log)

    t2i = bench.title_to_id()
    everything = bench.load()
    answerable = [q for q in everything if q.answerable]
    ver = bench.verified(everything, t2i)

    print(f"searching {len(ver)} verified + {len(answerable)} oracle-labelled "
          f"questions once each …", file=sys.stderr)
    cached_ver = collect(ver, t2i, args.pages, want_lexical=True)
    cached_ora = collect(answerable, t2i, args.pages, want_lexical=True)

    agreement_rows = [
        (a,
         score(cached_ver, a, None, args.pages),
         score(cached_ora, a, None, args.pages))
        for a in AGREEMENT_GRID
    ]
    lex_rows = [
        (w,
         score(cached_ver, retrieve.AGREEMENT, w, args.pages),
         score(cached_ora, retrieve.AGREEMENT, w, args.pages))
        for w in LEX_WEIGHT_GRID
    ]

    if args.json:
        print(json.dumps({
            "k": args.pages,
            "shipped": {"AGREEMENT": retrieve.AGREEMENT,
                        "LEX_WEIGHT": retrieve.LEX_WEIGHT,
                        "GIST_BONUS": retrieve.GIST_BONUS},
            "agreement": [{"value": a, "verified": v, "oracle": o}
                          for a, v, o in agreement_rows],
            "lex_weight": [{"value": w, "verified": v, "oracle": o}
                           for w, v, o in lex_rows],
        }, indent=2))
        return 0

    if args.joint:
        joint(cached_ver, args.pages)
        return 0

    table(f"AGREEMENT — `visual`, k={args.pages} (no BM25 in this mode)",
          retrieve.AGREEMENT, agreement_rows)
    table(f"LEX_WEIGHT — `hybrid`, k={args.pages}, "
          f"AGREEMENT={retrieve.AGREEMENT}",
          retrieve.LEX_WEIGHT, lex_rows)

    print("\n  verified: gold decided by literal string match — no model has an "
          "opinion.\n  oracle-labelled: gold from the Jev sweep; fair for these "
          "two modes\n  (pixels and BM25), circular for the Jev ones, which is "
          "why they are absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
