#!/usr/bin/env python
"""Measure retrieval on the real question set — and A/B the changes that claim to help.

Retrieval only: no reader, no API key, no tokens. If the right page is not in the
candidate set, no prompt can save the answer, so this is the number to move first.

    .venv/Scripts/python.exe scripts/evaluate_pl.py            # current config
    .venv/Scripts/python.exe scripts/evaluate_pl.py --ab       # vs stock chunk ranking
    .venv/Scripts/python.exe scripts/evaluate_pl.py -k 3

Ground truth lives in eval/questions_pl.yaml keyed by document *title*, because
article_ids are positions in the sorted pdfs/ glob and shift on every corpus edit.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import rag
import retrieve
import yaml

# The verbose table prints ✓/~/✗ and Polish page titles. Windows consoles
# default to cp1250 and the tick alone crashes the run after the numbers have
# already been computed — losing the whole eval to a glyph.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

EVAL = Path("eval/questions_pl.yaml")


CORPUS_DOC = "Cennik - Bramy garażowe"


def _title_to_id() -> dict[str, int]:
    return {a["title"]: i for i, a in enumerate(rag.articles())}


def _gold(q: dict, t2i: dict[str, int]) -> tuple[set[tuple[int, int]], tuple | None]:
    """(every acceptable page, the single best page).

    Ground truth is stated as page numbers within the main catalogue, so a
    corpus edit that shifts article_ids does not invalidate the file. `q["doc"]`
    overrides the default document for a question that lives elsewhere.
    """
    aid = t2i.get(str(q.get("doc", CORPUS_DOC)))
    if aid is None:
        return set(), None
    pages = {(aid, int(p)) for p in (q.get("pages") or [])}
    prim = (aid, int(q["primary"])) if q.get("primary") else None
    if prim:
        pages.add(prim)
    return pages, prim


def _baseline(question: str, k: int, depth: int | None = None) -> list[tuple[int, int]]:
    """Stock behaviour: take the top-k chunks, dedupe to pages, keep order.

    This is what the system did before retrieve.py existed, and it is the
    honest thing to compare against.

    CAVEAT, and it is a real one: at the default depth this reads 12 chunks
    while `_improved` reads 24 per variant across 2 variants — 4x the candidate
    pool. Part of any gap between them is pool depth, not the aggregation
    method being tested. `--ab` therefore runs this at both depths.
    """
    pages, seen = [], set()
    for h in rag.searcher("visual")(question, depth or max(k * 3, 12)):
        key = (h["article_id"], h["tile_index"] + 1)
        if key in seen:
            continue
        seen.add(key)
        pages.append(key)
        if len(pages) >= k:
            break
    return pages


def _improved(question: str, k: int) -> list[tuple[int, int]]:
    # `rag.searcher("visual")`, not `rag.search`: the latter follows whatever
    # PIXELRAG_JEV/PIXELRAG_HYBRID are set to, and a strategy in this file has
    # to be the strategy it is named after or the table means nothing.
    ranked, _ = retrieve.retrieve_pages(
        rag.searcher("visual"), question, rag._scale_of, n_pages=k,
        per_query=rag._per_query(k))
    return [(p.article_id, p.page) for p in ranked]


def _lexical(question: str, k: int) -> list[tuple[int, int]]:
    """BM25 over the PDF text layer, ALONE — an ablation, not a reference point.

    This exists to answer one question honestly: how much of the retrieval is
    PixelRAG earning, and how much would `grep` have got for free? The answer
    turned out to be "neither one is enough", so this is no longer a rival
    outside the system — it is one half of it (see `_hybrid`). Kept as a
    single-retriever ablation so the fusion has to keep justifying itself.

    It cannot see the 3 pages in this corpus with no text layer, which is exactly
    where a pixel index should win, and it is blind to 2-D price grids.

    Note it is a *tuned* baseline, not plain BM25: Polish diacritic folding,
    suffix stripping, a corpus-specific stop list and modified idf smoothing.
    Comparing it against PixelRAG-without-the-LoRA flatters the text side.
    """
    import lexical

    return [(h["article_id"], h["page"]) for h in lexical.search_text(question, k)]


def _hybrid(question: str, k: int, w: float = retrieve.LEX_WEIGHT
            ) -> list[tuple[int, int]]:
    """Both retrievers, fused by rank. The thing this eval exists to justify.

    Neither list alone reaches every question: measured here, the pixel index
    and the text layer miss DISJOINT sets, so the union is the whole set and
    the only question is whether rank fusion can actually surface it at k.
    """
    import lexical

    ranked, _ = retrieve.retrieve_pages(
        rag.searcher("visual"), question, rag._scale_of, n_pages=k,
        per_query=rag._per_query(k),
        lexical_fn=lexical.search_text, lex_weight=w)
    return [(p.article_id, p.page) for p in ranked]


def _jev(question: str, k: int, mode: str = "jev") -> list[tuple[int, int]]:
    """A Jev mode, scored like any other strategy. COSTS MONEY — see --jev.

    Every other strategy in this file is local CPU, so the whole eval could be
    re-run on a whim. These are not: `jev-expand` spends one TypeSafe call per
    question and `jev` spends two, the second carrying the whole candidate pool
    as text. That is the only reason they are behind a flag rather than in the
    default run — not because they are less interesting, but because a sweep
    that silently bills per question is a trap.

    Goes through rag.retrieve_for, which is the same call the app answers with.
    """
    ranked, _, _ = rag.retrieve_for(question, mode, n_pages=k)
    return [(p.article_id, p.page) for p in ranked]


def _oracle(question: str, k: int) -> list[tuple[int, int]]:
    """Upper bound: everything either retriever put in its top k.

    Not a system anyone can build — it returns up to 2k pages, and picking
    which k of them to keep is the entire problem fusion has to solve. It is
    here to separate "the evidence is not in the candidate pool" from "the
    evidence is there and the ranking buried it", which need different fixes.
    """
    seen, out = set(), []
    for page in _improved(question, k) + _lexical(question, k):
        if page not in seen:
            seen.add(page)
            out.append(page)
    return out


def run(qs: list[dict], k: int, fn, t2i) -> dict:
    per_kind: dict[str, list[bool]] = defaultdict(list)
    per_kind_cov: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rows, hit1, hitk, n = [], 0, 0, 0
    cov_sum = ceil_sum = 0.0
    absent_flagged = absent_total = 0

    for q in qs:
        gold, prim = _gold(q, t2i)
        got = fn(q["q"], k)
        kind = q.get("kind", "spec")

        if kind == "absent" or not gold:
            absent_total += 1
            rows.append((q["q"], kind, None, got))
            continue

        n += 1
        # top-1 counts any acceptable page in slot 1, not only `primary`: when an
        # option is priced identically in four families' tables, ranking a
        # different one first is not an error.
        top1 = bool(got) and got[0] in gold
        ink = any(g in gold for g in got)

        # COVERAGE, and it measures a different failure from recall.
        #
        # recall@k asks "did we surface AT LEAST ONE acceptable page". That is
        # the right question for "ile kosztuje X" — one page carries the number.
        # It is the wrong question for "pakiet antywłamaniowy", whose answer is
        # spread over the descriptive pages AND one options-table row per product
        # family. Measured: retrieval scored a clean recall ✓ on that question
        # while surfacing 3 of the 7 pages the answer needs, and the metric could
        # not tell the difference between that and finding all 7.
        #
        # `ceiling` is reported next to it because coverage is capped by k: a
        # question with 8 acceptable pages cannot exceed 4/8 at k=4. Without the
        # ceiling a low coverage number looks like a retrieval failure when it is
        # arithmetic. Read the gap between coverage and ceiling, not coverage.
        cov = len(set(got) & gold) / len(gold)
        ceiling = min(k, len(gold)) / len(gold)
        cov_sum += cov
        ceil_sum += ceiling

        hit1 += top1
        hitk += ink
        per_kind[kind].append(ink)
        per_kind_cov[kind].append((cov, ceiling))
        rows.append((q["q"], kind, (top1, ink), got))

    return {"n": n, "top1": hit1, "recall": hitk, "per_kind": dict(per_kind),
            "coverage": cov_sum, "ceiling": ceil_sum,
            "per_kind_cov": dict(per_kind_cov),
            "rows": rows, "absent_total": absent_total,
            "absent_flagged": absent_flagged}


def show(name: str, res: dict, k: int, t2i: dict[str, int], verbose: bool) -> None:
    i2t = {v: kk for kk, v in t2i.items()}
    n = res["n"] or 1
    print(f"\n=== {name} ===")
    print(f"  top-1  {res['top1']}/{res['n']}  ({100*res['top1']/n:.0f}%)")
    print(f"  recall@{k}  {res['recall']}/{res['n']}  ({100*res['recall']/n:.0f}%)")
    cov, ceil = 100 * res["coverage"] / n, 100 * res["ceiling"] / n
    print(f"  coverage@{k}  {cov:.0f}%  (ceiling at k={k}: {ceil:.0f}%"
          f" — the gap is what retrieval lost)")
    ck = res.get("per_kind_cov", {})
    for kind, vals in sorted(res["per_kind"].items()):
        cv = ck.get(kind, [])
        extra = ""
        if cv:
            c = 100 * sum(a for a, _ in cv) / len(cv)
            t = 100 * sum(b for _, b in cv) / len(cv)
            extra = f"   coverage {c:3.0f}% / {t:3.0f}%"
        print(f"      {kind:11s} {sum(vals)}/{len(vals)}{extra}")
    if res["absent_total"]:
        print(f"  (+{res['absent_total']} question(s) with no answer in the corpus — "
              f"reader must refuse; not scored here)")
    if not verbose:
        return
    print()
    for q, kind, score, got in res["rows"]:
        if score is None:
            mark, detail = "—", "brak odpowiedzi w korpusie"
        else:
            top1, ink = score
            mark = "✓" if top1 else ("~" if ink else "✗")
            detail = ", ".join(f"{i2t.get(a,'?')} s.{p}" for a, p in got[:k])
        print(f"  {mark} {q[:58]:58s} [{kind:10s}] {detail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=4, help="pages the reader is shown")
    ap.add_argument("--ab", action="store_true",
                    help="also run the stock chunk-ranking baseline")
    ap.add_argument("--lexical", action="store_true",
                    help="also run a BM25-over-text-layer reference point "
                         "(not part of the system; needs index/text.json)")
    ap.add_argument("--hybrid", action="store_true",
                    help="also run visual+lexical rank fusion, and the oracle "
                         "union that bounds what fusion could ever reach")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the lexical weight in the hybrid fusion")
    ap.add_argument("--jev", action="store_true",
                    help="also run the Jev modes. COSTS MONEY: one TypeSafe "
                         "call per question for jev-expand, two for jev — the "
                         "second carries the candidate pool as text. Run it to "
                         "decide whether the reranker earns that bill")
    ap.add_argument("-q", "--quiet", action="store_true", help="totals only")
    args = ap.parse_args()

    if not EVAL.exists():
        sys.exit(f"{EVAL} missing")
    qs = yaml.safe_load(EVAL.read_text(encoding="utf-8"))
    t2i = _title_to_id()

    missing = {str(q.get("doc", CORPUS_DOC)) for q in qs} - set(t2i)
    if missing:
        sys.exit(f"Ground truth names documents not in the index: "
                 f"{', '.join(sorted(missing))}\n"
                 f"Indexed: {', '.join(sorted(t2i))}\n"
                 f"Every question would score 0 — fix eval/questions_pl.yaml or "
                 f"rebuild the index before trusting any number here.")

    try:
        rag.searcher("visual")("rozgrzewka", 1)
    except Exception as e:
        sys.exit(f"Search API unreachable on {rag.SEARCH_API}: {e}\n"
                 f"Start it first (see README).")

    if args.ab:
        show("stock: top-k chunks, deduped to pages",
             run(qs, args.k, _baseline, t2i), args.k, t2i, not args.quiet)
        # Same method, matched candidate pool. Any gap that survives this is
        # attributable to page aggregation; the rest was pool depth.
        show("stock, matched pool (48 chunks — controls for candidate depth)",
             run(qs, args.k, lambda q, k: _baseline(q, k, depth=48), t2i),
             args.k, t2i, not args.quiet)
    show("page-aggregated + multi-query (retrieve.py)",
         run(qs, args.k, _improved, t2i), args.k, t2i, not args.quiet)
    if args.lexical or args.hybrid or args.sweep:
        try:
            show("ablation: BM25 over text layer alone (half of the hybrid)",
                 run(qs, args.k, _lexical, t2i), args.k, t2i, not args.quiet)
        except FileNotFoundError as e:
            print(f"\n(skipped lexical reference: {e})")
            return

    if args.sweep:
        print("\n=== lexical weight sweep (hybrid RRF) ===")
        print(f"  {'w':>5}  {'top-1':>7}  {'recall@'+str(args.k):>9}")
        for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            r = run(qs, args.k, lambda q, k, w=w: _hybrid(q, k, w), t2i)
            print(f"  {w:5.2f}  {r['top1']:3d}/{r['n']:<3d}  {r['recall']:5d}/{r['n']:<3d}")

    if args.jev:
        for mode in ("jev-expand", "jev", "jev+hybrid"):
            blocked = rag.retrieval_blocked(mode)
            if blocked:
                print(f"\n(skipped {mode}: {blocked})")
                continue
            show(f"{mode} (rag.retrieve_for — the path the app answers with)",
                 run(qs, args.k, lambda q, k, m=mode: _jev(q, k, m), t2i),
                 args.k, t2i, not args.quiet)

    if args.hybrid:
        show(f"HYBRID: visual + lexical, RRF (lex_weight={retrieve.LEX_WEIGHT})",
             run(qs, args.k, _hybrid, t2i), args.k, t2i, not args.quiet)
        show(f"oracle union of both top-{args.k} (upper bound, not a system)",
             run(qs, args.k, _oracle, t2i), args.k, t2i, False)


if __name__ == "__main__":
    main()
