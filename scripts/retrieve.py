#!/usr/bin/env python
"""Page-level retrieval over PixelRAG's chunk vectors.

PixelRAG returns a ranked list of *chunks*. What the reader needs is a ranked
list of *pages*, and the two are not the same ordering. Three things happen here,
all on top of unmodified PixelRAG search:

1. AGGREGATE CHUNKS TO PAGES.
   A page whose gist vector and three of its regions all match the query is far
   more likely to be the right page than one with a single fluky region hit —
   but a flat chunk ranking cannot express that, and taking the top-k chunks
   often returns four crops of the same page while the actual answer page sits
   at rank 6. Score = best chunk + a decaying bonus for the rest, so agreement
   across a page counts without letting a page with many mediocre chunks beat a
   page with one excellent one.

2. FUSE SEVERAL PHRASINGS OF THE QUESTION.
   Measured on the real question set, terse Polish noun phrases were the failure
   mode: "Dzielony wał" and "Prowadzenie pod kątem" returned near-tied scores
   across unrelated documents. Interrogative scaffolding ("Pokaż…", "Jaka jest…",
   "Ile kosztuje…") is a large fraction of a short query's tokens and none of its
   meaning, so we also search the bare noun phrase and fuse the rankings with
   RRF. Costs one extra encode (~160ms against the sidecar), no extra GPU work.

3. WEIGHT BY CHUNK SCALE.
   chunk_multiscale.py emits a whole-page gist vector (chunk_index 0) plus
   overlapping region vectors. "Pokaż rodzaje paneli" is a question about what a
   page *is* and wants the gist; "Ile kosztuje wkładka antywłamaniowa" is a
   question about one row and wants a region. Detecting that from the question is
   guesswork, so instead both are kept and the gist gets a small bonus — enough
   to break ties toward the page that is topically about the subject.

This module does no IO and imports nothing that does. `search_fn`, `lexical_fn`
and `scale_of` are all injected, so the ranking policy — the part the README
quotes recall figures against — can be exercised with three plain functions.
"""

from __future__ import annotations

import re
from typing import Callable

import pagehit
from pagehit import Chunk, PageHit

# (article_id, tile_index, chunk_index) -> "page" | "region". See chunkmeta.
ScaleFn = Callable[[int, int, int], str]

# Fraction of each additional matching chunk's score added to a page's total.
# Low on purpose: agreement is evidence, but four half-matches must not outrank
# one clean hit. Swept over 0.0-0.35: worth about +1 recall@4 and costs about
# -1 top-1, both within noise on 20 questions. Kept because agreement is a
# principled signal, not because the sweep proved it.
AGREEMENT = 0.18

# Reweighting for whole-page gist vectors. MEASURED AT 1.0 — i.e. off.
#
# The hypothesis was that "Pokaż rodzaje paneli" is a question about what a page
# *is*, so a whole-page vector should be favoured. The sweep says otherwise: at
# 1.06 top-1 dropped from 11/20 to 9/20, and excluding gist vectors from
# retrieval entirely changed nothing at all (top-1 10/20, recall@4 17/20 either
# way). Across the whole question set the rank-1 page was won by a gist chunk
# once, and never for a page that was actually correct.
#
# Left as a named constant rather than deleted because the negative result is
# specific to this corpus — 875x1024 regions of an A4 catalogue page already
# carry the page's identity (header, product family, table shape), so the gist
# adds no information. On a corpus of dense full-bleed diagrams it might.
GIST_BONUS = 1.0

# Interrogative and imperative scaffolding, stripped to leave the noun phrase.
# Ordered longest-first so "jaka jest" wins over "jaka".
_PREFIXES = [
    "pokaz mi", "pokaz", "pokaż mi", "pokaż", "podaj mi", "podaj",
    "jaka jest", "jaki jest", "jakie sa", "jakie są", "jaka", "jaki", "jakie",
    "ile kosztuje", "ile kosztuj", "ile kosztuja", "ile kosztują", "ile",
    "czy", "gdzie", "kiedy", "wymien", "wymień", "opisz", "co to jest", "co to",
    "informacje na temat", "informacje o", "informacja o",
    "what is", "what are", "show me", "show", "how much", "list",
]
_FILLER = re.compile(
    r"^\s*(?:pokaz|pokaż|podaj|wymien|wymień|opisz)?\s*"
    r"(?:informacje|informacja)\s+(?:na\s+temat|o)\s+", re.IGNORECASE)


def noun_phrase(q: str) -> str:
    """Strip interrogative scaffolding. Returns '' when nothing is left to strip."""
    s = _FILLER.sub("", q).strip()
    low = s.lower()
    for p in _PREFIXES:
        if low.startswith(p + " "):
            s = s[len(p):].strip(" ?.,:")
            break
    s = s.strip(" ?.,:")
    return s if s and s.lower() != q.strip(" ?.,:").lower() else ""


def query_variants(q: str) -> list[str]:
    """The phrasings to search. Always includes the question as asked, first."""
    out = [q.strip()]
    np_ = noun_phrase(q)
    if np_ and len(np_) >= 3:
        out.append(np_)
    return out


def aggregate(hits: list[dict], scale_of: ScaleFn) -> list[PageHit]:
    """Chunk hits -> page-scored candidates, best first.

    Each candidate keeps the chunks that put it there, so the caller can show
    *where* on the page the match was and can pick a region crop to highlight.

    `scale_of` reports whether a chunk is the whole-page gist or one region.
    Injected rather than read off disk: this module is the ranking policy and
    the manifest that answers the question belongs to the index — see chunkmeta.
    """
    by_page: dict[tuple[int, int], list[Chunk]] = {}
    for h in hits:
        aid, ti = h["article_id"], h["tile_index"]
        scale = scale_of(aid, ti, h["chunk_index"])
        weighted = h["score"] * (GIST_BONUS if scale == "page" else 1.0)
        by_page.setdefault((aid, ti), []).append(
            Chunk(chunk_index=h["chunk_index"], score=h["score"],
                  weighted=weighted, scale=scale))

    out = []
    for (aid, ti), chunks in by_page.items():
        chunks.sort(key=lambda c: -c.weighted)
        # Best chunk carries the page; the rest contribute a decaying bonus, so
        # agreement counts without letting four half-matches beat one clean hit.
        rest = sum(c.weighted for c in chunks[1:])
        out.append(PageHit(article_id=aid, tile_index=ti,
                           score=chunks[0].weighted + AGREEMENT * rest,
                           chunks=tuple(chunks)))
    out.sort(key=lambda p: -p.score)
    return out


def fuse(rankings: list[list[PageHit]]) -> list[PageHit]:
    """Fuse per-variant page rankings by best normalised score.

    RRF was the first thing tried here and it is the wrong tool for this case.
    Measured on the real question set, rank fusion of two phrasings bought +2
    recall@4 but cost -2 top-1: it discards score magnitude, so a page that one
    variant matched weakly at rank 1 outranks a page the other variant matched
    decisively. With only two variants there are not enough voters for ranks to
    carry more information than the scores do.

    Normalising each variant's scores by that variant's own best score fixes the
    scale problem RRF exists to solve — a two-word query scores lower against
    everything, so its absolute numbers are not comparable — while keeping the
    ordering that strength of evidence implies. A page found by only one variant
    still enters the candidate set, which is where the recall gain came from.
    """
    merged: dict[tuple[int, int], PageHit] = {}
    for ranking in rankings:
        top = max((p.score for p in ranking), default=0.0) or 1.0
        for i, p in enumerate(ranking):
            scored = p.with_(norm=p.score / top, rank=i)
            cur = merged.get(scored.key)
            # Strictly greater, so a tie keeps the earlier variant — variant 0
            # is the question as asked, and that is the ranking to prefer.
            if cur is None or scored.norm > cur.norm:
                # The whole record moves together: keeping the strongest
                # evidence means keeping the chunk list that produced it, and
                # the focus region that drives the highlight box comes from it.
                merged[scored.key] = scored
    out = sorted(merged.values(), key=lambda p: -p.norm)
    return out


# --------------------------------------------------------------------------
# hybrid: fusing the pixel index with the text layer
# --------------------------------------------------------------------------

# RRF is used HERE and score-normalisation is used in fuse() above, and the
# difference is not inconsistency — it is the two cases pulling opposite ways.
#
# fuse() merges two phrasings of ONE retriever. Its scores are the same encoder's
# cosines, so magnitude is real information and discarding it costs top-1.
#
# This fuses TWO retrievers whose scores share no scale, no distribution and no
# units: PixelRAG cosines land in a compressed 0.35-0.45 band, BM25 scores are
# unbounded and heavy-tailed (a strong rank-1 routinely triples rank-2).
# Normalise-by-max on those gives BM25's rank-2 a norm of ~0.3 while PixelRAG's
# rank-2 sits at ~0.97, so the visual list wins every slot below rank 1 for
# reasons that are entirely an artefact of score shape. Ranks are the only
# comparable quantity, which is exactly the case RRF was designed for.
RRF_K = 60

# Weight on the lexical list. 1.0 = the two retrievers vote equally.
# Swept on the real question set — see evaluate_pl.py --sweep.
LEX_WEIGHT = 1.0

# How deep to read each retriever's list before fusing. Deeper than n_pages on
# purpose: a page that RRF should promote to the top 4 has to be IN both lists
# first, and the whole point is the page one retriever ranked 7th.
FUSE_DEPTH = 20


def rrf(rankings: list[list[PageHit]], weights: list[float] | None = None,
        k: int = RRF_K) -> list[PageHit]:
    """Reciprocal-rank fusion across retrievers with incomparable scores.

    Each list contributes weight/(k + rank) to every page it returns. k=60 is
    the standard damping constant: it flattens the difference between ranks 1
    and 2 enough that one retriever's confident-but-wrong top hit cannot
    outvote agreement further down.
    """
    weights = weights or [1.0] * len(rankings)
    merged: dict[tuple[int, int], PageHit] = {}
    for w, ranking in zip(weights, rankings):
        for i, p in enumerate(ranking):
            cur = merged.get(p.key)
            if cur is None:
                merged[p.key] = p.with_(rrf=w / (k + i + 1))
                continue
            merged[p.key] = cur.with_(
                rrf=cur.rrf + w / (k + i + 1),
                sources=cur.sources | p.sources,
                # Keep whichever list carried chunk geometry. A page the text
                # index found alone has no region to box, and citations.py
                # already falls back to page-level highlighting for that case.
                chunks=cur.chunks or p.chunks,
            )
    return sorted(merged.values(), key=lambda p: -p.rrf)


def retrieve_pages(search_fn, question: str, scale_of: ScaleFn,
                   n_pages: int = 4, per_query: int = 24,
                   lexical_fn=None, lex_weight: float = LEX_WEIGHT,
                   fuse_depth: int = FUSE_DEPTH
                   ) -> tuple[list[PageHit], list[dict]]:
    """Full retrieval: variants -> chunk search -> page aggregate -> fuse.

    `search_fn(query, n_results)` is injected rather than imported so this stays
    testable and so rag.py keeps sole ownership of the encoder/sidecar path.

    `scale_of(article_id, tile_index, chunk_index)` reports gist versus region;
    chunkmeta.scale_of is the production one.

    `lexical_fn(query, n)` is optional and turns this hybrid. Passing it in the
    same way keeps the text layer a caller's choice, not a hard dependency: a
    corpus of scans has no text layer to fuse and should not pay for the import.

    Returns (top pages, per-retriever debug rows).
    """
    variants = query_variants(question)

    # Concurrently, because the two phrasings are independent and each costs an
    # encode: sequentially that is ~2x140ms of pure wall clock on the critical
    # path, for work that has no reason to be ordered. The encoder sidecar is an
    # HTTP call so the GIL is released; the in-process fallback queues onto its
    # own model thread and simply serialises, which is correct rather than fast.
    # executor.map preserves order, which fuse() depends on — variant 0 is the
    # question as asked and carries the ranking the others are compared against.
    if len(variants) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(variants)) as pool:
            hit_lists = list(pool.map(lambda v: search_fn(v, per_query), variants))
    else:
        hit_lists = [search_fn(variants[0], per_query)]

    rankings, debug = [], []
    for v, hits in zip(variants, hit_lists):
        ranked = aggregate(hits, scale_of)
        rankings.append(ranked)
        debug.append(_debug_row(v, "visual", ranked))
    visual = fuse(rankings)
    if lexical_fn is None:
        return visual[:n_pages], debug

    lex = [pagehit.from_lexical(h) for h in lexical_fn(question, fuse_depth)]
    debug.append(_debug_row(question, "lexical", lex))
    fused = rrf([visual[:fuse_depth], lex], weights=[1.0, lex_weight])
    return fused[:n_pages], debug


def _debug_row(query: str, retriever: str, ranked: list[PageHit]) -> dict:
    """What one retriever saw, for the UI's search trace. `score` here is that
    retriever's own number — see PageHit on why it is not comparable across
    retrievers, and why the fused list reports ranking_score instead."""
    return {
        "query": query, "retriever": retriever,
        "top": [{"article_id": p.article_id, "page": p.page,
                 "score": round(p.score, 4), "n_chunks": p.n_chunks}
                for p in ranked[:5]],
    }
