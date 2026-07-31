"""The record that travels from retrieval to the reader to the browser.

It used to be a bare dict with an unwritten schema, created in
retrieve.aggregate with eight keys, grown two more by fuse(), two more by rrf(),
reshaped in rag._oneshot_pages, projected into an SSE event and finally read by
JavaScript. Nothing declared what was in it at any point, `_lexical_as_pages`
had to hand-build the shape from placeholders to fake membership, and `score`
meant a cosine or an RRF rank score depending on a flag three modules away.

Two things this fixes beyond writing the fields down.

RANK SCORES ARE SEPARATE FIELDS. `score` is the retriever's own number and is
only comparable within one retriever. `norm` is that score divided by its
variant's best, and `rrf` is the cross-retriever rank score. They were being
collapsed into one key on the way out — `p.get("rrf", p["score"])` — so a
consumer could not tell which it had. `ranking_score` says which one orders the
list and `score` stays the raw evidence, both reported.

FUSION NO LONGER MUTATES ITS INPUTS. fuse() wrote `norm` and `rank` into the
caller's dicts and rrf() added `rrf` and `found_by` to copies of them, so what a
page contained depended on which functions had already seen it. `with_` returns
a new record; the ranking functions never touch what they were handed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

VISUAL = "visual"
LEXICAL = "lexical"


@dataclass(frozen=True)
class Chunk:
    """One matching chunk of a page, with the weight aggregation gave it."""

    chunk_index: int
    score: float
    weighted: float
    scale: str


@dataclass(frozen=True)
class PageHit:
    """One candidate page, at whatever stage of ranking it has reached.

    `tile_index` is 0-based, as tiles are on disk; `page` is 1-based, as the
    reader is told, as citations quote, and as the UI shows. Both are carried
    rather than derived at each use, because deriving one from the other in the
    wrong direction is the off-by-one this codebase has already paid for once.
    """

    article_id: int
    tile_index: int

    # -- evidence ---------------------------------------------------------
    #
    # The retriever's own number. Comparable within one retriever and one
    # phrasing, and nowhere else: visual scores are cosines in a 0.3-0.7 band,
    # BM25 scores are unbounded and land near 8.
    score: float = 0.0
    chunks: tuple[Chunk, ...] = ()
    sources: frozenset[str] = frozenset({VISUAL})

    # -- ranking ----------------------------------------------------------
    #
    # `norm` is `score` over its own variant's best, which is what fuse()
    # orders by. `rrf` is the cross-retriever rank score, which is what the
    # hybrid path orders by. Whichever applies, `ranking_score` reports it.
    norm: float = 0.0
    rank: int = 0
    rrf: float | None = None

    @property
    def page(self) -> int:
        """1-based page number — what the reader cites and the UI shows."""
        return self.tile_index + 1

    @property
    def key(self) -> tuple[int, int]:
        return (self.article_id, self.tile_index)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def focus(self) -> int | None:
        """Best region chunk — the one a highlight should box.

        Never the gist: that chunk spans the whole page, so drawing it would
        mark everything and mean nothing. None when only the gist matched, or
        when the page came from the text index and has no chunk geometry.
        """
        for c in self.chunks:
            if c.scale != "page":
                return c.chunk_index
        return None

    @property
    def has_geometry(self) -> bool:
        """False for a text-only hit; citations.py falls back to page-level."""
        return bool(self.chunks)

    @property
    def ranking_score(self) -> float:
        """The number that actually orders the list this page is in."""
        return self.rrf if self.rrf is not None else self.score

    @property
    def found_by(self) -> str:
        """Provenance for display: visual, lexical, or both."""
        return "both" if len(self.sources) > 1 else next(iter(self.sources))

    def with_(self, **changes) -> PageHit:
        """A copy with fields replaced. The ranking functions never mutate."""
        return replace(self, **changes)


def from_lexical(hit: dict) -> PageHit:
    """A BM25 hit as a page candidate.

    tile_index is page-1 by the same convention chunk_multiscale.py writes, so
    a text-only hit still resolves to a rendered page image for the reader. It
    carries no chunks, which is not a placeholder: there is genuinely no region
    geometry for a page the text index found alone, and `has_geometry` is how
    the rest of the pipeline asks.
    """
    return PageHit(article_id=hit["article_id"], tile_index=hit["page"] - 1,
                   score=hit["score"], sources=frozenset({LEXICAL}))
