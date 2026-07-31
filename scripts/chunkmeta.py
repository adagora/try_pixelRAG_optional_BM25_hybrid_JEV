"""Chunk geometry and scale, read once from the index.

chunk_multiscale.py writes one `chunks.json` per article describing every vector
in the index: where the crop sits on the page, and whether it is the whole-page
gist or one region. Two consumers wanted complementary halves of that record and
each grew its own reader — rag.py took the boxes for highlight overlays,
retrieve.py took the scale for gist weighting — so the same file was opened
twice, parsed twice, and had its "manifest missing or corrupt" case handled
twice, in modules three levels apart.

One reader, one record, one degradation rule: a missing or unparseable manifest
yields an empty mapping, because an index built before `scale` was recorded must
still rank and still answer. Callers fall back to sensible defaults rather than
failing — see `scale_of`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import layout

# Whole-page vector versus one region of it. chunk_multiscale.py emits a gist at
# chunk_index 0 followed by overlapping region crops.
GIST = "page"
REGION = "region"


@dataclass(frozen=True)
class Chunk:
    """One indexed vector's provenance on the page. Pixels, page coordinates."""

    tile_index: int
    chunk_index: int
    scale: str
    x: int
    y: int
    width: int
    height: int

    @property
    def is_gist(self) -> bool:
        return self.scale == GIST

    @property
    def has_box(self) -> bool:
        """A zero-area crop cannot be drawn, and the stock PDF path emits some."""
        return bool(self.width and self.height)


@lru_cache(maxsize=None)
def _for_article(index_dir: str, article_id: int) -> dict[tuple[int, int], Chunk]:
    # Keyed on the directory string rather than the IndexLayout so the cache
    # stays hashable and a test pointing at a temporary index gets its own entry.
    manifest = layout.IndexLayout(Path(index_dir)).chunks_manifest(article_id)
    if not manifest.exists():
        return {}
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8")).get("chunks", [])
    except (json.JSONDecodeError, OSError, AttributeError):
        return {}

    out: dict[tuple[int, int], Chunk] = {}
    for c in raw:
        ti = c.get("tile_index", 0)
        ci = c.get("chunk_index", 0)
        out[(ti, ci)] = Chunk(
            tile_index=ti,
            chunk_index=ci,
            # Indexes built before `scale` was recorded follow the convention
            # chunk_multiscale.py has always written: chunk 0 is the gist.
            scale=c.get("scale", GIST if ci == 0 else REGION),
            # chunk.py records x_offset; the stock PDF path omits it because its
            # single chunk always starts at x=0.
            x=c.get("x_offset", 0),
            y=c.get("y_offset", 0),
            width=c.get("width", 0),
            height=c.get("height", 0),
        )
    return out


def for_article(article_id: int,
                index: layout.IndexLayout | None = None) -> dict[tuple[int, int], Chunk]:
    """(tile_index, chunk_index) -> Chunk for one article. Empty if unavailable."""
    index = index or layout.DEFAULT
    return _for_article(str(index.index_dir), article_id)


def get(article_id: int, tile_index: int, chunk_index: int,
        index: layout.IndexLayout | None = None) -> Chunk | None:
    return for_article(article_id, index).get((tile_index, chunk_index))


def scale_of(article_id: int, tile_index: int, chunk_index: int,
             index: layout.IndexLayout | None = None) -> str:
    """Gist or region, defaulting to region when the manifest says nothing.

    Region is the safe default: it costs a hit the small gist bonus rather than
    granting one it did not earn, and it keeps the chunk eligible to be a
    highlight box, which a gist is not.
    """
    c = get(article_id, tile_index, chunk_index, index)
    return c.scale if c else REGION
