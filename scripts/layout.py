"""Where things are on disk — the one copy of PixelRAG's storage convention.

Two problems this exists to close.

CWD DEPENDENCE. Every consumer used to hold its own `Path("index")`, resolved
relative to whatever directory the process happened to start in. Run
`python scripts/ask.py` from anywhere but the repo root and retrieval found no
articles.json, with no error that said so — `articles()` simply raised
FileNotFoundError several frames from the cause. Paths here are anchored to the
repo root, which is knowable from this file's own location.

REPLICATED CONVENTION. `{article_id}.png.tiles/` and `tile_{n:04d}.jpg` are
UPSTREAM PixelRAG's layout, not ours — the most volatile external detail in the
system — and they were spelled out in eight places across rag.py, retrieve.py,
app.py, transcribe.py, build_text_index.py, prepare_hires.py and
chunk_multiscale.py, at every level from the ranking policy down to the build
scripts. When upstream renames a directory, that should be one edit.

`PIXELRAG_INDEX_DIR` moves the whole index tree; it is resolved against the repo
root, so a relative value still means what it looks like it means.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# This file lives in scripts/, so the repo root is one level up. Resolved once,
# at import, from __file__ rather than from the process working directory.
ROOT = Path(__file__).resolve().parents[1]


def _anchored(value: str | os.PathLike, default: Path) -> Path:
    """An override, resolved against the repo root when it is relative."""
    p = Path(value) if value else default
    return p if p.is_absolute() else (ROOT / p)


@dataclass(frozen=True)
class IndexLayout:
    """The index tree. Frozen because these are facts about a build, not knobs.

    Callers ask this for paths instead of composing them, so `tile_dir` is the
    only place that knows a page directory is named after the article id with a
    `.png.tiles` suffix.
    """

    index_dir: Path

    # -- directories -------------------------------------------------------

    @property
    def tiles_dir(self) -> Path:
        return self.index_dir / "tiles"

    def tile_dir(self, article_id: int) -> Path:
        """Per-article directory of rendered pages and their chunk crops."""
        return self.tiles_dir / f"{article_id}.png.tiles"

    def tile_dirs(self) -> list[Path]:
        """Every article's tile directory, in article order."""
        return sorted(self.tiles_dir.glob("*.png.tiles"),
                      key=lambda p: _article_id_of(p))

    # -- files -------------------------------------------------------------

    def page_image(self, article_id: int, tile_index: int) -> Path:
        """The rendered full-page JPEG. `tile_index` is 0-based, as on disk."""
        return self.tile_dir(article_id) / f"tile_{tile_index:04d}.jpg"

    def page_images(self, article_id: int) -> list[Path]:
        return sorted(self.tile_dir(article_id).glob("tile_*.jpg"))

    def chunks_manifest(self, article_id: int) -> Path:
        """Chunk geometry and scale, as chunk_multiscale.py writes it."""
        return self.tile_dir(article_id) / "chunks.json"

    @property
    def articles_json(self) -> Path:
        return self.index_dir / "articles.json"

    @property
    def summary_json(self) -> Path:
        return self.index_dir / "summary.json"

    @property
    def faiss_index(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def text_sidecar(self) -> Path:
        """BM25 source, built by build_text_index.py."""
        return self.index_dir / "text.json"

    @property
    def answer_cache_vectors(self) -> Path:
        return self.index_dir / "answer_cache.npz"

    @property
    def answer_cache_meta(self) -> Path:
        return self.index_dir / "answer_cache.json"

    @property
    def fit_cache(self) -> Path:
        """Resized page bytes, keyed by provider. Derived, safe to delete."""
        return self.tiles_dir / "_fit"

    @property
    def transcripts_dir(self) -> Path:
        return self.index_dir / "transcripts"

    def exists(self) -> bool:
        return self.index_dir.exists()


def _article_id_of(tile_dir: Path) -> int:
    """'3.png.tiles' -> 3. Sorts tile directories numerically rather than
    lexically, so article 10 does not land between 1 and 2."""
    try:
        return int(tile_dir.name.split(".", 1)[0])
    except ValueError:
        return -1


def default() -> IndexLayout:
    """The layout every module shares unless it was handed another one."""
    return IndexLayout(_anchored(os.environ.get("PIXELRAG_INDEX_DIR", ""),
                                 ROOT / "index"))


DEFAULT = default()
