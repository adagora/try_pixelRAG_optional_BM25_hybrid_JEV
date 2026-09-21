"""What is in the index: which documents, and where each page sits on disk.

Four accessors over `articles.json` and the rendered page images. They were in
rag.py, which meant that anything wanting a document's title — bench.py,
oracle.py, evaluate_pl.py all want exactly that and nothing else — had to
import the orchestration layer, and with it providers, jev, xray, requests and
a .env load. A question set builder does not need a reader backend to turn an
article id into a name.

`layout.py` owns where the index IS; this owns what is IN it. The split is that
`layout` answers "what path would article 3's page 7 have" without the index
existing, and this answers "how big is it" by opening the file.

THE BACKSLASH NORMALISATION IN `articles()` IS NOT COSMETIC. An index built on
Windows writes `pdfs\\foo.pdf`, and PurePosixPath treats that as one filename
containing a backslash rather than a directory and a file — so every source PDF
reads as missing, citations silently lose their geometry, and nothing raises.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import layout

# THE ONE OWNER of where the index is, for everything above layout.py. It is a
# module global rather than a constant because the test suite redirects it at a
# temporary index, and a second copy of it in another module means half the
# accessors quietly keep reading the real one — which is exactly what happened
# when these four functions lived in rag.py and it kept its own `LAYOUT`.
# Anything that needs it reads `corpus.LAYOUT` rather than importing the name.
LAYOUT = layout.DEFAULT
INDEX_DIR = LAYOUT.index_dir
TILES_DIR = LAYOUT.tiles_dir


@lru_cache(maxsize=None)
def articles() -> list[dict]:
    # Without the article list there is no corpus, so this is one of the few
    # places that fails instead of degrading. It still names the file and the
    # command that rebuilds it: the alternative is a JSONDecodeError surfacing
    # several frames up inside a request handler, which says neither.
    if not LAYOUT.articles_json.exists():
        raise FileNotFoundError(
            f"{LAYOUT.articles_json} missing — run scripts/build_index.py")
    try:
        arts = json.loads(LAYOUT.articles_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{LAYOUT.articles_json} is not valid JSON ({e}) — the index is "
            f"incomplete; rebuild it with scripts/build_index.py") from e
    # Index may have been built on Windows (`pdfs\\foo.pdf`); Path on macOS/Linux
    # treats that as a single filename with a backslash, so the source is "missing".
    for a in arts:
        url = a.get("url")
        if isinstance(url, str):
            a["url"] = url.replace("\\", "/")
    return arts


def doc_title(article_id: int) -> str:
    a = articles()
    return a[article_id]["title"] if article_id < len(a) else "?"


def page_path(article_id: int, tile_index: int) -> Path:
    return LAYOUT.page_image(article_id, tile_index)


@lru_cache(maxsize=None)
def page_size(article_id: int, tile_index: int) -> tuple[int, int]:
    from PIL import Image

    with Image.open(page_path(article_id, tile_index)) as im:
        return im.size


def source_pdf(article_id: int) -> Path | None:
    """The original PDF behind an article, if there is one on disk.

    None rather than a missing path, because every caller's next move is to
    read a text layer out of it and the answer for a scanned corpus, a moved
    file and a non-PDF source is the same: there is no text layer, degrade.
    """
    src = Path(articles()[article_id].get("url") or "")
    return src if src.exists() and src.suffix.lower() == ".pdf" else None
