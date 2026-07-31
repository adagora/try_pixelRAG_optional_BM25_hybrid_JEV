#!/usr/bin/env python
"""Pre-render PDFs so the stock pipeline indexes them at full resolution.

WHY THIS EXISTS
---------------
PixelRAG's PDF path costs you half your resolution and most of your retrieval
granularity, for reasons that are invisible from the config file:

  1. render/backends/pdf.py writes its own chunks.json at render time,
     specifically so embed/chunk.py will skip the page ("each PDF page is
     already a natural semantic unit"). One page becomes one vector.
  2. embed/embed_cpu.py then runs every image through _clamp_width(875),
     shrinking a 1653px-wide A4 page to 868px. Effective DPI: 200 -> 105.

Left alone, a dense technical drawing is embedded as a single vector at half
resolution, at an image width the model was never trained on.

Remove that pre-written chunks.json and chunk.py handles the page normally:
it splits it into a grid of <=1024px-tall x <=875px-wide pieces. Every piece
is already inside the clamp, so no downscaling happens — full 200 DPI — and
one A4 page yields 6 vectors instead of 1.

HOW IT WORKS
------------
Renders tiles exactly where the pipeline expects them, then deletes the
chunks.json. `pixelrag index build` then sees tiles whose manifest `source`
matches the document at that position, skips re-rendering (_needs_render),
and proceeds to chunk / embed / index normally.

Nothing here is forked or patched. To go back to stock behaviour, skip this
script and run `pixelrag index build --force`.
"""

import json
import shutil
import sys
from pathlib import Path

import yaml
from pixelrag_render.backends.pdf import render_pdf

import layout


def _is_stale(tile_dir: Path, pdf: Path) -> bool:
    """True if tile_dir holds pixels from a different document than `pdf`.

    Tile directories are named by the document's *position* in the sorted
    glob, so adding or renaming a PDF shifts every later position. Reusing a
    directory on position alone would pair one document's pixels with another
    document's metadata — silently, and with plausible-looking results. This
    mirrors the check `pipelines._needs_render` does upstream.
    """
    manifest = tile_dir / "tiles.json"
    if not manifest.exists():
        return True
    try:
        recorded = json.loads(manifest.read_text()).get("source")
    except (json.JSONDecodeError, OSError):
        return True  # corrupt manifest — re-render rather than trust it
    return recorded != str(pdf)


def main():
    cfg = yaml.safe_load((layout.ROOT / "pixelrag.yaml").read_text())
    src = Path(cfg["source"]["path"]).expanduser()
    # The index tree comes from pixelrag.yaml, not from PIXELRAG_INDEX_DIR:
    # `pixelrag index` reads that file too, and the two must agree about where
    # they are writing. Only the naming inside it is layout's business.
    index = layout.IndexLayout(layout._anchored(cfg.get("output", "./index"),
                                                layout.ROOT / "index"))
    tiles_dir = index.tiles_dir
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Must match PDFSource exactly: sorted(path.glob("**/*.pdf")). The tile
    # directory name is the document's *position* in this list, so the
    # ordering here has to be identical or the pipeline will re-render.
    pdfs = sorted(src.glob("**/*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {src}/ — add some and re-run.")

    print(f"Pre-rendering {len(pdfs)} PDF(s) at full resolution\n")
    total_pages = 0
    for idx, pdf in enumerate(pdfs):
        tile_dir = index.tile_dir(idx)

        if _is_stale(tile_dir, pdf):
            # Drop the whole directory, not just the manifest: leftover tiles
            # from the previous occupant would otherwise be chunked and
            # embedded alongside the new ones.
            if tile_dir.exists():
                print(f"  [{idx}] stale — re-rendering", flush=True)
                shutil.rmtree(tile_dir)
            render_pdf(str(pdf), str(tiles_dir), stem=str(idx))

        # render_pdf records `source` as the string it was handed; the
        # pipeline compares that against doc.path from PDFSource. Both are
        # str(pdf) from the same glob, so they match and rendering is skipped.
        # Drop chunks.json again in case render_pdf just recreated it.
        (tile_dir / "chunks.json").unlink(missing_ok=True)

        pages = len(list(tile_dir.glob("tile_*.jpg")))
        total_pages += pages
        print(f"  [{idx}] {pdf.name} — {pages} pages")

    print(
        f"\n{total_pages} pages ready. chunks.json removed so chunk.py will "
        f"split them at full resolution."
    )
    print("Next: .venv/bin/pixelrag index build")


if __name__ == "__main__":
    main()
