#!/usr/bin/env python
"""Measure how much small-text detail survives each PixelRAG PDF path.

Two paths exist:

  default   render_pdf writes chunks.json itself, so chunk.py skips the page.
            The whole page becomes one chunk, and embed_cpu._clamp_width
            shrinks it to <=875px wide before the model sees it.

  chunked   chunks.json is removed so chunk.py processes the page, splitting
            it into a grid of <=1024px-tall x <=875px-wide pieces. Every piece
            is already within the clamp, so no downscaling happens at all.

Renders a real crop of a dimension callout under both, at the exact pixel
scale the embedding model receives, so the difference is visible rather than
theoretical.

Usage:  .venv/bin/python scripts/resolution_probe.py pdfs/_TEST_SG2400_manual.pdf
"""

import shutil
import sys
from pathlib import Path

from PIL import Image
from pixelrag_embed.embed_cpu import _clamp_width, _MAX_CHUNK_WIDTH
from pixelrag_render.backends.pdf import render_pdf

OUT = Path("index/_probe")


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else "pdfs/_TEST_SG2400_manual.pdf"

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Stage 1 exactly as the pipeline calls it — note dpi is NOT configurable
    # from pixelrag.yaml; the pipeline calls render_pdf(path, dir, stem=idx)
    # with no ingest kwargs, so 200 is effectively hardcoded.
    tile_dirs = render_pdf(pdf, str(OUT), stem="0")
    tile_dir = tile_dirs[0]

    page = Image.open(tile_dir / "tile_0000.jpg")
    w, h = page.size
    print(f"\nrendered page      : {w} x {h} px  (dpi=200, hardcoded)")

    # --- default path: whole page, then clamped -------------------------
    clamped = _clamp_width(page)
    cw, chh = clamped.size
    scale = cw / w
    print(f"clamp limit        : {_MAX_CHUNK_WIDTH} px wide")
    print(f"default path sees  : {cw} x {chh} px  (scale {scale:.3f})")
    print(f"effective DPI      : {200 * scale:.0f}")

    # --- chunked path: grid pieces are already under the clamp ----------
    print(f"chunked path sees  : <=875 x 1024 px pieces at scale 1.000")
    print(f"effective DPI      : 200")
    grid_cols = -(-w // 875)
    grid_rows = -(-h // 1024)
    print(f"chunks per page    : {grid_cols} cols x {grid_rows} rows = "
          f"{grid_cols * grid_rows} vectors (default: 1)")

    # --- visual proof: the "2400 mm" dimension callout -------------------
    # Callout sits around y=402pt of 842pt, x=260pt of 595pt in PDF space.
    px = int(w * (240 / 595))
    py = int(h * (392 / 842))
    box = (px, py, px + 420, py + 60)

    native = page.crop(box)
    native.save(OUT / "callout_chunked_200dpi.png")

    # Same region as the default path renders it, upscaled back to native
    # size so the two images are directly comparable on screen.
    sbox = tuple(int(v * scale) for v in box)
    degraded = clamped.crop(sbox).resize(native.size, Image.NEAREST)
    degraded.save(OUT / "callout_default_106dpi.png")

    combo = Image.new("RGB", (native.width, native.height * 2 + 8), "white")
    combo.paste(degraded, (0, 0))
    combo.paste(native, (0, native.height + 8))
    combo.save(OUT / "callout_comparison.png")

    print(f"\nwrote {OUT}/callout_comparison.png")
    print("  top = default path (what the model sees today)")
    print("  bottom = chunked path (full resolution)")


if __name__ == "__main__":
    main()
