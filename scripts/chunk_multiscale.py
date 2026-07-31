#!/usr/bin/env python
"""Multi-scale, overlapping page chunker — replaces PixelRAG's stock grid.

Runs after prepare_hires.py and before `pixelrag index build`. Writes the same
chunks.json contract pixelrag_embed reads (file / tile_index / chunk_index /
x_offset / y_offset / width / height), so nothing upstream is patched or forked.

Three changes from the stock 2x3 grid, each measured against the real question
set on this corpus:

1. A WHOLE-PAGE GIST VECTOR per page (chunk_index 0).
   Stock PixelRAG embeds one vector per page; prepare_hires.py replaces that
   with six region vectors and no page vector. Both are needed. "Pokaż rodzaje
   paneli w bramach segmentowych" is a question about what a page *is* — a
   sheet of panel drawings and a structure table — and no single 875x1024
   quadrant carries that. Region vectors answer "where on the page", the gist
   vector answers "which page".

2. VERTICAL OVERLAP between row strips (stride < height).
   The stock grid cuts at a fixed 1024px with no overlap, so a table row landing
   on a boundary is half-height in two chunks and legible in neither. On page 61
   — a 30-row options table where each row is one answer — that is the difference
   between reading "+114 za kpl." and reading a sliced glyph. Overlap costs
   vectors, not accuracy.

3. NEAR-BLANK REGIONS ARE NOT EMBEDDED.
   Measured on the previous build: 82 of 354 vectors (23%) were page margin and
   footer whitespace. Because every blank chunk embeds to nearly the same vector,
   they surfaced as a tied block at the top of any query the encoder had no real
   signal for — on "Dzielony wał" the entire top-5 was blank slivers scoring an
   identical 0.386. Dropping them costs no recall (there is nothing on them) and
   saves the embed time too.

Chunks stay <= CHUNK_WIDTH (875) wide because both embed paths resize anything
wider (embed.py:_clamp_width_pil, embed_cpu.py:_clamp_width) — 875 is the width
the model was trained on and the width at which no pixels are thrown away.

x_offset/y_offset/width/height are recorded in ORIGINAL page pixels, including
for the downscaled gist chunk. They are what the UI turns into a highlight
rectangle over the full page, so they must describe the page, not the crop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# The model's native width. Wider crops get resized by the embedder anyway.
CHUNK_WIDTH = 875
# Row-strip height. Matches stock CHUNK_HEIGHT so crops stay in distribution.
CHUNK_HEIGHT = 1024
# Vertical stride. 640 => 37% overlap: any 384px-tall band of the page appears
# whole in at least one strip, which covers a table row plus its header line.
CHUNK_STRIDE = 640
# Below this, a Qwen3-VL patch grid has nothing to bite on (one patch = 28px).
MIN_SIDE = 56
# Fraction of sub-200 grey pixels below which a crop is treated as empty.
BLANK_INK = 0.005
# Whole-page gist vector per page. Default off: measured neutral on this corpus
# (retrieval is bit-identical with and without — see retrieve.py GIST_BONUS) and
# it costs ~11% more vectors and build time. The index currently in ./index was
# built WITH them; rebuilding without changes no measured number.
EMIT_GIST = False


def _ink(img: Image.Image) -> float:
    return float((np.asarray(img.convert("L")) < 200).mean())


def _row_offsets(h: int) -> list[int]:
    """Overlapping strip origins covering [0, h).

    The last strip is pinned to the bottom edge rather than left short, so the
    page footer is never clipped to a sliver — the stock grid's 291px tail on
    A4 was both useless and 23% of the index.
    """
    if h <= CHUNK_HEIGHT:
        return [0]
    ys = list(range(0, h - CHUNK_HEIGHT + 1, CHUNK_STRIDE))
    if ys[-1] + CHUNK_HEIGHT < h:
        ys.append(h - CHUNK_HEIGHT)
    return ys


def chunk_page(img: Image.Image, tile_idx: int, out_dir: Path,
               next_index: int) -> tuple[list[dict], int, int]:
    """Gist chunk + overlapping region chunks for one page. Returns (infos, next, skipped)."""
    w, h = img.size
    infos: list[dict] = []
    ci = next_index
    skipped = 0

    # --- 1. gist: the whole page at the model's native width -------------
    # Off by default: measured neutral on this corpus (see retrieve.py GIST_BONUS).
    if EMIT_GIST:
        gist = img
        if w > CHUNK_WIDTH:
            gist = img.resize((CHUNK_WIDTH, max(1, round(h * CHUNK_WIDTH / w))),
                              Image.LANCZOS)
        name = f"chunk_{tile_idx:04d}_{ci:02d}.png"
        gist.save(out_dir / name, format="PNG")
        infos.append({
            "tile": f"tile_{tile_idx:04d}.jpg", "tile_index": tile_idx,
            "chunk_index": ci, "file": name,
            # Full page, in page pixels — a citation here highlights the page.
            "x_offset": 0, "y_offset": 0, "width": w, "height": h,
            "scale": "page",
        })
        ci += 1

    # --- 2. regions: 875-wide columns x overlapping 1024-tall strips ------
    for y in _row_offsets(h):
        ch = min(CHUNK_HEIGHT, h - y)
        if ch < MIN_SIDE:
            continue
        x = 0
        while x < w:
            cw = min(CHUNK_WIDTH, w - x)
            if cw < MIN_SIDE:
                break
            crop = img.crop((x, y, x + cw, y + ch))
            if _ink(crop) < BLANK_INK:
                skipped += 1
                x += cw
                continue
            name = f"chunk_{tile_idx:04d}_{ci:02d}.png"
            crop.save(out_dir / name, format="PNG")
            infos.append({
                "tile": f"tile_{tile_idx:04d}.jpg", "tile_index": tile_idx,
                "chunk_index": ci, "file": name,
                "x_offset": x, "y_offset": y, "width": cw, "height": ch,
                "scale": "region",
            })
            ci += 1
            x += cw
    return infos, ci, skipped


def process_article(tile_dir: Path, article_id: int, source: str,
                    dry_run: bool = False) -> dict | None:
    tiles = sorted(tile_dir.glob("tile_*.jpg"))
    if not tiles:
        return None

    # Stale chunk PNGs from a previous run would otherwise be embedded
    # alongside the new ones — the manifest is rewritten, the files are not.
    if not dry_run:
        for old in tile_dir.glob("chunk_*.png"):
            old.unlink()

    all_infos: list[dict] = []
    hashes: dict[str, str] = {}
    skipped = 0
    page_height = 0
    for tile_idx, t in enumerate(tiles):
        with Image.open(t) as im:
            im.load()
            page_height = max(page_height, im.size[1])
            if dry_run:
                w, h = im.size
                n = 1 + len(_row_offsets(h)) * max(1, -(-w // CHUNK_WIDTH))
                all_infos.extend([{}] * n)
                continue
            infos, _, sk = chunk_page(im, tile_idx, tile_dir, 0)
        all_infos.extend(infos)
        skipped += sk
        hashes[t.name] = hashlib.md5(t.read_bytes()).hexdigest()

    if dry_run:
        return {"num_chunks": len(all_infos), "num_tiles": len(tiles), "skipped": 0}

    manifest = {
        "page_height": page_height,
        "viewport_width": CHUNK_WIDTH,
        "tile_height": CHUNK_HEIGHT,
        "chunk_height": CHUNK_HEIGHT,
        "chunk_stride": CHUNK_STRIDE,
        "multiscale": True,
        "num_tiles": len(tiles),
        "num_chunks": len(all_infos),
        "blank_skipped": skipped,
        "tile_hashes": hashes,
        "chunks": all_infos,
        "article_id": article_id,
        "source": source,
    }
    (tile_dir / "chunks.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"num_chunks": len(all_infos), "num_tiles": len(tiles), "skipped": skipped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="count the chunks a real run would write, and stop")
    ap.add_argument("--gist", action="store_true",
                    help="also emit a whole-page vector per page. Measured "
                         "neutral on this corpus (see GIST note in retrieve.py) "
                         "and costs ~11%% more vectors and build time, so off "
                         "by default.")
    args = ap.parse_args()
    global EMIT_GIST
    EMIT_GIST = args.gist

    cfg = yaml.safe_load(Path("pixelrag.yaml").read_text())
    src = Path(cfg["source"]["path"]).expanduser()
    tiles_root = Path(cfg.get("output", "./index")) / "tiles"

    # Must mirror PDFSource.glob exactly — the tile directory name is the
    # document's position in this list.
    pdfs = sorted(src.glob("**/*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {src}/")

    total_chunks = total_pages = total_skipped = 0
    for idx, pdf in enumerate(pdfs):
        tile_dir = tiles_root / f"{idx}.png.tiles"
        if not tile_dir.exists():
            print(f"  [{idx}] {pdf.name} — no tiles, run prepare_hires.py first")
            continue
        r = process_article(tile_dir, idx, str(pdf), dry_run=args.dry_run)
        if not r:
            continue
        total_chunks += r["num_chunks"]
        total_pages += r["num_tiles"]
        total_skipped += r["skipped"]
        print(f"  [{idx}] {pdf.name} — {r['num_tiles']} pages -> "
              f"{r['num_chunks']} chunks (skipped {r['skipped']} blank)")

    per = total_chunks / total_pages if total_pages else 0
    print(f"\n{total_pages} pages -> {total_chunks} chunks ({per:.1f}/page), "
          f"{total_skipped} blank regions skipped")
    if args.dry_run:
        print("(dry run — nothing written)")
    else:
        print("Next: pixelrag index build (embed + faiss)")


if __name__ == "__main__":
    main()
