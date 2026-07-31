#!/usr/bin/env python
"""Build the lexical sidecar (index/text.json). LOAD-BEARING — not just for eval.

This started as the eval's reference point ("what would plain BM25 have got?").
It is now the second retriever in the answer path: rag.py fuses BM25 over this
sidecar with the pixel index, worth +6 top-1 and +2 recall@4 on the real question
set because the two retrievers miss disjoint sets of questions.

Consequence for operations: if this file is stale or missing, retrieval silently
degrades to visual-only (85% recall@4 instead of 95%) rather than failing loudly.
Re-run it after ANY corpus change or build_index.py. Cheap derived artefact over
an existing ./index — seconds, no GPU, no model.

Also reports which pages have no text layer at all — those are both where lexical
search is blind and where citation highlighting has to fall back to PixelRAG's
region box instead of exact word rectangles.

    .venv/Scripts/python.exe scripts/build_text_index.py
"""

from __future__ import annotations

import json
import sys

import lexical
import rag


def main() -> None:
    if not rag.INDEX_DIR.exists():
        sys.exit("No ./index — build the visual index first (see README).")

    arts = rag.articles()
    data = lexical.build_sidecar(arts)
    pgs = data["pages"]
    vis_only = [p for p in pgs if p["visual_only"]]
    revs = sorted({p["revision"] for p in pgs if p["revision"]})

    print(f"text sidecar : {len(pgs)} pages from {len(arts)} documents "
          f"-> {lexical.TEXT_SIDECAR}")
    print(f"  visual-only (no text layer, lexical search cannot reach): "
          f"{len(vis_only)}")
    for p in vis_only:
        print(f"      {p['title']} p{p['page']}")
    print(f"  price-list revisions present: {', '.join(revs) or 'none detected'}")
    if len(revs) > 1:
        print("      NOTE: multiple editions in one index — answers must cite "
              "which, and prefer the newest.")

    # Blank chunks are no longer indexed at all — chunk_multiscale.py skips them
    # before they are ever embedded — so the old search-time blank mask is dead.
    # Reported here only as a check that the chunker did its job.
    total = blanks = 0
    for d in sorted(rag.TILES_DIR.glob("*.png.tiles")):
        m = d / "chunks.json"
        if not m.exists():
            continue
        j = json.loads(m.read_text(encoding="utf-8"))
        total += j.get("num_chunks", 0)
        blanks += j.get("blank_skipped", 0)
    print(f"chunks       : {total} indexed, {blanks} blank regions skipped "
          f"before embedding")


if __name__ == "__main__":
    main()
