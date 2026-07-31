#!/usr/bin/env python
"""Build / rebuild the visual index as fast as this box allows.

Speed levers (RTX 5070 Ti):
  1. pixelrag.yaml: device=cuda, backend=direct_gpu  → batched GPU embed
  2. prepare_hires                                 → full-res (slower, better)
  3. --stock                                       → 1 vector/page (~6× fewer embeds)
  4. Only keep the PDFs you need in pdfs/          → index is not append-friendly

    .venv/Scripts/python.exe scripts/build_index.py
    .venv/Scripts/python.exe scripts/build_index.py --stock     # faster first pass
    .venv/Scripts/python.exe scripts/build_index.py --skip-render
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--stock",
        action="store_true",
        help="Skip prepare_hires; stock 1-chunk/page path (faster, worse for matrices)",
    )
    ap.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip prepare_hires (tiles already on disk)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Clean output and rebuild from scratch",
    )
    ap.add_argument(
        "--adapter",
        default=None,
        metavar="DIR",
        help="PEFT LoRA adapter dir (the screenshot LoRA `pixelrag index build` "
             "never applies). Requires `pip install peft`. Queries must be "
             "encoded with the same adapter — see PIXELRAG_ADAPTER.",
    )
    args = ap.parse_args()
    os.chdir(ROOT)

    import torch

    if not torch.cuda.is_available():
        sys.exit(
            "CUDA not available in this interpreter.\n"
            f"  python={PY}\n"
            f"  torch={torch.__version__}\n"
            "Need cu128 torch in .venv and embed.device: cuda in pixelrag.yaml."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}", flush=True)

    t0 = time.perf_counter()
    if not args.stock and not args.skip_render:
        print("\n=== prepare_hires (full-res tiles) ===", flush=True)
        subprocess.run([PY, str(ROOT / "scripts" / "prepare_hires.py")], check=True)
    elif args.stock:
        print("\n=== stock path (no prepare_hires) ===", flush=True)

    # Multi-scale chunking must run AFTER prepare_hires (which deletes
    # chunks.json) and BEFORE the build (whose Stage 2 runs pixelrag_embed.chunk
    # without --force, so it skips any directory that already has a manifest —
    # which is exactly how ours survives without patching upstream).
    if not args.stock:
        print("\n=== chunk_multiscale (gist + overlapping regions) ===", flush=True)
        subprocess.run([PY, str(ROOT / "scripts" / "chunk_multiscale.py")], check=True)

    # Old shard_*.npz from embed_cpu lack keys the GPU embedder expects
    # (page_heights, image_hashes, …). Incremental merge then KeyErrors.
    emb_dir = ROOT / "index" / "embeddings"
    if emb_dir.exists() and not args.force:
        import numpy as np

        stale = False
        for f in emb_dir.glob("shard_*.npz"):
            keys = set(np.load(f).files)
            if "page_heights" not in keys or "image_hashes" not in keys:
                stale = True
                break
        if stale or list(emb_dir.glob("partial_*.npz")):
            print("\n=== clearing stale embeddings (format mismatch / partial) ===",
                  flush=True)
            import shutil
            shutil.rmtree(emb_dir)

    if args.stock:
        # Stock path: let the pipeline do everything, batch size and all. Kept as
        # the A/B baseline, not because it is fast.
        print("\n=== pixelrag index build (stock pipeline) ===", flush=True)
        cmd = [PY, "-m", "pixelrag_index.pipelines", "build", "--device", "cuda",
               "--force"]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    else:
        print("\n=== embed + faiss (batch size tuned for this GPU) ===", flush=True)
        embed_and_index(force=args.force, adapter=args.adapter)


# --------------------------------------------------------------------------
# embed + index, driven directly so the batch size is ours to set
# --------------------------------------------------------------------------

# pipelines.py hardcodes pixelrag_embed.embed's default batch_size=128. Measured
# on this 16GB RTX 5070 Ti, that is catastrophic: 128 chunks of 875x1024 need
# ~22 GiB of activations, so it spills to host memory and runs at 1.90 s/chunk
# while reporting 100% GPU utilization at 64W of a ~300W budget. Batch 16 peaks
# at 12.9 GiB, stays resident, and runs at 0.161 s/chunk — 12x faster.
#
#   batch=4   0.162 s/chunk   peak  6.2 GiB
#   batch=8   0.160 s/chunk   peak  8.4 GiB
#   batch=16  0.161 s/chunk   peak 12.9 GiB   <- headroom without spilling
#   batch=32  1.897 s/chunk   peak 21.8 GiB   <- spills, 12x slower
#
# Throughput is flat from 4 to 16, so this is not a tuning knob to chase — it is
# a cliff to stay off. On a card with more VRAM, raise it; the cliff moves.
BATCH_SIZE = int(os.environ.get("PIXELRAG_BATCH_SIZE", "16"))


def embed_and_index(force: bool, adapter: str | None = None) -> None:
    """Run stages 3 and 4 directly, with a batch size that fits this GPU."""
    import numpy as np

    tiles = ROOT / "index" / "tiles"
    emb = ROOT / "index" / "embeddings"
    emb.mkdir(parents=True, exist_ok=True)

    cmd = [PY, "-m", "pixelrag_embed.embed",
           "--shard-dir", str(tiles), "--output-dir", str(emb),
           "--gpu-ids", "0", "--backend", "direct_gpu",
           "--mode", "chunks", "--batch-size", str(BATCH_SIZE)]
    if adapter:
        # The half of PixelRAG's pitch `pixelrag index build` never applies:
        # pipelines.py passes an adapter flag in neither branch. Only meaningful
        # with direct_gpu, and the query encoder must use the same adapter or the
        # two live in mismatched spaces.
        cmd += ["--adapter", adapter]
    if not force:
        cmd.append("--resume")
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    print(f"embed: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    shards = sorted(emb.glob("shard_*.npz"))
    if not shards:
        sys.exit(f"No shard_*.npz in {emb} — embedding produced nothing.")
    total = sum(np.load(f, mmap_mode="r")["embeddings"].shape[0] for f in shards)
    nlist = min(4096, max(1, total // 40))
    print(f"\n=== faiss index ({total} vectors, nlist={nlist}) ===", flush=True)
    subprocess.run([PY, "-m", "pixelrag_embed.index", "build",
                    "--embeddings-dir", str(emb),
                    "--output-dir", str(ROOT / "index"),
                    "--nlist", str(nlist)], check=True)

    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min.", flush=True)
    print("Restart pixelrag serve so it reloads ./index.")


if __name__ == "__main__":
    main()
