#!/usr/bin/env python
"""Shrink page screenshots to the resolution the reader actually bills for.

The index renders pages at 200 DPI (1654x2339 for A4). No provider charges for
those pixels as-is.

GEMINI -- THE TILE THEORY WAS MEASURED AND DID NOT HOLD. The documented model is
768x768 tiles at 258 tokens each, which for a 1654x2339 page predicts
3 x 4 = 12 tiles = 3096 tokens, halving to 6 tiles if you scale 7% to 1536x2172
and shed a mostly-empty tile row. Against gemini-3.6-flash that prediction is
simply wrong: the same 4-page question billed 5725 input tokens with the resize
and 5725 without it, identical. That also puts the real per-image cost near 1080
tokens, not 3096 — this model normalises images to its own working resolution
before billing, so there is no tile boundary left to chase.

Tile chasing is therefore OFF by default (GEMINI_TILE_CHASING). Re-measure
before enabling it: this is one model on one page geometry, and the arithmetic
above is what the docs describe, so another Gemini model may well bill that way.
What the pass still buys on this path is upload bytes, which is latency rather
than spend.

ANTHROPIC is unverified here — this box has no Anthropic credentials, so the
figures below come from the documented billing rule, not from a measurement.
Tokens ~= w*h/750, and which ceiling applies depends on the model tier:
  * high-res tier (Opus 4.7+, Opus 5, Sonnet 5): long edge <= 2576px, capped at
    ~4784 tokens/image. A 200-DPI A4 page is UNDER that ceiling, so it is sent
    near full resolution and costs ~4784 tokens.
  * older tier: long edge clamped to 1568px, so the same page arrives as
    1109x1568 and costs ~2318 tokens.
Opting into the 1568 ceiling ourselves should therefore roughly halve the image
bill on a current Opus. Confirm that against a real `usage.input_tokens` before
relying on it — the Gemini result above is precisely what assuming a documented
billing model gets you. It is also a real accuracy tradeoff on small print, not
a free win: see resolution_probe.py, and sweep it before trusting it on a price
matrix.

Both paths cut upload bytes, which is latency nobody measures: four
full-resolution pages are ~1.5MB raw and ~2MB once base64'd into the request.

Resized bytes are cached on disk. The resize itself is only ~30ms/page, but
paying it per query for pages that never change is pure waste.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path

# Gemini's tile size. Images are diced into this grid and every tile costs the
# same 258 tokens whether it is full of table or full of margin.
GEMINI_TILE = 768
GEMINI_TOKENS_PER_TILE = 258

# Anthropic's long-edge ceiling for the tier we choose to target, and the
# divisor in its area->tokens rule. 1568 is the pre-high-res ceiling; raise it
# to 2576 to send high-res and pay for it.
ANTHROPIC_LONG_EDGE = int(os.environ.get("PIXELRAG_ANTHROPIC_LONG_EDGE", "1568"))
ANTHROPIC_TOKEN_DIVISOR = 750

# How much linear shrink is allowed while chasing a cheaper tile grid. At 0.85
# a 200-DPI render still lands near 170 DPI, which resolution_probe.py puts
# comfortably above the point where dimension callouts stop being legible.
# Lower it to buy more savings, but re-run the probe when you do.
SCALE_FLOOR = float(os.environ.get("PIXELRAG_IMAGE_FLOOR", "0.85"))

QUALITY = int(os.environ.get("PIXELRAG_IMAGE_QUALITY", "85"))

# Shrink Gemini pages to land on a cheaper 768px tile grid. OFF because it was
# measured to save exactly zero tokens on gemini-3.6-flash (see module docstring)
# while still costing resolution. Turn on only with a measurement to back it.
GEMINI_TILE_CHASING = os.environ.get("PIXELRAG_GEMINI_TILE_CHASING", "0") != "0"

# Set to 0 to send pages exactly as rendered, e.g. to A/B the accuracy cost.
ENABLED = os.environ.get("PIXELRAG_IMAGE_FIT", "1") != "0"

CACHE_DIR = Path(os.environ.get("PIXELRAG_FIT_CACHE", "index/tiles/_fit"))


# --------------------------------------------------------------------------
# token accounting -- the thing being optimised, so it is worth stating
# --------------------------------------------------------------------------

def gemini_tiles(w: int, h: int) -> int:
    return math.ceil(w / GEMINI_TILE) * math.ceil(h / GEMINI_TILE)


def gemini_tokens(w: int, h: int) -> int:
    return gemini_tiles(w, h) * GEMINI_TOKENS_PER_TILE


def anthropic_tokens(w: int, h: int, long_edge: int = ANTHROPIC_LONG_EDGE) -> int:
    """Tokens after the provider's own clamp to `long_edge`.

    Sending something larger than the clamp does not cost more -- it costs the
    same and uploads more bytes, which is the worst of both.
    """
    s = min(1.0, long_edge / max(w, h))
    return round(w * s * h * s / ANTHROPIC_TOKEN_DIVISOR)


# --------------------------------------------------------------------------
# choosing a target size
# --------------------------------------------------------------------------

def _gemini_target(w: int, h: int, floor: float = SCALE_FLOOR) -> tuple[int, int]:
    """Largest size that lands on the cheapest reachable tile grid.

    Enumerates candidate grids rather than sweeping scales: for a target of
    `cols` x `rows` tiles the biggest image that fits is bounded by both
    dimensions, so the scale is determined exactly. Picks the fewest tiles, and
    among grids that tie, the least shrink -- there is no reason to throw away
    resolution that is already paid for.
    """
    best = (w, h)
    best_tiles = gemini_tiles(w, h)
    best_scale = 1.0
    for cols in range(1, math.ceil(w / GEMINI_TILE) + 1):
        for rows in range(1, math.ceil(h / GEMINI_TILE) + 1):
            s = min(cols * GEMINI_TILE / w, rows * GEMINI_TILE / h, 1.0)
            if s < floor:
                continue
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            tiles = gemini_tiles(nw, nh)
            if tiles < best_tiles or (tiles == best_tiles and s > best_scale):
                best, best_tiles, best_scale = (nw, nh), tiles, s
    return best


def _anthropic_target(w: int, h: int,
                      long_edge: int = ANTHROPIC_LONG_EDGE) -> tuple[int, int]:
    """Clamp to the ceiling we intend to be billed at.

    No cliff here -- tokens track area continuously -- so this only avoids
    uploading pixels the provider is about to discard.
    """
    s = min(1.0, long_edge / max(w, h))
    return max(1, int(w * s)), max(1, int(h * s))


def target_size(w: int, h: int, provider: str) -> tuple[int, int]:
    if provider == "gemini":
        # Measured to buy no tokens on gemini-3.6-flash; the re-encode alone
        # still trims upload bytes, so the default is "same pixels, tighter
        # JPEG" rather than a resize.
        return _gemini_target(w, h) if GEMINI_TILE_CHASING else (w, h)
    if provider == "anthropic":
        return _anthropic_target(w, h)
    return w, h


# --------------------------------------------------------------------------
# the resize itself
# --------------------------------------------------------------------------

def _cache_key(path: Path, provider: str) -> str:
    st = path.stat()
    # Every knob that changes the OUTPUT belongs in the key, or flipping one
    # silently serves bytes produced under the old setting.
    raw = (f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{provider}|"
           f"{ANTHROPIC_LONG_EDGE}|{SCALE_FLOOR}|{QUALITY}|"
           f"{int(GEMINI_TILE_CHASING)}")
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def fit(path: Path, provider: str) -> tuple[bytes, str]:
    """Page bytes sized for `provider`, plus their mime type.

    Falls back to the original bytes on any failure. A page that reaches the
    reader slightly too large is a cost bug; a page that does not reach it at
    all is a broken answer.
    """
    path = Path(path)
    if not ENABLED:
        return path.read_bytes(), "image/jpeg"

    cached = CACHE_DIR / provider / f"{_cache_key(path, provider)}.jpg"
    if cached.exists():
        try:
            return cached.read_bytes(), "image/jpeg"
        except OSError:
            pass

    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            tw, th = target_size(im.width, im.height, provider)
            resized = (tw, th) != im.size
            if resized:
                im = im.resize((tw, th), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=QUALITY, optimize=True)
            data = buf.getvalue()
    except Exception:
        return path.read_bytes(), "image/jpeg"

    # When nothing was resized, re-encoding can come out larger than a source
    # JPEG that was already tight — and it can only lose quality. Keep whichever
    # is smaller; they are billed identically either way.
    if not resized:
        original = path.read_bytes()
        if len(original) <= len(data):
            data = original

    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    except OSError:
        pass
    return data, "image/jpeg"


def _size_of(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# --------------------------------------------------------------------------
# report -- run this before trusting the numbers in any write-up
# --------------------------------------------------------------------------

def _report() -> None:
    import glob

    pages = sorted(glob.glob("index/tiles/*.tiles/tile_*.jpg"))
    if not pages:
        print("No rendered pages under index/tiles/. Build the index first.")
        return

    rows = []
    for p in pages:
        w, h = _size_of(Path(p))
        gw, gh = _gemini_target(w, h)
        aw, ah = _anthropic_target(w, h)
        rows.append((
            gemini_tokens(w, h), gemini_tokens(gw, gh),
            # before: what a high-res-tier model bills at full resolution
            anthropic_tokens(w, h, 2576), anthropic_tokens(aw, ah),
            os.path.getsize(p),
        ))

    n = len(rows)
    gb = sum(r[0] for r in rows) / n
    ga = sum(r[1] for r in rows) / n
    ab = sum(r[2] for r in rows) / n
    aa = sum(r[3] for r in rows) / n
    w0, h0 = _size_of(Path(pages[0]))

    print(f"{n} pages, {w0}x{h0} source\n")
    print(f"gemini    {gb:8.0f} -> {ga:8.0f} tokens/page  ({100*(1-ga/gb):.0f}% off)")
    print(f"anthropic {ab:8.0f} -> {aa:8.0f} tokens/page  ({100*(1-aa/ab):.0f}% off)")
    print(f"          (before = high-res tier at 2576px; after = {ANTHROPIC_LONG_EDGE}px)")
    print()
    k = int(os.environ.get("PIXELRAG_ONESHOT_PAGES", "4"))
    sysx = 1400
    for name, before, after, rin, rout in (
        ("opus-5", ab, aa, 5.0, 25.0),
        ("sonnet-5", ab, aa, 3.0, 15.0),
    ):
        cb = (before * k + sysx) / 1e6 * rin + 800 / 1e6 * rout
        ca = (after * k + sysx) / 1e6 * rin + 800 / 1e6 * rout
        print(f"{name:9s} {k} pages: ${cb:.4f} -> ${ca:.4f} per query")


if __name__ == "__main__":
    _report()
