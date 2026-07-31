"""Crop a rendered page around citation rectangles, with the match highlighted.

What the chat shows next to an answer: not the whole page, which is unreadable
at chat width, and not the bare quote rectangle, which is a strip of text with
no row label and no table header to give it meaning.

This is page-layout policy, and it lived inside a FastAPI route handler — 85
lines of geometry interleaved with Response construction, unreachable from the
CLI and untestable without the web framework. The decisions are:

  * pad the union of the rectangles, because a quote pressed against its own
    bounding box tells you nothing about what it is a row of;
  * enforce a minimum height, because a one-line quote in a price matrix needs
    the rows above and below it to be checkable;
  * enforce a minimum width for the same reason in the other axis, since the
    row label sits to the left of the cell that was quoted;
  * leave a large retrieval region alone — it already fills the crop;
  * fall back to a readable mid-page band when there is nothing to highlight,
    so an unverified citation still shows *something* rather than a blank.

Rectangles are percentages of the page, the same convention citations.py
returns and the PDF viewer overlays, so nothing here needs the render DPI.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

# Padding around the union of the highlighted rectangles, in percent of page.
DEFAULT_PAD = 4.0

# A crop shorter than this is a floating strip of text. 10% of an A4 page is
# about five lines at 200 DPI — the quoted row plus its neighbours.
MIN_HEIGHT = 8.0
GROWN_HEIGHT = 10.0

# Same argument horizontally: the row label is to the left of the quoted cell.
MIN_WIDTH = 20.0
GROWN_WIDTH = 24.0

# Where to look when a citation could not be verified and has no geometry.
# Not the whole page: a readable band through the middle, where body content is.
FALLBACK_BAND = (4.0, 28.0, 96.0, 72.0)         # left, top, right, bottom

# Highlighter wash and border, RGBA. The wash has to be light enough to read
# the text through and saturated enough to find at a glance.
WASH = (255, 210, 63, 100)
BORDER = (180, 83, 31, 220)
BORDER_WIDTH = 3

JPEG_QUALITY = 88


@dataclass(frozen=True)
class Box:
    """A rectangle as percentages of the page. Matches citations.py's output."""

    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @classmethod
    def from_dict(cls, d: dict) -> Box:
        return cls(left=float(d.get("left", 0)), top=float(d.get("top", 0)),
                   width=float(d.get("width", 0)), height=float(d.get("height", 0)))


def parse_boxes(raw) -> list[Box]:
    """Boxes from whatever the wire carried. Anything unusable is dropped."""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            try:
                out.append(Box.from_dict(item))
            except (TypeError, ValueError):
                continue
    return out


def crop_window(boxes: list[Box], pad: float = DEFAULT_PAD) -> Box:
    """The region of the page to show, given what should be highlighted.

    Pure geometry in percent — no image, no IO — which is what makes the policy
    above assertable.
    """
    if not boxes:
        left, top, right, bottom = FALLBACK_BAND
    else:
        left = min(b.left for b in boxes)
        top = min(b.top for b in boxes)
        right = max(b.right for b in boxes)
        bottom = max(b.bottom for b in boxes)

    left = max(0.0, left - pad)
    top = max(0.0, top - pad)
    right = min(100.0, right + pad)
    bottom = min(100.0, bottom + pad)

    # Thin quote lines need air so the row label stays in frame. Large
    # retrieval regions already fill the crop — leave them alone.
    if boxes and (bottom - top) < MIN_HEIGHT:
        top, bottom = _grow(top, bottom, GROWN_HEIGHT)
    if boxes and (right - left) < MIN_WIDTH:
        left, right = _grow(left, right, GROWN_WIDTH)

    return Box(left=left, top=top, width=right - left, height=bottom - top)


def _grow(lo: float, hi: float, target: float) -> tuple[float, float]:
    """Widen a span to `target` about its midpoint, clamped to the page."""
    mid = (lo + hi) / 2
    half = target / 2
    return max(0.0, mid - half), min(100.0, mid + half)


def render(page_image, boxes: list[Box], pad: float = DEFAULT_PAD,
           quality: int = JPEG_QUALITY) -> bytes:
    """JPEG bytes of the crop, with `boxes` highlighted.

    `page_image` is a path or an open PIL image of the RENDERED page.
    """
    from PIL import Image, ImageDraw

    opened = not hasattr(page_image, "convert")
    im = Image.open(page_image) if opened else page_image
    try:
        im = im.convert("RGB")
        pw, ph = im.size
        window = crop_window(boxes, pad)

        x0, y0 = int(pw * window.left / 100), int(ph * window.top / 100)
        # Round the far edge up, so a sub-pixel window is still one pixel wide.
        x1 = int(pw * window.right / 100 + 0.999)
        y1 = int(ph * window.bottom / 100 + 0.999)
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        crop = im.crop((x0, y0, x1, y1)).convert("RGBA")

        if boxes:
            overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for b in boxes:
                rx0 = int(pw * b.left / 100) - x0
                ry0 = int(ph * b.top / 100) - y0
                rx1 = int(pw * b.right / 100) - x0
                ry1 = int(ph * b.bottom / 100) - y0
                draw.rectangle([rx0, ry0, rx1, ry1], fill=WASH)
                draw.rectangle([rx0, ry0, rx1, ry1], outline=BORDER,
                               width=BORDER_WIDTH)
            crop = Image.alpha_composite(crop, overlay)

        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=quality,
                                 optimize=True)
        return buf.getvalue()
    finally:
        if opened:
            im.close()
