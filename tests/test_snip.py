"""Crop policy for the inline citation image.

All of this used to live inside a FastAPI route handler, so none of it could be
asserted without standing up the web app.
"""

from __future__ import annotations

import io

import pytest

import snip

PAGE = (600, 850)


def box(left, top, width, height):
    return snip.Box(left=left, top=top, width=width, height=height)


# -- the window -------------------------------------------------------------

def test_the_window_pads_the_union_of_the_boxes():
    """A quote pressed against its own bounding box says nothing about what it
    is a row of."""
    w = snip.crop_window([box(30, 40, 40, 20)], pad=4)
    assert w.left == 26 and w.top == 36
    assert w.right == 74 and w.bottom == 64


def test_the_window_spans_several_boxes():
    """A quote wrapping several lines has one rectangle per line, and the crop
    has to cover all of them. Sized past the minimums so this tests the union
    and nothing else."""
    w = snip.crop_window([box(20, 40, 25, 2), box(20, 50, 30, 2)], pad=0)
    assert (w.left, w.top, w.right, w.bottom) == (20, 40, 50, 52)


def test_a_thin_quote_is_grown_so_the_row_label_stays_in_frame():
    """One line of a price matrix needs the rows around it to be checkable."""
    w = snip.crop_window([box(40, 50, 30, 0.4)], pad=0)
    assert w.height == pytest.approx(snip.GROWN_HEIGHT)
    assert w.top + w.height / 2 == pytest.approx(50.2)   # still centred on it


def test_a_narrow_quote_is_grown_the_same_way():
    """The row label sits to the LEFT of the cell that was quoted."""
    w = snip.crop_window([box(60, 50, 3, 20)], pad=0)
    assert w.width == pytest.approx(snip.GROWN_WIDTH)


def test_a_large_region_is_left_alone():
    """A retrieval region already fills the crop; growing it would zoom out of
    the thing being shown."""
    w = snip.crop_window([box(5, 10, 90, 60)], pad=0)
    assert (w.width, w.height) == (90, 60)


def test_the_window_never_leaves_the_page():
    w = snip.crop_window([box(0, 0, 100, 100)], pad=20)
    assert w.left == 0 and w.top == 0
    assert w.right == 100 and w.bottom == 100


def test_growing_at_the_edge_stays_on_the_page():
    w = snip.crop_window([box(0, 0, 2, 0.5)], pad=0)
    assert w.left >= 0 and w.top >= 0


def test_no_boxes_gives_a_readable_band_not_the_whole_page():
    """An unverified citation still shows something rather than a blank."""
    w = snip.crop_window([], pad=0)
    assert (w.left, w.top, w.right, w.bottom) == snip.FALLBACK_BAND
    assert w.height < 100                        # a band, not the page


def test_the_fallback_band_is_not_grown():
    """The minimums only apply to real highlights; the band is already sized."""
    w = snip.crop_window([], pad=0)
    assert w.height == snip.FALLBACK_BAND[3] - snip.FALLBACK_BAND[1]


# -- parsing ----------------------------------------------------------------

def test_parse_boxes_reads_the_citations_wire_format():
    got = snip.parse_boxes([{"left": 1, "top": 2, "width": 3, "height": 4}])
    assert got == [box(1, 2, 3, 4)]


def test_parse_boxes_defaults_missing_fields_to_zero():
    assert snip.parse_boxes([{"left": 5}]) == [box(5, 0, 0, 0)]


def test_parse_boxes_drops_anything_unusable():
    assert snip.parse_boxes(["nonsense", 42, None, {"left": 1}]) == [box(1, 0, 0, 0)]


def test_parse_boxes_rejects_a_non_list():
    assert snip.parse_boxes({"left": 1}) == []


# -- rendering --------------------------------------------------------------

@pytest.fixture
def page(tmp_path):
    from PIL import Image

    p = tmp_path / "tile_0000.jpg"
    Image.new("RGB", PAGE, (255, 255, 255)).save(p)
    return p


def size_of(data: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.size


def test_render_crops_to_the_window(page):
    data = snip.render(page, [box(25, 40, 50, 20)], pad=0)
    w, h = size_of(data)
    assert w == pytest.approx(PAGE[0] * 0.50, abs=2)
    assert h == pytest.approx(PAGE[1] * 0.20, abs=2)


def test_render_produces_a_jpeg(page):
    assert snip.render(page, [box(25, 40, 50, 20)])[:2] == b"\xff\xd8"


def test_render_marks_the_highlight(page):
    """A white page comes back with the wash on it, or the crop is pointless."""
    from PIL import Image

    data = snip.render(page, [box(25, 40, 50, 20)], pad=2)
    with Image.open(io.BytesIO(data)) as im:
        centre = im.convert("RGB").getpixel((im.width // 2, im.height // 2))
    assert centre != (255, 255, 255)


def test_render_without_boxes_draws_no_highlight(page):
    from PIL import Image

    data = snip.render(page, [])
    with Image.open(io.BytesIO(data)) as im:
        centre = im.convert("RGB").getpixel((im.width // 2, im.height // 2))
    assert centre == (255, 255, 255)


def test_render_never_produces_a_zero_size_crop(page):
    """A sub-pixel window must still be one pixel, not a crash."""
    w, h = size_of(snip.render(page, [box(50, 50, 0.001, 0.001)], pad=0))
    assert w >= 1 and h >= 1


def test_render_accepts_an_open_image_and_does_not_close_it(page):
    from PIL import Image

    with Image.open(page) as im:
        snip.render(im, [box(25, 40, 50, 20)])
        assert im.size == PAGE          # still usable
