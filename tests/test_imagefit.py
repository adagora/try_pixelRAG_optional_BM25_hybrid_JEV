"""Token arithmetic and target sizing. The README quotes these numbers, so a
change here is a change to a published cost claim."""

from __future__ import annotations

import io

import pytest

import imagefit

A4_200DPI = (1654, 2339)


def test_gemini_tiles_counts_a_partial_tile_as_a_whole_one():
    assert imagefit.gemini_tiles(769, 769) == 4
    assert imagefit.gemini_tiles(768, 768) == 1


def test_gemini_tokens_for_a_200dpi_a4_page():
    assert imagefit.gemini_tiles(*A4_200DPI) == 12
    assert imagefit.gemini_tokens(*A4_200DPI) == 12 * imagefit.GEMINI_TOKENS_PER_TILE


def test_anthropic_tokens_clamp_before_charging():
    """Sending more than the ceiling costs the same and uploads more — the whole
    reason the clamp is applied on our side."""
    # Within rounding: _anthropic_target truncates both dimensions, so
    # pre-clamping is a couple of tokens cheaper, never more expensive.
    clamped = imagefit._anthropic_target(*A4_200DPI, long_edge=1568)
    assert imagefit.anthropic_tokens(*A4_200DPI, long_edge=1568) == \
        pytest.approx(imagefit.anthropic_tokens(*clamped, long_edge=1568), rel=0.01)
    # And that is roughly half the high-res tier bill, as the README claims.
    assert imagefit.anthropic_tokens(*A4_200DPI, long_edge=1568) == 2318
    assert imagefit.anthropic_tokens(*A4_200DPI, long_edge=2576) == 5158


def test_anthropic_tokens_track_area_below_the_ceiling():
    assert imagefit.anthropic_tokens(750, 750, long_edge=2576) == 750


def test_anthropic_target_preserves_aspect_ratio():
    w, h = imagefit._anthropic_target(*A4_200DPI, long_edge=1568)
    assert h == 1568
    assert w / h == pytest.approx(A4_200DPI[0] / A4_200DPI[1], rel=1e-3)


def test_anthropic_target_never_upscales():
    assert imagefit._anthropic_target(800, 600, long_edge=1568) == (800, 600)


def test_gemini_target_finds_a_cheaper_grid_within_the_floor():
    w, h = imagefit._gemini_target(*A4_200DPI, floor=0.85)
    assert imagefit.gemini_tiles(w, h) < imagefit.gemini_tiles(*A4_200DPI)
    assert w / A4_200DPI[0] >= 0.85


def test_gemini_target_respects_the_scale_floor():
    """The floor is what keeps dimension callouts legible; it must bind."""
    assert imagefit._gemini_target(*A4_200DPI, floor=1.0) == A4_200DPI


def test_gemini_target_prefers_least_shrink_among_equal_grids():
    """Resolution already paid for should not be thrown away for nothing."""
    w, h = imagefit._gemini_target(1000, 1000, floor=0.1)
    assert (w, h) == (768, 768)                  # 1 tile, largest that fits


def test_target_size_is_identity_for_an_unknown_provider():
    assert imagefit.target_size(*A4_200DPI, "someone-else") == A4_200DPI


def test_target_size_gemini_is_a_no_op_unless_tile_chasing_is_on(monkeypatch):
    """Measured to save zero tokens on gemini-3.6-flash, so it is off."""
    monkeypatch.setattr(imagefit, "GEMINI_TILE_CHASING", False)
    assert imagefit.target_size(*A4_200DPI, "gemini") == A4_200DPI
    monkeypatch.setattr(imagefit, "GEMINI_TILE_CHASING", True)
    assert imagefit.target_size(*A4_200DPI, "gemini") != A4_200DPI


# -- the resize itself ------------------------------------------------------

@pytest.fixture
def page(tmp_path):
    from PIL import Image

    p = tmp_path / "tile_0000.jpg"
    Image.new("RGB", A4_200DPI, (250, 250, 250)).save(p, quality=95)
    return p


def _size(data):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.size


def test_fit_downscales_for_anthropic(page, tmp_path, monkeypatch):
    monkeypatch.setattr(imagefit, "CACHE_DIR", tmp_path / "fit")
    data, mime = imagefit.fit(page, "anthropic")
    assert mime == "image/jpeg"
    assert max(_size(data)) == imagefit.ANTHROPIC_LONG_EDGE


def test_fit_keeps_original_pixels_for_gemini(page, tmp_path, monkeypatch):
    monkeypatch.setattr(imagefit, "CACHE_DIR", tmp_path / "fit")
    data, _ = imagefit.fit(page, "gemini")
    assert _size(data) == A4_200DPI


def test_fit_is_cached_on_disk(page, tmp_path, monkeypatch):
    monkeypatch.setattr(imagefit, "CACHE_DIR", tmp_path / "fit")
    first, _ = imagefit.fit(page, "anthropic")
    cached = list((tmp_path / "fit" / "anthropic").glob("*.jpg"))
    assert len(cached) == 1
    assert imagefit.fit(page, "anthropic")[0] == first


def test_cache_key_changes_with_every_output_affecting_knob(page, monkeypatch):
    base = imagefit._cache_key(page, "anthropic")
    monkeypatch.setattr(imagefit, "ANTHROPIC_LONG_EDGE", 2576)
    assert imagefit._cache_key(page, "anthropic") != base


def test_fit_disabled_returns_the_file_untouched(page, monkeypatch):
    monkeypatch.setattr(imagefit, "ENABLED", False)
    assert imagefit.fit(page, "anthropic")[0] == page.read_bytes()


def test_fit_falls_back_to_original_bytes_on_a_broken_image(tmp_path, monkeypatch):
    """A page that reaches the reader too large is a cost bug; one that does not
    reach it at all is a broken answer."""
    monkeypatch.setattr(imagefit, "CACHE_DIR", tmp_path / "fit")
    bad = tmp_path / "tile_0001.jpg"
    bad.write_bytes(b"not a jpeg")
    assert imagefit.fit(bad, "anthropic")[0] == b"not a jpeg"
