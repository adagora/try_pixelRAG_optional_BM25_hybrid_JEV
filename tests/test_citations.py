"""Quote -> rectangle. Built against a synthesised PDF so the three documented
failure modes (hyphenation, typeset thin spaces, multi-line quotes) are
reproducible without shipping a 82 MB price list."""

from __future__ import annotations

import pytest

import citations

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")


@pytest.fixture
def pdf(tmp_path):
    """One page carrying the shapes the real corpus breaks on."""
    path = tmp_path / "cennik.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)          # A4 points
    page.insert_text((72, 100), "Pakiet antywlamaniowy RC2")
    page.insert_text((72, 130), "Wkladka antywlamaniowa")
    page.insert_text((400, 130), "3 956")               # typeset with a space
    page.insert_text((72, 160), "skrzydlo wyko-")       # hyphenated across
    page.insert_text((72, 180), "nane jest ze stali")   #   two lines
    page.insert_text((72, 220), "2500")
    page.insert_text((200, 220), "2500")                # repeated figure
    page.insert_text((72, 260), "unikat 7016")
    doc.save(path)
    doc.close()
    return path


def test_locate_quote_finds_a_simple_row_label(pdf):
    found = citations.locate_quote(pdf, 1, "Pakiet antywlamaniowy RC2")
    assert found and found[0]["exact"]
    r = found[0]["rects"][0]
    assert 0 < r["left"] < 100 and 0 < r["top"] < 100


def test_locate_quote_returns_percentages_of_the_page(pdf):
    r = citations.locate_quote(pdf, 1, "Wkladka antywlamaniowa")[0]["rects"][0]
    assert r["left"] == pytest.approx(100 * 72 / 595, abs=1.5)


def test_locate_quote_spans_hyphenation(pdf):
    """The page literally contains 'wyko- nane', the reader writes 'wykonane'."""
    found = citations.locate_quote(pdf, 1, "skrzydlo wykonane jest")
    assert found
    assert found[0]["coverage"] >= 0.6


def test_locate_quote_returns_one_rect_per_line(pdf):
    """One box around both lines would cover the whole column and look broken."""
    found = citations.locate_quote(pdf, 1, "skrzydlo wykonane jest ze stali")
    assert len(found[0]["rects"]) >= 2


def test_locate_quote_missing_text_returns_nothing(pdf):
    """A citation the page does not contain is the clearest signal the reader
    drifted — it must not be silently approximated."""
    assert citations.locate_quote(pdf, 1, "cena montazu bramy roletowej") == []


def test_locate_quote_ignores_a_single_shared_stopword(pdf):
    """Below MIN_COVERAGE the best run is coincidence."""
    assert citations.locate_quote(pdf, 1, "jest") == []


def test_locate_quote_out_of_range_page(pdf):
    assert citations.locate_quote(pdf, 99, "Pakiet antywlamaniowy RC2") == []


def test_locate_quote_folds_diacritics_both_ways(pdf):
    assert citations.locate_quote(pdf, 1, "Pakiet antywłamaniowy RC2")


# -- number pinning ---------------------------------------------------------

def test_locate_numbers_pins_a_unique_figure(pdf):
    pinned, repeated = citations.locate_numbers(pdf, 1, "kosztuje 7016 PLN netto")
    assert [p["value"] for p in pinned] == ["7016"]
    assert repeated == []


def test_locate_numbers_reports_repeats_without_geometry(pdf):
    """Fourteen boxes of noise around the one that matters is worse than none."""
    pinned, repeated = citations.locate_numbers(pdf, 1, "wymiar 2500 mm")
    assert pinned == []
    assert repeated == [{"value": "2500", "count": 2}]


def test_locate_numbers_ignores_figures_outside_3_to_6_digits(pdf):
    pinned, repeated = citations.locate_numbers(pdf, 1, "pozycja 20 z roku 12345678")
    assert pinned == [] and repeated == []


def test_locate_numbers_normalises_typeset_spacing(pdf):
    """The page has '3 956'; the reader writes '3956'."""
    pinned, _ = citations.locate_numbers(pdf, 1, "cena 3956 zl")
    assert [p["value"] for p in pinned] == ["3956"]


def test_locate_numbers_on_a_page_without_text(tmp_path):
    blank = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(blank)
    doc.close()
    assert citations.locate_numbers(blank, 1, "3956") == ([], [])


# -- token normalisation ----------------------------------------------------

def test_norm_tokens_keeps_digits_with_letters():
    assert citations._norm_tokens("RC2 W3-1") == ["rc2", "w3", "1"]


def test_norm_tokens_folds_polish():
    assert citations._norm_tokens("skrzydło") == ["skrzydlo"]
