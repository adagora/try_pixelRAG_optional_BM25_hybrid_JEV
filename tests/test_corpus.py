"""Index metadata, and the two couplings that break silently when it moves.

Both tests below exist because the refactor that created `corpus.py` broke them
and nothing said so until a behavioural test failed several layers up. Neither
is about what the accessors return — that is ordinary and covered wherever it
is used — they are about the two facts that are true of the module's PLACE in
the repo rather than of its code, and which a later move would quietly undo.
"""

from __future__ import annotations

import json

import pytest

import citeparse
import corpus


@pytest.fixture
def redirected(tmp_path, monkeypatch):
    """Point the whole system at a throwaway index."""
    import layout

    idx = layout.IndexLayout(tmp_path / "index")
    idx.tile_dir(0).mkdir(parents=True)
    idx.articles_json.parent.mkdir(parents=True, exist_ok=True)
    idx.articles_json.write_text(
        json.dumps([{"title": "fixture-doc", "url": "pdfs/nope.pdf"}]),
        encoding="utf-8")
    monkeypatch.setattr(corpus, "LAYOUT", idx)
    corpus.articles.cache_clear()
    corpus.page_size.cache_clear()
    yield idx
    corpus.articles.cache_clear()
    corpus.page_size.cache_clear()


# --------------------------------------------------------------------------
# One owner for LAYOUT
# --------------------------------------------------------------------------

def test_redirecting_the_layout_redirects_every_accessor(redirected):
    """THE BUG THIS PINS: `articles()` and `page_path()` used to live in rag.py
    and read `rag.LAYOUT`. Moving them to a module with its own `LAYOUT` left
    the test suite redirecting one copy while the accessors read the other, so
    half the system was reading a temporary index and half the real one — and
    the only symptom was a page count being wrong two layers away.

    Every accessor must follow the same global, so redirecting it once is
    enough."""
    assert corpus.articles() == [{"title": "fixture-doc", "url": "pdfs/nope.pdf"}]
    assert corpus.doc_title(0) == "fixture-doc"
    assert redirected.index_dir in corpus.page_path(0, 0).parents


def test_nothing_else_keeps_its_own_copy_of_the_layout(redirected):
    """A second module-level `LAYOUT = layout.DEFAULT` anywhere above
    layout.py recreates the split this module exists to prevent."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "scripts"
    owners = [p.name for p in sorted(root.glob("*.py"))
              if "\nLAYOUT = " in p.read_text(encoding="utf-8")]
    assert owners == ["corpus.py"], f"LAYOUT has more than one owner: {owners}"


def test_a_missing_article_index_names_the_command_that_builds_it(tmp_path,
                                                                  monkeypatch):
    """One of the few places that fails rather than degrades, so the message is
    the whole feature."""
    import layout

    monkeypatch.setattr(corpus, "LAYOUT", layout.IndexLayout(tmp_path / "gone"))
    corpus.articles.cache_clear()
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        corpus.articles()
    corpus.articles.cache_clear()


def test_a_corrupt_article_index_says_so_rather_than_raising_json_errors(
        redirected):
    redirected.articles_json.write_text("{not json", encoding="utf-8")
    corpus.articles.cache_clear()
    with pytest.raises(ValueError, match="not valid JSON"):
        corpus.articles()
    corpus.articles.cache_clear()


def test_a_windows_built_index_still_resolves_its_sources(redirected):
    """`pdfs\\foo.pdf` is ONE filename containing a backslash to PurePosixPath,
    so without normalisation every source PDF reads as missing, citations lose
    their geometry, and nothing raises."""
    redirected.articles_json.write_text(
        json.dumps([{"title": "d", "url": "pdfs\\\\sub\\\\foo.pdf"}]),
        encoding="utf-8")
    corpus.articles.cache_clear()
    assert "\\" not in corpus.articles()[0]["url"]
    corpus.articles.cache_clear()


def test_source_pdf_is_none_rather_than_a_path_that_is_not_there(redirected):
    """Callers' next move is to read a text layer; a scanned corpus, a moved
    file and a .png source all answer the same way."""
    assert corpus.source_pdf(0) is None


def test_an_article_id_past_the_end_is_named_rather_than_an_IndexError(redirected):
    assert corpus.doc_title(99) == "?"


# --------------------------------------------------------------------------
# The instruction string's drift check knows where the constant lives
# --------------------------------------------------------------------------

def test_the_parity_check_still_points_at_the_file_that_owns_the_instruction():
    """THE OTHER SILENT BREAK. check_parity.py compares every copy of
    DEFAULT_INSTRUCTION at SOURCE level, by globbing file paths — because both
    of its encode arms share one body, so a wrong chat template cancels out and
    reads as cosine 1.0. A file that no longer matches the regex is skipped,
    not reported, so moving the constant out of a listed file turns the check
    off without failing anything.

    It moved once already, from rag.py to queryembed.py."""
    import check_parity

    found = {}
    for pattern in check_parity.INSTRUCTION_OWNERS:
        for path in sorted(check_parity.ROOT.glob(pattern)):
            m = check_parity._INSTRUCTION_RE.search(path.read_text(encoding="utf-8"))
            found[path.name] = bool(m)

    assert found, "INSTRUCTION_OWNERS matched no files at all"
    silent = [name for name, ok in found.items() if not ok]
    assert not silent, f"listed but no longer declares the constant: {silent}"
    # The repo's own copies must be among them — the site-packages one may not
    # be installed in every environment, so it is not required here.
    assert {"queryembed.py", "encoder.py", "profile_search.py"} <= set(found)


def test_every_repo_copy_of_the_instruction_agrees():
    import check_parity

    assert check_parity.instruction_mismatches() == []


# --------------------------------------------------------------------------
# citeparse reaches the index only through corpus
# --------------------------------------------------------------------------

def test_citeparse_locates_nothing_when_there_is_no_text_layer(redirected):
    """It must ask corpus for the source rather than keep its own copy of the
    lookup — the duplicate check it used to carry is how the two drifted."""
    assert citeparse.locate_values(0, 1, "cena 3555 zł") == []


def test_a_citation_naming_an_unattached_page_is_dropped(redirected):
    out = citeparse.resolve_citations(
        "odpowiedź", [{"page": 9, "quote": "x"}],
        [{"article_id": 0, "page": 1, "document": "d"}])
    assert out == []
