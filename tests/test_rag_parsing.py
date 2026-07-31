"""rag.py's pure helpers: the citation block, page-number conversion, provider
detection, cache namespacing and cost arithmetic.

Everything here used to be reachable only through a live search service and a
billed reader call.
"""

from __future__ import annotations

import pytest

import providers
import rag


# -- the ---CYTATY--- block -------------------------------------------------

def test_split_citations_separates_prose_from_the_block():
    body, cites = rag.split_citations(
        'Cena to 3956 zł netto.\n'
        '---CYTATY---\n'
        '{"page": 61, "quote": "Wkladka antywlamaniowa", "supports": "pozycja 20"}\n')
    assert body == "Cena to 3956 zł netto."
    assert cites == [{"page": 61, "quote": "Wkladka antywlamaniowa",
                      "supports": "pozycja 20"}]


def test_split_citations_without_a_block():
    assert rag.split_citations("Nie znalazłem.") == ("Nie znalazłem.", [])


def test_split_citations_tolerates_code_fences():
    """A reader that fences the block must still produce a usable answer."""
    _, cites = rag.split_citations(
        'A.\n---CYTATY---\n```\n{"page": 1, "quote": "x"}\n```\n')
    assert len(cites) == 1


def test_split_citations_drops_malformed_lines_and_keeps_the_rest():
    _, cites = rag.split_citations(
        'A.\n---CYTATY---\n'
        '{"page": 1, "quote": "dobry"}\n'
        '{"page": 2, "quote": nieprawidlowy}\n'
        'zwykly tekst\n'
        '{"page": 3, "quote": "tez dobry"}\n')
    assert [c["quote"] for c in cites] == ["dobry", "tez dobry"]


def test_split_citations_requires_a_quote():
    _, cites = rag.split_citations('A.\n---CYTATY---\n{"page": 1}\n')
    assert cites == []


def test_split_citations_strips_the_block_from_the_answer():
    """The marker exists so raw citation JSON never reaches the user."""
    body, _ = rag.split_citations('Odpowiedź.\n---CYTATY---\n{"page":1,"quote":"q"}')
    assert rag.CITE_MARK not in body


# -- page spec conversion ---------------------------------------------------

def test_parse_pages():
    assert rag._parse_pages("0:0-5,1:0-4") == {0: (0, 5), 1: (0, 4)}


def test_parse_pages_ignores_junk():
    assert rag._parse_pages("0:0-5,rubbish,,2:1-3") == {0: (0, 5), 2: (1, 3)}


def test_parse_pages_empty():
    assert rag._parse_pages(None) == {} and rag._parse_pages("") == {}


def test_pages_1based_shifts_only_the_page():
    """Regions stay 0-based; only pages are renumbered, and they must match the
    numbering pixelrag_tile accepts and citations quote."""
    assert rag._pages_1based("0:0-5,1:0-5") == "1:0-5,2:0-5"


def test_pages_1based_sorts():
    assert rag._pages_1based("2:0-1,0:0-1") == "1:0-1,3:0-1"


# -- provider detection -----------------------------------------------------

def test_detect_provider_prefers_a_pasted_key_over_the_environment(monkeypatch):
    """An exported ANTHROPIC_API_KEY used to steal Gemini pastes and surface as
    a connection error against the wrong host."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-whatever")
    monkeypatch.delenv("PIXELRAG_PROVIDER", raising=False)
    assert rag.detect_provider("AIzaSyPasted") == "gemini"


def test_detect_provider_recognises_key_shapes(monkeypatch):
    monkeypatch.delenv("PIXELRAG_PROVIDER", raising=False)
    assert rag.detect_provider("sk-ant-x") == "anthropic"
    assert rag.detect_provider("ya29.x") == "gemini"


def test_detect_provider_env_override_wins(monkeypatch):
    monkeypatch.setenv("PIXELRAG_PROVIDER", "anthropic")
    assert rag.detect_provider("AIzaSy") == "anthropic"


def test_detect_provider_falls_back_to_gemini(monkeypatch):
    for v in ("PIXELRAG_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    assert rag.detect_provider(None) == "gemini"


# -- cache namespace --------------------------------------------------------

def test_cache_namespace_separates_everything_that_moves_the_answer():
    """Only the question is fuzzy-matched; these are exact-match, because a
    stale hit quotes last week's price list."""
    base = rag._cache_namespace("gemini", "m", "oneshot")
    assert base != rag._cache_namespace("anthropic", "m", "oneshot")
    assert base != rag._cache_namespace("gemini", "other", "oneshot")
    assert base != rag._cache_namespace("gemini", "m", "agent")


def test_cache_namespace_includes_the_index_identity():
    assert rag.answer_cache.index_fingerprint(rag.INDEX_DIR) in \
        rag._cache_namespace("gemini", "m", "oneshot")


# -- cost -------------------------------------------------------------------

def test_anthropic_cost_bills_cached_tokens_at_their_own_rate():
    """input_tokens is the UNCACHED remainder; summing everything at full rate
    would overstate the bill and hide the saving the cache exists to produce."""
    r = rag._done("a", [], providers.Usage(input=1_000_000, output=0,
                                           cache_read=1_000_000,
                                           cache_write=1_000_000),
                  1, providers.AnthropicReader())
    rin, _ = providers.ANTHROPIC_RATES
    assert r["usage"]["cost_usd"] == pytest.approx(rin + rin * 0.1 + rin * 1.25)


def test_gemini_reports_tokens_but_not_a_rate():
    """Pricing varies by model and tier, so that path does not guess."""
    r = rag._done("a", [], providers.Usage(input=100, output=50, thoughts=10),
                  1, providers.GeminiReader())
    assert r["usage"]["cost_usd"] is None
    assert r["usage"]["thoughts"] == 10


def test_done_splits_the_citation_block_out_of_the_answer(fake_reader):
    r = rag._done('Odp.\n---CYTATY---\n{"page":1,"quote":"q"}', [],
                  providers.Usage(), 1, fake_reader())
    assert r["answer"] == "Odp."


def test_done_never_fails_the_answer_over_highlighting(monkeypatch, fake_reader):
    """Highlighting is a nicety; a broken PDF must not lose a good answer."""
    def boom(*a, **k):
        raise RuntimeError("no text layer")

    monkeypatch.setattr(rag, "resolve_citations", boom)
    r = rag._done('Odp.\n---CYTATY---\n{"page":1,"quote":"q"}',
                  [{"type": "tile", "article_id": 0, "page": 1, "document": "d"}],
                  providers.Usage(), 1, fake_reader())
    assert r["answer"] == "Odp." and r["citations"] == []


# -- resolving a citation to a document -------------------------------------

def _pages():
    """Two catalogues that both have a page 61 — the case that used to draw a
    highlight in the wrong document."""
    return [{"article_id": 1, "page": 61, "document": "ECHO"},
            {"article_id": 0, "page": 61, "document": "BR-77"}]


def test_resolve_citations_uses_the_article_id_when_the_reader_gives_one(monkeypatch):
    monkeypatch.setattr(rag, "_source_pdf", lambda aid: None)
    out = rag.resolve_citations(
        "", [{"page": 61, "quote": "q", "article_id": 0}], _pages())
    assert out[0]["document"] == "BR-77"


def test_resolve_citations_falls_back_to_the_highest_ranked_page(monkeypatch):
    """The reader names only a page number, so the article is recovered from the
    pages actually attached, preferring the one retrieval ranked first."""
    monkeypatch.setattr(rag, "_source_pdf", lambda aid: None)
    out = rag.resolve_citations("", [{"page": 61, "quote": "q"}], _pages())
    assert out[0]["article_id"] == 1


def test_resolve_citations_drops_a_page_that_was_never_attached(monkeypatch):
    monkeypatch.setattr(rag, "_source_pdf", lambda aid: None)
    assert rag.resolve_citations("", [{"page": 999, "quote": "q"}], _pages()) == []


def test_resolve_citations_drops_a_non_numeric_page(monkeypatch):
    monkeypatch.setattr(rag, "_source_pdf", lambda aid: None)
    assert rag.resolve_citations("", [{"page": "sześćdziesiąt", "quote": "q"}],
                                 _pages()) == []


def test_resolve_citations_flags_an_unverified_quote(monkeypatch):
    """A quote that cannot be found on the page it names is kept and flagged —
    the clearest available signal that the reader drifted."""
    monkeypatch.setattr(rag, "_source_pdf", lambda aid: None)
    out = rag.resolve_citations("", [{"page": 61, "quote": "q"}], _pages())
    assert out[0]["verified"] is False and out[0]["rects"] == []


# -- ask-mode validation ----------------------------------------------------

def test_run_agent_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="Unknown ask mode"):
        rag.run_agent("q", mode="browse")


def test_run_agent_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.delenv("PIXELRAG_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="Unknown provider"):
        rag.run_agent("q", provider="openai")
