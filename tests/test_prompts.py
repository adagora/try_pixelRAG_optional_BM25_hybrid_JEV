"""The citation contract, which is spread across three files and has no seam.

`prompts.py` DEMANDS a `---CYTATY---` block and a page number; `page_header`
STATES that page number in the shape the reader is told to echo; `citeparse.py`
PARSES what comes back. Nothing connects them but agreement, and a change to
any one alone produces answers that still read perfectly and whose citations
silently stop resolving — no exception, no empty result, just highlights that
quietly disappear.

So these tests take the literal example out of the prompt the model is given
and push it through the parser, rather than asserting on a string either side
invented. If the format changes on one side only, the round trip breaks here.
"""

from __future__ import annotations

import json
import re

import pytest

import citeparse
import prompts


# --------------------------------------------------------------------------
# The contract holds end to end
# --------------------------------------------------------------------------

def _example_lines() -> list[str]:
    """The JSONL lines written out as an example inside ONESHOT_SYSTEM itself."""
    _, _, tail = prompts.ONESHOT_SYSTEM.partition(citeparse.CITE_MARK)
    return [ln.strip() for ln in tail.splitlines() if ln.strip().startswith("{")]


def test_the_prompt_demands_the_marker_the_parser_looks_for():
    assert citeparse.CITE_MARK in prompts.ONESHOT_SYSTEM
    assert citeparse.CITE_MARK in prompts.oneshot_preamble("q", [])


def test_the_example_citations_in_the_prompt_are_parseable_by_the_parser():
    """If the worked example the model is shown does not survive our own
    parser, the model copying it faithfully is the failure case."""
    lines = _example_lines()
    assert lines, "ONESHOT_SYSTEM no longer shows an example citation block"

    answer = "Odpowiedź.\n\n" + citeparse.CITE_MARK + "\n" + "\n".join(lines)
    body, cites = citeparse.split_citations(answer)

    assert body == "Odpowiedź."
    assert len(cites) == len(lines)
    for c in cites:
        assert isinstance(c["page"], int)
        assert c["quote"]


def test_a_citation_using_the_prompts_own_example_resolves_to_a_page(monkeypatch):
    """The whole round trip: the shape the prompt asks for, matched against
    the attached-page records rag.py builds.

    No source PDF, deliberately — this is about the format agreeing across the
    three files, and reaching into pdfs/ would make it a test of whichever
    catalogue happens to be checked in."""
    import corpus

    monkeypatch.setattr(corpus, "source_pdf", lambda aid: None)
    line = json.loads(_example_lines()[0])
    pages = [{"article_id": 0, "page": line["page"], "document": "kat"}]

    out = citeparse.resolve_citations("Odpowiedź", [line], pages)

    assert len(out) == 1
    assert out[0]["page"] == line["page"]
    assert out[0]["document"] == "kat"
    # No PDF behind the fixture, so there is no geometry — and the citation is
    # KEPT and flagged rather than dropped, which is the documented behaviour.
    assert out[0]["verified"] is False


# --------------------------------------------------------------------------
# page_header states the number in the form the prompt promises
# --------------------------------------------------------------------------

def test_the_page_header_states_the_number_the_prompt_tells_the_model_to_echo():
    """`"page" — numer strony podany w nagłówku obrazu, dokładnie tak jak
    podano.` The prompt makes a promise about the header; this is the header."""
    header = prompts.page_header({"document": "kat", "page": 61})
    assert "page=61" in header
    assert "strona 61" in header


@pytest.mark.parametrize("page", [1, 7, 61, 214])
def test_the_number_in_the_header_is_recoverable_as_an_integer(page):
    """What the model reads back out of the header has to survive `int()`,
    because that is what the parser does to it."""
    header = prompts.page_header({"document": "d", "page": page})
    found = re.search(r"page=(\d+)", header)
    assert found and int(found.group(1)) == page


def test_the_header_carries_the_document_so_two_page_61s_stay_apart():
    """Both catalogues have a page 61. The header naming the document is what
    lets a reader disambiguate, and citeparse keys on (article_id, page) for
    the same reason."""
    assert "kat-a" in prompts.page_header({"document": "kat-a", "page": 61})


# --------------------------------------------------------------------------
# The preamble's inventory
# --------------------------------------------------------------------------

def test_the_preamble_lists_every_attached_page_with_its_match_count():
    """The family-disambiguation rule depends on the reader being able to see
    it was handed three pages from one catalogue."""
    pages = [{"document": "kat", "page": 3, "n_chunks": 2},
             {"document": "kat", "page": 4, "n_chunks": 1}]
    text = prompts.oneshot_preamble("Ile kosztuje brama?", pages)

    assert "Ile kosztuje brama?" in text
    assert "strona 3" in text and "strona 4" in text
    assert "2" in text and "1" in text


def test_a_page_with_no_chunk_count_still_appears_in_the_inventory():
    """`n_chunks` is absent for a page BM25 nominated that the pixel index
    never saw — exactly the page most worth telling the reader about."""
    text = prompts.oneshot_preamble("q", [{"document": "kat", "page": 9}])
    assert "strona 9" in text


def test_no_attached_pages_still_produces_a_usable_preamble():
    text = prompts.oneshot_preamble("q", [])
    assert "q" in text and isinstance(text, str)


# --------------------------------------------------------------------------
# The two prompts are for two different jobs
# --------------------------------------------------------------------------

def test_the_agent_prompt_names_the_tools_the_agent_is_actually_given():
    """A prompt describing a tool the model was not handed is how a browse loop
    stalls calling something that does not exist."""
    named = {t["name"] for t in prompts.TOOLS}
    assert named == {"pixelrag_search", "pixelrag_tile"}
    for name in named:
        assert name in prompts.SYSTEM


def test_the_one_shot_prompt_offers_no_tools_at_all():
    """There is nothing to call on that path; telling it otherwise produces a
    prompt that hedges about tools that are not there."""
    for name in (t["name"] for t in prompts.TOOLS):
        assert name not in prompts.ONESHOT_SYSTEM


def test_every_tool_declares_a_schema_the_provider_layer_can_send():
    for t in prompts.TOOLS:
        assert t["description"].strip()
        schema = t["input_schema"]
        assert schema["type"] == "object"
        for required in schema.get("required", []):
            assert required in schema["properties"], (t["name"], required)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

def test_the_two_refusals_say_different_things():
    """One sentence used to cover both, and it was wrong for one of them:
    'rebuild your index' is unhelpful advice to someone who asked a door
    catalogue about engine oil."""
    assert set(prompts.REFUSAL_PL) == {"not-in-corpus", "out-of-scope"}
    assert prompts.REFUSAL_PL["not-in-corpus"] != prompts.REFUSAL_PL["out-of-scope"]


def test_only_the_in_corpus_refusal_suggests_a_different_document():
    """The out-of-scope case is not fixed by finding another catalogue."""
    assert "cennik" in prompts.REFUSAL_PL["not-in-corpus"].lower()
    assert "cennik" not in prompts.REFUSAL_PL["out-of-scope"].lower()


def test_the_no_pages_answer_tells_the_user_what_to_check():
    """It is shown when retrieval returned nothing, which is usually a missing
    document or a stale index, and both are actionable."""
    assert "pdfs/" in prompts.NO_PAGES_PL
