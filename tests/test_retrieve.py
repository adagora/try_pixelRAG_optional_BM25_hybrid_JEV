"""Ranking policy: aggregation, fusion, RRF, query variants.

Pinned because these are the numbers the README quotes recall figures against.
A refactor that changes an ordering here has changed the product.
"""

from __future__ import annotations

import pytest

import retrieve


# -- query variants ---------------------------------------------------------

def test_noun_phrase_strips_interrogative_scaffolding():
    assert retrieve.noun_phrase("Ile kosztuje wkładka antywłamaniowa?") == \
        "wkładka antywłamaniowa"
    assert retrieve.noun_phrase("Pokaż rodzaje paneli") == "rodzaje paneli"


def test_noun_phrase_prefers_longest_prefix():
    # "jaka jest" must win over "jaka", or the leftover "jest" pollutes the query.
    assert retrieve.noun_phrase("Jaka jest cena bramy") == "cena bramy"


def test_noun_phrase_empty_when_nothing_to_strip():
    assert retrieve.noun_phrase("dzielony wał") == ""


def test_query_variants_always_leads_with_the_question_as_asked():
    v = retrieve.query_variants("Ile kosztuje dzielony wał?")
    assert v[0] == "Ile kosztuje dzielony wał?"
    assert v[1] == "dzielony wał"


def test_query_variants_single_when_no_scaffolding():
    assert retrieve.query_variants("dzielony wał") == ["dzielony wał"]


# -- aggregation ------------------------------------------------------------

def test_aggregate_groups_chunks_by_page(scale_of, hits):
    out = retrieve.aggregate(hits((0, 0, 1, 0.5), (0, 0, 2, 0.4), (0, 1, 1, 0.45)),
                             scale_of)
    assert len(out) == 2
    top = out[0]
    assert (top["article_id"], top["tile_index"]) == (0, 0)
    assert top["n_chunks"] == 2
    assert top["page"] == 1                      # 1-based for display


def test_aggregate_agreement_bonus_is_applied(scale_of, hits):
    """best + AGREEMENT * sum(rest) — the whole reason pages beat chunks."""
    out = retrieve.aggregate(hits((0, 0, 1, 0.5), (0, 0, 2, 0.4)), scale_of)
    assert out[0]["score"] == pytest.approx(0.5 + retrieve.AGREEMENT * 0.4)


def test_aggregate_one_clean_hit_beats_several_mediocre(scale_of, hits):
    """The stated design constraint: agreement is evidence, not a majority vote."""
    out = retrieve.aggregate(
        hits((0, 0, 1, 0.90),
             (0, 1, 1, 0.50), (0, 1, 2, 0.50)),
        scale_of)
    assert out[0]["tile_index"] == 0


def test_aggregate_focus_is_a_region_never_the_gist(scale_of, hits):
    """A gist box spans the page, so highlighting it would mean nothing."""
    out = retrieve.aggregate(hits((0, 0, 0, 0.9), (0, 0, 2, 0.4)), scale_of)
    assert out[0]["focus"] == 2


def test_aggregate_focus_none_when_only_gist_matched(scale_of, hits):
    out = retrieve.aggregate(hits((0, 0, 0, 0.9)), scale_of)
    assert out[0]["focus"] is None


def test_aggregate_missing_manifest_degrades_to_region(hits):
    """An index built before `scale` was recorded must still rank."""
    out = retrieve.aggregate(hits((0, 0, 1, 0.5)), lambda a, t, c: "region")
    assert out[0]["n_chunks"] == 1


# -- fusion of phrasings ----------------------------------------------------

def test_fuse_normalises_per_variant():
    """A two-word query scores lower against everything; magnitudes are not
    comparable across variants, only within one."""
    a = [{"article_id": 0, "tile_index": 0, "score": 0.60, "chunks": [],
          "focus": 1, "n_chunks": 1}]
    b = [{"article_id": 0, "tile_index": 1, "score": 0.30, "chunks": [],
          "focus": 2, "n_chunks": 1}]
    out = retrieve.fuse([a, b])
    assert {p["norm"] for p in out} == {1.0}     # each is its own variant's best


def test_fuse_keeps_strongest_evidence_and_its_geometry():
    """Strongest means highest NORMALISED score — raw magnitudes are not
    comparable between a full question and its noun phrase. Page 0 is variant
    A's also-ran and variant B's best, so B's chunk list is the one that
    survives, and the highlight box comes from B."""
    a = [{"article_id": 0, "tile_index": 9, "score": 0.8, "chunks": [],
          "focus": None, "n_chunks": 0},
         {"article_id": 0, "tile_index": 0, "score": 0.5, "chunks": [{"x": 1}],
          "focus": 1, "n_chunks": 1}]
    b = [{"article_id": 0, "tile_index": 0, "score": 0.6, "chunks": [{"x": 2}],
          "focus": 5, "n_chunks": 3}]
    page0 = next(p for p in retrieve.fuse([a, b]) if p["tile_index"] == 0)
    assert page0["focus"] == 5
    assert page0["n_chunks"] == 3


def test_fuse_ties_on_norm_keep_the_first_variant():
    """Both variants ranked it first, so both norms are 1.0. The question as
    asked is variant 0 and wins the tie — deliberate, not incidental."""
    asked = [{"article_id": 0, "tile_index": 0, "score": 0.5, "chunks": [],
              "focus": 1, "n_chunks": 1}]
    phrase = [{"article_id": 0, "tile_index": 0, "score": 0.9, "chunks": [],
               "focus": 5, "n_chunks": 3}]
    assert retrieve.fuse([asked, phrase])[0]["focus"] == 1


def test_fuse_admits_pages_found_by_only_one_variant():
    """Where the recall gain comes from."""
    a = [{"article_id": 0, "tile_index": 0, "score": 0.9, "chunks": [],
          "focus": 1, "n_chunks": 1}]
    b = [{"article_id": 0, "tile_index": 7, "score": 0.4, "chunks": [],
          "focus": 1, "n_chunks": 1}]
    assert {p["tile_index"] for p in retrieve.fuse([a, b])} == {0, 7}


# -- hybrid RRF -------------------------------------------------------------

def test_rrf_scores_by_rank_not_magnitude():
    visual = [{"article_id": 0, "tile_index": 0, "score": 0.42, "source": "visual"},
              {"article_id": 0, "tile_index": 1, "score": 0.41, "source": "visual"}]
    lex = [{"article_id": 0, "tile_index": 1, "score": 8.0, "source": "lexical"}]
    out = retrieve.rrf([visual, lex])
    k = retrieve.RRF_K
    top = out[0]
    assert top["tile_index"] == 1                # agreed on by both retrievers
    assert top["rrf"] == pytest.approx(1 / (k + 2) + 1 / (k + 1))


def test_rrf_records_provenance():
    visual = [{"article_id": 0, "tile_index": 0, "score": 0.4, "source": "visual"}]
    lex = [{"article_id": 0, "tile_index": 0, "score": 8.0, "source": "lexical"}]
    out = retrieve.rrf([visual, lex])
    assert sorted(out[0]["found_by"]) == ["lexical", "visual"]


def test_rrf_keeps_whichever_list_carried_geometry():
    """A text-only hit has no region to box; the visual list's chunks must win."""
    visual = [{"article_id": 0, "tile_index": 0, "score": 0.4, "source": "visual",
               "chunks": [{"c": 1}], "focus": 3, "n_chunks": 2}]
    lex = [{"article_id": 0, "tile_index": 0, "score": 8.0, "source": "lexical",
            "chunks": [], "focus": None, "n_chunks": 0}]
    assert retrieve.rrf([lex, visual])[0]["focus"] == 3


def test_rrf_weight_shifts_the_lexical_vote():
    visual = [{"article_id": 0, "tile_index": 0, "score": 0.4, "source": "visual"}]
    lex = [{"article_id": 0, "tile_index": 1, "score": 8.0, "source": "lexical"}]
    assert retrieve.rrf([visual, lex], weights=[1.0, 0.0])[0]["tile_index"] == 0
    assert retrieve.rrf([visual, lex], weights=[0.0, 1.0])[0]["tile_index"] == 1


# -- the whole pipeline, with retrievers injected ---------------------------

def test_retrieve_pages_visual_only(scale_of, hits):
    calls = []

    def fake_search(q, n):
        calls.append((q, n))
        return hits((0, 0, 1, 0.9), (0, 1, 1, 0.4))

    pages, debug = retrieve.retrieve_pages(
        fake_search, "Ile kosztuje dzielony wał?", scale_of, n_pages=2)
    assert len(calls) == 2                       # question + noun phrase
    assert [p["page"] for p in pages] == [1, 2]
    assert [d["retriever"] for d in debug] == ["visual", "visual"]


def test_retrieve_pages_respects_n_pages(scale_of, hits):
    pages, _ = retrieve.retrieve_pages(
        lambda q, n: hits((0, 0, 1, 0.9), (0, 1, 1, 0.4)),
        "dzielony wał", scale_of, n_pages=1)
    assert len(pages) == 1


def test_retrieve_pages_hybrid_fuses_the_text_layer(scale_of, hits):
    def fake_lex(q, n):
        return [{"article_id": 0, "page": 2, "score": 8.0}]

    pages, debug = retrieve.retrieve_pages(
        lambda q, n: hits((0, 0, 1, 0.9)), "dzielony wał", scale_of,
        n_pages=2, lexical_fn=fake_lex)
    assert {p["page"] for p in pages} == {1, 2}
    assert debug[-1]["retriever"] == "lexical"


def test_retrieve_pages_variant_results_stay_in_variant_order(scale_of, hits):
    """The two phrasings are searched concurrently, but fuse() reads variant 0
    as the question as asked. Results must be re-paired with their own query,
    not with whichever thread finished first."""
    per_query = {"Ile kosztuje dzielony wał?": hits((0, 0, 1, 0.5)),
                 "dzielony wał": hits((0, 1, 1, 0.5))}

    _, debug = retrieve.retrieve_pages(
        lambda q, n: per_query[q], "Ile kosztuje dzielony wał?", scale_of)
    assert [d["query"] for d in debug] == ["Ile kosztuje dzielony wał?",
                                           "dzielony wał"]
    assert debug[0]["top"][0]["page"] == 1       # what variant 0 actually found
    assert debug[1]["top"][0]["page"] == 2
