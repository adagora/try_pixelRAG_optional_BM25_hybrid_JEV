"""Retrieval modes: what each one runs, and what it reports having run.

The comparison UI is only worth looking at if a row labelled `jev` really ran
Jev and a row labelled `visual` really did not. Two things make that fragile and
both are covered here: the modes are resolved from environment flags that mean
different things (PIXELRAG_HYBRID picks a default, PIXELRAG_JEV is a kill
switch), and every stage degrades quietly by design, so a mode that fell back
produces a perfectly normal-looking row unless the numbers say otherwise.

No network, no index, no key: the faiss call and the Jev call are both scripted.
"""

from __future__ import annotations

import pytest

import compare
import jev
import rag


def chunks(*triples):
    return [{"article_id": a, "tile_index": t, "chunk_index": c, "score": s}
            for a, t, c, s in triples]


@pytest.fixture
def wired(monkeypatch):
    """rag with the faiss HTTP call scripted and no Jev key in the environment."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("PIXELRAG_JEV", raising=False)
    monkeypatch.setattr(rag, "HYBRID", False)
    monkeypatch.setattr(rag, "_scale_of", lambda a, t, c: "region")
    monkeypatch.setattr(rag, "doc_title", lambda aid: f"doc{aid}")
    seen = []

    def raw(query, n, timeout=120):
        seen.append(query)
        return chunks((0, 0, 1, 0.62), (0, 0, 2, 0.55), (0, 1, 1, 0.48))

    monkeypatch.setattr(rag, "_raw_search", raw)
    return seen


def jev_wired(monkeypatch, *, scores=None):
    """A key, scripted Jev answers, and chunk text for every candidate."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(rag, "_chunk_texts",
                        lambda hits: ["tekst strony"] * len(hits))

    def evaluate(state, questions, report=None, kind=""):
        if report is not None:
            report.calls.append(jev.Call(kind, 40.0, 900, 60))
        if kind == "expand":
            return {k: {"noul": 0.95 if k == "0" else 0.1} for k in questions}
        return {k: (scores or {}).get(k, {"score": 2.0}) for k in questions}

    monkeypatch.setattr(jev, "evaluate", evaluate)


# -- resolving a mode -------------------------------------------------------

def test_auto_is_whatever_the_two_flags_ask_for(monkeypatch, wired):
    assert rag.resolve_retrieval("auto") == "visual"
    monkeypatch.setattr(rag, "HYBRID", True)
    assert rag.resolve_retrieval(None) == "hybrid"
    # A key makes a Jev mode the default; `manual` keeps it selectable only.
    # The Jev default does NOT depend on PIXELRAG_HYBRID any more: `jev-page`
    # runs BM25 to build its own candidate pool, so there is no hybrid variant
    # of it to choose between. See rag.default_retrieval for the measurement
    # that moved this off `jev`/`jev+hybrid`.
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert rag.resolve_retrieval("") == "jev-page"
    monkeypatch.setenv("PIXELRAG_JEV", "manual")
    assert rag.resolve_retrieval("") == "hybrid"
    assert rag.retrieval_blocked("jev+hybrid") is None     # still selectable
    monkeypatch.setenv("PIXELRAG_JEV", "0")
    assert "PIXELRAG_JEV" in rag.retrieval_blocked("jev+hybrid")
    monkeypatch.delenv("PIXELRAG_JEV")
    monkeypatch.setattr(rag, "HYBRID", False)
    assert rag.default_retrieval() == "jev-page"


def test_an_unknown_mode_is_refused_rather_than_guessed(wired):
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        rag.resolve_retrieval("rerank-everything")


def test_naming_hybrid_overrides_the_default_flag_but_not_a_missing_sidecar(
        monkeypatch, wired, tmp_path):
    """PIXELRAG_HYBRID picks the default; only the sidecar decides if it can run."""
    import lexical

    monkeypatch.setattr(lexical, "TEXT_SIDECAR", tmp_path / "text.json")
    assert rag.retrieval_blocked("hybrid") == (
        "no BM25 text sidecar — run scripts/build_text_index.py")
    (tmp_path / "text.json").write_text("{}")
    assert rag.retrieval_blocked("hybrid") is None
    assert rag._lexical_fn() is None                      # default still off
    assert rag._lexical_fn(required=True) is lexical.search_text


def test_the_jev_kill_switch_cannot_be_overridden_from_the_browser(
        monkeypatch, wired):
    assert "TYPESAFE_API_KEY" in rag.retrieval_blocked("jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert rag.retrieval_blocked("jev") is None
    monkeypatch.setenv("PIXELRAG_JEV", "0")
    assert "PIXELRAG_JEV" in rag.retrieval_blocked("jev+hybrid")


# -- what a mode actually runs ----------------------------------------------

def test_visual_mode_never_calls_jev_even_with_a_key(monkeypatch, wired):
    """A visual row under a configured key must still be a visual row."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev, "search", lambda *a, **k: pytest.fail("Jev was called"))
    pages, _, stats = rag.retrieve_for("Ile kosztuje brama?", "visual", n_pages=2)
    assert [p.page for p in pages] == [1, 2]
    assert stats["jev"] is None and stats["reranker"] == "none"


def test_stats_count_the_funnel_that_actually_happened(wired):
    _, debug, stats = rag.retrieve_for("Ile kosztuje brama?", "visual", n_pages=2)
    assert stats["searches"] == len(wired) == len(debug)   # one per phrasing
    assert stats["chunk_hits"] == 3 * stats["searches"]    # with duplicates
    assert stats["unique_chunks"] == 3                     # without
    assert stats["candidate_pages"] == 2
    assert stats["pages"] == 2 and stats["ranked_pages"] == 2
    assert stats["cost_usd"] == 0.0  # ubs:ignore — literal 0.0, not arithmetic: no paid API at all
    assert stats["ms"] >= 0 and stats["lexical_hits"] == 0


def test_jev_mode_searches_its_own_expansions_and_reports_the_bill(
        monkeypatch, wired):
    jev_wired(monkeypatch)
    _, _, stats = rag.retrieve_for("Ile kosztuje brama?", "jev", n_pages=2)
    assert stats["reranker"] == "jev"
    assert stats["jev"]["reranked"] and not stats["jev"]["fallbacks"]
    # The facet variant is searched on top of the question and its noun phrase.
    assert stats["searches"] == 3 and stats["jev"]["candidates"] == 3
    assert stats["jev"]["input_tokens"] == 1800          # two calls, 900 each
    assert stats["jev"]["cost_usd"] == pytest.approx(1800 * 0.042 / 1e6)
    assert any(f["used"] for f in stats["jev"]["facets"])


def test_jev_expand_buys_the_phrasings_and_not_the_reranker(monkeypatch, wired):
    """The whole point of the mode: one cheap call, then `visual` verbatim."""
    jev_wired(monkeypatch)
    monkeypatch.setattr(jev, "search", lambda *a, **k: pytest.fail("reranked"))
    _, debug, stats = rag.retrieve_for("Ile kosztuje brama?", "jev-expand", n_pages=2)
    assert stats["reranker"] == "none"
    assert stats["fusion"] == "score-norm(variants)"
    # One call, not two — and the facet variant is searched like any phrasing,
    # which is what makes this comparable to visual by subtraction.
    assert [c["kind"] for c in stats["jev"]["calls"]] == ["expand"]
    assert stats["searches"] == 3 and len(debug) == 3
    assert stats["jev"]["input_tokens"] == 900


def test_jev_expand_survives_typesafe_being_down(monkeypatch, wired):
    """Falling back lands on exactly the phrasings `visual` would have used, so
    an outage costs the timeout and nothing else."""
    import requests

    jev_wired(monkeypatch)

    def fail(*a, **k):
        raise requests.Timeout()

    monkeypatch.setattr(jev, "evaluate", fail)
    _, debug, stats = rag.retrieve_for("Ile kosztuje brama?", "jev-expand", n_pages=2)
    assert stats["notes"] and "expansion failed" in stats["notes"][0]
    assert [row["query"] for row in debug] == ["Ile kosztuje brama?", "brama"]


def test_a_jev_run_that_fell_back_says_so_in_the_row(monkeypatch, wired):
    """Without this the comparison shows a Jev row that was billed for an API
    call and ranked by cosine, labelled exactly like one that was not."""
    jev_wired(monkeypatch)
    monkeypatch.setattr(rag, "_chunk_texts", lambda hits: [""] * len(hits))
    _, _, stats = rag.retrieve_for("q", "jev", n_pages=2)
    assert stats["reranker"] == "none"
    assert stats["notes"] and "PDF text" in stats["notes"][0]


def test_bm25_votes_once_per_mode_not_twice(monkeypatch, wired, tmp_path):
    """In jev+hybrid the text layer is already in the candidate pool, so fusing
    it again would let one retriever vote twice on the same evidence."""
    import lexical

    monkeypatch.setattr(lexical, "TEXT_SIDECAR", tmp_path / "text.json")
    (tmp_path / "text.json").write_text("{}")
    monkeypatch.setattr(lexical, "search_text",
                        lambda q, n: [{"article_id": 0, "page": 2, "score": 7.4}])
    jev_wired(monkeypatch)

    _, debug, stats = rag.retrieve_for("q", "hybrid", n_pages=2)
    assert stats["fusion"] == "rrf(visual+bm25)"
    assert [row["retriever"] for row in debug][-1] == "lexical"

    _, debug, stats = rag.retrieve_for("q", "jev+hybrid", n_pages=2)
    assert stats["fusion"] == "jev score"
    assert all(row["retriever"] == "visual" for row in debug)
    assert stats["lexical_hits"] and stats["jev"]["candidates"] == 4


# -- the comparison ---------------------------------------------------------

def test_a_blocked_mode_is_a_row_that_says_why_and_runs_nothing(wired):
    row = compare.run("q", "jev")
    assert not row["ok"] and "TYPESAFE_API_KEY" in row["error"]
    assert row["pages"] == [] and row["stats"] is None


def test_a_failing_mode_does_not_take_the_others_with_it(monkeypatch, wired):
    def boom(*a, **k):
        raise RuntimeError("faiss is down")

    monkeypatch.setattr(rag, "_raw_search", boom)
    row = compare.run("q", "visual")
    assert not row["ok"] and "faiss is down" in row["error"]


def test_a_comparison_row_carries_the_pages_and_the_numbers(wired):
    row = compare.run("Ile kosztuje brama?", "visual", n_pages=2)
    assert row["ok"] and len(row["pages"]) == 2
    assert {"document", "page", "score", "found_by", "n_chunks"} <= set(row["pages"][0])
    assert row["stats"]["mode"] == "visual"


def test_agreement_groups_pages_by_the_modes_that_found_them(wired):
    rows = [compare.run("q", "visual", n_pages=2),
            compare.run("q", "visual", n_pages=1)]
    rows[1]["mode"] = "hybrid"
    agree = compare.agreement(rows)
    assert agree[0]["modes"] == ["hybrid", "visual"]      # both found page 1
    assert agree[-1]["modes"] == ["visual"]               # page 2 only in one
