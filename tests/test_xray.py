"""The exhaustive sweep, and the claims it is only allowed to make when true.

Faked at `jev.evaluate`, the single HTTP call, so sharding, facet folding,
degradation and the token accounting are all real code. No network, no key.
"""

from __future__ import annotations

import pytest

import jev
import xray


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("PIXELRAG_JEV", "1")


def corpus(n=6, text="tekst strony o bramach"):
    return [(f"a0:s{i + 1}", f"{text} {i}") for i in range(n)]


def scripted(monkeypatch, *, scores=None, facets=None, fail=None):
    """Answer every question the sweep asks, recording what it asked."""
    asked = []

    def evaluate(state, questions, report=None, kind=""):
        asked.append(questions)
        if fail is not None:
            raise fail
        if report is not None:
            report.calls.append(jev.Call(kind, 12.0, 1000, 80))
        out = {}
        for key in questions:
            if questions[key]["type"] == "score":
                out[key] = {"score": (scores or {}).get(key, 1.5)}
            elif questions[key]["type"] == "noul":
                out[key] = {"noul": (facets or {}).get(key, 0.5)}
            else:
                out[key] = {"choice": (facets or {}).get(key, "spec"),
                            "confidence": 0.9}
        return out

    monkeypatch.setattr(jev, "evaluate", evaluate)
    return asked


def test_every_page_is_scored_and_the_report_says_so(monkeypatch):
    """`exhaustive` is the only claim this module makes; it has to be checked."""
    scripted(monkeypatch)
    result = xray.xray("kolory", corpus(6))
    assert result.pages == 6 and result.judged == 6 and result.blind == 0
    assert result.exhaustive and result.unjudged == 0
    assert set(result.scores) == {f"a0:s{i}" for i in range(1, 7)}


def test_scores_are_the_rubric_position_not_the_raw_level(monkeypatch):
    """0..3 in, 0..1 out — the same normalisation jev.py uses, so a score here
    and a score there are the same number."""
    scripted(monkeypatch, scores={"a0:s1": 3.0, "a0:s2": 0.0})
    result = xray.xray("q", corpus(2))
    assert result.scores["a0:s1"] == 1.0
    assert result.scores["a0:s2"] == 0.0


def test_ranking_is_best_first_and_ties_keep_corpus_order(monkeypatch):
    scripted(monkeypatch, scores={"a0:s1": 1.0, "a0:s2": 3.0, "a0:s3": 1.0})
    assert [k for k, _ in xray.xray("q", corpus(3)).ranked()] == [
        "a0:s2", "a0:s1", "a0:s3"]


def test_a_page_with_no_text_is_blind_not_zero(monkeypatch):
    """Scoring a page on text it does not have would invent evidence; the page
    is left out and counted, which is what `visual` mode is for."""
    scripted(monkeypatch)
    result = xray.xray("q", [("a0:s1", "tekst"), ("a0:s2", "   ")])
    assert result.blind == 1 and result.judged == 1
    assert "a0:s2" not in result.scores
    assert result.exhaustive                 # every page that HAD text was judged
    assert any("no text" in note for note in result.report.fallbacks)


def test_a_corpus_with_no_text_at_all_is_refused_not_ranked(monkeypatch):
    scripted(monkeypatch)
    result = xray.xray("q", [("a0:s1", ""), ("a0:s2", " ")])
    assert result.scores == {} and result.judged == 0
    assert any("X-ray skipped" in n for n in result.report.fallbacks)


# -- sharding ---------------------------------------------------------------

def test_a_corpus_that_fits_is_one_request(monkeypatch):
    asked = scripted(monkeypatch)
    result = xray.xray("q", corpus(6))
    assert result.shards == 1 and len(asked) == 1


def test_a_corpus_that_does_not_fit_is_sharded_and_every_page_survives(monkeypatch):
    """The sweep's whole claim is that no page is left out; sharding is the one
    place a page could silently be."""
    monkeypatch.setattr(xray, "BUDGET_TOKENS", 1200)
    asked = scripted(monkeypatch)
    result = xray.xray("q", corpus(20, text="x" * 300))
    assert result.shards > 1 and len(asked) == result.shards
    assert result.judged == 20 and result.exhaustive


def test_query_facets_are_asked_once_however_many_shards(monkeypatch):
    """They are about the question text, so a second shard cannot change them
    and paying for them again would be waste with a rounding error attached."""
    monkeypatch.setattr(xray, "BUDGET_TOKENS", 1200)
    asked = scripted(monkeypatch)
    xray.xray("q", corpus(20, text="x" * 300))
    assert sum(xray.KIND in questions for questions in asked) == 1
    # evidence facets ride on every shard
    assert all(xray.ANSWERABLE in questions for questions in asked)


def test_evidence_facets_fold_by_max_across_shards(monkeypatch):
    """If any shard holds the answer, the corpus holds the answer. An average
    would let seven shards of silence outvote the one that knows."""
    monkeypatch.setattr(xray, "BUDGET_TOKENS", 1200)
    seen = {"n": 0}

    def evaluate(state, questions, report=None, kind=""):
        seen["n"] += 1
        noul = 0.9 if seen["n"] == 1 else 0.05
        return {k: ({"score": 1.5} if q["type"] == "score" else
                    {"noul": noul} if q["type"] == "noul" else
                    {"choice": "spec", "confidence": 0.9})
                for k, q in questions.items()}

    monkeypatch.setattr(jev, "evaluate", evaluate)
    result = xray.xray("q", corpus(20, text="x" * 300))
    assert result.shards > 1
    assert result.facets.answerable == 0.9


def test_a_corpus_too_big_to_sweep_is_refused_with_a_reason(monkeypatch):
    """"Exhaustive" is the only thing this mode sells. A partial sweep is not a
    cheaper version of it, it is a different product, so it is not offered."""
    monkeypatch.setattr(xray, "BUDGET_TOKENS", 200)
    monkeypatch.setattr(xray, "MAX_SHARDS", 2)
    asked = scripted(monkeypatch)
    result = xray.xray("q", corpus(40, text="x" * 400))
    assert result.scores == {} and asked == []
    assert any("use a retrieval mode" in n for n in result.report.fallbacks)


def test_one_failed_shard_does_not_discard_the_others(monkeypatch):
    """jev.search used to lose a whole rerank to a single unreadable crop. The
    sweep pays per shard, so dropping the paid-for ones costs twice."""
    monkeypatch.setattr(xray, "BUDGET_TOKENS", 1200)
    calls = {"n": 0}

    def evaluate(state, questions, report=None, kind=""):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("shard died")
        return {k: ({"score": 2.0} if q["type"] == "score" else
                    {"noul": 0.5} if q["type"] == "noul" else
                    {"choice": "spec", "confidence": 0.9})
                for k, q in questions.items()}

    monkeypatch.setattr(jev, "evaluate", evaluate)
    result = xray.xray("q", corpus(20, text="x" * 300))
    assert result.failed_shards == 1
    assert result.judged > 0                       # the survivors still rank
    assert not result.exhaustive                   # and the claim is withdrawn
    assert result.unjudged > 0
    assert any("shard" in n for n in result.report.fallbacks)


# -- facets -----------------------------------------------------------------

def test_a_choice_outside_its_criteria_is_dropped_rather_than_routed_on(monkeypatch):
    """`granularity` picks the reader's page budget. An unknown label would put
    a number nobody chose into PAGES_FOR, or a KeyError into an answer."""
    scripted(monkeypatch, facets={xray.GRANULARITY: "enormous"})
    result = xray.xray("q", corpus(3))
    assert result.facets.granularity is None
    assert result.facets.suggested_pages is None


def test_granularity_sets_the_page_budget(monkeypatch):
    scripted(monkeypatch, facets={xray.GRANULARITY: "survey"})
    assert xray.xray("q", corpus(3)).facets.suggested_pages == xray.PAGES_FOR["survey"]


def test_an_unjudgeable_facet_is_none_not_zero(monkeypatch):
    """A facet that failed and a facet that answered "no" must not look the
    same: one of them is a reason to distrust the sweep."""
    def evaluate(state, questions, report=None, kind=""):
        return {k: ({"score": 1.5} if q["type"] == "score" else {})
                for k, q in questions.items()}

    monkeypatch.setattr(jev, "evaluate", evaluate)
    facets = xray.xray("q", corpus(3)).facets
    assert facets.answerable is None and facets.scope is None
    assert facets.kind is None and facets.injection is None


def test_the_bill_is_reported_per_call(monkeypatch):
    scripted(monkeypatch)
    result = xray.xray("q", corpus(4))
    assert result.report.input_tokens == 1000
    assert result.report.output_tokens == 80
    assert [c.kind for c in result.report.calls] == ["xray"]
    assert result.as_dict()["jev"]["cost_usd"] == pytest.approx(1000 * 0.042 / 1e6)


def test_the_whole_sweep_failing_is_survivable(monkeypatch):
    scripted(monkeypatch, fail=ValueError("typesafe down"))
    result = xray.xray("q", corpus(4))
    assert result.scores == {} and not result.exhaustive
    assert result.failed_shards == 1


# -- planning ---------------------------------------------------------------

def test_the_plan_keeps_corpus_order_so_a_failure_is_a_page_range():
    shards = xray.plan(["x" * 1000] * 10, budget=1000)
    assert [i for shard in shards for i in shard] == list(range(10))
    assert len(shards) > 1


def test_a_single_oversized_page_still_gets_a_shard():
    """It is truncated to PAGE_CHARS long before here; dropping it would be a
    page missing from a sweep that calls itself exhaustive."""
    assert xray.plan(["x" * 99999], budget=10) == [[0]]
