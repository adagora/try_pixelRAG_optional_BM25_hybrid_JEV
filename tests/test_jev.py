import pytest
import requests

import jev


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("PIXELRAG_JEV", "1")


def hit(i, score=0.8):
    return {"article_id": 0, "tile_index": i, "chunk_index": 1, "score": score}


def texts(text):
    """A `texts_fn`: the same string for every candidate in the pool."""
    return lambda hits: [text] * len(hits)


def test_disabled_makes_no_jev_call(monkeypatch):
    monkeypatch.setenv("PIXELRAG_JEV", "0")
    assert jev.search("q", 1, lambda q, n: [hit(0)], None) == [hit(0)]


def test_expansion_selects_facets_and_preserves_original(monkeypatch):
    monkeypatch.setattr(jev, "evaluate", lambda *args: {
        str(i): {"noul": 0.9 if i == 0 else 0.1} for i in range(len(jev.FACETS))})
    variants = jev.expand("Ile kosztuje brama?")
    assert variants[:2] == ["Ile kosztuje brama?", "brama"]
    assert variants[2] == "brama " + jev.FACETS[0]


def test_rerank_promotes_expanded_chunk_before_cutoff(monkeypatch):
    monkeypatch.setattr(jev, "expand", lambda q, report=None: [q, "expanded"])
    seen = []
    def evaluate(state, questions, report=None, kind=""):
        seen.append(state)
        return {"0": {"score": 0.3}, "1": {"score": 2.9}}
    monkeypatch.setattr(jev, "evaluate", evaluate)
    result = jev.search("q", 1,
                        lambda q, n: [hit(0)] if q == "q" else [hit(0), hit(1)],
                        lambda hs: [f"chunk {h['tile_index']}" for h in hs])
    assert result[0]["tile_index"] == 1
    assert result[0]["retrieval_score"] == 0.8
    assert seen[0]["query"] == "q"
    assert len(seen[0]["chunks"]) == 2


@pytest.mark.parametrize("bad", [float("nan"), -1, 4, "2", True])
def test_invalid_scores_fall_back_atomically(monkeypatch, bad):
    monkeypatch.setattr(jev, "expand", lambda q, report=None: [q])
    monkeypatch.setattr(jev, "evaluate", lambda *args: {"0": {"score": bad}})
    assert jev.search("q", 1, lambda q, n: [hit(0)], texts("text")) == [hit(0)]


def test_api_failure_and_missing_text_preserve_retrieval(monkeypatch):
    def fail(*args):
        raise requests.Timeout()
    monkeypatch.setattr(jev, "evaluate", fail)
    assert jev.search("q", 1, lambda q, n: [hit(0)], texts("text")) == [hit(0)]
    assert jev.search("q", 1, lambda q, n: [hit(0)], texts("")) == [hit(0)]


def test_http_contract(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"answers": {"a": {"noul": 0.9}},
                    "usage": {"input_tokens": 312, "output_tokens": 48}}
    def post(url, **kwargs):
        assert url == jev.ENDPOINT
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["json"]["model"] == "jev-latest"
        assert kwargs["timeout"] == 20
        return Response()
    monkeypatch.setattr(jev.requests, "post", post)
    assert jev.evaluate("state", {"a": {"type": "noul"}})["a"]["noul"] == 0.9


def test_chunk_text_uses_crop_geometry(monkeypatch, tmp_path):
    import fitz
    import rag
    import chunkmeta
    path = tmp_path / "source.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=200, height=200)
        page.insert_text((20, 30), "inside")
        page.insert_text((20, 150), "outside")
        pdf.save(path)
    monkeypatch.setattr(rag, "_source_pdf", lambda aid: path)
    monkeypatch.setattr(rag, "page_size", lambda aid, ti: (400, 400))
    monkeypatch.setattr(chunkmeta, "get", lambda *args:
                        chunkmeta.Chunk(0, 1, "region", 0, 0, 400, 200))
    text = rag._chunk_text(hit(0))
    assert "inside" in text
    assert "outside" not in text


# -- what the stage reports having done -------------------------------------

def test_the_report_counts_the_funnel_and_the_tokens(monkeypatch):
    """The comparison table is only worth reading if these numbers are real."""
    monkeypatch.setattr(jev, "expand", lambda q, report=None: [q, "expanded"])

    def evaluate(state, questions, report=None, kind=""):
        report.calls.append(jev.Call(kind, 40.0, 900, 60))
        return {k: {"score": 2.0} for k in questions}

    monkeypatch.setattr(jev, "evaluate", evaluate)
    report = jev.Report()
    jev.search("q", 2, lambda q, n: [hit(0)] if q == "q" else [hit(0), hit(1)],
               texts("tekst"), report)
    assert report.candidates == 2 and report.scored == 2
    assert report.reranked and not report.fallbacks
    assert report.input_tokens == 900 and report.output_tokens == 60
    assert set(report.stages) == {"expand", "candidates", "text", "rerank"}


def test_a_degraded_run_says_so_rather_than_looking_like_a_clean_one(monkeypatch):
    """Missing text preserves the original retrieval — which is indistinguishable
    from a successful rerank in every field except this one."""
    monkeypatch.setattr(jev, "expand", lambda q, report=None: [q])
    report = jev.Report()
    out = jev.search("q", 1, lambda q, n: [hit(0)], texts(""), report)
    assert out == [hit(0)]
    assert not report.reranked and report.fallbacks
    assert "PDF text" in report.fallbacks[0]


def test_one_unreadable_crop_does_not_abort_the_whole_rerank(monkeypatch):
    """Measured cause of the stage silently not running on 1 question in 6: a
    single blind candidate out of forty used to discard the other thirty-nine."""
    monkeypatch.setattr(jev, "expand", lambda q, report=None: [q])
    asked = {}

    def evaluate(state, questions, report=None, kind=""):
        asked.update(state["chunks"])
        return {k: {"score": 3.0} for k in questions}

    monkeypatch.setattr(jev, "evaluate", evaluate)
    report = jev.Report()
    out = jev.search("q", 2, lambda q, n: [hit(0), hit(1), hit(2)],
                     # hit 1 is an image-only crop with no text layer
                     lambda hits: ["tekst", "   ", "tekst"], report)
    assert report.reranked and report.scored == 2 and report.blind == 1
    assert sorted(asked) == ["0", "2"]          # keyed by position in the pool
    assert [h["tile_index"] for h in out] == [0, 2]
    assert report.fallbacks and "no PDF text" in report.fallbacks[0]


def test_the_existence_check_rides_along_with_the_rerank(monkeypatch):
    """One extra Noul on a state that was already being sent."""
    monkeypatch.setattr(jev, "expand", lambda q, report=None: [q])
    asked = {}

    def evaluate(state, questions, report=None, kind=""):
        asked.update(questions)
        return {**{k: {"score": 1.5} for k in questions if k != jev.ANSWERABLE},
                jev.ANSWERABLE: {"noul": 0.04}}

    monkeypatch.setattr(jev, "evaluate", evaluate)
    report = jev.Report()
    out = jev.search("q", 2, lambda q, n: [hit(0), hit(1)], texts("tekst"), report)
    assert len(out) == 2                      # the ranking is unaffected by it
    assert asked[jev.ANSWERABLE]["type"] == "noul"
    assert report.answerable == 0.04
    assert report.best_score == 0.5


def test_a_broken_existence_check_does_not_lose_the_ranking(monkeypatch):
    monkeypatch.setattr(jev, "expand", lambda q, report=None: [q])
    monkeypatch.setattr(jev, "evaluate", lambda *a, **k: {"0": {"score": 3.0}})
    report = jev.Report()
    out = jev.search("q", 1, lambda q, n: [hit(0)], texts("tekst"), report)
    assert out[0]["score"] == 1.0 and report.reranked
    assert report.answerable is None


def test_a_failed_call_is_still_billed_and_still_reported(monkeypatch):
    """A malformed response cost tokens and took time; hiding that would make a
    broken mode look like the cheap one."""
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"answers": ["not", "an", "object"],
                    "usage": {"input_tokens": 120, "output_tokens": 8}}

    monkeypatch.setattr(jev.requests, "post", lambda url, **kw: Response())
    report = jev.Report()
    with pytest.raises(ValueError):
        jev.evaluate("state", {"a": {"type": "noul"}}, report, "expand")
    assert report.input_tokens == 120 and report.calls[0].kind == "expand"


def test_cost_uses_the_published_rate_and_output_is_free(monkeypatch):
    """docs.typesafe.ai/models.md: $0.042 per 1M in, output tokens free. The
    re-ranking cookbook's own total is input-token arithmetic exactly, so `free`
    is quoted, not assumed."""
    monkeypatch.delenv("TYPESAFE_PRICE_IN", raising=False)
    monkeypatch.delenv("TYPESAFE_PRICE_OUT", raising=False)
    assert jev.price(1_000_000, 1_000_000) == pytest.approx(0.042)
    # A full rerank of this index's pool, for scale.
    assert jev.price(7_760, 447) == pytest.approx(0.000326, abs=1e-6)


def test_an_invoice_rate_overrides_the_list_price(monkeypatch):
    monkeypatch.setenv("TYPESAFE_PRICE_IN", "2")
    monkeypatch.setenv("TYPESAFE_PRICE_OUT", "10")
    assert jev.price(1_000_000, 500_000) == pytest.approx(7.0)
    monkeypatch.setenv("TYPESAFE_PRICE_IN", "free please")
    assert jev.price(1, 1) is None
