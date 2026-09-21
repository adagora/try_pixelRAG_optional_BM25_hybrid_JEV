"""The Reader seam, and the answer pipeline running through it.

Every test here exercises rag.run_agent end to end — retrieval, page selection,
prompt assembly, streaming, citation resolution, cost, and the answer cache —
with no SDK imported, no API key, and no network. That was impossible before
providers.py existed, which is the point of it.
"""

from __future__ import annotations

import pytest

import answer_cache
import imagefit
import providers
import prompts
import queryembed
import corpus
import rag


@pytest.fixture
def wired(monkeypatch, index, fake_reader, tmp_path):
    """rag, pointed at the fixture index and a scripted reader.

    Returns a function that installs a reader and gives back a runner.
    """
    import chunkmeta

    monkeypatch.setattr(corpus, "LAYOUT", index)
    # The project's own .env sets PIXELRAG_HYBRID=1, which rag.py reads at
    # import — so without this a gate test that also sets TYPESAFE_API_KEY
    # resolves to `jev+hybrid` and BM25 answers from the REAL text sidecar,
    # nominating an article_id the one-article fixture does not have. It
    # depended on whether the question's words happened to be in the corpus.
    # test_modes.py's fixture has always pinned this; this one had not.
    monkeypatch.setattr(rag, "HYBRID", False)
    monkeypatch.setattr(rag, "_scale_of",
                        lambda a, t, c: chunkmeta.scale_of(a, t, c, index))
    monkeypatch.setattr(corpus, "doc_title", lambda aid: f"doc{aid}")
    monkeypatch.setattr(corpus, "page_size", lambda a, t: (1654, 2339))
    monkeypatch.setattr(corpus, "articles", lambda: [{"title": "doc0", "url": ""}])
    # A page image on disk for every tile the fake search returns.
    from PIL import Image

    for tile in (0, 1):
        p = index.page_image(0, tile)
        Image.new("RGB", (200, 280), (240, 240, 240)).save(p)
    # The faiss HTTP call is the seam, not rag.search: `search` is now one of
    # four mode-bound searchers, and patching it over would leave the mode
    # machinery every answer goes through unexercised.
    monkeypatch.setattr(rag, "_raw_search", lambda q, n, timeout=120: [
        {"article_id": 0, "tile_index": 0, "chunk_index": 1, "score": 0.62},
        {"article_id": 0, "tile_index": 0, "chunk_index": 2, "score": 0.55},
        {"article_id": 0, "tile_index": 1, "chunk_index": 1, "score": 0.48},
    ])
    # Isolate the answer cache and the resized-page cache from the real ones.
    monkeypatch.setattr(imagefit, "CACHE_DIR", tmp_path / "fit")
    monkeypatch.setattr(
        answer_cache, "_default",
        answer_cache.AnswerCache(tmp_path / "v.npz", tmp_path / "m.json"))
    monkeypatch.setattr(queryembed, "embed_query", lambda q, instruction=None: [1.0, 0.0])

    def install(**kwargs):
        reader = fake_reader(**kwargs)
        monkeypatch.setattr(providers, "reader_for",
                            lambda provider=None, api_key=None: reader)
        return reader

    return install


# -- the pipeline, without a model ------------------------------------------

def test_oneshot_answers_through_a_fake_reader(wired):
    reader = wired(answer="Cena to 3956 zł netto.")
    result = rag.run_agent("Ile kosztuje brama?")
    assert result["answer"] == "Cena to 3956 zł netto."
    assert result["provider"] == "fake" and result["model"] == "fake-1"
    assert reader.calls == 1


def test_the_reader_is_given_the_retrieved_pages(wired):
    reader = wired()
    rag.run_agent("Ile kosztuje brama?")
    assert len(reader.pages) == 2                # two distinct pages, deduped
    assert all(p["image"] and p["mime"] for p in reader.pages)


def test_the_reader_is_given_the_one_shot_system_prompt(wired):
    reader = wired()
    rag.run_agent("q")
    assert reader.system == prompts.ONESHOT_SYSTEM


def test_each_page_is_labelled_with_the_number_the_reader_must_cite(wired):
    """The header is the only place the page number is stated, and the citation
    block is parsed against it."""
    reader = wired()
    rag.run_agent("q")
    assert all("page=" in h for h in reader.headers)
    assert "page=1" in reader.headers[0]


def test_the_preamble_lists_the_page_inventory(wired):
    """A reader that can see it was given three pages from one catalogue knows
    to check whether they are different product lines."""
    reader = wired()
    rag.run_agent("Ile kosztuje brama?")
    assert "Ile kosztuje brama?" in reader.preamble
    assert "strona 1" in reader.preamble and "strona 2" in reader.preamble


def test_pages_are_sized_by_the_readers_own_policy(wired):
    """The provider name no longer reaches imagefit; the policy does."""
    reader = wired(policy=imagefit.ANTHROPIC_POLICY)
    rag.run_agent("q")
    from PIL import Image
    import io

    with Image.open(io.BytesIO(reader.pages[0]["image"])) as im:
        assert max(im.size) <= imagefit.ANTHROPIC_LONG_EDGE


def test_the_answer_streams_before_it_is_finished(wired):
    wired(answer="Cena to 3956 zł.")
    seen = []
    rag.run_agent("q", on_event=lambda ev: seen.append(ev))
    deltas = [e["text"] for e in seen if e["type"] == "answer_delta"]
    assert len(deltas) == 2 and "".join(deltas) == "Cena to 3956 zł."


def test_the_trace_reports_search_then_pages(wired):
    wired()
    seen = []
    rag.run_agent("q", on_event=lambda ev: seen.append(ev))
    kinds = [e["type"] for e in seen]
    assert kinds[0] == "search"
    assert kinds.count("tile") == 2
    assert kinds[-1] == "answer_delta"


def test_the_citation_block_is_stripped_from_the_answer(wired):
    wired(answer='Cena to 3956 zł.\n---CYTATY---\n{"page":1,"quote":"Cennik"}')
    result = rag.run_agent("q")
    assert result["answer"] == "Cena to 3956 zł."
    assert [c["quote"] for c in result["citations"]] == ["Cennik"]


def test_cost_comes_from_the_reader(wired):
    wired(usage=providers.Usage(input=1000, output=200), cost=0.0125)
    u = rag.run_agent("q")["usage"]
    assert u["input"] == 1000 and u["output"] == 200 and u["cost_usd"] == 0.0125  # ubs:ignore — fixture literal, passed through unchanged


def test_an_unpriced_reader_reports_no_cost(wired):
    wired(cost=None)
    assert rag.run_agent("q")["usage"]["cost_usd"] is None


def test_no_pages_gives_the_documented_non_answer(wired, monkeypatch):
    """Never invent an answer when retrieval found nothing."""
    reader = wired()
    monkeypatch.setattr(rag, "_raw_search", lambda q, n, timeout=120: [])
    result = rag.run_agent("q")
    assert result["answer"] == prompts.NO_PAGES_PL
    assert reader.calls == 0                     # and never pay for the read


def test_the_answer_says_which_mode_retrieved_it_and_where_the_time_went(wired):
    """Retrieval is the part a mode changes and the reader is the part that
    bills; an answer that reported one number for both could not show either."""
    wired()
    result = rag.run_agent("q", retrieval="visual")
    assert result["retrieval"] == "visual"
    t = result["timings"]
    assert t["retrieval_ms"] >= 0
    assert t["ttft_ms"] is not None                # this reader streams
    assert t["total_ms"] >= t["retrieval_ms"]
    assert t["reader_ms"] is not None


def test_the_gate_judges_the_attached_pages_and_can_refuse(wired, monkeypatch):
    """The reader is the expensive call; this is what stands in front of it."""
    import jev

    reader = wired()
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(rag, "_page_texts", lambda pages: ["tekst"] * len(pages))
    seen = {}

    def judged(q, texts, report=None):
        seen["n"] = len(texts)
        return jev.Gate(answerable=0.02, scope=0.91)

    monkeypatch.setattr(jev, "gate", judged)
    monkeypatch.setattr(rag, "JEV_REFUSE", 0.3)

    result = rag.run_agent("Ile kosztuje samochód?")
    # In scope (0.91) but not in these pages (0.02): the corpus is the right
    # one and the answer is not in it, which is its own sentence.
    assert result["answer"] == prompts.REFUSAL_PL["not-in-corpus"]
    assert result["answerable"] == 0.02 and result["scope"] == 0.91  # ubs:ignore — stub literals, nothing computes on them
    assert result["refused"] == "not-in-corpus"
    assert reader.calls == 0                 # never paid for the read
    assert seen["n"] == 2                    # judged the pages, not the pool


def test_out_of_scope_refuses_on_scope_alone_and_says_something_else(wired, monkeypatch):
    """Answerability cannot tell "we don't stock that" from "wrong shop".

    Measured over this index, a question the corpus is about but does not
    answer scores 0.04 answerable / 0.64 scope; a question from another domain
    scores 0.01 / 0.05. Routing on answerability alone collapses those into one
    event and shows the user advice meant for the other one.
    """
    import jev

    reader = wired()
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(rag, "_page_texts", lambda pages: ["tekst"] * len(pages))
    monkeypatch.setattr(jev, "gate",
                        lambda q, t, report=None: jev.Gate(answerable=0.01, scope=0.05))
    monkeypatch.setattr(rag, "JEV_REFUSE", 0.3)

    result = rag.run_agent("Jak wymienić olej w silniku?")
    assert result["refused"] == "out-of-scope"
    assert result["answer"] == prompts.REFUSAL_PL["out-of-scope"]
    assert result["answer"] != prompts.REFUSAL_PL["not-in-corpus"]
    assert reader.calls == 0


def test_the_gate_events_carry_what_the_ui_draws(wired, monkeypatch):
    """`scope` and `verdict` are read by scripts/static/index.html to decide
    which refusal sentence to show. They are a wire contract, not a detail:
    drop either and the browser silently falls back to the old message, which
    is the wrong one for exactly the case scope was added to catch."""
    import jev

    wired()
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(rag, "_page_texts", lambda pages: ["tekst"] * len(pages))
    monkeypatch.setattr(jev, "gate",
                        lambda q, t, report=None: jev.Gate(answerable=0.02, scope=0.05))
    monkeypatch.setattr(rag, "JEV_REFUSE", 0.3)

    seen = []
    rag.run_agent("q", on_event=seen.append)
    gate = next(e for e in seen if e["type"] == "answerable")
    assert gate["value"] == 0.02 and gate["scope"] == 0.05  # ubs:ignore — stub literals, nothing computes on them
    refused = next(e for e in seen if e["type"] == "refused")
    assert refused["verdict"] == "out-of-scope"
    assert refused["scope"] == 0.05 and refused["answerable"] == 0.02  # ubs:ignore — stub literals, nothing computes on them


def test_a_missing_scope_judgment_still_gates_on_answerability(wired, monkeypatch):
    """Scope was added after the gate shipped; losing it must not open the gate."""
    import jev

    reader = wired()
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(rag, "_page_texts", lambda pages: ["tekst"] * len(pages))
    monkeypatch.setattr(jev, "gate",
                        lambda q, t, report=None: jev.Gate(answerable=0.02, scope=None))
    monkeypatch.setattr(rag, "JEV_REFUSE", 0.3)

    assert rag.run_agent("q")["refused"] == "not-in-corpus"
    assert reader.calls == 0


def test_a_refusal_is_never_cached(wired, monkeypatch):
    """A gated refusal has tile events and a non-empty answer, so the old
    `worth caching` test would have pinned it for every rephrasing."""
    import jev

    wired()
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(rag, "_page_texts", lambda pages: ["tekst"] * len(pages))
    monkeypatch.setattr(jev, "gate",
                        lambda q, t, report=None: jev.Gate(answerable=0.01, scope=0.9))
    monkeypatch.setattr(rag, "JEV_REFUSE", 0.3)
    rag.run_agent("q")

    monkeypatch.setattr(rag, "JEV_REFUSE", 0.0)
    reader = wired(answer="Teraz działa")
    assert rag.run_agent("q")["answer"] == "Teraz działa"
    assert reader.calls == 1


def test_a_failing_gate_never_costs_an_answer(wired, monkeypatch):
    import jev

    def boom(*a, **k):
        raise RuntimeError("typesafe down")

    reader = wired(answer="Odpowiedź.")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev, "answerable", boom)
    monkeypatch.setattr(rag, "JEV_REFUSE", 0.9)
    result = rag.run_agent("q")
    assert result["answer"] == "Odpowiedź." and result["answerable"] is None
    assert reader.calls == 1


# -- the answer cache, end to end -------------------------------------------

def test_a_repeated_question_skips_the_reader(wired):
    reader = wired(answer="Cena to 3956 zł.")
    first = rag.run_agent("Ile kosztuje brama?")
    second = rag.run_agent("Ile kosztuje brama?")
    assert reader.calls == 1                     # the second never reached it
    assert second["answer"] == first["answer"]
    assert second["cached"] is True
    assert second["usage"]["input"] == 0 and second["usage"]["cost_usd"] == 0.0  # ubs:ignore — cache hit assigns literal 0.0


def test_a_cache_hit_still_draws_the_pages(wired):
    """Without the replay the answer would cite pages the user never saw
    appear, which reads as a bug even though the answer is right."""
    wired()
    rag.run_agent("q")
    seen = []
    rag.run_agent("q", on_event=lambda ev: seen.append(ev))
    assert [e["type"] for e in seen].count("tile") == 2


def test_a_cache_hit_does_not_report_the_time_the_original_answer_took(wired):
    """Keeping the stored timings would make the cache look exactly as slow as
    the work it exists to skip."""
    wired()
    rag.run_agent("q")
    result = rag.run_agent("q")
    assert result["cached"] and result["timings"]["retrieval_ms"] == 0.0
    assert result["timings"]["total_ms"] < 1000


def test_a_different_model_misses_the_cache(wired):
    """Model is part of the exact-match namespace: a stale hit from another
    model is a wrong answer, not a saving."""
    first = wired(answer="A")
    rag.run_agent("q")
    second = wired(answer="B", model="fake-2")
    assert rag.run_agent("q")["answer"] == "B"
    assert second.calls == 1


def test_a_non_answer_is_never_cached(wired, monkeypatch):
    """'no pages matched' is usually a transient search-service problem, and
    pinning it would keep answering that way."""
    monkeypatch.setattr(rag, "_raw_search", lambda q, n, timeout=120: [])
    wired()
    rag.run_agent("q")
    reader = wired(answer="Teraz działa")
    monkeypatch.setattr(rag, "_raw_search", lambda q, n, timeout=120: [
        {"article_id": 0, "tile_index": 0, "chunk_index": 1, "score": 0.6}])
    assert rag.run_agent("q")["answer"] == "Teraz działa"
    assert reader.calls == 1


def test_a_dead_encoder_answers_the_slow_way_rather_than_failing(wired, monkeypatch):
    def down(*a, **k):
        raise ConnectionError("encoder sidecar is gone")

    reader = wired(answer="Nadal działa")
    monkeypatch.setattr(queryembed, "embed_query", down)
    assert rag.run_agent("q")["answer"] == "Nadal działa"
    assert reader.calls == 1


# -- registry ---------------------------------------------------------------

def test_reader_for_maps_a_name_to_a_class():
    assert isinstance(providers.reader_for("gemini"), providers.GeminiReader)
    assert isinstance(providers.reader_for("anthropic"), providers.AnthropicReader)


def test_reader_for_rejects_an_unknown_provider():
    with pytest.raises(RuntimeError, match="Unknown provider"):
        providers.reader_for("openai")


def test_a_provider_can_be_registered(monkeypatch):
    """Adding a backend is writing one class, not editing four call sites."""
    monkeypatch.setitem(providers._REGISTRY, "mine", providers.GeminiReader)
    assert providers.reader_for("mine").name == "gemini"


def test_gemini_usage_from_a_stream_is_cumulative_not_summed():
    """Streaming reports running totals; summing them would multiply the bill."""
    class Chunk:
        def __init__(self, p, c, t):
            self.usage_metadata = type("U", (), {
                "prompt_token_count": p, "candidates_token_count": c,
                "thoughts_token_count": t})()

    u = providers.Usage()
    for c in (Chunk(5000, 10, 0), Chunk(5000, 40, 0), Chunk(5000, 90, 0)):
        providers._accumulate_gemini(u, c, cumulative=True)
    assert u.input == 5000 and u.output == 90


def test_gemini_usage_from_an_agent_loop_adds_up():
    """Discrete calls, so those really do sum."""
    class Resp:
        def __init__(self, p, c):
            self.usage_metadata = type("U", (), {
                "prompt_token_count": p, "candidates_token_count": c,
                "thoughts_token_count": 0})()

    u = providers.Usage()
    for r in (Resp(1000, 20), Resp(1500, 30)):
        providers._accumulate_gemini(u, r, cumulative=False)
    assert u.input == 2500 and u.output == 50


def test_a_tool_exception_becomes_a_result_the_model_can_read():
    """A traceback is more useful to the model than a dead turn, and it must not
    abort the browse loop."""
    def boom(name, args):
        raise ValueError("no such region")

    result, ev = providers._safe_dispatch(boom, "pixelrag_tile", {})
    assert result["ok"] is False and "ValueError" in result["message"]
    assert ev is None


def test_usage_reports_the_cache_split():
    u = providers.Usage(input=100, output=50, cache_read=900, cache_write=10)
    d = u.as_dict(cost_usd=1.5)
    assert d == {"input": 100, "output": 50, "thoughts": 0,
                 "cache_read": 900, "cache_write": 10, "cost_usd": 1.5}


# -- agent mode -------------------------------------------------------------

def test_agent_mode_uses_the_browse_prompt(wired):
    reader = wired(answer="Znalazłem.", steps=3)
    result = rag.run_agent("q", mode="agent")
    assert result["answer"] == "Znalazłem." and result["steps"] == 3
    assert reader.system == prompts.SYSTEM


def test_agent_mode_is_never_cached(wired):
    """Its value is the browse trace, and replaying a canned one would be a lie."""
    reader = wired(answer="Znalazłem.")
    rag.run_agent("q", mode="agent")
    rag.run_agent("q", mode="agent")
    assert reader.calls == 2
