"""The web UI's routes — the whole user-facing surface, previously untested.

`app.py` is thin by design, and thin is not the same as trivial: every route
here carries at least one decision that a reader of the URL cannot see. Pages
are 1-based in the URL and 0-based on disk. A Polish file name has to leave in
two encodings or the response crashes. A mode that cannot run must come back
named, with a reason, rather than missing. Those are the things this file
pins.

No search service, no encoder, no reader, no network: `rag` and `compare` are
substituted at the boundary, which is exactly the boundary app.py was written
against.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as webapp
import corpus


@pytest.fixture
def client():
    return TestClient(webapp.app)


def _events(response) -> list[dict]:
    """Parse an SSE body into the payload dicts it carried."""
    return [json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ")]


# --------------------------------------------------------------------------
# The page itself
# --------------------------------------------------------------------------

def test_index_is_served_as_utf8(client):
    """Explicit charset, because a Windows locale mangles 'ąęł' otherwise —
    the comment in app.py says so, so the header is worth asserting."""
    r = client.get("/")
    assert r.status_code == 200
    assert "charset=utf-8" in r.headers["content-type"].lower()


def test_index_actually_carries_the_ui(client):
    body = client.get("/").text
    assert "porównaj tryby" in body        # the comparison checkbox
    assert "<title" in body.lower()


# --------------------------------------------------------------------------
# /api/modes — a blocked mode must be named, not hidden
# --------------------------------------------------------------------------

def test_modes_reports_default_and_page_budget(client, monkeypatch):
    monkeypatch.setattr(webapp.compare_modes, "modes",
                        lambda: [{"mode": "visual", "blocked": None}])
    monkeypatch.setattr(webapp.rag, "default_retrieval", lambda: "hybrid")
    monkeypatch.setattr(webapp.rag, "ONESHOT_PAGES", 7)

    body = client.get("/api/modes").json()
    assert body["default"] == "hybrid"
    assert body["pages"] == 7


def test_a_blocked_mode_is_listed_with_its_reason(client, monkeypatch):
    """'jev — TYPESAFE_API_KEY is not set' is a setup instruction; a mode that
    silently vanished is a bug report. app.py's own docstring, asserted."""
    monkeypatch.setattr(webapp.compare_modes, "modes", lambda: [
        {"mode": "visual", "blocked": None},
        {"mode": "jev", "blocked": "TYPESAFE_API_KEY is not set"},
    ])
    listed = client.get("/api/modes").json()["modes"]
    blocked = [m for m in listed if m["blocked"]]
    assert [m["mode"] for m in blocked] == ["jev"]
    assert "TYPESAFE_API_KEY" in blocked[0]["blocked"]


# --------------------------------------------------------------------------
# /api/docs
# --------------------------------------------------------------------------

def test_docs_lists_articles_with_their_page_counts(client, monkeypatch):
    # IndexLayout is a frozen dataclass, so the layout is replaced rather than
    # patched in place — which is also how rag.py is meant to be redirected.
    monkeypatch.setattr(corpus, "articles",
                        lambda: [{"title": "a"}, {"title": "b"}])
    monkeypatch.setattr(corpus, "LAYOUT",
                        SimpleNamespace(page_images=lambda i: ["p"] * (i + 1)))
    docs = client.get("/api/docs").json()["docs"]
    assert [d["article_id"] for d in docs] == [0, 1]
    assert [d["pages"] for d in docs] == [1, 2]


# --------------------------------------------------------------------------
# /api/page — the 1-based/0-based seam
# --------------------------------------------------------------------------

def test_page_url_is_one_based_against_zero_based_storage(client, monkeypatch,
                                                          tmp_path):
    """The single most repeatable bug in this codebase's shape. Page 1 in the
    URL must read tile 0 off disk."""
    asked = []

    def fake_path(aid, idx):
        asked.append((aid, idx))
        p = tmp_path / f"{aid}-{idx}.jpg"
        p.write_bytes(b"\xff\xd8\xff")
        return p

    monkeypatch.setattr(corpus, "page_path", fake_path)
    assert client.get("/api/page/0/1").status_code == 200
    assert asked == [(0, 0)]


def test_missing_page_is_404_not_a_crash(client, monkeypatch, tmp_path):
    monkeypatch.setattr(corpus, "page_path",
                        lambda a, i: tmp_path / "absent.jpg")
    r = client.get("/api/page/0/9999")
    assert r.status_code == 404
    assert r.json()["error"] == "no such page"


# --------------------------------------------------------------------------
# /api/snip
# --------------------------------------------------------------------------

def test_snip_rejects_malformed_rects_with_400(client, monkeypatch, tmp_path):
    src = tmp_path / "p.jpg"
    src.write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(corpus, "page_path", lambda a, i: src)
    r = client.get("/api/snip/0/1", params={"rects": "{not json"})
    assert r.status_code == 400
    assert r.json()["error"] == "bad rects json"


def test_snip_on_a_missing_page_is_404_before_any_rendering(client,
                                                            monkeypatch, tmp_path):
    monkeypatch.setattr(corpus, "page_path", lambda a, i: tmp_path / "gone.jpg")
    called = []
    monkeypatch.setattr(webapp.snip, "render",
                        lambda *a, **k: called.append(1) or b"")
    assert client.get("/api/snip/0/1").status_code == 404
    assert not called


def test_snip_passes_boxes_and_padding_through(client, monkeypatch, tmp_path):
    src = tmp_path / "p.jpg"
    src.write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(corpus, "page_path", lambda a, i: src)
    seen = {}

    def fake_render(path, boxes, pad):
        seen["boxes"], seen["pad"] = boxes, pad
        return b"\xff\xd8\xff"

    monkeypatch.setattr(webapp.snip, "render", fake_render)
    rects = json.dumps([{"left": 1, "top": 2, "width": 3, "height": 4}])
    r = client.get("/api/snip/0/1", params={"rects": rects, "pad": 9.5})
    assert r.status_code == 200
    assert seen["pad"] == 9.5
    assert len(seen["boxes"]) == 1


def test_snip_padding_is_bounded(client):
    """`pad` is a query parameter on a public local route; the ge/le bounds are
    the validation, so a value outside them must be rejected by FastAPI."""
    assert client.get("/api/snip/0/1", params={"pad": 999}).status_code == 422


# --------------------------------------------------------------------------
# /api/pdf — the header-encoding bug this route exists to have fixed
# --------------------------------------------------------------------------

def test_pdf_out_of_range_article_is_404(client, monkeypatch):
    monkeypatch.setattr(corpus, "articles", lambda: [{"title": "a", "url": "x"}])
    assert client.get("/api/pdf/5").status_code == 404
    assert client.get("/api/pdf/0").status_code == 404   # url 'x' does not exist


def test_a_polish_filename_leaves_in_both_encodings(client, monkeypatch, tmp_path):
    """Headers go out as latin-1, so 'garażowe.pdf' crashes the response unless
    it is percent-encoded per RFC 5987 with an ASCII fallback. Both halves must
    be present or one client or the other breaks."""
    src = tmp_path / "Cennik - Bramy garażowe.pdf"
    src.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(corpus, "articles",
                        lambda: [{"title": "c", "url": str(src)}])

    disp = client.get("/api/pdf/0").headers["content-disposition"]
    assert disp.startswith("inline;")           # not a download prompt
    assert "filename*=UTF-8''" in disp          # RFC 5987 half
    assert "gara" in disp                       # ASCII-stripped fallback half
    assert disp.isascii()                       # the crash this prevents


def test_pdf_is_served_inline_for_the_viewer_link(client, monkeypatch, tmp_path):
    src = tmp_path / "plain.pdf"
    src.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(corpus, "articles",
                        lambda: [{"title": "c", "url": str(src)}])
    r = client.get("/api/pdf/0")
    assert r.headers["content-type"] == "application/pdf"
    assert "attachment" not in r.headers["content-disposition"]


# --------------------------------------------------------------------------
# /api/tile — an upstream failure must not look like an empty tile
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, ok, content=b"", ctype="image/png"):
        self.ok, self.content = ok, content
        self.headers = {"content-type": ctype}


def test_tile_upstream_error_is_502(client, monkeypatch):
    monkeypatch.setattr(webapp.rag._HTTP, "get", lambda *a, **k: _Resp(False))
    r = client.get("/api/tile/0/1/2")
    assert r.status_code == 502


def test_tile_unreachable_search_api_is_503(client, monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(webapp.rag._HTTP, "get", boom)
    r = client.get("/api/tile/0/1/2")
    assert r.status_code == 503
    assert "refused" in r.json()["error"]


def test_tile_passes_the_upstream_content_type_through(client, monkeypatch):
    monkeypatch.setattr(webapp.rag._HTTP, "get",
                        lambda *a, **k: _Resp(True, b"JPEGDATA", "image/jpeg"))
    r = client.get("/api/tile/0/1/2")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == b"JPEGDATA"


# --------------------------------------------------------------------------
# /api/compare — SSE
# --------------------------------------------------------------------------

def test_compare_streams_a_row_per_mode_then_agreement(client, monkeypatch):
    monkeypatch.setattr(webapp.compare_modes, "compare",
                        lambda q, m, p: iter([{"mode": "visual"},
                                              {"mode": "hybrid"}]))
    monkeypatch.setattr(webapp.compare_modes, "agreement",
                        lambda rows: [{"page": 1, "modes": ["visual", "hybrid"]}])

    events = _events(client.post("/api/compare", json={"question": "q"}))
    assert [e["kind"] for e in events] == ["row", "row", "done"]
    assert events[-1]["payload"]["agreement"][0]["page"] == 1


def test_compare_reports_a_failure_as_an_event_not_a_dead_stream(client,
                                                                 monkeypatch):
    """The browser has already received 200 and headers by the time a mode
    raises, so the only way to say 'this broke' is in the stream."""
    def boom(*a, **k):
        raise RuntimeError("index gone")

    monkeypatch.setattr(webapp.compare_modes, "compare", boom)
    events = _events(client.post("/api/compare", json={"question": "q"}))
    assert events[-1]["kind"] == "error"
    assert "RuntimeError: index gone" in events[-1]["payload"]


def test_compare_forwards_the_requested_modes_and_page_budget(client, monkeypatch):
    seen = {}

    def fake(q, modes, pages):
        seen.update(q=q, modes=modes, pages=pages)
        return iter([])

    monkeypatch.setattr(webapp.compare_modes, "compare", fake)
    monkeypatch.setattr(webapp.compare_modes, "agreement", lambda rows: [])
    client.post("/api/compare",
                json={"question": "x", "modes": ["visual"], "pages": 2})
    assert seen == {"q": "x", "modes": ["visual"], "pages": 2}


# --------------------------------------------------------------------------
# /api/ask — SSE over a worker thread
# --------------------------------------------------------------------------

def test_ask_streams_events_then_the_result(client, monkeypatch):
    def fake_run_agent(question, on_event=None, api_key=None, retrieval=None):
        on_event({"type": "search"})
        on_event({"type": "answer_delta", "text": "hi"})
        return {"answer": "hi"}

    monkeypatch.setattr(webapp.rag, "run_agent", fake_run_agent)
    events = _events(client.post("/api/ask", json={"question": "q"}))
    assert [e["kind"] for e in events] == ["event", "event", "done"]
    assert events[-1]["payload"]["answer"] == "hi"


def test_ask_forwards_the_retrieval_mode_and_pasted_key(client, monkeypatch):
    seen = {}

    def fake(question, on_event=None, api_key=None, retrieval=None):
        seen.update(api_key=api_key, retrieval=retrieval)
        return {}

    monkeypatch.setattr(webapp.rag, "run_agent", fake)
    client.post("/api/ask", json={"question": "q", "api_key": "  sk-1  ",
                                  "retrieval": "jev-page"})
    assert seen == {"api_key": "sk-1", "retrieval": "jev-page"}   # trimmed


def test_a_blank_pasted_key_falls_back_to_the_environment(client, monkeypatch):
    """The UI always sends the field; whitespace must mean 'unset', not a key
    of spaces that providers.py would then try to authenticate with."""
    seen = {}
    monkeypatch.setattr(webapp.rag, "run_agent",
                        lambda q, on_event=None, api_key=None, retrieval=None:
                        seen.update(api_key=api_key) or {})
    client.post("/api/ask", json={"question": "q", "api_key": "   "})
    assert seen["api_key"] is None


def test_ask_reports_a_worker_exception_in_the_stream(client, monkeypatch):
    def boom(*a, **k):
        raise ValueError("no key")

    monkeypatch.setattr(webapp.rag, "run_agent", boom)
    events = _events(client.post("/api/ask", json={"question": "q"}))
    assert events[-1]["kind"] == "error"
    assert "ValueError: no key" in events[-1]["payload"]


def test_the_stream_terminates_even_when_the_worker_dies(client, monkeypatch):
    """The sentinel is pushed in a `finally`; without it the response hangs
    forever and the browser spinner never stops."""
    monkeypatch.setattr(webapp.rag, "run_agent",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    r = client.post("/api/ask", json={"question": "q"})
    assert r.status_code == 200
    assert _events(r)                       # completed rather than timing out
