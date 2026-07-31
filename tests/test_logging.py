"""Every silent fallback now says so.

These assert on observability, not behaviour: each fallback is correct and stays
correct: the bug was that a system running four levels degraded looked identical
to one running at full speed.
"""

from __future__ import annotations

import logging

import pytest

import answer_cache
import imagefit
import rag


@pytest.fixture(autouse=True)
def _at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="pixelrag")


def test_a_dead_encoder_sidecar_is_reported(monkeypatch, caplog):
    """Losing the sidecar costs ~30s on the next query and nothing said so."""
    import requests

    monkeypatch.setattr(rag, "_sidecar_ok", None)
    monkeypatch.setattr(rag._HTTP, "get",
                        lambda *a, **k: (_ for _ in ()).throw(
                            requests.RequestException("refused")))
    assert rag._sidecar_available() is False
    assert "encoder sidecar not reachable" in caplog.text


def test_falling_back_to_server_side_encoding_is_reported(monkeypatch, caplog):
    """The 7x latency regression that looks like a fast path from outside."""
    monkeypatch.setattr(rag, "LOCAL_ENCODE", True)
    monkeypatch.setattr(rag, "embed_query",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cuda")))
    monkeypatch.setattr(rag._HTTP, "post", _fake_search_response)
    rag.search("q")
    assert "server-side encoding" in caplog.text


def test_a_disabled_answer_cache_is_reported(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(rag, "embed_query",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(rag, "_oneshot_pages", lambda *a, **k: [])
    monkeypatch.setattr(rag.providers, "reader_for",
                        lambda provider=None, api_key=None: _NullReader())
    rag.run_agent("q")
    assert "answer cache disabled" in caplog.text


def test_a_missing_text_sidecar_under_hybrid_is_reported(monkeypatch, caplog):
    """PIXELRAG_HYBRID=1 with no sidecar silently answered visual-only, which
    is exactly the comparison the flag exists to make."""
    import lexical

    monkeypatch.setattr(rag, "HYBRID", True)
    monkeypatch.setattr(lexical, "TEXT_SIDECAR", lexical.Path("/nope/text.json"))
    assert rag._lexical_fn() is None
    assert "build_text_index" in caplog.text


def test_a_failed_resize_is_reported(tmp_path, monkeypatch, caplog):
    """Silently paying full high-res rates is the failure nobody notices."""
    monkeypatch.setattr(imagefit, "CACHE_DIR", tmp_path / "fit")
    bad = tmp_path / "tile_0000.jpg"
    bad.write_bytes(b"not a jpeg")
    imagefit.fit(bad, imagefit.ANTHROPIC_POLICY)
    assert "full resolution" in caplog.text


def test_a_corrupt_answer_cache_is_reported(tmp_path, caplog):
    (tmp_path / "m.json").write_text("{not json")
    (tmp_path / "v.npz").write_bytes(b"rubbish")
    answer_cache.AnswerCache(tmp_path / "v.npz", tmp_path / "m.json")
    assert "unreadable" in caplog.text


def test_an_inconsistent_answer_cache_is_reported(tmp_path, caplog):
    """Pairing a question with another question's vector is the failure mode."""
    import json

    import numpy as np

    np.savez_compressed(tmp_path / "v.npz", v=np.eye(3, 8, dtype=np.float32))
    (tmp_path / "m.json").write_text(json.dumps([{"question": "one"}]))
    answer_cache.AnswerCache(tmp_path / "v.npz", tmp_path / "m.json")
    assert "inconsistent" in caplog.text


def test_a_changed_embedding_model_is_reported(tmp_path, caplog):
    c = answer_cache.AnswerCache(tmp_path / "v.npz", tmp_path / "m.json")
    c.put([1.0] * 8, "old", "ns", {"answer": "a"})
    c.put([1.0] * 4, "new", "ns", {"answer": "b"})
    assert "embedding width changed" in caplog.text


def test_an_absent_cache_is_not_a_warning(tmp_path, caplog):
    """The normal first run must be quiet, or the warnings mean nothing."""
    answer_cache.AnswerCache(tmp_path / "v.npz", tmp_path / "m.json")
    assert caplog.text == ""


# -- configuration ----------------------------------------------------------

def test_configure_logging_does_not_hijack_the_root_logger():
    """A library that installs a root handler on import steals logging from
    whatever embeds it."""
    rag.configure_logging()
    assert logging.getLogger("pixelrag").propagate is False
    assert not logging.getLogger().handlers or \
        logging.getLogger("pixelrag").handlers


def test_the_level_is_configurable(monkeypatch):
    monkeypatch.setenv("PIXELRAG_LOG", "debug")
    rag.configure_logging()
    assert logging.getLogger("pixelrag").level == logging.DEBUG
    rag.configure_logging("error")
    assert logging.getLogger("pixelrag").level == logging.ERROR


# -- helpers ----------------------------------------------------------------

class _NullReader:
    name, model = "null", "null-1"
    image_policy = imagefit.PASSTHROUGH

    def price(self, usage):
        return None

    def read_pages(self, *a, **k):
        raise AssertionError("should not be reached: no pages were retrieved")


def _fake_search_response(*a, **k):
    class R:
        ok = True

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"results": [{"hits": []}]}
    return R()
