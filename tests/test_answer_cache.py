"""Cache semantics. A stale hit quotes last week's price list, so the namespace
and threshold rules are the load-bearing part, not the storage."""

from __future__ import annotations

import numpy as np
import pytest

import answer_cache


def vec(*xs):
    v = np.zeros(8, dtype=np.float32)
    for i, x in enumerate(xs):
        v[i] = x
    return v.tolist()


@pytest.fixture
def cache(tmp_path):
    return answer_cache.AnswerCache(tmp_path / "v.npz", tmp_path / "m.json")


def result(answer="odpowiedź"):
    return {"answer": answer, "trace": [{"type": "tile"}], "usage": {}}


def test_miss_on_empty_cache(cache):
    assert cache.get(vec(1), "ns") is None
    assert cache.stats()["misses"] == 1


def test_identical_question_hits(cache):
    cache.put(vec(1), "Ile kosztuje brama?", "ns", result())
    hit = cache.get(vec(1), "ns")
    assert hit["question"] == "Ile kosztuje brama?"
    assert hit["similarity"] == pytest.approx(1.0)


def test_unrelated_question_misses(cache):
    cache.put(vec(1), "q", "ns", result())
    assert cache.get(vec(0, 1), "ns") is None


def test_threshold_is_the_cut(cache, monkeypatch):
    cache.put(vec(1), "q", "ns", result())
    near = vec(0.96, 0.28)                       # cos ~0.96 to vec(1)
    monkeypatch.setattr(answer_cache, "THRESHOLD", 0.99)
    assert cache.get(near, "ns") is None
    monkeypatch.setattr(answer_cache, "THRESHOLD", 0.90)
    assert cache.get(near, "ns") is not None


def test_namespace_is_exact_match(cache):
    """Model, page budget, hybrid flag and index identity all change the answer
    to the same question — they must miss, not fuzzy-match."""
    cache.put(vec(1), "q", "gemini|k4|hyb0", result())
    assert cache.get(vec(1), "gemini|k8|hyb0") is None
    assert cache.get(vec(1), "gemini|k4|hyb1") is None
    assert cache.get(vec(1), "gemini|k4|hyb0") is not None


def test_a_near_miss_in_another_namespace_cannot_win(cache):
    """Masking, not filtering: the wrong-namespace neighbour must not be picked
    just because it is nearest overall."""
    cache.put(vec(1), "exact", "other", result("wrong"))
    cache.put(vec(0.98, 0.2), "close", "mine", result("right"))
    hit = cache.get(vec(1), "mine")
    assert hit["result"]["answer"] == "right"


def test_zero_vector_is_not_a_key(cache):
    cache.put(vec(0), "q", "ns", result())
    assert cache.get(vec(0), "ns") is None


def test_survives_a_reload(cache, tmp_path):
    cache.put(vec(1), "q", "ns", result("zapisane"))
    reopened = answer_cache.AnswerCache(tmp_path / "v.npz", tmp_path / "m.json")
    assert reopened.get(vec(1), "ns")["result"]["answer"] == "zapisane"


def test_mismatched_vector_and_meta_lengths_are_discarded(tmp_path):
    """Pairing a question with someone else's vector is worse than no cache."""
    import json

    np.savez_compressed(tmp_path / "v.npz", v=np.eye(3, 8, dtype=np.float32))
    (tmp_path / "m.json").write_text(json.dumps([{"question": "only one"}]))
    assert answer_cache.AnswerCache(tmp_path / "v.npz", tmp_path / "m.json").get(
        vec(1), "ns") is None


def test_a_new_embedding_model_resets_rather_than_mixes_spaces(cache):
    cache.put(vec(1), "old", "ns", result())
    cache.put([1.0, 0.0], "new", "ns", result("nowa"))   # 2-d, different model
    assert cache.stats()["entries"] == 1
    assert cache.get([1.0, 0.0], "ns")["result"]["answer"] == "nowa"


def test_eviction_keeps_the_most_recently_useful(cache, monkeypatch):
    monkeypatch.setattr(answer_cache, "MAX_ENTRIES", 2)
    cache.put(vec(1), "first", "ns", result())
    cache.put(vec(0, 1), "second", "ns", result())
    cache.get(vec(1), "ns")                      # first becomes recently used
    cache.put(vec(0, 0, 1), "third", "ns", result())
    assert {m["question"] for m in cache._meta} == {"first", "third"}


def test_clear_empties_and_removes_the_files(cache):
    cache.put(vec(1), "q", "ns", result())
    cache.clear()
    assert cache.stats()["entries"] == 0
    assert not cache.vec_path.exists() and not cache.meta_path.exists()


def test_disabled_never_reads_or_writes(cache, monkeypatch):
    monkeypatch.setattr(answer_cache, "ENABLED", False)
    cache.put(vec(1), "q", "ns", result())
    assert cache.stats()["entries"] == 0
    assert cache.get(vec(1), "ns") is None


def test_index_fingerprint_changes_when_the_index_is_rebuilt(tmp_path):
    (tmp_path / "index.faiss").write_bytes(b"a")
    (tmp_path / "articles.json").write_text("[]")
    (tmp_path / "summary.json").write_text("{}")
    before = answer_cache.index_fingerprint(tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"aa")
    assert answer_cache.index_fingerprint(tmp_path) != before


def test_index_fingerprint_tolerates_a_missing_index(tmp_path):
    assert answer_cache.index_fingerprint(tmp_path / "nope")
