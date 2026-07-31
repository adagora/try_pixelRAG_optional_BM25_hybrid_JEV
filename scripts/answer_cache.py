#!/usr/bin/env python
"""Reuse a previous answer when the question means the same thing.

The expensive part of a query is not finding the pages, it is the reader: four
page screenshots is ~9k image tokens and a multi-second round trip, and asking
the same thing twice pays for it twice. Exact-string caching barely helps —
people rephrase — but the pipeline already computes a 2048-d embedding of every
question, 138ms before the reader is ever called. Two questions that embed to
the same point have the same answer in this corpus, so that vector is a
free-and-already-paid-for cache key.

Lookup is a dot product against a few hundred stored vectors: sub-millisecond,
against a reader call measured in seconds and cents.

WHAT MAKES AN ENTRY REUSABLE. Only the question is fuzzy-matched. Everything
that would change the answer for the *same* question — provider, model, ask
mode, page budget, whether BM25 is fused in, and the identity of the index
itself — is folded into an exact-match namespace. Rebuilding the index or
flipping PIXELRAG_HYBRID does not return stale answers, it misses the cache.
That is deliberate: a wrong cached price is far more expensive than a cache
miss.

THRESHOLD -- measured on this corpus's encoder, not assumed. Against
"Jakie kolory i wzory paneli sa dostepne?":

    1.0000  identical
    0.9704  reordered   "Jakie wzory i kolory paneli sa dostepne?"
    0.9540  politeness  "Prosze podac jakie kolory i wzory paneli sa dostepne"
    0.8963  synonyms    "Jakie sa dostepne barwy i desenie paneli?"
    0.8762  terse       "kolory i wzory paneli"
    0.2405  unrelated   "Ile kosztuje montaz bramy roletowej?"

Two things to take from that. The floor is far lower than a similarity-cache
rule of thumb suggests -- an unrelated question sits at 0.24, not 0.6-0.8 --
so there is a lot of headroom below the paraphrases. But genuine rephrasings
land at 0.88-0.97, NOT above 0.98, so a 0.97 cut catches little more than exact
repeats and near-identical word order.

0.95 is the default: it picks up reordering and politeness prefixes while
staying ~0.7 clear of anything unrelated. Going lower to catch synonym swaps
(0.90) is defensible on a corpus this small, where a wrong hit is obvious, but
it should be an eval decision rather than a guess. Err high when unsure: a false
hit confidently answers a question nobody asked.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

import layout

ENABLED = os.environ.get("PIXELRAG_ANSWER_CACHE", "1") != "0"
THRESHOLD = float(os.environ.get("PIXELRAG_CACHE_THRESHOLD", "0.95"))
MAX_ENTRIES = int(os.environ.get("PIXELRAG_CACHE_MAX", "2000"))

# Beside the index by default, because an entry is only valid for the index it
# was read out of (see index_fingerprint). PIXELRAG_CACHE_DIR moves it; a
# relative value is anchored to the repo root, not to the working directory.
_CACHE_LAYOUT = layout.IndexLayout(
    layout._anchored(os.environ.get("PIXELRAG_CACHE_DIR", ""),
                     layout.DEFAULT.index_dir))
CACHE_DIR = _CACHE_LAYOUT.index_dir
VEC_PATH = _CACHE_LAYOUT.answer_cache_vectors
META_PATH = _CACHE_LAYOUT.answer_cache_meta


def index_fingerprint(index_dir: Path) -> str:
    """Identity of the index the cached answers were read out of.

    Size and mtime rather than a content hash: index.faiss is hundreds of MB
    and this runs on every question. A rebuild always moves both.
    """
    parts = []
    for name in ("index.faiss", "articles.json", "summary.json"):
        p = Path(index_dir) / name
        try:
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{name}:-")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class AnswerCache:
    """Vectors in one array, payloads in a parallel list, both rewritten on put.

    Not a database on purpose. A few hundred entries of a few KB each is small
    enough that the simplest thing that survives a restart is the right thing,
    and a corrupt cache is recoverable by deleting two files.
    """

    def __init__(self, vec_path: Path = VEC_PATH, meta_path: Path = META_PATH):
        self.vec_path = Path(vec_path)
        self.meta_path = Path(meta_path)
        self._lock = threading.Lock()
        self._vecs: np.ndarray | None = None
        self._meta: list[dict] = []
        self.hits = 0
        self.misses = 0
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            vecs = np.load(self.vec_path)["v"].astype(np.float32)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return
        # A half-written pair is worse than none: drop both rather than pair a
        # question with someone else's vector.
        if isinstance(meta, list) and len(meta) == len(vecs):
            self._meta, self._vecs = meta, vecs

    def _save(self) -> None:
        try:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            if self._vecs is None or not len(self._meta):
                self.meta_path.unlink(missing_ok=True)
                self.vec_path.unlink(missing_ok=True)
                return
            # Vectors first: a crash between the two leaves extra vectors, which
            # the length check discards. The reverse would orphan answers.
            np.savez_compressed(self.vec_path, v=self._vecs)
            self.meta_path.write_text(
                json.dumps(self._meta, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    # -- api ---------------------------------------------------------------

    def get(self, embedding, namespace: str) -> dict | None:
        """Nearest cached answer within THRESHOLD, or None."""
        if not ENABLED:
            return None
        with self._lock:
            if self._vecs is None or not len(self._meta):
                self.misses += 1
                return None
            q = np.asarray(embedding, dtype=np.float32)
            n = float(np.linalg.norm(q))
            if not n:
                self.misses += 1
                return None
            q /= n

            sims = self._vecs @ q
            # Namespace is exact-match, so a near-miss in the wrong namespace
            # must not win. Masking beats filtering: the array stays contiguous.
            mask = np.array([m.get("namespace") == namespace for m in self._meta])
            if not mask.any():
                self.misses += 1
                return None
            sims = np.where(mask, sims, -np.inf)

            i = int(np.argmax(sims))
            if sims[i] < THRESHOLD:
                self.misses += 1
                return None

            self.hits += 1
            entry = self._meta[i]
            entry["used"] = entry.get("used", 0) + 1
            entry["last_used"] = time.time()
            return {
                "similarity": float(sims[i]),
                "question": entry["question"],
                "result": entry["result"],
            }

    def put(self, embedding, question: str, namespace: str, result: dict) -> None:
        if not ENABLED:
            return
        q = np.asarray(embedding, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if not n:
            return
        q = (q / n).reshape(1, -1)

        with self._lock:
            if self._vecs is None or not len(self._meta):
                self._vecs = q
                self._meta = []
            elif self._vecs.shape[1] != q.shape[1]:
                # A different embedding model was configured. Its vectors are
                # not comparable to these, and silently mixing spaces would
                # make every similarity meaningless.
                self._vecs, self._meta = q, []
            else:
                self._vecs = np.vstack([self._vecs, q])
            self._meta.append({
                "question": question,
                "namespace": namespace,
                "result": result,
                "created": time.time(),
                "used": 0,
            })
            if len(self._meta) > MAX_ENTRIES:
                # Evict least-recently-useful: never-used entries by age.
                keep = sorted(
                    range(len(self._meta)),
                    key=lambda i: (self._meta[i].get("last_used")
                                   or self._meta[i]["created"]),
                    reverse=True,
                )[:MAX_ENTRIES]
                keep.sort()
                self._vecs = self._vecs[keep]
                self._meta = [self._meta[i] for i in keep]
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._vecs, self._meta = None, []
            self._save()

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._meta),
                "hits": self.hits,
                "misses": self.misses,
                "threshold": THRESHOLD,
                "enabled": ENABLED,
            }


_default: AnswerCache | None = None


def default() -> AnswerCache:
    global _default
    if _default is None:
        _default = AnswerCache()
    return _default


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect or clear the answer cache.")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    c = default()
    if args.clear:
        c.clear()
        print("cleared")
        return
    print(json.dumps(c.stats(), indent=2))
    for m in c._meta:
        print(f"  [{m.get('used', 0):3d} uses] {m['question'][:70]}")


if __name__ == "__main__":
    _main()
