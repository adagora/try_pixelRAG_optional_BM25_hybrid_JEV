#!/usr/bin/env python
"""Polish-aware lexical (BM25) sidecar over the PDF text layer.

Why this exists: PixelRAG matches on how a page *looks*. That works for
"pokaż rodzaje paneli" (a page of panel drawings looks distinctive) and fails
for "dzielony wał" — one row out of thirty in an options table, where the whole
875x1024 chunk looks like every other block of small numbers. Measured on the
real question set, terse Polish noun phrases returned near-tied scores across
unrelated documents: no visual signal at all.

Lexical search is exactly right for that case and nearly free, because these
price lists are digital-born and carry a full text layer.

Self-contained BM25 rather than rank_bm25: it is 40 lines, and Polish needs
custom normalisation anyway (diacritic folding + suffix stripping), which is
where the actual retrieval quality lives.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import layout

INDEX_DIR = layout.DEFAULT.index_dir
TEXT_SIDECAR = layout.DEFAULT.text_sidecar

# Query words that carry intent, not content. "Pokaż próg uszczelniający" must
# match on "próg uszczelniający"; leaving "pokaż"/"jaka"/"jest" in dilutes every
# score and, worse, matches document boilerplate.
STOP = {
    # interrogatives / imperatives
    "pokaz", "pokazac", "podaj", "jaka", "jaki", "jakie", "jak", "ile", "czy",
    "gdzie", "kiedy", "co", "ktory", "ktora", "ktore", "wymien", "opisz",
    # copulas / function words
    "jest", "sa", "byc", "ma", "maja", "moze", "mozna", "w", "we", "z", "ze",
    "na", "do", "od", "za", "o", "u", "i", "oraz", "lub", "a", "the", "of",
    "dla", "przy", "po", "pod", "nad", "bez", "przez",
    # near-universal in this corpus: every page is about bramy
    "temat", "informacje", "informacja",
}

# Polish inflection, handled by suffix stripping. A real stemmer (Morfologik)
# would be better, but these are the endings that actually collide in this
# corpus: "roletowe/roletowa/roletowych", "uszczelniajacy/uszczelniajace".
_SUFFIXES = (
    "iajacych", "iajacego", "ajacych", "ajacego", "iajacy", "iajace", "ajacy",
    "ajace", "owych", "owego", "iach", "ami", "ach", "emu", "ego", "ych", "ymi",
    "owe", "owa", "owi", "ow", "em", "ie", "ia", "ym", "ej", "ie", "y", "a",
    "e", "i", "u", "o",
)

_WORD = re.compile(r"[0-9a-ząćęłńóśźż]+", re.IGNORECASE)


def fold(s: str) -> str:
    """Strip Polish diacritics so 'wał' and 'wal' are the same token.

    Users type without diacritics constantly ("Pokaz" in the real question
    set), and OCR drops them. Folding both sides costs nothing.
    """
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def stem(w: str) -> str:
    """Crude but effective: strip the longest matching inflectional ending.

    Guarded at 4 chars so short content words ('wal', 'RC2', 'SNP', 'okno')
    survive intact — over-stemming them would merge unrelated terms.
    """
    if len(w) <= 4 or w.isdigit():
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def tokenise(text: str, *, drop_stop: bool = True) -> list[str]:
    out = []
    for m in _WORD.finditer(fold(text.lower())):
        w = m.group(0)
        if drop_stop and w in STOP:
            continue
        if len(w) < 2:
            continue
        out.append(stem(w))
    return out


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

class BM25:
    """Standard Okapi BM25 over a small in-memory page collection.

    The corpus here is hundreds of pages, not millions, so a plain dict
    postings list is faster than anything that needs a build step.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lens = [len(d) for d in docs]
        self.avglen = (sum(self.lens) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        # +0.5/+0.5 smoothing keeps idf positive for terms in most documents,
        # which matters at this collection size (a term in 12 of 16 docs is
        # still informative here, and plain BM25 idf would go negative).
        self.idf = {
            w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.n
        for w in query_tokens:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avglen)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


# --------------------------------------------------------------------------
# sidecar
# --------------------------------------------------------------------------

def build_sidecar(articles: list[dict], out_path: Path = TEXT_SIDECAR) -> dict:
    """Extract per-page text for every PDF and record which pages have none.

    Pages with no text layer are flagged rather than dropped: the visual index
    is the only thing that can reach them, and a caller that knows a page is
    visual-only can say so instead of silently under-ranking it.
    """
    import fitz

    pages = []
    for aid, a in enumerate(articles):
        src = Path(a.get("url") or "")
        if not src.exists() or src.suffix.lower() != ".pdf":
            continue
        with fitz.open(src) as doc:
            for pno in range(doc.page_count):
                text = doc[pno].get_text()
                pages.append({
                    "article_id": aid,
                    "title": a.get("title", str(aid)),
                    "page": pno + 1,
                    "text": text,
                    "visual_only": len(text.strip()) < 20,
                    "revision": _revision(text),
                })
    data = {"pages": pages}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


_REV = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")


def _revision(text: str) -> str | None:
    """Price-list vintage, e.g. 'CBG/PL-PLN/10.04.2025' -> '2025-04-10'.

    Load-bearing: this corpus holds two editions of the same price list, and
    they disagree (wkladka antywlamaniowa is +99 in the 2024 one and +130 in
    the 2025 one). Without a date the reader has no way to prefer the current
    price, and quoting a stale one to a customer is the expensive failure.
    """
    m = re.search(r"CBG/PL-PLN/" + _REV.pattern, text)
    if not m:
        m = _REV.search(text[:400])         # fall back to any date in the header
    if not m:
        return None
    d, mo, y = m.groups()
    return f"{y}-{mo}-{d}"


@lru_cache(maxsize=1)
def _loaded() -> tuple[list[dict], BM25]:
    if not TEXT_SIDECAR.exists():
        raise FileNotFoundError(
            f"{TEXT_SIDECAR} missing — run scripts/build_text_index.py")
    # A half-written sidecar reads as "hybrid is broken" rather than "hybrid is
    # not built", and the two have different fixes, so say which one this is.
    try:
        pages = json.loads(TEXT_SIDECAR.read_text(encoding="utf-8"))["pages"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(
            f"{TEXT_SIDECAR} is unreadable ({e!r}) — the sidecar is incomplete; "
            f"rebuild it with scripts/build_text_index.py") from e
    return pages, BM25([tokenise(p["text"]) for p in pages])


def pages() -> list[dict]:
    return _loaded()[0]


def revision_of(article_id: int, page: int) -> str | None:
    for p in pages():
        if p["article_id"] == article_id and p["page"] == page:
            return p["revision"]
    return None


def search_text(query: str, n: int = 8) -> list[dict]:
    """Top-n pages by BM25. Returns [] when no query term is in the vocabulary."""
    pgs, bm = _loaded()
    toks = tokenise(query)
    if not toks:
        return []
    scores = bm.scores(toks)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order[:n]:
        if scores[i] <= 0:
            break
        p = pgs[i]
        out.append({
            "article_id": p["article_id"],
            "document": p["title"],
            "page": p["page"],
            "score": round(scores[i], 4),
            "revision": p["revision"],
        })
    return out


def matched_terms(query: str) -> list[str]:
    """Query tokens that exist in the collection — what is worth highlighting."""
    _, bm = _loaded()
    return [t for t in dict.fromkeys(tokenise(query)) if t in bm.idf]
