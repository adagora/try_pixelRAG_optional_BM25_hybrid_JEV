#!/usr/bin/env python
"""Turn a quote the reader copied out of a page into highlight rectangles.

The reader answers from page *images*, so its citations are text it read off
pixels. To draw a highlight we have to find that text again in the PDF's own
text layer and get its coordinates. `page.search_for()` alone is not enough on
this corpus, for three reasons that all show up on the first page you try:

  * Justified Polish body text is hyphenated across lines — the page literally
    contains "skrzydło wyko- nane jest", so searching for "skrzydło wykonane"
    matches nothing.
  * Prices are typeset with thin/no-break spaces ("3 956"), and the reader
    writes them back as "3956".
  * A quote spanning two lines has no single rectangle. One box around both
    lines covers the whole column width and looks like a bug.

So we match at the *word* level against `page.get_text("words")`, which gives a
box per word, normalise both sides, find the best consecutive run, and return
one rectangle per line of the match. That is what a highlighter pen does, and it
degrades gracefully: a partial match still highlights the part it found.

Coordinates come back as percentages of the page so the UI can overlay them on
the rendered page image at any zoom without knowing the render DPI.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

# Keep digits together with letters: "A-1", "RC2", "W3-1" are content words here.
_TOKEN = re.compile(r"[0-9a-ząćęłńóśźż]+", re.IGNORECASE)

# Minimum share of a quote's tokens that must line up before we call it a match.
# Below this the "best run" is coincidence — usually a single shared stopword.
MIN_COVERAGE = 0.55
# A one-token match is only meaningful if the token is distinctive.
MIN_RUN = 2


def _fold(s: str) -> str:
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _norm_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN.finditer(_fold(text).lower())]


@lru_cache(maxsize=512)
def _page_words(pdf: str, page: int) -> tuple[tuple, ...]:
    """(x0, y0, x1, y1, normalised_token, line_key) per word, reading order.

    A single PDF "word" can normalise to several tokens ("2500x2100" -> 2500,
    2100) or to none (a bare "-"), so the mapping is not 1:1 and each emitted
    token carries its own source rectangle.
    """
    import fitz

    with fitz.open(pdf) as doc:
        if not 0 <= page - 1 < doc.page_count:
            return ()
        pg = doc[page - 1]
        out = []
        for x0, y0, x1, y1, w, block, line, _no in pg.get_text("words"):
            toks = _norm_tokens(w)
            if not toks:
                continue
            # Split the word box proportionally when one word yields several
            # tokens, so highlighting "2500" in "2500x2100" boxes only "2500".
            total = sum(len(t) for t in toks)
            cur = x0
            for t in toks:
                frac = len(t) / total
                nx = cur + (x1 - x0) * frac
                out.append((cur, y0, nx, y1, t, (block, line)))
                cur = nx
        return tuple(out)


def _best_run(words: tuple[tuple, ...], q: list[str]) -> tuple[int, int, int]:
    """Longest consecutive run of `q` found in `words`. Returns (start, end, matched).

    Greedy rather than full alignment: walk every position where the quote's
    first token occurs and extend, allowing a bounded number of skips on the
    page side so hyphen fragments and stray typeset marks do not end the run.
    Quote-side skips are not allowed — the reader's words must all appear.
    """
    if not q or not words:
        return (0, 0, 0)
    page_toks = [w[4] for w in words]
    best = (0, 0, 0)
    starts = [i for i, t in enumerate(page_toks) if t == q[0]]
    # If the quote's first token never appears (reader paraphrased the opening),
    # try anchoring on its longest, most distinctive token instead.
    if not starts:
        anchor = max(range(len(q)), key=lambda i: len(q[i]))
        starts = [i - anchor for i, t in enumerate(page_toks)
                  if t == q[anchor] and i - anchor >= 0]
    for s in starts:
        qi, pi, matched, skips = 0, s, 0, 0
        while qi < len(q) and pi < len(page_toks):
            if page_toks[pi] == q[qi]:
                matched += 1
                qi += 1
                pi += 1
            elif q[qi].startswith(page_toks[pi]) or page_toks[pi].startswith(q[qi]):
                # Hyphenation: page has "wyko" + "nane", quote has "wykonane".
                joined = page_toks[pi]
                pj = pi + 1
                while pj < len(page_toks) and len(joined) < len(q[qi]):
                    joined += page_toks[pj]
                    pj += 1
                if joined == q[qi]:
                    matched += 1
                    qi += 1
                    pi = pj
                else:
                    skips += 1
                    pi += 1
            else:
                skips += 1
                pi += 1
            if skips > 2 + len(q) // 4:
                break
        if matched > best[2]:
            best = (s, pi, matched)
    return best


def locate_quote(pdf: str | Path, page: int, quote: str) -> list[dict]:
    """Highlight rectangles (percent of page) for `quote` on `page`.

    One rectangle per line of text the match spans. Empty list when the quote
    cannot be found — which is itself worth surfacing: a citation the page does
    not contain is the single most useful signal that the reader drifted.
    """
    pdf = str(pdf)
    q = _norm_tokens(quote)
    if not q:
        return []
    words = _page_words(pdf, page)
    if not words:
        return []                     # image-only page: caller falls back to region box

    start, end, matched = _best_run(words, q)
    if matched < MIN_RUN or matched < MIN_COVERAGE * len(q):
        return []

    import fitz

    with fitz.open(pdf) as doc:
        pw, ph = doc[page - 1].rect.width, doc[page - 1].rect.height

    # Group the matched span into lines, then union each line's boxes.
    by_line: dict[tuple, list[tuple]] = {}
    for w in words[start:end]:
        if w[4] in q or any(w[4] in t or t in w[4] for t in q):
            by_line.setdefault(w[5], []).append(w)

    rects = []
    for line in sorted(by_line, key=lambda k: (k[0], k[1])):
        ws = by_line[line]
        x0 = min(w[0] for w in ws)
        y0 = min(w[1] for w in ws)
        x1 = max(w[2] for w in ws)
        y1 = max(w[3] for w in ws)
        rects.append({
            "left": 100 * x0 / pw, "top": 100 * y0 / ph,
            "width": 100 * (x1 - x0) / pw, "height": 100 * (y1 - y0) / ph,
        })
    if not rects:
        return []
    return [{
        "rects": rects,
        "coverage": round(matched / len(q), 2),
        "exact": matched == len(q),
    }]


def locate_numbers(pdf: str | Path, page: int, text: str,
                   max_terms: int = 8) -> list[dict]:
    """Boxes for price/dimension figures in `text` that appear ONCE on `page`.

    Complements locate_quote: a reader that answers "+130 PLN za szt." without
    quoting the surrounding row still names a number, and pinning that number is
    what a user checking a quote actually wants to see. 3-6 digits covers prices
    and millimetre dimensions; shorter matches half a table, longer is a document
    code or a date.

    Returns (pinned, repeated).

    Only figures occurring exactly once are pinned, and that restriction is the
    point. Measured on the SNP matrix (page 39) for a one-line answer, boxing
    every occurrence drew 15 rectangles: the price 3956 once, then 2500 six
    times, 2100 four times and 7016 four times — axis thresholds and colour
    codes repeated across that page's four tables. Fourteen boxes of noise
    around the one that matters is worse than drawing nothing.

    Repeated figures come back as counts without geometry, so the UI can say
    "2500 appears 6x on this page" rather than point at all six. Which copy the
    reader meant is not recoverable from the number alone — that is what the
    quote rectangle is for.
    """
    pdf = str(pdf)
    words = _page_words(pdf, page)
    if not words:
        return [], []

    wanted: list[str] = []
    for m in re.finditer(r"\d[\d\s.,  ]{1,12}\d|\d{2,}", text):
        digits = re.sub(r"[\s.,  ]", "", m.group(0))
        if digits.isdigit() and 3 <= len(digits) <= 6 and digits not in wanted:
            wanted.append(digits)

    import fitz

    with fitz.open(pdf) as doc:
        pw, ph = doc[page - 1].rect.width, doc[page - 1].rect.height

    pinned, repeated = [], []
    for term in wanted[:max_terms]:
        hits = [w for w in words if w[4] == term]
        if not hits:
            continue
        if len(hits) > 1:
            repeated.append({"value": term, "count": len(hits)})
            continue
        w = hits[0]
        pinned.append({
            "value": term,
            "left": 100 * w[0] / pw, "top": 100 * w[1] / ph,
            "width": 100 * (w[2] - w[0]) / pw,
            "height": 100 * (w[3] - w[1]) / ph,
        })
    return pinned, repeated
