"""The reader's citation block: parsing it, and turning it into rectangles.

A one-shot answer arrives as prose followed by a `---CYTATY---` block of JSONL,
one line per claim, each naming a page and quoting the span it came from. This
module is everything between that text and a highlight the UI can draw.

It is separate from `citations.py`, which it uses: that one is pure geometry
over a PDF's word boxes and knows nothing about this index, while this one
knows about articles, attached pages and what the reader was asked. Keeping the
geometry ignorant is what lets it be tested against any PDF.

TWO INDEPENDENT SOURCES OF GEOMETRY, because they fail in different places and
a citation with neither is still worth showing:

  * the quoted phrase, located in the PDF text layer — precise, and the thing
    the reader actually claims to have read;
  * every figure in the answer prose, located on the same page — this is what
    someone checking a quoted price wants boxed, and the reader rarely quotes
    the number and its row label in one span.

Everything here degrades rather than raises. A reader that omits the block,
fences it in backticks or writes one malformed line still produces a usable
answer; a quote that cannot be found on the page it names is KEPT and flagged
`verified: false`, because that is the clearest available signal that the
reader drifted, and hiding it would make a wrong citation look like no
citation.
"""

from __future__ import annotations

import json
import logging
import re

import citations as C
import corpus

log = logging.getLogger("pixelrag")

CITE_MARK = "---CYTATY---"


def split_citations(text: str) -> tuple[str, list[dict]]:
    """Separate the prose answer from the trailing ---CYTATY--- JSONL block.

    Tolerant by design: a reader that omits the block, fences it in ```, or
    writes one malformed line should still produce a usable answer. Anything
    unparseable is dropped rather than shown.
    """
    if CITE_MARK not in text:
        return text.strip(), []
    body, _, tail = text.partition(CITE_MARK)
    cites = []
    for line in tail.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(c, dict) and c.get("quote"):
            cites.append(c)
    return body.strip(), cites


def resolve_citations(answer: str, cites: list[dict],
                      pages: list[dict]) -> list[dict]:
    """Attach highlight rectangles to each citation, over the ORIGINAL page.

    Two independent sources of geometry, because they fail in different places:
      * the quoted phrase, located in the PDF text layer — precise, and the thing
        the reader actually claims to have read;
      * every figure in the answer prose, located on the same page — this is what
        someone checking a quoted price wants boxed, and the reader rarely quotes
        the number and its row label in one span.
    A citation whose quote cannot be found on the page it names is kept and
    flagged: that is the clearest available signal that the reader drifted.
    """
    # Keyed on (article_id, page), NOT page alone: both catalogues have a page
    # 61, so a bare page number resolved a citation against whichever document
    # happened to be last in the list — then located the quote in that
    # document's PDF and drew the highlight there. The reader names only a page
    # number, so the article is recovered by looking for that page among the
    # ones actually attached, preferring the highest-ranked.
    by_key = {(p["article_id"], p["page"]): p for p in pages}
    first_by_page: dict[int, dict] = {}
    for p in pages:
        first_by_page.setdefault(p["page"], p)

    out = []
    for c in cites:
        try:
            pno = int(c.get("page"))
        except (TypeError, ValueError):
            continue
        aid = c.get("article_id")
        target = (by_key.get((aid, pno)) if aid is not None
                  else first_by_page.get(pno))
        if target is None:
            continue
        pdf = corpus.source_pdf(target["article_id"])
        quote = str(c["quote"])
        found = C.locate_quote(pdf, pno, quote) if pdf else []
        out.append({
            "document": target["document"],
            "article_id": target["article_id"],
            "page": pno,
            "quote": quote,
            "supports": c.get("supports") or "",
            "rects": found[0]["rects"] if found else [],
            "coverage": found[0]["coverage"] if found else 0.0,
            # No text layer at all → the region box from PixelRAG is the only
            # geometry available, and the UI falls back to it.
            "text_layer": bool(pdf and C._page_words(str(pdf), pno)),
            "verified": bool(found),
        })

    # Numbers are per (document, page), not per citation — attach to the first
    # citation of each so the UI draws each box once. Grouping by page alone
    # would pin one document's figures onto another's page of the same number.
    for aid, pno in {(c["article_id"], c["page"]) for c in out}:
        pdf = corpus.source_pdf(aid)
        if not pdf:
            continue
        pinned, repeated = C.locate_numbers(pdf, pno, answer)
        for c in out:
            if c["article_id"] == aid and c["page"] == pno:
                c["numbers"] = pinned
                c["repeated"] = repeated
                break
    return out


_NUM = re.compile(r"\d[\d   .,]{1,12}\d|\d{2,}")


def locate_values(article_id: int, page: int, answer: str) -> list[dict]:
    """Find where the numbers the model quoted actually sit on the page.

    The rendered tiles are pixels, but these PDFs still carry a text layer, so
    every figure in the answer can be located exactly rather than approximated
    by the region the model happened to open. Returns boxes as percentages of
    the page, the same convention rag.box_pct uses for chunk overlays.

    This is a *check on the model*, not a source of truth: a quoted number that
    cannot be found on the cited page is worth seeing.
    """
    src = corpus.source_pdf(article_id)
    if src is None:
        return []

    # "3 555 zł" / "3.555" / "3555" should all find the cell reading 3555.
    # Prices and millimetre dimensions are 3-6 digits. Anything longer is a
    # date or a document code (answers cite "CBG_PL-PLN_10.10.2024"); anything
    # shorter matches half the table.
    stem = src.stem
    wanted: set[str] = set()
    for m in _NUM.finditer(answer):
        raw = m.group(0).strip(" .,\u00a0\u202f")
        if raw in stem:                       # part of the document identifier
            continue
        digits = re.sub(r"[\s.,\u00a0\u202f]", "", raw)
        if not digits.isdigit() or not 3 <= len(digits) <= 6:
            continue
        wanted.add(digits)
        if raw != digits and len(raw) - len(digits) == 1:
            wanted.add(raw)                   # "3 555" as typeset in the page
    if not wanted:
        return []

    import fitz

    out: list[dict] = []
    with fitz.open(src) as doc:
        if not 0 <= page - 1 < doc.page_count:
            return []
        pg = doc[page - 1]
        pw, ph = pg.rect.width, pg.rect.height
        for term in sorted(wanted, key=len, reverse=True)[:12]:
            try:
                rects = pg.search_for(term)
            except Exception:
                continue
            for r in rects[:6]:
                out.append({
                    "value": term,
                    "left": 100 * r.x0 / pw, "top": 100 * r.y0 / ph,
                    "width": 100 * (r.x1 - r.x0) / pw,
                    "height": 100 * (r.y1 - r.y0) / ph,
                    "ambiguous": len(rects) > 1,
                })
    return _confirm_intersections(out)


def _confirm_intersections(pins: list[dict], tol: float = 1.5) -> list[dict]:
    """Resolve which copy of a repeated figure is the one actually cited.

    In a price matrix the answer names the row and column thresholds as well as
    the price, so the right cell is the one sitting under a quoted column header
    and level with a quoted row header. A value that repeats elsewhere in the
    table will not satisfy both. This turns two dashed guesses into one
    confirmed cell without asking the model where it looked.
    """
    for p in pins:
        has_col = any(q is not p and abs(q["left"] - p["left"]) < tol
                      and q["top"] < p["top"] - tol for q in pins)
        has_row = any(q is not p and abs(q["top"] - p["top"]) < tol
                      and q["left"] < p["left"] - tol for q in pins)
        p["confirmed"] = bool(has_col and has_row)
    # A confirmed cell is no longer an open question, whatever its twin does.
    for p in pins:
        if p["confirmed"]:
            p["ambiguous"] = False
    return pins


