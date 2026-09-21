"""Visual retrieval over the local PixelRAG index.

Ask modes (PIXELRAG_ASK):

    oneshot  (default)  pixelrag search → top page screenshots → one VLM read
    agent               multi-turn tool browse (search + tile), like PixelRAG's
                        web/agent-server.mjs — slower, for hard navigation

Why oneshot is the default: it *is* PixelRAG (retrieve screenshots, read them)
without the multi-round image re-send tax of an agent loop. Agent mode remains
for cases where the reader must navigate region-by-region.

Shared by ask.py (CLI) and app.py (web UI) so prompt and policy cannot drift.
"""

import base64
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load project credentials before providers read their environment settings.
# Explicitly exported values always take precedence over .env.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

import answer_cache
import chunkmeta
import citeparse
import corpus
import encoder_device
import imagefit
import jev
import layout
import pagehit
import prompts
import providers
import queryembed
import xray

# This system degrades rather than fails, in six separate places: the encoder
# sidecar, local encoding, the answer cache, the text sidecar, citation
# geometry, and page resizing. Each fallback is correct — a slower answer beats
# no answer — but a silent one means nobody can tell which of them is active,
# and one of them (server-side encoding) is a 7x latency regression that looks
# exactly like a fast path from outside. Everything below logs on the way down.
log = logging.getLogger("pixelrag")


def configure_logging(level: str | None = None, stream=None) -> None:
    """Send pixelrag's warnings somewhere a human will see them.

    Called by the entry points (ask.py, app.py), not at import: a library that
    installs a root handler on import steals logging from whatever embeds it.
    PIXELRAG_LOG=debug turns up the detail; PIXELRAG_LOG=critical quiets it.
    """
    name = (level or os.environ.get("PIXELRAG_LOG", "warning")).strip().upper()
    root = logging.getLogger("pixelrag")
    root.setLevel(getattr(logging, name, logging.WARNING))
    if not root.handlers:
        h = logging.StreamHandler(stream)
        h.setFormatter(logging.Formatter("pixelrag: %(levelname)s: %(message)s"))
        root.addHandler(h)
    # Warnings are diagnostics about how the answer was produced; they must not
    # interleave with an answer being streamed to stdout.
    root.propagate = False

# One connection pool for the process. A search makes at least two HTTP calls
# (encode, then /search) and each used to open, use and discard its own socket:
# a handshake and a lingering TIME_WAIT per call, in the path being measured.
_HTTP = requests.Session()

SEARCH_API = "http://127.0.0.1:30001"
# Where the index lives, and the only thing that knows how it is laid out.
# Anchored to the repo root, so a CLI invoked from another directory still finds
# it — see layout.py.
# Where the index is lives in corpus.py — one owner, so redirecting it at a
# test index redirects every accessor rather than half of them.
INDEX_DIR = corpus.INDEX_DIR
TILES_DIR = corpus.TILES_DIR
MAX_STEPS = 10
# Ask modes — both PixelRAG-native:
#   oneshot  search → attach top page screenshots → one VLM call (default)
#   agent    multi-turn tool browse (slower, for hard navigation)
ASK_MODE = os.environ.get("PIXELRAG_ASK", "oneshot").strip().lower()
# Pages attached to the reader.
#
# The old justification — "recall plateaus at 4 (k=2 70%, k=4 85%, k=8 90%)" — was
# measured while search() silently clamped to 10 chunks, so the curve past ~5
# pages was an artefact of the cap, not of the index. Uncapped, recall@4 is 90%
# and recall@12 is 95%, and coverage@k keeps climbing well past 4 (28% -> 51%).
#
# 4 is kept as the default for image-token cost, not because the curve flattens:
# every extra page is a full-page JPEG in the prompt. Raise it for questions whose
# answer spans families ("pakiet antywłamaniowy" needs 7 pages); PIXELRAG_PER_QUERY
# follows automatically.
ONESHOT_PAGES = int(os.environ.get("PIXELRAG_ONESHOT_PAGES", "4"))
# Chunks pulled per query phrasing before aggregating to pages. 0 = derive it
# from ONESHOT_PAGES.
#
# This exists because raising ONESHOT_PAGES alone silently does nothing. Chunks
# are 6 per page, and several chunks of the same page routinely land in one
# result set, so 24 chunks collapse to about 5 distinct pages — measured on
# "pakiet antywłamaniowy", where asking for 12 pages returned 5 at any setting.
# The page count the reader sees is bounded by the CHUNK budget, not by k.
PER_QUERY = int(os.environ.get("PIXELRAG_PER_QUERY", "0"))


def _per_query(n_pages: int) -> int:
    """Chunk budget per phrasing. 6 chunks/page is the observed collapse rate."""
    return PER_QUERY or max(24, n_pages * 6)
# Fuse BM25 over the PDF text layer into page selection. PIXELRAG_HYBRID=1.
#
# OFF BY DEFAULT, and the reason is the experiment, not the measurement. Hybrid
# is strictly better on this question set (80% top-1 / 95% recall@4 against
# 50%/85% visual-only) — but this project exists to test one claim, "search any
# document by how it LOOKS, not just the text it contains". A default that lets
# BM25 supply most of the top-1 wins makes a good answer uninformative: you
# cannot tell whether the pixel index earned it. The thing under test has to be
# what runs by default, even when it scores worse.
#
# So: visual-only is the product, hybrid is the control. Flip it on to see the
# ceiling, and if this experiment graduates, flip the default with it.
HYBRID = os.environ.get("PIXELRAG_HYBRID", "0").strip().lower() in ("1", "true", "yes")
# Refuse without calling the reader when Jev says the answer is not in the
# retrieved pages. 0 = off (report the signal, act on nothing).
JEV_REFUSE = float(os.environ.get("PIXELRAG_JEV_REFUSE", "0") or 0)
# Characters of the page's own head prepended to every crop's text before it is
# reranked. 0 = off. A 875x1024 crop of a catalogue page usually does not say
# which product it belongs to — measured on the fabric colour pages, 4 of 32
# crops named their brand — so the reranker is handed a table with no owner.
CHUNK_CONTEXT = int(os.environ.get("PIXELRAG_CHUNK_CONTEXT", "0") or 0)

# Which backend answers, and what it costs, lives in providers.py — model names,
# thinking/effort settings, token accounting, image sizing and the SDK calls.
# Re-exported here because ask.py and the tests reach for them by these names.
detect_provider = providers.detect


def list_models(provider: str | None = None) -> list[str]:
    """What the configured key can actually reach — model names drift."""
    return providers.reader_for(provider).models()


# --------------------------------------------------------------------------
# index metadata
# --------------------------------------------------------------------------
#
# The accessors themselves live in corpus.py, and are called through it rather
# than imported by name. That is deliberate: `from corpus import articles`
# would bind a second reference here, and a test redirecting `corpus.articles`
# at a fake index would then reach corpus's callers but not this module's —
# half the system reading a temporary index and half reading the real one.
# One name, one owner, patched in one place.

def box_pct(article_id: int, tile_index: int, chunk_index: int) -> dict | None:
    """Chunk box as percentages of the page, for overlaying in the UI."""
    c = chunkmeta.get(article_id, tile_index, chunk_index, corpus.LAYOUT)
    if c is None or not c.has_box:
        return None
    pw, ph = corpus.page_size(article_id, tile_index)
    return {
        "left": 100 * c.x / pw, "top": 100 * c.y / ph,
        "width": 100 * c.width / pw, "height": 100 * c.height / ph,
    }


def _parse_pages(spec: str | None) -> dict[int, tuple[int, int]]:
    """'0:0-5,1:0-4' -> {0: (0, 5), 1: (0, 4)}"""
    out = {}
    for part in (spec or "").split(","):
        m = re.fullmatch(r"\s*(\d+):(\d+)-(\d+)\s*", part)
        if m:
            out[int(m[1])] = (int(m[2]), int(m[3]))
    return out


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def _raw_search(query: str, n_results: int = 5, timeout: int = 120) -> list[dict]:
    """Raw chunk hits from the faiss service. NOT clamped — see below.

    This used to clamp n to 10, which silently capped the whole retrieval stack:
    retrieve.py asked for 24 chunks per phrasing and got 10, so page aggregation
    only ever saw ~5 distinct pages and `PIXELRAG_ONESHOT_PAGES` above 5 was a
    no-op. Every measurement taken before 2026-07-31 — the 85% recall@4, the
    AGREEMENT and GIST_BONUS sweeps — ran under that cap.
    The service itself honours any n_docs (200 -> 200 hits, 106 pages).

    The clamp belongs on the agent-mode tool path, where hits become rows in an
    LLM prompt and a long list is a real cost. It does not belong here, where the
    caller is retrieve.py and more candidates are strictly better.
    """
    n = max(1, n_results)
    if queryembed.LOCAL_ENCODE:
        try:
            q = {"embedding": queryembed.embed_query(query)}
        except Exception:
            # The server encodes on one OpenMP thread — p95 1673ms against
            # 450ms here. Correct, and slow enough to be worth saying so.
            log.warning("local encode failed; falling back to server-side "
                        "encoding for this query", exc_info=True)
            q = {"text": query}
    else:
        q = {"text": query}
    r = _HTTP.post(f"{SEARCH_API}/search",
                   json={"queries": [q], "n_docs": n}, timeout=timeout)
    r.raise_for_status()
    return r.json()["results"][0]["hits"]


def _chunk_texts(hits: list[dict]) -> list[str]:
    """Text inside each candidate's crop, in PDF coordinates. One string per hit.

    Grouped by article because the per-hit version opened, mapped and closed the
    same catalogue once per candidate: 44 opens of an 82 MB file to answer one
    question, measured at 573ms of a 2258ms Jev retrieval. A quarter of the
    stage's latency was re-reading a file it already had open a moment earlier.

    A candidate whose article has no source PDF, or whose chunk carries no box,
    yields "" — and one empty string is enough to make jev.search skip the
    rerank, which is the intended behaviour and why this returns rather than
    raises.
    """
    import fitz

    out = [""] * len(hits)
    by_article: dict[int, list[int]] = {}
    for i, hit in enumerate(hits):
        by_article.setdefault(hit["article_id"], []).append(i)

    for aid, idxs in by_article.items():
        path = corpus.source_pdf(aid)
        if path is None:
            continue
        heads: dict[int, str] = {}
        with fitz.open(path) as pdf:
            for i in idxs:
                hit = hits[i]
                ti, ci = hit["tile_index"], hit["chunk_index"]
                chunk = chunkmeta.get(aid, ti, ci, corpus.LAYOUT)
                if chunk is None or not chunk.has_box:
                    continue
                pw, ph = corpus.page_size(aid, ti)
                page = pdf[ti]
                sx, sy = page.rect.width / pw, page.rect.height / ph
                text = page.get_text(clip=fitz.Rect(
                    chunk.x * sx, chunk.y * sy,
                    (chunk.x + chunk.width) * sx,
                    (chunk.y + chunk.height) * sy))
                if CHUNK_CONTEXT and text.strip():
                    if ti not in heads:
                        heads[ti] = " ".join(
                            page.get_text().split())[:CHUNK_CONTEXT]
                    text = f"[{heads[ti]}]\n{text}"
                out[i] = text
    return out


def _page_texts(pages: list[dict]) -> list[str]:
    """Full text of each attached page, one PDF open per article."""
    import fitz

    out = [""] * len(pages)
    by_article: dict[int, list[int]] = {}
    for i, page in enumerate(pages):
        by_article.setdefault(page["article_id"], []).append(i)

    for aid, idxs in by_article.items():
        path = corpus.source_pdf(aid)
        if path is None:
            continue
        with fitz.open(path) as pdf:
            for i in idxs:
                ti = pages[i]["tile_index"]
                if 0 <= ti < pdf.page_count:
                    out[i] = pdf[ti].get_text()
    return out


def _answerable_gate(question: str, pages: list[dict], emit) -> jev.Gate:
    """Ask whether the answer is in the pages about to be attached — and whether
    the question belongs to this corpus at all.

    Never raises: a failed check must not lose an answer the reader could still
    give. Returns an empty Gate when nothing could be judged, so callers read
    `.answerable is None` rather than distinguishing None from a Gate.
    """
    if not jev.enabled():
        return jev.Gate()
    started = time.perf_counter()
    report = jev.Report()
    try:
        result = jev.gate(question, _page_texts(pages), report)
    except Exception:
        log.warning("Jev answerability gate failed; answering anyway", exc_info=True)
        return jev.Gate()
    if result.answerable is None:
        return result
    emit({"type": "answerable", "value": result.answerable,
          "scope": result.scope, "pages": len(pages),
          "ms": round((time.perf_counter() - started) * 1000, 1),
          "jev": report.as_dict()})
    return result


def _chunk_text(hit: dict) -> str:
    """One candidate's crop text. The batch above is what production calls."""
    return _chunk_texts([hit])[0]


# --------------------------------------------------------------------------
# retrieval modes
# --------------------------------------------------------------------------
#
# Seven named ways to get from a question to a ranked list of pages:
#
#   visual      the pixel index alone — the claim this project exists to test
#   hybrid      visual + BM25 over the PDF text layer, fused by rank (RRF)
#   jev-expand  Jev-chosen phrasings, then `visual` verbatim — no reranker
#   jev         visual candidates, Jev-expanded queries, Jev-reranked chunks
#   jev+hybrid  the same Jev pool, with BM25 page candidates poured into it
#   jev-page    `hybrid`, then Jev reranks the candidate PAGES on full text
#   xray        no retrieval at all — every page in the corpus, judged
#
# `jev-expand` exists to make one subtraction possible. Jev's two stages cost
# about 500 input tokens and about 11k respectively, and before this they could
# only be bought as a pair — so "is the reranker earning its bill" had no
# experiment. visual → jev-expand → jev isolates one stage per step.
#
# They are named — rather than derived at four call sites from two independent
# environment flags — because the UI now runs them against each other on one
# question. A mode you can compare has to be a value you can pass, and
# "whatever PIXELRAG_JEV was when this process started" is not one.
#
# `xray` is the odd one out and belongs in the list anyway: it is the control
# that says what the other five are FOR. It skips retrieval entirely and scores
# every page of the corpus in one request, so its recall is 1.0 by construction
# and any page a retrieval mode misses was missed by ranking, not by the model.
# Measured here: 38 pages, 31k input tokens, 1.3 s, $0.0013 — under a quarter of
# one reader call. It is a mode on a 38-page corpus and a benchmark on a large
# one; see xray.py for where it stops being either.
#
# `auto` is that environment default, and is what every existing caller gets.
RETRIEVAL_MODES = ("visual", "hybrid", "jev-expand", "jev", "jev+hybrid",
                   "jev-page", "xray")

# How deep BM25 is read when its pages are poured into the Jev candidate pool.
# The RRF path has its own depth (retrieve.FUSE_DEPTH) and this is not it: there
# the two lists are fused by rank, here the pages are just extra candidates for
# a reranker that will score them on their text anyway.
LEXICAL_POOL = 20


def default_retrieval() -> str:
    """The mode PIXELRAG_JEV and PIXELRAG_HYBRID ask for between them.

    THE JEV DEFAULT IS `jev-page`, AND IT USED TO BE `jev`. Measured by
    scripts/bench.py over the 13 questions whose gold page is decided by
    literal string match — so no mode is graded by its own model — at k=4:

        mode          top-1  recall     ms     $/q
        visual          62%     92%    194       0
        jev-expand      62%     92%    878  0.00004
        jev             77%     92%   1846  0.00064
        hybrid          85%     92%      6       0
        jev-page        92%     92%   1109  0.00088
        xray            92%    100%   1288  0.00242

    Two results decided this. `jev` — chunk reranking, the mode that was the
    default — is BEATEN BY `hybrid`, which is free and 300x faster: scoring
    875x1024 crop text loses to BM25 over the same pages. And `jev-expand`
    scores exactly what `visual` scores on every column, so the expansion stage
    buys nothing for 684 ms. Both are the same finding at different scales:
    recall here is 92-100% before Jev is called, so nothing Jev does to WIDEN
    the funnel can help, and everything it does to the text UNIT matters.
    Crops 77%, BM25 pages 85%, whole pages 92% — monotone in chunk size.

    `xray` is not the default despite tying on top-1 and winning recall,
    because it is the only mode whose cost scales with the corpus rather than
    with the question. It is the right default for 38 pages and the wrong one
    for 3,800; see xray.py.

    PIXELRAG_JEV=manual keeps the modes selectable without defaulting to them.
    """
    if jev.enabled() and not jev.declined():
        # `jev-page` already runs BM25 to build its candidate pool, so there is
        # no separate hybrid variant of it to pick between.
        return "jev-page"
    return "hybrid" if HYBRID else "visual"


def resolve_retrieval(mode: str | None) -> str:  # ubs:ignore — allowlist; no eval sink
    """Name a mode, or resolve `auto`/None to whatever the environment sets."""
    name = (mode or "auto").strip().lower()
    if name in ("", "auto", "default"):
        return default_retrieval()
    if name not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r} (expected auto or one "
                         f"of {', '.join(RETRIEVAL_MODES)}).")
    return name


def retrieval_blocked(mode: str) -> str | None:
    """Why `mode` cannot run here as named, or None.

    The two environment flags are deliberately asymmetric and this is where it
    shows. PIXELRAG_HYBRID only picks the DEFAULT — asking for hybrid explicitly
    turns BM25 on, because the only real question is whether the text sidecar
    exists. PIXELRAG_JEV=0 is a KILL SWITCH: it means "do not call TypeSafe with
    this corpus", and a dropdown in a browser must not be able to overrule it.
    """
    if mode in TYPESAFE_MODES and not jev.enabled():
        return ("TYPESAFE_API_KEY is not set" if not jev.configured()
                else "turned off by PIXELRAG_JEV")
    # `xray` reads the corpus out of the same text sidecar BM25 uses, so it has
    # the same prerequisite and, on a corpus of scans, the same answer: it
    # cannot run, and `visual` is the mode for that.
    if ("hybrid" in mode or mode == "xray") and _lexical_fn(required=True) is None:
        return "no BM25 text sidecar — run scripts/build_text_index.py"
    return None


def retrieval_modes() -> list[dict]:
    """Every mode with the reason it cannot run here. For the UI's picker."""
    return [{"mode": m, "blocked": retrieval_blocked(m), "default": m == default_retrieval()}
            for m in RETRIEVAL_MODES]


# Which modes spend TypeSafe tokens. Stated as a list rather than matched as a
# substring of the name, because `xray` pays and does not say "jev", and because
# a mode that is billed by accident of spelling is the kind of bug that only
# shows up on an invoice.
TYPESAFE_MODES = ("jev-expand", "jev", "jev+hybrid", "jev-page", "xray")


def _jev_active(mode: str) -> bool:
    """Does this mode call TypeSafe at all? True for every mode that bills."""
    return mode in TYPESAFE_MODES and jev.enabled()


def _jev_reranks(mode: str) -> bool:
    """Does it pay for a CHUNK rerank? `jev-expand`, `jev-page`, `xray` do not."""
    return _jev_active(mode) and mode not in ("jev-expand", "jev-page", "xray")


def new_stats() -> dict:
    """The mutable sink `searcher` records measurements into.

    Sets rather than counters for the two funnel numbers: the same chunk comes
    back from several phrasings and the same page from several chunks, and a
    running total of hits would report the pool as several times its real size.
    """
    return {"searches": [], "lexical": [], "chunks": set(), "pages": set()}


def _record(stats: dict | None, key: str, query: str, started: float,
            rows: list[dict]) -> None:
    """One retriever call, as measured. Lexical rows carry a 1-based `page` and
    no chunk at all, which is exactly the shape difference this flattens."""
    if stats is None:
        return
    stats[key].append({"query": query, "hits": len(rows),
                       "ms": round((time.perf_counter() - started) * 1000, 1)})
    for row in rows:
        page = (row["article_id"],
                row["tile_index"] if "tile_index" in row else row["page"] - 1)
        stats["pages"].add(page)
        if "chunk_index" in row:
            stats["chunks"].add((*page, row["chunk_index"]))


def searcher(mode: str = "auto", *, stats: dict | None = None,
             report: jev.Report | None = None, keep: int | None = None,
             timeout: int = 120):
    """The chunk-search function ONE retrieval mode runs on.

    Injected into retrieve.py — and used directly by the evals — so the mode is
    a property of the function rather than of the process. Given a `new_stats()`
    sink, every call records what it returned and what it took; the comparison
    table's chunk counts and search timings all come from here, which keeps
    retrieve.py free of measurement it has no other reason to carry.
    """
    mode = resolve_retrieval(mode)  # ubs:ignore — allowlist; no eval sink

    def raw(query: str, n: int) -> list[dict]:
        started = time.perf_counter()
        hits = _raw_search(query, n, timeout)
        _record(stats, "searches", query, started, hits)
        return hits

    if not _jev_reranks(mode):
        # jev-expand searches with the plain retriever; its Jev call happens in
        # retrieve.py's variants_fn, once, before any of this runs.
        return raw

    pool_lexical = _lexical_fn(required=True) if "hybrid" in mode else None

    def pool(query: str, n: int) -> list[dict]:
        """Jev's candidate pool: chunk hits, plus whole pages BM25 nominated.

        A lexical page enters as chunk 0 with score 0.0 — it has no cosine and
        must not pretend to one. Jev scores it on its text like every other
        candidate, which is the only route by which a page the pixel index never
        saw reaches the reader in this mode.
        """
        hits = raw(query, n)
        if pool_lexical is not None:
            started = time.perf_counter()
            pages = pool_lexical(query, LEXICAL_POOL)
            _record(stats, "lexical", query, started, pages)
            for page in pages:
                hits.append({"article_id": page["article_id"],
                             "tile_index": page["page"] - 1,
                             "chunk_index": 0, "score": 0.0})
        return hits

    def jev_search(query: str, n: int) -> list[dict]:
        return jev.search(query, n, pool, _chunk_texts, report, keep)

    return jev_search


def search(query: str, n_results: int = 5, timeout: int = 120) -> list[dict]:
    """Chunk hits under whatever mode the environment configures — see `searcher`."""
    return searcher(timeout=timeout)(query, n_results)


def _timed_lexical(fn, stats: dict | None):
    """Wrap the BM25 retriever so the RRF path's call lands in the sink too."""
    if fn is None:
        return None

    def timed(query: str, n: int) -> list[dict]:
        started = time.perf_counter()
        rows = fn(query, n)
        _record(stats, "lexical", query, started, rows)
        return rows

    return timed


def retrieve_for(question: str, mode: str | None = "auto",
                 n_pages: int = ONESHOT_PAGES) -> tuple[list, list[dict], dict]:
    """Question -> ranked pages, under ONE named retrieval mode.

    Returns (pages, per-retriever debug rows, stats). The one-shot reader and
    the comparison endpoint both come through here, and that is the point: a
    mode the UI offers to compare has to be the same code that answers with it,
    or the comparison measures something nobody runs.
    """
    import retrieve

    mode = resolve_retrieval(mode)  # ubs:ignore — allowlist; no eval sink
    if mode == "xray":
        return _xray_pages(question, n_pages)
    reranks = _jev_reranks(mode)
    page_rerank = mode == "jev-page" and _jev_active(mode)
    report = jev.Report() if _jev_active(mode) else None
    stats = new_stats()
    started = time.perf_counter()
    # In jev+hybrid the text layer is already in the candidate pool, so fusing
    # it a second time here would let BM25 vote twice on the same evidence.
    lexical = (_lexical_fn(required=True)
               if ("hybrid" in mode or page_rerank) and not reranks else None)
    if reranks:
        # Jev expands the query itself and returns one already-ranked list, so
        # retrieve.py must not fan the question out a second time.
        variants_fn = lambda q: [q]
    elif _jev_active(mode) and not page_rerank:
        variants_fn = lambda q: jev.expanded_variants(q, report)
    else:
        variants_fn = None
    pages, debug = retrieve.retrieve_pages(
        # Keep every scored candidate: page aggregation ranks pages itself and
        # retrieve_pages takes the top n_pages at the end, so cutting the list
        # here only throws away evidence that was already paid for.
        searcher(mode, stats=stats, report=report, keep=jev.RERANK_POOL),
        question, _scale_of,
        # jev-page reranks AFTER aggregation, so it asks for the candidate
        # list, not the four pages that would otherwise survive the cut.
        n_pages=max(n_pages, jev.PAGE_POOL) if page_rerank else n_pages,
        per_query=_per_query(n_pages),
        lexical_fn=_timed_lexical(lexical, stats),
        variants_fn=variants_fn)
    if page_rerank:
        pages = _rerank_pages(question, pages, report)
    pages = pages[:n_pages]
    return pages, debug, retrieval_stats(mode, stats, report, debug,
                                         len(pages), started)


def _rerank_pages(question: str, pages: list, report) -> list:
    """Reorder candidate pages by Jev's score of each page's full text.

    A page is the unit the reader is given and the unit a heading belongs to.
    Scoring crops asks the model to judge a table with the title cropped off.
    Pages that cannot be scored keep their retrieval rank, below those that can.
    """
    if not pages:
        return pages
    started = time.perf_counter()
    try:
        texts = _page_texts([{"article_id": p.article_id, "tile_index": p.tile_index}
                             for p in pages])
        scores, _ = jev.score_pages(question, texts, report)
    except Exception:
        log.warning("Jev page reranking failed; keeping retrieval order", exc_info=True)
        if report is not None:
            report.note("page reranking failed — retrieval order kept")
        return pages
    if report is not None:
        report.stage("rerank", started)
    scored = [(s, i, p) for i, (s, p) in enumerate(zip(scores, pages)) if s is not None]
    blind = [p for s, p in zip(scores, pages) if s is None]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [p.with_(norm=s, rrf=None, score=s) for s, _, p in scored] + blind


def retrieval_stats(mode: str, stats: dict, report, debug: list[dict],
                    kept: int, started: float, sweep=None) -> dict:
    """What one retrieval did, stated once: the wire contract for the UI.

    Consumers: the `search` event (the trace line and its details panel) and
    compare.py's table. Every number counts something that actually happened,
    and the funnel reads in order — `chunk_hits` rows came back from `searches`
    queries, `unique_chunks` of them were distinct, `candidate_pages` pages were
    in play, `ranked_pages` survived aggregation and `pages` reached the reader.

    `unique_chunks` counts the PIXEL index only. `candidate_pages` can exceed
    the pages those chunks sit on, because BM25 nominates whole pages that have
    no chunk hit at all — which is the entire point of fusing it in, and would
    be invisible if the two were reported as one number.

    `search_ms` is a SUM over calls retrieve.py may have run CONCURRENTLY, so it
    exceeds `ms` (wall clock) whenever more than one phrasing was searched.
    Both are here because the first is what the mode costs the search service
    and the second is what the user waits for.

    `cost_usd` is retrieval money only — the reader's tokens are counted in the
    answer's own `usage`, and adding the two would double-count nothing but
    would hide which half moved. 0.0 means "this mode calls no paid API";
    None means "it does, and nobody told us the rate".
    """
    jev_stats = report.as_dict() if report is not None else None
    notes = list(jev_stats["fallbacks"]) if jev_stats else []
    if "hybrid" in mode and not stats["lexical"]:
        notes.append("BM25 did not run — no text sidecar")
    # A sweep that lost a shard still returns pages and still looks like a
    # sweep. Saying so here puts it on the trace line and in every comparison
    # row, next to the word "exhaustive" it has just stopped deserving.
    if sweep is not None and not sweep.exhaustive:
        notes.append(f"NOT exhaustive — {sweep.unjudged} of {sweep.pages} pages "
                     f"carry no score")
    lexical_fused = any(row["retriever"] == "lexical" for row in debug)
    return {
        "mode": mode,
        "reranker": "jev" if (jev_stats and jev_stats["reranked"]) else "none",
        "fusion": ("xray score (no retrieval)" if sweep is not None else
                   "rrf(visual+bm25)" if lexical_fused else
                   "jev score" if (jev_stats and jev_stats["reranked"])
                   else "score-norm(variants)"),
        "queries": [row["query"] for row in debug] or (
            jev_stats["variants"] if jev_stats else []),
        "searches": len(stats["searches"]),
        "chunk_hits": sum(row["hits"] for row in stats["searches"]),
        "unique_chunks": len(stats["chunks"]),
        "blind_chunks": (jev_stats or {}).get("blind", 0),
        "answerable": (jev_stats or {}).get("answerable"),
        "best_score": (jev_stats or {}).get("best_score"),
        "lexical_calls": len(stats["lexical"]),
        "lexical_hits": sum(row["hits"] for row in stats["lexical"]),
        "candidate_pages": len(stats["pages"]),
        "ranked_pages": max((row.get("pages", 0) for row in debug), default=0),
        "pages": kept,
        "ms": round((time.perf_counter() - started) * 1000, 1),
        "search_ms": round(sum(row["ms"] for row in stats["searches"]), 1),
        "lexical_ms": round(sum(row["ms"] for row in stats["lexical"]), 1),
        "jev": jev_stats,
        "xray": sweep.as_dict() if sweep is not None else None,
        "notes": notes,
        "cost_usd": jev_stats["cost_usd"] if jev_stats else 0.0,
    }


def _xray_pages(question: str, n_pages: int) -> tuple[list, list[dict], dict]:
    """`retrieve_for` for the mode that does not retrieve.

    There is no encoder, no faiss call, no BM25 query and no fusion here: the
    corpus text is read straight off the sidecar and every page of it is judged
    against the question in one request. What comes back is the same
    (pages, debug, stats) triple every other mode returns, so the reader, the
    comparison table and the UI cannot tell the difference — which is the point.
    Recall is 1.0 by construction, so the only thing a comparison row for this
    mode measures is ranking.

    The debug row calls itself `xray` rather than `visual`, because compare.py
    attributes results by that name and a sweep filed under `visual` would be a
    lie in the one table whose entire job is attribution.
    """
    import lexical

    started = time.perf_counter()
    corpus = lexical.pages()
    keys = [f"a{p['article_id']}:s{p['page']}" for p in corpus]
    sweep = xray.xray(question, list(zip(keys, (p["text"] for p in corpus))))
    by_key = {key: p for key, p in zip(keys, corpus)}
    ranked = [pagehit.from_xray(by_key[key]["article_id"], by_key[key]["page"], score)
              for key, score in sweep.ranked()]
    debug = [{"query": question, "retriever": "xray", "pages": len(ranked),
              "chunks": 0,
              "top": [{"article_id": p.article_id, "page": p.page,
                       "score": round(p.score, 4), "n_chunks": 0}
                      for p in ranked[:5]]}]
    stats = new_stats()
    stats["pages"] = {(p.article_id, p.tile_index) for p in ranked}
    kept = ranked[:n_pages]
    return kept, debug, retrieval_stats("xray", stats, sweep.report, debug,
                                        len(kept), started, sweep=sweep)


# Agent-mode tool result: every hit becomes a row the model reads, and the browse
# loop re-sends them each turn. 10 is the budget that used to live in search() and
# starve retrieve.py; applied here it costs nothing.
AGENT_SEARCH_HITS = 10


def _do_search(query: str, n_results: int = 5,
                retrieval: str | None = None) -> tuple[str, dict]:
    mode = resolve_retrieval(retrieval)  # ubs:ignore — allowlist; no eval sink
    report = jev.Report() if _jev_active(mode) else None
    stats = new_stats()
    started = time.perf_counter()
    hits = searcher(mode, stats=stats, report=report)(
        query, max(1, min(n_results, AGENT_SEARCH_HITS)))
    rows = []
    for h in hits:
        # Expose only `page`/`region`, the exact names and numbering
        # pixelrag_tile accepts. Leaking the 0-based tile_index alongside a
        # 1-based page invites the model to pass one where the other is meant.
        rows.append({
            "article_id": h["article_id"],
            "document": corpus.doc_title(h["article_id"]),
            "page": h["tile_index"] + 1,
            "region": h["chunk_index"],
            "score": round(h["score"], 4),
            "available": _pages_1based(h.get("article_pages")),
        })
    # `hits` here are chunks, not pages: in agent mode the rows ARE what the
    # reader gets, so `pages` counts rows and the page funnel above it still
    # counts pages. The one-shot event means the same thing by the same names.
    event = {"type": "search", "query": query, "hits": rows,
             "stats": retrieval_stats(mode, stats, report, [], len(rows), started)}
    return json.dumps({"results": rows}, ensure_ascii=False), event


def _do_tile(article_id: int, tile_index: int, chunk_index: int) -> tuple[dict, dict]:
    """Fetch one region. Returns a provider-neutral result plus a UI event.

    Result is either {"ok": False, "message": str} or
    {"ok": True, "label": str, "image": bytes, "mime": str}; each backend
    formats that into its own tool-result shape.
    """
    chunks = chunkmeta.for_article(article_id, corpus.LAYOUT)
    if (tile_index, chunk_index) not in chunks:
        # Report in the same 1-based page numbering the tool accepts, so the
        # correction the model makes is directly usable.
        pages = sorted({t + 1 for t, _ in chunks})
        here = sorted(c for t, c in chunks if t == tile_index)
        msg = (f"No such region. Article {article_id} has pages {pages}; "
               + (f"page {tile_index + 1} has regions {here}."
                  if here else f"page {tile_index + 1} does not exist."))
        return ({"ok": False, "message": msg},
                {"type": "tile_error", "article_id": article_id,
                 "tile_index": tile_index, "chunk_index": chunk_index})

    url = f"{SEARCH_API}/tile/{article_id}/{tile_index}/{chunk_index}"
    resp = _HTTP.get(url, timeout=60)
    if not resp.ok or not resp.headers.get("content-type", "").startswith("image/"):
        return ({"ok": False, "message": f"Tile fetch failed: HTTP {resp.status_code}"},
                {"type": "tile_error", "article_id": article_id,
                 "tile_index": tile_index, "chunk_index": chunk_index})

    pw, ph = corpus.page_size(article_id, tile_index)
    event = {
        "type": "tile",
        "article_id": article_id,
        "document": corpus.doc_title(article_id),
        "page": tile_index + 1,
        "tile_index": tile_index,
        "chunk_index": chunk_index,
        "box": box_pct(article_id, tile_index, chunk_index),
        # Lets the UI reserve the right space before the full page JPEG loads,
        # so cards don't collapse and the overlays don't stack on one line.
        "page_w": pw,
        "page_h": ph,
    }
    result = {
        "ok": True,
        "label": f"{corpus.doc_title(article_id)} — page {tile_index + 1}, region {chunk_index}",
        "image": resp.content,
        "mime": resp.headers["content-type"].split(";")[0],
    }
    return result, event


# --------------------------------------------------------------------------
# agent loop
# --------------------------------------------------------------------------

def _pages_1based(spec: str | None) -> str:
    """'0:0-5,1:0-5' -> '1:0-5,2:0-5' so page numbers match everywhere."""
    parts = []
    for tile, (lo, hi) in sorted(_parse_pages(spec).items()):
        parts.append(f"{tile + 1}:{lo}-{hi}")
    return ",".join(parts)


def _dispatcher(retrieval: str | None = None):
    """Bind the agent's tools to one retrieval mode.

    A factory rather than a module-level function because the mode now arrives
    per request: a browse loop calls `pixelrag_search` several times and every
    one of them has to run the mode the caller asked for, not the process's.
    """

    def dispatch(name: str, args: dict) -> tuple[object, dict | None]:
        """Run one tool call. Returns (provider-neutral result, UI event)."""
        if name == "pixelrag_search":
            return _do_search(args["query"], args.get("n_results", 5), retrieval)
        if name == "pixelrag_tile":
            # `page` is 1-based on the wire; tiles are 0-indexed on disk. Clamp
            # rather than pass a negative index through to a confusing lookup.
            page = int(args["page"])
            if page < 1:
                return ({"ok": False,
                         "message": f"page must be 1 or greater (got {page}); "
                                    f"pages are numbered as in the citation."}, None)
            return _do_tile(int(args["article_id"]), page - 1, int(args["region"]))
        return f"Unknown tool {name}", None

    return dispatch


def _cache_namespace(provider: str, model: str, mode: str, retrieval: str) -> str:
    """Everything that changes the answer to the *same* question.

    Only the question is matched fuzzily. Model, page budget and which
    retrieval mode ran all move the answer, and a rebuilt index moves what the
    page numbers even refer to — so those are exact-match. A miss costs one
    reader call; a stale hit quotes last week's price list.

    `retrieval` is the resolved mode, which is why the old `hyb0/1` flag is gone
    — it is now one of four values that flag could not express. jev.namespace()
    stays on top of it because the Jev MODEL can change under a mode whose name
    did not.
    """
    return "|".join([
        provider, model, mode, f"ret:{retrieval}",
        f"k{ONESHOT_PAGES}", jev.namespace(),
        answer_cache.index_fingerprint(INDEX_DIR),
    ])


def _replay(cached: dict, emit, started: float) -> dict:
    """Return a cached answer, re-firing its trace so the UI still draws pages.

    Without the replay a cache hit would produce an answer citing pages the
    user never saw appear, which reads as a bug even though the answer is right.
    """
    result = dict(cached["result"])
    for ev in result.get("trace") or []:
        emit(ev)
    result["usage"] = {**result.get("usage", {}),
                       "input": 0, "output": 0, "thoughts": 0,
                       "cache_read": 0, "cache_write": 0, "cost_usd": 0.0}
    result["cached"] = True
    result["cache_similarity"] = round(cached["similarity"], 4)
    result["cached_from"] = cached["question"]
    # The stored timings are what the ORIGINAL answer took; keeping them would
    # make the cache look exactly as slow as the thing it exists to skip.
    result["timings"] = {"retrieval_ms": 0.0, "reader_ms": 0.0, "ttft_ms": None,
                         "total_ms": round((time.perf_counter() - started) * 1000, 1)}
    return result


def run_agent(question: str, on_event=None, max_steps: int = MAX_STEPS,
              provider: str | None = None, api_key: str | None = None,
              mode: str | None = None, retrieval: str | None = None) -> dict:
    """Answer a question over the visual index.

    mode (or PIXELRAG_ASK): oneshot | agent
      oneshot — PixelRAG retrieve + one VLM read (default)
      agent   — multi-turn tile browse

    retrieval (or auto): any of RETRIEVAL_MODES — visual | hybrid | jev-expand
    | jev | jev+hybrid | jev-page | xray. Resolved once here and carried down,
    so everything that runs under one answer — the cache key, both ask modes,
    every tool call in a browse loop — agrees on which mode produced it.
    """
    mode = (mode or ASK_MODE).strip().lower()
    if mode not in ("oneshot", "agent"):
        raise ValueError(f"Unknown ask mode {mode!r} (expected oneshot or agent).")
    retrieval = resolve_retrieval(retrieval)  # ubs:ignore — allowlist; no eval sink

    reader = providers.reader_for(provider, api_key)
    trace: list[dict] = []
    started = time.perf_counter()

    def emit(ev):
        trace.append(ev)
        if on_event:
            on_event(ev)

    # The question's embedding is needed by retrieval anyway ~140ms from now, so
    # asking for it here is free (it is memoised) and buys a lookup that can
    # skip the entire reader call. Agent mode is excluded: its value is the
    # browse trace, and replaying a canned one would be a lie.
    key = ns = None
    if mode == "oneshot":
        ns = _cache_namespace(reader.name, reader.model, mode, retrieval)
        try:
            key = queryembed.embed_query(question.strip())
        except Exception:
            # Encoder down: answer the slow way rather than fail, but the
            # answer cache is off for this question and nothing else says so.
            log.warning("could not embed the question — answer cache disabled "
                        "for this query", exc_info=True)
            key = None
        if key is not None:
            hit = answer_cache.default().get(key, ns)
            if hit is not None:
                return _replay(hit, emit, started)

    if mode == "oneshot":
        result = _read_once(reader, question, emit, trace, retrieval, started)
    else:
        result = _browse(reader, question, emit, trace, max_steps, retrieval, started)

    if key is not None and _worth_caching(result):
        try:
            answer_cache.default().put(key, question.strip(), ns, result)
        except Exception:
            # A cache write must not fail a good answer — but a cache that
            # never writes is a cache that never hits, which looks like nothing.
            log.warning("answer cache write failed", exc_info=True)
    return result


def _worth_caching(result: dict) -> bool:
    """Was this an answer, or a report that there was nothing to answer from?

    Never cache the latter: "no pages matched" is usually a transient
    search-service problem, and pinning it keeps answering that way.

    A non-empty `trace` is NOT that signal, though it was used as one. The
    search event fires before page selection and fires even when retrieval
    returned nothing, so the trace is never empty and the non-answer was cached
    every time — one search outage became a permanent wrong answer for every
    rephrasing in that namespace, until the index was rebuilt or the cache
    cleared by hand. A `tile` event is the real signal: it means at least one
    page actually reached the reader.
    """
    return bool(result.get("answer")) and not result.get("refused") and any(
        e.get("type") == "tile" for e in result.get("trace") or [])


def _read_once(reader, question: str, emit, trace: list[dict],
               retrieval: str, started: float) -> dict:
    """One-shot: retrieve pages, attach them, one streamed read.

    Provider-independent by construction — everything below this line is the
    same for every backend, and everything that is not lives in providers.py.
    """
    marks: dict[str, float] = {}
    pages, stats = _oneshot_pages(question, emit, reader.image_policy,
                                  retrieval=retrieval)
    marks["retrieval"] = time.perf_counter()
    if not pages:
        return _done(prompts.NO_PAGES_PL, trace, providers.Usage(), 1, reader,
                     retrieval, _timings(started, marks))

    # The gate judges the pages being attached, not the candidate pool the
    # reranker saw. In a Jev mode you get both, and the pair is a diagnosis:
    # high in the pool but low here means retrieval found it and ranking lost it.
    gate = _answerable_gate(question, pages, emit)
    marks["gate"] = time.perf_counter()
    verdict = gate.verdict(JEV_REFUSE) if JEV_REFUSE else "ok"
    if verdict != "ok":
        log.info("Jev gate refused (%s): answerable=%s scope=%s, threshold %.2f",
                 verdict, gate.answerable, gate.scope, JEV_REFUSE)
        emit({"type": "refused", "answerable": gate.answerable,
              "scope": gate.scope, "verdict": verdict, "threshold": JEV_REFUSE})
        result = _done(prompts.REFUSAL_PL[verdict], trace, providers.Usage(), 1, reader,
                       retrieval, _timings(started, marks))
        result["answerable"] = gate.answerable
        result["scope"] = gate.scope
        result["refused"] = verdict
        return result

    def on_text(piece: str) -> None:
        # First token, not first chunk of prose: setdefault is the whole point,
        # and it has to sit on the path the reader actually calls.
        marks.setdefault("first_token", time.perf_counter())
        _stream_answer(emit, piece)

    reply = reader.read_pages(
        system=prompts.ONESHOT_SYSTEM,
        preamble=prompts.oneshot_preamble(question, pages),
        pages=pages,
        header_of=prompts.page_header,
        on_text=on_text,
    )
    result = _done(reply.text, trace, reply.usage, reply.steps, reader,
                   retrieval, _timings(started, marks))
    result["answerable"] = gate.answerable
    result["scope"] = gate.scope
    return result


def _browse(reader, question: str, emit, trace: list[dict],
            max_steps: int, retrieval: str, started: float) -> dict:
    """Agent mode: the reader drives, opening regions until it can answer."""
    reply = reader.browse(
        system=prompts.SYSTEM, question=question, tools=prompts.TOOLS,
        dispatch=_dispatcher(retrieval), on_event=emit, max_steps=max_steps)
    # No retrieval or first-token mark: retrieval happens inside the loop, once
    # per tool call, and each of those search events carries its own stats.
    return _done(reply.text, trace, reply.usage, reply.steps, reader,
                 retrieval, _timings(started, {}))


def _timings(started: float, marks: dict) -> dict:
    """Wall clock, in ms, for the parts a person actually feels.

    `ttft_ms` is time to the first token of the answer, and it is the number a
    retrieval mode moves most: every millisecond retrieval spends is spent
    before the reader has said anything at all. None where the number does not
    exist for this path rather than 0, which would read as "instant".
    """
    now = time.perf_counter()

    def ms(mark):
        return None if mark is None else round((mark - started) * 1000, 1)

    retrieval, gate = marks.get("retrieval"), marks.get("gate")
    # The reader's clock starts after the gate, so a refusal reports the gate's
    # ~840ms as gate_ms and a reader_ms of 0 — not 840ms of a call never made.
    base = gate or retrieval
    return {"retrieval_ms": ms(retrieval),
            "gate_ms": None if (gate is None or retrieval is None)
                       else round((gate - retrieval) * 1000, 1),
            "reader_ms": None if base is None else round((now - base) * 1000, 1),
            "ttft_ms": ms(marks.get("first_token")),
            "total_ms": ms(now)}


def _lexical_fn(required: bool = False):
    """BM25 over the text layer, or None when there is no sidecar to read.

    Returns None rather than raising: a corpus of scans has no usable text layer
    and must still answer from pixels alone. Degrading to visual-only is the
    correct behaviour there, not an error.

    `required=True` means a caller NAMED a hybrid mode, so PIXELRAG_HYBRID — a
    default-picker, not a kill switch — has nothing to say about it. Only the
    sidecar's existence decides.
    """
    if not (required or HYBRID):
        return None
    try:
        import lexical

        if not lexical.TEXT_SIDECAR.exists():
            log.warning("PIXELRAG_HYBRID=1 but %s is missing — answering "
                        "visual-only; run scripts/build_text_index.py",
                        lexical.TEXT_SIDECAR)
            return None
        return lexical.search_text
    except ImportError:
        log.warning("PIXELRAG_HYBRID=1 but the lexical module could not be "
                    "imported — answering visual-only", exc_info=True)
        return None


def _scale_of(article_id: int, tile_index: int, chunk_index: int) -> str:
    """retrieve.py's ScaleFn, bound to this process's index."""
    return chunkmeta.scale_of(article_id, tile_index, chunk_index, corpus.LAYOUT)


def _hit_row(p: pagehit.PageHit) -> dict:
    """What every retrieved-page payload says about a page.

    `score` is whichever number ORDERS the list — the fused RRF score under
    hybrid, the aggregated cosine otherwise — and `raw_score` is always the
    retriever's own. They are separate keys because they are not comparable:
    visual scores are cosines in a 0.3-0.7 band and BM25 scores are unbounded
    near 8, so putting both in one column invites reading a text hit as ten
    times more confident than a pixel hit. `found_by` says which produced it.

    The matched region is NOT here: the search event calls it `region` and the
    tile event calls it `chunk_index`, and each adds its own name rather than
    shipping both for one value.
    """
    return {
        "article_id": p.article_id,
        "document": corpus.doc_title(p.article_id),
        "page": p.page,
        "score": round(p.ranking_score, 5),
        "raw_score": round(p.score, 4),
        "found_by": p.found_by,
        "n_chunks": p.n_chunks,
    }


def _oneshot_pages(question: str, emit, policy: imagefit.Policy,
                   n_pages: int = ONESHOT_PAGES,
                   retrieval: str | None = None) -> tuple[list[dict], dict]:
    """Hybrid search → best whole pages (retrieve, fuse, read pages).

    Page selection is delegated to retrieve.py, which aggregates chunk hits per
    page and fuses several phrasings of the question. Taking the top-k *chunks*
    instead — the previous behaviour — routinely spent all k slots on crops of
    one page while the answer page sat below the cut.

    Pure pixel retrieval by default — that is the claim under test. Set
    PIXELRAG_HYBRID=1 to fuse BM25 over the text layer in as a second retriever.
    Measured on eval/questions_pl.yaml the two miss disjoint sets: visual-only
    50% top-1 / 85% recall@4, BM25-only 75% / 80%, fused 80% / 95%. Pages the
    text index found alone carry no region box; citations.py already falls back
    to page-level highlighting for those.
    """
    pages_ranked, debug, stats = retrieve_for(question, retrieval, n_pages)

    emit({"type": "search", "query": question, "variants": debug, "stats": stats,
          "hits": [{**_hit_row(p), "region": p.focus} for p in pages_ranked]})

    pages: list[dict] = []
    for p in pages_ranked:
        path = corpus.page_path(p.article_id, p.tile_index)
        if not path.exists():
            continue
        pw, ph = corpus.page_size(p.article_id, p.tile_index)
        # Highlight the strongest *region* chunk. The gist chunk covers the whole
        # page, so boxing it would mark everything and mean nothing.
        box = (box_pct(p.article_id, p.tile_index, p.focus)
               if p.focus is not None else None)
        # Sized for whoever is about to read it. The rendered page is 200 DPI;
        # every provider bills a smaller number than that, and Gemini's is a
        # step function with a cliff just below A4 — see imagefit.
        img, mime = imagefit.fit(path, policy)

        # The wire contract, stated once. Consumers: ask.py's trace printer and
        # index.html's SSE handler. It carries no image bytes — the UI fetches
        # the page from /api/page, and base64ing a JPEG into an SSE frame would
        # put every retrieved page on the wire twice.
        event = {
            **_hit_row(p),
            "tile_index": p.tile_index,
            "chunk_index": p.focus,
            "box": box,
            # Geometry stays in ORIGINAL page pixels: box_pct and the UI's
            # highlight overlays are percentages of the rendered page, and the
            # reader's downscale must not leak into them.
            "page_w": pw,
            "page_h": ph,
        }
        emit({"type": "tile", **event})
        # What the reader gets, which is the event plus the pixels themselves.
        pages.append({**event, "image": img, "mime": mime,
                      "label": f"{corpus.doc_title(p.article_id)} — strona {p.page}"})
    return pages, stats


# --------------------------------------------------------------------------
# citations
# --------------------------------------------------------------------------

def _stream_answer(emit, text: str) -> None:
    """Push a fragment of the answer to the caller as it arrives.

    The reader spends seconds on a four-page read, and until it finished the UI
    had nothing to show but a spinner — the search and page events all fire in
    the first ~300ms. Streaming does not make the answer arrive sooner, it makes
    the first sentence arrive ~10x sooner, which is the number a user feels.

    Consumers that do not know this event ignore it and still get the final
    answer in the `done` payload, so this is additive.
    """
    if text:
        emit({"type": "answer_delta", "text": text})


def _done(answer: str, trace: list[dict], usage: providers.Usage, steps: int,
          reader, retrieval: str, timings: dict) -> dict:
    """Assemble the answer payload: prose, citations, cost, trace.

    Pricing is the reader's own business — an unpriced provider returns None
    rather than having a rate guessed for it here.
    """
    usage_out = usage.as_dict(reader.price(usage))
    provider, model = reader.name, reader.model

    # The reader's own citations, geometry resolved against the source PDF.
    body, raw_cites = citeparse.split_citations(answer or "")
    pages_seen = [{"article_id": e["article_id"], "page": e["page"],
                   "document": e["document"]}
                  for e in trace if e["type"] == "tile"]
    try:
        cites = citeparse.resolve_citations(body, raw_cites, pages_seen)
    except Exception:
        # Highlighting is a nicety; never fail the answer over it. It is still
        # the reader's own evidence, so losing it is worth a line.
        log.warning("could not resolve citations for this answer", exc_info=True)
        cites = []

    # Legacy number pins, kept so the agent mode (which has no citation block)
    # still marks the figures it quoted.
    pins: list[dict] = []
    if not cites:
        for aid, page in sorted({(e["article_id"], e["page"])
                                 for e in trace if e["type"] == "tile"}):
            try:
                for hit in citeparse.locate_values(aid, page, body):
                    pins.append({"article_id": aid, "page": page, **hit})
            except Exception:
                log.debug("number pinning failed for article %s page %s",
                          aid, page, exc_info=True)

    return {"answer": body, "trace": trace, "usage": usage_out, "steps": steps,
            "provider": provider, "model": model, "pins": pins,
            "citations": cites, "retrieval": retrieval, "timings": timings}
