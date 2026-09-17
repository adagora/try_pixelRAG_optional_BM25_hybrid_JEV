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
import encoder_device
import imagefit
import jev
import layout
import pagehit
import providers

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
LAYOUT = layout.DEFAULT
INDEX_DIR = LAYOUT.index_dir
TILES_DIR = LAYOUT.tiles_dir
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


SYSTEM = """You answer questions about a manufacturing company's own documentation
(gates, doors, and components) by *reading screenshot tiles* of its catalogues,
price lists, and technical manuals. Documents may be in Polish — answer in the
language the question was asked in.

You cannot see any document until you look at it. Never answer from memory or
from general knowledge about gates and pricing.

How to work (keep the loop short — every tool round trip re-sends every image):
1. Call pixelrag_search ONCE with a short descriptive query. Prefer the
   document's own vocabulary over the user's phrasing.
2. From the hits, pick the 2–4 regions you need and call pixelrag_tile for ALL
   of them in the SAME turn (parallel tool calls). Do not open one region,
   wait, then open another unless the first batch was wrong.
3. Never re-open a (page, region) you have already seen. If the cell is still
   unclear, open a *different* neighbouring region, or answer that you cannot
   read it confidently.
4. Answer as soon as the tiles show the figure. Typical price question: 1 search
   + 2–3 tiles + answer — about 3 round trips, not 8+.

Reading a tile: each tile is one region of a page, not a whole page. Search
results give you `available` — the valid page:region ranges per article, e.g.
"1:0-5,2:0-5" means page 1 has regions 0-5. Regions run left-to-right then
top-to-bottom, so on a two-column split region 0 is top-left, region 1 is
top-right, region 2 is middle-left, and so on. Use that to navigate deliberately.

Price matrices — the most common task, and the easiest to get wrong:
- These tables index gate HEIGHT down the left edge and gate WIDTH across the
  top, both as "up to and including" (Polish: "do") thresholds in mm. A
  2350 x 3050 mm gate uses the 2400 row and the 3100 column, not 2300/3000.
- The axis headers and the cell you want are usually in DIFFERENT regions.
  Open the header region and the cell region together in one turn, then count
  rows and columns across the two. Say which row and column header you landed
  on so the user can check you.
- Cells often carry TWO values: the unshaded one is the standard RAL colour
  price, the shaded one is the wood-decor (zloty dab, orzech) price. State which.
- Prices are net; VAT is added per the page footer. Say so when quoting.

If the tiles do not contain the answer, or the text is too small to read with
confidence, say exactly that. Never guess a price, dimension, or part number —
a wrong number quoted to a customer is expensive. Naming the digits you are
unsure about is far more useful than a confident wrong answer.

Be decisive: stop searching once you have the answer, and lead with it."""

# One-shot reader: images are already attached; no tools.
#
# Written against failures observed on price-list / options-table corpora,
# in rough order of how often they burned an answer:
#   1. quoting one price when the option is priced per product family
#   2. reading an options-table price out of the wrong family column
#   3. answering a question the corpus does not cover, from a nearby document
#   4. silently picking one drive/variant when the question named none
ONESHOT_SYSTEM = """Odpowiadasz na pytania o dokumentację techniczną i cenniki
producenta bram, drzwi i okien, czytając ZAŁĄCZONE ZRZUTY STRON (PixelRAG).

JĘZYK: odpowiadaj w języku pytania. Pytanie po polsku → odpowiedź po polsku.

DOWODY: obrazy są jedynym źródłem. Nigdy nie odpowiadaj z pamięci ani z ogólnej
wiedzy o bramach. Jeśli na stronach nie ma odpowiedzi — powiedz to wprost i
napisz, czego dokument nie zawiera. Nie zgaduj ceny, wymiaru ani numeru części:
błędna liczba podana klientowi jest kosztowna. Lepiej wskazać, których cyfr nie
jesteś pewien, niż podać pewną, ale złą wartość.

RODZINA PRODUKTU — najczęstsze źródło błędnych odpowiedzi:
Ta sama opcja ma różne ceny w różnych liniach produktowych (np. pakiet
antywłamaniowy RC2: +819 dla UniPro, +850 dla PRIME). Tabela opcji ma KILKA
KOLUMN CENOWYCH — nagłówki to rodziny (UniPro | SNP, SNP 2.0 | RenoSystem SSt |
RenoSystem SNP). Zawsze:
  • sprawdź, z której kolumny czytasz, i nazwij tę rodzinę w odpowiedzi;
  • jeśli pytanie nie wskazuje rodziny, podaj WSZYSTKIE dostępne ceny z
    etykietami, a nie jedną wybraną;
  • jeśli w danej kolumnie jest pusto — ta opcja nie jest dostępna dla tej
    rodziny. Napisz to, nie przenoś ceny z kolumny obok.

TABELE CENOWE (wymiary): wysokość otworu (Ho) w wierszach po lewej, szerokość
otworu (So) w kolumnach u góry — progi „do” w mm. Brama 2350 × 3050 mm to
wiersz 2400 i kolumna 3100, nie 2300/3000. Jedna strona może mieć kilka takich
tabel (brama ręczna, MOTO, METRO, SPARK) — powiedz, z której czytasz. Jeśli
pytanie nie wskazuje napędu, podaj wszystkie warianty. Zapis „2500x2100”
traktuj jako szerokość × wysokość, ale napisz, jak go zinterpretowałeś.
Odcienie szarości w komórkach oznaczają zakresy wykonania lub ograniczenia
kolorystyczne — sprawdź legendę pod tabelą, zanim zacytujesz taką komórkę.

CENY: netto, VAT wg przepisów (patrz stopka strony) — wspomnij o tym przy
każdej cenie. Podaj jednostkę dokładnie jak w dokumencie (za szt., za kpl.,
za m2, za mb., do ceny bramy).

„POKAŻ …”: pytanie o rysunek lub tabelę, nie o liczbę. Zacznij od tego, co
przedstawia strona i gdzie to jest (numer rysunku/tabeli, np. „Rys. 7”,
„Tab. 2”), potem podaj istotne dane. Strona i tak zostanie pokazana obok.

CYTATY — wymagane. Po odpowiedzi dodaj blok w dokładnie tym formacie:

---CYTATY---
{"page": 61, "quote": "Wkładka antywłamaniowa", "supports": "nazwa pozycji"}
{"page": 61, "quote": "Pakiet antywłamaniowy RC2", "supports": "pozycja 20"}

Zasady cytatów:
  • "page" — numer strony podany w nagłówku obrazu, dokładnie tak jak podano.
  • "quote" — tekst przepisany DOSŁOWNIE ze strony (nagłówek sekcji, nazwa
    pozycji w tabeli, etykieta wiersza). Służy do podświetlenia miejsca w
    dokumencie, więc musi istnieć na stronie znak w znak. Nie parafrazuj.
  • Cytuj etykietę wiersza/nagłówek, nie samą liczbę — liczby z odpowiedzi są
    lokalizowane automatycznie.
  • Jeden wiersz JSON na cytat, bez dodatkowego tekstu w bloku.

Zacznij od odpowiedzi. Bez wstępów."""


# --------------------------------------------------------------------------
# index metadata
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def articles() -> list[dict]:
    arts = json.loads(LAYOUT.articles_json.read_text(encoding="utf-8"))
    # Index may have been built on Windows (`pdfs\\foo.pdf`); Path on macOS/Linux
    # treats that as a single filename with a backslash, so the source is "missing".
    for a in arts:
        url = a.get("url")
        if isinstance(url, str):
            a["url"] = url.replace("\\", "/")
    return arts


def doc_title(article_id: int) -> str:
    a = articles()
    return a[article_id]["title"] if article_id < len(a) else "?"


def page_path(article_id: int, tile_index: int) -> Path:
    return LAYOUT.page_image(article_id, tile_index)


@lru_cache(maxsize=None)
def page_size(article_id: int, tile_index: int) -> tuple[int, int]:
    from PIL import Image

    with Image.open(page_path(article_id, tile_index)) as im:
        return im.size


def box_pct(article_id: int, tile_index: int, chunk_index: int) -> dict | None:
    """Chunk box as percentages of the page, for overlaying in the UI."""
    c = chunkmeta.get(article_id, tile_index, chunk_index, LAYOUT)
    if c is None or not c.has_box:
        return None
    pw, ph = page_size(article_id, tile_index)
    return {
        "left": 100 * c.x / pw, "top": 100 * c.y / ph,
        "width": 100 * c.width / pw, "height": 100 * c.height / ph,
    }


_NUM = re.compile(r"\d[\d   .,]{1,12}\d|\d{2,}")


def locate_values(article_id: int, page: int, answer: str) -> list[dict]:
    """Find where the numbers the model quoted actually sit on the page.

    The rendered tiles are pixels, but these PDFs still carry a text layer, so
    every figure in the answer can be located exactly rather than approximated
    by the region the model happened to open. Returns boxes as percentages of
    the page, same convention as box_pct.

    This is a *check on the model*, not a source of truth: a quoted number that
    cannot be found on the cited page is worth seeing.
    """
    src = Path(articles()[article_id].get("url") or "")
    if not src.exists() or src.suffix.lower() != ".pdf":
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

TOOLS = [
    {
        "name": "pixelrag_search",
        "description": (
            "Search the visual document index by text. Returns ranked tiles with "
            "their document, page, and `pages` — the article's valid tile:chunk "
            "ranges (e.g. '0:0-5,1:0-5' means page 0 has chunks 0-5). Call this "
            "first, then pixelrag_tile to actually read the content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Short descriptive query, in the document's own vocabulary."},
                "n_results": {"type": "integer", "description": "How many tiles to return (default 5, max 10)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "pixelrag_tile",
        "description": (
            "Look at one region of one page and read it. Returns that region as an "
            "image. `page` is the same 1-based number search results report and you "
            "cite. Regions run left-to-right then top-to-bottom, so on a two-column "
            "page region 0 is top-left, 1 is top-right, 2 is middle-left, and so on. "
            "Call this multiple times in one turn to open several regions at once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article_id": {"type": "integer", "description": "From search results."},
                # Deliberately 1-based to match the `page` field in search results and
                # in citations. An earlier 0-based `tile_index` invited the model to
                # pass the page number it had just been shown, and miss by one.
                "page": {"type": "integer", "description": "1-based page number, as reported by search."},
                "region": {"type": "integer", "description": "0-based region within that page."},
            },
            "required": ["article_id", "page", "region"],
        },
    },
]


# ---- query encoding -------------------------------------------------------
#
# `pixelrag serve` encodes the query itself, but it is forced to do so on ONE
# CPU thread: faiss and torch each ship their own libomp, and two OpenMP
# runtimes in one process are unsafe, so OMP_NUM_THREADS=1 is the price of
# having faiss there at all. That single thread is nearly all of search latency
# (server-side encode p95 1673ms), and profiling puts 98% of it in the forward.
#
# Nothing requires the encoder to live in the faiss process — /search accepts a
# precomputed `embedding`. Encoding here instead takes the same work to p95
# ~450ms on CPU threads and ~24ms on this box's CUDA GPU in fp16, graphed.
# Set PIXELRAG_LOCAL_ENCODE=0 to fall back to server-side.

LOCAL_ENCODE = os.environ.get("PIXELRAG_LOCAL_ENCODE", "1") != "0"
EMBED_MODEL = os.environ.get("PIXELRAG_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
# Prefer the sidecar (scripts/encoder.py) so the ~30s model load is paid once
# at its startup rather than per process, and one copy of the weights serves
# every client. Falls back to loading in-process if it isn't running.
ENCODER_URL = os.environ.get("PIXELRAG_ENCODER_URL", "http://127.0.0.1:8001")
_encoder: dict | None = None
_sidecar_ok: bool | None = None


def _sidecar_available() -> bool:
    global _sidecar_ok
    if _sidecar_ok is None:
        try:
            r = _HTTP.get(f"{ENCODER_URL}/health", timeout=2)
            _sidecar_ok = r.ok and r.json().get("status") == "ok"
        except requests.RequestException as e:
            _sidecar_ok = False
            log.info("encoder sidecar not reachable at %s (%s) — loading the "
                     "model in-process instead; first query pays ~30s",
                     ENCODER_URL, e)
    return _sidecar_ok


def _get_encoder() -> dict:
    """Load the query encoder once. ~28s here — weights, then graph capture.

    Never in the faiss process.
    """
    global _encoder
    if _encoder is None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dev, dtype = encoder_device.resolve(torch)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            EMBED_MODEL, dtype=dtype).eval().to(dev)
        processor = AutoProcessor.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        encoder_device.use_fast_tokenise(processor, EMBED_MODEL)
        encoder_device.graph_forward(torch, model, dev)
        _encoder = {"processor": processor, "model": model, "device": dev,
                    "torch": torch}
        # Record a graph per padding width before returning. Whoever asks first
        # would otherwise pay ~6s for whichever shape their query happens to
        # have, and this path is already the slow fallback — its costs belong at
        # load. Assigning _encoder first is what keeps this from recursing, and
        # calling _forward_pass rather than embed_query keeps it from queueing
        # onto the thread it is already running on.
        for width in encoder_device.warm_widths():
            _forward_pass("warmup", None, width)
    return _encoder


# pixelrag_serve wraps every query in this chat template before tokenising.
# Passing raw text instead changes the token sequence entirely and moves the
# embedding — measured as 1/10 identical rankings until this was replicated.
DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."


@lru_cache(maxsize=256)
def _embed_cached(text: str, instruction: str | None) -> tuple[float, ...]:
    """Memoised encode. Tuple so the cache cannot hand out a mutable list.

    The same text gets encoded more than once per question: retrieve.py searches
    the question and its noun phrase, and the answer cache needs the question's
    vector before either search runs. At ~140ms a call that adds up, and the
    encoder is deterministic for a given text.
    """
    if _sidecar_available():
        r = _HTTP.post(f"{ENCODER_URL}/embed",
                       json={"text": text, "instruction": instruction}, timeout=60)
        r.raise_for_status()
        return tuple(r.json()["embedding"])
    return tuple(_encode_local(text, instruction))


def embed_query(text: str, instruction: str | None = None) -> list[float]:
    """Encode as pixelrag_serve._encode_queries does.

    Same chat template and instruction, same base-model forward, same
    last-token pooling over last_hidden_state, same L2 normalisation. Any
    *structural* divergence puts the query in a different space from the indexed
    documents and silently degrades retrieval. The arithmetic does differ — fp16,
    padding and the CUDA graph together move an index score by up to 7.9e-04 at
    cos 0.999994, which check_parity.py exists to keep honest.
    """
    return list(_embed_cached(text, instruction))


def _encode_local(text: str, instruction: str | None = None) -> list[float]:
    """embed_query's in-process case, queued onto the thread that owns the model.

    Which thread runs the forward is not a detail here: it is the thread that
    recorded the CUDA graphs, and only that thread may replay them. See
    encoder_device.on_model_thread.
    """
    return encoder_device.on_model_thread(_forward_pass, text, instruction, 0)


def _forward_pass(text: str, instruction: str | None,
                  pad_width: int) -> list[float]:
    """The encode itself — same steps, same order as pixelrag_serve does them.

    Runs on the model thread. Every numerical decision it makes comes from
    encoder_device, so this path and the sidecar's cannot drift apart; pad_width
    is how _get_encoder records a graph for a length no query it has happens to
    produce.
    """
    enc = _get_encoder()
    torch, dev = enc["torch"], enc["device"]
    messages = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": DEFAULT_INSTRUCTION if instruction is None else instruction}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    prompt = enc["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = enc["processor"](text=[prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs, forward = encoder_device.prepare_forward(torch, enc["model"],
                                                     enc["processor"], inputs,
                                                     pad_width)
    with torch.no_grad():
        out = forward(**inputs)
    h = out.last_hidden_state
    idx = inputs["attention_mask"].sum(dim=1) - 1
    pooled = h[torch.arange(h.size(0), device=h.device), idx]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().float().numpy()[0].tolist()


def warm_encoder() -> None:
    """Pay the model load at startup rather than on a user's first question."""
    if LOCAL_ENCODE:
        try:
            embed_query("warmup")
        except Exception:
            log.warning("query encoder failed to warm up — queries will be "
                        "encoded server-side, which is ~7x slower "
                        "(p95 1673ms against 450ms)", exc_info=True)


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
    if LOCAL_ENCODE:
        try:
            q = {"embedding": embed_query(query)}
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
        path = _source_pdf(aid)
        if path is None:
            continue
        heads: dict[int, str] = {}
        with fitz.open(path) as pdf:
            for i in idxs:
                hit = hits[i]
                ti, ci = hit["tile_index"], hit["chunk_index"]
                chunk = chunkmeta.get(aid, ti, ci, LAYOUT)
                if chunk is None or not chunk.has_box:
                    continue
                pw, ph = page_size(aid, ti)
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
        path = _source_pdf(aid)
        if path is None:
            continue
        with fitz.open(path) as pdf:
            for i in idxs:
                ti = pages[i]["tile_index"]
                if 0 <= ti < pdf.page_count:
                    out[i] = pdf[ti].get_text()
    return out


def _answerable_gate(question: str, pages: list[dict], emit) -> float | None:
    """Ask whether the answer is in the pages about to be attached.

    Never raises: a failed check must not lose an answer the reader could still
    give. Returns None when the question was not asked or could not be judged.
    """
    if not jev.enabled():
        return None
    started = time.perf_counter()
    report = jev.Report()
    try:
        value = jev.answerable(question, _page_texts(pages), report)
    except Exception:
        log.warning("Jev answerability gate failed; answering anyway", exc_info=True)
        return None
    if value is None:
        return None
    emit({"type": "answerable", "value": value, "pages": len(pages),
          "ms": round((time.perf_counter() - started) * 1000, 1),
          "jev": report.as_dict()})
    return value


def _chunk_text(hit: dict) -> str:
    """One candidate's crop text. The batch above is what production calls."""
    return _chunk_texts([hit])[0]


# --------------------------------------------------------------------------
# retrieval modes
# --------------------------------------------------------------------------
#
# Four named ways to get from a question to a ranked list of pages:
#
#   visual      the pixel index alone — the claim this project exists to test
#   hybrid      visual + BM25 over the PDF text layer, fused by rank (RRF)
#   jev-expand  Jev-chosen phrasings, then `visual` verbatim — no reranker
#   jev         visual candidates, Jev-expanded queries, Jev-reranked chunks
#   jev+hybrid  the same Jev pool, with BM25 page candidates poured into it
#   jev-page    `hybrid`, then Jev reranks the candidate PAGES on full text
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
# `auto` is that environment default, and is what every existing caller gets.
RETRIEVAL_MODES = ("visual", "hybrid", "jev-expand", "jev", "jev+hybrid", "jev-page")

# How deep BM25 is read when its pages are poured into the Jev candidate pool.
# The RRF path has its own depth (retrieve.FUSE_DEPTH) and this is not it: there
# the two lists are fused by rank, here the pages are just extra candidates for
# a reranker that will score them on their text anyway.
LEXICAL_POOL = 20


def default_retrieval() -> str:
    """The mode PIXELRAG_JEV and PIXELRAG_HYBRID ask for between them.

    A key makes a Jev mode the default. That is thin evidence — two questions
    with verified gold, one of which `hybrid` misses entirely and `jev+hybrid`
    ranks second — but it is the only evidence there is, and it points this way.
    PIXELRAG_JEV=manual keeps the modes selectable without defaulting to them.
    """
    if jev.enabled() and not jev.declined():
        return "jev+hybrid" if HYBRID else "jev"
    return "hybrid" if HYBRID else "visual"


def resolve_retrieval(mode: str | None) -> str:
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
    if "jev" in mode and not jev.enabled():
        return ("TYPESAFE_API_KEY is not set" if not jev.configured()
                else "turned off by PIXELRAG_JEV")
    if "hybrid" in mode and _lexical_fn(required=True) is None:
        return "no BM25 text sidecar — run scripts/build_text_index.py"
    return None


def retrieval_modes() -> list[dict]:
    """Every mode with the reason it cannot run here. For the UI's picker."""
    return [{"mode": m, "blocked": retrieval_blocked(m), "default": m == default_retrieval()}
            for m in RETRIEVAL_MODES]


def _jev_active(mode: str) -> bool:
    """Does this mode call TypeSafe at all? True for every Jev mode."""
    return "jev" in mode and jev.enabled()


def _jev_reranks(mode: str) -> bool:
    """Does it pay for a CHUNK rerank? `jev-expand` and `jev-page` do not."""
    return _jev_active(mode) and mode not in ("jev-expand", "jev-page")


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
    mode = resolve_retrieval(mode)

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

    mode = resolve_retrieval(mode)
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
                    kept: int, started: float) -> dict:
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
    lexical_fused = any(row["retriever"] == "lexical" for row in debug)
    return {
        "mode": mode,
        "reranker": "jev" if (jev_stats and jev_stats["reranked"]) else "none",
        "fusion": ("rrf(visual+bm25)" if lexical_fused else
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
        "notes": notes,
        "cost_usd": jev_stats["cost_usd"] if jev_stats else 0.0,
    }


# Agent-mode tool result: every hit becomes a row the model reads, and the browse
# loop re-sends them each turn. 10 is the budget that used to live in search() and
# starve retrieve.py; applied here it costs nothing.
AGENT_SEARCH_HITS = 10


def _do_search(query: str, n_results: int = 5,
                retrieval: str | None = None) -> tuple[str, dict]:
    mode = resolve_retrieval(retrieval)
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
            "document": doc_title(h["article_id"]),
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
    chunks = chunkmeta.for_article(article_id, LAYOUT)
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

    pw, ph = page_size(article_id, tile_index)
    event = {
        "type": "tile",
        "article_id": article_id,
        "document": doc_title(article_id),
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
        "label": f"{doc_title(article_id)} — page {tile_index + 1}, region {chunk_index}",
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

    retrieval (or auto): visual | hybrid | jev | jev+hybrid — RETRIEVAL_MODES.
    Resolved once here and carried down, so everything that runs under one
    answer — the cache key, both ask modes, every tool call in a browse loop —
    agrees on which mode produced it.
    """
    mode = (mode or ASK_MODE).strip().lower()
    if mode not in ("oneshot", "agent"):
        raise ValueError(f"Unknown ask mode {mode!r} (expected oneshot or agent).")
    retrieval = resolve_retrieval(retrieval)

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
            key = embed_query(question.strip())
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
        return _done(NO_PAGES_PL, trace, providers.Usage(), 1, reader,
                     retrieval, _timings(started, marks))

    # The gate judges the pages being attached, not the candidate pool the
    # reranker saw. In a Jev mode you get both, and the pair is a diagnosis:
    # high in the pool but low here means retrieval found it and ranking lost it.
    gate = _answerable_gate(question, pages, emit)
    marks["gate"] = time.perf_counter()
    if JEV_REFUSE and gate is not None and gate < JEV_REFUSE:
        log.info("Jev answerability %.3f < %.3f — refusing without a reader call",
                 gate, JEV_REFUSE)
        emit({"type": "refused", "answerable": gate, "threshold": JEV_REFUSE})
        result = _done(NO_PAGES_PL, trace, providers.Usage(), 1, reader,
                       retrieval, _timings(started, marks))
        result["answerable"] = gate
        result["refused"] = True
        return result

    def on_text(piece: str) -> None:
        # First token, not first chunk of prose: setdefault is the whole point,
        # and it has to sit on the path the reader actually calls.
        marks.setdefault("first_token", time.perf_counter())
        _stream_answer(emit, piece)

    reply = reader.read_pages(
        system=ONESHOT_SYSTEM,
        preamble=_oneshot_preamble(question, pages),
        pages=pages,
        header_of=_page_header,
        on_text=on_text,
    )
    result = _done(reply.text, trace, reply.usage, reply.steps, reader,
                   retrieval, _timings(started, marks))
    result["answerable"] = gate
    return result


def _browse(reader, question: str, emit, trace: list[dict],
            max_steps: int, retrieval: str, started: float) -> dict:
    """Agent mode: the reader drives, opening regions until it can answer."""
    reply = reader.browse(
        system=SYSTEM, question=question, tools=TOOLS,
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
    return chunkmeta.scale_of(article_id, tile_index, chunk_index, LAYOUT)


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
        "document": doc_title(p.article_id),
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
        path = page_path(p.article_id, p.tile_index)
        if not path.exists():
            continue
        pw, ph = page_size(p.article_id, p.tile_index)
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
                      "label": f"{doc_title(p.article_id)} — strona {p.page}"})
    return pages, stats


# --------------------------------------------------------------------------
# citations
# --------------------------------------------------------------------------

CITE_MARK = "---CYTATY---"


def _source_pdf(article_id: int) -> Path | None:
    src = Path(articles()[article_id].get("url") or "")
    return src if src.exists() and src.suffix.lower() == ".pdf" else None


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
    import citations as C

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
        pdf = _source_pdf(target["article_id"])
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
        pdf = _source_pdf(aid)
        if not pdf:
            continue
        pinned, repeated = C.locate_numbers(pdf, pno, answer)
        for c in out:
            if c["article_id"] == aid and c["page"] == pno:
                c["numbers"] = pinned
                c["repeated"] = repeated
                break
    return out


NO_PAGES_PL = (
    "Nie znalazłem w zaindeksowanych dokumentach żadnej strony pasującej do tego "
    "pytania, więc nie mogę na nie odpowiedzieć. Sprawdź, czy dokument na ten "
    "temat jest w katalogu pdfs/ i czy indeks został przebudowany."
)


def _page_header(p: dict) -> str:
    """Label above each attached image.

    The page number here is what the reader must echo back in its citation
    block, so it is stated once, unambiguously, in the same form we parse.
    """
    return f"\n[{p['document']} — strona {p['page']}]  (page={p['page']})"


def _oneshot_preamble(question: str, pages: list[dict]) -> str:
    """User-turn framing: the question, the page inventory, and the ask.

    Listing which pages are attached (and how many regions of each matched)
    matters for the family-disambiguation rule: a reader that can see it was
    given three pages from one catalogue knows to check whether they are
    different product lines rather than assuming one answer.
    """
    inventory = "\n".join(
        f"  • strona {p['page']} — {p['document']}"
        f" (dopasowane regiony: {p.get('n_chunks', 1)})"
        for p in pages
    )
    return (
        f"Pytanie: {question}\n\n"
        f"Poniżej {len(pages)} zrzut(y) stron znalezionych przez wyszukiwanie "
        f"wizualne:\n{inventory}\n\n"
        "Przeczytaj je i odpowiedz. Jeśli odpowiedzi tam nie ma — napisz to. "
        "Pamiętaj o bloku ---CYTATY--- na końcu."
    )


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
    body, raw_cites = split_citations(answer or "")
    pages_seen = [{"article_id": e["article_id"], "page": e["page"],
                   "document": e["document"]}
                  for e in trace if e["type"] == "tile"]
    try:
        cites = resolve_citations(body, raw_cites, pages_seen)
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
                for hit in locate_values(aid, page, body):
                    pins.append({"article_id": aid, "page": page, **hit})
            except Exception:
                log.debug("number pinning failed for article %s page %s",
                          aid, page, exc_info=True)

    return {"answer": body, "trace": trace, "usage": usage_out, "steps": steps,
            "provider": provider, "model": model, "pins": pins,
            "citations": cites, "retrieval": retrieval, "timings": timings}
