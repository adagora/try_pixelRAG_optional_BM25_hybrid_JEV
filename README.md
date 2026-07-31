# Visual RAG (PixelRAG)

Search technical PDFs by how pages **look** (drawings, tables, dimension callouts),
then answer from page screenshots with a VLM.

Thin layer over [PixelRAG](https://github.com/StarTrail-org/PixelRAG): index +
serve stay upstream; this repo adds page-level retrieval, optional BM25 hybrid,
citations, and a web UI.

![Web UI — answer with citations over the source PDF](playground1.png)

## PDFS source
https://www.wisniowski.pl/images/foldery-online/pl/k75-pl-ECHO-wisniowski/#page=24

## Setup

```bash
brew install poppler jpeg-turbo
uv venv --python 3.12
uv pip install 'pixelrag[index,serve,pdf]'
# plus whatever ask.py / app.py need (google-genai / anthropic, fastapi, …)
```

Put PDFs in `pdfs/`. Copy API keys into `.env` (`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`).

## Index

```bash
.venv/bin/python scripts/prepare_hires.py
.venv/bin/python scripts/build_index.py
.venv/bin/python scripts/build_text_index.py   # BM25 sidecar (hybrid)
```

## Run

```bash
# 1) FAISS search API
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/pixelrag serve \
  --index-dir ./index --tiles-dir ./index/tiles \
  --articles-json ./index/articles.json --port 30001 --device cpu

# 2) query-encoder sidecar (optional but much faster on Apple Silicon)
.venv/bin/python scripts/encoder.py

# 3) ask CLI or UI
.venv/bin/python scripts/ask.py "…"
.venv/bin/python scripts/app.py          # http://127.0.0.1:8000
```

## Ask modes

| `PIXELRAG_ASK` | Behaviour |
|---|---|
| `oneshot` (default) | retrieve top pages → one VLM read of page JPEGs |
| `agent` | multi-turn `search` / `tile` browse |

## Useful env

| Var | Default | Meaning |
|---|---|---|
| `PIXELRAG_HYBRID` | `0` | fuse BM25 text layer with visual retrieval (`1` = on) |
| `PIXELRAG_ONESHOT_PAGES` | `4` | pages attached to the reader |
| `PIXELRAG_ASK` | `oneshot` | `oneshot` \| `agent` |
| `GEMINI_MODEL` | `gemini-3.6-flash` | reader model |
| `PIXELRAG_PROVIDER` | auto | force `gemini` \| `anthropic` |
| `ANTHROPIC_EFFORT` | `medium` | thinking depth for the read; empty = API default |
| `PIXELRAG_ANSWER_CACHE` | `1` | reuse answers for near-identical questions |
| `PIXELRAG_CACHE_THRESHOLD` | `0.97` | cosine cut for a cache hit |
| `PIXELRAG_IMAGE_FIT` | `1` | downscale pages to the reader's billed resolution |
| `PIXELRAG_IMAGE_FLOOR` | `0.85` | how much shrink `imagefit` may spend chasing a cheaper tile grid |
| `PIXELRAG_ANTHROPIC_LONG_EDGE` | `1568` | Anthropic tier to target; `2576` sends high-res |

## Cost

The reader is the whole bill — retrieval is ~145ms of local CPU and free.

**Measured** on this index, `gemini-3.6-flash`, 4 pages attached:

| | input tokens | wall clock |
|---|---|---|
| cold question | 5725 | ~8.7 s |
| repeat question (answer cache) | **0** | **~0 ms** |

**Not measured** — no Anthropic credentials on this box. From the documented
`w*h/750` rule, `claude-opus-5` sits in the high-res tier (2576px) and bills a
200-DPI page at ~5158 tokens, which clamping to 1568px should roughly halve.
Confirm against a real `usage.input_tokens` before believing it: the same
reasoning applied to Gemini's published 768px-tile model predicted a 50% saving
and delivered **exactly zero** (5725 tokens either way — the model normalises
images before billing). Gemini tile-chasing is off by default for that reason;
`PIXELRAG_GEMINI_TILE_CHASING=1` re-enables it if you measure otherwise.

`scripts/imagefit.py` run directly prints the predicted table for your index —
predicted, not billed. `scripts/answer_cache.py` prints hit rate and contents;
`--clear` empties it. The cache keys on the question's embedding plus an
exact-match namespace (model, page budget, hybrid flag, index fingerprint), so
rebuilding the index or changing any of those misses rather than answering from
stale pages.

Measured cosine for the cache, same encoder as retrieval: identical question
1.00, reordered 0.97, politeness prefix 0.95, synonym swap 0.90, unrelated
question 0.24. The default `0.95` catches rewordings with a wide margin over
anything unrelated; drop to `0.90` to catch synonym swaps if your evals support
it.

### Transcribing pages to text does not pay here — measured

`scripts/transcribe.py` reads a page once into structured markdown so the reader
never re-reads the pixels. On `gemini-3.6-flash` it **costs more than it saves**:
transcripts averaged **1.25× the tokens of the image** across 9 pages, because
Gemini bills any page image at a flat ~1093 tokens while text scales with
density. A 21-column weight matrix cost 3707 tokens as markdown against 1093 as
pixels — and tables like that are precisely the pages that answer questions.

The tool is kept because the verdict is provider-specific: Anthropic bills a
200-DPI page at ~2318–5158 tokens by area, where even the worst transcript wins.
Run `--measure` before assuming either way. It is not wired into `rag.py`.

Visual-only is the default on purpose (the experiment). Hybrid usually ranks
better when PDFs have a text layer; turn it on with `PIXELRAG_HYBRID=1`.

## Layout

| Path | Role |
|---|---|
| `pdfs/` | source documents |
| `index/` | tiles, vectors, `text.json` |
| `scripts/rag.py` | retrieve + reader (shared by CLI/UI) |
| `scripts/retrieve.py` | chunk→page aggregation, RRF hybrid |
| `scripts/imagefit.py` | downscale pages to what the reader actually bills |
| `scripts/answer_cache.py` | reuse answers for near-identical questions |
| `scripts/lexical.py` | BM25 over PDF text |
| `scripts/citations.py` | quote → highlight rectangles |
| `scripts/app.py` + `static/` | web UI |
| `eval/` | place for a fresh question set when you rebuild one |

`scripts/evaluate.py` / `evaluate_pl.py` / `answer_all.py` expect YAML under
`eval/` — add questions there when you want measured recall again.
