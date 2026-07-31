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

Visual-only is the default on purpose (the experiment). Hybrid usually ranks
better when PDFs have a text layer; turn it on with `PIXELRAG_HYBRID=1`.

## Layout

| Path | Role |
|---|---|
| `pdfs/` | source documents |
| `index/` | tiles, vectors, `text.json` |
| `scripts/rag.py` | retrieve + reader (shared by CLI/UI) |
| `scripts/retrieve.py` | chunk→page aggregation, RRF hybrid |
| `scripts/lexical.py` | BM25 over PDF text |
| `scripts/citations.py` | quote → highlight rectangles |
| `scripts/app.py` + `static/` | web UI |
| `eval/` | place for a fresh question set when you rebuild one |

`scripts/evaluate.py` / `evaluate_pl.py` / `answer_all.py` expect YAML under
`eval/` — add questions there when you want measured recall again.
