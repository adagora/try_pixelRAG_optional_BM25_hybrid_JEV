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

## Retrieval modes

Which retriever finds the pages is separate from which reader answers. Five
named modes, pickable per question — in the UI header, with `ask.py
--retrieval`, or as `retrieval=` on `POST /api/ask`:

| Mode | What runs |
|---|---|
| `visual` | the pixel index alone — the claim this project exists to test |
| `hybrid` | visual + BM25 over the PDF text layer, fused by rank (RRF) |
| `jev-expand` | Jev-chosen phrasings, then `visual` verbatim — no reranker |
| `jev` | visual candidates, Jev-expanded queries, Jev-reranked chunks |
| `jev+hybrid` | the same Jev pool, with BM25 page candidates poured into it |
| `jev-page` | `hybrid`, then Jev reranks the candidate **pages** on full text |

The two Jev modes also return an **answerability** score: does the retrieved
content actually contain an answer, independent of how it ranked. Ranking always
produces a rank 1, so no score can say "nothing here answers this" — see
[Is the answer even in there?](#is-the-answer-even-in-there).

`jev-expand` exists to make one subtraction possible: Jev's two stages cost
about 500 input tokens and about 11k respectively, and they used to be
purchasable only as a pair, so "is the reranker earning its bill" had no
experiment. `visual → jev-expand → jev` isolates one stage per step.

`auto` (the default everywhere) resolves to `hybrid` where the text sidecar
exists and `visual` otherwise. **A TYPESAFE_API_KEY makes the Jev modes
selectable; it does not make one the default** — `PIXELRAG_JEV=1` does that.
Jev adds two sequential round trips (~1.9s against ~20ms) for a benefit one
verified question supports and no question set has measured, and the mode that
runs by default should be the measured one.

The flags are deliberately asymmetric in the other direction too: naming
`hybrid` explicitly turns BM25 on regardless of `PIXELRAG_HYBRID`, since the
only real question is whether the sidecar exists. `PIXELRAG_JEV=0` stays a kill
switch that a dropdown in a browser cannot overrule. A mode that cannot run here
stays in the list, greyed, with the reason.

The answer cache keys on the resolved mode, so switching modes re-answers
rather than replaying pages the new mode never retrieved.

### What each mode is actually made of

Every mode runs the same spine — **query → encode → faiss over pixel-crop
embeddings → aggregate crops into pages → top-`k` whole page JPEGs to the
reader**. Only two slots differ:

| mode | phrasings from | retrievers | reranker | Jev calls |
|---|---|---|---|---|
| `visual` | local regex strip | pixel faiss | — | 0 |
| `hybrid` | local regex strip | pixel faiss **+ BM25**, fused by RRF | — | 0 |
| `jev-expand` | **Jev** | pixel faiss | — | 1 |
| `jev` | **Jev** | pixel faiss | **Jev** | 2 |
| `jev+hybrid` | **Jev** | pixel faiss + BM25 **into the same pool** | **Jev** | 2 |

`hybrid` fuses BM25 as a second voter. `jev+hybrid` instead pours BM25's pages
into the candidate pool as extra chunks for Jev to score — so it is not "hybrid
plus jev", it is different wiring, and it is why BM25 is not also fused by RRF
there.

### What a "chunk" is — pixels, text, or both

Three different units travel under that word, and only the first is what the
chunk counts report:

1. **The retrieval unit is pixels.** A chunk is an **875x1024 crop** of a
   1654x2339 rendered page — 8 per page, 2 columns by 4 overlapping rows,
   embedded by the visual model. "27 chunks" in a comparison row means 27 image
   crops.
2. **BM25's unit is a whole page of text.** `index/text.json` holds one entry
   per page, not per chunk. In `hybrid` the pixel side votes with crops and the
   text side votes with entire pages.
3. **Jev's rerank unit is both.** The geometry is a pixel chunk's rectangle; the
   content is PDF text clipped to that box, capped at 6,000 characters.

**None of them reach the reader's context.** In one-shot mode the reader is sent
`PIXELRAG_ONESHOT_PAGES` **whole-page JPEGs** — never crops, never text — and
those images are the ~5,700 input tokens. Chunks only decide *which pages get
attached*, so the chunk counts are funnel width, not context size, and context
is the same in every mode. (Agent mode is the exception: `pixelrag_tile` hands
the reader individual crop images.)

Note for anyone reading `GIST_BONUS` in `retrieve.py`: **this index has no gist
chunks.** All 88 chunks in article 0 and all 213 in article 1 are `scale:
region`, so the whole-page vector that constant reweights does not exist here
and the constant is inert. Rebuilding with multiscale gist vectors is what would
make it — and any page-versus-region routing — mean anything.

### Comparing them

Tick **porównaj tryby** in the UI and ask a question: every available mode runs
its retrieval and reports back, side by side. **No reader is called**, so a
four-mode comparison costs four retrievals and zero image tokens; each row has
an "odpowiedz tym trybem" button when you want the answer from one of them.

Each row reports the funnel and the bill: pages returned, unique chunks pulled,
candidate pages considered, how many queries were issued, which reranker ran,
wall clock, and retrieval cost. `szczegóły` opens the rest — query variants,
the Jev facets with their scores, per-stage timings, per-call tokens, and each
retriever's own top-5 list. A page only one mode found is drawn with a dashed
border, and a closing "zgodność" card says which modes agreed on what.

Read the notes line first. Every Jev stage degrades quietly by design (failed
expansion, chunks with no PDF text, a rejected score), and a run that fell back
to the original retrieval is billed like one that worked and looks like one
everywhere else.

The same comparison from a terminal, including `--json`:

```sh
.venv/bin/python scripts/compare.py "Ile kosztuje brama Connect?"
.venv/bin/python scripts/compare.py --modes visual,jev --json "…"
```

`evaluate_pl.py` answers the other question — which mode is better across the
labelled question set, offline. This one answers what happened to *this*
question just now, which is what you need when one answer cited the wrong page:
the right page was either missing from the candidate pool, below the cut, or
there and reranked away, and those are three different bugs.

An answer's footer reports the same shape for the mode that produced it:
retrieval time, time to first token, total, and the reader's own tokens and
cost — split rather than summed, because retrieval is the part the mode changes
and the reader is the part that bills.

## Useful env

| Var | Default | Meaning |
|---|---|---|
| `PIXELRAG_HYBRID` | `0` | default-picks BM25 fusion (`1` = on); see "Retrieval modes" |
| `PIXELRAG_JEV` | `auto` | `1` makes a Jev mode the default; `auto` selectable-only; `0` kills it |
| `PIXELRAG_JEV_REFUSE` | `0` | refuse below this answerability, without paying for the reader |
| `PIXELRAG_JEV_DEPTH` | `24` | chunks pulled per phrasing before the rerank |
| `PIXELRAG_JEV_POOL` | `64` | chunks scored in one rerank call |
| `PIXELRAG_JEV_PAGES` | `16` | candidate pages scored by `jev-page` |
| `PIXELRAG_CHUNK_CONTEXT` | `0` | chars of the page's head prepended to each crop's text |
| `TYPESAFE_API_KEY` | — | enables `jev` / `jev+hybrid` |
| `TYPESAFE_MODEL` | `jev-latest` | Jev model |
| `TYPESAFE_PRICE_IN` / `_OUT` | `0.042` / `0` | USD per million tokens; defaults are the published jev-1.13 rate |
| `PIXELRAG_ONESHOT_PAGES` | `4` | pages attached to the reader |
| `PIXELRAG_ASK` | `oneshot` | `oneshot` \| `agent` |
| `GEMINI_MODEL` | `gemini-3.6-flash` | reader model |
| `PIXELRAG_PROVIDER` | auto | force `gemini` \| `anthropic` |
| `ANTHROPIC_EFFORT` | `medium` | thinking depth for the read; empty = API default |
| `PIXELRAG_ANSWER_CACHE` | `1` | reuse answers for near-identical questions |
| `PIXELRAG_CACHE_THRESHOLD` | `0.95` | cosine cut for a cache hit |
| `PIXELRAG_LOG` | `warning` | `debug`\|`info`\|`warning`\|`error` — see "When it degrades" |
| `PIXELRAG_INDEX_DIR` | `./index` | index tree; relative paths resolve against the repo root |
| `PIXELRAG_IMAGE_FIT` | `1` | downscale pages to the reader's billed resolution |
| `PIXELRAG_IMAGE_FLOOR` | `0.85` | how much shrink `imagefit` may spend chasing a cheaper tile grid |
| `PIXELRAG_ANTHROPIC_LONG_EDGE` | `1568` | Anthropic tier to target; `2576` sends high-res |

## Cost

The reader is the whole bill — retrieval is ~145ms of local CPU and free, in
every mode but the Jev ones, which add two TypeSafe calls per question.

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
exact-match namespace (model, page budget, retrieval mode, index fingerprint), so
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
| `scripts/rag.py` | orchestration: retrieve → read → cite (shared by CLI/UI) |
| `scripts/providers.py` | the model backends behind one `Reader` interface |
| `scripts/retrieve.py` | chunk→page aggregation, RRF hybrid — pure, no IO |
| `scripts/pagehit.py` | the retrieved-page record, shared by all three layers |
| `scripts/layout.py` | where the index is on disk, and how it is named |
| `scripts/chunkmeta.py` | chunk geometry and scale, read once |
| `scripts/imagefit.py` | downscale pages to what the reader actually bills |
| `scripts/answer_cache.py` | reuse answers for near-identical questions |
| `scripts/lexical.py` | BM25 over PDF text |
| `scripts/citations.py` | quote → highlight rectangles |
| `scripts/snip.py` | crop a page around a citation, for chat inline |
| `scripts/app.py` + `static/` | web UI |
| `tests/` | the pure logic, runnable with no services and no API key |
| `eval/` | place for a fresh question set when you rebuild one |

`scripts/evaluate.py` / `evaluate_pl.py` / `answer_all.py` expect YAML under
`eval/` — add questions there when you want measured recall again.

## Tests

```bash
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest
```

No search service, no encoder, no API key, no network. The answer pipeline runs
end to end against a `FakeReader` (see `tests/conftest.py`), so prompt assembly,
streaming, citations, cost and the answer cache are all covered without paying
for a read.

## When it degrades

The system falls back rather than failing — six times over. Each fallback is
correct and each one now says so:

| Symptom | What you get |
|---|---|
| encoder sidecar down | model loads in-process; first query pays ~30s (`INFO`) |
| local encode fails | server-side encoding, ~7x slower (`WARNING`) |
| question can't be embedded | answer cache off for that query (`WARNING`) |
| `PIXELRAG_HYBRID=1`, no `text.json` | answers visual-only (`WARNING`) |
| page can't be resized | sent at full resolution, billed accordingly (`WARNING`) |
| citations can't be resolved | answer stands, no highlights (`WARNING`) |

`PIXELRAG_LOG=info` or `scripts/ask.py --log info` to see them; the CLI sends
them to stderr so they never interleave with a streamed answer.

### Jev query expansion and chunk reranking

Set `TYPESAFE_API_KEY` in the project `.env` file or export it to enable the
TypeSafe Jev stage automatically. CLI and UI load `.env`; exported values take
precedence:

```sh
export TYPESAFE_API_KEY=...
.venv/bin/python scripts/ask.py "Ile kosztuje brama Connect?"
```

Both one-shot retrieval and agent searches select query expansions with Jev,
retrieve and deduplicate candidate chunks, then score their relevance to the
original question before selecting context for the reader. In `jev+hybrid`,
lexical page candidates enter that same scoring stage — a page BM25 nominated
enters the pool as chunk 0 with score 0.0 (it has no cosine and must not
pretend to one) and is judged on its text like any other candidate, which is
the only route by which a page the pixel index never saw reaches the reader.
BM25 is therefore *not* fused a second time by RRF in that mode: it would be
one retriever voting twice on the same evidence. The answer cache separates
every mode.

Jev's [typed API](https://docs.typesafe.ai/api) evaluates supplied alternatives;
it does not generate free-form rewrites. Expansion selects up to two relevant
Polish/English facets (price, dimensions, installation, specifications,
accessories, security), retaining the original query and local noun phrase.
Reranking uses PDF text clipped to each indexed chunk's geometry. It evaluates
up to 64 unique candidates (or the requested hit count if larger), with at most
6,000 characters per chunk, using a four-level relevance rubric. Original
retrieval order breaks score ties. The reader still receives page images.

`PIXELRAG_JEV=0` disables the stage; `TYPESAFE_MODEL` overrides `jev-latest`.
Requests time out after 20 seconds. Failed expansion uses local query variants;
failed reranking or candidates without extractable PDF text use the original
retrieval. Scanned PDFs therefore need a text layer for this stage.

Every one of those fallbacks is billed like a success and, apart from one
field, looks like one. The stage reports what it did — variants searched,
facets and their scores, candidates pooled, candidates scored, per-call tokens
and milliseconds, and the fallbacks it took — and the UI shows it in the search
trace and in every comparison row. Read the notes first.

Jev sends queries and candidate text to TypeSafe and incurs separate API usage,
which is *not* included in the reader's token/cost totals: the answer footer
bills the reader, the retrieval stats bill Jev. The rate is
[published](https://docs.typesafe.ai/models.md) — $0.042 per million input
tokens for jev-1.13, output free — so retrieval cost is shown in dollars;
`TYPESAFE_PRICE_IN` / `_OUT` override it with what your invoice actually says.
Cached answers skip retrieval entirely.

**The money is not the objection.** A full rerank of this index's candidate pool
is ~10,500 input tokens, or **$0.0004 a question** — against ~5,700 tokens of
page images to the reader, at roughly 7x the rate per token. Jev's price here is
latency, not dollars.

#### Is the answer even in there?

`jev` and `jev+hybrid` ask one extra `Noul` alongside the rerank — *do these
chunks contain the information needed to answer the query, for the exact product
the query names?* The chunk text is already in that request's state, so it costs
output tokens only: no extra round trip, no extra input tokens.

This is the signal ranking cannot give you. Sorting always yields a rank 1, so
`best_score` tells you how good the top chunk was relative to the others and
never that none of them answer the question. Measured over the candidate pool,
`k=4`:

| question | actually in the corpus? | `jev` | `jev+hybrid` |
|---|---|---|---|
| co to jest pergola? | yes, 27 pages | 52% | 71% |
| bramy roletowe kolory | yes | 91% | 91% |
| Ile kosztuje brama Connect? | **no — no prices at all** | **5%** | **4%** |
| Jak wymienić olej w silniku? | no | **1%** | **1%** |

Four for four, with clean separation. It is reported in the API (`answerable`),
on every comparison card, and in the search trace.

#### The gate: the same question, in front of the reader

Judging the candidate pool answers *"is it in the corpus"*. The decision worth
money is *"is it in the four pages I am about to spend a reader call on"*, so
the check also runs over the attached pages themselves, whatever mode retrieved
them — it needs only a key, not a Jev retrieval mode.

| | payload | guards |
|---|---|---|
| over the pool | 37 chunks, 5,517 tok, 780 ms | retrieval (20 ms, free) |
| **over the attached pages** | **4 pages, ~2,900 tok, ~840 ms** | **the reader (~6,200 ms, ~5,722 tok)** |

Measured against known ground truth, `hybrid` retrieval, 4 pages:

| question | answerable? | gate |
|---|---|---|
| co to jest pergola? | yes | 0.72 |
| bramy roletowe kolory | yes | 0.95 |
| Jak zamontować bramę roletową? | **no** — see below | 0.13 |
| Ile kosztuje brama Connect? | no, no prices in the index | 0.03 |
| Jak wymienić olej w silniku? | no | 0.01 |

Five for five, at $0.00012 and ~840 ms against a reader call worth ~6,200 ms.

The third row is the interesting one. `montaż` appears on four pages, so BM25
votes for them confidently — but every occurrence is a mounting *position*
("montaż przed otworem", "wymiary montażowe", "montaż natynkowy"), never an
installation *procedure*. A KARTA TECHNICZNA says what a product is, not how to
fit it. Keyword retrieval sees the word; the gate reads what it means.

In a Jev retrieval mode you get both numbers, and the pair is a diagnosis: high
over the pool and low at the gate means retrieval found the answer and ranking
lost it.

`PIXELRAG_JEV_REFUSE=<0..1>` refuses below that threshold without calling the
reader — **0 (off) by default**. It emits `answerable` and `refused` events
either way, so you can watch where it *would* have fired before trusting it.
A gated refusal is never written to the answer cache.

#### What that check found, on the first day it ran

**This index contains no prices.** Zero pages match `zł`, `netto`, `cena`,
`cennik`, `PLN` or `EUR`. Both documents are KARTA TECHNICZNA datasheets.

It is worse than that. Of six questions used to benchmark retrieval above, four
have no answer anywhere in this index:

| term | pages |
|---|---|
| `segmentow` (bramy segmentowe) | 0 |
| `wkładka` / `antywłaman` | 0 |
| `dzielony wał` | 0 |
| `Connect`, `SNP`, any price | 0 |
| `napęd` | 10 |
| `montaż` | 4 |

Those terms are not incidental. `retrieve.py` cites "Dzielony wał" and
"Prowadzenie pod kątem" as the queries that motivated noun-phrase fusion,
`rag.py` cites "pakiet antywłamaniowy" as the question needing 7 pages, and
`evaluate_pl.py` keys its ground truth to `CORPUS_DOC = "Cennik - Bramy
garażowe"` — a price list absent from this index.

**So the retrieval figures in `rag.py` (visual 50%/85%, BM25 75%/80%, fused
80%/95%) and the sweeps behind `AGREEMENT`, `GIST_BONUS` and `LEX_WEIGHT` were
measured against a corpus this index no longer contains.** They may well still
hold; nothing here reproduces them, and `evaluate_pl.py` cannot run at all —
with or without `--jev` — until that corpus and its question set come back.

A $0.0002 typed question surfaced this. Four turns of reading page lists,
comparing rankings and timing stages did not — because a retrieval mode looks
exactly the same whether it ranked the right pages or ranked noise.

#### Cost and speed per mode

Averages over 4 questions, `k=4`, this index:

| mode | Jev calls | chunks | candidate pages | Jev tokens | cost/question | answerability |
|---|---|---|---|---|---|---|
| `visual` | 0 | 27.2 | 10.8 | 0 | $0 | — |
| `hybrid` | 0 | 27.2 | **15.8** | 0 | $0 | — |
| `jev-expand` | 1 | 28.5 | 10.8 | 500 | $0.00002 | — |
| `jev` | 2 | 28.5 | 10.8 | 7,118 | $0.00030 | yes |
| `jev+hybrid` | 2 | 28.5 | **16.8** | 9,326 | $0.00039 | yes |

One thing that table says out loud: `hybrid` carries ~50% more candidate pages,
because BM25 nominates pages the pixel index never surfaced — that breadth is
the whole reason it exists.

`jev+hybrid` used to throw most of it away. The rerank returned only
`per_query` hits, so 16 BM25 pages collapsed to 6 candidates before aggregation
ever saw them — breadth paid for and discarded. It now returns everything it
scored (the candidates were already paid for; truncating first only destroys
evidence), and the same question carries 37 chunks into ranking and 15 candidate
pages instead of 6. Measured before/after, no extra tokens.

Not done, and for a reason: widening first-stage depth (`PIXELRAG_PER_QUERY`)
would push BM25's pages *out* of the Jev pool, because they are appended after
each variant's chunk hits and the pool is capped at `RERANK_POOL`. Widening
needs the pool to interleave by source first.

**Latency is a delta, not a column.** Modes run sequentially and the first one to
touch a question pays the query encode, so raw per-mode numbers from one run are
contaminated — a run showing `visual` at 946ms and `hybrid` at 17ms is measuring
cache order, not retrievers, since `hybrid` *is* `visual` plus BM25. What is
stable:

| | added latency |
|---|---|
| retrieval itself, encode warm | 10-300 ms |
| `+ jev-expand` | **+750 ms** — one round trip |
| `+ jev` / `jev+hybrid` | **+1900 ms** — two round trips |
| the reader call, for scale | ~6200 ms |

So Jev is +25-30% on an end-to-end answer and about a quarter of what the reader
costs in money. **Cost is noise here. Latency is the whole price**, and it is
two sequential round trips — sequential because the expansion decides which
queries run, which decides which chunks the reranker sees.

#### Can we just score a lot more chunks? — measured, and no

Output tokens are free and input runs at $0.042/M, so the obvious move is to
stop being careful about the candidate pool and let Jev filter a big one.
`PIXELRAG_JEV_DEPTH` (chunks per phrasing) and `PIXELRAG_JEV_POOL` (chunks
scored in one call) exist to try exactly that. Measured, `jev+hybrid`, against
the two questions with verified gold:

| depth/pool | scored | latency | input tokens | cost | pergola gold | kolory gold |
|---|---|---|---|---|---|---|
| 24/64 (default) | 48 | 2,552 ms | 9,303 | $0.00039 | rank 2 | rank 1 |
| 48/150 | 69 | 3,302 ms | 13,687 | $0.00057 | rank 2 | rank 1 |
| 100/320 | 112 | 3,462 ms | 25,176 | $0.00106 | **rank 3** | rank 1 |

The economics say yes: scoring 112 chunks — a third of this corpus's 301 — costs
a tenth of a cent, still less than the reader. Latency grows sub-linearly,
+900 ms for 2.3x the pool.

The results say no. Nothing improved and one gold page got *worse*, because at
38 pages first-stage recall is not the bottleneck — the answer is already in the
top 24, so a wider pool adds only distractors. At 112 candidates the ECHO cover
page overtook the characteristics page for "co to jest pergola?".

**Widening the pool pays when recall is the constraint. Here it is saturated.**
On a corpus where the right page routinely sits at rank 60, the same knobs
would read the other way — which is why they are knobs.

#### "kolory tkanin soltis" — what a crop cannot know

The sharpest case this corpus offers. Two fabric brands each have a colour
table, in the same layout, pages apart:

* gold — `a1:s21`, `a1:s22`: "Kolorystyka tkanin **Serge Ferrari Group**",
  SOLTIS 7635 / VEOZIP / 92
* distractors — `a1:s19`, `a1:s20`: "Kolorystyka tkanin **Copaco**" / Aliplast

Every mode retrieved all four pages. Only the ordering differed:

| mode | top 4 | latency |
|---|---|---|
| `hybrid` | **s21 s22** · s19 s20 | 163 ms |
| `jev` / `jev+hybrid` | s21 · **s20** · s22 · **s19** | ~1,950 ms |
| `jev-page` | **s22 s21** · s20 s19 | 1,137 ms |

Chunk-level reranking put a wrong-brand page between the two right ones. The
reason is not the model:

| page | brand | crops naming it |
|---|---|---|
| a1:s19 | Copaco | 1 of 8 |
| a1:s20 | Copaco | **0 of 8** |
| a1:s21 | SOLTIS | 1 of 8 |
| a1:s22 | SOLTIS | 2 of 8 |

**Four of thirty-two crops know which fabric they belong to.** The crop with the
most colour evidence on the SOLTIS page carries no brand name at all, and
neither does its counterpart on the Copaco page — as text they are nearly the
same object. The 875x1024 grid severs the heading from the table it heads, and
every stage downstream inherits that.

Given the same four pages *whole*, the same model separates them cleanly:

| rank | page | score |
|---|---|---|
| 1 | a1:s22 | **2.80**/3 |
| 2 | a1:s21 | **2.80**/3 |
| 3 | a1:s20 | 1.06/3 |
| 4 | a1:s19 | 1.06/3 |

So `hybrid` wins this question for a structural reason, not a smarter one: BM25
scores the whole page, and the whole page has its heading.

Two things came out of this, both partial:

**`PIXELRAG_CHUNK_CONTEXT=200`** prepends the page's own head to every crop
before reranking. Brand attribution goes from 4/32 crops to **18/32** — and the
ranking does not move, because the pages it cannot fix (`s20`, `s22`) are
continuation pages whose heads name no brand either. Fixing those needs section
carry-over, i.e. real structure, not a prefix.

**`jev-page`** reranks the candidate *pages* instead of the crops, which is also
about half the latency (one call, ~16 pages, no expansion). Scored against every
question here with verified gold it is **1 win, 1 loss, 1 tie**: it gets the
Soltis ordering right where chunk reranking does not, and drops "co to jest
pergola?" from rank 1 to rank 3 — there a single crop of the characteristics
page is a sharper match than the whole page diluted by everything else on it.

Which is the honest summary of the whole question: **chunks give precision about
where an answer sits, pages give attribution about what it is about, and neither
subsumes the other.** Three questions cannot choose between them.

#### Does the reranker earn its bill? — still open

The reranker is not a no-op: `jev` and `jev-expand` disagreed on 6/6 questions.
It won the one case with verified ground truth — "co to jest pergola?", where it
moved the definitional page from rank 3 to rank 1 and every other mode left it
at 3 or missed it. Expansion alone moved the page set on 2/6 and, given a colour
facet, spent the reader's page budget on the wrong catalogue.

That is one verified win, no verified losses, and a great deal of unverifiable
churn. It stays open until there is a question set whose answers are in this
index.

#### A single unreadable crop used to abort the whole stage

Measured before the fix: one blind candidate out of 27-41 skipped the rerank on
1 question in 6 (`jev`) and 2 in 6 (`jev+hybrid`) — the `expand` call still
billed, cosine order still served, under a mode named after a reranker that never
ran. Blind candidates are now dropped from the rerank instead and counted in
`blind_chunks`; all 6 questions rerank. The trade: a page with no text layer can
no longer be *promoted* by this stage, only ranked by the pixel index that found
it, which is what `visual` is for.
