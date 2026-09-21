# Visual RAG (PixelRAG)

Search technical PDFs by how pages **look** (drawings, tables, dimension callouts),
then answer from page screenshots with a VLM.

Thin layer over [PixelRAG](https://github.com/StarTrail-org/PixelRAG): index +
serve stay upstream; this repo adds page-level retrieval, optional BM25 hybrid,
citations, and a web UI.

# Jev experiments

[TypeSafe](https://docs.typesafe.ai/) / Jev latency-focused demos. Each app lives in its own top-level directory with its own README, TESTING.md and screenshots.

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

Which retriever finds the pages is separate from which reader answers. Seven
named modes, pickable per question — in the UI header, with `ask.py
--retrieval`, or as `retrieval=` on `POST /api/ask`:

| Mode | What runs | top-1 | recall |
|---|---|---|---|
| `visual` | the pixel index alone — the claim this project exists to test | 55% | 95% |
| `hybrid` | visual + BM25 over the PDF text layer, fused by rank (RRF) | **84%** | 97% |
| `jev-expand` | Jev-chosen phrasings, then `visual` verbatim — no reranker | 55% | 92% |
| `jev` | visual candidates, Jev-expanded queries, Jev-reranked chunks | 63% | 90% |
| `jev+hybrid` | the same Jev pool, with BM25 page candidates poured into it | 68% | 90% |
| `jev-page` | `hybrid`, then Jev reranks the candidate **pages** on full text | 87% | **100%** |
| `xray` | **no retrieval** — every page of the corpus, judged in one call | **92%** | **100%** |

top-1 is `scripts/bench.py --verified`, k=4, on the 38 questions whose gold page
is decided by literal string match rather than by a model. `jev-page` is the
default when a key is present; see
[Does the reranker earn its bill?](#does-the-reranker-earn-its-bill--the-page-one-may-the-chunk-one-does-not)
for why it is not `jev`, and why `jev-expand` is kept despite scoring what
`visual` scores.

**The one result in that table that is statistically resolved is the free
one**: `hybrid` beats `visual` 11-0 on the questions they disagree about,
p=0.001. `jev-page`'s 3-point lead over `hybrid` is not resolved — they split
nine discordant questions 5-4 — so the case for paying for it is the recall
column, not the top-1 one. `bench.py` prints Wilson intervals and an exact
McNemar for every pair;
[read the interval before the ordering](#read-the-interval-before-the-ordering).

The two Jev modes also return an **answerability** score: does the retrieved
content actually contain an answer, independent of how it ranked. Ranking always
produces a rank 1, so no score can say "nothing here answers this" — see
[Is the answer even in there?](#is-the-answer-even-in-there).

`jev-expand` exists to make one subtraction possible: Jev's two stages cost
about 500 input tokens and about 11k respectively, and they used to be
purchasable only as a pair, so "is the reranker earning its bill" had no
experiment. `visual → jev-expand → jev` isolates one stage per step — and the
subtraction has now been done. **`jev-expand` scores exactly what `visual`
scores**, so the expansion stage is measured dead weight and the mode is kept
only to keep proving it.

`auto` resolves to `jev-page` when a key is present, and to `hybrid`/`visual`
otherwise. It used to resolve to `jev`/`jev+hybrid`; it does not any more,
because `jev` is beaten by `hybrid` on the question set at 300x the latency.
`PIXELRAG_JEV=manual` keeps the Jev modes selectable without defaulting to them,
and `PIXELRAG_JEV=0` stops TypeSafe being called at all.

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
| `jev-page` | local regex strip | pixel faiss + BM25, fused by RRF | **Jev, on whole pages** | 1 |
| `xray` | — (none searched) | **none** — the text sidecar is read whole | **Jev, on every page** | 1 |

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
| `PIXELRAG_JEV_REFUSE` | `0` | refuse below this answerability, without paying for the reader; **0.3** if you turn it on |
| `PIXELRAG_JEV_DEPTH` | `24` | chunks pulled per phrasing before the rerank |
| `PIXELRAG_JEV_POOL` | `64` | chunks scored in one rerank call |
| `PIXELRAG_JEV_PAGES` | `16` | candidate pages scored by `jev-page` |
| `PIXELRAG_XRAY_BUDGET` | `48000` | input tokens per `xray` shard; under Jev's 64k window |
| `PIXELRAG_XRAY_SHARDS` | `8` | above this the corpus is too big to sweep and `xray` refuses |
| `PIXELRAG_XRAY_PAGE_CHARS` | `6000` | per-page truncation inside a sweep |
| `PIXELRAG_ORACLE_GOLD` | `0.66` | score at which a page becomes gold in the generated set |
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
| `scripts/corpus.py` | what is in the index: articles, page images, source PDFs |
| `scripts/queryembed.py` | question → vector: the sidecar, the model, the fallbacks |
| `scripts/citeparse.py` | the reader's `---CYTATY---` block → highlight rectangles |
| `scripts/prompts.py` | everything the reader is told: prompts, tools, refusal copy |
| `scripts/providers.py` | the model backends behind one `Reader` interface |
| `scripts/retrieve.py` | chunk→page aggregation, RRF hybrid — pure, no IO |
| `scripts/pagehit.py` | the retrieved-page record, shared by all three layers |
| `scripts/layout.py` | where the index is on disk, and how it is named |
| `scripts/chunkmeta.py` | chunk geometry and scale, read once |
| `scripts/imagefit.py` | downscale pages to what the reader actually bills |
| `scripts/answer_cache.py` | reuse answers for near-identical questions |
| `scripts/lexical.py` | BM25 over PDF text |
| `scripts/xray.py` | the whole corpus, every question, in one call |
| `scripts/oracle.py` | builds and labels `eval/questions_pl.yaml` |
| `scripts/bench.py` | every mode over that set; `--verified` for a non-circular referee |
| `scripts/sweep_ranking.py` | re-sweeps `AGREEMENT` / `LEX_WEIGHT` on cached hits |
| `scripts/calibration.py` | order-invariance and OOD calibration; the bar for any replacement |
| `scripts/citations.py` | quote → rectangles, pure geometry over a PDF |
| `scripts/snip.py` | crop a page around a citation, for chat inline |
| `scripts/app.py` + `static/` | web UI |
| `tests/` | the pure logic, runnable with no services and no API key |
| `eval/` | the generated question set (`scripts/oracle.py`) |

`evaluate_pl.py`, `answer_all.py`, `bench.py`, `sweep_ranking.py` and
`check_parity.py` all read `eval/questions_pl.yaml`. `scripts/oracle.py`
generates it from the corpus, `scripts/bench.py` reads it for a per-mode
comparison, and `evaluate_pl.py` runs against it for the first time since the
corpus behind its old ground truth was replaced.

## Tests

```bash
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest
```

No search service, no encoder, no API key, no network. The answer pipeline runs
end to end against a `FakeReader` (see `tests/conftest.py`), so prompt assembly,
streaming, citations, cost and the answer cache are all covered without paying
for a read.

Two of those files exist because the things they cover are the things a reader
cannot check: `tests/test_bench.py` pins the grading and the statistics behind
every figure quoted in this README — the benchmark that decides what the repo
claims was itself unasserted — and `tests/test_app.py` covers all eight routes
of the UI, including the 1-based page URLs, the RFC 5987 header a Polish file
name needs, and the SSE error paths that are the only way a stream can report a
failure after its 200.

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

##### One number could not tell two failures apart

The gate asked one Noul and routed on it. Measured over this index:

| question | `answerable` | `scope` |
|---|---|---|
| kolory tkanin soltis | 0.68 | 0.81 |
| Ile kosztuje brama Connect? | **0.04** | **0.64** |
| Jak wymienić olej w silniku? | **0.01** | **0.05** |

On answerability alone the last two rows are the same event — 0.04 against 0.01
is not a distinction anyone should route on. They are not the same event. One
asked a door catalogue for a price it does not print; the other asked a door
catalogue about engine oil. The first is a corpus that should be extended, and
the user should be told what *is* here; the second is a question that was never
going to work. `scope` is the only field that separates them, and it costs about
**forty input tokens**, because the page text is already in that request.

This is the RAG form of a result TypeSafe's own docs make about Choice: a
relative judgment always points at *something*, so it cannot report that
everything on offer is wrong. Answerability is absolute about the pages and
still cannot report that the whole corpus is the wrong one.

So `jev.gate()` returns both, and a refusal says which it was:

| verdict | what the user is told |
|---|---|
| `not-in-corpus` | these documents are about this, but do not carry the answer — you probably want a different document (a cennik, not a karta techniczna) |
| `out-of-scope` | this question is about another domain entirely |

`out-of-scope` is tested first and against a fixed 0.5, not against the caller's
threshold: tightening the answerability gate must not silently start
reclassifying near-misses as wrong-library.

Measured by `scripts/bench.py` over the **49** generated questions this corpus
cannot answer: **48/49 refused (98%)**. No amount of better retrieval reaches
that number, because ranking always returns a rank 1.

That was 10/10 on the narrow question set, and the one miss that appeared at
five times the negatives is the more useful number: a gate reported as perfect
on ten questions is a gate that has not been tested.

`PIXELRAG_JEV_REFUSE=<0..1>` refuses below that threshold without calling the
reader — **0 (off) by default**. It emits `answerable` and `refused` events
either way, so you can watch where it *would* have fired before trusting it.
A gated refusal is never written to the answer cache.

**If you turn it on, 0.3 is the number**, measured over 122 questions — see
[the operating point](#the-two-things-open-replications-do-not-reproduce--tested).
It stays off by default
because refusing to answer is the one failure a user cannot work around, and
that should be a decision someone makes rather than one they inherit.

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
measured against a corpus this index no longer contains.**

That has since been closed rather than disclaimed. `scripts/oracle.py` built a
question set for the corpus that *is* here, `evaluate_pl.py` runs against it,
and `scripts/sweep_ranking.py` re-ran both live constants — below. The prose
figures in `rag.py` remain unreproduced and are still to be read as history;
the constants are not.

#### The ranking constants, re-swept

`AGREEMENT` and `LEX_WEIGHT` are the two numbers in `retrieve.py` that change
what comes back. Neither touches retrieval — one reweights chunks a search
already returned, the other weights a BM25 list already fetched — so every
query is encoded once and each value replays `aggregate`/`fuse`/`rrf` over the
same hits. A 10-point sweep costs 13 searches, and no two values are compared
against different draws of anything.

```sh
.venv/bin/python scripts/sweep_ranking.py          # both, ~13 searches, $0
```

Only `visual` and `hybrid` are swept: they are the two modes graded fairly, so
a winner read out of the table is not a model agreeing with itself.

**On 13 questions this sweep said both constants were switches with wide
indifference bands and no setting worth arguing about. On 38 it says both are
mis-set.** That reversal is itself the most useful thing in this section, so
both readings are kept.

`AGREEMENT`, over `visual` (no BM25 in that mode), top-1 of 38:

| 0.0 | 0.05 | 0.1 | 0.15 | **0.18** | 0.2 | 0.25 | 0.3 | 0.35 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| **28** | 23 | 22 | 21 | **21** ← shipped | 20 | 19 | 19 | 18 | 18 |

Monotone decreasing, with no plateau anywhere: **turning agreement off is worth
7 top-1 questions and costs 1 of recall** (35/38 against 36/38). Head to head
it is **9-2, p=0.065** — short of resolved, and far past what justified setting
it in the first place. The narrow set showed -1 top-1 for +1 recall and called
it a wash; at three times the questions the trade is 7-for-1 and the wrong way
round.

`LEX_WEIGHT`, over `hybrid`, top-1 of 38:

| 0.0 | 0.25 | 0.5 | 0.75 | **1.0** | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| 21 | 28 | 29 | 32 | **32** ← shipped | **35** | **35** | **35** | **35** |

The shipped value sits on the shoulder, not the plateau. BM25 is being
under-weighted by exactly the amount the narrow set could not see.

Because a fused ranking mixes a list `AGREEMENT` orders with one `LEX_WEIGHT`
weights, they can interact, and `--joint` sweeps the grid (free — the hits are
already in memory):

```sh
.venv/bin/python scripts/sweep_ranking.py --joint
```

| | best cell | shipped |
|---|---|---|
| `AGREEMENT` | **0.0** | 0.18 |
| `LEX_WEIGHT` | **2.0** | 1.0 |
| `hybrid` top-1 | **36/38 (95%)** | 32/38 (84%) |

**Four questions, free, at no latency — and 36/38 is above `xray`'s 35/38**, the
most expensive mode in the repo. The `AGREEMENT=0` row dominates every other
row at every weight, so the direction is not one lucky cell.

**None of it is shipped, and this is the point.** Against the shipped pair the
best cell is **5-1, p=0.22**, and a 6-0 sweep is the smallest margin that
reaches p<0.05. Picking the best of 90 cells on 38 questions is exactly the
procedure that produced the 13-question conclusions this README has spent the
last section retracting. The finding is *"both constants are probably wrong and
worth another 100 questions to settle"*, not *"set them to 0.0 and 2.0"* — and
writing the second sentence is how the first one gets forgotten.

`GIST_BONUS` was not swept, because on this index it is not merely neutral but
inert, and that is checkable rather than measurable: **all 301 chunks are
`region`** (88 in article 0, 213 in article 1) and not one is a gist, so the
branch it weights never executes. A sweep would draw a flat line by
construction. Rebuilding with `scripts/chunk_multiscale.py` is what would make
it — and any page-versus-region routing — mean anything.

A $0.0002 typed question surfaced this. Four turns of reading page lists,
comparing rankings and timing stages did not — because a retrieval mode looks
exactly the same whether it ranked the right pages or ranked noise.

#### Cost and speed per mode

Averages over 4 questions, `k=4`, this index. **This table is about the shape
of the funnel** — how many chunks and candidate pages each mode carries — and
its cost column is a 4-question average kept for that context. For cost per
question, prefer the
[38-question benchmark](#does-the-reranker-earn-its-bill--the-page-one-may-the-chunk-one-does-not),
which runs the same modes over nine times the questions and puts `jev` at
$0.00082 rather than the $0.00030 below: longer questions pull more candidate
text into the rerank, and four questions do not sample that.

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

### `xray` — when the corpus is small enough not to retrieve at all

38 pages of this index are **31,298 input tokens**. Jev's window is 64k. So the
whole corpus fits in one request, and scoring every page of it against a query
costs **1.3 s and $0.0013** — under a quarter of one reader call (~6,200 ms,
~5,700 image tokens at roughly 7x the rate per token).

Which reframes the funnel: on a corpus this size, the entire retrieval pipeline
exists to avoid doing something cheaper than the call it is protecting.

Three properties follow, and they are why this is a module and not a flag:

1. **Recall stops being a variable.** Every page is scored, so no page can fall
   below a cut. Measured: 100% recall, the only mode that reaches it. `xray`
   cannot *miss* a page, only rank it badly — one failure mode instead of three.
2. **The scores are absolute.** A rubric level means the same thing in every
   request, unlike cosine or BM25 which only compare within one query. That is
   what makes sharding sound, and it makes the score distribution itself a
   signal: for a question this corpus cannot answer the top page scores
   **0.24/1** rather than 0.81/1 — a fact ranking destroys, since sorting always
   yields a rank 1.
3. **Extra judgments are nearly free.** Measured: seven more questions over the
   same state cost **293 input tokens total**, ~42 each — the tokens of the
   question text, because the state is sent once and output tokens are unbilled.

So the sweep asks what ranking could never afford to. Every `xray` query returns,
in the same request as the ranking:

| facet | what it is for |
|---|---|
| `answerable` | is the answer anywhere in the corpus |
| `scope` | is the question about this library at all |
| `premise` | does the corpus *contradict* something the query asserts |
| `injection` | is the query trying to steer the system rather than ask it |
| `kind` | price / spec / procedure / catalogue — what is being asked for |
| `granularity` | value / passage / survey — how many pages the answer needs |

**The ceiling is real and the module refuses to hide it.** 824 tokens per page
here means ~72 pages per request. Past that the corpus is sharded into
concurrent calls — same wall clock, multiplied bill — and past `MAX_SHARDS` it
refuses outright, because "exhaustive" is the only claim it makes and a partial
sweep is not a cheaper version of it. A 10,000-page corpus needs retrieval, and
that is what the other six modes are. `xray` is a mode at 38 pages and a
benchmark at 3,800.

A sweep that lost a shard still returns pages and still looks like a sweep, so
`exhaustive: false` and `unjudged` are in the stats, and the note lands on the
trace line and every comparison row.

### Where the question set comes from

Everything above depends on `eval/questions_pl.yaml`, which for most of this
repo's life did not exist — `evaluate_pl.py` could not run at all, and every
retrieval figure in it was measured against a corpus this index no longer
contains. `scripts/oracle.py` builds one in three stages, and the split between
them is the design:

1. **Mine, in code, free.** Headings are spans of page text, so finding them is
   string work. Running headers ("KARTA TECHNICZNA", on all 11 pages of article
   0) and wrapped sentences are dropped by rules — *"appears on 27 of 27 pages"*
   is a fact, and asking a model about it would be paying for arithmetic.
2. **Select the question, with Jev.** Four Polish phrasings per subject are
   generated locally; Jev picks which one the subject invites, and judges
   whether the subject is a real thing, specific enough to have one answer, and
   a physical product. No page text is sent — none is needed to judge a
   sentence.
3. **Label, with the X-ray.** Gold is whatever the *corpus* answers with, not
   the page the heading came from.

Run: `.venv/bin/python scripts/oracle.py --limit 150` — **114 questions in 53 s
for $0.21**, from 198 mined subjects, of which 84 are dropped by the four
checks below. 150 is the whole corpus: `subjects()` finds 150 page-specific
headings in these 38 pages and a higher limit mines nothing more.

The shipped set was 33 questions until it was 114, and
[that changed several of this README's conclusions](#read-the-interval-before-the-ordering)
— including one that had been stated as a negative result. Regenerating is
cheap; believing a 13-question benchmark is not.

**The negative class is discovered, not asserted.** For subjects Jev confirms
are physical products, the builder also emits `Ile kosztuje "X"?` and
`Jak zamontować "X"?` and lets the X-ray decide. On this corpus they come back
unanswerable — 49 of the 114 — which is the README's own day-one discovery
(*"this index contains no prices"*) reproduced automatically, per question, as
labelled data. They are the hardest negatives available: well-formed, in scope
(`scope` 0.84-0.95), about a product demonstrably in the corpus, and still
unanswerable.

Four checks keep the labels honest, and each exists because the first run got
something wrong:

| check | what it caught |
|---|---|
| both judgments must agree | `Ile kosztuje "WAGA PERGOLI WOLNOSTOJĄCEJ"?` — a weight table cleared the gold threshold in a corpus with no prices, while the Noul in the same response said 0.1 |
| gold must contain the subject | `HI MARINA HORIZON — jakie są parametry techniczne?` was labelled onto a garage-door dimensions table; the string occurs on exactly one page and it is not that one |
| the negative class is led by the Noul | a price question about a real product still scores "related topic" on its own pages, so the Score can never say "nothing answers this" and every probe landed in the unsure band |
| soft hyphens are normalised | the text layer carries U+00AD inside wrapped words, so a heading failed to match itself |

**What this oracle is not evidence for**, stated in the generated file itself
rather than a footnote: the labels come from Jev reading page text, so `jev`,
`jev-page` and `xray` are partly self-marked. `bench.py --verified` is the
answer — it keeps only questions whose subject string occurs on exactly one page
and makes that page the gold, so `str.__contains__` decides and no mode is
circular. **38 of 65 qualify, and the Jev oracle independently agreed with the
string on 35 of them.** That 35/38 is the only evidence in this repo that the
oracle is worth anything, and it is why the tables above quote `--verified`.
The agreement rate held exactly when the set tripled — it was 12/13 on the
narrow set, 92% either way — which is the check that the wider set is not
wider by being sloppier.

#### Does the reranker earn its bill? — the page one may, the chunk one does not

This was open for one reason — there was no question set. `scripts/oracle.py`
builds one (see [Where the question set comes from](#where-the-question-set-comes-from)),
`scripts/bench.py` runs every mode over it, and `--verified` restricts it to the
38 questions whose gold page is decided by **literal string match**, so no mode
is graded by its own model. At `k=4`:

| mode | top-1 | 95% CI | recall | cover | ms | $/q |
|---|---|---|---|---|---|---|
| `visual` | 55% | 40-70 | 95% | 95% | 170 | $0 |
| `jev-expand` | **55%** | 40-70 | 92% | 92% | 1069 | $0.00005 |
| `jev` | 63% | 47-77 | 90% | 90% | 1837 | $0.00082 |
| `jev+hybrid` | 68% | 52-81 | 90% | 90% | 1940 | $0.00101 |
| `hybrid` | **84%** | 70-93 | 97% | 97% | **6** | **$0** |
| `jev-page` | 87% | 73-94 | **100%** | **100%** | 1117 | $0.00117 |
| `xray` | **92%** | 79-97 | **100%** | **100%** | 1297 | $0.00313 |

##### Read the interval before the ordering

**38 questions means one question is 2.6 points**, and `bench.py` prints a
Wilson interval next to every rate and then tests each pair head to head on the
questions the two modes actually disagreed about. The ones they both got right
carry no information about which is better, and neither do the ones they both
got wrong. Exact McNemar, because the discordant counts are single digits and
no approximation survives that.

| pair | discordant | p | |
|---|---|---|---|
| `hybrid` vs `visual` | **11-0** | **0.001** | **resolved** |
| `xray` vs `jev-expand` | **15-1** | **0.001** | **resolved** |
| `xray` vs `jev` | **11-0** | **0.001** | **resolved** |
| `xray` vs `jev+hybrid` | **9-0** | **0.004** | **resolved** |
| `jev-page` vs `hybrid` | 5-4 | 1.000 | not resolved |

**This table was 13 questions until it was 38, and widening it changed the
answer.** What survived, what died, and what turned out to be noise:

**Pixel retrieval alone loses to free BM25, decisively.** `hybrid` 84% against
`visual` 55%, and the split is **11-0** — eleven questions where adding a text
retriever fixed the ranking, none where it broke it, p=0.001. At 13 questions
this pair was 4-0 and unresolved. It is the clearest result in the repo and it
is the one that costs nothing.

**`jev-page`'s lead over `hybrid` was noise.** At 13 questions it was 92%
against 85% and looked like the reason to make it the default. At 38 it is
33/38 against 32/38 — **5-4 across nine discordant questions**, a coin flip, at
187x the latency and $0.00117 a question against free. Three times the
questions turned a 7-point lead into one, and revealed that the two modes trade
wins rather than one dominating.

**But `jev-page` buys the column nobody was reading.** 100% recall and 100%
cover against `hybrid`'s 97%: it never loses a gold page, it just does not rank
better. That is a real property and a different one from the headline, and it
is the honest case for the mode.

**Chunk reranking got worse under scrutiny, not better.** `jev` and
`jev+hybrid` do not merely trail `hybrid` on top-1 — their **recall falls to
90%, below `visual`'s 95%**. The rerank is discarding gold pages the pixel
index had already found. That is a stronger claim than the 13-question set
could make, and it points the same way as the mechanism below.

With that carried, three findings, and two of them retire a stage:

**Expansion buys nothing, and at 38 questions it is worse than nothing.**
`jev-expand` scores exactly what `visual` scores on top-1 — 55% against 55%,
the same number for the second question set running, which is an identity
rather than a margin. But the wider set separates them where the narrow one
could not: **recall drops from 95% to 92%**. A Jev-chosen phrasing loses a gold
page the plain query found. The reason is visible in the funnel above it:
recall is already saturated before Jev is called, so there is nothing for a
wider query to recover and only something to lose. The README measured this
from the other end too — widening the candidate pool 2.3x also changed nothing,
because *"widening pays when recall is the constraint. Here it is saturated."*
Expansion is the same medicine for the same absent disease, with a side effect.

**Chunk reranking loses to BM25 and destroys recall doing it.** `jev` at 63%
and `jev+hybrid` at 68% against `hybrid` at 84%, and both at **90% recall —
below `visual`'s 95%**, which is the damning number: the rerank is throwing
away gold pages the pixel index had already retrieved. The mechanism was
measured separately and points the same way: the `kolory tkanin soltis` case
shows the 875x1024 grid severing headings from the tables they head, with only
4 of 32 crops naming the brand they belong to. Jev is being asked to judge an
artifact of the image grid, and it judges it badly enough to discard evidence.

**The unit is the whole finding.** Sorted by how much text travels together:

| unit given to the ranker | top-1 | of 38 | recall |
|---|---|---|---|
| 875x1024 crop text (`jev`) | 63% | 24 | 90% |
| whole page, BM25 (`hybrid`) | 84% | 32 | 97% |
| whole page, Jev (`jev-page`) | 87% | 33 | 100% |
| whole corpus, Jev (`xray`) | 92% | 35 | 100% |

Still monotone at three times the questions, and now with a resolved step in
it: crop text to whole page is 11-0 for the page. The middle two rungs remain
one question apart and are *not* separated — `hybrid` and `jev-page` trade
wins — so the ladder is really two groups, crops below and pages above, rather
than four ranks. **Jev is good exactly when it reads the unit a person would
read**, and pixel retrieval is what had been choosing that unit for it.

So `default_retrieval()` returns `jev-page` rather than `jev`/`jev+hybrid`.
`jev-expand` stays in the mode list, unblocked and never default, because it is
the control that proves the subtraction — a stage removed without a mode to
re-run is a claim, not a measurement.

**The case for that default changed when the question set widened, and it is
worth stating plainly rather than leaving the line above to imply the old
one.** `jev-page` was made the default because it ranked 7 points above
`hybrid`. It does not: at 38 questions that lead is one question and a 5-4
split. What it still has is **100% recall against 97%** — it never drops a gold
page. So the defensible reason to pay 187x the latency and $0.00117 a question
is recall, not ranking, and anyone who cares about neither should set
`PIXELRAG_JEV=manual` and run `hybrid`, which is free and 6 ms.

##### And a routing layer — this section used to say "would not help", and the data reversed it

At 13 questions the modes agreed almost everywhere: the ceiling for a perfect
oracle router was 12/13, exactly what `jev-page` scored alone, so there was
nothing to route. **That conclusion was an artifact of the sample size.**

At 38 questions `jev-page` and `hybrid` disagree about nine, and they split
them **5-4**. Which means:

| | questions |
|---|---|
| both right | 28 |
| `jev-page` only | **5** |
| `hybrid` only | **4** |
| neither | 1 |

A perfect router over just those two reaches **37/38 (97%)**, against 33/38 for
`jev-page` alone and 32/38 for `hybrid`. **Ten points of headroom, where the
narrow set showed none** — and the cheaper mode wins four of the nine, so the
routing is not "spend money when the question is hard".

That does not make a router the right next move; it makes it a real one, which
it demonstrably was not before. The prerequisite is a signal that predicts
*which* of the two will win a given question, and nothing here has looked for
one. Note also that this is the pair whose ceiling is cheapest to reach — one
of them is free — so the headroom is worth more than the same number would be
between two paid modes.

The general lesson is the one this whole section is now an example of: **a
negative result on 13 questions is a statement about 13 questions.**

#### The two things open replications do not reproduce — tested

The community result on open Jev reimplementations is that ~80% of the value
needs no training: read next-token logits over the allowed labels instead of
decoding JSON, and any open model does it. What replications are reported *not*
to reproduce is **OOD calibration** and **option-order robustness**.

That is not trivia for this repo, it is the invoice. Ranking — the half logits
commoditise — is worth +7 top-1 over free BM25 here. The gate — which runs
entirely on calibration — breaks even at 1 unanswerable question in 381. **The
half you could self-host is the half barely worth paying for; the half carrying
the return is precisely the half replications are said to miss.**

So `scripts/calibration.py` tests both claims directly:

```sh
.venv/bin/python scripts/calibration.py            # both, ~30 s, ~$0.05
```

**Claim 1 — option-order robustness.** Every page scored twice, the second time
with the rubric written backwards and the result mirrored back:

| | run 1 | run 2 |
|---|---|---|
| mean drift | 0.058 (**1.9%** of a 0-3 scale) | 0.062 (**2.1%**) |
| max drift | 0.39 | 0.32 |
| top-1 unchanged | 8/8 | 8/8 |
| top-3 set unchanged | 8/8 | 7/8 |

**Two runs, because this one is stochastic and a single sample of it would be
the same mistake as a 13-question benchmark.** The sampling moves the third
decimal and one top-3 set; it does not move the finding. Drift is ~2% either
way and the top-1 page never changed in sixteen paired sweeps.

If this drifted, every threshold in this repo would be an artifact of criteria
order — the gate, the oracle's gold cut, `jev-page`'s ranking. It does not.

**Claim 2 — OOD calibration.** All 114 generated questions plus 8 supplied
out-of-domain ones, judged against the whole corpus:

| declared | n | actually answerable |
|---|---|---|
| 0.0 – 0.2 | 56 | **0.00** |
| 0.2 – 0.4 | 1 | 0.00 |
| 0.4 – 0.6 | 2 | 1.00 |
| 0.6 – 0.8 | 27 | 1.00 |
| 0.8 – 1.0 | 36 | 1.00 |

| signal | graded against | AUC | medians |
|---|---|---|---|
| `answerable` | 65 answerable vs 57 not | **1.00** | 0.87 vs 0.06 |
| `scope` | 114 in-domain vs 8 out | **1.00** | 0.90 vs 0.02 |

**Perfect separation on both, at three times the questions.** This is the one
measurement in this README that survived widening the set unchanged — the
retrieval table did not, the ranking constants did not, and the `jev-page`
default did not. AUC 1.00 on 41 questions was a number to distrust; AUC 1.00 on
122, with 57 negatives instead of 18, is a result.

**The operating point, which is the number to set `PIXELRAG_JEV_REFUSE` from:**

| gate at | coverage | correct | wrongly refused |
|---|---|---|---|
| 0.1 | 72% | 74% | 0 |
| **0.3** | **53%** | **100%** | **0** |
| 0.5 | 53% | 100% | **1** |
| 0.7 | 41% | 100% | **15** |
| 0.9 | 22% | 100% | **38** |

**0.3 is the operating point — and it is now a point rather than a plateau.**
On the narrow set 0.3 and 0.5 were indistinguishable and the README said "0.3
to 0.5". With 57 negatives instead of 18, 0.5 turns away its first answerable
question while buying no coverage at all: identical 53% coverage, one customer
refused. Everything above that is worse for the same reason, steeply — 15
wrongly refused at 0.7. Set it to 0.3.

##### Why `scope` needed questions the oracle cannot write

The first run scored `scope` at **AUC 0.51 — a coin flip** — and that was the
test's fault, not the signal's. Every question in `eval/questions_pl.yaml` is
mined *from* this corpus, so all of them are in scope by construction, including
every one of the 49 negatives: they are price and installation questions about
products that are demonstrably here. Grading `scope` on answerability asks it to
separate a class that is not in the data.

`calibration.OUT_OF_DOMAIN` supplies the missing class — eight fluent, specific
Polish questions from other domains (engine oil, ZUS contributions, risotto).
With a negative class to separate, `scope` scores 1.00. Note the asymmetry the
widened set makes stark: 114 in-domain questions against 8 out. The oracle
generated the first number and a human wrote the second, and no amount of
regenerating changes that ratio. **An oracle that reads
the corpus cannot invent out-of-domain questions; they are the one input this
measurement needs a human for**, and leaving them out silently turns a working
signal into a failing number.

##### This is also the acceptance test for replacing Jev

Point `calibration.py` at any candidate — a local logit reader over an open
model, another vendor — and it answers "is this good enough to swap in" with the
same two numbers. The bar the hosted model set here is **AUC 1.00 on both
signals over 122 questions, and ~2% order drift with the top-1 page unmoved.**
That is what a replication has to clear before the ranking savings mean
anything — and the question count is part of the bar, because clearing it on 41
is most of a coin flip away from clearing it at all.

#### Does Jev work with pixel RAG at all?

Worth stating plainly, because the answer is partly no and the repo is named
after the part that is.

**Jev never sees a pixel.** It is [text-only](https://docs.typesafe.ai/models.md);
every call this repo makes sends `{"query": …}`, `{"query", "pages"}` or
`{"query", "chunks"}`. In a pixel-retrieval system Jev does not judge the
retrieved evidence — it judges a *text shadow* of it, extracted from the PDF
layer at coordinates the pixel index chose. Two consequences that the table
above is measuring without naming:

- The geometry is hostile. A crop rectangle is a good unit for an image encoder
  and a bad one for a reader, and `jev` at 63% top-1 with 90% recall — below
  what `visual` retrieved before the rerank touched it — is what that costs.
- On a corpus of scans there is no shadow at all and every Jev stage degrades
  to nothing, which is exactly the corpus `visual` exists for.

What Jev does add is the thing neither pixels nor BM25 can produce at any price:
a **typed judgment about the question** rather than a ranking of documents.
Answerability, scope, premise. The gate below refuses 48 of the 49 questions
this corpus cannot answer; no amount of better retrieval reaches that number,
because ranking always returns a rank 1.

**The short version: use Jev for the judgment layer and for page-level ranking;
do not use it to expand a query, and do not feed it crops.**

#### A single unreadable crop used to abort the whole stage

Measured before the fix: one blind candidate out of 27-41 skipped the rerank on
1 question in 6 (`jev`) and 2 in 6 (`jev+hybrid`) — the `expand` call still
billed, cosine order still served, under a mode named after a reranker that never
ran. Blind candidates are now dropped from the rerank instead and counted in
`blind_chunks`; all 6 questions rerank. The trade: a page with no text layer can
no longer be *promoted* by this stage, only ranked by the pixel index that found
it, which is what `visual` is for.
