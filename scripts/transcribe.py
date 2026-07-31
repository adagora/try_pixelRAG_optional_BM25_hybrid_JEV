#!/usr/bin/env python
"""Read every page once, keep the text, stop paying to re-read pixels.

VERDICT ON gemini-3.6-flash: DO NOT USE THIS AS A COST OPTIMISATION. Measured
over 9 pages, transcripts averaged 1.25x the tokens of the page image they
replace. The reason is that Gemini bills a page image at a FLAT ~1093 tokens
regardless of what is on it, while text scales with density -- and the densest
content here is numeric tables, which are close to the worst case for a
tokeniser. One 21-column weight matrix transcribed to 4744 chars carrying 2067
digits and 737 pipe characters: 3707 tokens as text, 1093 as pixels, 3.4x worse.

    page   image tok   text tok   ratio
    0/p4        1093        156   0.14x   sparse cover page
    0/p1        1093        458   0.42x
    1/p17       1093        885   0.81x
    1/p9        1093        981   0.90x
    0/p11       1093       1197   1.10x
    1/p12       1093       3707   3.39x   weight matrix
    1/p14       1093       3687   3.37x   weight matrix
    mean        1093       1364   1.25x

The pages where transcription loses are exactly the tables and dimensioned
drawings that carry the answers, so the average understates how bad the trade is
in practice.

This inverts on a provider that bills images by area. Anthropic charges a
200-DPI A4 page ~2318 tokens at the 1568px tier and ~5158 at the high-res tier,
so even the 3707-token worst case would win there. Re-run --measure before
assuming either way; that is the whole point of the flag.

The pass itself is correct and idempotent, so it is kept as a measurement tool
and for the day this moves to a per-area-billed reader. It is deliberately NOT
wired into rag.py.

--- original rationale, still true where the arithmetic works ---

A page that answers five questions is currently sent to the reader five times,
as an image, at full price each time. Its content does not change between those
five reads. Transcribing it once and reusing the text turns a recurring
per-question image cost into a one-off cost per page.

Retrieval is untouched: pages are still FOUND by how they look. This only
changes what the reader is handed once retrieval has chosen them, which is a
separate axis from the claim the project exists to test.

WHAT A TRANSCRIPT MUST PRESERVE, because these are what the answers depend on:
  * every digit, exactly, including thousands separators as typeset
  * table structure -- which row label and column header a cell sits under, and
    which product family a price column belongs to
  * the meaning carried by SHADING, which markdown cannot represent at all.
    Grey cells in these price matrices mark execution ranges and colour
    restrictions; a transcript that silently drops that is worse than no
    transcript, because it reads as complete.
  * figure and table numbers ("Rys. 7", "Tab. 2"), which answers cite

THE HONEST RISK. Anything the transcriber misreads is baked in permanently and
re-served to every future question, where a bad image read is at least a fresh
mistake each time. That is why transcripts record the model that produced them
and why --check re-reads a sample against the page.

    .venv/bin/python scripts/transcribe.py --limit 4     # pilot, then measure
    .venv/bin/python scripts/transcribe.py               # all pages
    .venv/bin/python scripts/transcribe.py --measure     # compare token costs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import imagefit
import layout
import providers
import rag

OUT = layout._anchored(os.environ.get("PIXELRAG_TRANSCRIPTS", ""),
                       layout.DEFAULT.transcripts_dir)
MANIFEST = OUT / "manifest.json"

# Deliberately not the reader's system prompt. The reader's job is to answer;
# this job is to reproduce, and asking for interpretation here is how a wrong
# reading gets frozen into the cache.
SYSTEM = """Przepisujesz stronę katalogu/cennika technicznego na tekst. Nie
odpowiadasz na pytania i nie interpretujesz treści — odtwarzasz ją.

ZASADY:
1. LICZBY dosłownie, co do cyfry, wraz ze spacjami/separatorami tak jak na
   stronie ("3 555", nie "3555"). Nie zaokrąglaj, nie przeliczaj, nie poprawiaj.
2. TABELE jako tabele markdown. Zachowaj WSZYSTKIE nagłówki wierszy i kolumn.
   Jeśli kolumny odpowiadają rodzinom produktów (UniPro, SNP, RenoSystem…),
   nazwij je w nagłówku dokładnie tak jak w dokumencie. Pusta komórka pozostaje
   pusta — nie przenoś wartości z sąsiedniej kolumny.
3. CIENIOWANIE I KOLOR KOMÓREK niosą znaczenie, którego markdown nie zapisze.
   Po każdej tabeli, w której występują, dodaj linię
   `> CIENIOWANIE: <opis które komórki/zakresy są wyróżnione i co mówi legenda>`.
   Jeśli pod tabelą jest legenda, przepisz ją w całości.
4. RYSUNKI: nie opisuj ich wyglądu w prozie. Podaj numer i podpis
   ("Rys. 7 — Przekrój poziomy"), wypisz wszystkie wymiary i etykiety widoczne
   na rysunku jako listę, i zaznacz `> RYSUNEK` w tej sekcji.
5. NAGŁÓWKI sekcji i nazwy produktów zachowaj dosłownie — po nich odbywa się
   późniejsze wyszukiwanie cytatów.
6. Jeśli fragment jest nieczytelny, napisz `[NIECZYTELNE]` w tym miejscu.
   Nie zgaduj. Zgadnięta cyfra jest gorsza niż jej brak.
7. Jednostki i dopiski ("za szt.", "za kpl.", "netto", "do ceny bramy")
   przepisz razem z wartością, do której należą.

Zacznij od razu od treści strony. Bez wstępu i bez komentarza."""


def _page_key(path: Path, model: str) -> str:
    st = path.stat()
    return hashlib.sha256(
        f"{path.name}|{st.st_size}|{st.st_mtime_ns}|{model}".encode()).hexdigest()[:20]


def _load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1),
                        encoding="utf-8")


def transcript_path(article_id: int, tile_index: int) -> Path:
    return OUT / str(article_id) / f"{tile_index:04d}.md"


def load(article_id: int, tile_index: int) -> str | None:
    """The cached transcript for a page, or None if it was never made."""
    p = transcript_path(article_id, tile_index)
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def _pages() -> list[tuple[int, int, Path]]:
    out = []
    for d in rag.LAYOUT.tile_dirs():
        aid = layout._article_id_of(d)
        if aid < 0:
            continue
        for f in sorted(d.glob("tile_*.jpg")):
            out.append((aid, int(f.stem.split("_")[1]), f))
    return out


def _transcribe_one(client, types, path: Path, model: str) -> tuple[str, int, int]:
    img, mime = imagefit.fit(path, providers.GeminiReader().image_policy)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[
            types.Part.from_bytes(data=img, mime_type=mime),
            types.Part.from_text(text="Przepisz tę stronę zgodnie z zasadami."),
        ])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            # Transcription is not a reasoning task; thinking here is spend with
            # nothing to show for it.
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL),
        ),
    )
    um = getattr(resp, "usage_metadata", None)
    tin = (um.prompt_token_count or 0) if um else 0
    tout = (um.candidates_token_count or 0) if um else 0
    return providers.text_of(resp), tin, tout


def run(limit: int | None, article: int | None, force: bool) -> None:
    from google.genai import types

    reader = providers.GeminiReader()
    model = reader.model
    client = reader._client()
    manifest = _load_manifest()

    todo = [(a, t, p) for a, t, p in _pages()
            if article is None or a == article]
    todo = [(a, t, p) for a, t, p in todo
            if force or manifest.get(f"{a}/{t}", {}).get("key") != _page_key(p, model)]
    if limit:
        todo = todo[:limit]

    if not todo:
        print("nothing to do — every page already transcribed for this model")
        return

    print(f"transcribing {len(todo)} page(s) with {model}\n")
    tin = tout = 0
    t0 = time.perf_counter()
    for i, (aid, tile, path) in enumerate(todo, 1):
        try:
            text, a, b = _transcribe_one(client, types, path, model)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {aid}/p{tile + 1}  FAILED: "
                  f"{type(e).__name__}: {e}")
            continue
        tin += a
        tout += b
        dest = transcript_path(aid, tile)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        manifest[f"{aid}/{tile}"] = {
            "key": _page_key(path, model), "model": model,
            "chars": len(text), "out_tokens": b, "at": time.time(),
        }
        _save_manifest(manifest)
        print(f"  [{i}/{len(todo)}] {aid}/p{tile + 1}  {len(text):6d} chars  "
              f"{b:5d} out tok")

    dt = time.perf_counter() - t0
    print(f"\n{len(todo)} pages in {dt:.0f}s · {tin} in / {tout} out")


def measure() -> None:
    """The go/no-go: does a transcript actually cost less than the image?

    Compares like with like — the same page, counted by the same tokeniser the
    reader will bill against. Predicting this from page geometry is what put a
    50%-saving claim in this repo that measured zero.
    """
    from google.genai import types

    reader = providers.GeminiReader()
    model = reader.model
    client = reader._client()
    manifest = _load_manifest()
    done = [k for k in manifest]
    if not done:
        print("no transcripts yet — run with --limit 4 first")
        return

    print(f"{'page':>12}  {'image tok':>10}  {'text tok':>9}  {'ratio':>7}")
    print("-" * 45)
    tot_i = tot_t = 0
    for key in done:
        aid, tile = (int(x) for x in key.split("/"))
        path = rag.page_path(aid, tile)
        text = load(aid, tile) or ""

        img, mime = imagefit.fit(path, providers.GeminiReader().image_policy)
        n_img = client.models.count_tokens(
            model=model,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=img, mime_type=mime)])],
        ).total_tokens
        n_txt = client.models.count_tokens(
            model=model,
            contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=text)])],
        ).total_tokens
        tot_i += n_img
        tot_t += n_txt
        print(f"{aid}/p{tile + 1:<8}  {n_img:10d}  {n_txt:9d}  "
              f"{n_txt / max(n_img, 1):6.2f}x")

    n = len(done)
    print("-" * 45)
    print(f"{'mean':>12}  {tot_i / n:10.0f}  {tot_t / n:9.0f}  "
          f"{tot_t / max(tot_i, 1):6.2f}x")
    print()
    if tot_t < tot_i:
        k = int(os.environ.get("PIXELRAG_ONESHOT_PAGES", "4"))
        print(f"VERDICT: transcripts are cheaper. At {k} pages/question that is "
              f"{(tot_i - tot_t) / n * k:.0f} fewer input tokens per question.")
    else:
        print("VERDICT: transcripts cost MORE than the images. The premise for "
              "this cache does not hold on this model — stop here.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, help="transcribe at most N pages")
    ap.add_argument("--article", type=int, help="only this article_id")
    ap.add_argument("--force", action="store_true", help="redo existing")
    ap.add_argument("--measure", action="store_true",
                    help="compare transcript vs image tokens, then exit")
    args = ap.parse_args()

    if args.measure:
        measure()
        return
    run(args.limit, args.article, args.force)


if __name__ == "__main__":
    main()
