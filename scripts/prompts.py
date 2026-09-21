"""Everything the reader is told: system prompts, tool schemas, refusal copy.

All of it is text, and nearly all of it is Polish. It is here rather than in
rag.py so that changing what the model is asked does not mean reading an
orchestration module, and so that a diff touching only wording is obviously
only wording.

THE TWO SYSTEM PROMPTS ARE NOT VARIANTS OF EACH OTHER. `SYSTEM` drives agent
mode, where the model has tools and must be told to stop using them — every
round trip re-sends every image, so the loop length IS the bill. `ONESHOT`
drives the default path, where the pages are already attached, there is nothing
to call, and the entire job is reading what is in front of it. Merging them has
been tried and produces a prompt that hedges about tools that are not there.

THE CITATION CONTRACT LIVES IN THREE PLACES AND THEY MUST AGREE: the format is
demanded here (`ONESHOT`, closing instruction), the page number is stated in
`page_header` in the exact shape that will be parsed, and `citeparse.py` parses
it. A change to any one of them without the others produces answers whose
citations silently stop resolving — the answer still reads fine, which is what
makes it worth saying out loud.

WHY THE REFUSALS ARE TWO SENTENCES AND NOT ONE: one refusal used to cover both
"this corpus does not carry the answer" and "this question is about another
domain", and the sentence it showed was wrong for one of them. "Rebuild your
index" is unhelpful advice to someone who asked a door catalogue about engine
oil. `jev.Gate` separates the cases; these are what each should say.
"""

from __future__ import annotations

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
# agent-mode tools
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


# --------------------------------------------------------------------------
# one-shot framing, and what to say when there is nothing to say
# --------------------------------------------------------------------------

NO_PAGES_PL = (
    "Nie znalazłem w zaindeksowanych dokumentach żadnej strony pasującej do tego "
    "pytania, więc nie mogę na nie odpowiedzieć. Sprawdź, czy dokument na ten "
    "temat jest w katalogu pdfs/ i czy indeks został przebudowany."
)

# One refusal used to cover two different failures, and the sentence it showed
# was wrong for one of them. `jev.Gate` separates them; these are what each one
# should actually say, because "rebuild your index" is unhelpful advice to
# someone who asked a door catalogue about engine oil.
REFUSAL_PL = {
    "not-in-corpus": (
        "Te dokumenty dotyczą tego tematu, ale nie zawierają informacji "
        "potrzebnej do odpowiedzi na to pytanie — nie odpowiadam, zamiast "
        "zgadywać. Prawdopodobnie potrzebny jest inny dokument (np. cennik "
        "zamiast karty technicznej)."),
    "out-of-scope": (
        "To pytanie dotyczy innej dziedziny niż zaindeksowane dokumenty, "
        "więc nie ma tu na nie odpowiedzi."),
}


def page_header(p: dict) -> str:
    """Label above each attached image.

    The page number here is what the reader must echo back in its citation
    block, so it is stated once, unambiguously, in the same form we parse.
    """
    return f"\n[{p['document']} — strona {p['page']}]  (page={p['page']})"


def oneshot_preamble(question: str, pages: list[dict]) -> str:
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


