"""Polish normalisation and BM25. The retrieval quality lives in the folding
and stemming, so that is what is pinned here."""

from __future__ import annotations

import lexical


def test_fold_strips_polish_diacritics():
    assert lexical.fold("wał") == "wal"
    assert lexical.fold("Wiśniowski") == "Wisniowski"
    assert lexical.fold("ŁĄCZNIK") == "LACZNIK"


def test_stem_leaves_short_content_words_intact():
    """Over-stemming 'wal'/'RC2'/'SNP' would merge unrelated terms."""
    for w in ("wal", "rc2", "snp", "okno"):
        assert lexical.stem(w) == w


def test_stem_collapses_inflections_of_one_word():
    assert lexical.stem("roletowe") == lexical.stem("roletowa")
    assert lexical.stem("roletowych") == lexical.stem("roletowe")


def test_stem_never_leaves_a_stub():
    """The >=4 guard: stripping must not take a word below the point where it
    still identifies anything."""
    for w in ("panele", "bramy", "drzwiowe", "uszczelniajacy"):
        assert len(lexical.stem(w)) >= 4


def test_stem_leaves_digits_alone():
    assert lexical.stem("123456") == "123456"


def test_tokenise_drops_intent_words():
    toks = lexical.tokenise("Pokaż jaka jest cena progu uszczelniającego")
    assert not {"pokaz", "jaka", "jest"} & set(toks)
    assert lexical.stem("progu") in toks


def test_tokenise_can_keep_stopwords():
    assert "jest" in lexical.tokenise("jest brama", drop_stop=False)


def test_tokenise_folds_and_lowercases():
    assert lexical.tokenise("Dzielony WAŁ") == lexical.tokenise("dzielony wal")


# -- BM25 -------------------------------------------------------------------

def _bm(*docs):
    return lexical.BM25([lexical.tokenise(d) for d in docs])


def test_bm25_ranks_the_matching_document_first():
    bm = _bm("dzielony wal napedu", "panele przetloczenia kolory",
             "brama segmentowa uszczelka")
    assert max(range(3), key=lambda i: bm.scores(lexical.tokenise("dzielony wal"))[i]) == 0


def test_bm25_unknown_term_scores_nothing():
    bm = _bm("dzielony wal", "panele kolory")
    assert bm.scores(["nieistniejacyterminxyz"]) == [0.0, 0.0]


def test_bm25_idf_stays_positive_for_common_terms():
    """+0.5/+0.5 smoothing: at this collection size a term in most documents is
    still informative, and plain BM25 idf would go negative and rank it away."""
    bm = _bm("brama segmentowa", "brama roletowa", "brama garazowa", "panel")
    assert bm.idf[lexical.stem("brama")] > 0


def test_bm25_empty_collection_does_not_divide_by_zero():
    bm = lexical.BM25([])
    assert bm.scores(["cokolwiek"]) == []


def test_revision_prefers_the_price_list_code():
    """Two editions of one price list disagree on prices; the date decides."""
    text = "nagłówek 01.01.2020 stopka CBG/PL-PLN/10.04.2025"
    assert lexical._revision(text) == "2025-04-10"


def test_revision_falls_back_to_a_header_date():
    assert lexical._revision("Cennik 10.04.2025 reszta strony") == "2025-04-10"


def test_revision_none_when_undated():
    assert lexical._revision("brak daty w tym tekscie") is None
