"""The question-set builder, and the checks that keep its labels honest.

The mining stage is pure string work and is tested without any model at all.
The labelling stage is faked at `jev.evaluate`, so the agreement rule, the
grounding check and the negative class are all real code.
"""

from __future__ import annotations

import pytest

import jev
import oracle
import xray


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("PIXELRAG_JEV", "1")


# -- mining, with no model involved -----------------------------------------

def test_headings_are_the_all_caps_lines():
    text = ("BRAMY GARAŻOWE\n"
            "Idealne rozwiązanie do obiektów, gdzie\n"
            "ANTYKOROZYJNOŚĆ\n"
            "Kurtyna bramy oraz prowadnice wykonane\n")
    assert oracle.headings(text) == ["BRAMY GARAŻOWE", "ANTYKOROZYJNOŚĆ"]


def test_a_wrapped_sentence_is_not_a_heading():
    """"Idealne rozwiązanie do obiektów, gdzie" is a line of prose the PDF
    happened to break; a question built on it would be nonsense."""
    assert oracle.headings("Idealne rozwiązanie do obiektów, gdzie\n") == []
    assert oracle.headings("COS TAM W\n") == []          # dangling preposition


def test_a_table_cell_is_not_a_heading():
    assert oracle.headings("Ho – 130 [mm]\n") == []
    assert oracle.headings("2300 2400 2500\n") == []


def test_a_short_document_is_not_emptied_by_the_boilerplate_filter():
    """In a 2-page document every heading sits on "50%" of it, so an unguarded
    ratio deletes the whole document rather than its furniture."""
    pages = [{"article_id": 0, "page": i + 1, "text": f"TEMAT NUMER {i}\n"}
             for i in range(2)]
    assert len(oracle.subjects(pages)) == 2


def test_running_headers_are_dropped_per_document():
    """"KARTA TECHNICZNA" is on all 11 pages of article 0 and names nothing.
    Per DOCUMENT, so a 27-page catalogue's furniture cannot survive by being
    diluted across a corpus that also holds an 11-page one."""
    pages = [{"article_id": 0, "page": i + 1,
              "text": f"KARTA TECHNICZNA\nTEMAT NUMER {i}\n"} for i in range(10)]
    mined = dict(oracle.subjects(pages))
    assert "KARTA TECHNICZNA" not in mined
    assert "TEMAT NUMER 3" in mined


def test_the_same_subject_is_mined_once_ignoring_brackets():
    """The text layer emits "RAL 9005" and "(RAL 9005)" on the same page, and
    the first run produced two identical questions about one colour code."""
    pages = [{"article_id": 0, "page": 1, "text": "RAL 9005\nINNE RZECZY\n"},
             {"article_id": 0, "page": 2, "text": "(RAL 9005)\nKOLEJNE RZECZY\n"},
             {"article_id": 0, "page": 3, "text": "TRZECIA STRONA\n"},
             {"article_id": 0, "page": 4, "text": "CZWARTA STRONA\n"}]
    subjects = [s for s, _ in oracle.subjects(pages)]
    assert subjects.count("RAL 9005") == 1
    assert not any(s.startswith("(") for s in subjects)


def test_soft_hyphens_do_not_stop_a_subject_matching_itself():
    """This corpus's text layer carries U+00AD inside words the PDF wrapped, so
    a heading mined from one line failed to be found in its own page text."""
    assert oracle._flat("WI\xadSNIOWSKI") == oracle._flat("WISNIOWSKI")


# -- labelling --------------------------------------------------------------

def sweep(monkeypatch, scores, answerable):
    """Script xray.xray directly — the labelling stage's only dependency."""
    def fake(question, pages, report=None):
        result = xray.XRay(pages=len(pages), judged=len(scores), shards=1)
        result.scores = dict(scores)
        result.facets = xray.Facets(answerable=answerable, scope=0.9)
        return result

    monkeypatch.setattr(oracle.xray, "xray", fake)


def candidate(**kwargs):
    base = {"subject": "FOTOKOMÓRKI", "page": "a0:s6",
            "question": 'Do czego służy "FOTOKOMÓRKI"?', "kind": "spec"}
    return oracle.Candidate(**{**base, **kwargs})


CORPUS = [{"article_id": 0, "page": 6, "title": "doc0",
           "text": "FOTOKOMÓRKI chronią bramę przed zamknięciem na przeszkodzie."},
          {"article_id": 0, "page": 1, "title": "doc0", "text": "COS INNEGO"}]
TITLES = {0: "doc0"}


def test_a_positive_needs_both_judgments_to_agree(monkeypatch):
    """The page Score and the corpus Noul come from one request and are
    independent; a label that rests on only one of them is a coin flip."""
    c = candidate()
    c.scores = {"a0:s6": 0.9}
    c.answerable = 0.05                       # the Noul disagrees with the Score
    assert oracle.rows([c], TITLES, CORPUS) == []

    c2 = candidate()
    c2.scores = {"a0:s6": 0.9}
    c2.answerable = 0.9
    assert len(oracle.rows([c2], TITLES, CORPUS)) == 1


def test_the_negative_class_is_led_by_the_noul_not_the_score(monkeypatch):
    """`Ile kosztuje "FOTOKOMÓRKI"?` in a corpus with no prices: the photocell's
    own pages still score "related topic", so the Score lands mid-band and can
    never say "nothing answers this". The Noul can, and does."""
    c = candidate(question='Ile kosztuje "FOTOKOMÓRKI"?', kind="price")
    c.scores = {"a0:s6": 0.45}                # related, not an answer
    c.answerable = 0.04
    rows = oracle.rows([c], TITLES, CORPUS)
    assert len(rows) == 1
    assert rows[0]["answerable"] is False
    assert rows[0]["pages"] == [] and rows[0]["primary"] is None


def test_a_gold_page_that_does_not_contain_the_subject_is_rejected():
    """`HI MARINA HORIZON — jakie są parametry techniczne?` was labelled onto a
    garage-door dimensions table, because the form's own words outweighed its
    subject. The string occurs on exactly one page and it is not that one."""
    c = candidate(subject="HI MARINA HORIZON", page="a1:s16",
                  question='HI MARINA HORIZON — jakie są parametry techniczne?')
    c.scores = {"a0:s1": 0.9}                 # a page that never names it
    c.answerable = 0.9
    assert oracle.rows([c], TITLES, CORPUS) == []
    assert c.rejected and "do not contain the subject" in c.rejected


def test_grounding_needs_only_one_anchor_page():
    """A continuation page answers without repeating the heading above it —
    the `kolory tkanin soltis` case. Requiring every gold page to name the
    subject would throw those away."""
    corpus = CORPUS + [{"article_id": 0, "page": 7, "title": "doc0",
                        "text": "dalszy ciag tabeli bez naglowka"}]
    c = candidate()
    c.scores = {"a0:s6": 0.9, "a0:s7": 0.8}
    c.answerable = 0.9
    rows = oracle.rows([c], TITLES, corpus)
    assert rows[0]["pages"] == [6, 7]


def test_gold_is_whichever_page_the_corpus_answers_with(monkeypatch):
    """Not the page the heading was mined from. A question mined from page 7
    whose real answer is on page 6 has to get page 6 or the set teaches the
    retriever the wrong target."""
    c = candidate(page="a0:s1")               # mined from page 1
    c.scores = {"a0:s6": 0.95, "a0:s1": 0.2}
    c.answerable = 0.9
    rows = oracle.rows([c], TITLES, CORPUS)
    assert rows[0]["primary"] == 6
    assert rows[0]["oracle"]["from"] == "a0:s1"


def test_probes_are_only_built_on_subjects_that_are_things():
    """"How much does the WEIGHT OF A FREESTANDING PERGOLA cost" is not a
    question, so the answer to it was noise that reached the eval file."""
    thing = candidate()
    thing.product = 0.95
    prop = candidate(subject="WAGA PERGOLI", question='Co to jest "WAGA PERGOLI"?')
    prop.product = 0.1
    built = oracle.probes([thing, prop], every=1)
    assert {c.subject for c in built} == {"FOTOKOMÓRKI"}
    assert {c.kind for c in built} == set(oracle.PROBES)


def test_a_rejected_candidate_never_reaches_the_file():
    c = candidate()
    c.scores = {"a0:s6": 0.9}
    c.answerable = 0.9
    c.rejected = "too general"
    assert oracle.rows([c], TITLES, CORPUS) == []


# -- the file it writes ------------------------------------------------------

def test_the_written_file_is_the_schema_evaluate_pl_reads(tmp_path):
    import yaml

    c = candidate()
    c.scores = {"a0:s6": 0.9}
    c.answerable = 0.9
    out = tmp_path / "questions.yaml"
    oracle.dump(oracle.rows([c], TITLES, CORPUS), out)
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert loaded[0]["q"] == 'Do czego służy "FOTOKOMÓRKI"?'
    assert loaded[0]["doc"] == "doc0"
    assert loaded[0]["pages"] == [6] and loaded[0]["primary"] == 6
    assert loaded[0]["kind"] == "spec"


def test_the_file_says_what_it_is_not_evidence_for(tmp_path):
    """The circularity caveat has to travel with the labels, not live in a
    commit message nobody reads next year."""
    text = oracle.dump([], tmp_path / "q.yaml")
    assert "agreement, not accuracy" in text
    assert "Generated by scripts/oracle.py" in text
