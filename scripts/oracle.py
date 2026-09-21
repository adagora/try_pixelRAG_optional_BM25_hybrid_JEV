#!/usr/bin/env python
"""Build the question set this repo has never had, and label it from the corpus.

The README's most-repeated sentence is that nothing here can be measured:
`evaluate_pl.py` keys its ground truth to a price list this index no longer
contains, four of the six benchmark questions have no answer anywhere in the
corpus, and every retrieval figure quoted — visual 50/85, BM25 75/80, fused
80/95 — was measured against documents that are gone. "Does the reranker earn
its bill" is open for exactly one reason: there is no question set.

Writing one by hand means reading 38 pages of Polish datasheets and inventing
questions whose answers are on them. This does it instead, in three stages, and
the split between them is the whole design:

  1. MINE, in code, for free. Headings are spans of the page text, so finding
     them is string work, not judgment. Running headers and wrapped sentences
     are dropped by rules, because "appears on 27 of 27 pages" is a fact and
     asking a model about it would be paying for arithmetic.
  2. SELECT, with Jev, over the question FORM only. Several Polish phrasings of
     each subject are generated locally; Jev picks the well-formed one, says
     whether it is specific enough to have a single answer, and classifies it.
     No page text is sent, because none is needed to judge a sentence — and
     because whether the corpus answers it is stage 3's job, done better.
  3. LABEL, with the X-ray. Every surviving question is scored against every
     page of the corpus (xray.py), so gold is whatever the corpus actually
     answers with — not whichever page the heading was mined from. A question
     mined from page 7 whose real answer is on page 9 gets page 9, and a
     question nothing answers is kept and labelled `answerable: false`, which
     is the negative class scripts/bench.py scores the refusal gate against,
     and which could not honestly be written by hand.

WHAT THIS ORACLE IS NOT EVIDENCE FOR, stated here rather than in a footnote:
the labels come from Jev reading page text, so a mode that ranks by Jev reading
page text is being graded by its own reading. Against `jev`, `jev-page` and
`xray` this file measures agreement, not accuracy, and a win there is worth
nothing. Against `visual` it is a fair test — pixels against text, different
evidence entirely — and against `hybrid` it is nearly fair, since BM25 matches
strings where the oracle judges meaning. Read the per-mode table with that in
mind, or use `scripts/bench.py --verified`, which keeps only the questions
whose gold page is decided by literal string match and is therefore a referee
no mode shares. Measured on the generated set: 13 of 23 questions qualify and
this oracle agreed with the string on 12 of them.

    .venv/bin/python scripts/oracle.py --limit 40          # write the set
    .venv/bin/python scripts/oracle.py --limit 8 --dry-run # see it, write nothing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import jev
import xray

log = logging.getLogger("pixelrag")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

OUT = Path("eval/questions_pl.yaml")

# A page scores 0..1 (xray normalises the 4-level rubric). 0.66 is the bottom of
# "Partial answer", so gold means "this page at least partly answers it" and a
# page that is merely on-topic does not qualify.
GOLD_SCORE = float(os.environ.get("PIXELRAG_ORACLE_GOLD", "0.66"))
# Reported alongside each label so an auditor can see how close to the line it
# was. It is deliberately NOT the negative test — see the comment in `rows`.
DEAD_SCORE = float(os.environ.get("PIXELRAG_ORACLE_DEAD", "0.34"))
# Jev must agree the question is a real, answerable-in-principle question.
WELL_FORMED = 0.6
SPECIFIC = 0.5
# Probes ask what a thing costs or how to fit it, so they need a thing.
PRODUCT = 0.6
# The X-ray answers "is it on this page" (a per-page Score) and "is it in this
# corpus" (one Noul) in the same request, from the same state. They are
# independent questions and Jev guarantees no arithmetic between them, so when
# they disagree the honest reading is that neither is trustworthy for a LABEL.
# Measured: `Ile kosztuje "WAGA PERGOLI WOLNOSTOJĄCEJ"?` put a weight table
# above the gold threshold in a corpus with no prices in it, while the Noul in
# the same response said 0.1. Requiring both is what catches that.
AGREE_YES = 0.5
AGREE_NO = 0.3
# Stage 2 asks 4 questions per candidate; this keeps one request well inside
# the 64k window and well inside anything the API might cap question count at.
BATCH = int(os.environ.get("PIXELRAG_ORACLE_BATCH", "30"))
WORKERS = int(os.environ.get("PIXELRAG_ORACLE_WORKERS", "4"))

# --------------------------------------------------------------------------
# stage 1 — mine, in code
# --------------------------------------------------------------------------

# A heading in these datasheets is a short all-caps line. The ratio rather than
# `.isupper()` because Polish headings carry digits, units and model codes
# ("BRAMA BR-77s | BR-77E"), and because `.isupper()` is False for any of them.
UPPER_RATIO = 0.85
# Above this share of a document's pages a line is furniture, not a subject:
# "KARTA TECHNICZNA" is on all 11 pages of article 0 and names nothing.
BOILERPLATE = 0.30
# Below this the ratio is noise, not repetition — see `subjects`.
BOILERPLATE_MIN_PAGES = 4
# Mostly numbers and units — a table cell that survived the line filter.
_NUMERIC = re.compile(r"^[\W\d\s]*$|^\S{1,3}\s*[–-]\s*\d")
# Polish function words that only ever end a WRAPPED sentence, never a heading.
_DANGLING = {"i", "oraz", "lub", "w", "we", "z", "ze", "na", "do", "od", "za",
             "o", "u", "dla", "przy", "po", "pod", "nad", "bez", "przez",
             "gdzie", "które", "który", "która", "jest", "są", "aby", "że"}


def headings(text: str) -> list[str]:
    """Candidate subject phrases from one page's text layer.

    Pure string work on purpose. Every rule here encodes something that is
    true of the document rather than something that needs judging, and each
    one was added because the corpus produced the case it removes.
    """
    out = []
    for line in text.splitlines():
        s = " ".join(line.split()).strip(" -–—:|")
        words = s.split()
        if not (3 <= len(s) <= 60 and 1 <= len(words) <= 6):
            continue
        if s.endswith((".", ",", ";", ":")) or words[-1].lower() in _DANGLING:
            continue
        if _NUMERIC.match(s):
            continue
        letters = [c for c in s if c.isalpha()]
        if len(letters) < 3:
            continue
        if sum(c.isupper() for c in letters) / len(letters) < UPPER_RATIO:
            continue
        out.append(s)
    return out


def subjects(pages: list[dict]) -> list[tuple[str, str]]:
    """(subject, page key) for every heading that names something page-specific.

    Boilerplate is removed per DOCUMENT rather than per corpus: a line on every
    page of article 0 is that document's letterhead even if article 1 has never
    heard of it, and pooling the two would let a 27-page catalogue's furniture
    survive by dilution.
    """
    per_doc: dict[int, Counter] = {}
    mined: list[tuple[str, str, int]] = []
    doc_pages: Counter = Counter()
    for page in pages:
        aid = page["article_id"]
        doc_pages[aid] += 1
        found = list(dict.fromkeys(headings(page["text"])))  # once per page
        per_doc.setdefault(aid, Counter()).update(found)
        for head in found:
            mined.append((head, f"a{aid}:s{page['page']}", aid))

    seen: set[str] = set()
    out = []
    for head, key, aid in mined:
        # The boilerplate ratio needs enough pages to mean anything. In a
        # 2-page document every heading is on "50%" of it and the filter
        # deletes the entire document; the rule is about repetition, and three
        # pages is the fewest that can show any.
        if (doc_pages[aid] >= BOILERPLATE_MIN_PAGES
                and per_doc[aid][head] / doc_pages[aid] > BOILERPLATE):
            continue
        # Dedupe on letters and digits only. The text layer emits the same
        # subject with and without its brackets — "RAL 9005" and "(RAL 9005)"
        # both survived the first run and produced two identical questions
        # about one colour code.
        fold = re.sub(r"[^0-9a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", "", head).casefold()
        if not fold or fold in seen:   # first page that carries it wins
            continue
        seen.add(fold)
        out.append((head.strip("()[]{}"), key))
    return out


# Phrasings, generated locally, for Jev to choose between.
#
# Polish is inflected and these subjects are mined in the nominative, so
# "parametry techniczne KURTYNA BRAMY" is ungrammatical and no amount of
# templating fixes it. Every form below therefore keeps the subject in the
# nominative — quoted, or in apposition after a dash or colon, both of which
# are ordinary written Polish. Jev picks the one that reads as a question a
# person would type, and `well_formed` is the escape when none of them do.
FORMS = {
    "a": 'Co to jest "{s}"?',
    "b": '{s} — jakie są parametry techniczne?',
    "c": '{s} — jakie warianty są dostępne?',
    "d": 'Do czego służy "{s}"?',
}

# What each form ASKS, which is what the Choice is actually choosing between.
#
# First attempt handed Jev the templates themselves as criteria, with the
# subject replaced by the words "the subject". Every option then read as a
# near-identical sentence about nothing and the model picked "a" 8 times out of
# 8 — a Choice whose options are indistinguishable is a constant with a bill.
# Describing the INTENT makes them distinct, and makes this the same judgment
# `kind` makes, which is why the two agree.
FORM_INTENT = {
    "a": "Asks for a definition: what the subject is",
    "b": "Asks for technical parameters, dimensions or figures",
    "c": "Asks which variants, sizes, colours or options exist",
    "d": "Asks what the subject is for, or what problem it solves",
}

# Probes: well-formed questions about a subject the corpus definitely contains,
# along a DIMENSION it may not cover.
#
# The README's sharpest finding was that this index holds no prices at all, and
# that four of six hand-written benchmark questions had no answer anywhere —
# discovered by accident, months in. A question set that cannot produce that
# class of question cannot catch it again. These are generated deliberately and
# then labelled by the X-ray like everything else, so if the corpus does answer
# them they become positives; if it does not they become the hardest negatives
# available — in scope, well-formed, about a subject that is demonstrably here,
# and still unanswerable. Nothing about the outcome is asserted in advance.
PROBES = {
    "price": 'Ile kosztuje "{s}"?',
    "procedure": 'Jak zamontować "{s}"?',
}


@dataclass
class Candidate:
    """One mined subject on its way to being a question."""

    subject: str
    page: str
    question: str | None = None
    kind: str | None = None
    well_formed: float | None = None
    specific: float | None = None
    product: float | None = None
    # filled by stage 3
    scores: dict[str, float] = field(default_factory=dict)
    answerable: float | None = None
    scope: float | None = None
    rejected: str | None = None


# --------------------------------------------------------------------------
# stage 2 — select the question, with Jev
# --------------------------------------------------------------------------

def _select_batch(batch: list[Candidate], report: jev.Report) -> None:
    """Choose a phrasing, and judge it, for a whole batch in one request.

    No page text is sent. That is not an economy — it is the point: these four
    questions are about a sentence, and mixing the corpus in would invite the
    model to answer "is this answerable" here, where it can see one page,
    instead of in stage 3, where it can see all of them.
    """
    state = {"candidates": {str(i): {"subject": c.subject, "options": {
        key: form.format(s=c.subject) for key, form in FORMS.items()}}
        for i, c in enumerate(batch)}}
    questions = {}
    for i, _ in enumerate(batch):
        questions[f"{i}.form"] = {"type": "choice", "instructions": (
            f"For candidate {i}, which question about its subject would a person "
            "most usefully ask — the one this kind of subject most invites?"),
            "criteria": dict(FORM_INTENT)}
        questions[f"{i}.well_formed"] = {"type": "noul", "instructions": (
            f"Is candidate {i}'s subject a real thing a document can describe — "
            "a product, component, feature or property — rather than a fragment, "
            "a table cell, a page header or a meaningless string?")}
        questions[f"{i}.specific"] = {"type": "noul", "instructions": (
            f"Is candidate {i}'s subject specific enough that a question about it "
            "would have one definite answer, rather than being so general that "
            "half a catalogue would answer it?")}
        # Only a THING can have a price or an installation procedure. The first
        # run built a price probe on "WAGA PERGOLI WOLNOSTOJĄCEJ" — the weight
        # of a freestanding pergola — and asked what that costs, which is not a
        # question, so the answer to it was noise. Probes need a noun that names
        # an object, and this is the field that says so.
        questions[f"{i}.product"] = {"type": "noul", "instructions": (
            f"Is candidate {i}'s subject a physical product, component or "
            "accessory — a thing that could be bought, shipped or fitted — rather "
            "than a property, a measurement, a material or a section title?")}
        questions[f"{i}.kind"] = {"type": "choice", "instructions": (
            f"What kind of information would a question about candidate {i}'s "
            "subject most likely ask for?"),
            "criteria": {
                "price": "A cost, price or quotation",
                "spec": "A technical parameter, dimension, material or figure",
                "procedure": "How to install, assemble, operate or maintain it",
                "catalogue": "Which variants, colours, sizes or options exist"}}
    answers = jev.evaluate(state, questions, report, "select")
    for i, c in enumerate(batch):
        try:
            form = answers[f"{i}.form"]["choice"]
            c.question = FORMS[form].format(s=c.subject)
        except (KeyError, TypeError):
            c.rejected = "no usable phrasing"
            continue
        c.well_formed = _noul(answers, f"{i}.well_formed")
        c.specific = _noul(answers, f"{i}.specific")
        c.product = _noul(answers, f"{i}.product")
        try:
            c.kind = answers[f"{i}.kind"]["choice"]
        except (KeyError, TypeError):
            c.kind = "spec"
        if (c.well_formed or 0) < WELL_FORMED:
            c.rejected = f"not a real subject ({c.well_formed})"
        elif (c.specific or 0) < SPECIFIC:
            c.rejected = f"too general ({c.specific})"


def _noul(answers: dict, key: str) -> float | None:
    try:
        return round(jev.number(answers[key], "noul", 1), 4)
    except (ValueError, KeyError, TypeError):
        return None


def select(candidates: list[Candidate], report: jev.Report | None = None) -> None:
    """Stage 2 over every candidate, in batches, concurrently."""
    report = report if report is not None else jev.Report()
    batches = [candidates[i:i + BATCH] for i in range(0, len(candidates), BATCH)]

    def run(batch):
        local = jev.Report()
        try:
            _select_batch(batch, local)
        except Exception:  # noqa: BLE001
            log.warning("Oracle selection batch failed; %d candidates dropped",
                        len(batch), exc_info=True)
            for c in batch:
                c.rejected = "selection call failed"
        return local

    if len(batches) == 1:
        reports = [run(batches[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(WORKERS, len(batches))) as pool:
            reports = list(pool.map(run, batches))
    for local in reports:
        report.calls.extend(local.calls)


# --------------------------------------------------------------------------
# stage 3 — label, with the X-ray
# --------------------------------------------------------------------------

def label(candidates: list[Candidate], corpus: list[dict],
          report: jev.Report | None = None) -> None:
    """Score every surviving question against every page of the corpus.

    One X-ray per question — this is the expensive stage and the only one whose
    cost scales with the question set, at about $0.0013 and 1.3 s each. The
    questions go out concurrently, so 40 of them is a minute, not forty.
    """
    report = report if report is not None else jev.Report()
    live = [c for c in candidates if c.rejected is None and c.question]
    if not live:
        return
    pages = [(f"a{p['article_id']}:s{p['page']}", p["text"]) for p in corpus]

    def run(c: Candidate):
        try:
            return c, xray.xray(c.question, pages)
        except Exception:  # noqa: BLE001
            log.warning("Oracle X-ray failed for %r", c.question, exc_info=True)
            return c, None

    with ThreadPoolExecutor(max_workers=min(WORKERS, len(live))) as pool:
        for c, sweep in pool.map(run, live):
            if sweep is None or not sweep.scores:
                c.rejected = "X-ray failed"
                continue
            report.calls.extend(sweep.report.calls)
            c.scores = sweep.scores
            c.answerable = sweep.facets.answerable
            c.scope = sweep.facets.scope
            if not sweep.exhaustive:
                # A partial sweep cannot certify gold: a page it never judged
                # might be the best one, which is the one claim this file makes.
                c.rejected = "X-ray was not exhaustive"


def gold(c: Candidate) -> tuple[int | None, list[int], int | None]:
    """(article_id, acceptable pages, best page) for one labelled candidate.

    Gold is confined to ONE document because that is the shape
    eval/questions_pl.yaml can express — `doc` names a title and `pages` are
    numbers within it. The document chosen is the one holding the best page,
    and any qualifying page in the other document is dropped rather than
    silently renumbered into this one. `spilled` reports how often that
    happened, because a question whose answer is split across two catalogues is
    a bad eval question and the count is how you would know.
    """
    if not c.scores:
        return None, [], None
    best_key, best = max(c.scores.items(), key=lambda kv: kv[1])
    if best < GOLD_SCORE:
        return None, [], None
    aid = int(best_key.split(":")[0][1:])
    pages = sorted(int(key.split(":s")[1]) for key, score in c.scores.items()
                   if score >= GOLD_SCORE and key.startswith(f"a{aid}:"))
    return aid, pages, int(best_key.split(":s")[1])


def spilled(c: Candidate) -> int:
    """Qualifying pages that fell outside the chosen document."""
    aid, pages, _ = gold(c)
    if aid is None:
        return 0
    return sum(1 for key, score in c.scores.items()
               if score >= GOLD_SCORE and not key.startswith(f"a{aid}:"))


def best_score(c: Candidate) -> float:
    return max(c.scores.values()) if c.scores else 0.0


# --------------------------------------------------------------------------
# writing it out
# --------------------------------------------------------------------------

def _flat(text: str) -> str:
    """Comparable form of a span of PDF text.

    Soft hyphens are the reason this is not just `.casefold()`: this corpus's
    text layer carries U+00AD inside words it wrapped ("WI\xadSNIOWSKI",
    "wy\xadłącznikami"), so a heading mined from one line and searched for in
    the page's full text can fail to match itself.
    """
    return re.sub(r"\s+", " ", text.replace("\xad", "").replace("\u00a0", " ")).casefold()


def _grounded(c: Candidate, aid: int, pages: list[int],
              text_of: dict[str, str]) -> bool:
    """Does the subject literally appear on any page the oracle called gold?

    A free, non-model check on a model's label, and it earns its place. The
    first run produced `HI MARINA HORIZON — jakie są parametry techniczne?`
    with gold on article 0 page 5, a garage-door dimensions table; the string
    "HI MARINA HORIZON" occurs on exactly one page of this corpus and it is
    a1:s16. The question form's own words ("parametry techniczne") outweighed
    its subject — the same dilution the README documents for query expansion,
    where adding topic signal drowned the constraint.

    ANY gold page, not all of them: a continuation page legitimately answers a
    question without repeating the heading it sits under, which is the whole
    `kolory tkanin soltis` problem. Requiring one anchor keeps those and still
    catches a gold set that has drifted to another document entirely.
    """
    needle = _flat(c.subject)
    return any(needle in _flat(text_of.get(f"a{aid}:s{p}", "")) for p in pages)


def rows(candidates: list[Candidate], titles: dict[int, str],
         corpus: list[dict] | None = None) -> list[dict]:
    """The eval file's records: positives with gold pages, negatives without.

    Both classes are kept. `evaluate_pl.py` reads `q/kind/doc/pages/primary`
    and ignores the rest; `calibration.py` reads `answerable` and needs the
    negatives, which is why a question the corpus cannot answer is a result
    here rather than a reject.
    """
    text_of = {f"a{p['article_id']}:s{p['page']}": p["text"]
               for p in (corpus or [])}
    out = []
    for c in candidates:
        if c.rejected is not None or not c.scores:
            continue
        aid, pages, primary = gold(c)
        if aid is not None and text_of and not _grounded(c, aid, pages, text_of):
            c.rejected = "gold pages do not contain the subject"
            continue
        top = best_score(c)
        # Two judgments from the same request, and each label needs both.
        #
        # WHICH ONE LEADS DIFFERS BY CLASS, and the first version of this got
        # it backwards. A positive is led by the page Score, because a label
        # has to name pages and only the Score knows which. A negative is led
        # by the Noul, because the Score cannot express "nothing answers this":
        # ask `Ile kosztuje "FOTOKOMÓRKI"?` of a corpus with no prices and the
        # photocell's own pages still score "Related topic only" (~0.4) — they
        # are about the right product, they just do not carry the fact. Every
        # price probe therefore landed in the unsure band and the negative
        # class came out empty, which is the opposite of what it measures.
        # The Noul said 0.04. `top < GOLD_SCORE` is now only the guard that
        # stops a negative being declared over a page that plainly answers.
        says_yes = (c.answerable or 0) >= AGREE_YES
        says_no = c.answerable is not None and c.answerable <= AGREE_NO
        if aid is not None and not says_yes:
            continue
        if aid is not None:
            out.append({"q": c.question, "kind": c.kind or "spec",
                        "doc": titles[aid], "pages": pages, "primary": primary,
                        "answerable": True,
                        "oracle": {"best": round(top, 3),
                                   "answerable": c.answerable,
                                   "scope": c.scope,
                                   "spilled": spilled(c),
                                   "from": c.page}})
        elif says_no and top < GOLD_SCORE:
            # In scope by construction — it was mined from these very pages —
            # and still unanswerable. That combination is the hardest negative
            # a corpus can produce and it cannot be written by hand.
            out.append({"q": c.question, "kind": c.kind or "spec",
                        "doc": titles[int(c.page.split(":")[0][1:])],
                        "pages": [], "primary": None, "answerable": False,
                        "oracle": {"best": round(top, 3),
                                   "answerable": c.answerable,
                                   "scope": c.scope, "spilled": 0,
                                   "from": c.page}})
        # Anything the two judgments do not agree on is dropped: the oracle is
        # unsure, and an eval label nobody believes is worse than a smaller set.
    return out


def dump(records: list[dict], out: Path = OUT) -> str:
    """YAML, written by hand so the file reads as documentation.

    pyyaml would fold the Polish into escapes and sort the keys into an order
    that hides what matters. This is a generated file people are meant to
    audit, so the question goes first and the provenance goes last.
    """
    def q(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [
        "# Generated by scripts/oracle.py — audit before trusting.",
        "#",
        "# `pages`/`primary` are 1-based page numbers inside `doc`, labelled by",
        "# an exhaustive Jev sweep of the corpus (scripts/xray.py), not by hand.",
        "# `answerable: false` rows are questions mined FROM this corpus that the",
        "# corpus still cannot answer — the negative class, and not fabricated.",
        "#",
        "# The labels come from Jev reading page text, so measuring `jev`,",
        "# `jev-page` or `xray` against them measures agreement, not accuracy.",
        "# `visual` (pixels) and `hybrid` (string match) are graded fairly.",
        "",
    ]
    for r in records:
        lines.append(f"- q: {q(r['q'])}")
        lines.append(f"  kind: {r['kind']}")
        lines.append(f"  doc: {q(r['doc'])}")
        lines.append(f"  pages: [{', '.join(str(p) for p in r['pages'])}]")
        if r["primary"] is not None:
            lines.append(f"  primary: {r['primary']}")
        lines.append(f"  answerable: {str(r['answerable']).lower()}")
        o = r["oracle"]
        lines.append(f"  oracle: {{best: {o['best']}, answerable: {o['answerable']}, "
                     f"scope: {o['scope']}, spilled: {o['spilled']}, from: {o['from']}}}")
        lines.append("")
    text = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def probes(candidates: list[Candidate], every: int = 2) -> list[Candidate]:
    """Probe questions over subjects that survived selection.

    Built from accepted candidates only, so the subject is known to be real and
    known to be in the corpus — which is what makes a negative from here a
    finding about the CORPUS rather than about a bad question. They skip stage 2
    because they are templated, not mined: their form is fixed and their subject
    has already been judged.
    """
    live = [c for c in candidates if c.rejected is None and c.question
            and (c.product or 0) >= PRODUCT]
    out = []
    for n, c in enumerate(live):
        if n % every:
            continue
        for kind, form in PROBES.items():
            out.append(Candidate(subject=c.subject, page=c.page,
                                 question=form.format(s=c.subject), kind=kind,
                                 well_formed=c.well_formed, specific=c.specific))
    return out


def build(limit: int | None = None, report: jev.Report | None = None,
          probe: bool = True
          ) -> tuple[list[Candidate], list[dict], dict[int, str]]:
    """The whole pipeline. Returns (candidates, eval records, doc titles)."""
    import lexical

    report = report if report is not None else jev.Report()
    corpus = lexical.pages()
    titles = {p["article_id"]: p["title"] for p in corpus}
    mined = subjects(corpus)
    if limit:
        # Even stride rather than the first N, so a truncated run still covers
        # both documents instead of stopping inside article 0.
        step = max(1, len(mined) // limit)
        mined = mined[::step][:limit]
    candidates = [Candidate(subject=s, page=k) for s, k in mined]
    select(candidates, report)
    if probe:
        candidates += probes(candidates)
    label(candidates, corpus, report)
    return candidates, rows(candidates, titles, corpus), titles


def main() -> None:
    import rag

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=40,
                    help="mined subjects to carry forward (default 40)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the set; write nothing")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    rag.configure_logging(args.log, stream=sys.stderr)

    if not jev.enabled():
        sys.exit("TYPESAFE_API_KEY is not set (or PIXELRAG_JEV=0) — "
                 "the oracle is the Jev sweep, so there is nothing to run.")

    started = time.perf_counter()
    report = jev.Report()
    candidates, records, _ = build(args.limit, report)
    ms = (time.perf_counter() - started) * 1000

    if args.json:
        print(json.dumps({"records": records, "rejected": [
            {"subject": c.subject, "why": c.rejected}
            for c in candidates if c.rejected], "bill": report.as_dict()},
            ensure_ascii=False, indent=2))
        return

    kept = [r for r in records if r["answerable"]]
    negative = [r for r in records if not r["answerable"]]
    rejected = Counter(c.rejected.split(" (")[0]
                       for c in candidates if c.rejected)
    print(f"mined {len(candidates)} subjects -> "
          f"{len(kept)} answerable, {len(negative)} labelled-unanswerable, "
          f"{len(candidates) - len(records)} dropped")
    for why, n in rejected.most_common():
        print(f"    {n:3d} dropped: {why}")
    unsure = len(candidates) - len(records) - sum(rejected.values())
    if unsure:
        print(f"    {unsure:3d} dropped: the two oracle judgments disagreed")
    cost = jev.price(report.input_tokens, report.output_tokens)
    print(f"\n{report.input_tokens:,} input tokens, "
          f"{len(report.calls)} calls, {ms / 1000:.1f}s, "
          f"${cost:.4f}" if cost is not None else "")
    spill = sum(r["oracle"]["spilled"] for r in records)
    if spill:
        print(f"{spill} gold page(s) fell outside the chosen document and were dropped")
    print()
    for r in kept[:12]:
        print(f"  ✓ {r['q'][:64]:64s} {r['doc'][:18]:18s} "
              f"p{r['primary']} {r['pages']}")
    for r in negative[:6]:
        print(f"  ∅ {r['q'][:64]:64s} best={r['oracle']['best']:.2f} "
              f"scope={r['oracle']['scope']}")

    if args.dry_run:
        print(f"\n--dry-run: {args.out} not written")
        return
    dump(records, args.out)
    print(f"\nwrote {len(records)} questions to {args.out}")
    print(f"now runnable:  .venv/bin/python scripts/evaluate_pl.py")


if __name__ == "__main__":
    main()
