#!/usr/bin/env python
"""Every retrieval mode, over the whole question set, in one table.

`evaluate_pl.py` measures one strategy at a time and hard-codes
`rag.searcher("visual")` inside each of them, which was right when the question
was "does page aggregation beat top-k chunks". The question now is which of the
seven named modes to run, and that needs them side by side on the same
questions with the same k — including the two columns the README says are the
real price: wall clock and dollars.

    .venv/bin/python scripts/bench.py                    # every runnable mode
    .venv/bin/python scripts/bench.py --modes visual,jev,xray
    .venv/bin/python scripts/bench.py --json

Reading the table, in the order the numbers deserve:

  top-1      the oracle's single best page came back at rank 1
  recall@k   any acceptable page came back at all
  cover@k    what fraction of the acceptable pages were retrieved, against the
             most any mode could have retrieved at this k (`ceiling`), because
             a question with 9 gold pages cannot score 100% at k=4 and counting
             it against the mode measures the question, not the retriever
  refuse     of the questions the corpus CANNOT answer, how many the gate
             caught — the only column where the negatives count for anything

BEFORE READING A WINNER OUT OF IT: the labels come from Jev reading page text
(scripts/oracle.py), so `jev`, `jev-page` and `xray` are graded by a judge that
shares their evidence and their model. Their rows measure agreement with the
oracle and are not evidence of accuracy. `visual` is graded fairly — pixels
against text — and `hybrid` nearly so. The header prints this next to the rows
that need it, because a table is read long after its caveat is forgotten.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import corpus
import jev
import rag

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

EVAL = Path("eval/questions_pl.yaml")
# Modes whose ranking evidence is the same text the oracle labelled from.
CIRCULAR = {"jev", "jev+hybrid", "jev-page", "xray"}


@dataclass
class Question:
    q: str
    kind: str
    doc: str
    pages: list[int]
    primary: int | None
    answerable: bool
    oracle: dict = field(default_factory=dict)


def load(path: Path = EVAL) -> list[Question]:
    """The oracle's question set, keyed to document titles rather than ids.

    Article ids are positions in the sorted pdfs/ glob and move whenever the
    corpus does; titles do not. Same reasoning as evaluate_pl.py, same file.
    """
    import yaml

    if not path.exists():
        sys.exit(f"{path} missing — run scripts/oracle.py to build it.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Question(q=r["q"], kind=r.get("kind", "spec"), doc=str(r["doc"]),
                     pages=list(r.get("pages") or []), primary=r.get("primary"),
                     answerable=bool(r.get("answerable", True)),
                     oracle=r.get("oracle") or {})
            for r in raw]


def title_to_id() -> dict[str, int]:
    return {a["title"]: i for i, a in enumerate(corpus.articles())}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval on a proportion, Wilson score.

    Not the textbook normal interval: at n=13 and p near 1.0 that one produces
    bounds above 100% and is simply wrong at the end of the scale this table
    lives at. Wilson stays inside [0, 1] and does not collapse to zero width
    when a mode gets everything right.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on two modes' per-question top-1 outcomes.

    THE ONLY COMPARISON THAT MEANS ANYTHING BETWEEN TWO ROWS OF THIS TABLE.
    Both modes answer the SAME questions, so the interesting count is not how
    many each got right but on how many they disagreed: the questions both got
    right carry no information about which is better, and neither do the
    questions both got wrong. Only the discordant pairs do, and there are
    usually one or two of them.

    Returns (a-wins, b-wins, p). p is the exact binomial sign test on the
    discordant pairs — no chi-square approximation, which needs counts this
    question set will never have.
    """
    shared = a.keys() & b.keys()
    wins_a = sum(1 for q in shared if a[q] and not b[q])
    wins_b = sum(1 for q in shared if b[q] and not a[q])
    n = wins_a + wins_b
    if n == 0:
        return (0, 0, 1.0)
    tail = sum(math.comb(n, i) for i in range(min(wins_a, wins_b) + 1))
    return (wins_a, wins_b, min(1.0, 2 * tail / (2 ** n)))


def _subject(question: str) -> str:
    """The mined heading a generated question was built around.

    oracle.py emits exactly two shapes — `... "SUBJECT" ...` and
    `SUBJECT — ...` — so recovering the subject is a string operation, not a
    guess. A hand-written question has neither shape and simply does not
    qualify for the verified subset, which is the correct outcome.
    """
    quoted = re.search(r'"([^"]+)"', question)
    if quoted:
        return quoted.group(1)
    return question.split(" — ")[0]


def verified(questions: list[Question], t2i: dict[str, int]) -> list[Question]:
    """The subset whose gold page can be checked without asking a model.

    THE POINT OF THIS FUNCTION is that every other number in this file is
    graded by Jev reading page text, which makes `jev`, `jev-page` and `xray`
    partly self-marked. Here the answer is decided by `str.__contains__`: keep
    only the questions whose subject string occurs on exactly ONE page of the
    corpus, and make that page the gold. No model has an opinion, so no mode
    can be circular against it.

    Measured on the generated set: 13 of 23 questions qualify, and the Jev
    oracle independently agreed with the string on 12 of those 13 — which is
    the only evidence in this repo that the oracle is worth anything at all.
    The disagreement is a continuation page, the same failure the README
    documents for `kolory tkanin soltis`.
    """
    import lexical
    import oracle

    flat = {(p["article_id"], p["page"]): oracle._flat(p["text"])
            for p in lexical.pages()}
    id_to_title = {i: t for t, i in t2i.items()}
    out = []
    for q in questions:
        if not q.answerable:
            continue
        needle = oracle._flat(_subject(q.q))
        hits = [key for key, text in flat.items() if needle in text]
        if len(hits) != 1:
            continue
        aid, page = hits[0]
        if aid not in id_to_title:
            continue
        out.append(Question(q=q.q, kind=q.kind, doc=id_to_title[aid],
                            pages=[page], primary=page, answerable=True,
                            oracle={**q.oracle, "verified_by": "literal string",
                                    "agreed": [aid, page] == [t2i[q.doc], q.primary]}))
    return out


@dataclass
class Result:
    """One mode's run over the whole set."""

    mode: str
    blocked: str | None = None
    top1: int = 0
    recall: int = 0
    covered: int = 0
    ceiling: int = 0
    gold_total: int = 0
    scored: int = 0
    refused: int = 0
    negatives: int = 0
    latencies: list[float] = field(default_factory=list)
    cost: float = 0.0
    errors: int = 0
    misses: list[tuple[str, list]] = field(default_factory=list)
    # question -> did this mode put the gold page at rank 1. Kept per question
    # rather than summed, because comparing two modes needs to know WHICH ones
    # they disagreed about — see mcnemar().
    per_q: dict[str, bool] = field(default_factory=dict)

    @property
    def ms(self) -> float:
        """Median, not mean: one cold encoder load is not this mode's latency."""
        return round(statistics.median(self.latencies), 1) if self.latencies else 0.0

    def as_dict(self) -> dict:
        pct = lambda n, d: round(100 * n / d, 1) if d else None  # noqa: E731
        lo, hi = wilson(self.top1, self.scored)
        return {"mode": self.mode, "blocked": self.blocked,
                "questions": self.scored,
                "top1": pct(self.top1, self.scored),
                "top1_n": self.top1,
                "top1_ci95": [round(100 * lo, 1), round(100 * hi, 1)],
                "recall": pct(self.recall, self.scored),
                "coverage": pct(self.covered, self.gold_total),
                "ceiling": pct(self.ceiling, self.gold_total),
                "refused": pct(self.refused, self.negatives),
                "negatives": self.negatives,
                "median_ms": self.ms, "cost_usd": round(self.cost, 5),
                "errors": self.errors, "circular": self.mode in CIRCULAR,
                # Per question, so two --json runs made at different times (or
                # at different budgets) can be compared pairwise offline. A
                # mode costs real money per run; re-running one only to learn
                # which questions it disagreed with another about is a bill
                # this field exists to avoid.
                "per_question": self.per_q}


def run_mode(mode: str, questions: list[Question], k: int,
             gate: bool = True) -> Result:
    """One mode over every question, retrieval only — no reader, ever.

    The gate is asked on the negatives only. It costs a TypeSafe call per
    question and its answer is meaningless for a question whose answer IS in
    the corpus: `refuse` measures whether a system would have declined to spend
    a reader call it could not have used, so the questions where it should have
    spent one are not the experiment.
    """
    result = Result(mode=mode)
    blocked = rag.retrieval_blocked(mode)
    if blocked:
        result.blocked = blocked
        return result
    t2i = title_to_id()

    for q in questions:
        aid = t2i.get(q.doc)
        if aid is None:
            continue
        started = time.perf_counter()
        try:
            pages, _, stats = rag.retrieve_for(q.q, mode, n_pages=k)
        except Exception as exc:  # noqa: BLE001 — a broken mode is a row, not a crash
            result.errors += 1
            print(f"  ! {mode}: {type(exc).__name__} on {q.q[:40]!r}",
                  file=sys.stderr)
            continue
        result.latencies.append((time.perf_counter() - started) * 1000)
        result.cost += stats.get("cost_usd") or 0.0
        got = [(p.article_id, p.page) for p in pages]

        if q.answerable:
            result.scored += 1
            gold = {(aid, p) for p in q.pages}
            result.gold_total += len(gold)
            # The most any retriever could have got at this k. A question with
            # nine acceptable pages is not a failure of the mode that returned
            # four of them.
            result.ceiling += min(len(gold), k)
            hit = [g for g in got if g in gold]
            result.covered += len(hit)
            if hit:
                result.recall += 1
            is_top1 = bool(q.primary is not None and got
                           and got[0] == (aid, q.primary))
            result.per_q[q.q] = is_top1
            if is_top1:
                result.top1 += 1
            elif not hit:
                result.misses.append((q.q, got))
        else:
            result.negatives += 1
            if not gate or not jev.enabled():
                continue
            rows = [{"article_id": a, "tile_index": p - 1} for a, p in got]
            try:
                verdict = jev.gate(q.q, rag._page_texts(rows))
            except Exception:  # noqa: BLE001
                result.errors += 1
                continue
            if verdict.verdict(rag.JEV_REFUSE or 0.5) != "ok":
                result.refused += 1
    return result


def table(results: list[Result]) -> str:
    head = (f"{'mode':<11} {'top-1':>7} {'95% CI':>11} {'recall':>7} "
            f"{'cover':>7} {'ceil':>6} {'refuse':>7} {'ms':>7} {'$/q':>9}")
    lines = [head, "-" * len(head)]
    for r in results:
        if r.blocked:
            lines.append(f"{r.mode:<11}  — {r.blocked}")
            continue
        d = r.as_dict()
        star = "*" if d["circular"] else " "
        cost = f"{r.cost / r.scored:.5f}" if r.scored else "0"
        refuse = "—" if d["refused"] is None else f"{d['refused']:.0f}%"
        lo, hi = d["top1_ci95"]
        ci = f"[{lo:.0f}-{hi:.0f}]" if r.scored else "—"
        lines.append(
            f"{r.mode:<11}{star}{d['top1'] or 0:6.0f}% {ci:>11} "
            f"{d['recall'] or 0:6.0f}% "
            f"{d['coverage'] or 0:6.0f}% {d['ceiling'] or 0:5.0f}% "
            f"{refuse:>7} {r.ms:6.0f} {cost:>9}")
    return "\n".join(lines)


def resolution(results: list[Result]) -> str:
    """Which differences in the table above are differences, and which are one
    question.

    THIS BLOCK EXISTS BECAUSE THE TABLE IS PERSUASIVE AND THE QUESTION SET IS
    SMALL. At n=13 a single question is 7.7 points, so two rows seven points
    apart differ by one question and a row ordering is not a ranking. Every
    pair is tested head to head on the questions they disagreed about, which is
    the only evidence in the table about which mode is better.
    """
    scored = [r for r in results if not r.blocked and r.per_q]
    if len(scored) < 2:
        return ""
    n = max(r.scored for r in scored)
    step = 100.0 / n if n else 0.0
    out = [f"\nResolution: {n} questions, so one question is {step:.1f} points.",
           ("Head to head on the questions two modes disagreed about "
            "(exact McNemar):"), ""]
    best = max(scored, key=lambda r: r.top1)
    for r in scored:
        if r is best:
            continue
        wins_b, wins_r, p = mcnemar(best.per_q, r.per_q)
        disc = wins_b + wins_r
        if disc == 0:
            verdict = "identical on every question"
        elif p > 0.05:
            verdict = f"p={p:.3f} — not resolved by this set"
        else:
            verdict = f"p={p:.3f} — resolved"
        out.append(f"  {best.mode:<11} vs {r.mode:<11} "
                   f"{wins_b}-{wins_r} of {disc:>2} discordant   {verdict}")
    out.append("\n  A row that beats another by one or two questions has not "
               "been shown to be\n  better than it. Widen eval/questions_pl.yaml "
               "(scripts/oracle.py) before\n  reading a winner out of the "
               "ordering.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modes", help="comma-separated subset")
    ap.add_argument("-k", "--pages", type=int, default=rag.ONESHOT_PAGES)
    ap.add_argument("--eval", type=Path, default=EVAL)
    ap.add_argument("--verified", action="store_true",
                    help="only questions with string-checkable gold — no mode "
                         "can be circular against it")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the refusal column (saves a call per negative)")
    ap.add_argument("--misses", action="store_true", help="list what each mode missed")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    rag.configure_logging(args.log, stream=sys.stderr)

    questions = load(args.eval)
    if args.verified:
        negatives = [q for q in questions if not q.answerable]
        questions = verified(questions, title_to_id()) + negatives
    names = ([m.strip() for m in args.modes.split(",")] if args.modes
             else list(rag.RETRIEVAL_MODES))
    positives = sum(1 for q in questions if q.answerable)

    # Warm the encoder outside the measurement, exactly as compare.py does:
    # the first mode to touch the index otherwise pays ~26 s of model load and
    # the table reports it as that mode's latency.
    try:
        rag.searcher("visual")("rozgrzewka", 1)
    except Exception:  # noqa: BLE001
        pass

    results = []
    for name in names:
        print(f"  {name} …", file=sys.stderr, flush=True)
        results.append(run_mode(name, questions, args.pages,
                                gate=not args.no_gate))

    if args.json:
        # Every pair, not just each against the best. The printed block stays
        # short because a terminal table is read top to bottom; the JSON is
        # read by something that wants the whole matrix.
        scored = [r for r in results if not r.blocked and r.per_q]
        pairs = []
        for i, a in enumerate(scored):
            for b in scored[i + 1:]:
                wins_a, wins_b, p = mcnemar(a.per_q, b.per_q)
                pairs.append({"a": a.mode, "b": b.mode, "a_wins": wins_a,
                              "b_wins": wins_b, "discordant": wins_a + wins_b,
                              "p": round(p, 4)})
        print(json.dumps({"k": args.pages, "questions": len(questions),
                          "answerable": positives,
                          "rows": [r.as_dict() for r in results],
                          "resolution": pairs},
                         ensure_ascii=False, indent=2))
        return

    print(f"\n{positives} answerable questions, "
          f"{len(questions) - positives} the corpus cannot answer, k={args.pages}\n")
    print(table(results))
    print(resolution(results))
    if args.verified:
        agreed = sum(1 for q in questions if q.oracle.get("agreed"))
        print(f"\n  gold decided by literal string match, not by a model — "
              f"no row is circular.")
        print(f"  the Jev oracle independently agreed with the string on "
              f"{agreed}/{positives}.")
    else:
        print("\n* graded by an oracle that shares this mode's model and evidence "
              "— agreement, not accuracy. Re-run with --verified for a referee "
              "no mode shares.")
    print("  cover is of every acceptable page; ceil is the most any mode "
          "could reach at this k.")
    if args.misses:
        for r in results:
            if r.misses:
                print(f"\n{r.mode} missed {len(r.misses)}:")
                for q, got in r.misses:
                    print(f"  {q[:58]:58s} -> {got}")


if __name__ == "__main__":
    main()
