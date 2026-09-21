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
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    return {a["title"]: i for i, a in enumerate(rag.articles())}


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

    @property
    def ms(self) -> float:
        """Median, not mean: one cold encoder load is not this mode's latency."""
        return round(statistics.median(self.latencies), 1) if self.latencies else 0.0

    def as_dict(self) -> dict:
        pct = lambda n, d: round(100 * n / d, 1) if d else None  # noqa: E731
        return {"mode": self.mode, "blocked": self.blocked,
                "questions": self.scored,
                "top1": pct(self.top1, self.scored),
                "recall": pct(self.recall, self.scored),
                "coverage": pct(self.covered, self.gold_total),
                "ceiling": pct(self.ceiling, self.gold_total),
                "refused": pct(self.refused, self.negatives),
                "negatives": self.negatives,
                "median_ms": self.ms, "cost_usd": round(self.cost, 5),
                "errors": self.errors, "circular": self.mode in CIRCULAR}


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
            if q.primary is not None and got and got[0] == (aid, q.primary):
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
    head = (f"{'mode':<11} {'top-1':>7} {'recall':>7} {'cover':>7} "
            f"{'ceil':>6} {'refuse':>7} {'ms':>7} {'$/q':>9}")
    lines = [head, "-" * len(head)]
    for r in results:
        if r.blocked:
            lines.append(f"{r.mode:<11}  — {r.blocked}")
            continue
        d = r.as_dict()
        star = "*" if d["circular"] else " "
        cost = f"{r.cost / r.scored:.5f}" if r.scored else "0"
        refuse = "—" if d["refused"] is None else f"{d['refused']:.0f}%"
        lines.append(
            f"{r.mode:<11}{star}{d['top1'] or 0:6.0f}% {d['recall'] or 0:6.0f}% "
            f"{d['coverage'] or 0:6.0f}% {d['ceiling'] or 0:5.0f}% "
            f"{refuse:>7} {r.ms:6.0f} {cost:>9}")
    return "\n".join(lines)


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
        print(json.dumps({"k": args.pages, "questions": len(questions),
                          "answerable": positives,
                          "rows": [r.as_dict() for r in results]},
                         ensure_ascii=False, indent=2))
        return

    print(f"\n{positives} answerable questions, "
          f"{len(questions) - positives} the corpus cannot answer, k={args.pages}\n")
    print(table(results))
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
