#!/usr/bin/env python
"""Answer the whole eval set end to end and report answer + citation quality.

Retrieval is measured separately by evaluate_pl.py; this measures what actually
reaches the user. Three things worth knowing per question, none of which the
retrieval metric can see:

  cited page in ground truth  — did the reader answer from a page that really
                                covers the question, or from a near-miss?
  citation verified           — was the quote found in that page's text layer?
                                An unverified quote is the clearest signal the
                                reader drifted.
  family named                — the corpus prices the same option separately per
                                product family, so an answer that names no
                                family is ambiguous even when its number is right.

    GEMINI_API_KEY=... .venv/Scripts/python.exe scripts/answer_all.py
    ... scripts/answer_all.py --only 14 15      # just those questions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import rag
import yaml

EVAL = Path("eval/questions_pl.yaml")
OUT = Path("eval/answers.json")
CORPUS_DOC = "Cennik - Bramy garażowe"

FAMILIES = ["UniPro SNP", "UniPro", "PRIME", "RenoSystem", "DoorPro",
            "roletow", "uchyln", "Novum", "Progress", "Komfort", "Select",
            "City", "Connect", "MakroPro"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, metavar="N",
                    help="1-based question numbers to run")
    ap.add_argument("--mode", choices=["oneshot", "agent"], default=None)
    args = ap.parse_args()

    qs = yaml.safe_load(EVAL.read_text(encoding="utf-8"))
    t2i = {a["title"]: i for i, a in enumerate(rag.articles())}
    aid = t2i.get(CORPUS_DOC)
    if aid is None:
        sys.exit(f"{CORPUS_DOC} not in the index — rebuild first.")

    rows, results = [], []
    for i, q in enumerate(qs, 1):
        if args.only and i not in args.only:
            continue
        gold = {int(p) for p in (q.get("pages") or [])}
        if q.get("primary"):
            gold.add(int(q["primary"]))

        t0 = time.perf_counter()
        try:
            r = rag.run_agent(q["q"], mode=args.mode)
        except Exception as e:
            print(f"{i:2d}. ERROR {type(e).__name__}: {e}", flush=True)
            continue
        dt = time.perf_counter() - t0

        cites = r.get("citations") or []
        cited = {c["page"] for c in cites}
        on_gold = bool(cited & gold) if gold else None
        verified = sum(1 for c in cites if c["verified"])
        pinned = sum(len(c.get("numbers") or []) for c in cites)
        fam = [f for f in FAMILIES if f.lower() in r["answer"].lower()]
        refused = bool(re.search(
            r"nie ma|nie zawiera|nie znalaz|brak informacji|nie mogę",
            r["answer"], re.I))

        mark = "?" if on_gold is None else ("✓" if on_gold else "✗")
        print(f"{i:2d}. {mark} {q['q'][:46]:46s} "
              f"cyt={len(cites)}({verified}✓) pin={pinned} "
              f"str={sorted(cited) or '-'} rodz={','.join(fam[:2]) or '-'}"
              f"{' REFUSED' if refused else ''} {dt:.0f}s", flush=True)

        rows.append((i, on_gold, len(cites), verified, bool(fam), refused))
        results.append({
            "n": i, "q": q["q"], "kind": q.get("kind"),
            "gold_pages": sorted(gold), "cited_pages": sorted(cited),
            "cited_on_gold": on_gold, "citations": len(cites),
            "verified": verified, "pinned_numbers": pinned,
            "families_named": fam, "looks_like_refusal": refused,
            "answer": r["answer"],
            "usage": r["usage"], "seconds": round(dt, 1),
        })

    scored = [r for r in rows if r[1] is not None]
    if scored:
        print(f"\ncited a ground-truth page : {sum(r[1] for r in scored)}/{len(scored)}")
        print(f"answers with >=1 citation : {sum(1 for r in rows if r[2]) }/{len(rows)}")
        print(f"all citations verified    : "
              f"{sum(1 for r in rows if r[2] and r[2] == r[3])}/{len(rows)}")
        print(f"named a product family    : {sum(1 for r in rows if r[4])}/{len(rows)}")
    # A --only run must not destroy the full-set results: merge by question
    # number so re-testing three failures keeps the other seventeen.
    if args.only and OUT.exists():
        try:
            prev = {r["n"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, OSError):
            prev = {}
        prev.update({r["n"]: r for r in results})
        results = [prev[k] for k in sorted(prev)]
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nfull answers -> {OUT}")


if __name__ == "__main__":
    main()
