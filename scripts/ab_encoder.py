#!/usr/bin/env python
"""Compare two encoder configurations by alternating them query by query.

sweep_encoder.py runs one config per process, which is the right shape for
"what will a fresh sidecar do" but the wrong shape for "is bf16 faster than
fp16". This box drives a desktop, idles its SM clock at 517MHz of 3090MHz, and
gets measurably faster as a sweep proceeds — so in a sequence of process runs the
config that happened to run late looks better than it is. A 6ms apparent win was
pure position-in-round drift.

So: load both configs into one process and interleave A,B,A,B on the same query.
Every source of drift then hits both arms equally, and the statistic is the
paired difference per query rather than two independent distributions. The sign
test at the bottom is what decides a tie: if A does not win clearly more often
than it loses, the configs are tied and should be reported as tied.

Both models must fit in VRAM at once (two fp16 copies is ~9GB of 16GB here).

    python scripts/ab_encoder.py --a fp16/sdpa --b bf16/sdpa --n 100
    python scripts/ab_encoder.py --a fp16/sdpa --b fp16/eager --tokenise fast
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import profile_search as ps  # noqa: E402
from bench_search import query_set  # noqa: E402


def p95(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * len(s))) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="fp16/sdpa", help="dtype/attn, e.g. bf16/eager")
    ap.add_argument("--b", default="bf16/sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenise", choices=["stock", "memo", "fast"], default="fast")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    arms = {}
    for name in ("a", "b"):
        dtype, attn = getattr(args, name).split("/")
        # tokenise=memo/fast patch global or per-processor state; applying the same
        # setting to both arms keeps the comparison about dtype and attention.
        torch, model, processor = ps.load(args.device, dtype, attn, args.tokenise)
        arms[name] = (getattr(args, name), torch, model, processor)

    qs = query_set(args.n + args.warmup)
    for text in qs[:args.warmup]:
        for _, torch, model, processor in arms.values():
            ps.encode(torch, model, processor, text, args.device)

    acc = {"a": [], "b": []}
    fwd = {"a": [], "b": []}
    for i, text in enumerate(qs[args.warmup:]):
        # Swap which arm goes first on alternate queries: the second call on a
        # query runs against a hotter cache and would otherwise always be arm b.
        order = ("a", "b") if i % 2 == 0 else ("b", "a")
        for name in order:
            _, torch, model, processor = arms[name]
            r = ps.time_phases(torch, model, processor, text, args.device)
            acc[name].append(r["TOTAL"])
            fwd[name].append(r["forward"])

    la, lb = arms["a"][0], arms["b"][0]
    print(f"\n{'arm':<16}{'TOTAL med':>11}{'TOTAL p95':>11}{'fwd med':>10}{'fwd p95':>10}")
    print("-" * 58)
    for name, label in (("a", la), ("b", lb)):
        print(f"{label:<16}{statistics.median(acc[name]):>11.2f}{p95(acc[name]):>11.2f}"
              f"{statistics.median(fwd[name]):>10.2f}{p95(fwd[name]):>10.2f}")

    d = [x - y for x, y in zip(acc["a"], acc["b"])]
    wins = sum(1 for x in d if x < 0)
    print(f"\npaired delta per query (a - b), n={len(d)}:")
    print(f"  median {statistics.median(d):+.2f}ms  mean {statistics.fmean(d):+.2f}ms  "
          f"stdev {statistics.stdev(d):.2f}ms")
    print(f"  {la} faster on {wins}/{len(d)} queries ({wins/len(d):.0%})")
    # 40-60% is coin-flip territory for n=100; call that tied rather than
    # promoting a sub-millisecond median difference to a recommendation.
    verdict = ("TIED" if 0.40 <= wins / len(d) <= 0.60
               else f"{la if wins/len(d) > 0.5 else lb} wins")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
