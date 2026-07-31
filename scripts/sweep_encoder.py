#!/usr/bin/env python
"""Run profile_search.py over a set of encoder configurations and rank them.

One subprocess per (config, repeat) on purpose. A config measured three times in
one process shares a warmed allocator, a hot clock and one set of autotuned
cuBLAS choices, which makes it look more repeatable than it is; a fresh process
per repeat is what a restarted sidecar will actually see. The spread across
repeats is reported, and it is the only honest basis for calling two configs
tied.

Repeats run round-robin across configs rather than all-of-one-then-the-next.
This GPU drives a desktop and idles at 517MHz of 3090MHz, so a 15-minute sweep
sees real clock and contention drift; grouping the repeats would hand that drift
to whichever config ran during the quiet stretch.

profile_search.py stays the instrument — this only drives it and parses its
table, so there is one implementation of the CUDA-synchronised timing.

    python scripts/sweep_encoder.py --repeats 3 --n 40
    python scripts/sweep_encoder.py --only fp16/sdpa/fast --repeats 5
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (label, dtype, attn, tokenise, compile, pad_to). Labels are what the report ranks.
CONFIGS = [
    ("fp16/sdpa/stock",           "fp16", "sdpa",  "stock", "off",             0),
    ("fp16/sdpa/memo",            "fp16", "sdpa",  "memo",  "off",             0),
    ("fp16/sdpa/fast",            "fp16", "sdpa",  "fast",  "off",             0),
    ("bf16/sdpa/fast",            "bf16", "sdpa",  "fast",  "off",             0),
    ("fp16/eager/fast",           "fp16", "eager", "fast",  "off",             0),
    ("bf16/eager/fast",           "bf16", "eager", "fast",  "off",             0),
    ("fp32/sdpa/fast",            "fp32", "sdpa",  "fast",  "off",             0),
    # Inductor modes are listed to be shown failing: there is no Triton wheel for
    # Windows here, so all three raise TritonMissing on the first call.
    ("fp16/sdpa/fast+inductor",   "fp16", "sdpa",  "fast",  "default",         0),
    ("fp16/sdpa/fast+reduceovh",  "fp16", "sdpa",  "fast",  "reduce-overhead", 0),
    ("fp16/sdpa/fast+graphs-dyn", "fp16", "sdpa",  "fast",  "cudagraphs",      0),
    # Longest query in eval/questions.yaml tokenises to 64, so pad 64 is one
    # shape for every query with nothing truncated.
    ("fp16/sdpa/pad64+graphs",    "fp16", "sdpa",  "fast",  "cudagraphs",     64),
    ("bf16/sdpa/pad64+graphs",    "bf16", "sdpa",  "fast",  "cudagraphs",     64),
    ("fp16/eager/pad64+graphs",   "fp16", "eager", "fast",  "cudagraphs",     64),
    ("fp16/sdpa/pad64",           "fp16", "sdpa",  "fast",  "off",            64),
]

_ROW = re.compile(r"^(template|tokenise|to_device|forward|pool|to_host|TOTAL)\s+"
                  r"([\d.]+)\s+([\d.]+)\s+([\d.]+)")
_WARM = re.compile(r"^warmup (\d+): (\d+)ms")


def run(dtype, attn, tokenise, compile_mode, pad_to, n, warmup, device):
    cmd = [sys.executable, str(HERE / "profile_search.py"), "--mode", "phases",
           "--device", device, "--dtype", dtype, "--attn", attn,
           "--tokenise", tokenise, "--compile", compile_mode,
           "--pad-to", str(pad_to), "--n", str(n), "--warmup", str(warmup)]
    # HF_HUB_OFFLINE because a hub lookup mid-run is seconds of network in the
    # middle of a millisecond measurement.
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE.parent,
                       env={**os.environ, "HF_HUB_OFFLINE": "1"})
    if p.returncode != 0:
        return None, p.stderr.strip().splitlines()[-3:]
    phases, warmups = {}, []
    for line in p.stdout.splitlines():
        m = _ROW.match(line.strip())
        if m:
            phases[m[1]] = {"mean": float(m[2]), "median": float(m[3]), "p95": float(m[4])}
        w = _WARM.match(line.strip())
        if w:
            warmups.append(int(w[2]))
    return {"phases": phases, "warmups": warmups}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", action="append",
                    help="label to run; repeatable. default: all")
    args = ap.parse_args()

    todo = [c for c in CONFIGS if not args.only or c[0] in args.only]
    runs: dict[str, list] = {c[0]: [] for c in todo}
    for r in range(args.repeats):
        for label, dtype, attn, tok, comp, pad in todo:
            res, err = run(dtype, attn, tok, comp, pad, args.n, args.warmup,
                           args.device)
            if res is None:
                print(f"{label} repeat {r}: FAILED {err}", flush=True)
                continue
            runs[label].append(res)
            t = res["phases"]["TOTAL"]
            print(f"{label} repeat {r}: TOTAL p95 {t['p95']:.2f} mean {t['mean']:.2f} "
                  f"forward p95 {res['phases']['forward']['p95']:.2f} "
                  f"first-call {res['warmups'][0] if res['warmups'] else '?'}ms",
                  flush=True)

    def med(xs):
        return statistics.median(xs)

    rows = [(label, rs) for label, rs in runs.items() if rs]
    print(f"\n{'config':<28}{'p95 med':>9}{'p95 min':>9}{'p95 max':>9}"
          f"{'fwd med':>9}{'tok mean':>10}{'first call':>12}")
    print("-" * 86)
    for label, rs in sorted(rows, key=lambda r: med([x["phases"]["TOTAL"]["p95"]
                                                     for x in r[1]])):
        p95s = [x["phases"]["TOTAL"]["p95"] for x in rs]
        fwd = [x["phases"]["forward"]["p95"] for x in rs]
        tok = [x["phases"]["tokenise"]["mean"] for x in rs]
        first = [x["warmups"][0] for x in rs if x["warmups"]]
        print(f"{label:<28}{med(p95s):>9.2f}{min(p95s):>9.2f}{max(p95s):>9.2f}"
              f"{med(fwd):>9.2f}{med(tok):>10.2f}"
              f"{(med(first) if first else 0):>11.0f}ms")
    print(f"\nn={args.n} per run, {len(rs)} runs per config, warmup={args.warmup} "
          f"excluded. Columns are the median across repeats except p95 min/max, "
          f"which bound the run-to-run spread. `first call` is the first encode "
          f"after load — the compile cost, where there is one.")


if __name__ == "__main__":
    main()
