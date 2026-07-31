#!/usr/bin/env python
"""Start the three local services and wait until each is actually answering.

The README's start commands are macOS-flavoured (KMP_DUPLICATE_LIB_OK, --device
cpu) and predate the encoder sidecar. On this box — Windows, CUDA — the correct
invocation is different, and the ordering matters: the query encoder must be warm
before the first question or it pays 30s of model load inline.

    .venv/Scripts/python.exe scripts/serve_all.py          # all three, foreground
    .venv/Scripts/python.exe scripts/serve_all.py --stop    # kill whatever is up

Logs go to logs/. Ctrl-C stops everything.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

import layout

ROOT = layout.ROOT
INDEX = layout.DEFAULT
PY = sys.executable
LOGS = ROOT / "logs"

SEARCH_PORT = 30001
ENCODER_PORT = 8001
UI_PORT = 8000


def _up(url: str, timeout: float = 1.5) -> bool:
    try:
        return requests.get(url, timeout=timeout).ok
    except requests.RequestException:
        return False


def _wait(name: str, url: str, proc: subprocess.Popen, log: Path,
          limit: float = 300.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if proc.poll() is not None:
            sys.exit(f"{name} exited with code {proc.returncode}.\n"
                     f"  log: {log}\n"
                     f"  tail: {_tail(log)}")
        if _up(url):
            print(f"  {name} ready in {time.perf_counter() - t0:.0f}s", flush=True)
            return
        time.sleep(1.5)
    sys.exit(f"{name} did not become ready in {limit:.0f}s — see {log}")


def _tail(p: Path, n: int = 6) -> str:
    try:
        return "\n        ".join(p.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def stop() -> None:
    """Kill anything listening on our three ports."""
    import re

    # netstat emits OEM-codepage bytes that cp1250 cannot decode on a Polish
    # Windows locale, so decode explicitly rather than trusting text=True.
    out = subprocess.run(["netstat", "-ano"], capture_output=True,
                         text=True, encoding="utf-8", errors="replace").stdout
    pids = set()
    for line in out.splitlines():
        m = re.search(r":(\d+)\s+.*LISTENING\s+(\d+)", line)
        if m and int(m.group(1)) in (SEARCH_PORT, ENCODER_PORT, UI_PORT):
            pids.add(m.group(2))
    if not pids:
        print("nothing listening on 30001/8001/8000")
        return
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/F"],
                       capture_output=True, text=True)
        print(f"  killed pid {pid}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true", help="kill the services and exit")
    ap.add_argument("--no-ui", action="store_true", help="search + encoder only")
    args = ap.parse_args()

    if args.stop:
        stop()
        return

    os.chdir(ROOT)
    if not (ROOT / "index" / "index.faiss").exists():
        sys.exit("No index/index.faiss — build it first:\n"
                 "  .venv/Scripts/python.exe scripts/build_index.py")
    LOGS.mkdir(exist_ok=True)

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    procs: list[tuple[str, subprocess.Popen, Path]] = []

    def launch(name: str, cmd: list[str], log_name: str) -> subprocess.Popen:
        log = LOGS / log_name
        fh = open(log, "w", encoding="utf-8")
        print(f"starting {name} …", flush=True)
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                             cwd=ROOT)
        procs.append((name, p, log))
        return p

    # 1. faiss search API. OMP_NUM_THREADS=1 is not a tuning choice: faiss and
    #    torch each bundle their own libomp and the process dies without it.
    #    The encoder lives elsewhere precisely so that cap costs nothing.
    search_env_note = {**env, "OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    log = LOGS / "serve.log"
    fh = open(log, "w", encoding="utf-8")
    print("starting search API (faiss) …", flush=True)
    p_search = subprocess.Popen(
        [PY, "-m", "pixelrag_serve.api",
         "--index-dir", str(INDEX.index_dir),
         "--tiles-dir", str(INDEX.tiles_dir),
         "--articles-json", str(INDEX.articles_json),
         "--port", str(SEARCH_PORT), "--device", "cpu"],
        stdout=fh, stderr=subprocess.STDOUT, env=search_env_note, cwd=ROOT)
    procs.append(("search API", p_search, log))

    # 2. query encoder on CUDA, no faiss in the process, no thread cap.
    p_enc = launch("query encoder (CUDA)",
                   [PY, str(ROOT / "scripts" / "encoder.py")], "encoder.log")

    _wait("search API", f"http://127.0.0.1:{SEARCH_PORT}/health", p_search,
          LOGS / "serve.log")
    _wait("query encoder", f"http://127.0.0.1:{ENCODER_PORT}/health", p_enc,
          LOGS / "encoder.log")

    if not args.no_ui:
        p_ui = launch("web UI", [PY, str(ROOT / "scripts" / "app.py")], "app.log")
        _wait("web UI", f"http://127.0.0.1:{UI_PORT}/api/docs", p_ui, LOGS / "app.log")
        print(f"\n  open http://127.0.0.1:{UI_PORT}")

    print("\nCtrl-C to stop everything.")
    try:
        while True:
            for name, p, lg in procs:
                if p.poll() is not None:
                    print(f"\n{name} died (code {p.returncode}):\n        {_tail(lg)}")
                    raise KeyboardInterrupt
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopping …")
        for name, p, _ in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        time.sleep(1.5)
        for name, p, _ in procs:
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    main()
