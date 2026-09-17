#!/usr/bin/env python
"""Local chat UI over the visual index, showing the agent's browsing path.

Thin web front end over rag.py. Local rather than a static page because it needs
the Anthropic key server-side and has to reach the search API on localhost.

    # 1. search API (leave running)
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/pixelrag serve \
      --index-dir ./index --tiles-dir ./index/tiles \
      --articles-json ./index/articles.json --port 30001 --device cpu

    # 2. this UI
    export ANTHROPIC_API_KEY=...           # or paste a key in the UI
    .venv/bin/python scripts/app.py        # http://127.0.0.1:8000
"""

import json
import queue
import sys
import threading
from pathlib import Path
from urllib.parse import quote, unquote

import requests
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import compare as compare_modes
import rag
import snip

HERE = Path(__file__).parent
app = FastAPI(title="PixelRAG manuals")

# pdf.js, vendored (scripts/static/vendor). Local rather than a CDN because the
# app is expected to work on a box with no internet, and because the worker
# script must be same-origin.
app.mount("/vendor", StaticFiles(directory=HERE / "static" / "vendor"), name="vendor")


class Ask(BaseModel):
    question: str
    api_key: str | None = None  # optional paste from the UI; overrides env for this ask
    # Which retrieval mode answers. None/"auto" is the environment's default,
    # which is what every client that predates the picker sends.
    retrieval: str | None = None


class Compare(BaseModel):
    """Retrieval only, several modes, one question. No reader, no image tokens."""

    question: str
    modes: list[str] | None = None
    pages: int | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    # Explicit UTF-8: Windows locale encoding mangles Polish (ąęł…) and dashes.
    return HTMLResponse(
        (HERE / "static" / "index.html").read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8",
    )


@app.get("/api/modes")
def retrieval_modes():
    """The retrieval modes this box can actually run, and why not where it can't.

    The UI greys out a blocked mode and shows the reason rather than hiding it:
    "jev — TYPESAFE_API_KEY is not set" is a setup instruction, and a mode that
    silently vanished is a bug report.
    """
    return {"modes": compare_modes.modes(), "default": rag.default_retrieval(),
            "pages": rag.ONESHOT_PAGES}


@app.post("/api/compare")
def compare_retrieval(req: Compare):
    """Stream one row per retrieval mode: what it found and what it cost.

    Server-sent events for the same reason /api/ask uses them — each mode is a
    second or more of real retrieval and they run sequentially by design (see
    compare.py), so the table fills in as the modes finish instead of arriving
    all at once after the slowest one.
    """
    def stream():
        try:
            rows = []
            for row in compare_modes.compare(req.question, req.modes, req.pages):
                rows.append(row)
                yield _sse({"kind": "row", "payload": row})
            yield _sse({"kind": "done",
                        "payload": {"agreement": compare_modes.agreement(rows)}})
        except Exception as e:
            yield _sse({"kind": "error", "payload": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/docs")
def list_docs():
    out = []
    for i, a in enumerate(rag.articles()):
        out.append({"article_id": i, "title": a["title"],
                    "pages": len(rag.LAYOUT.page_images(i))})
    return {"docs": out}


@app.get("/api/page/{article_id}/{page}")
def page_image(article_id: int, page: int):
    """Full page image; `page` is 1-based, matching what the UI shows."""
    p = rag.page_path(article_id, page - 1)
    if not p.exists():
        return JSONResponse({"error": "no such page"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/snip/{article_id}/{page}")
def citation_snip(
    article_id: int,
    page: int,
    rects: str | None = Query(
        None,
        description="JSON list of {left,top,width,height} in %% of the page",
    ),
    pad: float = Query(4.0, ge=0, le=40, description="padding around the union box, %%"),
):
    """Crop of the rendered page around citation rectangles, for chat inline.

    Rects are the same percent-of-page boxes `citations.py` and the PDF viewer
    use. What the crop actually shows — padding, minimum size, the fallback band
    for an unverified citation — is snip.py's business, not this route's.
    """
    src = rag.page_path(article_id, page - 1)
    if not src.exists():
        return JSONResponse({"error": "no such page"}, status_code=404)

    boxes = []
    if rects:
        try:
            boxes = snip.parse_boxes(json.loads(unquote(rects)))
        except json.JSONDecodeError:
            return JSONResponse({"error": "bad rects json"}, status_code=400)

    return Response(
        snip.render(src, boxes, pad=pad),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/pdf/{article_id}")
def source_pdf(article_id: int):
    """The original PDF — this is what the viewer actually renders.

    Served with byte ranges (Starlette's FileResponse honours Range) so pdf.js
    can fetch only the pages it draws. That is not a micro-optimisation: the gate
    price list is 82 MB, and without ranges every question would pull the whole
    file before the first pixel appears.

    `inline`, not `attachment`: the same URL backs the "otwórz PDF ↗" link, and
    a download prompt there is not what the user asked for.
    """
    arts = rag.articles()
    if not 0 <= article_id < len(arts):
        return JSONResponse({"error": "no such document"}, status_code=404)
    src = Path(arts[article_id].get("url") or "")
    if not src.exists():
        return JSONResponse({"error": "source file missing"}, status_code=404)
    # Headers go out as latin-1, so a Polish file name ("Cennik - Bramy
    # garażowe.pdf") crashes the response unless it is percent-encoded per
    # RFC 5987, with an ASCII-stripped fallback for anything that ignores that.
    ascii_name = src.name.encode("ascii", "replace").decode("ascii").replace('"', "")
    disp = (f'inline; filename="{ascii_name}"; '
            f"filename*=UTF-8''{quote(src.name, safe='')}")
    return FileResponse(
        src, media_type="application/pdf",
        headers={"Content-Disposition": disp,
                 "Cache-Control": "private, max-age=86400"},
    )


@app.get("/api/tile/{article_id}/{tile_index}/{chunk_index}")
def tile_image(article_id: int, tile_index: int, chunk_index: int):
    """Proxy a single region image, so the UI can show exactly what the agent saw.

    Through rag's pooled session, not a bare requests.get: this fires once per
    region the UI draws, which is precisely the traffic the connection pool was
    introduced for — a handshake and a lingering TIME_WAIT per call otherwise.
    """
    try:
        r = rag._HTTP.get(
            f"{rag.SEARCH_API}/tile/{article_id}/{tile_index}/{chunk_index}", timeout=60)
        if not r.ok:
            return JSONResponse({"error": "tile fetch failed"}, status_code=502)
        return StreamingResponse(iter([r.content]),
                                 media_type=r.headers.get("content-type", "image/png"))
    except requests.RequestException as e:
        return JSONResponse({"error": str(e)}, status_code=503)


@app.post("/api/ask")
def ask(req: Ask):
    """Server-sent events: the agent's steps as they happen, then the answer.

    Streaming matters here — a browsing agent takes several model round trips,
    and watching it search and open tiles is most of the diagnostic value.
    """
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            key = (req.api_key or "").strip() or None
            result = rag.run_agent(
                req.question,
                on_event=lambda ev: q.put(("event", ev)),
                api_key=key,
                retrieval=req.retrieval,
            )
            q.put(("done", result))
        except Exception as e:
            q.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            q.put((None, None))

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            kind, payload = q.get()
            if kind is None:
                break
            yield _sse({"kind": kind, "payload": payload})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    rag.configure_logging()
    if not rag.INDEX_DIR.exists():
        sys.exit(f"No index at {rag.INDEX_DIR} — build it first (see README).")
    # The query encoder lives here rather than in the faiss process (see rag.py).
    # Load it now, in the background, so the first question doesn't pay for it.
    if rag.LOCAL_ENCODE:
        threading.Thread(target=rag.warm_encoder, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
