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

import io
import json
import queue
import sys
import threading
from pathlib import Path
from urllib.parse import quote, unquote

import rag
import requests
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).parent
app = FastAPI(title="PixelRAG manuals")

# pdf.js, vendored (scripts/static/vendor). Local rather than a CDN because the
# app is expected to work on a box with no internet, and because the worker
# script must be same-origin.
app.mount("/vendor", StaticFiles(directory=HERE / "static" / "vendor"), name="vendor")


class Ask(BaseModel):
    question: str
    api_key: str | None = None  # optional paste from the UI; overrides env for this ask


@app.get("/", response_class=HTMLResponse)
def index():
    # Explicit UTF-8: Windows locale encoding mangles Polish (ąęł…) and dashes.
    return HTMLResponse(
        (HERE / "static" / "index.html").read_text(encoding="utf-8"),
        media_type="text/html; charset=utf-8",
    )


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

    Rects are the same percent-of-page boxes `citations.py` / the PDF viewer use.
    When `rects` is empty or missing, returns a mid-height strip of the page so
    the chat still has *something* visual for an unverified citation.
    """
    from PIL import Image, ImageDraw

    src = rag.page_path(article_id, page - 1)
    if not src.exists():
        return JSONResponse({"error": "no such page"}, status_code=404)

    boxes: list[dict] = []
    if rects:
        try:
            raw = json.loads(unquote(rects))
            if isinstance(raw, list):
                boxes = [b for b in raw if isinstance(b, dict)]
        except json.JSONDecodeError:
            return JSONResponse({"error": "bad rects json"}, status_code=400)

    with Image.open(src) as im:
        im = im.convert("RGB")
        pw, ph = im.size
        if boxes:
            left = min(float(b.get("left", 0)) for b in boxes)
            top = min(float(b.get("top", 0)) for b in boxes)
            right = max(float(b.get("left", 0)) + float(b.get("width", 0)) for b in boxes)
            bottom = max(float(b.get("top", 0)) + float(b.get("height", 0)) for b in boxes)
        else:
            # Unverified / no text layer: a readable band, not the whole page.
            left, right = 4.0, 96.0
            top, bottom = 28.0, 72.0

        left = max(0.0, left - pad)
        top = max(0.0, top - pad)
        right = min(100.0, right + pad)
        bottom = min(100.0, bottom + pad)
        # Thin quote lines need air so the row label stays in frame.
        # Large retrieval regions already fill the crop — leave them alone.
        if boxes and (bottom - top) < 8.0:
            mid = (top + bottom) / 2
            top, bottom = max(0.0, mid - 5.0), min(100.0, mid + 5.0)
        if boxes and (right - left) < 20.0:
            mid = (left + right) / 2
            left, right = max(0.0, mid - 12.0), min(100.0, mid + 12.0)

        x0, y0 = int(pw * left / 100), int(ph * top / 100)
        x1, y1 = int(pw * right / 100 + 0.999), int(ph * bottom / 100 + 0.999)
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        crop = im.crop((x0, y0, x1, y1)).convert("RGBA")

        # Highlighter wash + border so the matched span/region is obvious.
        if boxes:
            overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for b in boxes:
                bl = float(b.get("left", 0))
                bt = float(b.get("top", 0))
                bw = float(b.get("width", 0))
                bh = float(b.get("height", 0))
                rx0 = int(pw * bl / 100) - x0
                ry0 = int(ph * bt / 100) - y0
                rx1 = int(pw * (bl + bw) / 100) - x0
                ry1 = int(ph * (bt + bh) / 100) - y0
                draw.rectangle([rx0, ry0, rx1, ry1], fill=(255, 210, 63, 100))
                draw.rectangle([rx0, ry0, rx1, ry1], outline=(180, 83, 31, 220), width=3)
            crop = Image.alpha_composite(crop, overlay)

        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
        return Response(
            buf.getvalue(),
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
    """Proxy a single region image, so the UI can show exactly what the agent saw."""
    try:
        r = requests.get(
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
            yield f"data: {json.dumps({'kind': kind, 'payload': payload}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    if not rag.INDEX_DIR.exists():
        sys.exit("No ./index — build it first (see README).")
    # The query encoder lives here rather than in the faiss process (see rag.py).
    # Load it now, in the background, so the first question doesn't pay for it.
    if rag.LOCAL_ENCODE:
        threading.Thread(target=rag.warm_encoder, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
