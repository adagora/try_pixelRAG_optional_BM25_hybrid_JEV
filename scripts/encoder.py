#!/usr/bin/env python
"""Query-encoder sidecar — the model that `pixelrag serve` cannot run fast here.

Why this exists
---------------
faiss and torch each bundle their own libomp, and two OpenMP runtimes in one
process is a hazard on every platform — only the symptom is local (a segfault
on Apple Silicon, an "already initialized" abort elsewhere). `pixelrag serve`
needs faiss, so its query encoder is pinned to a single CPU thread by
OMP_NUM_THREADS=1, and profiling puts nearly all of search latency in the
transformer forward on that one thread (server-side encode p95 1673ms).

Nothing requires the encoder to share that process: /search accepts a
precomputed `embedding`. Running it here instead, with no faiss imported, frees
the same work to use every core — or an accelerator, which is where the win
actually is: p95 ~450ms on CPU threads against ~24ms on CUDA in fp16 with the
forward graphed, and identical rankings either way.

Keeping it as a service rather than loading in-process means the ~30s model load
and the ~17s of CUDA graph capture are paid once at startup, not per CLI
invocation, and there is exactly one copy of the weights no matter how many
clients there are. Requests are served from one pinned thread — see
encoder_device.on_model_thread, which is not optional once graphs are involved.

    python scripts/encoder.py          # :8001

Start it with an interpreter whose torch can see the accelerator. The .venv here
is deliberately CPU-only torch — it exists to host faiss — so a sidecar launched
from it resolves to cpu and gives that order of magnitude straight back.
PIXELRAG_ENCODER_DEVICE / _DTYPE / _PAD / _COMPILE override what runs and how;
see encoder_device.py.

Encoding must stay faithful to pixelrag_serve._encode_queries — same chat
template, same instruction, same last-token pooling — or queries land in a
different space from the indexed documents. Faithful, not bit-identical: fp16,
the padding and the graph each move the last bits, together 7.9e-04 on an index
score at cos 0.999994, every ranking unchanged. scripts/check_parity.py is the
gate; scripts/parity_check.py sweeps the knobs behind it.
"""

import os
import sys
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import encoder_device

MODEL = os.environ.get("PIXELRAG_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."

app = FastAPI(title="pixelrag query encoder")
_state: dict = {}


class EmbedRequest(BaseModel):
    text: str
    instruction: str | None = None


def load() -> None:
    # On encoder_device's model thread, not this one: whichever thread records the
    # CUDA graphs is the only thread that may replay them, and uvicorn will serve
    # /embed from a threadpool. See encoder_device.on_model_thread.
    encoder_device.on_model_thread(_load)


def _load() -> None:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dev, dtype = encoder_device.resolve(torch)
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=dtype).eval().to(dev)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    encoder_device.use_fast_tokenise(processor, MODEL)
    graphed = encoder_device.graph_forward(torch, model, dev)
    _state.update(processor=processor, model=model, device=dev, dtype=dtype,
                  torch=torch, graphed=graphed)
    load_s = time.perf_counter() - t0

    # The first encode on a fresh context is not representative: CUDA compiles
    # and autotunes its kernels and allocates cuBLAS workspaces on first use, and
    # a graphed forward additionally captures ~6s per padding width. Spend all of
    # it here, before uvicorn accepts, rather than on whoever asks first.
    t0 = time.perf_counter()
    widths = encoder_device.warm_widths()
    _encode("warmup")
    for width in widths:
        _encode("warmup", pad_width=width)
    print(f"encoder ready: {MODEL} on {dev} ({dtype}) in {load_s:.0f}s, "
          f"warmed {1 + len(widths)} shape(s) in "
          f"{(time.perf_counter()-t0)*1000:.0f}ms", flush=True)


@app.get("/health")
def health():
    # Reports dtype and whether the forward is graphed: "what is this actually
    # running" is the first question of every latency comparison, and every part
    # of the answer is env-dependent.
    return {"status": "ok" if _state else "loading", "model": MODEL,
            "device": _state.get("device"), "dtype": str(_state.get("dtype")),
            "graphed": _state.get("graphed"),
            "pad": encoder_device.PAD_TO_MULTIPLE}


@app.post("/embed")
def embed(req: EmbedRequest):
    return {"embedding": encoder_device.on_model_thread(_encode, req.text,
                                                        req.instruction)}


def _encode(text: str, instruction: str | None = None,
            pad_width: int = 0) -> list[float]:
    """The encode itself. Runs on the model thread; load() calls it directly.

    `pad_width` exists so warm-up can record a graph for a length no probe query
    happens to have.
    """
    torch, dev = _state["torch"], _state["device"]
    processor, model = _state["processor"], _state["model"]

    messages = [
        {"role": "system", "content": [{"type": "text",
         "text": instruction or DEFAULT_INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs, forward = encoder_device.prepare_forward(torch, model, processor,
                                                     inputs, pad_width)
    with torch.no_grad():
        out = forward(**inputs)
    h = out.last_hidden_state
    idx = inputs["attention_mask"].sum(dim=1) - 1
    pooled = h[torch.arange(h.size(0), device=h.device), idx]
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().float().numpy()[0].tolist()


if __name__ == "__main__":
    # The hazard is two OpenMP runtimes in one process, which is not specific to
    # any OS even though the Apple Silicon segfault is how this project met it.
    # Nothing above imports faiss; the guard is here to catch the day something
    # does, on whatever platform, rather than to describe one machine.
    if "faiss" in sys.modules:
        sys.exit("faiss must not be imported here — that is the crash this avoids.")
    load()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PIXELRAG_ENCODER_PORT", 8001)),
                log_level="warning")
