"""The model backends: what each one is called, how it is billed, how big an
image it wants, and how to stream an answer out of it.

Everything that differs between Gemini and Anthropic used to be spread through
rag.py as `if provider == "..."` and four hand-copied functions. Two one-shot
runners repeated the same skeleton — build the preamble, attach header + image
per page, stream, accumulate usage — and two agent runners repeated a second
one. Provider identity also travelled *downwards* as a bare string into
imagefit, so a low-level sizing utility branched on the names of its callers.

A Reader owns all of it:

  * `name` / `model` — identity, for the cache namespace and the UI footer.
  * `image_policy` — what resolution this provider bills for, handed to
    imagefit as data rather than as its own name.
  * `read_pages` — one call over attached page screenshots, streamed.
  * `browse` — the multi-turn tool loop, for agent mode.
  * `price` — tokens to dollars, or None where we decline to guess.

Adding a provider is writing one class. Testing the pipeline without paying for
one is writing a smaller class — see tests/test_providers.py, which does exactly
that and never imports an SDK.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Protocol

import imagefit

GEMINI = "gemini"
ANTHROPIC = "anthropic"


# --------------------------------------------------------------------------
# what a read costs and what it produced
# --------------------------------------------------------------------------

@dataclass
class Usage:
    """Tokens for one answer.

    `input` is the UNCACHED remainder on providers that report a prompt cache
    separately — cached reads and writes are billed at their own rates, so
    summing them at the full rate would overstate the bill and hide the saving
    the cache exists to produce.
    """

    input: int = 0
    output: int = 0
    thoughts: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def as_dict(self, cost_usd: float | None) -> dict:
        return {"input": self.input, "output": self.output,
                "thoughts": self.thoughts, "cache_read": self.cache_read,
                "cache_write": self.cache_write, "cost_usd": cost_usd}


@dataclass
class Reply:
    """A reader's output: the raw text, what it cost, and how many round trips."""

    text: str
    usage: Usage = field(default_factory=Usage)
    steps: int = 1


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------

class Reader(Protocol):
    """A model that can read page screenshots and answer questions about them."""

    name: str
    model: str

    @property
    def image_policy(self) -> imagefit.Policy:
        """How pages should be sized before they are sent to this reader."""

    def read_pages(self, system: str, preamble: str, pages: list[dict],
                   header_of, on_text) -> Reply:
        """One call over attached page screenshots, streaming text to `on_text`.

        `pages` are the dicts rag._oneshot_pages produces: `image` bytes, `mime`,
        and the metadata `header_of(page)` turns into the label above each image.
        """

    def browse(self, system: str, question: str, tools: list[dict], dispatch,
               on_event, max_steps: int) -> Reply:
        """Multi-turn tool loop. `dispatch(name, args)` returns (result, event).

        A result is `{"ok": True, "label", "image", "mime"}`, `{"ok": False,
        "message"}`, or a plain string; turning that into this SDK's tool-result
        shape is the whole reason each backend implements this separately.
        """

    def price(self, usage: Usage) -> float | None:
        """Dollars, or None where the rate depends on things we do not know."""


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

# 3.6 Flash: fewer agent turns/tool calls than 3.5 Flash, stronger multimodal
# (price matrices), cheaper output. Override: GEMINI_MODEL=gemini-3.5-flash-lite
# for max throughput (set GEMINI_THINKING=medium if tool loops truncate early).
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
# Unset → API default (medium on 3.6 Flash / minimal on 3.5 Flash-Lite).
# We pin low: browse loops re-send every tile image each turn, so thinking
# cost compounds. Override: GEMINI_THINKING=minimal|low|medium|high.
GEMINI_THINKING = os.environ.get("GEMINI_THINKING", "low").strip().lower()


class GeminiReader:
    name = GEMINI

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key
        self.model = model or GEMINI_MODEL

    # -- client ------------------------------------------------------------

    def _client(self):
        """Cached — a transient Client is GC'd mid-call, closing its httpx
        session ("Cannot send a request, as the client has been closed")."""
        key = self._api_key or os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        return _gemini_client_for(key)

    @property
    def image_policy(self) -> imagefit.Policy:
        return imagefit.GEMINI_POLICY

    def price(self, usage: Usage) -> float | None:
        """None on purpose: Gemini pricing varies by model and by tier, and a
        guessed rate in the UI footer is worse than no number at all."""
        return None

    def models(self) -> list[str]:
        out = []
        for m in self._client().models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                out.append(m.name.removeprefix("models/"))
        return sorted(out)

    # -- one-shot ----------------------------------------------------------

    def read_pages(self, system, preamble, pages, header_of, on_text) -> Reply:
        from google.genai import types

        parts: list = [types.Part.from_text(text=preamble)]
        for p in pages:
            parts.append(types.Part.from_text(text=header_of(p)))
            parts.append(types.Part.from_bytes(data=p["image"],
                                               mime_type=p["mime"]))

        chunks: list[str] = []
        usage = Usage()
        for chunk in self._client().models.generate_content_stream(
                model=self.model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    thinking_config=_thinking_config(types))):
            piece = text_of(chunk)
            if piece:
                chunks.append(piece)
                on_text(piece)
            _accumulate_gemini(usage, chunk, cumulative=True)

        return Reply("".join(chunks) or "The model returned nothing.", usage)

    # -- agent -------------------------------------------------------------

    def browse(self, system, question, tools, dispatch, on_event,
               max_steps) -> Reply:
        from google.genai import types

        decls = [
            types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["input_schema"])
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=decls)],
            # We drive the loop ourselves so the UI can stream each step.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True),
            thinking_config=_thinking_config(types),
        )
        contents = [types.Content(role="user",
                                  parts=[types.Part.from_text(text=question)])]
        usage = Usage()
        client = self._client()

        for step in range(max_steps):
            resp = client.models.generate_content(
                model=self.model, contents=contents, config=config)
            _accumulate_gemini(usage, resp, cumulative=False)

            cand = (resp.candidates or [None])[0]
            if cand is None or not cand.content or not cand.content.parts:
                return Reply(text_of(resp) or "The model returned nothing.",
                             usage, step + 1)

            contents.append(cand.content)
            calls = [p.function_call for p in cand.content.parts if p.function_call]
            if not calls:
                return Reply(text_of(resp), usage, step + 1)

            replies = []
            for call in calls:
                result, ev = _safe_dispatch(dispatch, call.name,
                                            dict(call.args or {}))
                if ev:
                    on_event(ev)
                replies.append(self._tool_reply(types, call.name, result))
            contents.append(types.Content(role="user", parts=replies))

        return Reply("Stopped after the step limit without settling on an answer.",
                     usage, max_steps)

    @staticmethod
    def _tool_reply(types, name: str, result):
        if isinstance(result, dict) and result.get("ok"):
            # Gemini tool results carry images natively via inline_data.
            return types.Part.from_function_response(
                name=name,
                response={"status": "ok", "description": result["label"]},
                parts=[types.FunctionResponsePart(
                    inline_data=types.FunctionResponseBlob(
                        mime_type=result["mime"], data=result["image"]))])
        if isinstance(result, dict):
            return types.Part.from_function_response(
                name=name,
                response={"status": "error",
                          "message": result.get("message", "failed")})
        return types.Part.from_function_response(
            name=name, response={"status": "ok", "results": result})


def _thinking_config(types):
    """Map GEMINI_THINKING → ThinkingConfig. Unknown values fall back to low."""
    level = {
        "minimal": types.ThinkingLevel.MINIMAL,
        "low": types.ThinkingLevel.LOW,
        "medium": types.ThinkingLevel.MEDIUM,
        "high": types.ThinkingLevel.HIGH,
    }.get(GEMINI_THINKING, types.ThinkingLevel.LOW)
    return types.ThinkingConfig(thinking_level=level)


def _accumulate_gemini(usage: Usage, resp, *, cumulative: bool) -> None:
    """Fold one response's usage in.

    Streaming reports CUMULATIVE totals, so the last chunk carrying them is
    authoritative and assignment is correct. The agent loop makes discrete
    calls, so those add. Getting this backwards inflates or flattens the whole
    bill, which is why it is one function rather than two inline blocks.
    """
    um = getattr(resp, "usage_metadata", None)
    if not um:
        return
    thoughts = um.thoughts_token_count or 0
    output = (um.candidates_token_count or 0) + thoughts
    if cumulative:
        usage.input = um.prompt_token_count or usage.input
        usage.thoughts = thoughts or usage.thoughts
        usage.output = output or usage.output
    else:
        usage.input += um.prompt_token_count or 0
        usage.thoughts += thoughts
        usage.output += output


def text_of(resp) -> str:
    cand = (resp.candidates or [None])[0]
    if not cand or not cand.content or not cand.content.parts:
        return ""
    return "".join(p.text for p in cand.content.parts if getattr(p, "text", None))


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
# Thinking depth for the one-shot read. Empty = the API default (high).
#
# Worth setting, because thinking is ON BY DEFAULT on Opus 5 and thinking
# tokens bill as OUTPUT ($25/M) — a "read this cell and quote it" task was
# silently paying reasoning rates. `medium` is the starting point, not a
# conclusion: sweep low/medium/high with evaluate_pl.py before settling, since
# the whole ONESHOT_SYSTEM prompt exists because misreading a row is expensive.
ANTHROPIC_EFFORT = os.environ.get("ANTHROPIC_EFFORT", "medium").strip().lower()

# List price for the default model, $/1M tokens (input, output).
ANTHROPIC_RATES = (5.0, 25.0)

# Thinking shares this budget with the visible answer on models where thinking
# is on by default, so this is not just answer length.
ANTHROPIC_MAX_TOKENS = 8000


class AnthropicReader:
    name = ANTHROPIC

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key
        self.model = model or ANTHROPIC_MODEL

    def _client(self):
        import anthropic

        return (anthropic.Anthropic(api_key=self._api_key) if self._api_key
                else anthropic.Anthropic())

    @property
    def image_policy(self) -> imagefit.Policy:
        return imagefit.ANTHROPIC_POLICY

    def price(self, usage: Usage) -> float | None:
        rin, rout = ANTHROPIC_RATES
        return round(
            usage.input / 1e6 * rin
            + usage.cache_read / 1e6 * rin * 0.1
            + usage.cache_write / 1e6 * rin * 1.25
            + usage.output / 1e6 * rout, 4)

    def models(self) -> list[str]:
        return sorted(m.id for m in self._client().models.list())

    # -- one-shot ----------------------------------------------------------

    def read_pages(self, system, preamble, pages, header_of, on_text) -> Reply:
        content: list = [{"type": "text", "text": preamble}]
        for p in pages:
            content.append({"type": "text", "text": header_of(p)})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": p["mime"],
                    "data": base64.standard_b64encode(p["image"]).decode(),
                },
            })

        kwargs: dict = {
            "model": self.model,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            # The system prompt is ~1.2k tokens of Polish, byte-identical on
            # every call, and was being re-billed at full rate every time.
            # Caching is a prefix match, so it has to be a block with
            # cache_control rather than a bare string. Reads cost 0.1x; the
            # images that follow still cost full price because the retrieved
            # set changes per question.
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
        }
        if ANTHROPIC_EFFORT:
            # Reading a value out of a table is not a reasoning-heavy task, and
            # the default effort spends thinking tokens (billed as output)
            # accordingly. Sweep against eval/ before trusting a low setting on
            # a price matrix — cheaper is not free if it misreads a row.
            kwargs["output_config"] = {"effort": ANTHROPIC_EFFORT}

        chunks: list[str] = []
        with self._client().messages.stream(**kwargs) as stream:
            for piece in stream.text_stream:
                chunks.append(piece)
                on_text(piece)
            final = stream.get_final_message()
        return Reply("".join(chunks), _anthropic_usage(final.usage))

    # -- agent -------------------------------------------------------------

    def browse(self, system, question, tools, dispatch, on_event,
               max_steps) -> Reply:
        client = self._client()
        messages: list = [{"role": "user", "content": question}]
        usage = Usage()

        for step in range(max_steps):
            resp = client.messages.create(
                model=self.model, max_tokens=ANTHROPIC_MAX_TOKENS,
                system=system, tools=tools, messages=messages)
            got = _anthropic_usage(resp.usage)
            usage.input += got.input
            usage.output += got.output
            usage.cache_read += got.cache_read
            usage.cache_write += got.cache_write

            # A refusal returns HTTP 200 with empty/partial content — check first.
            if resp.stop_reason == "refusal":
                return Reply(
                    f"The model declined this request ({resp.stop_details}).",
                    usage, step + 1)

            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return Reply(text, usage, step + 1)

            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                result, ev = _safe_dispatch(dispatch, block.name, dict(block.input))
                if ev:
                    on_event(ev)
                results.append(self._tool_result(block.id, result))
            messages.append({"role": "user", "content": results})

        return Reply("Stopped after the step limit without settling on an answer.",
                     usage, max_steps)

    @staticmethod
    def _tool_result(tool_use_id: str, result) -> dict:
        if isinstance(result, dict) and result.get("ok"):
            content = [
                {"type": "text", "text": result["label"] + ":"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": result["mime"],
                    "data": base64.standard_b64encode(result["image"]).decode()}},
            ]
        elif isinstance(result, dict):
            content = result.get("message", "failed")
            if result.get("error"):
                return {"type": "tool_result", "tool_use_id": tool_use_id,
                        "content": content, "is_error": True}
        else:
            content = result
        return {"type": "tool_result", "tool_use_id": tool_use_id,
                "content": content}


def _anthropic_usage(u) -> Usage:
    return Usage(
        input=u.input_tokens,
        output=u.output_tokens,
        cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------

def _safe_dispatch(dispatch, name: str, args: dict):
    """Run a tool call; turn a raised exception into a result the model can read.

    A traceback is more useful to the model than a dead turn — it can correct a
    bad argument and retry — and it must not abort the browse loop.
    """
    try:
        return dispatch(name, args)
    except Exception as e:
        return {"ok": False, "error": True,
                "message": f"{type(e).__name__}: {e}"}, None


try:
    from functools import lru_cache

    @lru_cache(maxsize=4)
    def _gemini_client_for(key: str):
        from google import genai

        return genai.Client(api_key=key)
except ImportError:  # pragma: no cover
    pass


def detect(api_key: str | None = None) -> str:
    """Pick gemini or anthropic.

    A key pasted in the UI wins over shell env — otherwise an exported
    ANTHROPIC_API_KEY steals Gemini pastes and surfaces as a cryptic
    APIConnectionError against api.anthropic.com.
    """
    forced = os.environ.get("PIXELRAG_PROVIDER", "").strip().lower()
    if forced:
        return forced
    key = (api_key or "").strip()
    if key.startswith("AIza") or key.startswith("ya29."):
        return GEMINI
    if key.startswith("sk-ant"):
        return ANTHROPIC
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return GEMINI
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return ANTHROPIC
    return GEMINI


_REGISTRY = {GEMINI: GeminiReader, ANTHROPIC: AnthropicReader}


def reader_for(provider: str | None = None,
               api_key: str | None = None) -> Reader:
    """The Reader for a provider name. The one place a name becomes a class."""
    provider = (provider or detect(api_key)).strip().lower()
    cls = _REGISTRY.get(provider)
    if cls is None:
        raise RuntimeError(
            f"Unknown provider {provider!r} (expected "
            f"{' or '.join(sorted(_REGISTRY))}).")
    return cls(api_key=api_key)


def register(name: str, cls) -> None:
    """Add a Reader implementation. Exists so a test can install a fake one
    without an SDK, an API key, or a network."""
    _REGISTRY[name] = cls
