"""LLM access. Every pipeline call goes through `complete(task, prompt, ...)`.

`task` is a short label ("plan", "figure_code", ...) so that the replay/demo
LLM and the tests can route canned answers without a network call.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Protocol, Sequence

MODELS = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}

RULES = """You are paperforge, a tool that drafts a scientific manuscript from a research code repository.

Rules that hold for every answer, because the authors will submit what you help write:
- Never invent a number, result, dataset size or experimental detail. Use only what the repository material shows. Where a needed value is missing, write [TODO: what is missing] instead.
- Never invent a citation. Cite only keys you were explicitly given as verified.
- The manuscript is a draft the authors will check; prefer a plain, accurate sentence over an impressive unsupported one.
- Answer in the exact output format the request asks for, with no commentary around it."""


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(self, task: str, prompt: str, context: str = "", images: Sequence[Path] = ()) -> str: ...


class ClaudeLLM:
    """Claude via the Anthropic SDK (streaming, adaptive thinking, cached context)."""

    def __init__(self, model: str = "sonnet", effort: str = "high", max_tokens: int = 32000):
        import anthropic

        self.model = MODELS.get(model, model)
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / configured profile
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}

    def complete(self, task: str, prompt: str, context: str = "", images: Sequence[Path] = ()) -> str:
        system = [{"type": "text", "text": RULES}]
        if context:
            # The repo summary is identical across the many section/figure calls: cache it.
            system.append({"type": "text", "text": context, "cache_control": {"type": "ephemeral"}})
        content: list[dict] = []
        for img in images:
            data = base64.standard_b64encode(Path(img).read_bytes()).decode("ascii")
            media = "image/jpeg" if str(img).lower().endswith((".jpg", ".jpeg")) else "image/png"
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        content.append({"type": "text", "text": prompt})

        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            # If a safety classifier declines, the API reruns the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()

        self.usage["calls"] += 1
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            self.usage[key] += getattr(msg.usage, key, 0) or 0
        if msg.stop_reason == "refusal":
            raise LLMError(f"model declined the '{task}' request")
        if msg.stop_reason == "max_tokens":
            raise LLMError(f"'{task}' answer was cut off at max_tokens={self.max_tokens}")
        return "".join(b.text for b in msg.content if b.type == "text")


def extract_block(text: str, lang: str = "") -> str:
    """Return the body of the first fenced code block (any language if not found), else the text."""
    for pattern in (rf"```{lang}[^\n]*\n(.*?)```", r"```[^\n]*\n(.*?)```"):
        m = re.search(pattern, text, re.S)
        if m:
            return m.group(1).strip("\n")
    return text.strip()


def parse_json(text: str):
    body = extract_block(text, "json")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        start = min((i for i in (body.find("{"), body.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        return json.JSONDecoder().raw_decode(body[start:])[0]


def ask_json(llm: LLM, task: str, prompt: str, context: str = "", images: Sequence[Path] = ()):
    """Ask for JSON; retry once if the answer does not parse."""
    answer = llm.complete(task, prompt, context, images)
    try:
        return parse_json(answer)
    except (json.JSONDecodeError, ValueError):
        answer = llm.complete(task, prompt + "\n\nYour previous answer was not valid JSON. Reply with JSON only.", context, images)
        try:
            return parse_json(answer)
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"'{task}' did not return valid JSON") from exc
