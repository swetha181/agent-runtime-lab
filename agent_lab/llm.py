"""LLM clients behind one async interface.

- AnthropicLLM: the real Claude API (async), with server-side refusal fallback.
- ScriptedLLM: returns pre-built SDK messages in order; used by integration
  tests so the agent loop runs end to end with mocked LLM responses.
- RecordingLLM / ReplayLLM: record real responses to a cassette file, then
  replay them offline to detect runtime regressions without API cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLM(Protocol):
    async def create(self, **params: Any) -> BetaMessage: ...


class AnthropicLLM:
    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        # Resolves credentials from ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = client or AsyncAnthropic()

    async def create(self, **params: Any) -> BetaMessage:
        # fallbacks="default": if a safety classifier declines, the API retries the
        # request on Anthropic's recommended fallback model inside the same call.
        return await self.client.beta.messages.create(
            betas=[FALLBACK_BETA], fallbacks="default", **params
        )


class ScriptedLLM:
    """Plays back a fixed list of responses and records every request."""

    def __init__(self, responses: list[BetaMessage | dict[str, Any]]) -> None:
        self._responses = [r if isinstance(r, BetaMessage) else BetaMessage.model_validate(r) for r in responses]
        self.requests: list[dict[str, Any]] = []

    async def create(self, **params: Any) -> BetaMessage:
        # Snapshot the message list: the agent keeps appending to the same list object.
        self.requests.append({**params, "messages": list(params.get("messages", []))})
        if not self._responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)


class RecordingLLM:
    def __init__(self, inner: LLM, cassette: Path) -> None:
        self.inner = inner
        self.cassette = cassette
        self.recorded: list[dict[str, Any]] = []

    async def create(self, **params: Any) -> BetaMessage:
        message = await self.inner.create(**params)
        self.recorded.append(message.to_dict())
        return message

    def save(self) -> None:
        self.cassette.parent.mkdir(parents=True, exist_ok=True)
        self.cassette.write_text(json.dumps(self.recorded, indent=1), encoding="utf-8")


class ReplayLLM(ScriptedLLM):
    def __init__(self, cassette: Path) -> None:
        super().__init__(json.loads(cassette.read_text(encoding="utf-8")))


# ---------- helpers for building scripted responses ----------

_counter = 0


def _msg_id() -> str:
    global _counter
    _counter += 1
    return f"msg_scripted_{_counter:04d}"


def text_message(text: str, stop_reason: str = "end_turn", model: str = "claude-opus-5-5") -> BetaMessage:
    return BetaMessage.model_validate({
        "id": _msg_id(), "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40},
    })


def tool_message(calls: list[tuple[str, dict[str, Any]]], model: str = "claude-opus-5-5") -> BetaMessage:
    content = [
        {"type": "tool_use", "id": f"toolu_{_msg_id()}_{i}", "name": name, "input": args}
        for i, (name, args) in enumerate(calls)
    ]
    return BetaMessage.model_validate({
        "id": _msg_id(), "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 150, "output_tokens": 30},
    })
