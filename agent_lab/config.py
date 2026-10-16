"""Runtime settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# USD per million tokens (input, output), first-party Claude API rates.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "claude-opus-5-5"))
    router_model: str = field(default_factory=lambda: os.getenv("AGENT_ROUTER_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: os.getenv("AGENT_EFFORT", "medium"))
    router_effort: str = field(default_factory=lambda: os.getenv("AGENT_ROUTER_EFFORT", "low"))
    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "16000"))
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "6"))
    tool_timeout_s: float = float(os.getenv("AGENT_TOOL_TIMEOUT_S", "8"))
    max_tool_result_chars: int = int(os.getenv("AGENT_MAX_TOOL_RESULT_CHARS", "6000"))
    memory_char_budget: int = int(os.getenv("AGENT_MEMORY_CHAR_BUDGET", "12000"))
    offline: bool = field(default_factory=lambda: os.getenv("AGENT_OFFLINE", "0") == "1")


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = PRICING.get(model)
    if rates is None:
        return None
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
