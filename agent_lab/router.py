"""Intent routing and query rewriting.

RuleRouter is a deterministic keyword baseline. LLMRouter asks Claude for a
structured {intent, rewritten_query, confidence} and falls back to the
baseline when the call fails or returns low confidence. The eval harness
compares the two on the same labelled cases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from .config import Settings
from .llm import LLM

Intent = Literal["engineering_docs", "ci_triage", "market_data", "out_of_scope"]
INTENTS: tuple[Intent, ...] = ("engineering_docs", "ci_triage", "market_data", "out_of_scope")

# Which tools each intent may use. Enforced at execution time, not just in the prompt.
TOOL_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "engineering_docs": ("search_docs",),
    "ci_triage": ("list_ci_runs", "get_ci_failure", "search_docs"),
    "market_data": ("get_market_price",),
    "out_of_scope": (),
}


@dataclass(frozen=True)
class Route:
    intent: Intent
    rewritten_query: str
    confidence: float
    router: str


_KEYWORDS: dict[str, list[str]] = {
    "ci_triage": [
        r"\bci\b", r"pipeline", r"build (failed|failing|broke)", r"\brun[- ]?\d+", r"failing test",
        r"flak", r"\bjob\b.*fail", r"why did .* fail", r"triage", r"test(s)? (failed|failing)",
    ],
    "market_data": [
        r"\b(btc|eth|bnb|sol|xrp|doge)\b", r"usdt", r"price", r"24h", r"\bmarket\b", r"bitcoin",
        r"ethereum", r"trading pair",
    ],
    "engineering_docs": [
        r"how (do|should|can) (i|we)", r"\bpolicy\b", r"deploy", r"code review", r"on[- ]call",
        r"runbook", r"rate limit", r"feature flag", r"secret", r"incident", r"\bsev[ -]?\d\b",
        r"handbook", r"guideline", r"rollback", r"what is our",
    ],
}


class RuleRouter:
    def route(self, query: str) -> Route:
        text = query.lower()
        scores = {
            intent: sum(1 for p in patterns if re.search(p, text))
            for intent, patterns in _KEYWORDS.items()
        }
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return Route("out_of_scope", query, 0.3, "rules")
        total = sum(scores.values())
        return Route(best, query, round(scores[best] / total, 2), "rules")  # type: ignore[arg-type]


_ROUTER_SYSTEM = """You route requests for an engineering assistant.

Intents:
- engineering_docs: questions about how the engineering team works (CI policy, deploys, \
rollbacks, code review, on-call, incidents, API rate limits, feature flags, secrets).
- ci_triage: questions about a specific failing CI run, pipeline or test.
- market_data: questions about current crypto prices or 24h market moves for a trading pair.
- out_of_scope: anything else, including trading advice, executing trades, or general chit-chat.

Also rewrite the request as a short, keyword-rich search query that keeps every \
identifier (run IDs, symbols, service names). Use uppercase pairs such as BTCUSDT for market data."""

_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "rewritten_query": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "rewritten_query", "confidence"],
    "additionalProperties": False,
}


class LLMRouter:
    def __init__(self, llm: LLM, settings: Settings, fallback: RuleRouter | None = None,
                 min_confidence: float = 0.5) -> None:
        self.llm = llm
        self.settings = settings
        self.fallback = fallback or RuleRouter()
        self.min_confidence = min_confidence
        self.last_usage: tuple[int, int] = (0, 0)

    async def route(self, query: str) -> Route:
        self.last_usage = (0, 0)
        try:
            msg = await self.llm.create(
                model=self.settings.router_model,
                max_tokens=1024,
                system=_ROUTER_SYSTEM,
                output_config={"effort": self.settings.router_effort,
                               "format": {"type": "json_schema", "schema": _ROUTER_SCHEMA}},
                messages=[{"role": "user", "content": query}],
            )
            self.last_usage = (msg.usage.input_tokens, msg.usage.output_tokens)
            if msg.stop_reason == "refusal":
                return Route("out_of_scope", query, 1.0, "llm_refusal")
            text = next(b.text for b in msg.content if b.type == "text")
            data = json.loads(text)
            route = Route(data["intent"], data["rewritten_query"] or query, float(data["confidence"]), "llm")
        except Exception:  # network, parse or schema failure: degrade to the baseline
            return self.fallback.route(query)
        if route.intent not in INTENTS or route.confidence < self.min_confidence:
            return self.fallback.route(query)
        return route
