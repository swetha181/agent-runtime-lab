"""Tool server: typed, validated, read-only tools shared by the agent and MCP.

Every tool has a strict JSON schema (sent to Claude with `strict: true`) and a
Pydantic model that re-validates the input before the handler runs, so a
malformed or truncated call fails fast with a clear `is_error` result instead
of reaching the handler.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import DATA_DIR, Settings
from .retrieval import TfidfIndex

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"


class ToolError(Exception):
    """Raised for expected tool failures; the message is shown to the model."""


@dataclass
class ToolContext:
    settings: Settings
    index: TfidfIndex
    ci_dir: Path = DATA_DIR / "ci_runs"
    market_fixture: Path = DATA_DIR / "market_fixtures.json"
    http: httpx.AsyncClient | None = None


# ---------- inputs ----------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchDocsInput(_Strict):
    query: str = Field(min_length=2, max_length=300)
    k: Literal[1, 2, 3, 4, 5]


class CiFailureInput(_Strict):
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,40}$")


class ListCiRunsInput(_Strict):
    limit: Literal[1, 3, 5, 10]


class MarketPriceInput(_Strict):
    symbol: str = Field(pattern=r"^[A-Z0-9]{5,12}$")


# ---------- handlers ----------


async def search_docs(ctx: ToolContext, args: SearchDocsInput) -> dict[str, Any]:
    hits = ctx.index.search(args.query, k=args.k)
    return {
        "query": args.query,
        "results": [
            {"source": h.chunk.source, "heading": h.chunk.heading, "score": h.score, "text": h.chunk.text}
            for h in hits
        ],
    }


_ERROR_RULES: list[tuple[str, str]] = [
    ("dependency", r"(ResolutionImpossible|No matching distribution|Could not find a version|npm ERR! 404)"),
    ("timeout", r"(Timed? ?out|TimeoutError|exceeded the maximum execution time|deadline exceeded)"),
    ("out_of_memory", r"(Killed|MemoryError|OOMKilled|exit code 137)"),
    ("network_flake", r"(ConnectionResetError|ECONNRESET|Temporary failure in name resolution|503 Service Unavailable)"),
    ("lint", r"(ruff|flake8|eslint).{0,40}(error|found \d+ error)"),
    ("assertion", r"(AssertionError|assert .+ ==|Expected .+ but got)"),
]
_FAILED_TEST = re.compile(r"FAILED ([\w/.\-]+::[\w\[\]\-]+)")


async def get_ci_failure(ctx: ToolContext, args: CiFailureInput) -> dict[str, Any]:
    path = ctx.ci_dir / f"{args.run_id}.log"
    if not path.is_file():
        raise ToolError(f"CI run '{args.run_id}' not found. Use list_ci_runs to see valid IDs.")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = {k.strip(): v.strip() for k, v in (l[2:].split(":", 1) for l in lines if l.startswith("# ") and ":" in l)}
    text = "\n".join(lines)
    category = next((name for name, rx in _ERROR_RULES if re.search(rx, text)), "unknown")
    failed_tests = sorted(set(_FAILED_TEST.findall(text)))
    error_lines = [l for l in lines if re.search(r"(Error|FAILED|error:|Killed|Timeout|ERR!)", l)][:12]
    retried_ok = "retry: passed" in text.lower()
    return {
        "run_id": args.run_id,
        "pipeline": header.get("pipeline"),
        "step": header.get("step"),
        "commit": header.get("commit"),
        "category": category,
        "likely_flaky": category in {"network_flake", "timeout"} or retried_ok,
        "failed_tests": failed_tests,
        "key_lines": error_lines,
    }


async def list_ci_runs(ctx: ToolContext, args: ListCiRunsInput) -> dict[str, Any]:
    runs = sorted(ctx.ci_dir.glob("*.log"), key=lambda p: p.name)[: args.limit]
    out = []
    for path in runs:
        first = path.read_text(encoding="utf-8").splitlines()[:4]
        meta = {k.strip(): v.strip() for k, v in (l[2:].split(":", 1) for l in first if l.startswith("# ") and ":" in l)}
        out.append({"run_id": path.stem, "pipeline": meta.get("pipeline"), "status": meta.get("status")})
    return {"runs": out}


async def get_market_price(ctx: ToolContext, args: MarketPriceInput) -> dict[str, Any]:
    if ctx.settings.offline or ctx.http is None:
        fixtures = json.loads(ctx.market_fixture.read_text(encoding="utf-8"))
        if args.symbol not in fixtures:
            raise ToolError(f"Unknown symbol '{args.symbol}' (offline fixtures only).")
        return {**fixtures[args.symbol], "symbol": args.symbol, "source": "offline_fixture"}
    resp = await ctx.http.get(BINANCE_TICKER_URL, params={"symbol": args.symbol})
    if resp.status_code == 400:
        raise ToolError(f"Unknown symbol '{args.symbol}'. Use a trading pair such as BTCUSDT.")
    resp.raise_for_status()
    data = resp.json()
    return {
        "symbol": args.symbol,
        "last_price": data["lastPrice"],
        "change_24h_pct": data["priceChangePercent"],
        "high_24h": data["highPrice"],
        "low_24h": data["lowPrice"],
        "source": "binance_public_api",
    }


# ---------- registry ----------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[_Strict]
    handler: Callable[[ToolContext, Any], Awaitable[dict[str, Any]]]
    input_schema: dict[str, Any] = field(default_factory=dict)

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": self.input_schema,
        }

    def validate(self, raw: Any) -> _Strict:
        try:
            return self.input_model.model_validate(raw)
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise ToolError(f"Invalid input for {self.name}: {errors}") from exc


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "search_docs",
            "Search the internal engineering handbook (CI, deploys, on-call, code review, "
            "API limits, security). Returns the most relevant sections with their source file. "
            "Use it for any question about how the team works; cite the source in the answer.",
            SearchDocsInput,
            search_docs,
            _schema({
                "query": {"type": "string", "description": "Short, keyword-rich search query."},
                "k": {"type": "integer", "enum": [1, 2, 3, 4, 5], "description": "Number of sections to return."},
            }),
        ),
        Tool(
            "get_ci_failure",
            "Fetch and triage the log of one CI run: failure category (assertion, timeout, "
            "dependency, network_flake, out_of_memory, lint), failed tests and key error lines.",
            CiFailureInput,
            get_ci_failure,
            _schema({"run_id": {"type": "string", "description": "CI run ID, e.g. 'run-1042'."}}),
        ),
        Tool(
            "list_ci_runs",
            "List recent CI runs with their pipeline and status. Use it when the user does not give a run ID.",
            ListCiRunsInput,
            list_ci_runs,
            _schema({"limit": {"type": "integer", "enum": [1, 3, 5, 10]}}),
        ),
        Tool(
            "get_market_price",
            "Get the latest price and 24h change for a Binance spot trading pair (read-only market data). "
            "Use uppercase pair symbols such as BTCUSDT or ETHUSDT.",
            MarketPriceInput,
            get_market_price,
            _schema({"symbol": {"type": "string", "description": "Trading pair, e.g. BTCUSDT."}}),
        ),
    ]
}
