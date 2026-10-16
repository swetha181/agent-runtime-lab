"""MCP server exposing the same validated tools over the Model Context Protocol.

Run:  python -m agent_lab.mcp_server        (stdio transport)

Any MCP client (Claude Code, Claude Desktop, the Anthropic SDK MCP helpers) can
then discover and call search_docs, list_ci_runs, get_ci_failure and
get_market_price. Inputs go through the same Pydantic validation as the agent.
"""

from __future__ import annotations

from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

from . import build_context
from .config import Settings
from .tools import TOOLS, ToolContext

server = MCPServer(
    name="agent-runtime-lab",
    instructions="Read-only engineering tools: handbook search, CI failure triage and market prices.",
)
_ctx: ToolContext | None = None


def _context() -> ToolContext:
    global _ctx
    if _ctx is None:
        settings = Settings()
        _ctx = build_context(settings, http=None if settings.offline else httpx.AsyncClient(timeout=5.0))
    return _ctx


async def _call(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    tool = TOOLS[name]
    return await tool.handler(_context(), tool.validate(raw))


@server.tool(description=TOOLS["search_docs"].description)
async def search_docs(query: str, k: int = 3) -> dict[str, Any]:
    return await _call("search_docs", {"query": query, "k": k})


@server.tool(description=TOOLS["list_ci_runs"].description)
async def list_ci_runs(limit: int = 5) -> dict[str, Any]:
    return await _call("list_ci_runs", {"limit": limit})


@server.tool(description=TOOLS["get_ci_failure"].description)
async def get_ci_failure(run_id: str) -> dict[str, Any]:
    return await _call("get_ci_failure", {"run_id": run_id})


@server.tool(description=TOOLS["get_market_price"].description)
async def get_market_price(symbol: str) -> dict[str, Any]:
    return await _call("get_market_price", {"symbol": symbol})


if __name__ == "__main__":
    server.run("stdio")
