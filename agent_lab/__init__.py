"""Agent Runtime Lab: a tool-calling Claude agent for engineering workflows."""

from __future__ import annotations

import httpx

from .agent import Agent, RunResult
from .config import DATA_DIR, Settings
from .llm import LLM, AnthropicLLM
from .retrieval import TfidfIndex
from .router import LLMRouter, RuleRouter
from .tools import ToolContext

__all__ = ["Agent", "RunResult", "Settings", "build_context", "build_agent"]


def build_context(settings: Settings | None = None, http: httpx.AsyncClient | None = None) -> ToolContext:
    settings = settings or Settings()
    return ToolContext(settings=settings, index=TfidfIndex.from_directory(DATA_DIR / "docs"), http=http)


def build_agent(llm: LLM | None = None, settings: Settings | None = None,
                http: httpx.AsyncClient | None = None, use_llm_router: bool = True) -> Agent:
    ctx = build_context(settings, http)
    llm = llm or AnthropicLLM()
    router = LLMRouter(llm, ctx.settings) if use_llm_router else RuleRouter()
    return Agent(llm, ctx, router=router)
