"""Async HTTP service around the agent.

POST /v1/agent/run    run one request (optionally within a session)
GET  /v1/traces/{id}  spans for a recent run
GET  /metrics         request, tool, guardrail, latency and cost counters
GET  /health          liveness
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import build_agent
from .agent import Agent
from .llm import LLM


class RunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


class Metrics:
    def __init__(self, window: int = 500) -> None:
        self.counters: Counter[str] = Counter()
        self.latencies: deque[float] = deque(maxlen=window)
        self.cost_usd = 0.0

    def record(self, result: Any) -> None:
        self.counters["requests_total"] += 1
        self.counters[f"intent.{result.intent}"] += 1
        self.counters[f"stop_reason.{result.stop_reason}"] += 1
        for call in result.tool_calls:
            self.counters[f"tool.{call.name}.{'ok' if call.ok else 'error'}"] += 1
            if call.flagged_injection:
                self.counters["guardrail.tool_output_flagged"] += 1
        if result.blocked_reason:
            self.counters["guardrail.input_blocked"] += 1
        self.latencies.append(result.latency_ms)
        self.cost_usd += result.cost_usd or 0.0

    def snapshot(self) -> dict[str, Any]:
        lat = sorted(self.latencies)

        def pct(p: float) -> float | None:
            return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None

        return {"counters": dict(self.counters), "latency_ms": {"p50": pct(0.5), "p95": pct(0.95)},
                "estimated_cost_usd": round(self.cost_usd, 6)}


def create_app(llm: LLM | None = None, agent: Agent | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=5.0) as http:
            state["agent"] = agent or build_agent(llm=llm, http=http)
            yield

    app = FastAPI(title="Agent Runtime Lab", lifespan=lifespan)
    metrics = Metrics()
    traces: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/agent/run")
    async def run(req: RunRequest) -> dict[str, Any]:
        result = await state["agent"].run(req.query, req.session_id)
        metrics.record(result)
        traces[result.trace_id] = result.trace
        while len(traces) > 200:
            traces.popitem(last=False)
        body = result.to_dict()
        body.pop("trace")
        return body

    @app.get("/v1/traces/{trace_id}")
    async def trace(trace_id: str) -> dict[str, Any]:
        if trace_id not in traces:
            raise HTTPException(404, "trace not found")
        return {"trace_id": trace_id, "spans": traces[trace_id]}

    @app.get("/metrics")
    async def get_metrics() -> dict[str, Any]:
        return metrics.snapshot()

    return app
