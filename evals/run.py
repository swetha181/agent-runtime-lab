"""Evaluation harness.

  python -m evals.run --mode offline   deterministic components, no API key, runs in CI
  python -m evals.run --mode live      full agent on Claude; records a cassette per case
  python -m evals.run --mode replay    re-runs the agent on recorded cassettes (regression check)

Offline measures: rule-router accuracy, retrieval recall@3 / MRR, CI triage
accuracy, guardrail block rate and false-positive rate, tool-output flagging.
Live adds: LLM-router accuracy, tool-selection accuracy, answer checks,
latency p50/p95, tokens and estimated cost. Replay compares the runtime's
intent and tool sequence against the recorded live run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from agent_lab import build_context
from agent_lab.agent import Agent
from agent_lab.config import Settings
from agent_lab.guardrails import check_input, wrap_tool_output
from agent_lab.llm import AnthropicLLM, RecordingLLM, ReplayLLM
from agent_lab.router import LLMRouter, RuleRouter
from agent_lab.tools import TOOLS

EVAL_DIR = Path(__file__).resolve().parent
CASSETTES = EVAL_DIR / "cassettes"
RESULTS = EVAL_DIR / "results"


def load_cases() -> list[dict[str, Any]]:
    lines = (EVAL_DIR / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def intent_ok(expected: str, actual: str) -> bool:
    return actual in expected.split("|")


def answer_checks(case: dict[str, Any], answer: str) -> tuple[bool, list[str]]:
    text = answer.lower()
    failures = [f"missing '{group}'" for group in case.get("must_contain", [])
                if not any(alt.lower() in text for alt in group.split("|"))]
    failures += [f"contains '{bad}'" for bad in case.get("must_not_contain", []) if bad.lower() in text]
    return not failures, failures


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(p * len(ordered)))], 1)


# ---------------- offline ----------------


async def run_offline(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ctx = build_context(Settings(offline=True))
    router = RuleRouter()
    report: dict[str, Any] = {"mode": "offline", "cases": len(cases)}

    # 1. Routing baseline (attack cases are handled by the input guard, not the router)
    routed = [c for c in cases if not c.get("attack")]
    per_cat: dict[str, list[bool]] = defaultdict(list)
    misses = []
    for c in routed:
        r = router.route(c["query"])
        ok = intent_ok(c["expected_intent"], r.intent)
        per_cat[c["category"]].append(ok)
        if not ok:
            misses.append({"id": c["id"], "expected": c["expected_intent"], "got": r.intent})
    report["rule_router"] = {
        "accuracy": round(sum(sum(v) for v in per_cat.values()) / len(routed), 3),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_cat.items())},
        "misses": misses,
    }

    # 2. Retrieval quality
    retrieval = [c for c in cases if c.get("expected_sources")]
    hits3, rr = 0, []
    retrieval_misses = []
    for c in retrieval:
        results = [h.chunk.source for h in ctx.index.search(c["query"], k=3)]
        rank = next((i + 1 for i, s in enumerate(results) if s in c["expected_sources"]), None)
        hits3 += rank is not None
        rr.append(1 / rank if rank else 0.0)
        if rank is None:
            retrieval_misses.append({"id": c["id"], "got": results})
    report["retrieval"] = {
        "queries": len(retrieval),
        "recall_at_3": round(hits3 / len(retrieval), 3),
        "mrr": round(statistics.mean(rr), 3),
        "misses": retrieval_misses,
    }

    # 3. CI triage classifier
    ci_cases = [c for c in cases if c.get("expected_ci_category")]
    ci_ok = 0
    for c in ci_cases:
        out = await TOOLS["get_ci_failure"].handler(ctx, TOOLS["get_ci_failure"].validate({"run_id": c["ci_run"]}))
        ci_ok += out["category"] == c["expected_ci_category"]
    report["ci_triage"] = {"cases": len(ci_cases), "accuracy": round(ci_ok / len(ci_cases), 3)}

    # 4. Input guardrail: block rate on attacks, false positives on benign requests
    attacks = [c for c in cases if c.get("attack")]
    benign = [c for c in cases if not c.get("attack")]
    blocked = sum(not check_input(c["query"]).allowed for c in attacks)
    false_pos = [c["id"] for c in benign if not check_input(c["query"]).allowed]
    report["input_guard"] = {
        "attacks": len(attacks), "blocked": blocked, "block_rate": round(blocked / len(attacks), 3),
        "benign": len(benign), "false_positives": false_pos,
        "false_positive_rate": round(len(false_pos) / len(benign), 3),
    }

    # 5. Tool-output guard: every handbook section scanned; only the poisoned one should flag
    flagged_sources = sorted({ch.source for ch in ctx.index.chunks if wrap_tool_output("search_docs", ch.text)[1]})
    report["tool_output_guard"] = {"chunks_scanned": len(ctx.index.chunks), "flagged_sources": flagged_sources}
    return report


# ---------------- live / replay ----------------


async def run_agent_cases(cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    settings = Settings()
    rows = []
    async with httpx.AsyncClient(timeout=5.0) as http:
        ctx = build_context(settings, http)
        for c in cases:
            cassette = CASSETTES / f"{c['id']}.json"
            if mode == "replay":
                if not cassette.exists():
                    rows.append({"id": c["id"], "skipped": "no cassette"})
                    continue
                llm = ReplayLLM(cassette)
                ctx.settings = Settings(offline=True)  # replay must not hit live market data
            else:
                llm = RecordingLLM(AnthropicLLM(), cassette)
            agent = Agent(llm, ctx, router=LLMRouter(llm, ctx.settings))
            started = time.perf_counter()
            try:
                result = await agent.run(c["query"])
            except Exception as exc:  # record and keep going; a crash is a failed case
                rows.append({"id": c["id"], "category": c["category"], "error": f"{type(exc).__name__}: {exc}"})
                continue
            if mode == "live":
                llm.save()
            tools_used = [t.name for t in result.tool_calls]
            answer_ok, answer_fail = answer_checks(c, result.answer)
            rows.append({
                "id": c["id"], "category": c["category"],
                "intent_ok": intent_ok(c["expected_intent"], result.intent), "intent": result.intent,
                "router": result.router,
                "tools_ok": set(c.get("expected_tools", [])) <= set(tools_used), "tools": tools_used,
                "tool_errors": sum(not t.ok for t in result.tool_calls),
                "flagged_output": any(t.flagged_injection for t in result.tool_calls),
                "answer_ok": answer_ok, "answer_failures": answer_fail,
                "steps": result.steps, "stop_reason": result.stop_reason,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd or 0.0,
            })

    done = [r for r in rows if "intent_ok" in r]
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for r in done:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["intent"] += r["intent_ok"]
        by_cat[r["category"]]["tools"] += r["tools_ok"]
        by_cat[r["category"]]["answer"] += r["answer_ok"]
    lat = [r["latency_ms"] for r in done]
    summary = {
        "mode": mode, "model": settings.model, "cases": len(cases), "completed": len(done),
        "errors": [r for r in rows if "error" in r or "skipped" in r],
        "intent_accuracy": round(sum(r["intent_ok"] for r in done) / max(len(done), 1), 3),
        "tool_selection_accuracy": round(sum(r["tools_ok"] for r in done) / max(len(done), 1), 3),
        "answer_pass_rate": round(sum(r["answer_ok"] for r in done) / max(len(done), 1), 3),
        "by_category": {k: {m: f"{v[m]}/{v['n']}" for m in ("intent", "tools", "answer")} for k, v in sorted(by_cat.items())},
        "latency_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95)},
        "avg_steps": round(statistics.mean(r["steps"] for r in done), 2) if done else None,
        "tokens": {"input": sum(r["input_tokens"] for r in done), "output": sum(r["output_tokens"] for r in done)},
        "estimated_cost_usd": round(sum(r["cost_usd"] for r in done), 4),
        "failures": [r for r in done if not (r["intent_ok"] and r["tools_ok"] and r["answer_ok"])],
    }
    if mode == "replay":
        live = RESULTS / "live.json"
        if live.exists():
            baseline = {r["id"]: r for r in json.loads(live.read_text())["rows"]}
            summary["regressions"] = [
                {"id": r["id"], "recorded": [baseline[r["id"]].get("intent"), baseline[r["id"]].get("tools")],
                 "replayed": [r["intent"], r["tools"]]}
                for r in done if r["id"] in baseline
                and (r["intent"], r["tools"]) != (baseline[r["id"]].get("intent"), baseline[r["id"]].get("tools"))
            ]
    return {**summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "live", "replay"], default="offline")
    parser.add_argument("--only", help="comma-separated case IDs or categories")
    args = parser.parse_args()

    cases = load_cases()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted or c["category"] in wanted]

    report = asyncio.run(run_offline(cases) if args.mode == "offline" else run_agent_cases(cases, args.mode))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.mode}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(printable, indent=2))
    if args.mode == "replay" and report.get("regressions"):
        sys.exit(1)


if __name__ == "__main__":
    main()
