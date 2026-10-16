# Agent Runtime Lab

A tool-calling **Claude agent runtime** for engineering workflows. It routes each request by intent, retrieves from an engineering handbook (RAG), triages CI failures, looks up read-only market data, and exposes the same tools through an **MCP server**. Every run is guarded, traced and measurable through a **replayable evaluation harness**.

It is a learning and portfolio project built with synthetic data. It is not affiliated with any company and is not production-ready (see [Limitations](#limitations)).

```text
request ─► input guard ─► router (Claude structured output, keyword fallback)
                               │  intent + rewritten query
                               ▼
                 tool allowlist for that intent
                               │
            ┌──── Claude tool-calling loop (max N steps) ────┐
            │  tool_use ─► validate ─► run (timeout) ─►       │
            │  wrap as untrusted <tool_output> ─► tool_result │
            └──────────────────────────────────────────────────┘
                               │
                 answer + trace spans + tokens/cost ─► /metrics
```

## What it does

| Area | Implementation |
| --- | --- |
| **Intent routing and query rewriting** | `LLMRouter` asks Claude for `{intent, rewritten_query, confidence}` via structured outputs (JSON schema). Falls back to a deterministic `RuleRouter` on errors or low confidence. |
| **Tool calling** | Four read-only tools with strict JSON schemas (`strict: true`) plus Pydantic re-validation: `search_docs`, `list_ci_runs`, `get_ci_failure`, `get_market_price`. Parallel tool calls run concurrently with per-tool timeouts. |
| **Tool server (MCP)** | `agent_lab/mcp_server.py` exposes the same tools and validation over the Model Context Protocol (stdio). |
| **RAG** | Heading-aware Markdown chunking, TF-IDF vectors and cosine-similarity search, with source citations. Pure Python, so it is deterministic and testable offline. |
| **Developer productivity** | CI log triage classifies failures (assertion, timeout, dependency, network flake, OOM, lint), lists failed tests and flags likely flakes. Handbook Q&A covers CI, deploys, code review, on-call and more. |
| **Guardrails** | Input guard against prompt injection and exfiltration; tool output wrapped and flagged as untrusted (indirect injection); per-intent tool allowlist enforced at execution time; step limit. |
| **Context management and memory** | Per-session memory keeps only user turns and final answers within a size budget; oversized tool results are truncated. |
| **Observability** | Nested trace spans for router, every model call and every tool call (latency, tokens, stop reason, errors). `/metrics` reports counters, p50/p95 latency and estimated cost. |
| **Reliability** | Handles `tool_use`, `end_turn`, `max_tokens`, `pause_turn` and `refusal` stop reasons; server-side refusal fallback (`fallbacks: "default"`); invalid tool input and timeouts go back to Claude as `is_error` results so it can recover. |
| **Service** | Async FastAPI app: `POST /v1/agent/run`, `GET /v1/traces/{id}`, `GET /metrics`, `GET /health`. |

## Quick start

Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                               # 48 tests, no API key needed
python -m evals.run --mode offline      # deterministic evaluation, no API key needed
```

With a Claude API key (`export ANTHROPIC_API_KEY=...`):

```bash
python -m agent_lab "Why did run-1043 fail, and should I quarantine the test?" --trace
uvicorn agent_lab.api:create_app --factory --port 8000    # then POST /v1/agent/run
python -m agent_lab.mcp_server                            # MCP server over stdio
```

The default model is `claude-opus-5-5` (`AGENT_MODEL`, `AGENT_ROUTER_MODEL` and `AGENT_EFFORT` override it). Set `AGENT_OFFLINE=1` to use market-data fixtures instead of the Binance public API.

## Evaluation

`evals/cases.jsonl` holds 42 labelled cases: handbook questions, CI triage, market data, out-of-scope requests, direct prompt-injection attacks and one indirect injection hidden in a handbook page.

| Mode | Needs API key | Measures |
| --- | --- | --- |
| `offline` | No | Keyword-router accuracy, retrieval recall@3 and MRR, CI triage accuracy, input-guard block rate and false-positive rate, tool-output flagging |
| `live` | Yes | Full agent on Claude: router accuracy, tool-selection accuracy, answer checks, p50/p95 latency, tokens, estimated cost. Records one cassette per case. |
| `replay` | No | Re-runs the agent on recorded responses and fails if intent or tool sequence changed (regression check in CI) |

### Offline results (deterministic, reproducible with `python -m evals.run --mode offline`)

| Metric | Result |
| --- | --- |
| Keyword-router baseline accuracy | 73.5% (25/34 non-attack cases) |
| Retrieval recall@3 / MRR | 1.00 / 1.00 (15 queries) |
| CI triage category accuracy | 8/8 |
| Input guard: attacks blocked | 8/8 |
| Input guard: false positives on benign requests | 0/34 |
| Tool-output guard | flags only the poisoned page (`vendor-notes.md`) out of 37 chunks |

**Read these honestly.** The data set is small and synthetic, and the guardrail patterns and test cases were written by the same person, so 100% here shows that the components work as designed, not that they generalise. The keyword router's 73.5% is the baseline that the Claude router has to beat.

### Live results

Not yet recorded. Run `python -m evals.run --mode live`; results go to `evals/results/live.json` and cassettes to `evals/cassettes/`.

## Testing

`pytest` runs 48 tests: unit tests for retrieval, routing, guardrails, tools, memory and tracing, plus **integration tests that drive the full agent loop with mocked Claude responses** (`ScriptedLLM` builds real SDK `BetaMessage` objects). They cover the happy path, parallel tool calls, a disallowed tool, invalid tool input with recovery, tool timeouts, the step limit, a refusal, prompt-injection blocking, a poisoned document, router fallback, session memory, the HTTP API and the MCP server. GitHub Actions runs the tests and the offline evaluation on Python 3.11–3.13.

## Project layout

```text
agent_lab/
  agent.py        agent loop, tool execution, run result
  router.py       LLM router (structured output) + keyword baseline + tool allowlist
  tools.py        tool schemas, validation and handlers
  retrieval.py    chunking + TF-IDF vector search
  guardrails.py   input and tool-output guards
  memory.py       session memory with a size budget
  tracing.py      spans and JSONL export
  llm.py          Anthropic client, scripted / recording / replay clients
  api.py          FastAPI service
  mcp_server.py   MCP tool server
data/             synthetic handbook, CI logs, market fixtures
evals/            cases, harness, results, cassettes
tests/            unit and integration tests
```

## Limitations

- Synthetic data only: a nine-page handbook and six CI logs. Retrieval is lexical (TF-IDF), so it misses synonyms that dense embeddings would catch.
- Guardrails are pattern-based. They catch known phrasings, not paraphrases or encoded attacks.
- Memory is in-process; traces and metrics are not persisted across restarts.
- No authentication or rate limiting on the HTTP API; keep it on localhost.
- Market data is read-only and informational. The agent never places trades.

Possible next steps: dense embeddings with a vector store, an LLM-graded answer-quality metric, persisted traces (OpenTelemetry), and per-tool latency budgets.
