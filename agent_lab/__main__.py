"""Command line: python -m agent_lab "Why did run-1042 fail?" [--rules-router] [--trace]"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from . import build_agent


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Ask the engineering agent a question.")
    parser.add_argument("query")
    parser.add_argument("--rules-router", action="store_true", help="use the keyword baseline router")
    parser.add_argument("--trace", action="store_true", help="print the span trace")
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=5.0) as http:
        agent = build_agent(http=http, use_llm_router=not args.rules_router)
        result = await agent.run(args.query)

    print(result.answer)
    print(f"\n[intent={result.intent} router={result.router} steps={result.steps} "
          f"tools={[c.name for c in result.tool_calls]} tokens={result.input_tokens}/{result.output_tokens} "
          f"cost=${result.cost_usd or 0:.4f} latency={result.latency_ms}ms trace={result.trace_id}]")
    if args.trace:
        print(json.dumps(result.trace, indent=1))


if __name__ == "__main__":
    asyncio.run(_main())
