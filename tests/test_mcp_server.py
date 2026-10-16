from agent_lab import mcp_server


async def test_mcp_server_lists_all_tools():
    names = {t.name for t in await mcp_server.server.list_tools()}
    assert names == {"search_docs", "list_ci_runs", "get_ci_failure", "get_market_price"}


async def test_mcp_tool_call_uses_shared_validation(monkeypatch, ctx):
    monkeypatch.setattr(mcp_server, "_ctx", ctx)
    result = await mcp_server.get_ci_failure("run-1046")
    assert result["category"] == "out_of_memory"
    out = await mcp_server.search_docs("canary error rate", 2)
    assert out["results"][0]["source"] == "deploy-process.md"
