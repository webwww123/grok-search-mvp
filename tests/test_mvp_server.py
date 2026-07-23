import pytest

from grok_search.mvp_server import mcp


@pytest.mark.asyncio
async def test_mvp_exposes_exactly_two_tools():
    tools = await mcp.list_tools()
    assert {tool.name for tool in tools} == {"search_sources", "synthesize_search"}
