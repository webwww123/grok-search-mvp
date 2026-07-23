import pytest

from grok_search.mvp_server import mcp, search_guide


@pytest.mark.asyncio
async def test_mvp_exposes_search_and_guide_only():
    tools = await mcp.list_tools()
    assert {tool.name for tool in tools} == {"web_search", "search_guide"}


@pytest.mark.asyncio
async def test_search_guide_is_compact_and_mentions_instructions():
    guide = await search_guide()
    assert guide["tool"] == "GrokSearchMVP:web_search"
    assert "instructions" in guide["usage"]
    assert "raw search" in guide["returns"]
