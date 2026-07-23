import asyncio

import pytest

from grok_search.mvp_pipeline import (
    MVPSettings,
    SearchCache,
    SearchPipeline,
    canonicalize_url,
    normalize_and_deduplicate,
)


def _settings(**overrides):
    values = {
        "exa_api_key": "exa-test",
        "tavily_api_key": "tavily-test",
        "synth_api_url": "https://synth.example/v1",
        "synth_api_key": "synth-test",
        "synth_model": "deepseek-flash",
        "cache_ttl_seconds": 1800,
    }
    values.update(overrides)
    return MVPSettings(**values)


def test_canonicalize_url_removes_tracking_and_fragment():
    assert canonicalize_url("HTTPS://Example.COM/a/?utm_source=x&b=2#part") == "https://example.com/a?b=2"


def test_normalize_and_deduplicate_merges_provider_evidence():
    sources = normalize_and_deduplicate(
        {
            "exa": [
                {
                    "title": "Short title",
                    "url": "https://example.com/item?utm_source=exa",
                    "provider_score": 0.9,
                    "excerpt": "A shared excerpt with enough text to participate in exact matching. " * 2,
                    "raw_content": "Exa preserved detail. " * 10,
                }
            ],
            "tavily": [
                {
                    "title": "A more descriptive source title",
                    "url": "https://example.com/item",
                    "provider_score": 0.8,
                    "excerpt": "Tavily adds a second detail.",
                    "raw_content": "Tavily preserved detail. " * 10,
                }
            ],
        }
    )

    assert len(sources) == 1
    assert sources[0]["id"] == "S1"
    assert sources[0]["providers"] == ["exa", "tavily"]
    assert sources[0]["title"] == "A more descriptive source title"
    assert "Exa preserved detail" in sources[0]["raw_content"]
    assert "Tavily preserved detail" in sources[0]["raw_content"]


@pytest.mark.asyncio
async def test_search_runs_providers_concurrently_and_allows_partial_success():
    pipeline = SearchPipeline(_settings())
    started = set()
    both_started = asyncio.Event()

    async def exa_search(query, max_results):
        started.add("exa")
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.5)
        raise RuntimeError("temporary Exa failure")

    async def tavily_search(query, max_results):
        started.add("tavily")
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.5)
        return [
            {
                "title": "Working source",
                "url": "https://example.com/working",
                "excerpt": "Useful current evidence.",
                "raw_content": "Useful current evidence with more context.",
                "provider_score": 0.7,
            }
        ]

    pipeline._search_exa = exa_search
    pipeline._search_tavily = tavily_search
    result = await pipeline.search_sources("test query", 3)

    assert result["ok"] is True
    assert result["status"] == "partial"
    assert result["provider_status"]["exa"]["status"] == "error"
    assert result["provider_status"]["tavily"]["status"] == "ok"
    assert result["sources"][0]["providers"] == ["tavily"]


@pytest.mark.asyncio
async def test_cache_expires_entries():
    now = [100.0]
    cache = SearchCache(ttl_seconds=30, max_entries=2, clock=lambda: now[0])
    await cache.set("search_test", {"value": 1})
    assert await cache.get("search_test") == {"value": 1}
    now[0] = 131.0
    assert await cache.get("search_test") is None


@pytest.mark.asyncio
async def test_synthesis_parses_fenced_json_and_filters_invalid_citations():
    pipeline = SearchPipeline(_settings())

    async def tavily_search(query, max_results):
        return [
            {
                "title": "Source",
                "url": "https://example.com/source",
                "excerpt": "Evidence",
                "raw_content": "Evidence with sufficient context.",
                "provider_score": 0.9,
            }
        ]

    async def exa_search(query, max_results):
        return []

    async def synthesis(prompt):
        return """```json
        {
          "summary": "A cited summary.",
          "key_findings": [
            {"title": "Kept", "details": "Supported.", "source_ids": ["S1"]},
            {"title": "Dropped", "details": "Unsupported.", "source_ids": ["S99"]}
          ],
          "conflicts": []
        }
        ```"""

    pipeline._search_exa = exa_search
    pipeline._search_tavily = tavily_search
    pipeline._call_synthesis = synthesis

    search = await pipeline.search_sources("test query")
    result = await pipeline.synthesize_search(search["search_id"])

    assert result["ok"] is True
    assert result["summary"] == "A cited summary."
    assert [finding["title"] for finding in result["key_findings"]] == ["Kept"]
    assert result["validation_warnings"]
    assert result["sources"][0]["id"] == "S1"


@pytest.mark.asyncio
async def test_missing_synthesis_config_returns_cached_sources():
    pipeline = SearchPipeline(
        _settings(synth_api_url=None, synth_api_key=None, synth_model=None)
    )

    async def exa_search(query, max_results):
        return [
            {
                "title": "Source",
                "url": "https://example.com/source",
                "excerpt": "Evidence",
                "raw_content": "Evidence",
            }
        ]

    async def tavily_search(query, max_results):
        return []

    pipeline._search_exa = exa_search
    pipeline._search_tavily = tavily_search
    search = await pipeline.search_sources("test query")
    result = await pipeline.synthesize_search(search["search_id"])

    assert result["ok"] is False
    assert result["error"]["code"] == "SYNTHESIS_NOT_CONFIGURED"
    assert result["sources"][0]["url"] == "https://example.com/source"
