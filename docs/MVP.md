# Two-stage search MVP

This optional server is intentionally separate from the original GrokSearch server. It exposes two safe tools:

1. `web_search(query, instructions="")` searches Exa and Tavily with 30 results each, deduplicates the evidence, sends it to the configured OpenAI-compatible model, and returns a structured citation-backed answer. `instructions` is a free-form research brief from the calling AI describing needed information, time range, filtering rules, and emphasis.
2. `search_guide()` returns a short usage guide and example. It performs no search.

Raw results, excerpts, page content, and cache IDs remain server-side and are never returned through MCP. Only the synthesized answer and compact citation metadata are exposed. One search provider may fail without discarding the other provider's results. Tavily's generated answer is disabled.

## Configuration

Copy the variable names from `.env.mvp.example`. Keep real credentials outside the Git repository and load them into the process environment.

Required for full operation:

- `EXA_API_KEY`
- `TAVILY_API_KEY`
- `SYNTH_API_URL`
- `SYNTH_API_KEY`
- `SYNTH_MODEL`

## Run

```bash
uv sync --extra dev
uv run grok-search-mvp
```

The default transport is stdio. The existing `grok-search` entry point and deployment are unchanged.
