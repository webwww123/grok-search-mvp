# Two-stage search MVP

This optional server is intentionally separate from the original GrokSearch server. It exposes only:

1. `search_sources(query, max_results=6)` — searches Exa and Tavily concurrently, normalizes and deduplicates evidence, and returns a 30-minute `search_id`.
2. `synthesize_search(search_id, focus="")` — sends the cached evidence to an OpenAI-compatible model and returns a structured, citation-backed answer plus the original sources.

One search provider may fail without discarding the other provider's results. Synthesis never calls the web directly and Tavily's generated answer is disabled.

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
