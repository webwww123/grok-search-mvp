"""Two-tool FastMCP server for the minimal search pipeline."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from .mvp_pipeline import SearchPipeline


mcp = FastMCP("GrokSearchMVP")
pipeline = SearchPipeline()


@mcp.tool(
    name="search_sources",
    output_schema=None,
    description="""
Search Exa and Tavily concurrently, then normalize, rank, and deduplicate their evidence.

Use this first whenever the user needs current web information. It returns an opaque search_id,
provider status, and complete normalized sources. One provider may fail without discarding the
other provider's results. For a final written answer, pass search_id to
GrokSearchMVP:synthesize_search. If the call fails, follow the returned error.resolution exactly.
""",
)
async def search_sources(
    query: Annotated[
        str,
        Field(
            description="A concise, self-contained web search query of 1-400 characters.",
            min_length=1,
            max_length=400,
        ),
    ],
    max_results: Annotated[
        int,
        Field(
            description="Maximum results requested from each provider; use 6 normally.",
            ge=1,
            le=10,
        ),
    ] = 6,
) -> dict:
    return await pipeline.search_sources(query, max_results)


@mcp.tool(
    name="synthesize_search",
    output_schema=None,
    description="""
Turn cached search evidence into a structured, citation-backed answer using the configured
OpenAI-compatible synthesis model.

Use only after GrokSearchMVP:search_sources and pass its search_id. The result includes a summary,
deduplicated key findings, conflicts, source_ids, and the original normalized sources. If the ID is
expired, call search_sources again. If synthesis fails, the response still includes source evidence.
""",
)
async def synthesize_search(
    search_id: Annotated[
        str,
        Field(description="The exact search_id returned by GrokSearchMVP:search_sources."),
    ],
    focus: Annotated[
        str,
        Field(
            description="Optional emphasis for synthesis; leave empty to answer the original query.",
            max_length=1000,
        ),
    ] = "",
) -> dict:
    return await pipeline.synthesize_search(search_id, focus)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
