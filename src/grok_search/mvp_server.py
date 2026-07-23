"""Compact synthesized web search with a small usage guide."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from .mvp_pipeline import SearchPipeline


mcp = FastMCP("GrokSearchMVP")
pipeline = SearchPipeline()


@mcp.tool(
    name="web_search",
    output_schema=None,
    description="""
Search the live web and return one compact, citation-backed answer.

Use whenever the user needs current or externally verifiable information. The server searches Exa
and Tavily with 15 results each, deduplicates evidence, and synthesizes it internally. Returns only
the organized answer, key findings, conflicts, compact citation metadata, and provider status. Raw
search results, excerpts, cached IDs, and page content are never exposed. If the call fails, follow
error.resolution exactly.
""",
)
async def web_search(
    query: Annotated[
        str,
        Field(
            description="A concise, self-contained web search query of 1-400 characters.",
            min_length=1,
            max_length=400,
        ),
    ],
    instructions: Annotated[
        str,
        Field(
            description=(
                "Optional research brief for the internal AI. State what information is needed, "
                "relevant dates or time window, geography, preferred source types, comparison "
                "criteria, exclusions, and desired emphasis. Example: 'Focus on changes since "
                "2025, prefer official sources, compare security and cost, and flag conflicts.'"
            ),
            max_length=2000,
        ),
    ] = "",
) -> dict:
    return await pipeline.search_and_synthesize(query, instructions)


@mcp.tool(
    name="search_guide",
    output_schema=None,
    description="""
Return a short guide for using GrokSearchMVP:web_search effectively.

Use only when the calling AI is unsure how to form query or instructions. This tool performs no
search and returns no web content. It explains how to specify needed information, time window,
source preferences, comparisons, exclusions, and desired emphasis.
""",
)
async def search_guide() -> dict:
    return {
        "tool": "GrokSearchMVP:web_search",
        "usage": (
            "Put the core question in query. Use instructions as a research brief for the internal "
            "AI: say what information matters, when it must be from, how to filter sources, and "
            "what comparisons or conflicts to emphasize."
        ),
        "instructions_can_include": [
            "time window or cutoff date",
            "region, platform, company, or topic scope",
            "preferred source types such as official documentation",
            "comparison criteria and exclusions",
            "desired depth, emphasis, or answer language",
        ],
        "example": {
            "query": "Compare leading AI coding assistants",
            "instructions": (
                "Use information from 2026, prefer official pricing and security documentation, "
                "compare cost and long-context support, and clearly flag conflicting claims."
            ),
        },
        "returns": (
            "Only the internal AI's organized answer and compact citation metadata; raw search "
            "results and page content are never exposed."
        ),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
