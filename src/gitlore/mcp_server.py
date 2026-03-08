"""MCP server exposing planning-time gitlore tools."""

from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from gitlore.config import GitloreConfig
from gitlore.export import load_export_bundle
from gitlore.index import IndexStore, guidance_to_json
from gitlore.query import build_planning_brief, render_planning_brief


def create_mcp_server(config: GitloreConfig) -> FastMCP:
    """Create the MCP server for planning-time retrieval."""
    server = FastMCP(
        name="gitlore",
        instructions=(
            "Use gitlore during planning to retrieve repository-specific tribal knowledge. "
            "Prefer get_planning_brief before editing code, get_related_files to widen scope, "
            "and get_repo_guidance for stable repo-wide guidance."
        ),
    )

    @server.tool(description="Return a small planning brief for the current task and candidate scope.")
    def get_planning_brief(
        task: str,
        files: list[str] | None = None,
        diff_path: str | None = None,
        tentative_plan: str = "",
        question: str = "",
        format: str = "json",
        max_notes: int = 5,
    ) -> str:
        brief = build_planning_brief(
            config,
            task=task,
            files=files,
            diff_path=diff_path,
            tentative_plan=tentative_plan,
            question=question,
            max_notes=max_notes,
        )
        return render_planning_brief(brief, format_name=format)

    @server.tool(description="Return related files for a given repository path.")
    def get_related_files(path: str, limit: int = 5) -> str:
        store = IndexStore(config.repo_path)
        try:
            related = store.get_related_files(path, limit=limit)
        finally:
            store.close()
        return "\n".join(f"{item.path}\t{item.reason}\t{item.score:.2f}" for item in related)

    @server.tool(description="Return short repo-wide guidance cards.")
    def get_repo_guidance() -> str:
        bundle = load_export_bundle(config)
        return guidance_to_json(bundle.cards, bundle.build_metadata)

    return server


def run_stdio_server(config: GitloreConfig) -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server(config)
    asyncio.run(server.run_stdio_async())
