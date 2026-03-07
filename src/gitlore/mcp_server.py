"""MCP server exposing gitlore context tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from gitlore.config import GitloreConfig
from gitlore.export import load_export_bundle
from gitlore.index import IndexStore, bundle_to_json
from gitlore.query import build_context, render_context


def create_mcp_server(config: GitloreConfig) -> FastMCP:
    """Create the MCP server for agent-facing context lookup."""
    server = FastMCP(
        name="gitlore",
        instructions=(
            "Use gitlore to retrieve repository tribal knowledge before editing code. "
            "Prefer get_context for task-scoped briefs, get_related_files to widen scope, "
            "and get_repo_rules for stable repo-wide guidance."
        ),
    )

    @server.tool(description="Return a task-scoped repository context bundle.")
    def get_context(
        task: str,
        files: list[str] | None = None,
        diff_path: str | None = None,
        format: str = "json",
        max_items: int | None = None,
        max_tokens: int | None = None,
        compress: bool = False,
    ) -> str:
        bundle = build_context(
            config,
            task=task,
            files=files,
            diff_path=diff_path,
            format_name=format,
            max_items=max_items,
            max_tokens=max_tokens,
            compress=compress,
        )
        if format == "json":
            return bundle_to_json(bundle)
        return render_context(bundle, format_name=format, config=config, compress=compress)

    @server.tool(description="Return related files for a given repository path.")
    def get_related_files(path: str, limit: int = 5) -> str:
        store = IndexStore(config.repo_path)
        try:
            related = store.get_related_files(path, limit=limit)
        finally:
            store.close()
        return "\n".join(f"{item.path}\t{item.reason}\t{item.score:.2f}" for item in related)

    @server.tool(description="Return stable repository rules and doc guidance.")
    def get_repo_rules() -> str:
        bundle = load_export_bundle(config)
        payload = {
            "facts": [
                {
                    "id": fact.id,
                    "kind": fact.kind.value,
                    "title": fact.title,
                    "guidance": fact.guidance,
                    "files": fact.files,
                    "evidence": [asdict(item) for item in fact.evidence],
                }
                for fact in bundle.facts
            ],
            "build_metadata": asdict(bundle.build_metadata) if bundle.build_metadata else None,
        }
        return json.dumps(payload, indent=2, default=str)

    return server


def run_stdio_server(config: GitloreConfig) -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server(config)
    asyncio.run(server.run_stdio_async())
