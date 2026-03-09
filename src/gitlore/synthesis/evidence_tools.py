"""Evidence query tools for build-time repository investigation.

Exposes precomputed analysis results as MCP tools so the build-time agent
can pull evidence on demand rather than receiving a pre-digested feed.
Also provides the record_note tool for incremental knowledge capture.
"""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from gitlore.docs import DocSnippet
from gitlore.models import (
    AnalysisResult,
    CommentCategory,
)


def _ok(data: Any) -> dict[str, Any]:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _err(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def _create_evidence_tools(
    analysis: AnalysisResult,
    doc_snippets: list[DocSnippet],
) -> list[Any]:
    """Create all evidence query tools and the record_note tool."""

    # ── Evidence query tools ────────────────────────────────────────────

    @tool(
        "get_hotspots",
        "Get files with high churn and fix frequency. Returns file path, commit count, "
        "fix ratio, churn score. Use this to find areas that change often and break often.",
        {"limit": int},
    )
    async def get_hotspots(args: dict[str, Any]) -> dict[str, Any]:
        limit = min(args.get("limit", 20) or 20, 50)
        sorted_hs = sorted(analysis.hotspots, key=lambda h: h.score, reverse=True)[:limit]
        return _ok([
            {
                "path": h.path,
                "commit_count": h.commit_count,
                "fix_ratio": round(h.fix_ratio, 2),
                "churn_score": round(h.score, 1),
                "lines_added": h.lines_added,
                "lines_deleted": h.lines_deleted,
            }
            for h in sorted_hs
        ])

    @tool(
        "get_coupling_for_file",
        "Get files that frequently change together with the given file. Returns coupled "
        "file path, directional confidence, lift, and shared commit count.",
        {"path": str},
    )
    async def get_coupling_for_file(args: dict[str, Any]) -> dict[str, Any]:
        path = args["path"]
        results = []
        for pair in analysis.coupling_pairs:
            if pair.file_a == path:
                results.append({
                    "coupled_file": pair.file_b,
                    "confidence": round(pair.confidence_a_to_b, 2),
                    "reverse_confidence": round(pair.confidence_b_to_a, 2),
                    "lift": round(pair.lift, 2),
                    "shared_commits": int(pair.shared_commits),
                })
            elif pair.file_b == path:
                results.append({
                    "coupled_file": pair.file_a,
                    "confidence": round(pair.confidence_b_to_a, 2),
                    "reverse_confidence": round(pair.confidence_a_to_b, 2),
                    "lift": round(pair.lift, 2),
                    "shared_commits": int(pair.shared_commits),
                })
        results.sort(key=lambda r: float(str(r["confidence"])), reverse=True)
        if not results:
            return _ok(f"No coupling data found for {path}")
        return _ok(results)

    @tool(
        "get_hub_files",
        "Get files that act as coordination points, coupling with many other files. "
        "High-centrality files where changes often have wider scope than the diff suggests.",
        {"limit": int},
    )
    async def get_hub_files(args: dict[str, Any]) -> dict[str, Any]:
        limit = min(args.get("limit", 10) or 10, 30)
        sorted_hubs = sorted(analysis.hub_files, key=lambda h: h.coupled_file_count, reverse=True)[:limit]
        return _ok([
            {
                "path": h.path,
                "coupled_file_count": h.coupled_file_count,
                "total_coupling_weight": round(h.total_coupling_weight, 1),
            }
            for h in sorted_hubs
        ])

    @tool(
        "get_fix_after_chains",
        "Get commits that needed follow-up fixes. Shows the original commit subject, "
        "follow-up fix subjects, affected files, urgency tier, and time span. "
        "Use this to find areas where initial changes frequently miss something.",
        {"file": str, "limit": int},
    )
    async def get_fix_after_chains(args: dict[str, Any]) -> dict[str, Any]:
        limit = min(args.get("limit", 10) or 10, 30)
        file_filter = args.get("file", "") or ""
        chains = analysis.fix_after_chains
        if file_filter:
            chains = [c for c in chains if file_filter in c.files]
        sorted_chains = sorted(chains, key=lambda c: len(c.fixup_hashes), reverse=True)[:limit]
        return _ok([
            {
                "original_hash": c.original_hash[:10],
                "original_subject": c.original_subject,
                "fixup_count": len(c.fixup_hashes),
                "fixup_subjects": c.fixup_subjects,
                "tier": c.tier.value,
                "time_span": str(c.time_span),
                "files": c.files,
            }
            for c in sorted_chains
        ])

    @tool(
        "get_revert_chains",
        "Get commits that were reverted. Shows original subject, revert depth, "
        "affected files, and whether the change is effectively reverted or re-landed.",
        {"file": str, "limit": int},
    )
    async def get_revert_chains(args: dict[str, Any]) -> dict[str, Any]:
        limit = min(args.get("limit", 10) or 10, 30)
        file_filter = args.get("file", "") or ""
        chains = analysis.revert_chains
        if file_filter:
            chains = [c for c in chains if file_filter in c.files]
        sorted_chains = sorted(chains, key=lambda c: c.depth, reverse=True)[:limit]
        return _ok([
            {
                "original_hash": c.original_hash[:10],
                "original_subject": c.original_subject,
                "revert_depth": c.depth,
                "effectively_reverted": c.is_effectively_reverted,
                "files": c.files,
            }
            for c in sorted_chains
        ])

    @tool(
        "get_review_patterns",
        "Get recurring themes from PR review comments. Returns cluster labels, "
        "comment counts, affected files, and sample comment bodies. "
        "Use this to understand what reviewers consistently care about.",
        {"file": str},
    )
    async def get_review_patterns(args: dict[str, Any]) -> dict[str, Any]:
        file_filter = args.get("file", "") or ""
        if analysis.comment_clusters:
            clusters = analysis.comment_clusters
            if file_filter:
                clusters = [
                    c for c in clusters
                    if any(
                        item.comment.file_path and file_filter in item.comment.file_path
                        for item in c.comments
                    )
                ]
            return _ok([
                {
                    "label": c.label,
                    "comment_count": len(c.comments),
                    "coherence": round(c.coherence, 2),
                    "files": sorted({
                        item.comment.file_path
                        for item in c.comments
                        if item.comment.file_path
                    })[:10],
                    "sample_comments": [
                        item.comment.body[:200] for item in c.comments[:3]
                    ],
                }
                for c in sorted(clusters, key=lambda c: len(c.comments), reverse=True)
            ])
        return _ok("No review pattern data available. GitHub review enrichment may not have run.")

    @tool(
        "get_review_comments",
        "Get individual classified PR review comments, optionally filtered by file or "
        "category (bug, architecture, convention, security, performance). "
        "Use this to see specific reviewer feedback.",
        {"file": str, "category": str, "limit": int},
    )
    async def get_review_comments(args: dict[str, Any]) -> dict[str, Any]:
        limit = min(args.get("limit", 20) or 20, 50)
        file_filter = args.get("file", "") or ""
        category_filter = args.get("category", "") or ""
        comments = analysis.classified_comments
        if file_filter:
            comments = [
                c for c in comments
                if c.comment.file_path and file_filter in c.comment.file_path
            ]
        if category_filter:
            try:
                cat = CommentCategory(category_filter)
                comments = [c for c in comments if cat in c.categories]
            except ValueError:
                return _err(
                    f"Unknown category: {category_filter}. "
                    "Valid: bug, architecture, convention, security, performance, nitpick, question, praise"
                )
        notable = [
            c for c in comments
            if c.confidence >= 0.5 and any(
                cat not in (CommentCategory.NITPICK, CommentCategory.PRAISE)
                for cat in c.categories
            )
        ]
        notable.sort(key=lambda c: c.confidence, reverse=True)
        return _ok([
            {
                "pr_number": c.comment.pr_number,
                "file": c.comment.file_path,
                "categories": [cat.value for cat in c.categories],
                "confidence": round(c.confidence, 2),
                "body": c.comment.body[:300],
                "author": c.comment.author,
            }
            for c in notable[:limit]
        ])

    @tool(
        "get_doc_snippets",
        "Get extracted documentation and config snippets from the repo "
        "(README, CONTRIBUTING, CI configs, pyproject.toml, etc). "
        "Optionally filter by topic keyword.",
        {"topic": str},
    )
    async def get_doc_snippets(args: dict[str, Any]) -> dict[str, Any]:
        topic = (args.get("topic", "") or "").lower()
        snippets = doc_snippets
        if topic:
            snippets = [
                s for s in snippets
                if topic in s.title.lower() or topic in s.content.lower()
            ]
        return _ok([
            {
                "path": s.path,
                "title": s.title,
                "content": s.content[:500],
                "source_type": s.source_type,
            }
            for s in snippets[:20]
        ])

    @tool(
        "get_conventions",
        "Get detected commit message conventions: format style, adherence rate, "
        "common types and scopes, detected rules. Use this to understand how "
        "the team structures commits.",
        {},
    )
    async def get_conventions(args: dict[str, Any]) -> dict[str, Any]:
        conv = analysis.conventions
        if conv is None:
            return _ok("No commit convention data available.")
        top_types = sorted(conv.types_used.items(), key=lambda x: x[1], reverse=True)[:8]
        top_scopes = sorted(conv.scopes_used.items(), key=lambda x: x[1], reverse=True)[:8]
        return _ok({
            "primary_format": conv.primary_format,
            "format_adherence": round(conv.format_adherence, 2),
            "types_used": dict(top_types),
            "scopes_used": dict(top_scopes),
            "ticket_format": conv.ticket_format,
            "ticket_adherence": round(conv.ticket_adherence, 2),
            "detected_rules": conv.detected_rules,
        })

    @tool(
        "get_modules",
        "Get implicit module groupings detected from co-change patterns. "
        "Shows which files tend to form logical units based on how they change together.",
        {},
    )
    async def get_modules(args: dict[str, Any]) -> dict[str, Any]:
        if not analysis.implicit_modules:
            return _ok("No implicit module data available.")
        return _ok([
            {
                "module_id": m.module_id,
                "files": m.files,
                "internal_coupling_avg": round(m.internal_coupling_avg, 2),
            }
            for m in sorted(
                analysis.implicit_modules,
                key=lambda m: len(m.files),
                reverse=True,
            )
        ])

    return [
        get_hotspots,
        get_coupling_for_file,
        get_hub_files,
        get_fix_after_chains,
        get_revert_chains,
        get_review_patterns,
        get_review_comments,
        get_doc_snippets,
        get_conventions,
        get_modules,
    ]


def create_evidence_tools_server(
    analysis: AnalysisResult,
    doc_snippets: list[DocSnippet],
) -> Any:
    """Create an MCP server with evidence query tools."""
    tools = _create_evidence_tools(analysis, doc_snippets)
    return create_sdk_mcp_server(
        name="evidence",
        version="1.0.0",
        tools=tools,
    )
