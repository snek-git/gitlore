"""Run a single-session repository investigation via Claude Agent SDK.

The build-time agent receives evidence query tools (precomputed analysis)
and git tools (read-only repo access), investigates the repository freely,
and calls record_note to emit knowledge notes as it goes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import warnings
from collections.abc import Callable
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    query,
)

from gitlore.config import GitloreConfig
from gitlore.docs import DocSnippet
from gitlore.models import AnalysisResult, KnowledgeNote
from gitlore.prompts import load as load_prompt
from gitlore.synthesis.evidence_tools import create_evidence_tools_server
from gitlore.synthesis.tools import create_git_tools_server

log = logging.getLogger("gitlore.synthesis")

# Suppress noisy SDK internal logging
logging.getLogger("claude_agent_sdk").setLevel(logging.CRITICAL)
logging.getLogger("claude_agent_sdk._internal").setLevel(logging.CRITICAL)

# ── Tool permission guard ─────────────────────────────────────────────────

_BLOCKED_TOOLS = {"Write", "Edit", "NotebookEdit"}

_DESTRUCTIVE_BASH = re.compile(
    r"(?:^|\||\&\&|\;|\$\()\s*(?:"
    r"rm\b|rmdir\b|mv\b|cp\b|chmod\b|chown\b"
    r"|git\s+(?:push|reset|checkout|restore|clean|branch\s+-[dD]|merge|rebase|commit|add|stash|tag\s+-d)"
    r"|sudo\b|curl\b.*\|\s*(?:sh|bash)|wget\b.*\|\s*(?:sh|bash)"
    r"|pip\b|uv\b|npm\b|yarn\b"
    r"|docker\b|kubectl\b"
    r"|>\s*\S|>>"
    r"|tee\b"
    r")",
    re.MULTILINE,
)

_ALLOWED_TOOLS = [
    # Built-in read-only tools
    "Read",
    "Grep",
    "Glob",
    # MCP git tools
    "mcp__git__show_commit",
    "mcp__git__file_history",
    "mcp__git__file_content",
    "mcp__git__blame_range",
    "mcp__git__diff_between",
    "mcp__git__list_changes",
    "mcp__git__repo_tree",
    # MCP evidence tools
    "mcp__evidence__get_hotspots",
    "mcp__evidence__get_coupling_for_file",
    "mcp__evidence__get_hub_files",
    "mcp__evidence__get_fix_after_chains",
    "mcp__evidence__get_revert_chains",
    "mcp__evidence__get_review_patterns",
    "mcp__evidence__get_review_comments",
    "mcp__evidence__get_doc_snippets",
    "mcp__evidence__get_conventions",
    "mcp__evidence__get_modules",
    "mcp__evidence__record_note",
]


async def _check_tool_permission(
    tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext,
) -> PermissionResultAllow | PermissionResultDeny:
    """Allow read-only tools, block destructive ones."""
    if tool_name in _BLOCKED_TOOLS:
        return PermissionResultDeny(message=f"Tool {tool_name} is not allowed in investigation")

    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        if _DESTRUCTIVE_BASH.search(cmd):
            return PermissionResultDeny(message=f"Destructive bash command blocked: {cmd}")

    return PermissionResultAllow()


# ── OpenRouter configuration ──────────────────────────────────────────────


def _configure_openrouter_env(model: str) -> None:
    """Set env vars so the Claude Agent SDK routes through OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Required for agentic investigation via OpenRouter."
        )
    or_model = model.removeprefix("openrouter/")
    os.environ["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
    os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key
    os.environ["ANTHROPIC_API_KEY"] = ""
    os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] = or_model


async def _prompt_stream(text: str):  # type: ignore[no-untyped-def]
    """Yield a single user message as an async iterable."""
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
    }


# ── Agent runner ──────────────────────────────────────────────────────────

_MAX_RETRIES = 3


async def _run_agent(
    prompt: str,
    repo_path: str,
    model: str,
    *,
    mcp_servers: dict[str, Any],
    _log_fn: Callable[[str], None] | None = None,
    system_prompt_name: str = "investigation_system",
) -> None:
    """Run the investigation agent session.

    The agent communicates findings via the record_note tool, so the
    return value is not important -- notes are collected via the tool.
    """
    _print: Callable[[str], None] = _log_fn if callable(_log_fn) else lambda _: None

    _configure_openrouter_env(model)

    stderr_lines: list[str] = []

    def _capture_stderr(line: str) -> None:
        stderr_lines.append(line)
        log.debug("claude-cli stderr: %s", line)

    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        stderr_lines.clear()
        options = ClaudeAgentOptions(
            system_prompt=load_prompt(system_prompt_name),
            mcp_servers=mcp_servers,
            allowed_tools=_ALLOWED_TOOLS,
            permission_mode="bypassPermissions",
            can_use_tool=_check_tool_permission,
            cwd=repo_path,
            stderr=_capture_stderr,
        )

        log.debug(
            "Starting investigation attempt %d/%d model=%s repo=%s",
            attempt, _MAX_RETRIES, model, repo_path,
        )

        try:
            msg_count = 0
            async for msg in query(prompt=_prompt_stream(prompt), options=options):
                msg_count += 1
                if isinstance(msg, AssistantMessage):
                    tool_names = []
                    current_text = ""
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            current_text += block.text
                        elif isinstance(block, ToolUseBlock):
                            if block.name == "Bash" and isinstance(block.input, dict):
                                tool_names.append(f"Bash({block.input.get('command', '')[:60]})")
                            elif block.name == "mcp__evidence__record_note":
                                text_preview = ""
                                if isinstance(block.input, dict):
                                    text_preview = str(block.input.get("text", ""))[:60]
                                tool_names.append(f"record_note({text_preview}...)")
                            else:
                                tool_names.append(block.name)
                        elif isinstance(block, ToolResultBlock):
                            if block.is_error:
                                content_preview = str(block.content)[:200] if block.content else ""
                                _print(f"  [{msg_count}] tool error: {content_preview}")
                    if current_text:
                        preview = current_text[:200].replace("\n", " ").strip()
                        if len(current_text) > 200:
                            preview += "..."
                        _print(f"  [{msg_count}] {preview}")
                    if tool_names:
                        _print(f"  [{msg_count}] -> {', '.join(tool_names)}")
                elif isinstance(msg, ResultMessage):
                    log.debug(
                        "ResultMessage: num_turns=%s is_error=%s",
                        msg.num_turns, msg.is_error,
                    )

            _print(f"  Investigation complete -- {msg_count} messages")
            return
        except Exception as e:
            last_err = e
            stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr)"
            log.warning(
                "Investigation attempt %d/%d failed: %s\nstderr: %s",
                attempt, _MAX_RETRIES, e, stderr_output,
            )
            if attempt < _MAX_RETRIES:
                log.info("Retrying in %ds...", 2 * attempt)
                await asyncio.sleep(2 * attempt)

    raise RuntimeError(
        f"Investigation failed after {_MAX_RETRIES} attempts: {last_err}"
        f"\nLast stderr:\n{chr(10).join(stderr_lines) if stderr_lines else '(none)'}"
    )


# ── Public API ────────────────────────────────────────────────────────────


def run_investigation(
    analysis: AnalysisResult,
    doc_snippets: list[DocSnippet],
    config: GitloreConfig,
    repo_path: str,
    *,
    _log_fn: object = None,
) -> list[KnowledgeNote]:
    """Run a single-session repository investigation and return discovered notes.

    The agent receives evidence tools (querying precomputed analysis) and git
    tools (read-only repo access). It calls record_note as it discovers things.
    """
    if not config.models.synthesizer or not os.environ.get("OPENROUTER_API_KEY", ""):
        return []

    note_collector: list[KnowledgeNote] = []

    evidence_server = create_evidence_tools_server(analysis, doc_snippets, note_collector)
    git_server = create_git_tools_server(repo_path)

    sources = ["git history"]
    if analysis.classified_comments or analysis.comment_clusters:
        sources.append("PR review comments")
    if doc_snippets:
        sources.append("repo docs/config")

    prompt = (
        f"Repository: {repo_path}\n"
        f"Commits analyzed: {analysis.total_commits_analyzed}\n"
        f"Available evidence sources: {', '.join(sources)}\n"
        "\n"
        "Investigate this repository. Use the evidence tools to find patterns, "
        "then use git tools to understand the codebase directly. "
        "Record your findings with record_note as you go."
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
        asyncio.run(
            _run_agent(
                prompt,
                repo_path,
                config.models.synthesizer,
                mcp_servers={"evidence": evidence_server, "git": git_server},
                _log_fn=_log_fn if callable(_log_fn) else None,
            )
        )

    return note_collector
