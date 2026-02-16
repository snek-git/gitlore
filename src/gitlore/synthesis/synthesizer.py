"""Synthesize analysis patterns into actionable rules via agentic LLM loop.

Uses the Claude Agent SDK to run an agentic loop where the LLM receives
pre-computed analysis data plus git investigation tools, investigates the
most interesting patterns, and then synthesizes rules from real understanding.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import warnings

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)

from gitlore.config import GitloreConfig
from gitlore.models import (
    AnalysisResult,
    ChurnHotspot,
    ClassifiedComment,
    CommentCategory,
    CommentCluster,
    CouplingPair,
    EvidencePoint,
    Finding,
    FindingCategory,
    FindingSeverity,
    FixAfterChain,
    HubFile,
    RevertChain,
    SynthesisResult,
)
from gitlore.prompts import load as load_prompt
from gitlore.synthesis.tools import create_git_tools_server

log = logging.getLogger("gitlore.synthesis")

# ── XML serialization ────────────────────────────────────────────────────────


def _xml_escape(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _coupling_to_xml(pairs: list[CouplingPair], limit: int = 25) -> str:
    """Convert coupling pairs to XML."""
    if not pairs:
        return ""
    sorted_pairs = sorted(pairs, key=lambda p: p.strength, reverse=True)[:limit]
    lines = ['  <category name="co-change-coupling">']
    for p in sorted_pairs:
        conf = max(p.confidence_a_to_b, p.confidence_b_to_a)
        lines.append(
            f'    <pattern confidence="{conf:.2f}" occurrences="{p.shared_commits:.0f}" lift="{p.lift:.2f}">'
        )
        lines.append(
            f"      <files>{_xml_escape(p.file_a)}, {_xml_escape(p.file_b)}</files>"
        )
        lines.append(
            f"      <description>{_xml_escape(p.file_a)} and {_xml_escape(p.file_b)} change together"
            f" in {conf:.0%} of commits touching either file (lift {p.lift:.1f}x)</description>"
        )
        lines.append("    </pattern>")
    lines.append("  </category>")
    return "\n".join(lines)


def _hotspots_to_xml(hotspots: list[ChurnHotspot], limit: int = 20) -> str:
    """Convert churn hotspots to XML."""
    if not hotspots:
        return ""
    sorted_hs = sorted(hotspots, key=lambda h: h.score, reverse=True)[:limit]
    lines = ['  <category name="churn-hotspots">']
    for h in sorted_hs:
        lines.append(
            f'    <pattern confidence="{min(h.score / 10, 1.0):.2f}" occurrences="{h.commit_count}">'
        )
        lines.append(f"      <file>{_xml_escape(h.path)}</file>")
        lines.append(f"      <churn_score>{h.score:.1f}</churn_score>")
        lines.append(f"      <fix_ratio>{h.fix_ratio:.2f}</fix_ratio>")
        lines.append(
            f"      <description>{_xml_escape(h.path)} has high edit frequency"
            f" ({h.commit_count} commits, {h.lines_added + h.lines_deleted} lines churned,"
            f" {h.fix_ratio:.0%} are fixes)</description>"
        )
        lines.append("    </pattern>")
    lines.append("  </category>")
    return "\n".join(lines)


def _reverts_to_xml(chains: list[RevertChain], limit: int = 20) -> str:
    """Convert revert chains to XML."""
    if not chains:
        return ""
    sorted_chains = sorted(chains, key=lambda r: r.depth, reverse=True)[:limit]
    lines = ['  <category name="reverts">']
    for r in sorted_chains:
        lines.append(
            f'    <pattern confidence="0.90" occurrences="{r.depth}">'
        )
        lines.append(
            f"      <files>{', '.join(_xml_escape(f) for f in r.files)}</files>"
        )
        lines.append(
            f"      <original_subject>{_xml_escape(r.original_subject)}</original_subject>"
        )
        lines.append(
            f"      <description>Commit \"{_xml_escape(r.original_subject)}\" was reverted"
            f" {r.depth} time(s), affecting files: {_xml_escape(', '.join(r.files[:5]))}</description>"
        )
        lines.append("    </pattern>")
    lines.append("  </category>")
    return "\n".join(lines)


def _fix_after_to_xml(chains: list[FixAfterChain], limit: int = 20) -> str:
    """Convert fix-after chains to XML."""
    if not chains:
        return ""
    sorted_chains = sorted(chains, key=lambda c: len(c.fixup_hashes), reverse=True)[
        :limit
    ]
    lines = ['  <category name="fix-after">']
    for c in sorted_chains:
        lines.append(
            f'    <pattern confidence="0.80" occurrences="{len(c.fixup_hashes)}" tier="{c.tier.value}">'
        )
        lines.append(
            f"      <files>{', '.join(_xml_escape(f) for f in c.files)}</files>"
        )
        lines.append(
            f"      <original_subject>{_xml_escape(c.original_subject)}</original_subject>"
        )
        lines.append(
            f"      <fixup_subjects>{'; '.join(_xml_escape(s) for s in c.fixup_subjects)}</fixup_subjects>"
        )
        lines.append(
            f"      <description>\"{_xml_escape(c.original_subject)}\" required"
            f" {len(c.fixup_hashes)} follow-up fix(es) within {c.time_span}</description>"
        )
        lines.append("    </pattern>")
    lines.append("  </category>")
    return "\n".join(lines)


def _conventions_to_xml(result: AnalysisResult) -> str:
    """Convert commit conventions to XML."""
    conv = result.conventions
    if conv is None:
        return ""
    lines = ['  <category name="commit-conventions">']
    lines.append(f'    <pattern confidence="{conv.format_adherence:.2f}" occurrences="{sum(conv.types_used.values())}">')
    lines.append(f"      <convention>{_xml_escape(conv.primary_format)}</convention>")
    lines.append(f"      <adherence>{conv.format_adherence:.0%}</adherence>")
    if conv.types_used:
        top_types = sorted(conv.types_used.items(), key=lambda x: x[1], reverse=True)[:8]
        lines.append(f"      <types_used>{', '.join(f'{t}({c})' for t, c in top_types)}</types_used>")
    if conv.scopes_used:
        top_scopes = sorted(conv.scopes_used.items(), key=lambda x: x[1], reverse=True)[:8]
        lines.append(f"      <scopes_used>{', '.join(f'{s}({c})' for s, c in top_scopes)}</scopes_used>")
    if conv.ticket_format:
        lines.append(f"      <ticket_format>{_xml_escape(conv.ticket_format)} ({conv.ticket_adherence:.0%} adherence)</ticket_format>")
    for rule_text in conv.detected_rules:
        lines.append(f"      <detected_rule>{_xml_escape(rule_text)}</detected_rule>")
    lines.append("    </pattern>")
    lines.append("  </category>")
    return "\n".join(lines)


def _hub_files_to_xml(hubs: list[HubFile], limit: int = 15) -> str:
    """Convert hub files to XML."""
    if not hubs:
        return ""
    sorted_hubs = sorted(hubs, key=lambda h: h.coupled_file_count, reverse=True)[:limit]
    lines = ['  <category name="hub-files">']
    for h in sorted_hubs:
        lines.append(
            f'    <pattern confidence="0.85" occurrences="{h.coupled_file_count}">'
        )
        lines.append(f"      <file>{_xml_escape(h.path)}</file>")
        lines.append(
            f"      <description>{_xml_escape(h.path)} is a hub file coupled with"
            f" {h.coupled_file_count} other files (total weight {h.total_coupling_weight:.1f})</description>"
        )
        lines.append("    </pattern>")
    lines.append("  </category>")
    return "\n".join(lines)


def _review_patterns_to_xml(clusters: list[CommentCluster], limit: int = 25) -> str:
    """Convert review comment clusters to XML with rich per-sample metadata."""
    if not clusters:
        return ""
    sorted_clusters = sorted(clusters, key=lambda c: len(c.comments), reverse=True)[
        :limit
    ]
    lines = ['  <category name="review-patterns">']
    for cl in sorted_clusters:
        conf = cl.coherence if cl.coherence > 0 else 0.7
        lines.append(
            f'    <pattern confidence="{conf:.2f}" occurrences="{len(cl.comments)}">'
        )
        lines.append(f"      <label>{_xml_escape(cl.label)}</label>")
        for i, cc in enumerate(cl.comments[:8]):
            c = cc.comment
            cats = ", ".join(cat.value for cat in cc.categories) if cc.categories else ""
            resolved = ""
            if c.is_resolved is not None:
                resolved = f' resolved="{str(c.is_resolved).lower()}"'
            lines.append(f'      <sample author="{_xml_escape(c.author)}" pr="{c.pr_number}"{resolved}>')
            lines.append(f"        <body>{_xml_escape(c.body[:500])}</body>")
            if cats:
                lines.append(f"        <categories>{_xml_escape(cats)}</categories>")
            if c.file_path:
                lines.append(f"        <file>{_xml_escape(c.file_path)}</file>")
            if c.diff_context:
                lines.append(f"        <diff_context>{_xml_escape(c.diff_context[:300])}</diff_context>")
            for j, reply in enumerate(c.thread_comments[:3]):
                lines.append(f"        <reply>{_xml_escape(reply[:300])}</reply>")
            lines.append("      </sample>")
        file_paths = list({c.comment.file_path for c in cl.comments if c.comment.file_path})[:10]
        if file_paths:
            lines.append(f"      <affected_files>{', '.join(_xml_escape(f) for f in file_paths)}</affected_files>")
        lines.append(
            f"      <description>{len(cl.comments)} review comments clustered around:"
            f" {_xml_escape(cl.label)}</description>"
        )
        lines.append("    </pattern>")
    lines.append("  </category>")
    return "\n".join(lines)


def _notable_comments_to_xml(comments: list[ClassifiedComment], limit: int = 15) -> str:
    """Convert high-confidence unclustered comments to XML.

    These are classified comments that didn't fall into HDBSCAN clusters
    (noise points) but have high confidence and interesting categories.
    """
    if not comments:
        return ""
    # Filter to high-confidence, non-trivial categories
    notable = [
        c for c in comments
        if c.confidence >= 0.8 and any(
            cat not in (CommentCategory.NITPICK, CommentCategory.PRAISE, CommentCategory.QUESTION)
            for cat in c.categories
        )
    ]
    if not notable:
        return ""
    notable = sorted(notable, key=lambda c: c.confidence, reverse=True)[:limit]
    lines = ['  <category name="notable-comments">']
    for cc in notable:
        c = cc.comment
        cats = ", ".join(cat.value for cat in cc.categories)
        resolved = ""
        if c.is_resolved is not None:
            resolved = f' resolved="{str(c.is_resolved).lower()}"'
        lines.append(f'    <comment confidence="{cc.confidence:.2f}" author="{_xml_escape(c.author)}" pr="{c.pr_number}"{resolved}>')
        lines.append(f"      <body>{_xml_escape(c.body[:500])}</body>")
        lines.append(f"      <categories>{_xml_escape(cats)}</categories>")
        if c.file_path:
            lines.append(f"      <file>{_xml_escape(c.file_path)}</file>")
        if c.diff_context:
            lines.append(f"      <diff_context>{_xml_escape(c.diff_context[:300])}</diff_context>")
        for reply in c.thread_comments[:2]:
            lines.append(f"      <reply>{_xml_escape(reply[:300])}</reply>")
        lines.append("    </comment>")
    lines.append("  </category>")
    return "\n".join(lines)


def analysis_to_xml(result: AnalysisResult) -> str:
    """Convert an AnalysisResult into XML for LLM consumption.

    Review data comes first (headline), then git-derived patterns (supporting context).
    """
    sections = [
        _review_patterns_to_xml(result.comment_clusters),
        _notable_comments_to_xml(result.classified_comments),
        _coupling_to_xml(result.coupling_pairs),
        _hub_files_to_xml(result.hub_files),
        _hotspots_to_xml(result.hotspots),
        _reverts_to_xml(result.revert_chains),
        _fix_after_to_xml(result.fix_after_chains),
        _conventions_to_xml(result),
    ]
    body = "\n".join(s for s in sections if s)
    return f"<patterns>\n{body}\n</patterns>"


# ── System prompt ─────────────────────────────────────────────────────────────

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
]


# ── Pre-filtering ─────────────────────────────────────────────────────────────


def _pre_filter_analysis(result: AnalysisResult) -> AnalysisResult:
    """Pre-filter analysis data before sending to LLM.

    Drops low-confidence patterns and limits per-category counts.
    """
    filtered = AnalysisResult(
        total_commits_analyzed=result.total_commits_analyzed,
        analysis_date=result.analysis_date,
        conventions=result.conventions,
    )

    # Coupling: keep pairs with confidence > 0.5
    filtered.coupling_pairs = [
        p
        for p in result.coupling_pairs
        if max(p.confidence_a_to_b, p.confidence_b_to_a) >= 0.5
    ]

    # Hotspots: keep those with score > 0 (they're already scored)
    filtered.hotspots = sorted(result.hotspots, key=lambda h: h.score, reverse=True)[
        :30
    ]

    # Reverts: all are significant
    filtered.revert_chains = result.revert_chains[:20]

    # Fix-after: all are significant
    filtered.fix_after_chains = sorted(
        result.fix_after_chains,
        key=lambda c: len(c.fixup_hashes),
        reverse=True,
    )[:20]

    # Hub files
    filtered.hub_files = sorted(
        result.hub_files,
        key=lambda h: h.coupled_file_count,
        reverse=True,
    )[:15]

    # Implicit modules
    filtered.implicit_modules = result.implicit_modules

    # Comment clusters: keep those with enough comments
    filtered.comment_clusters = [
        c for c in result.comment_clusters if len(c.comments) >= 2
    ][:25]

    # Classified comments: pass through for notable-comments extraction
    filtered.classified_comments = result.classified_comments

    return filtered


# ── Output parsing ─────────────────────────────────────────────────────────────

_VALID_CATEGORIES = {c.value for c in FindingCategory}
_VALID_SEVERITIES = {s.value for s in FindingSeverity}

_FINDINGS_RE = re.compile(r"<findings>(.*)</findings>", re.DOTALL)
_FINDING_RE = re.compile(r"<finding\b([^>]*)>(.*?)</finding>", re.DOTALL)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
_TAG_RE = re.compile(r"<(\w+)(?:\s+[^>]*)?>([^<]*)</\1>")
_POINT_RE = re.compile(r'<point\s+source="([^"]*)">(.*?)</point>', re.DOTALL)
_FILE_RE = re.compile(r"<file>([^<]+)</file>")


def _parse_findings_xml(raw: str) -> list[Finding]:
    """Parse XML findings from agent output into Finding objects."""
    # Extract <findings> block
    findings_match = _FINDINGS_RE.search(raw)
    if not findings_match:
        log.warning("No <findings> block found in agent output")
        return []

    findings_xml = findings_match.group(1)
    findings: list[Finding] = []

    for match in _FINDING_RE.finditer(findings_xml):
        attrs_str, body = match.group(1), match.group(2)
        attrs = dict(_ATTR_RE.findall(attrs_str))

        cat_str = attrs.get("category", "code_pattern")
        sev_str = attrs.get("severity", "medium")
        category = FindingCategory(cat_str) if cat_str in _VALID_CATEGORIES else FindingCategory.CODE_PATTERN
        severity = FindingSeverity(sev_str) if sev_str in _VALID_SEVERITIES else FindingSeverity.MEDIUM

        # Extract simple tags
        title = ""
        insight = ""
        for tag_match in _TAG_RE.finditer(body):
            tag_name, tag_text = tag_match.group(1), tag_match.group(2).strip()
            if tag_name == "title":
                title = tag_text
            elif tag_name == "insight":
                # insight can be multiline, re-extract with dotall
                insight_match = re.search(r"<insight>(.*?)</insight>", body, re.DOTALL)
                if insight_match:
                    insight = insight_match.group(1).strip()

        # Extract evidence points
        evidence = []
        for point_match in _POINT_RE.finditer(body):
            evidence.append(EvidencePoint(
                source=point_match.group(1),
                text=point_match.group(2).strip(),
            ))

        # Extract files
        files = _FILE_RE.findall(body)

        if title:
            findings.append(Finding(
                title=title,
                category=category,
                severity=severity,
                insight=insight,
                evidence=evidence,
                files=[f.strip() for f in files],
            ))

    return findings


# ── Agent loop ────────────────────────────────────────────────────────────────


def _configure_openrouter_env(model: str) -> None:
    """Set env vars so the Claude Agent SDK routes through OpenRouter.

    Requires OPENROUTER_API_KEY to be set (loaded from .env by python-dotenv).
    The model is set via ANTHROPIC_DEFAULT_SONNET_MODEL so the SDK's
    default "sonnet" alias resolves to the configured synthesizer model.

    See: https://openrouter.ai/docs/guides/community/anthropic-agent-sdk
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Required for agentic synthesis via OpenRouter."
        )
    # Strip litellm's "openrouter/" prefix — OpenRouter model IDs don't include it
    or_model = model.removeprefix("openrouter/")
    os.environ["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
    os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key
    os.environ["ANTHROPIC_API_KEY"] = ""
    os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] = or_model


async def _prompt_stream(text: str):
    """Yield a single user message as an async iterable.

    The Claude Agent SDK closes stdin immediately when prompt is a string,
    which breaks bidirectional MCP communication. Passing an async iterable
    instead routes through stream_input() which keeps stdin open until the
    first result arrives.
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
    }


_MAX_RETRIES = 3

# Suppress noisy SDK internal logging (it prints "Fatal error in message reader"
# to the terminal on transient failures before our retry logic kicks in)
logging.getLogger("claude_agent_sdk").setLevel(logging.CRITICAL)
logging.getLogger("claude_agent_sdk._internal").setLevel(logging.CRITICAL)


async def _run_agent(prompt: str, repo_path: str, model: str, *, _log_fn: object = None) -> str:
    """Run the agentic synthesis loop via Claude Agent SDK.

    The agent receives analysis data as its prompt, investigates patterns
    using git tools, then outputs structured guidance content. Retries up to
    _MAX_RETRIES times on transient subprocess/API failures.
    """
    from typing import Callable
    _print: Callable[[str], None] = _log_fn if callable(_log_fn) else lambda _: None

    _configure_openrouter_env(model)

    stderr_lines: list[str] = []

    def _capture_stderr(line: str) -> None:
        stderr_lines.append(line)
        log.debug("claude-cli stderr: %s", line)

    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        stderr_lines.clear()
        server = create_git_tools_server(repo_path)
        options = ClaudeAgentOptions(
            system_prompt=load_prompt("synthesis_system"),
            mcp_servers={"git": server},
            allowed_tools=_ALLOWED_TOOLS,
            permission_mode="bypassPermissions",
            max_turns=50,
            cwd=repo_path,
            stderr=_capture_stderr,
        )

        log.debug("Starting agentic synthesis attempt %d/%d model=%s repo=%s", attempt, _MAX_RETRIES, model, repo_path)

        try:
            last_text = ""
            msg_count = 0
            has_tool_use = False
            async for msg in query(prompt=_prompt_stream(prompt), options=options):
                msg_count += 1
                log.debug("Message %d: %s", msg_count, type(msg).__name__)
                if isinstance(msg, AssistantMessage):
                    has_tool_use = False
                    current_text = ""
                    tool_names = []
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            log.debug("  TextBlock (%d chars)", len(block.text))
                            current_text += block.text
                        elif isinstance(block, ToolUseBlock):
                            log.debug("  ToolUse: %s(%s)", block.name, block.input)
                            has_tool_use = True
                            tool_names.append(block.name)
                        elif isinstance(block, ToolResultBlock):
                            content_preview = str(block.content)[:200] if block.content else "(empty)"
                            log.debug("  ToolResult: %s", content_preview)
                            if block.is_error:
                                _print(f"  [{msg_count}] !! tool error: {content_preview}")
                        else:
                            log.debug("  %s", type(block).__name__)
                    # Print agent activity to console
                    if current_text:
                        preview = current_text[:200].replace("\n", " ").strip()
                        if len(current_text) > 200:
                            preview += "..."
                        _print(f"  [{msg_count}] {preview}")
                    if tool_names:
                        _print(f"  [{msg_count}] -> {', '.join(tool_names)}")
                    # Only keep text from messages that don't have tool calls
                    # (intermediate messages are narration, final message is output)
                    if current_text and not has_tool_use:
                        last_text = current_text
                    if current_text:
                        log.debug("  Text content: %s", current_text)
                elif isinstance(msg, ResultMessage):
                    log.debug("  ResultMessage: num_turns=%s is_error=%s result_len=%s",
                              msg.num_turns, msg.is_error, len(msg.result) if msg.result else 0)
                    if msg.result:
                        log.debug("  Result content: %s", msg.result)
                        last_text = msg.result

            log.debug("Agent synthesis complete: %d messages, output length=%d", msg_count, len(last_text))
            _print(f"  Done — {msg_count} messages")
            return last_text
        except Exception as e:
            last_err = e
            stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
            log.warning(
                "Synthesis attempt %d/%d failed: %s\nstderr: %s",
                attempt, _MAX_RETRIES, e, stderr_output,
            )
            if attempt < _MAX_RETRIES:
                log.info("Retrying in %ds...", 2 * attempt)
                await asyncio.sleep(2 * attempt)

    raise RuntimeError(
        f"Agentic synthesis failed after {_MAX_RETRIES} attempts: {last_err}"
        f"\nLast stderr:\n{chr(10).join(stderr_lines) if stderr_lines else '(none)'}"
    )


# ── Main synthesizer ──────────────────────────────────────────────────────────


def synthesize(
    analysis: AnalysisResult,
    config: GitloreConfig,
    *,
    _log_fn: object = None,
) -> SynthesisResult:
    """Run the full synthesis pipeline: filter -> XML -> agent -> parse findings."""
    filtered = _pre_filter_analysis(analysis)
    xml_input = analysis_to_xml(filtered)

    has_review_data = bool(analysis.comment_clusters or analysis.classified_comments)
    review_note = (
        "PR review data is included. Use review clusters and git patterns together."
        if has_review_data
        else "No PR review data available (git-only mode). Focus on git history patterns."
    )

    user_prompt = (
        f"Analyze this codebase ({analysis.total_commits_analyzed} commits analyzed)."
        f" {review_note}\n\n{xml_input}"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
        raw_output = asyncio.run(
            _run_agent(user_prompt, config.repo_path, config.models.synthesizer, _log_fn=_log_fn)
        )
    findings = _parse_findings_xml(raw_output)

    return SynthesisResult(
        findings=findings,
        raw_xml=raw_output.strip(),
        analysis=analysis,
        has_review_data=has_review_data,
        model_used=config.models.synthesizer,
    )
