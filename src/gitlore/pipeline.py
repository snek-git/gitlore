"""Pipeline orchestrator — wires extraction, analysis, synthesis, and formatting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from gitlore.config import GitloreConfig
from gitlore.models import AnalysisResult, SynthesisResult


def run_pipeline(config: GitloreConfig, *, git_only: bool = False, use_cache: bool = True) -> SynthesisResult:
    """Run the full gitlore analysis pipeline."""
    from gitlore.analyzers.churn import analyze_churn
    from gitlore.analyzers.commit_classifier import classify_commits
    from gitlore.analyzers.conventions import analyze_conventions
    from gitlore.analyzers.coupling import analyze_coupling
    from gitlore.analyzers.fix_after import detect_fix_after
    from gitlore.analyzers.reverts import detect_reverts
    from gitlore.extractors.git_log import iter_commits
    from gitlore.synthesis.synthesizer import synthesize

    # ── Branch A: Git extraction + analysis ─────────────────────────────
    commits = list(
        iter_commits(
            config.repo_path,
            since_months=config.analysis.since_months,
            no_merges=True,
        )
    )

    classified = classify_commits(commits)
    conventions = analyze_conventions(classified)
    hotspots = analyze_churn(classified, config.analysis)
    revert_chains = detect_reverts(commits)
    fix_chains = detect_fix_after(commits)
    coupling_pairs, modules, hubs = analyze_coupling(commits, config.analysis)

    analysis = AnalysisResult(
        hotspots=hotspots,
        revert_chains=revert_chains,
        fix_after_chains=fix_chains,
        coupling_pairs=coupling_pairs,
        implicit_modules=modules,
        hub_files=hubs,
        conventions=conventions,
        total_commits_analyzed=len(commits),
        analysis_date=datetime.now(timezone.utc),
    )

    # ── Branch B: GitHub comments + LLM (optional) ──────────────────────
    if not git_only and config.github.owner and config.github.repo:
        _run_comment_pipeline(config, analysis, use_cache=use_cache)

    # ── Synthesis ───────────────────────────────────────────────────────
    return synthesize(analysis, config)


def _run_comment_pipeline(config: GitloreConfig, analysis: AnalysisResult, *, use_cache: bool = True) -> None:
    """Fetch, classify, and cluster PR review comments."""
    import asyncio

    from gitlore.cache import Cache
    from gitlore.classifiers.comment_classifier import classify_comments
    from gitlore.clustering.semantic import cluster_comments
    from gitlore.extractors.github_comments import fetch_review_comments

    cache = Cache(config.repo_path) if use_cache else None

    token = config.github.resolve_token()
    if not token:
        return

    # Check cache for comments
    comments = None
    if cache is not None:
        comments = cache.get_comments(config.github.owner, config.github.repo)

    if comments is None:
        comments = asyncio.run(
            fetch_review_comments(token, config.github.owner, config.github.repo)
        )
        # Always write to cache even if use_cache was True (it already is if we're here)
        if cache is not None and comments:
            cache.set_comments(config.github.owner, config.github.repo, comments)

    if not comments:
        return

    classified = asyncio.run(classify_comments(comments, config.models.classifier, cache=cache))
    analysis.classified_comments = classified
    clusters = cluster_comments(classified, config.models, cache=cache)
    analysis.comment_clusters = clusters


def write_outputs(result: SynthesisResult, config: GitloreConfig) -> list[str]:
    """Write formatted output files. Returns list of written paths."""
    from gitlore.formatters.agents_md import format_agents_md
    from gitlore.formatters.claude_md import format_claude_md
    from gitlore.formatters.copilot_instructions import format_copilot_instructions
    from gitlore.formatters.cursor_rules import format_cursor_rules
    from gitlore.formatters.html import format_html
    from gitlore.formatters.report import format_report

    formatter_map = {
        "report": (format_report, "gitlore-report.md"),
        "html": (format_html, "gitlore-report.html"),
        "claude_md": (format_claude_md, "CLAUDE.md"),
        "agents_md": (format_agents_md, "AGENTS.md"),
        "cursor_rules": (format_cursor_rules, ".cursor/rules/gitlore.mdc"),
        "copilot_instructions": (
            format_copilot_instructions,
            ".github/copilot-instructions.md",
        ),
    }

    written = []
    repo = Path(config.repo_path)

    for fmt in config.output.formats:
        if fmt not in formatter_map:
            continue
        formatter, rel_path = formatter_map[fmt]
        content = formatter(result)
        out_path = repo / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        written.append(str(out_path))

    return written
