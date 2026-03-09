"""Build the local knowledge index for gitlore."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

from gitlore.cache import Cache
from gitlore.config import GitloreConfig
from gitlore.docs import DocSnippet, extract_doc_snippets
from gitlore.index import IndexStore
from gitlore.models import (
    AnalysisResult,
    BuildMetadata,
    CouplingPair,
    FileEdge,
    SourceCoverage,
)
from gitlore.synthesis.synthesizer import run_investigation


def build_index(
    config: GitloreConfig,
    *,
    use_cache: bool = True,
    console: object | None = None,
) -> BuildMetadata:
    """Build a fresh `.gitlore/index.db` knowledge index for the configured repository."""
    from gitlore.analyzers.churn import analyze_churn
    from gitlore.analyzers.commit_classifier import classify_commits
    from gitlore.analyzers.conventions import analyze_conventions
    from gitlore.analyzers.coupling import analyze_coupling
    from gitlore.analyzers.fix_after import detect_fix_after
    from gitlore.analyzers.reverts import detect_reverts
    from gitlore.extractors.git_log import iter_commits

    def _log(message: str) -> None:
        if console is not None and hasattr(console, "print"):
            console.print(f"[dim]{message}[/dim]")

    # 1. Extract git history
    _log("Extracting git history...")
    commits = list(
        iter_commits(
            config.repo_path,
            since_months=config.build.since_months,
            no_merges=True,
        )
    )
    _log(f"Analyzing {len(commits)} commits...")

    # 2. Classify and run deterministic analyzers
    classified = classify_commits(commits)
    analysis = AnalysisResult(
        hotspots=analyze_churn(classified, config.build),
        revert_chains=detect_reverts(commits),
        fix_after_chains=detect_fix_after(commits),
        total_commits_analyzed=len(commits),
        analysis_date=datetime.now(UTC),
    )
    coupling_pairs, modules, hubs = analyze_coupling(commits, config.build)
    analysis.coupling_pairs = coupling_pairs
    analysis.implicit_modules = modules
    analysis.hub_files = hubs
    analysis.conventions = analyze_conventions(classified)

    coverage = SourceCoverage(git=True)
    cache = Cache(config.repo_path) if use_cache else None

    # 3. Optional GitHub review enrichment
    if config.sources.github:
        _log("Collecting GitHub review data...")
        _run_review_enrichment(config, analysis, coverage, cache=cache)

    # 4. Optional doc/config snippets
    doc_snippets: list[DocSnippet] = []
    if config.sources.docs:
        _log("Indexing repo docs and config...")
        doc_snippets = extract_doc_snippets(config.repo_path)
        coverage.docs = bool(doc_snippets)

    # 5. Build file edges from coupling
    file_edges = _build_file_edges(analysis.coupling_pairs)

    # 6. Run agent investigation
    _log("Running repository investigation...")
    notes = run_investigation(
        analysis, doc_snippets, config, config.repo_path, _log_fn=_log,
    )
    _log(f"Investigation complete: {len(notes)} notes recorded")

    # 7. Store index
    metadata = BuildMetadata(
        repo_path=config.repo_path,
        built_at=datetime.now(UTC),
        total_commits_analyzed=len(commits),
        note_count=len(notes),
        source_coverage=coverage,
    )

    store = IndexStore(config.repo_path)
    try:
        store.store_build(notes, file_edges, metadata)
    finally:
        store.close()
        if cache is not None:
            cache.close()

    return metadata


def _run_review_enrichment(
    config: GitloreConfig,
    analysis: AnalysisResult,
    coverage: SourceCoverage,
    *,
    cache: Cache | None,
) -> None:
    from gitlore.classifiers.comment_classifier import classify_comments
    from gitlore.clustering.semantic import cluster_comments
    from gitlore.extractors.github_comments import fetch_review_comments

    owner, repo = config.github.resolve_owner_repo(config.repo_path)
    token = config.github.resolve_token()
    if not owner or not repo or not token:
        return

    comments = cache.get_comments(owner, repo) if cache is not None else None
    if comments is None:
        comments = asyncio.run(fetch_review_comments(token, owner, repo))
        if cache is not None and comments:
            cache.set_comments(owner, repo, comments)
    if not comments:
        return

    coverage.github = True
    if not config.models.classifier:
        return

    try:
        analysis.classified_comments = asyncio.run(
            classify_comments(comments, config.models.classifier, cache=cache)
        )
        coverage.classified_reviews = bool(analysis.classified_comments)
    except Exception:
        analysis.classified_comments = []
        return

    if not config.query.semantic or not config.models.embedding or not os.environ.get(
        "OPENROUTER_API_KEY", ""
    ):
        return

    try:
        analysis.comment_clusters = cluster_comments(
            analysis.classified_comments,
            config.models,
            cache=cache,
        )
        coverage.semantic = bool(analysis.comment_clusters)
    except Exception:
        analysis.comment_clusters = []


def _build_file_edges(pairs: list[CouplingPair]) -> list[FileEdge]:
    """Convert strong coupling pairs into bidirectional file edges for retrieval."""
    edges: list[FileEdge] = []
    for pair in pairs:
        confidence = max(pair.confidence_a_to_b, pair.confidence_b_to_a)
        reason = f"co-change ({confidence:.0%} of relevant commits)"
        edges.append(FileEdge(
            src=pair.file_a, dst=pair.file_b,
            edge_type="cochange", score=pair.strength, reason=reason,
        ))
        edges.append(FileEdge(
            src=pair.file_b, dst=pair.file_a,
            edge_type="cochange", score=pair.strength, reason=reason,
        ))
    return edges
