"""Build the local knowledge index for gitlore."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from gitlore.cache import Cache
from gitlore.config import GitloreConfig
from gitlore.docs import DocSnippet, extract_doc_snippets
from gitlore.index import IndexStore
from gitlore.models import (
    AnalysisResult,
    BuildMetadata,
    CouplingPair,
    FileEdge,
    KnowledgeNote,
    SourceCoverage,
)
from gitlore.synthesis.synthesizer import run_investigation


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    )


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

    con = console if isinstance(console, Console) else Console()

    def _log(message: str) -> None:
        con.print(f"[dim]{message}[/dim]")

    # 1. Extract git history
    with con.status("Extracting git history..."):
        commits = list(
            iter_commits(
                config.repo_path,
                since_months=config.build.since_months,
                no_merges=True,
            )
        )
    con.print(f"  Extracted [bold]{len(commits)}[/bold] commits")

    # 2. Classify and run deterministic analyzers
    with con.status("Running analyzers..."):
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

    con.print(
        f"  Found [bold]{len(analysis.hotspots)}[/bold] hotspots, "
        f"[bold]{len(analysis.coupling_pairs)}[/bold] coupling pairs, "
        f"[bold]{len(analysis.fix_after_chains)}[/bold] fix-after chains, "
        f"[bold]{len(analysis.revert_chains)}[/bold] reverts"
    )

    coverage = SourceCoverage(git=True)
    cache = Cache(config.repo_path) if use_cache else None

    # 3. Optional GitHub review enrichment
    if config.sources.github:
        _run_review_enrichment(config, analysis, coverage, cache=cache, console=con)

    # 4. Optional doc/config snippets
    doc_snippets: list[DocSnippet] = []
    if config.sources.docs:
        with con.status("Indexing repo docs and config..."):
            doc_snippets = extract_doc_snippets(config.repo_path)
            coverage.docs = bool(doc_snippets)
        con.print(f"  Indexed [bold]{len(doc_snippets)}[/bold] doc snippets")

    # 5. Build file edges from coupling
    file_edges = _build_file_edges(analysis.coupling_pairs)

    # 6. Run agent investigation
    _log("Running repository investigation...")
    notes = run_investigation(
        analysis, doc_snippets, config, config.repo_path, _log_fn=_log,
    )
    con.print(f"  Investigation complete: [bold]{len(notes)}[/bold] notes recorded")

    # 7. Embed notes for semantic retrieval
    if notes and config.models.embedding and os.environ.get("OPENROUTER_API_KEY", ""):
        with con.status(f"Embedding {len(notes)} notes..."):
            _embed_notes(notes, config.models.embedding)
        con.print(f"  Embedded [bold]{len(notes)}[/bold] notes")

    # 8. Store index
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
    console: Console,
) -> None:
    from gitlore.classifiers.comment_classifier import classify_comments
    from gitlore.clustering.semantic import cluster_comments
    from gitlore.extractors.github_comments import fetch_review_comments

    owner, repo = config.github.resolve_owner_repo(config.repo_path)
    token = config.github.resolve_token()
    if not owner or not repo or not token:
        console.print("[dim]  Skipping GitHub reviews (no owner/repo/token)[/dim]")
        return

    # Fetch or load from cache
    comments = cache.get_comments(owner, repo) if cache is not None else None
    if comments is None:
        progress = _make_progress()
        with progress:
            task = progress.add_task(f"Fetching PR reviews from {owner}/{repo}", total=None)

            def _on_fetch(comments_so_far: int, prs_fetched: int) -> None:
                progress.update(task, description=f"Fetching reviews ({comments_so_far} comments, {prs_fetched} PRs)")

            comments = asyncio.run(fetch_review_comments(token, owner, repo, on_progress=_on_fetch))
        if cache is not None and comments:
            cache.set_comments(owner, repo, comments)
    else:
        console.print(f"[dim]  Loaded {len(comments)} cached review comments[/dim]")

    if not comments:
        console.print("[dim]  No review comments found[/dim]")
        return

    console.print(f"  Collected [bold]{len(comments)}[/bold] review comments")
    coverage.github = True

    # Classify
    if not config.models.classifier:
        return

    progress = _make_progress()
    with progress:
        task = progress.add_task(f"Classifying {len(comments)} comments", total=len(comments))

        def _on_classify(done: int, total: int) -> None:
            progress.update(task, completed=done)

        try:
            analysis.classified_comments = asyncio.run(
                classify_comments(
                    comments, config.models.classifier, cache=cache, on_progress=_on_classify,
                )
            )
            coverage.classified_reviews = bool(analysis.classified_comments)
        except Exception:
            analysis.classified_comments = []
            console.print("[yellow]  Classification failed, continuing without[/yellow]")
            return

    console.print(f"  Classified [bold]{len(analysis.classified_comments)}[/bold] comments")

    # Cluster
    if not config.query.semantic or not config.models.embedding or not os.environ.get(
        "OPENROUTER_API_KEY", ""
    ):
        return

    progress = _make_progress()
    with progress:
        embed_task = progress.add_task("Embedding comments", total=len(analysis.classified_comments))
        cluster_task: int | None = None
        summarize_task: int | None = None

        def _on_cluster_progress(stage: str, done: int, total: int) -> None:
            nonlocal cluster_task, summarize_task
            if stage == "embed":
                progress.update(embed_task, completed=done)
            elif stage == "cluster":
                if cluster_task is None:
                    progress.update(embed_task, visible=False)
                    cluster_task = progress.add_task("Running HDBSCAN", total=1)
                progress.update(cluster_task, completed=done)
            elif stage == "summarize":
                if summarize_task is None:
                    if cluster_task is not None:
                        progress.update(cluster_task, visible=False)
                    summarize_task = progress.add_task("Summarizing themes", total=total)
                progress.update(summarize_task, completed=done)

        try:
            analysis.comment_clusters = cluster_comments(
                analysis.classified_comments,
                config.models,
                cache=cache,
                on_progress=_on_cluster_progress,
            )
            coverage.semantic = bool(analysis.comment_clusters)
        except Exception as exc:
            analysis.comment_clusters = []
            console.print(f"[yellow]  Clustering failed ({exc}), continuing without[/yellow]")
            return

    console.print(f"  Found [bold]{len(analysis.comment_clusters)}[/bold] review themes")


def _embed_notes(notes: list[KnowledgeNote], embedding_model: str) -> None:
    """Embed note text + anchors for semantic retrieval."""
    from gitlore.utils.llm import embed

    texts = [f"{n.text} {' '.join(n.anchors)}" for n in notes]
    embeddings = asyncio.run(embed(embedding_model, texts))
    for note, emb in zip(notes, embeddings):
        note.embedding = emb


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
