"""Build the local knowledge index for gitlore."""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

from gitlore.cache import Cache
from gitlore.config import GitloreConfig
from gitlore.docs import DocSnippet, extract_doc_snippets
from gitlore.index import IndexStore
from gitlore.models import (
    AnalysisResult,
    BuildMetadata,
    ChurnHotspot,
    ClassifiedComment,
    CommentCategory,
    CommentCluster,
    ContextBundle,
    CouplingPair,
    EvidenceRef,
    FactKind,
    FactStability,
    FixAfterChain,
    HubFile,
    KnowledgeFact,
    KnowledgeSeverity,
    QueryIntent,
    RevertChain,
    ReviewComment,
    SourceCoverage,
)


def build_index(
    config: GitloreConfig,
    *,
    use_cache: bool = True,
    console: object | None = None,
) -> BuildMetadata:
    """Build a fresh `.gitlore/index.db` for the configured repository."""
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

    _log("Extracting git history...")
    commits = list(
        iter_commits(
            config.repo_path,
            since_months=config.build.since_months,
            no_merges=True,
        )
    )
    _log(f"Analyzing {len(commits)} commits...")

    classified = classify_commits(commits)
    conventions = analyze_conventions(classified)
    hotspots = analyze_churn(classified, config.build)
    revert_chains = detect_reverts(commits)
    fix_after_chains = detect_fix_after(commits)
    coupling_pairs, modules, hubs = analyze_coupling(commits, config.build)

    analysis = AnalysisResult(
        hotspots=hotspots,
        revert_chains=revert_chains,
        fix_after_chains=fix_after_chains,
        coupling_pairs=coupling_pairs,
        implicit_modules=modules,
        hub_files=hubs,
        conventions=conventions,
        total_commits_analyzed=len(commits),
        analysis_date=datetime.now(UTC),
    )

    coverage = SourceCoverage(git=True)
    cache = Cache(config.repo_path) if use_cache else None

    if config.sources.github:
        _log("Collecting GitHub review data...")
        _run_review_enrichment(config, analysis, coverage, cache=cache)

    doc_snippets: list[DocSnippet] = []
    if config.sources.docs:
        _log("Indexing repo docs and config...")
        doc_snippets = extract_doc_snippets(config.repo_path)
        coverage.docs = bool(doc_snippets)

    facts, relationships = _build_facts(analysis, doc_snippets)
    embeddings = _build_embeddings(config, facts, coverage, cache=cache)

    metadata = BuildMetadata(
        repo_path=config.repo_path,
        built_at=datetime.now(UTC),
        total_commits_analyzed=len(commits),
        fact_count=len(facts),
        source_coverage=coverage,
    )

    store = IndexStore(config.repo_path)
    try:
        store.store_build(facts, relationships, metadata, embeddings)
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

    comments = None
    if cache is not None:
        comments = cache.get_comments(owner, repo)

    if comments is None:
        comments = asyncio.run(fetch_review_comments(token, owner, repo))
        if cache is not None and comments:
            cache.set_comments(owner, repo, comments)

    if not comments:
        return

    coverage.github = True
    analysis.classified_comments = []
    analysis.comment_clusters = []

    if not config.models.classifier:
        return

    try:
        classified = asyncio.run(
            classify_comments(
                comments,
                config.models.classifier,
                cache=cache,
            )
        )
        analysis.classified_comments = classified
        coverage.classified_reviews = True
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
        if analysis.comment_clusters:
            coverage.semantic = True
    except Exception:
        analysis.comment_clusters = []


def _build_facts(
    analysis: AnalysisResult,
    doc_snippets: list[DocSnippet],
) -> tuple[list[KnowledgeFact], list[tuple[str, str, str, float]]]:
    facts: list[KnowledgeFact] = []
    relationships: list[tuple[str, str, str, float]] = []

    if analysis.conventions is not None:
        for rule_text in analysis.conventions.detected_rules:
            facts.append(
                _make_fact(
                    kind=FactKind.RULE,
                    stability=FactStability.STABLE,
                    title=rule_text,
                    guidance=rule_text,
                    support_count=max(1, int(analysis.conventions.format_adherence * 100)),
                    confidence=analysis.conventions.format_adherence,
                    severity=KnowledgeSeverity.NONE,
                    intents=[
                        QueryIntent.FEATURE,
                        QueryIntent.REFACTOR,
                        QueryIntent.REVIEW,
                        QueryIntent.GENERAL,
                    ],
                    evidence=[
                        EvidenceRef(
                            source_type="config",
                            label="commit conventions",
                            ref="commit-conventions",
                            excerpt=rule_text,
                        )
                    ],
                )
            )

    for pair in analysis.coupling_pairs:
        confidence = max(pair.confidence_a_to_b, pair.confidence_b_to_a)
        facts.append(_fact_from_coupling(pair, confidence))
        relationships.extend(
            [
                (
                    pair.file_a,
                    pair.file_b,
                    f"co-change ({confidence:.0%} of relevant commits)",
                    pair.strength,
                ),
                (
                    pair.file_b,
                    pair.file_a,
                    f"co-change ({confidence:.0%} of relevant commits)",
                    pair.strength,
                ),
            ]
        )
        test_fact = _test_association_fact(pair)
        if test_fact is not None:
            facts.append(test_fact)

    for hotspot in analysis.hotspots:
        facts.append(_fact_from_hotspot(hotspot))

    for hub in analysis.hub_files:
        facts.append(_fact_from_hub(hub))

    for chain in analysis.revert_chains:
        facts.append(_fact_from_revert(chain))

    for chain in analysis.fix_after_chains:
        facts.append(_fact_from_fix_after(chain))

    if analysis.comment_clusters:
        for cluster in analysis.comment_clusters:
            facts.append(_fact_from_cluster(cluster))
    elif analysis.classified_comments:
        for comment in analysis.classified_comments[:50]:
            facts.append(_fact_from_classified_comment(comment))

    for snippet in doc_snippets:
        facts.append(_fact_from_doc(snippet))

    deduped: dict[str, KnowledgeFact] = {}
    for fact in facts:
        deduped[fact.id] = fact
    ordered_facts = sorted(deduped.values(), key=lambda item: (item.kind.value, item.id))
    return ordered_facts, relationships


def _build_embeddings(
    config: GitloreConfig,
    facts: list[KnowledgeFact],
    coverage: SourceCoverage,
    *,
    cache: Cache | None,
) -> dict[str, list[float]]:
    if not config.query.semantic or not config.models.embedding:
        return {}
    if not os.environ.get("OPENROUTER_API_KEY", ""):
        return {}

    semantic_kinds = {
        FactKind.RULE,
        FactKind.DOC_GUIDANCE,
        FactKind.REVIEW_THEME,
        FactKind.HISTORICAL_EXAMPLE,
    }
    candidates = [fact for fact in facts if fact.kind in semantic_kinds]
    if not candidates:
        return {}

    texts = [fact.search_text or f"{fact.title}\n{fact.guidance}" for fact in candidates]
    cached_vectors = cache.get_embeddings(config.models.embedding, texts) if cache else None
    if cached_vectors is None:
        from gitlore.utils.llm import embed

        vectors = asyncio.run(embed(config.models.embedding, texts))
        if cache is not None:
            cache.set_embeddings(config.models.embedding, texts, vectors)
    else:
        vectors = cached_vectors

    coverage.semantic = True
    return {fact.id: vector for fact, vector in zip(candidates, vectors)}


def _make_fact(
    *,
    kind: FactKind,
    stability: FactStability,
    title: str,
    guidance: str,
    files: list[str] | None = None,
    subsystems: list[str] | None = None,
    support_count: int = 0,
    confidence: float = 0.0,
    severity: KnowledgeSeverity = KnowledgeSeverity.NONE,
    last_seen_at: datetime | None = None,
    intents: list[QueryIntent] | None = None,
    evidence: list[EvidenceRef] | None = None,
    extra_search: str = "",
) -> KnowledgeFact:
    files = files or []
    subsystems = subsystems or _subsystems(files)
    intents = intents or [QueryIntent.GENERAL]
    evidence = evidence or []
    search_text = " ".join([title, guidance, " ".join(files), " ".join(subsystems), extra_search]).strip()
    fact_id = hashlib.sha1(
        "\0".join([kind.value, title, guidance, ",".join(files)]).encode()
    ).hexdigest()[:16]
    return KnowledgeFact(
        id=fact_id,
        kind=kind,
        stability=stability,
        title=title,
        guidance=guidance,
        files=files,
        subsystems=subsystems,
        applicable_intents=intents,
        support_count=support_count,
        confidence=confidence,
        severity=severity,
        last_seen_at=last_seen_at,
        evidence=evidence,
        search_text=search_text,
    )


def _fact_from_coupling(pair: CouplingPair, confidence: float) -> KnowledgeFact:
    return _make_fact(
        kind=FactKind.FILE_RELATIONSHIP,
        stability=FactStability.SITUATIONAL,
        title=f"Editing {pair.file_a} usually implicates {pair.file_b}",
        guidance=(
            f"Changes to `{pair.file_a}` and `{pair.file_b}` frequently land together. "
            f"Inspect both when planning edits."
        ),
        files=[pair.file_a, pair.file_b],
        support_count=int(pair.shared_commits),
        confidence=confidence,
        severity=KnowledgeSeverity.NONE,
        intents=[
            QueryIntent.BUGFIX,
            QueryIntent.FEATURE,
            QueryIntent.REFACTOR,
            QueryIntent.REVIEW,
        ],
        evidence=[
            EvidenceRef(
                source_type="coupling",
                label="co-change pair",
                ref=f"{pair.file_a}::{pair.file_b}",
                excerpt=(
                    f"{pair.file_a} and {pair.file_b} change together in "
                    f"{confidence:.0%} of commits touching either file."
                ),
            )
        ],
        extra_search=f"co-change related files {pair.shared_commits:.0f}",
    )


def _test_association_fact(pair: CouplingPair) -> KnowledgeFact | None:
    files = [pair.file_a, pair.file_b]
    source_files = [path for path in files if not _is_test_path(path)]
    test_files = [path for path in files if _is_test_path(path)]
    if len(source_files) != 1 or len(test_files) != 1:
        return None
    source_file = source_files[0]
    test_file = test_files[0]
    confidence = max(pair.confidence_a_to_b, pair.confidence_b_to_a)
    return _make_fact(
        kind=FactKind.TEST_ASSOCIATION,
        stability=FactStability.SITUATIONAL,
        title=f"Changes to {source_file} often require {test_file}",
        guidance=f"When editing `{source_file}`, inspect and update `{test_file}`.",
        files=[source_file, test_file],
        support_count=int(pair.shared_commits),
        confidence=confidence,
        severity=KnowledgeSeverity.NONE,
        intents=[
            QueryIntent.BUGFIX,
            QueryIntent.FEATURE,
            QueryIntent.REFACTOR,
            QueryIntent.REVIEW,
        ],
        evidence=[
            EvidenceRef(
                source_type="coupling",
                label="test association",
                ref=f"{source_file}::{test_file}",
                excerpt=f"These files co-change in {confidence:.0%} of relevant commits.",
            )
        ],
    )


def _fact_from_hotspot(hotspot: ChurnHotspot) -> KnowledgeFact:
    severity = KnowledgeSeverity.HIGH if hotspot.fix_ratio >= 0.3 else KnowledgeSeverity.MEDIUM
    return _make_fact(
        kind=FactKind.FRAGILE_AREA,
        stability=FactStability.SITUATIONAL,
        title=f"{hotspot.path} is a high-churn area",
        guidance=(
            f"`{hotspot.path}` changes frequently and has a {hotspot.fix_ratio:.0%} fix ratio. "
            "Budget extra validation and test coverage."
        ),
        files=[hotspot.path],
        support_count=hotspot.commit_count,
        confidence=min(hotspot.score / 10.0, 1.0),
        severity=severity,
        intents=[
            QueryIntent.BUGFIX,
            QueryIntent.FEATURE,
            QueryIntent.REFACTOR,
            QueryIntent.REVIEW,
        ],
        evidence=[
            EvidenceRef(
                source_type="hotspot",
                label="churn hotspot",
                ref=hotspot.path,
                excerpt=(
                    f"{hotspot.commit_count} commits, {hotspot.lines_added + hotspot.lines_deleted} "
                    f"lines churned, {hotspot.fix_ratio:.0%} fixes."
                ),
            )
        ],
    )


def _fact_from_hub(hub: HubFile) -> KnowledgeFact:
    return _make_fact(
        kind=FactKind.FRAGILE_AREA,
        stability=FactStability.SITUATIONAL,
        title=f"{hub.path} fans out across the repo",
        guidance=(
            f"`{hub.path}` couples with {hub.coupled_file_count} other files. "
            "Treat it as a coordination point when changing surrounding code."
        ),
        files=[hub.path],
        support_count=hub.coupled_file_count,
        confidence=min(hub.total_coupling_weight / 10.0, 1.0),
        severity=KnowledgeSeverity.MEDIUM,
        intents=[
            QueryIntent.BUGFIX,
            QueryIntent.FEATURE,
            QueryIntent.REFACTOR,
            QueryIntent.REVIEW,
        ],
        evidence=[
            EvidenceRef(
                source_type="coupling",
                label="hub file",
                ref=hub.path,
                excerpt=f"Coupled with {hub.coupled_file_count} files.",
            )
        ],
    )


def _fact_from_revert(chain: RevertChain) -> KnowledgeFact:
    title = f"Past change to {', '.join(chain.files[:2]) or chain.original_subject} was reverted"
    guidance = (
        f"`{chain.original_subject}` was reverted {chain.depth} time(s). "
        "Review prior failure modes before repeating the approach."
    )
    return _make_fact(
        kind=FactKind.HISTORICAL_EXAMPLE,
        stability=FactStability.EXAMPLE,
        title=title,
        guidance=guidance,
        files=chain.files,
        support_count=chain.depth,
        confidence=0.9,
        severity=KnowledgeSeverity.HIGH,
        last_seen_at=chain.original_date,
        intents=[QueryIntent.BUGFIX, QueryIntent.REVIEW],
        evidence=[
            EvidenceRef(
                source_type="revert",
                label="revert chain",
                ref=chain.original_hash,
                excerpt=chain.original_subject,
            )
        ],
    )


def _fact_from_fix_after(chain: FixAfterChain) -> KnowledgeFact:
    title = f"{chain.original_subject} needed follow-up fixes"
    guidance = (
        f"`{chain.original_subject}` needed {len(chain.fixup_hashes)} follow-up fix(es) within "
        f"{chain.time_span}. Check the same edge cases before editing this area."
    )
    return _make_fact(
        kind=FactKind.HISTORICAL_EXAMPLE,
        stability=FactStability.EXAMPLE,
        title=title,
        guidance=guidance,
        files=chain.files,
        support_count=len(chain.fixup_hashes),
        confidence=0.8,
        severity=KnowledgeSeverity.HIGH if chain.tier.value == "immediate" else KnowledgeSeverity.MEDIUM,
        last_seen_at=chain.original_date,
        intents=[QueryIntent.BUGFIX, QueryIntent.REVIEW],
        evidence=[
            EvidenceRef(
                source_type="fix_after",
                label="follow-up chain",
                ref=chain.original_hash,
                excerpt="; ".join(chain.fixup_subjects[:3]) or chain.original_subject,
            )
        ],
    )


def _fact_from_cluster(cluster: CommentCluster) -> KnowledgeFact:
    files = sorted(
        {
            comment.comment.file_path
            for comment in cluster.comments
            if comment.comment.file_path
        }
    )
    severity = _severity_from_categories(
        [
            category
            for classified in cluster.comments
            for category in classified.categories
        ]
    )
    evidence = [
        EvidenceRef(
            source_type="review_cluster",
            label=f"PR #{classified.comment.pr_number}",
            ref=str(classified.comment.pr_number),
            excerpt=classified.comment.body[:180],
        )
        for classified in cluster.comments[:5]
    ]
    return _make_fact(
        kind=FactKind.REVIEW_THEME,
        stability=FactStability.SITUATIONAL,
        title=cluster.label,
        guidance=f"Reviewers repeatedly flag this theme: {cluster.label}.",
        files=files,
        support_count=len(cluster.comments),
        confidence=cluster.coherence or 0.7,
        severity=severity,
        intents=[
            QueryIntent.BUGFIX,
            QueryIntent.REFACTOR,
            QueryIntent.FEATURE,
            QueryIntent.REVIEW,
        ],
        evidence=evidence,
        extra_search=" ".join(
            comment.comment.body for comment in cluster.comments[:5]
        ),
    )


def _fact_from_classified_comment(comment: ClassifiedComment) -> KnowledgeFact:
    file_path = comment.comment.file_path or ""
    severity = _severity_from_categories(comment.categories)
    title = f"Review feedback near {file_path or 'general code'}"
    return _make_fact(
        kind=FactKind.REVIEW_THEME,
        stability=FactStability.SITUATIONAL,
        title=title,
        guidance=comment.comment.body,
        files=[file_path] if file_path else [],
        support_count=1,
        confidence=comment.confidence,
        severity=severity,
        intents=[
            QueryIntent.BUGFIX,
            QueryIntent.REFACTOR,
            QueryIntent.FEATURE,
            QueryIntent.REVIEW,
        ],
        evidence=[
            EvidenceRef(
                source_type="review_comment",
                label=f"PR #{comment.comment.pr_number}",
                ref=str(comment.comment.pr_number),
                excerpt=comment.comment.body[:180],
            )
        ],
        extra_search=" ".join(category.value for category in comment.categories),
    )


def _fact_from_doc(snippet: DocSnippet) -> KnowledgeFact:
    return _make_fact(
        kind=FactKind.DOC_GUIDANCE,
        stability=FactStability.STABLE,
        title=f"{snippet.title} ({snippet.path})",
        guidance=snippet.content,
        files=[snippet.path],
        support_count=1,
        confidence=0.7,
        severity=KnowledgeSeverity.NONE,
        intents=[
            QueryIntent.FEATURE,
            QueryIntent.REFACTOR,
            QueryIntent.REVIEW,
            QueryIntent.GENERAL,
        ],
        evidence=[
            EvidenceRef(
                source_type=snippet.source_type,
                label=snippet.path,
                ref=snippet.path,
                excerpt=snippet.content[:180],
            )
        ],
        extra_search=snippet.path,
    )


def _subsystems(files: list[str]) -> list[str]:
    subsystems: set[str] = set()
    for path in files:
        parts = Path(path).parts
        if len(parts) >= 2:
            subsystems.add("/".join(parts[:2]))
        elif parts:
            subsystems.add(parts[0])
    return sorted(subsystems)


def _is_test_path(path: str) -> bool:
    lower = path.lower()
    return lower.startswith("tests/") or "/test" in lower or lower.endswith("_test.py")


def _severity_from_categories(categories: list[CommentCategory]) -> KnowledgeSeverity:
    values = {category.value for category in categories}
    if "security" in values or "bug" in values:
        return KnowledgeSeverity.HIGH
    if "architecture" in values or "performance" in values:
        return KnowledgeSeverity.MEDIUM
    if values:
        return KnowledgeSeverity.LOW
    return KnowledgeSeverity.NONE
