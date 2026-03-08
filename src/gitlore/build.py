"""Build the local advice-card index for gitlore."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import defaultdict
from datetime import UTC, datetime

from gitlore.cache import Cache
from gitlore.config import GitloreConfig
from gitlore.docs import DocSnippet, extract_doc_snippets
from gitlore.index import IndexStore
from gitlore.models import (
    AdviceCard,
    AdvicePriority,
    AnalysisResult,
    BuildMetadata,
    ClassifiedComment,
    CommentCategory,
    CouplingPair,
    EvidenceRef,
    FileEdge,
    FixAfterChain,
    HubFile,
    InvestigationLead,
    LeadKind,
    QueryIntent,
    RevertChain,
    SourceCoverage,
)
from gitlore.synthesis.synthesizer import investigate_leads

MAX_HOTSPOT_LEADS = 20
MAX_COUPLING_LEADS = 20
MAX_REVIEW_LEADS = 15
MAX_HISTORY_LEADS = 10
MAX_GUIDANCE_LEADS = 10


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

    if config.sources.github:
        _log("Collecting GitHub review data...")
        _run_review_enrichment(config, analysis, coverage, cache=cache)

    doc_snippets: list[DocSnippet] = []
    if config.sources.docs:
        _log("Indexing repo docs and config...")
        doc_snippets = extract_doc_snippets(config.repo_path)
        coverage.docs = bool(doc_snippets)

    leads, file_edges = _build_leads(analysis, doc_snippets)
    _log(f"Generated {len(leads)} investigation leads...")

    cards_by_lead = investigate_leads(
        leads,
        config,
        config.repo_path,
        _log_fn=_log,
    )
    cards = _merge_cards(leads, cards_by_lead)

    metadata = BuildMetadata(
        repo_path=config.repo_path,
        built_at=datetime.now(UTC),
        total_commits_analyzed=len(commits),
        card_count=len(cards),
        source_coverage=coverage,
    )

    store = IndexStore(config.repo_path)
    try:
        store.store_build(cards, file_edges, metadata)
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


def _build_leads(
    analysis: AnalysisResult,
    doc_snippets: list[DocSnippet],
) -> tuple[list[InvestigationLead], list[FileEdge]]:
    leads: list[InvestigationLead] = []
    file_edges: list[FileEdge] = []

    for hotspot in analysis.hotspots[:MAX_HOTSPOT_LEADS]:
        leads.append(_lead_from_hotspot(hotspot))

    for hub in sorted(analysis.hub_files, key=lambda item: item.coupled_file_count, reverse=True)[:MAX_HOTSPOT_LEADS]:
        leads.append(_lead_from_hub(hub))

    for pair in sorted(analysis.coupling_pairs, key=lambda item: item.strength, reverse=True)[:MAX_COUPLING_LEADS]:
        confidence = max(pair.confidence_a_to_b, pair.confidence_b_to_a)
        leads.append(_lead_from_coupling(pair, confidence))
        file_edges.extend(_edges_from_coupling(pair, confidence))
        test_lead = _lead_from_test_association(pair, confidence)
        if test_lead is not None:
            leads.append(test_lead)

    for chain in sorted(analysis.fix_after_chains, key=lambda item: len(item.fixup_hashes), reverse=True)[:MAX_HISTORY_LEADS]:
        leads.append(_lead_from_fix_after(chain))

    for chain in sorted(analysis.revert_chains, key=lambda item: item.depth, reverse=True)[:MAX_HISTORY_LEADS]:
        leads.append(_lead_from_revert(chain))

    leads.extend(_review_leads(analysis)[:MAX_REVIEW_LEADS])
    leads.extend(_guidance_leads(analysis, doc_snippets)[:MAX_GUIDANCE_LEADS])

    deduped: dict[str, InvestigationLead] = {}
    for lead in leads:
        deduped.setdefault(lead.id, lead)
    ordered = sorted(
        deduped.values(),
        key=lambda item: (item.priority.sort_rank, -item.confidence, -item.support_count, item.title),
    )
    return ordered, file_edges


def _merge_cards(
    leads: list[InvestigationLead],
    cards_by_lead: dict[str, list[AdviceCard]],
) -> list[AdviceCard]:
    cards: list[AdviceCard] = []
    for lead in leads:
        generated = cards_by_lead.get(lead.id)
        if generated:
            cards.extend(generated)
        else:
            cards.extend(_fallback_cards_for_lead(lead))

    deduped: dict[str, AdviceCard] = {}
    for card in cards:
        existing = deduped.get(card.id)
        if existing is None or card.confidence > existing.confidence:
            deduped[card.id] = card
    return sorted(
        deduped.values(),
        key=lambda item: (item.priority.sort_rank, -item.confidence, -item.support_count, item.text),
    )


def _fallback_cards_for_lead(lead: InvestigationLead) -> list[AdviceCard]:
    text = lead.summary
    card_id = hashlib.sha1("\0".join([text, ",".join(lead.anchors)]).encode()).hexdigest()[:16]
    return [
        AdviceCard(
            id=card_id,
            text=text,
            priority=lead.priority,
            kind=lead.kind.advice_kind,
            applies_to=list(lead.applies_to) or [QueryIntent.GENERAL],
            anchors=list(lead.anchors),
            confidence=max(lead.confidence, 0.4),
            support_count=max(lead.support_count, 1),
            search_text=" ".join([text, " ".join(lead.anchors), lead.search_text]).strip(),
            created_by_build="deterministic",
            evidence=list(lead.evidence),
        )
    ]


def _make_lead(
    *,
    kind: LeadKind,
    title: str,
    summary: str,
    anchors: list[str] | None = None,
    applies_to: list[QueryIntent] | None = None,
    priority: AdvicePriority = AdvicePriority.MEDIUM,
    support_count: int = 0,
    confidence: float = 0.0,
    evidence: list[EvidenceRef] | None = None,
    search_text: str = "",
    prompt_context: str = "",
) -> InvestigationLead:
    anchors = anchors or []
    applies_to = applies_to or [QueryIntent.GENERAL]
    evidence = evidence or []
    lead_id = hashlib.sha1(
        "\0".join([kind.value, title, summary, ",".join(anchors)]).encode()
    ).hexdigest()[:16]
    return InvestigationLead(
        id=lead_id,
        kind=kind,
        title=title,
        summary=summary,
        anchors=anchors,
        applies_to=applies_to,
        priority=priority,
        support_count=support_count,
        confidence=confidence,
        evidence=evidence,
        search_text=" ".join([title, summary, " ".join(anchors), search_text]).strip(),
        prompt_context=prompt_context,
    )


def _lead_from_hotspot(hotspot) -> InvestigationLead:
    priority = AdvicePriority.HIGH if hotspot.fix_ratio >= 0.2 else AdvicePriority.MEDIUM
    return _make_lead(
        kind=LeadKind.HOTSPOT,
        title=f"Hotspot: {hotspot.path}",
        summary=(
            f"`{hotspot.path}` is a fragile area: it changes often and has a {hotspot.fix_ratio:.0%} "
            "fix ratio. Plan extra validation before editing it."
        ),
        anchors=[hotspot.path],
        applies_to=[QueryIntent.BUGFIX, QueryIntent.FEATURE, QueryIntent.REFACTOR, QueryIntent.REVIEW],
        priority=priority,
        support_count=hotspot.commit_count,
        confidence=min(hotspot.score / 10.0, 1.0),
        evidence=[
            EvidenceRef(
                source_type="hotspot",
                label="churn hotspot",
                ref=hotspot.path,
                excerpt=f"{hotspot.commit_count} commits, {hotspot.fix_ratio:.0%} fixes.",
            )
        ],
        prompt_context="Investigate why this file is fragile and what agents usually miss when changing it.",
    )


def _lead_from_hub(hub: HubFile) -> InvestigationLead:
    return _make_lead(
        kind=LeadKind.HOTSPOT,
        title=f"Coordination hub: {hub.path}",
        summary=(
            f"`{hub.path}` fans out across the repo and acts like a coordination point. "
            "Changes here often have a wider scope than the diff suggests."
        ),
        anchors=[hub.path],
        applies_to=[QueryIntent.BUGFIX, QueryIntent.FEATURE, QueryIntent.REFACTOR, QueryIntent.REVIEW],
        priority=AdvicePriority.MEDIUM,
        support_count=hub.coupled_file_count,
        confidence=min(hub.total_coupling_weight / 10.0, 1.0),
        evidence=[
            EvidenceRef(
                source_type="coupling",
                label="hub file",
                ref=hub.path,
                excerpt=f"Coupled with {hub.coupled_file_count} files.",
            )
        ],
        prompt_context="Investigate which neighboring files or tests usually matter when this file changes.",
    )


def _lead_from_coupling(pair: CouplingPair, confidence: float) -> InvestigationLead:
    return _make_lead(
        kind=LeadKind.COUPLING,
        title=f"Coupling: {pair.file_a} + {pair.file_b}",
        summary=(
            f"When editing `{pair.file_a}`, you should probably inspect `{pair.file_b}` too. "
            "These files change together often enough that plans are frequently too narrow."
        ),
        anchors=[pair.file_a, pair.file_b],
        applies_to=[QueryIntent.BUGFIX, QueryIntent.FEATURE, QueryIntent.REFACTOR, QueryIntent.REVIEW],
        priority=AdvicePriority.HIGH if confidence >= 0.5 else AdvicePriority.MEDIUM,
        support_count=int(pair.shared_commits),
        confidence=confidence,
        evidence=[
            EvidenceRef(
                source_type="coupling",
                label="co-change pair",
                ref=f"{pair.file_a}::{pair.file_b}",
                excerpt=f"Co-change confidence {confidence:.0%} across relevant commits.",
            )
        ],
        prompt_context="Confirm whether the coupling reflects a real planning dependency or just incidental churn.",
    )


def _lead_from_test_association(pair: CouplingPair, confidence: float) -> InvestigationLead | None:
    files = [pair.file_a, pair.file_b]
    source_files = [path for path in files if not _is_test_path(path)]
    test_files = [path for path in files if _is_test_path(path)]
    if len(source_files) != 1 or len(test_files) != 1:
        return None
    source_file = source_files[0]
    test_file = test_files[0]
    return _make_lead(
        kind=LeadKind.TEST_ASSOCIATION,
        title=f"Test association: {source_file} -> {test_file}",
        summary=(
            f"When editing `{source_file}`, inspect `{test_file}`. This area repeatedly needs "
            "test updates alongside code changes."
        ),
        anchors=[source_file, test_file],
        applies_to=[QueryIntent.BUGFIX, QueryIntent.FEATURE, QueryIntent.REFACTOR, QueryIntent.REVIEW],
        priority=AdvicePriority.HIGH,
        support_count=int(pair.shared_commits),
        confidence=confidence,
        evidence=[
            EvidenceRef(
                source_type="coupling",
                label="test association",
                ref=f"{source_file}::{test_file}",
                excerpt=f"These files co-change in {confidence:.0%} of relevant commits.",
            )
        ],
        prompt_context="Focus on what validation or scenario coverage this test file usually provides.",
    )


def _lead_from_fix_after(chain: FixAfterChain) -> InvestigationLead:
    priority = AdvicePriority.HIGH if chain.tier.value == "immediate" or len(chain.fixup_hashes) >= 3 else AdvicePriority.MEDIUM
    return _make_lead(
        kind=LeadKind.FIX_AFTER,
        title=f"Fix-after: {chain.original_subject}",
        summary=(
            f"Changes like `{chain.original_subject}` needed {len(chain.fixup_hashes)} follow-up fixes. "
            "Check the same edge cases before you plan a similar edit."
        ),
        anchors=list(chain.files),
        applies_to=[QueryIntent.BUGFIX, QueryIntent.REVIEW],
        priority=priority,
        support_count=len(chain.fixup_hashes),
        confidence=0.8,
        evidence=[
            EvidenceRef(
                source_type="fix_after",
                label="follow-up chain",
                ref=chain.original_hash,
                excerpt="; ".join(chain.fixup_subjects[:3]) or chain.original_subject,
            )
        ],
        search_text=chain.original_subject,
        prompt_context="Inspect the original and fixup commits to identify the repeated planning mistake.",
    )


def _lead_from_revert(chain: RevertChain) -> InvestigationLead:
    return _make_lead(
        kind=LeadKind.REVERT,
        title=f"Revert: {chain.original_subject}",
        summary=(
            f"`{chain.original_subject}` was reverted. Review the failure mode before repeating "
            "a similar approach in this area."
        ),
        anchors=list(chain.files),
        applies_to=[QueryIntent.BUGFIX, QueryIntent.REVIEW],
        priority=AdvicePriority.HIGH,
        support_count=chain.depth,
        confidence=0.9,
        evidence=[
            EvidenceRef(
                source_type="revert",
                label="revert chain",
                ref=chain.original_hash,
                excerpt=chain.original_subject,
            )
        ],
        search_text=chain.original_subject,
        prompt_context="Inspect the original and revert commits and determine what should change in future plans.",
    )


def _review_leads(analysis: AnalysisResult) -> list[InvestigationLead]:
    leads: list[InvestigationLead] = []
    if analysis.comment_clusters:
        for cluster in analysis.comment_clusters:
            anchors = sorted(
                {
                    item.comment.file_path
                    for item in cluster.comments
                    if item.comment.file_path
                }
            )
            leads.append(
                _make_lead(
                    kind=LeadKind.REVIEW,
                    title=f"Review theme: {cluster.label}",
                    summary=f"Reviewers repeatedly flag this theme: {cluster.label}.",
                    anchors=anchors,
                    applies_to=[QueryIntent.BUGFIX, QueryIntent.FEATURE, QueryIntent.REFACTOR, QueryIntent.REVIEW],
                    priority=AdvicePriority.HIGH if len(cluster.comments) >= 4 else AdvicePriority.MEDIUM,
                    support_count=len(cluster.comments),
                    confidence=cluster.coherence or 0.7,
                    evidence=[
                        EvidenceRef(
                            source_type="review_cluster",
                            label=f"PR #{item.comment.pr_number}",
                            ref=f"PR #{item.comment.pr_number}",
                            excerpt=item.comment.body[:180],
                        )
                        for item in cluster.comments[:5]
                    ],
                    search_text=" ".join(item.comment.body for item in cluster.comments[:5]),
                    prompt_context="Decide whether this review theme is a real planning constraint or just noise.",
                )
            )
        return leads

    grouped: dict[tuple[str, str], list[ClassifiedComment]] = defaultdict(list)
    for comment in analysis.classified_comments:
        categories = [item.value for item in comment.categories if item not in {
            CommentCategory.PRAISE,
            CommentCategory.QUESTION,
            CommentCategory.NITPICK,
        }]
        if not categories:
            continue
        anchor = comment.comment.file_path or "repo"
        grouped[(anchor, categories[0])].append(comment)

    for (anchor, category), comments in grouped.items():
        if len(comments) < 2:
            continue
        leads.append(
            _make_lead(
                kind=LeadKind.REVIEW,
                title=f"Review pattern in {anchor}: {category}",
                summary=(
                    f"Review history around `{anchor}` repeatedly raises {category}-style objections. "
                    "This likely reflects a real maintainer preference or risk."
                ),
                anchors=[] if anchor == "repo" else [anchor],
                applies_to=[QueryIntent.BUGFIX, QueryIntent.FEATURE, QueryIntent.REFACTOR, QueryIntent.REVIEW],
                priority=AdvicePriority.HIGH if len(comments) >= 3 else AdvicePriority.MEDIUM,
                support_count=len(comments),
                confidence=max(sum(item.confidence for item in comments) / len(comments), 0.6),
                evidence=[
                    EvidenceRef(
                        source_type="review_comment",
                        label=f"PR #{item.comment.pr_number}",
                        ref=f"PR #{item.comment.pr_number}",
                        excerpt=item.comment.body[:180],
                    )
                    for item in comments[:5]
                ],
                search_text=" ".join(item.comment.body for item in comments[:5]),
                prompt_context="Focus on repeated review objections that should change an agent's plan.",
            )
        )
    return sorted(
        leads,
        key=lambda item: (item.priority.sort_rank, -item.support_count, -item.confidence, item.title),
    )


def _guidance_leads(analysis: AnalysisResult, doc_snippets: list[DocSnippet]) -> list[InvestigationLead]:
    leads: list[InvestigationLead] = []
    if analysis.conventions is not None:
        for rule_text in analysis.conventions.detected_rules[:MAX_GUIDANCE_LEADS]:
            leads.append(
                _make_lead(
                    kind=LeadKind.CONVENTION,
                    title=f"Convention: {rule_text}",
                    summary=rule_text,
                    anchors=[],
                    applies_to=[QueryIntent.FEATURE, QueryIntent.REFACTOR, QueryIntent.REVIEW, QueryIntent.GENERAL],
                    priority=AdvicePriority.MEDIUM,
                    support_count=max(1, int(analysis.conventions.format_adherence * 100)),
                    confidence=analysis.conventions.format_adherence,
                    evidence=[
                        EvidenceRef(
                            source_type="config",
                            label="commit conventions",
                            ref="commit-conventions",
                            excerpt=rule_text,
                        )
                    ],
                    prompt_context="Only keep this as a planning card if it affects code or review outcomes, not just commit hygiene.",
                )
            )

    for snippet in doc_snippets:
        if not _is_guidance_snippet(snippet):
            continue
        leads.append(
            _make_lead(
                kind=LeadKind.DOC,
                title=f"Guidance: {snippet.title}",
                summary=snippet.content[:240],
                anchors=[snippet.path],
                applies_to=[QueryIntent.FEATURE, QueryIntent.REFACTOR, QueryIntent.REVIEW, QueryIntent.GENERAL],
                priority=AdvicePriority.LOW,
                support_count=1,
                confidence=0.6,
                evidence=[
                    EvidenceRef(
                        source_type=snippet.source_type,
                        label=snippet.path,
                        ref=snippet.path,
                        excerpt=snippet.content[:180],
                    )
                ],
                search_text=f"{snippet.title} {snippet.path}",
                prompt_context="Only turn this into a card if it encodes stable repo guidance that affects planning.",
            )
        )
    return leads


def _edges_from_coupling(pair: CouplingPair, confidence: float) -> list[FileEdge]:
    reason = f"co-change ({confidence:.0%} of relevant commits)"
    return [
        FileEdge(src=pair.file_a, dst=pair.file_b, edge_type="cochange", score=pair.strength, reason=reason),
        FileEdge(src=pair.file_b, dst=pair.file_a, edge_type="cochange", score=pair.strength, reason=reason),
    ]


def _is_guidance_snippet(snippet: DocSnippet) -> bool:
    title = snippet.title.lower()
    return any(token in title for token in {"build", "run", "environment", "config", "convention", "testing"})


def _is_test_path(path: str) -> bool:
    lower = path.lower()
    return lower.startswith("tests/") or "/test" in lower or lower.endswith("_test.py")
