"""Retrieve task-scoped context bundles from the local knowledge index."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from pathlib import Path

from gitlore.config import GitloreConfig
from gitlore.index import IndexStore, bundle_to_json
from gitlore.models import (
    ContextBundle,
    ContextItem,
    ContextQuery,
    FactKind,
    KnowledgeFact,
    QueryIntent,
    RelatedFile,
    SourceCoverage,
)
from gitlore.utils.llm import complete_sync

_BUGFIX_WORDS = {"fix", "bug", "flaky", "error", "fail", "regression"}
_REFACTOR_WORDS = {"refactor", "cleanup", "simplify", "restructure"}
_FEATURE_WORDS = {"add", "implement", "introduce", "support"}
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_/.-]+")


def infer_intent(task: str, *, diff_path: str | None = None) -> QueryIntent:
    """Infer the dominant task intent from simple keywords."""
    lowered = task.lower()
    tokens = set(_TOKEN_RE.findall(lowered))
    if diff_path or "review" in tokens:
        return QueryIntent.REVIEW
    if tokens & _BUGFIX_WORDS:
        return QueryIntent.BUGFIX
    if tokens & _REFACTOR_WORDS:
        return QueryIntent.REFACTOR
    if tokens & _FEATURE_WORDS:
        return QueryIntent.FEATURE
    return QueryIntent.GENERAL


def build_context(
    config: GitloreConfig,
    *,
    task: str,
    files: list[str] | None = None,
    diff_path: str | None = None,
    format_name: str | None = None,
    max_items: int | None = None,
    max_tokens: int | None = None,
    compress: bool = False,
) -> ContextBundle:
    """Return a task-scoped context bundle using only the local index."""
    files = _normalize_files(config.repo_path, files or [])
    diff_text = _read_diff(diff_path)
    intent = infer_intent(task, diff_path=diff_path)
    query = ContextQuery(
        task=task,
        intent=intent,
        files=files,
        diff_text=diff_text,
        diff_path=diff_path,
        max_items=max_items or config.query.max_items,
        max_tokens=max_tokens or config.query.max_tokens,
        format=format_name or config.query.default_format,
        compress=compress,
    )

    store = IndexStore(config.repo_path)
    try:
        if not store.has_index():
            raise FileNotFoundError("No gitlore index found. Run `gitlore build` first.")
        metadata = store.get_build_metadata()
        facts = store.load_facts()
        related = _collect_related_files(store, query.files)
        fts_scores = store.search_fts(_fts_query(query), limit=100)
    finally:
        store.close()

    ranked = _rank_facts(facts, query, related, fts_scores)
    rules, situational, examples = _package_items(ranked, query.max_items)
    suggested_tests = _suggest_tests(ranked)
    summary = _build_summary(task, rules, situational, examples)

    return ContextBundle(
        task=task,
        intent=intent,
        files=query.files,
        summary=summary,
        rules=rules,
        situational=situational,
        examples=examples,
        related_files=related[:5],
        suggested_tests=suggested_tests[:3],
        source_coverage=metadata.source_coverage if metadata else SourceCoverage(),
        build_metadata=metadata,
    )


def render_context(bundle: ContextBundle, *, format_name: str, config: GitloreConfig, compress: bool = False) -> str:
    """Render a ContextBundle into the requested output format."""
    if format_name == "json":
        return bundle_to_json(bundle)

    if format_name == "prompt":
        rendered = _render_prompt(bundle)
    else:
        rendered = _render_summary(bundle)

    if compress:
        if not config.models.compressor:
            raise ValueError("`--compress` requires [models].compressor to be configured.")
        rendered = complete_sync(
            model=config.models.compressor,
            system=(
                "Rewrite repo context into a compact, actionable brief for a coding agent. "
                "Preserve facts, file paths, and warnings. Do not invent details."
            ),
            user=rendered,
            temperature=0.0,
            max_tokens=400,
        ).strip()

    return rendered


def _normalize_files(repo_path: str, files: list[str]) -> list[str]:
    repo = Path(repo_path)
    normalized: list[str] = []
    for file_name in files:
        path = Path(file_name)
        if path.is_absolute():
            try:
                normalized.append(str(path.relative_to(repo)))
            except ValueError:
                continue
        else:
            normalized.append(str(path))
    return sorted(dict.fromkeys(normalized))


def _read_diff(diff_path: str | None) -> str:
    if not diff_path:
        return ""
    path = Path(diff_path)
    if not path.exists():
        return ""
    return path.read_text()[:4000]


def _collect_related_files(store: IndexStore, files: list[str]) -> list[RelatedFile]:
    related: dict[str, RelatedFile] = {}
    for path in files:
        for item in store.get_related_files(path, limit=5):
            existing = related.get(item.path)
            if existing is None or item.score > existing.score:
                related[item.path] = item
    return sorted(related.values(), key=lambda item: (-item.score, item.path))


def _rank_facts(
    facts: list[KnowledgeFact],
    query: ContextQuery,
    related_files: list[RelatedFile],
    fts_scores: dict[str, float],
) -> list[ContextItem]:
    lexical_scores = _lexical_scores(facts, query, fts_scores)
    semantic_scores = {
        fact.id: _semantic_similarity(query.task + "\n" + query.diff_text, fact.search_text)
        for fact in facts
    }
    support_max = max((fact.support_count for fact in facts), default=1)
    related_scores = {item.path: item.score for item in related_files}

    ranked: list[ContextItem] = []
    for fact in facts:
        file_match = _file_match(query.files, fact.files)
        graph_proximity = _graph_proximity(fact.files, related_scores)
        lexical = lexical_scores.get(fact.id, 0.0)
        semantic = semantic_scores.get(fact.id, 0.0)
        confidence = fact.confidence
        support = fact.support_count / max(support_max, 1)
        recency = _recency_score(fact)

        score = (
            0.30 * file_match
            + 0.25 * lexical
            + 0.15 * semantic
            + 0.10 * graph_proximity
            + 0.10 * confidence
            + 0.05 * support
            + 0.05 * recency
        )

        if query.intent in {QueryIntent.BUGFIX, QueryIntent.REVIEW} and fact.kind in {
            FactKind.FRAGILE_AREA,
            FactKind.HISTORICAL_EXAMPLE,
        }:
            score += 0.10
        if query.intent in {QueryIntent.FEATURE, QueryIntent.REFACTOR} and fact.kind in {
            FactKind.RULE,
            FactKind.DOC_GUIDANCE,
            FactKind.TEST_ASSOCIATION,
        }:
            score += 0.10

        score = max(0.0, min(score, 1.0))
        if score <= 0.05 and not file_match and not lexical:
            continue

        ranked.append(
            ContextItem(
                fact_id=fact.id,
                kind=fact.kind,
                title=fact.title,
                guidance=fact.guidance,
                files=fact.files,
                score=score,
                why_selected=_why_selected(file_match, lexical, graph_proximity, fact),
                evidence=fact.evidence[:3],
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.title))
    return ranked


def _lexical_scores(
    facts: list[KnowledgeFact],
    query: ContextQuery,
    fts_scores: dict[str, float],
) -> dict[str, float]:
    tokens = _query_tokens(query)
    if not tokens:
        return fts_scores

    query_tokens = set(tokens)
    scores: dict[str, float] = {}
    for fact in facts:
        fact_tokens = set(_TOKEN_RE.findall(fact.search_text.lower()))
        if not fact_tokens:
            continue
        overlap = len(query_tokens & fact_tokens)
        if overlap == 0:
            continue
        scores[fact.id] = max(overlap / len(query_tokens), fts_scores.get(fact.id, 0.0))
    for fact_id, score in fts_scores.items():
        scores.setdefault(fact_id, score)
    return scores


def _query_tokens(query: ContextQuery) -> list[str]:
    parts = [query.task] + query.files
    if query.diff_text:
        parts.append(query.diff_text[:500])
    return [token.lower() for token in _TOKEN_RE.findall(" ".join(parts))]


def _semantic_similarity(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_RE.findall(left.lower()))
    right_tokens = set(_TOKEN_RE.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))


def _file_match(query_files: list[str], fact_files: list[str]) -> float:
    if not query_files or not fact_files:
        return 0.0
    query_set = set(query_files)
    fact_set = set(fact_files)
    if query_set & fact_set:
        return 1.0
    query_names = {Path(path).name for path in query_files}
    fact_names = {Path(path).name for path in fact_files}
    if query_names & fact_names:
        return 0.5
    return 0.0


def _graph_proximity(fact_files: list[str], related_scores: dict[str, float]) -> float:
    if not fact_files:
        return 0.0
    return max((related_scores.get(path, 0.0) for path in fact_files), default=0.0)


def _recency_score(fact: KnowledgeFact) -> float:
    if fact.last_seen_at is None:
        return 0.2
    age_days = max((datetime.now(UTC) - fact.last_seen_at).total_seconds() / 86400.0, 0.0)
    return 1.0 / (1.0 + age_days / 90.0)


def _why_selected(file_match: float, lexical: float, graph_proximity: float, fact: KnowledgeFact) -> str:
    reasons: list[str] = []
    if file_match:
        reasons.append("direct file match")
    if graph_proximity:
        reasons.append("related file expansion")
    if lexical:
        reasons.append("task text overlap")
    if fact.support_count:
        reasons.append(f"support={fact.support_count}")
    return ", ".join(reasons) or "global repo guidance"


def _package_items(
    ranked: list[ContextItem],
    max_items: int,
) -> tuple[list[ContextItem], list[ContextItem], list[ContextItem]]:
    rules: list[ContextItem] = []
    situational: list[ContextItem] = []
    examples: list[ContextItem] = []

    for item in ranked:
        if len(rules) + len(situational) + len(examples) >= max_items:
            break
        if item.kind in {FactKind.RULE, FactKind.DOC_GUIDANCE} and len(rules) < 3:
            rules.append(item)
        elif item.kind in {FactKind.HISTORICAL_EXAMPLE} and len(examples) < 3:
            examples.append(item)
        elif item.kind not in {FactKind.TEST_ASSOCIATION} and len(situational) < 4:
            situational.append(item)

    return rules, situational, examples


def _suggest_tests(ranked: list[ContextItem]) -> list[str]:
    tests: list[str] = []
    for item in ranked:
        if item.kind != FactKind.TEST_ASSOCIATION:
            continue
        for path in item.files:
            if "test" in path.lower() and path not in tests:
                tests.append(path)
    return tests


def _build_summary(
    task: str,
    rules: list[ContextItem],
    situational: list[ContextItem],
    examples: list[ContextItem],
) -> list[str]:
    lines = [f"Task: {task}"]
    for item in (rules + situational + examples)[:4]:
        lines.append(f"- {item.title}")
    return lines


def _render_summary(bundle: ContextBundle) -> str:
    lines = [f"Task: {bundle.task}", "", "Relevant context"]
    for item in bundle.rules:
        lines.append(f"- {item.guidance}")
    for item in bundle.situational:
        lines.append(f"- {item.guidance}")
    if bundle.examples:
        lines.append("")
        lines.append("Useful examples")
        for item in bundle.examples:
            lines.append(f"- {item.guidance}")
    if bundle.related_files:
        lines.append("")
        lines.append("Related files")
        for item in bundle.related_files:
            lines.append(f"- {item.path} ({item.reason})")
    if bundle.suggested_tests:
        lines.append("")
        lines.append("Suggested tests")
        for path in bundle.suggested_tests:
            lines.append(f"- {path}")
    return "\n".join(lines)


def _render_prompt(bundle: ContextBundle) -> str:
    lines = [
        "Use this repository context for the current task:",
        "",
        f"Task: {bundle.task}",
        "",
        "Relevant tribal knowledge:",
    ]
    for item in bundle.rules + bundle.situational + bundle.examples:
        lines.append(f"- {item.guidance}")
    if bundle.suggested_tests:
        lines.append("")
        lines.append("Suggested tests:")
        for path in bundle.suggested_tests:
            lines.append(f"- {path}")
    return "\n".join(lines)


def _fts_query(query: ContextQuery) -> str:
    tokens = _query_tokens(query)
    if not tokens:
        return ""
    unique = sorted(dict.fromkeys(tokens))[:20]
    escaped = ['"' + token.replace('"', '""') + '"' for token in unique]
    return " OR ".join(escaped)
