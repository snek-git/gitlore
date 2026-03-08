"""Retrieve planning briefs from the local advice-card index."""

from __future__ import annotations

import re
from pathlib import Path

from gitlore.config import GitloreConfig
from gitlore.index import IndexStore, brief_to_json
from gitlore.models import (
    AdviceCard,
    PlanningBrief,
    PlanningNote,
    PlanningQuery,
    QueryIntent,
    RelatedFile,
    SourceCoverage,
)

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


def build_planning_brief(
    config: GitloreConfig,
    *,
    task: str,
    files: list[str] | None = None,
    diff_path: str | None = None,
    tentative_plan: str = "",
    question: str = "",
    max_notes: int = 5,
) -> PlanningBrief:
    """Return a task-scoped planning brief using local retrieval only."""
    query = PlanningQuery(
        task=task,
        intent=infer_intent(task, diff_path=diff_path),
        files=_normalize_files(config.repo_path, files or []),
        diff_text=_read_diff(diff_path),
        diff_path=diff_path,
        tentative_plan=tentative_plan,
        question=question,
        max_notes=max_notes,
    )

    store = IndexStore(config.repo_path)
    try:
        if not store.has_index():
            raise FileNotFoundError("No gitlore index found. Run `gitlore build` first.")
        metadata = store.get_build_metadata()
        cards = store.load_cards()
        related_files = _collect_related_files(store, query.files)
        fts_scores = store.search_fts(_fts_query(query), limit=100)
    finally:
        store.close()

    ranked = _rank_cards(cards, query, related_files, fts_scores)
    notes = [
        PlanningNote(
            text=card.text,
            refs=[item.ref for item in card.evidence[:3]],
            priority=card.priority,
        )
        for card in ranked[: query.max_notes]
    ]
    summary = f"{len(notes)} planning note{'s' if len(notes) != 1 else ''} for this change"
    return PlanningBrief(
        task=task,
        summary=summary,
        notes=notes,
        related_files=related_files[:5],
        source_coverage=metadata.source_coverage if metadata else SourceCoverage(),
        build_metadata=metadata,
    )


def render_planning_brief(brief: PlanningBrief, *, format_name: str = "summary") -> str:
    """Render the planning brief for CLI or MCP usage."""
    if format_name == "json":
        return brief_to_json(brief)

    lines = [brief.summary]
    for note in brief.notes:
        lines.append("")
        lines.append(f"- [{note.priority.value}] {note.text}")
        if note.refs:
            lines.append(f"  refs: {', '.join(note.refs)}")
    return "\n".join(lines)


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


def _rank_cards(
    cards: list[AdviceCard],
    query: PlanningQuery,
    related_files,
    fts_scores: dict[str, float],
) -> list[AdviceCard]:
    tokens = _query_tokens(query)
    token_set = set(tokens)
    related_scores = {item.path: item.score for item in related_files}
    support_max = max((card.support_count for card in cards), default=1)

    scored: list[tuple[float, AdviceCard]] = []
    for card in cards:
        anchor_match = _anchor_match(query.files, card.anchors)
        intent_match = 1.0 if query.intent in card.applies_to or QueryIntent.GENERAL in card.applies_to else 0.0
        graph_match = _graph_match(card.anchors, related_scores)
        lexical = _lexical_match(card, token_set)
        fts = fts_scores.get(card.id, 0.0)
        support = card.support_count / max(support_max, 1)

        score = (
            0.35 * anchor_match
            + 0.20 * intent_match
            + 0.20 * max(lexical, fts)
            + 0.10 * graph_match
            + 0.10 * card.confidence
            + 0.03 * support
            + 0.02 * card.priority.retrieval_weight
        )
        if score <= 0.05 and not anchor_match and not lexical and not fts:
            continue
        scored.append((score, card))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].priority.sort_rank,
            -item[1].confidence,
            -item[1].support_count,
            item[1].text,
        )
    )
    return [card for _, card in scored]


def _query_tokens(query: PlanningQuery) -> list[str]:
    parts = [query.task, query.tentative_plan, query.question, query.diff_text] + query.files
    return [token.lower() for token in _TOKEN_RE.findall(" ".join(parts))]


def _anchor_match(query_files: list[str], anchors: list[str]) -> float:
    if not query_files or not anchors:
        return 0.0
    query_set = set(query_files)
    anchor_set = set(anchors)
    if query_set & anchor_set:
        return 1.0
    query_names = {Path(item).name for item in query_files}
    anchor_names = {Path(item).name for item in anchors}
    if query_names & anchor_names:
        return 0.5
    return 0.0


def _graph_match(anchors: list[str], related_scores: dict[str, float]) -> float:
    if not anchors:
        return 0.0
    return max((related_scores.get(anchor, 0.0) for anchor in anchors), default=0.0)


def _lexical_match(card: AdviceCard, token_set: set[str]) -> float:
    if not token_set:
        return 0.0
    card_tokens = set(_TOKEN_RE.findall(card.search_text.lower()))
    if not card_tokens:
        return 0.0
    overlap = len(token_set & card_tokens)
    if overlap == 0:
        return 0.0
    return overlap / len(token_set)


def _fts_query(query: PlanningQuery) -> str:
    tokens = _query_tokens(query)
    if not tokens:
        return ""
    unique = sorted(dict.fromkeys(tokens))[:20]
    escaped = ['"' + token.replace('"', '""') + '"' for token in unique]
    return " OR ".join(escaped)
