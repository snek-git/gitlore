"""Retrieve knowledge notes from the local index."""

from __future__ import annotations

import asyncio
import math
import os
import re
from pathlib import Path

from gitlore.config import GitloreConfig
from gitlore.index import IndexStore, brief_to_json
from gitlore.models import (
    CONFIDENCE_WEIGHT,
    KnowledgeNote,
    PlanningBrief,
    PlanningNote,
    PlanningQuery,
    RelatedFile,
    SourceCoverage,
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_/.-]+")


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
        notes = store.load_notes()
        related_files = _collect_related_files(store, query.files)
        fts_scores = store.search_fts(_fts_query(query), limit=100)
    finally:
        store.close()

    query_embedding = _embed_query(query, config)
    ranked = _rank_notes(notes, query, related_files, fts_scores, query_embedding)
    planning_notes = [
        PlanningNote(
            text=note.text,
            refs=note.evidence_refs[:3],
            confidence=note.confidence,
        )
        for note in ranked[: query.max_notes]
    ]
    summary = f"{len(planning_notes)} note{'s' if len(planning_notes) != 1 else ''} for this change"
    return PlanningBrief(
        task=task,
        summary=summary,
        notes=planning_notes,
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
        lines.append(f"- [{note.confidence}] {note.text}")
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


# ── Semantic retrieval ────────────────────────────────────────────────────


def _embed_query(query: PlanningQuery, config: GitloreConfig) -> list[float] | None:
    """Embed the query text for semantic matching. Returns None if unavailable."""
    if not config.models.embedding or not os.environ.get("OPENROUTER_API_KEY", ""):
        return None

    query_text = f"{query.task} {query.tentative_plan} {query.question}".strip()
    if not query_text:
        return None

    try:
        from gitlore.utils.llm import embed
        result = asyncio.run(embed(config.models.embedding, [query_text]))
        return result[0] if result else None
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_match(
    note_embedding: list[float] | None,
    query_embedding: list[float] | None,
) -> float:
    """Cosine similarity between note and query embeddings, or 0.0 if unavailable."""
    if note_embedding is None or query_embedding is None:
        return 0.0
    return max(0.0, _cosine_similarity(note_embedding, query_embedding))


# ── Ranking ───────────────────────────────────────────────────────────────


def _rank_notes(
    notes: list[KnowledgeNote],
    query: PlanningQuery,
    related_files: list[RelatedFile],
    fts_scores: dict[str, float],
    query_embedding: list[float] | None = None,
) -> list[KnowledgeNote]:
    tokens = _query_tokens(query)
    token_set = set(tokens)
    related_scores = {item.path: item.score for item in related_files}
    has_embeddings = query_embedding is not None and any(n.embedding is not None for n in notes)

    scored: list[tuple[float, KnowledgeNote]] = []
    for note in notes:
        anchor_match = _anchor_match(query.files, note.anchors)
        graph_match = _graph_match(note.anchors, related_scores)
        lexical = _lexical_match(note, token_set)
        fts = fts_scores.get(note.id, 0.0)
        semantic = _semantic_match(note.embedding, query_embedding) if has_embeddings else 0.0
        confidence = CONFIDENCE_WEIGHT.get(note.confidence, 0.6)

        if has_embeddings:
            score = (
                0.30 * anchor_match
                + 0.20 * max(lexical, fts)
                + 0.20 * semantic
                + 0.12 * graph_match
                + 0.10 * confidence
                + 0.08 * (1.0 if anchor_match or lexical or fts or semantic > 0.5 else 0.0)
            )
        else:
            score = (
                0.40 * anchor_match
                + 0.25 * max(lexical, fts)
                + 0.15 * graph_match
                + 0.10 * confidence
                + 0.10 * (1.0 if anchor_match or lexical or fts else 0.0)
            )

        if score <= 0.05 and not anchor_match and not lexical and not fts and semantic < 0.3:
            continue
        scored.append((score, note))

    scored.sort(key=lambda item: (-item[0], item[1].confidence, item[1].text))
    return [note for _, note in scored]


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
    for qf in query_files:
        for anchor in anchors:
            if qf.startswith(anchor.rstrip("/") + "/") or anchor.startswith(qf.rstrip("/") + "/"):
                return 0.3
    return 0.0


def _graph_match(anchors: list[str], related_scores: dict[str, float]) -> float:
    if not anchors:
        return 0.0
    return max((related_scores.get(anchor, 0.0) for anchor in anchors), default=0.0)


def _lexical_match(note: KnowledgeNote, token_set: set[str]) -> float:
    if not token_set:
        return 0.0
    note_tokens = set(_TOKEN_RE.findall(note.search_text.lower()))
    if not note_tokens:
        return 0.0
    overlap = len(token_set & note_tokens)
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
