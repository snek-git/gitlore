"""SQLite-backed advice-card index for gitlore."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from gitlore.models import (
    AdviceCard,
    AdviceKind,
    AdvicePriority,
    BuildMetadata,
    EvidenceRef,
    FileEdge,
    PlanningBrief,
    QueryIntent,
    RelatedFile,
    SourceCoverage,
)


def index_path(repo_path: str) -> Path:
    """Return the on-disk location of the planning index."""
    return Path(repo_path) / ".gitlore" / "index.db"


def _encode_list(items: list[str]) -> str:
    return json.dumps(items)


def _decode_list(raw: str) -> list[str]:
    if not raw:
        return []
    return list(json.loads(raw))


class IndexStore:
    """Persist and query build-time advice cards for planning retrieval."""

    def __init__(self, repo_path: str) -> None:
        db_path = index_path(repo_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def close(self) -> None:
        self._conn.close()

    def reset(self) -> None:
        self._conn.executescript(
            """
            DELETE FROM advice_cards;
            DELETE FROM card_evidence;
            DELETE FROM file_edges;
            DELETE FROM build_metadata;
            DELETE FROM card_fts;
            """
        )
        self._conn.commit()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS advice_cards (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                priority TEXT NOT NULL,
                kind TEXT NOT NULL,
                applies_to TEXT NOT NULL,
                anchors TEXT NOT NULL,
                confidence REAL NOT NULL,
                support_count INTEGER NOT NULL,
                search_text TEXT NOT NULL,
                created_by_build TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS card_evidence (
                card_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                label TEXT NOT NULL,
                ref TEXT NOT NULL,
                excerpt TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (card_id, position)
            );

            CREATE TABLE IF NOT EXISTS file_edges (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                score REAL NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (src, dst, edge_type)
            );

            CREATE TABLE IF NOT EXISTS build_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS card_fts USING fts5(
                card_id UNINDEXED,
                text,
                anchors,
                search_text
            );
            """
        )
        self._conn.commit()

    def store_build(
        self,
        cards: list[AdviceCard],
        edges: list[FileEdge],
        metadata: BuildMetadata,
    ) -> None:
        """Replace the current index contents with a fresh build."""
        self.reset()

        for card in cards:
            self._conn.execute(
                """
                INSERT INTO advice_cards (
                    id, text, priority, kind, applies_to, anchors,
                    confidence, support_count, search_text, created_by_build
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card.id,
                    card.text,
                    card.priority.value,
                    card.kind.value,
                    _encode_list([item.value for item in card.applies_to]),
                    _encode_list(card.anchors),
                    card.confidence,
                    card.support_count,
                    card.search_text,
                    card.created_by_build,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO card_fts (card_id, text, anchors, search_text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    card.id,
                    card.text,
                    " ".join(card.anchors),
                    card.search_text,
                ),
            )
            for index, evidence in enumerate(card.evidence):
                self._conn.execute(
                    """
                    INSERT INTO card_evidence (
                        card_id, position, source_type, label, ref, excerpt, weight
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card.id,
                        index,
                        evidence.source_type,
                        evidence.label,
                        evidence.ref,
                        evidence.excerpt,
                        evidence.weight,
                    ),
                )

        for edge in edges:
            self._conn.execute(
                """
                INSERT INTO file_edges (src, dst, edge_type, score, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    edge.src,
                    edge.dst,
                    edge.edge_type,
                    edge.score,
                    edge.reason,
                ),
            )

        self._conn.execute(
            "INSERT OR REPLACE INTO build_metadata (key, value) VALUES (?, ?)",
            (
                "metadata",
                json.dumps(
                    {
                        "repo_path": metadata.repo_path,
                        "built_at": metadata.built_at.isoformat(),
                        "total_commits_analyzed": metadata.total_commits_analyzed,
                        "card_count": metadata.card_count,
                        "source_coverage": asdict(metadata.source_coverage),
                    }
                ),
            ),
        )
        self._conn.commit()

    def load_cards(self) -> list[AdviceCard]:
        rows = self._conn.execute(
            "SELECT * FROM advice_cards ORDER BY priority, confidence DESC, id"
        ).fetchall()
        return [self._row_to_card(row) for row in rows]

    def get_related_files(self, path: str, limit: int = 5) -> list[RelatedFile]:
        rows = self._conn.execute(
            """
            SELECT dst, reason, score
            FROM file_edges
            WHERE src = ?
            ORDER BY score DESC, dst
            LIMIT ?
            """,
            (path, limit),
        ).fetchall()
        return [
            RelatedFile(
                path=str(row["dst"]),
                reason=str(row["reason"]),
                score=float(row["score"]),
            )
            for row in rows
        ]

    def search_fts(self, query: str, limit: int = 50) -> dict[str, float]:
        if not query.strip():
            return {}
        rows = self._conn.execute(
            """
            SELECT card_id, bm25(card_fts) AS rank
            FROM card_fts
            WHERE card_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        scores: dict[str, float] = {}
        for row in rows:
            rank = float(row["rank"])
            scores[str(row["card_id"])] = 1.0 / (1.0 + max(rank, 0.0))
        return scores

    def get_build_metadata(self) -> BuildMetadata | None:
        row = self._conn.execute(
            "SELECT value FROM build_metadata WHERE key = ?",
            ("metadata",),
        ).fetchone()
        if row is None:
            return None
        raw = json.loads(row["value"])
        return BuildMetadata(
            repo_path=raw["repo_path"],
            built_at=datetime.fromisoformat(raw["built_at"]),
            total_commits_analyzed=raw["total_commits_analyzed"],
            card_count=raw.get("card_count", raw.get("fact_count", 0)),
            source_coverage=SourceCoverage(**raw["source_coverage"]),
        )

    def has_index(self) -> bool:
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM advice_cards"
        ).fetchone()
        return row is not None and int(row["count"]) > 0

    def _row_to_card(self, row: sqlite3.Row) -> AdviceCard:
        evidence_rows = self._conn.execute(
            """
            SELECT source_type, label, ref, excerpt, weight
            FROM card_evidence
            WHERE card_id = ?
            ORDER BY position
            """,
            (row["id"],),
        ).fetchall()
        evidence = [
            EvidenceRef(
                source_type=str(e_row["source_type"]),
                label=str(e_row["label"]),
                ref=str(e_row["ref"]),
                excerpt=str(e_row["excerpt"]),
                weight=float(e_row["weight"]),
            )
            for e_row in evidence_rows
        ]
        return AdviceCard(
            id=str(row["id"]),
            text=str(row["text"]),
            priority=AdvicePriority(str(row["priority"])),
            kind=AdviceKind(str(row["kind"])),
            applies_to=[QueryIntent(item) for item in _decode_list(str(row["applies_to"]))],
            anchors=_decode_list(str(row["anchors"])),
            confidence=float(row["confidence"]),
            support_count=int(row["support_count"]),
            search_text=str(row["search_text"]),
            created_by_build=str(row["created_by_build"]),
            evidence=evidence,
        )


def brief_to_json(brief: PlanningBrief) -> str:
    """Serialize a PlanningBrief into minimal JSON for CLI and MCP responses."""
    payload = {
        "summary": brief.summary,
        "notes": [
            {
                "text": note.text,
                "refs": note.refs,
                "priority": note.priority.value,
            }
            for note in brief.notes
        ],
    }
    return json.dumps(payload, indent=2, default=str)


def guidance_to_json(cards: list[AdviceCard], metadata: BuildMetadata | None) -> str:
    """Serialize repo guidance cards for MCP and CLI debugging."""
    payload = {
        "cards": [
            {
                "text": card.text,
                "priority": card.priority.value,
                "kind": card.kind.value,
                "refs": [evidence.ref for evidence in card.evidence[:3]],
            }
            for card in cards
        ],
        "build_metadata": asdict(metadata) if metadata else None,
    }
    return json.dumps(payload, indent=2, default=str)
