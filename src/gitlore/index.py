"""SQLite-backed knowledge index for gitlore."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from gitlore.models import (
    BuildMetadata,
    FileEdge,
    KnowledgeNote,
    PlanningBrief,
    RelatedFile,
    SourceCoverage,
)


def index_path(repo_path: str) -> Path:
    """Return the on-disk location of the knowledge index."""
    return Path(repo_path) / ".gitlore" / "index.db"


class IndexStore:
    """Persist and query build-time knowledge notes for retrieval."""

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
            DELETE FROM knowledge_notes;
            DELETE FROM file_edges;
            DELETE FROM build_metadata;
            DELETE FROM note_fts;
            """
        )
        self._conn.commit()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_notes (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                anchors TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                confidence TEXT NOT NULL,
                search_text TEXT NOT NULL,
                created_at TEXT
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

            CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
                note_id UNINDEXED,
                text,
                anchors,
                search_text
            );
            """
        )
        self._conn.commit()

    def store_build(
        self,
        notes: list[KnowledgeNote],
        edges: list[FileEdge],
        metadata: BuildMetadata,
    ) -> None:
        """Replace the current index contents with a fresh build."""
        self.reset()

        for note in notes:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_notes (
                    id, text, anchors, evidence_refs, confidence, search_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note.id,
                    note.text,
                    json.dumps(note.anchors),
                    json.dumps(note.evidence_refs),
                    note.confidence,
                    note.search_text,
                    note.created_at.isoformat() if note.created_at else None,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO note_fts (note_id, text, anchors, search_text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    note.id,
                    note.text,
                    " ".join(note.anchors),
                    note.search_text,
                ),
            )

        for edge in edges:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO file_edges (src, dst, edge_type, score, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (edge.src, edge.dst, edge.edge_type, edge.score, edge.reason),
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
                        "note_count": metadata.note_count,
                        "head_commit": metadata.head_commit,
                        "source_coverage": asdict(metadata.source_coverage),
                    }
                ),
            ),
        )
        self._conn.commit()

    def load_notes(self) -> list[KnowledgeNote]:
        rows = self._conn.execute(
            "SELECT * FROM knowledge_notes ORDER BY confidence, id"
        ).fetchall()
        return [self._row_to_note(row) for row in rows]

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
            SELECT note_id, bm25(note_fts) AS rank
            FROM note_fts
            WHERE note_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        scores: dict[str, float] = {}
        for row in rows:
            rank = float(row["rank"])
            scores[str(row["note_id"])] = 1.0 / (1.0 + max(rank, 0.0))
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
            note_count=raw["note_count"],
            head_commit=raw.get("head_commit", ""),
            source_coverage=SourceCoverage(**raw["source_coverage"]),
        )

    def has_index(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM build_metadata WHERE key = ?",
            ("metadata",),
        ).fetchone()
        if row is not None:
            return True
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM knowledge_notes"
        ).fetchone()
        return row is not None and int(row["count"]) > 0

    def _row_to_note(self, row: sqlite3.Row) -> KnowledgeNote:
        created_at = None
        if row["created_at"]:
            created_at = datetime.fromisoformat(row["created_at"])
        return KnowledgeNote(
            id=str(row["id"]),
            text=str(row["text"]),
            anchors=json.loads(row["anchors"]),
            evidence_refs=json.loads(row["evidence_refs"]),
            confidence=str(row["confidence"]),
            created_at=created_at,
            search_text=str(row["search_text"]),
        )


def brief_to_json(brief: PlanningBrief) -> str:
    """Serialize a PlanningBrief into minimal JSON for CLI and MCP responses."""
    payload = {
        "summary": brief.summary,
        "notes": [
            {
                "text": note.text,
                "refs": note.refs,
                "confidence": note.confidence,
            }
            for note in brief.notes
        ],
    }
    return json.dumps(payload, indent=2, default=str)


def guidance_to_json(notes: list[KnowledgeNote], metadata: BuildMetadata | None) -> str:
    """Serialize repo guidance notes for MCP and CLI debugging."""
    payload = {
        "notes": [
            {
                "text": note.text,
                "confidence": note.confidence,
                "refs": note.evidence_refs[:3],
            }
            for note in notes
        ],
        "build_metadata": asdict(metadata) if metadata else None,
    }
    return json.dumps(payload, indent=2, default=str)
