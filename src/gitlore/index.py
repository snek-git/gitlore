"""SQLite-backed knowledge index for gitlore."""

from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from gitlore.models import (
    BuildMetadata,
    ContextBundle,
    EvidenceRef,
    FactKind,
    FactStability,
    KnowledgeFact,
    KnowledgeSeverity,
    QueryIntent,
    RelatedFile,
    SourceCoverage,
)


def index_path(repo_path: str) -> Path:
    """Return the on-disk location of the context index."""
    return Path(repo_path) / ".gitlore" / "index.db"


def _encode_list(items: list[str]) -> str:
    return json.dumps(items)


def _decode_list(raw: str) -> list[str]:
    if not raw:
        return []
    return list(json.loads(raw))


def _encode_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _decode_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _floats_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}d", *vec)


def _blob_to_floats(blob: bytes) -> list[float]:
    n_items = len(blob) // 8
    return list(struct.unpack(f"{n_items}d", blob))


class IndexStore:
    """Persist and query retrieval-oriented knowledge facts."""

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
            DELETE FROM facts;
            DELETE FROM fact_evidence;
            DELETE FROM file_relationships;
            DELETE FROM build_metadata;
            DELETE FROM fact_embeddings;
            DELETE FROM fact_fts;
            """
        )
        self._conn.commit()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                stability TEXT NOT NULL,
                title TEXT NOT NULL,
                guidance TEXT NOT NULL,
                files TEXT NOT NULL,
                subsystems TEXT NOT NULL,
                applicable_intents TEXT NOT NULL,
                support_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                last_seen_at TEXT,
                search_text TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fact_evidence (
                fact_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                label TEXT NOT NULL,
                ref TEXT NOT NULL,
                excerpt TEXT NOT NULL,
                PRIMARY KEY (fact_id, position)
            );

            CREATE TABLE IF NOT EXISTS file_relationships (
                path TEXT NOT NULL,
                related_path TEXT NOT NULL,
                reason TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (path, related_path, reason)
            );

            CREATE TABLE IF NOT EXISTS build_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fact_embeddings (
                fact_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(
                fact_id UNINDEXED,
                title,
                guidance,
                files,
                subsystems,
                search_text
            );
            """
        )
        self._conn.commit()

    def store_build(
        self,
        facts: list[KnowledgeFact],
        relationships: list[tuple[str, str, str, float]],
        metadata: BuildMetadata,
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        """Replace the current index contents with a fresh build."""
        self.reset()

        for fact in facts:
            self._conn.execute(
                """
                INSERT INTO facts (
                    id, kind, stability, title, guidance, files, subsystems,
                    applicable_intents, support_count, confidence, severity,
                    last_seen_at, search_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.id,
                    fact.kind.value,
                    fact.stability.value,
                    fact.title,
                    fact.guidance,
                    _encode_list(fact.files),
                    _encode_list(fact.subsystems),
                    _encode_list([intent.value for intent in fact.applicable_intents]),
                    fact.support_count,
                    fact.confidence,
                    fact.severity.value,
                    _encode_datetime(fact.last_seen_at),
                    fact.search_text,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO fact_fts (fact_id, title, guidance, files, subsystems, search_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.id,
                    fact.title,
                    fact.guidance,
                    " ".join(fact.files),
                    " ".join(fact.subsystems),
                    fact.search_text,
                ),
            )
            for idx, evidence in enumerate(fact.evidence):
                self._conn.execute(
                    """
                    INSERT INTO fact_evidence (
                        fact_id, position, source_type, label, ref, excerpt
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact.id,
                        idx,
                        evidence.source_type,
                        evidence.label,
                        evidence.ref,
                        evidence.excerpt,
                    ),
                )

        for path, related_path, reason, score in relationships:
            self._conn.execute(
                """
                INSERT INTO file_relationships (path, related_path, reason, score)
                VALUES (?, ?, ?, ?)
                """,
                (
                    path,
                    related_path,
                    reason,
                    score,
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
                        "fact_count": metadata.fact_count,
                        "source_coverage": asdict(metadata.source_coverage),
                    }
                ),
            ),
        )

        if embeddings:
            for fact_id, vector in embeddings.items():
                self._conn.execute(
                    "INSERT INTO fact_embeddings (fact_id, vector) VALUES (?, ?)",
                    (fact_id, _floats_to_blob(vector)),
                )

        self._conn.commit()

    def load_facts(self) -> list[KnowledgeFact]:
        rows = self._conn.execute("SELECT * FROM facts ORDER BY id").fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_fact(self, fact_id: str) -> KnowledgeFact | None:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_fact(row)

    def get_related_files(self, path: str, limit: int = 5) -> list[RelatedFile]:
        rows = self._conn.execute(
            """
            SELECT path, related_path, reason, score
            FROM file_relationships
            WHERE path = ?
            ORDER BY score DESC, related_path
            LIMIT ?
            """,
            (path, limit),
        ).fetchall()
        return [
            RelatedFile(
                path=row["related_path"],
                reason=row["reason"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    def search_fts(self, query: str, limit: int = 50) -> dict[str, float]:
        if not query.strip():
            return {}
        rows = self._conn.execute(
            """
            SELECT fact_id, bm25(fact_fts) AS rank
            FROM fact_fts
            WHERE fact_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        scores: dict[str, float] = {}
        for row in rows:
            rank = float(row["rank"])
            scores[str(row["fact_id"])] = 1.0 / (1.0 + max(rank, 0.0))
        return scores

    def get_embeddings(self) -> dict[str, list[float]]:
        rows = self._conn.execute(
            "SELECT fact_id, vector FROM fact_embeddings"
        ).fetchall()
        return {str(row["fact_id"]): _blob_to_floats(row["vector"]) for row in rows}

    def get_build_metadata(self) -> BuildMetadata | None:
        row = self._conn.execute(
            "SELECT value FROM build_metadata WHERE key = ?",
            ("metadata",),
        ).fetchone()
        if row is None:
            return None
        raw = json.loads(row["value"])
        coverage = SourceCoverage(**raw["source_coverage"])
        return BuildMetadata(
            repo_path=raw["repo_path"],
            built_at=datetime.fromisoformat(raw["built_at"]),
            total_commits_analyzed=raw["total_commits_analyzed"],
            fact_count=raw["fact_count"],
            source_coverage=coverage,
        )

    def has_index(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM facts").fetchone()
        return row is not None and int(row["count"]) > 0

    def _row_to_fact(self, row: sqlite3.Row) -> KnowledgeFact:
        evidence_rows = self._conn.execute(
            """
            SELECT source_type, label, ref, excerpt
            FROM fact_evidence
            WHERE fact_id = ?
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
            )
            for e_row in evidence_rows
        ]
        return KnowledgeFact(
            id=str(row["id"]),
            kind=FactKind(str(row["kind"])),
            stability=FactStability(str(row["stability"])),
            title=str(row["title"]),
            guidance=str(row["guidance"]),
            files=_decode_list(str(row["files"])),
            subsystems=_decode_list(str(row["subsystems"])),
            applicable_intents=[
                QueryIntent(item) for item in _decode_list(str(row["applicable_intents"]))
            ],
            support_count=int(row["support_count"]),
            confidence=float(row["confidence"]),
            severity=KnowledgeSeverity(str(row["severity"])),
            last_seen_at=_decode_datetime(row["last_seen_at"]),
            evidence=evidence,
            search_text=str(row["search_text"]),
        )


def bundle_to_json(bundle: ContextBundle) -> str:
    """Serialize a ContextBundle into JSON for CLI and MCP responses."""
    payload = {
        "task": bundle.task,
        "intent": bundle.intent.value,
        "files": bundle.files,
        "summary": bundle.summary,
        "rules": [
            {
                "fact_id": item.fact_id,
                "kind": item.kind.value,
                "title": item.title,
                "guidance": item.guidance,
                "files": item.files,
                "score": item.score,
                "why_selected": item.why_selected,
                "evidence": [asdict(evidence) for evidence in item.evidence],
            }
            for item in bundle.rules
        ],
        "situational": [
            {
                "fact_id": item.fact_id,
                "kind": item.kind.value,
                "title": item.title,
                "guidance": item.guidance,
                "files": item.files,
                "score": item.score,
                "why_selected": item.why_selected,
                "evidence": [asdict(evidence) for evidence in item.evidence],
            }
            for item in bundle.situational
        ],
        "examples": [
            {
                "fact_id": item.fact_id,
                "kind": item.kind.value,
                "title": item.title,
                "guidance": item.guidance,
                "files": item.files,
                "score": item.score,
                "why_selected": item.why_selected,
                "evidence": [asdict(evidence) for evidence in item.evidence],
            }
            for item in bundle.examples
        ],
        "related_files": [asdict(item) for item in bundle.related_files],
        "suggested_tests": bundle.suggested_tests,
        "source_coverage": asdict(bundle.source_coverage),
        "build_metadata": asdict(bundle.build_metadata) if bundle.build_metadata else None,
    }
    return json.dumps(payload, indent=2, default=str)
