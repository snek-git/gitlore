"""SQLite cache for GitHub comments, LLM classifications, and embeddings."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from gitlore.models import ReviewComment


def _hash_key(model: str, text: str) -> str:
    """SHA-256 hash of model + text for cache keys."""
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()


def _serialize_comments(comments: list[ReviewComment]) -> str:
    """Serialize comments to JSON."""
    items = []
    for c in comments:
        d = asdict(c)
        d["created_at"] = c.created_at.isoformat()
        items.append(d)
    return json.dumps(items)


def _deserialize_comments(data: str) -> list[ReviewComment]:
    """Deserialize comments from JSON."""
    items = json.loads(data)
    comments = []
    for d in items:
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        comments.append(ReviewComment(**d))
    return comments


def _floats_to_blob(vec: list[float]) -> bytes:
    """Pack a float list into a compact binary blob."""
    return struct.pack(f"{len(vec)}d", *vec)


def _blob_to_floats(blob: bytes) -> list[float]:
    """Unpack a binary blob into a float list."""
    n = len(blob) // 8
    return list(struct.unpack(f"{n}d", blob))


class Cache:
    """SQLite cache stored at {repo}/.gitlore/cache.db."""

    def __init__(self, repo_path: str) -> None:
        cache_dir = Path(repo_path) / ".gitlore"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_dir / "cache.db"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS comments (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                fetched_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS classifications (
                key TEXT PRIMARY KEY,
                categories TEXT NOT NULL,
                confidence REAL NOT NULL,
                model TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL
            );
        """)

    def close(self) -> None:
        self._conn.close()

    # ── Comments ──────────────────────────────────────────────────────────────

    def get_comments(
        self, owner: str, repo: str, max_age_hours: float = 24
    ) -> list[ReviewComment] | None:
        key = f"{owner}/{repo}"
        cutoff = time.time() - max_age_hours * 3600
        row = self._conn.execute(
            "SELECT data FROM comments WHERE key = ? AND fetched_at > ?",
            (key, cutoff),
        ).fetchone()
        if row is None:
            return None
        return _deserialize_comments(row[0])

    def set_comments(
        self, owner: str, repo: str, comments: list[ReviewComment]
    ) -> None:
        key = f"{owner}/{repo}"
        data = _serialize_comments(comments)
        self._conn.execute(
            "INSERT OR REPLACE INTO comments (key, data, fetched_at) VALUES (?, ?, ?)",
            (key, data, time.time()),
        )
        self._conn.commit()

    # ── Classifications ───────────────────────────────────────────────────────

    def get_classification(
        self, model: str, body: str
    ) -> tuple[list[str], float] | None:
        key = _hash_key(model, body)
        row = self._conn.execute(
            "SELECT categories, confidence FROM classifications WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), row[1]

    def set_classification(
        self,
        model: str,
        body: str,
        categories: list[str],
        confidence: float,
    ) -> None:
        key = _hash_key(model, body)
        self._conn.execute(
            "INSERT OR REPLACE INTO classifications (key, categories, confidence, model) VALUES (?, ?, ?, ?)",
            (key, json.dumps(categories), confidence, model),
        )
        self._conn.commit()

    # ── Embeddings ────────────────────────────────────────────────────────────

    def get_embeddings(
        self, model: str, texts: list[str]
    ) -> list[list[float]] | None:
        """Get cached embeddings for all texts. Returns None if ANY text is missing."""
        vectors: list[list[float]] = []
        for text in texts:
            key = _hash_key(model, text)
            row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            vectors.append(_blob_to_floats(row[0]))
        return vectors

    def set_embeddings(
        self, model: str, texts: list[str], vectors: list[list[float]]
    ) -> None:
        for text, vec in zip(texts, vectors):
            key = _hash_key(model, text)
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, vector, model) VALUES (?, ?, ?)",
                (key, _floats_to_blob(vec), model),
            )
        self._conn.commit()
