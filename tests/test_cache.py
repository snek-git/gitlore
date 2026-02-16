"""Tests for SQLite cache module."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from gitlore.cache import Cache, _blob_to_floats, _floats_to_blob, _hash_key
from gitlore.models import ReviewComment


@pytest.fixture
def cache(tmp_path):
    return Cache(str(tmp_path))


def _make_comment(body: str = "test comment", pr: int = 1) -> ReviewComment:
    return ReviewComment(
        pr_number=pr,
        file_path="src/main.py",
        line=10,
        body=body,
        author="reviewer",
        created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


# ── Helper tests ─────────────────────────────────────────────────────────────


class TestHelpers:
    def test_hash_key_deterministic(self):
        assert _hash_key("model", "text") == _hash_key("model", "text")

    def test_hash_key_different_inputs(self):
        assert _hash_key("model_a", "text") != _hash_key("model_b", "text")
        assert _hash_key("model", "text_a") != _hash_key("model", "text_b")

    def test_float_blob_roundtrip(self):
        vec = [1.0, 2.5, -3.7, 0.0, 1e-10]
        assert _blob_to_floats(_floats_to_blob(vec)) == vec

    def test_empty_float_blob(self):
        assert _blob_to_floats(_floats_to_blob([])) == []


# ── Comment cache ────────────────────────────────────────────────────────────


class TestCommentCache:
    def test_set_and_get_comments(self, cache):
        comments = [_make_comment("first"), _make_comment("second", pr=2)]
        cache.set_comments("owner", "repo", comments)
        result = cache.get_comments("owner", "repo")
        assert result is not None
        assert len(result) == 2
        assert result[0].body == "first"
        assert result[1].body == "second"
        assert result[1].pr_number == 2

    def test_get_missing_returns_none(self, cache):
        assert cache.get_comments("owner", "repo") is None

    def test_ttl_expired_returns_none(self, cache):
        comments = [_make_comment()]
        cache.set_comments("owner", "repo", comments)
        # Manually set fetched_at to the past
        cache._conn.execute(
            "UPDATE comments SET fetched_at = ? WHERE key = ?",
            (time.time() - 25 * 3600, "owner/repo"),
        )
        cache._conn.commit()
        assert cache.get_comments("owner", "repo", max_age_hours=24) is None

    def test_ttl_not_expired(self, cache):
        comments = [_make_comment()]
        cache.set_comments("owner", "repo", comments)
        result = cache.get_comments("owner", "repo", max_age_hours=24)
        assert result is not None
        assert len(result) == 1

    def test_overwrite_comments(self, cache):
        cache.set_comments("owner", "repo", [_make_comment("old")])
        cache.set_comments("owner", "repo", [_make_comment("new")])
        result = cache.get_comments("owner", "repo")
        assert result is not None
        assert len(result) == 1
        assert result[0].body == "new"

    def test_different_repos_independent(self, cache):
        cache.set_comments("owner", "repo1", [_make_comment("repo1")])
        cache.set_comments("owner", "repo2", [_make_comment("repo2")])
        r1 = cache.get_comments("owner", "repo1")
        r2 = cache.get_comments("owner", "repo2")
        assert r1 is not None and r1[0].body == "repo1"
        assert r2 is not None and r2[0].body == "repo2"

    def test_comment_fields_roundtrip(self, cache):
        comment = ReviewComment(
            pr_number=42,
            file_path="src/foo.py",
            line=100,
            body="test body",
            author="alice",
            created_at=datetime(2026, 3, 1, 8, 30, 0, tzinfo=timezone.utc),
            is_resolved=True,
            diff_context="@@ -1,3 +1,4 @@",
            thread_comments=["reply 1", "reply 2"],
            review_state="APPROVED",
        )
        cache.set_comments("o", "r", [comment])
        result = cache.get_comments("o", "r")
        assert result is not None
        c = result[0]
        assert c.pr_number == 42
        assert c.file_path == "src/foo.py"
        assert c.line == 100
        assert c.author == "alice"
        assert c.is_resolved is True
        assert c.diff_context == "@@ -1,3 +1,4 @@"
        assert c.thread_comments == ["reply 1", "reply 2"]
        assert c.review_state == "APPROVED"
        assert c.created_at == datetime(2026, 3, 1, 8, 30, 0, tzinfo=timezone.utc)


# ── Classification cache ─────────────────────────────────────────────────────


class TestClassificationCache:
    def test_set_and_get(self, cache):
        cache.set_classification("model", "comment body", ["bug", "security"], 0.95)
        result = cache.get_classification("model", "comment body")
        assert result is not None
        cats, conf = result
        assert cats == ["bug", "security"]
        assert conf == pytest.approx(0.95)

    def test_get_missing_returns_none(self, cache):
        assert cache.get_classification("model", "unknown") is None

    def test_different_models_independent(self, cache):
        cache.set_classification("model_a", "body", ["bug"], 0.9)
        cache.set_classification("model_b", "body", ["nitpick"], 0.7)
        a = cache.get_classification("model_a", "body")
        b = cache.get_classification("model_b", "body")
        assert a is not None and a[0] == ["bug"]
        assert b is not None and b[0] == ["nitpick"]

    def test_overwrite_classification(self, cache):
        cache.set_classification("model", "body", ["bug"], 0.5)
        cache.set_classification("model", "body", ["architecture"], 0.9)
        result = cache.get_classification("model", "body")
        assert result is not None
        assert result[0] == ["architecture"]
        assert result[1] == pytest.approx(0.9)


# ── Embedding cache ──────────────────────────────────────────────────────────


class TestEmbeddingCache:
    def test_set_and_get(self, cache):
        texts = ["hello", "world"]
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        cache.set_embeddings("model", texts, vectors)
        result = cache.get_embeddings("model", texts)
        assert result is not None
        assert len(result) == 2
        assert result[0] == [1.0, 2.0, 3.0]
        assert result[1] == [4.0, 5.0, 6.0]

    def test_get_missing_returns_none(self, cache):
        assert cache.get_embeddings("model", ["unknown"]) is None

    def test_partial_miss_returns_none(self, cache):
        cache.set_embeddings("model", ["hello"], [[1.0, 2.0]])
        # Asking for hello + world, but world isn't cached
        assert cache.get_embeddings("model", ["hello", "world"]) is None

    def test_different_models_independent(self, cache):
        cache.set_embeddings("model_a", ["text"], [[1.0]])
        cache.set_embeddings("model_b", ["text"], [[2.0]])
        a = cache.get_embeddings("model_a", ["text"])
        b = cache.get_embeddings("model_b", ["text"])
        assert a is not None and a[0] == [1.0]
        assert b is not None and b[0] == [2.0]

    def test_empty_texts_returns_empty(self, cache):
        result = cache.get_embeddings("model", [])
        assert result is not None
        assert result == []

    def test_high_dimensional_vectors(self, cache):
        vec = list(range(1536))  # typical embedding dimension
        cache.set_embeddings("model", ["text"], [vec])
        result = cache.get_embeddings("model", ["text"])
        assert result is not None
        assert result[0] == [float(x) for x in range(1536)]
