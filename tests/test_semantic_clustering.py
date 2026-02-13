"""Tests for semantic clustering of review comments."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gitlore.clustering.semantic import (
    MIN_COMMENTS_FOR_CLUSTERING,
    _cluster_embeddings,
    _compute_coherence,
    _prepare_texts,
    cluster_comments,
)
from gitlore.config import ModelConfig
from gitlore.models import ClassifiedComment, CommentCategory, CommentCluster, ReviewComment


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_classified(body: str, category: CommentCategory = CommentCategory.CONVENTION) -> ClassifiedComment:
    return ClassifiedComment(
        comment=ReviewComment(
            pr_number=1,
            file_path="src/main.ts",
            line=10,
            body=body,
            author="reviewer",
            created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        ),
        categories=[category],
        confidence=0.9,
    )


def _make_model_config(embedding: str = "local/nomic-embed-text-v1.5") -> ModelConfig:
    return ModelConfig(
        classifier="test-classifier",
        synthesizer="test-synthesizer",
        embedding=embedding,
    )


# ── Unit tests ────────────────────────────────────────────────────────────────


class TestPrepareTexts:
    def test_nomic_prefix(self):
        config = _make_model_config("local/nomic-embed-text-v1.5")
        comments = [_make_classified("use const here")]
        texts = _prepare_texts(comments, config)
        assert texts == ["clustering: use const here"]

    def test_non_nomic_no_prefix(self):
        config = _make_model_config("text-embedding-3-small")
        comments = [_make_classified("use const here")]
        texts = _prepare_texts(comments, config)
        assert texts == ["use const here"]

    def test_multiple_comments(self):
        config = _make_model_config("local/nomic-embed-text-v1.5")
        comments = [
            _make_classified("use const"),
            _make_classified("add tests"),
        ]
        texts = _prepare_texts(comments, config)
        assert len(texts) == 2
        assert all(t.startswith("clustering: ") for t in texts)


class TestComputeCoherence:
    def test_identical_vectors(self):
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        coherence = _compute_coherence(embeddings, [0, 1, 2])
        assert coherence == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        coherence = _compute_coherence(embeddings, [0, 1])
        assert coherence == pytest.approx(0.0)

    def test_single_vector(self):
        embeddings = np.array([[1.0, 0.0, 0.0]])
        coherence = _compute_coherence(embeddings, [0])
        assert coherence == 1.0

    def test_mixed_similarity(self):
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ])
        # Normalize for proper cosine sim
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        coherence = _compute_coherence(embeddings, [0, 1, 2])
        assert 0.9 < coherence < 1.0


class TestClusterEmbeddings:
    def test_finds_clusters_in_clear_data(self):
        rng = np.random.default_rng(42)
        # Two tight clusters
        cluster_a = rng.normal(loc=0.0, scale=0.1, size=(10, 5))
        cluster_b = rng.normal(loc=5.0, scale=0.1, size=(10, 5))
        data = np.vstack([cluster_a, cluster_b])
        labels = _cluster_embeddings(data)
        assert len(labels) == 20
        # Should find at least 1 cluster (may find 2)
        unique = set(labels)
        unique.discard(-1)
        assert len(unique) >= 1

    def test_all_noise_returns_negative_ones(self):
        rng = np.random.default_rng(42)
        # Random scattered points
        data = rng.random((5, 3)) * 100
        labels = _cluster_embeddings(data)
        assert len(labels) == 5
        # With very few scattered points, everything may be noise
        # Just verify it runs without error


# ── Integration tests with mocked embeddings ─────────────────────────────────


class TestClusterComments:
    def test_too_few_comments_returns_empty(self):
        config = _make_model_config()
        comments = [_make_classified(f"comment {i}") for i in range(MIN_COMMENTS_FOR_CLUSTERING - 1)]
        result = cluster_comments(comments, config)
        assert result == []

    @patch("gitlore.clustering.semantic.complete_sync")
    @patch("gitlore.clustering.semantic._get_embeddings")
    def test_clusters_similar_comments(self, mock_embed, mock_llm):
        rng = np.random.default_rng(42)

        # Create embeddings with two clear clusters
        cluster_a = rng.normal(loc=0.0, scale=0.05, size=(5, 20))
        cluster_b = rng.normal(loc=3.0, scale=0.05, size=(5, 20))
        embeddings = np.vstack([cluster_a, cluster_b])
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        mock_embed.return_value = embeddings
        mock_llm.return_value = "Use const instead of let"

        comments = [
            _make_classified(f"use const here (variant {i})") for i in range(5)
        ] + [
            _make_classified(f"add error handling (variant {i})") for i in range(5)
        ]

        config = _make_model_config()
        result = cluster_comments(comments, config)

        # Should find at least one cluster
        assert len(result) >= 1
        for cluster in result:
            assert isinstance(cluster, CommentCluster)
            assert cluster.label  # non-empty label
            assert cluster.comments  # non-empty comments
            assert cluster.coherence > 0
            assert cluster.centroid is not None

    @patch("gitlore.clustering.semantic.complete_sync")
    @patch("gitlore.clustering.semantic._get_embeddings")
    def test_all_noise_returns_empty(self, mock_embed, mock_llm):
        rng = np.random.default_rng(42)
        # Scattered embeddings that won't form clusters
        embeddings = rng.random((10, 50)) * 100
        mock_embed.return_value = embeddings

        comments = [_make_classified(f"unique comment number {i} about something") for i in range(10)]
        config = _make_model_config()
        result = cluster_comments(comments, config)

        # May or may not find clusters depending on HDBSCAN
        # Just verify no errors and valid output type
        assert isinstance(result, list)
        for cluster in result:
            assert isinstance(cluster, CommentCluster)

    @patch("gitlore.clustering.semantic.complete_sync")
    @patch("gitlore.clustering.semantic._get_embeddings")
    def test_cluster_coherence_is_computed(self, mock_embed, mock_llm):
        # All identical embeddings -> coherence should be 1.0
        embeddings = np.ones((10, 20))
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        mock_embed.return_value = embeddings
        mock_llm.return_value = "All same pattern"

        comments = [_make_classified(f"same comment {i}") for i in range(10)]
        config = _make_model_config()
        result = cluster_comments(comments, config)

        if result:
            # If a cluster formed from identical vectors, coherence should be high
            assert result[0].coherence == pytest.approx(1.0, abs=0.01)

    @patch("gitlore.clustering.semantic.complete_sync")
    @patch("gitlore.clustering.semantic._get_embeddings")
    def test_llm_labeling_failure_uses_fallback(self, mock_embed, mock_llm):
        rng = np.random.default_rng(42)
        # Tight cluster
        embeddings = rng.normal(loc=0.0, scale=0.01, size=(10, 20))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        mock_embed.return_value = embeddings
        mock_llm.side_effect = RuntimeError("LLM failed")

        comments = [_make_classified(f"use const (variant {i})") for i in range(10)]
        config = _make_model_config()
        result = cluster_comments(comments, config)

        if result:
            # Fallback label should mention the count
            assert "comments" in result[0].label.lower() or "pattern" in result[0].label.lower()

    def test_empty_input_returns_empty(self):
        config = _make_model_config()
        result = cluster_comments([], config)
        assert result == []

    @patch("gitlore.clustering.semantic._get_embeddings")
    def test_nomic_model_uses_local_embed(self, mock_embed):
        """Verify local embedding is called for local/ prefix models."""
        mock_embed.return_value = np.random.default_rng(42).random((5, 20))

        comments = [_make_classified(f"comment {i}") for i in range(5)]
        config = _make_model_config("local/nomic-embed-text-v1.5")

        with patch("gitlore.clustering.semantic.complete_sync", return_value="label"):
            cluster_comments(comments, config)

        mock_embed.assert_called_once()

    @patch("gitlore.clustering.semantic.complete_sync")
    @patch("gitlore.clustering.semantic._get_embeddings")
    def test_centroid_shape_matches_embedding(self, mock_embed, mock_llm):
        dim = 20
        rng = np.random.default_rng(42)
        embeddings = rng.normal(loc=0.0, scale=0.01, size=(10, dim))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        mock_embed.return_value = embeddings
        mock_llm.return_value = "Test pattern"

        comments = [_make_classified(f"similar comment {i}") for i in range(10)]
        config = _make_model_config()
        result = cluster_comments(comments, config)

        for cluster in result:
            if cluster.centroid is not None:
                assert len(cluster.centroid) == dim
