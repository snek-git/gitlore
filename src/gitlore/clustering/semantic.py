"""Semantic clustering of classified PR review comments."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from gitlore.config import ModelConfig
from gitlore.models import ClassifiedComment, CommentCluster
from gitlore.prompts import load as load_prompt
from gitlore.utils.llm import complete_sync

logger = logging.getLogger(__name__)

# Minimum comments needed to attempt clustering
MIN_COMMENTS_FOR_CLUSTERING = 5


def _get_embeddings(
    texts: list[str], model_config: ModelConfig
) -> NDArray[np.float64]:
    """Get embeddings using litellm API."""
    import asyncio

    from gitlore.utils.llm import embed

    embedding_model = model_config.embedding
    # Strip "local/" prefix if present (legacy config compat)
    if embedding_model.startswith("local/"):
        embedding_model = embedding_model[len("local/") :]

    embeddings = asyncio.run(embed(embedding_model, texts))
    return np.array(embeddings, dtype=np.float64)


def _prepare_texts(comments: list[ClassifiedComment], model_config: ModelConfig) -> list[str]:
    """Prepare comment texts for embedding, adding task prefix if needed."""
    embedding_model = model_config.embedding.lower()
    use_prefix = "nomic" in embedding_model

    texts: list[str] = []
    for c in comments:
        text = c.comment.body
        if use_prefix:
            text = f"clustering: {text}"
        texts.append(text)

    return texts


def _cluster_embeddings(embeddings: NDArray[np.float64]) -> NDArray[np.int64]:
    """Run HDBSCAN clustering directly on embeddings with cosine metric."""
    n_samples = embeddings.shape[0]
    min_cluster_size = min(3, n_samples)
    min_samples = min(2, n_samples)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
        cluster_selection_method="eom",
    )
    labels: NDArray[np.int64] = clusterer.fit_predict(embeddings)
    return labels


def _compute_coherence(
    embeddings: NDArray[np.float64], indices: list[int]
) -> float:
    """Compute average pairwise cosine similarity for a cluster."""
    if len(indices) < 2:
        return 1.0
    cluster_embeddings = embeddings[indices]
    sim_matrix = cosine_similarity(cluster_embeddings)
    # Average of upper triangle (excluding diagonal)
    n = sim_matrix.shape[0]
    upper_sum = float(np.sum(np.triu(sim_matrix, k=1)))
    n_pairs = n * (n - 1) / 2
    return upper_sum / n_pairs


def _label_cluster(
    comments: list[ClassifiedComment],
    model: str,
) -> str:
    """Use an LLM to generate a human-readable label for a cluster."""
    sample = comments[:10]
    bodies = "\n".join(f"- {c.comment.body}" for c in sample)

    system = load_prompt("cluster_label_system")
    user = load_prompt("cluster_label_user").format(bodies=bodies)

    try:
        label = complete_sync(
            model=model,
            system=system,
            user=user,
            temperature=0.0,
            max_tokens=50,
        )
        return label.strip().strip('"').strip("'")
    except Exception:
        logger.warning("Failed to label cluster, using fallback")
        return f"Pattern ({len(comments)} comments)"


def cluster_comments(
    classified: list[ClassifiedComment],
    model_config: ModelConfig,
) -> list[CommentCluster]:
    """Cluster classified comments by semantic similarity.

    Pipeline: embed -> HDBSCAN cluster (cosine metric) -> LLM label.

    Args:
        classified: List of classified review comments.
        model_config: Model configuration for embedding and labeling.

    Returns:
        List of CommentCluster with labels, coherence scores, and centroids.
    """
    if len(classified) < MIN_COMMENTS_FOR_CLUSTERING:
        logger.info(
            "Too few comments for clustering (%d < %d)",
            len(classified),
            MIN_COMMENTS_FOR_CLUSTERING,
        )
        return []

    # 1. Prepare texts and embed
    texts = _prepare_texts(classified, model_config)
    embeddings = _get_embeddings(texts, model_config)

    # 2. Cluster (HDBSCAN with cosine metric directly on embeddings)
    labels = _cluster_embeddings(embeddings)

    # 4. Group comments by cluster, skip noise (label == -1)
    cluster_indices: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        cluster_indices.setdefault(int(label), []).append(i)

    if not cluster_indices:
        logger.info("No clusters found (all comments classified as noise)")
        return []

    noise_count = int(np.sum(labels == -1))
    logger.info(
        "Found %d clusters, %d noise comments out of %d total",
        len(cluster_indices),
        noise_count,
        len(classified),
    )

    # 5. Build CommentCluster objects with coherence and LLM labels
    clusters: list[CommentCluster] = []
    for cluster_id, indices in sorted(cluster_indices.items()):
        cluster_comments_list = [classified[i] for i in indices]
        coherence = _compute_coherence(embeddings, indices)
        centroid = np.mean(embeddings[indices], axis=0).tolist()

        label = _label_cluster(cluster_comments_list, model_config.classifier)

        clusters.append(
            CommentCluster(
                cluster_id=cluster_id,
                label=label,
                comments=cluster_comments_list,
                centroid=centroid,
                coherence=coherence,
            )
        )

    return clusters
