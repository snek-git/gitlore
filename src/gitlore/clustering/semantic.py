"""Semantic clustering of classified PR review comments."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from gitlore.config import ModelConfig
from gitlore.models import ClassifiedComment, CommentCluster
from gitlore.utils.llm import complete_sync

TYPE_CHECKING = False
if TYPE_CHECKING:
    from gitlore.cache import Cache

logger = logging.getLogger(__name__)

MIN_COMMENTS_FOR_CLUSTERING = 5


def _get_embeddings(
    texts: list[str], model_config: ModelConfig, cache: Cache | None = None,
) -> NDArray[np.float64]:
    """Get embeddings using litellm API, with optional cache."""
    import asyncio

    from gitlore.utils.llm import embed

    embedding_model = model_config.embedding
    if embedding_model.startswith("local/"):
        embedding_model = embedding_model[len("local/"):]

    if cache is not None:
        cached = cache.get_embeddings(embedding_model, texts)
        if cached is not None:
            return np.array(cached, dtype=np.float64)

    embeddings = asyncio.run(embed(embedding_model, texts))

    if cache is not None:
        cache.set_embeddings(embedding_model, texts, embeddings)

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
    min_cluster_size = max(5, n_samples // 100)
    min_samples = max(2, min_cluster_size // 3)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
        cluster_selection_epsilon=0.3,
        cluster_selection_method="leaf",
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
    n = sim_matrix.shape[0]
    upper_sum = float(np.sum(np.triu(sim_matrix, k=1)))
    n_pairs = n * (n - 1) / 2
    return upper_sum / n_pairs


def _format_comment_for_summary(cc: ClassifiedComment) -> str:
    """Format a single classified comment with its context for the summarizer."""
    c = cc.comment
    parts = [f"[PR #{c.pr_number}]"]
    if c.file_path:
        parts.append(f"[{c.file_path}")
        if c.line:
            parts[-1] += f":{c.line}"
        parts[-1] += "]"
    if c.review_state:
        parts.append(f"[{c.review_state}]")
    cats = [cat.value for cat in cc.categories]
    if cats:
        parts.append(f"({', '.join(cats)})")
    parts.append(f"\n{c.body}")
    if c.diff_context:
        parts.append(f"\nDiff context:\n{c.diff_context[:400]}")
    if c.thread_comments:
        for reply in c.thread_comments[:3]:
            parts.append(f"\nReply: {reply[:300]}")
    return " ".join(parts[:3]) + "".join(parts[3:])


def _summarize_cluster(
    comments: list[ClassifiedComment],
    model: str,
) -> str:
    """Summarize a cluster of review comments into a recurring theme description."""
    formatted = "\n\n---\n\n".join(
        _format_comment_for_summary(c) for c in comments
    )

    system = (
        "You analyze groups of PR review comments from the same repository that were "
        "clustered together because they express similar concerns. You receive the full "
        "comment text, diff context, and thread replies.\n\n"
        "Produce a short summary (2-4 sentences) of the recurring reviewer concern this "
        "cluster represents. Focus on what reviewers consistently care about and why. "
        "Be specific to this codebase, not generic.\n\n"
        "First line: a short label (5-15 words).\n"
        "Then: the summary."
    )
    user = (
        f"This cluster contains {len(comments)} review comments. "
        f"Summarize the recurring concern.\n\n{formatted}"
    )

    try:
        result = complete_sync(
            model=model,
            system=system,
            user=user,
            temperature=0.0,
        )
        return result.strip()
    except Exception:
        logger.warning("Failed to summarize cluster, using fallback")
        return f"Pattern ({len(comments)} comments)"


# Progress callback type: (stage, completed, total)
# stage is "embed", "cluster", or "summarize"
ClusterProgressFn = Callable[[str, int, int], None]


def cluster_comments(
    classified: list[ClassifiedComment],
    model_config: ModelConfig,
    cache: Cache | None = None,
    on_progress: ClusterProgressFn | None = None,
) -> list[CommentCluster]:
    """Cluster classified comments by semantic similarity.

    Pipeline: embed -> HDBSCAN cluster -> LLM summarize per cluster.
    """
    if len(classified) < MIN_COMMENTS_FOR_CLUSTERING:
        return []

    def _progress(stage: str, done: int, total: int) -> None:
        if on_progress is not None:
            on_progress(stage, done, total)

    # 1. Embed
    _progress("embed", 0, len(classified))
    texts = _prepare_texts(classified, model_config)
    embeddings = _get_embeddings(texts, model_config, cache)
    _progress("embed", len(classified), len(classified))

    # 2. Cluster
    _progress("cluster", 0, 1)
    labels = _cluster_embeddings(embeddings)
    _progress("cluster", 1, 1)

    # 3. Group by cluster, skip noise
    cluster_indices: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        cluster_indices.setdefault(int(label), []).append(i)

    if not cluster_indices:
        return []

    # 4. Summarize each cluster
    total_clusters = len(cluster_indices)
    clusters: list[CommentCluster] = []
    for idx, (cluster_id, indices) in enumerate(sorted(cluster_indices.items())):
        _progress("summarize", idx, total_clusters)

        cluster_comments_list = [classified[i] for i in indices]
        coherence = _compute_coherence(embeddings, indices)
        centroid = np.mean(embeddings[indices], axis=0).tolist()
        label = _summarize_cluster(cluster_comments_list, model_config.classifier)

        clusters.append(
            CommentCluster(
                cluster_id=cluster_id,
                label=label,
                comments=cluster_comments_list,
                centroid=centroid,
                coherence=coherence,
            )
        )

    _progress("summarize", total_clusters, total_clusters)
    return clusters
