# Semantic Clustering of PR Review Comments

Research into approaches for grouping similar PR review comments to detect repeated feedback patterns (unwritten team conventions).

## 1. Embedding Models

Embedding models convert review comment text into dense vector representations where semantically similar comments are close together in vector space.

### API-Based Models

| Model | Dimensions | Max Tokens | Cost per 1M tokens | Notes |
|-------|-----------|------------|-------------------|-------|
| OpenAI text-embedding-3-small | 1536 | 8191 | $0.02 | Best cost/quality ratio for API models. Supports dimension reduction via `dimensions` param. |
| OpenAI text-embedding-3-large | 3072 | 8191 | $0.13 | Higher quality, 6.5x more expensive. Also supports native dimension reduction. |
| Cohere embed-v3 (english) | 1024 | 512 | ~$0.10 | Good quality. Cohere now pushes embed-v4 ($0.12/1M) with 1536 dims. |

### Open Source / Local Models

| Model | Dimensions | Params | Speed (sent/sec) | MTEB Score | Notes |
|-------|-----------|--------|------------------|-----------|-------|
| all-MiniLM-L6-v2 | 384 | 22.7M | ~14,200 | Good but dated | Fastest option, 22MB, max 256 tokens. Architecture from 2019; outperformed by modern models on retrieval benchmarks. Still viable for clustering short text. |
| nomic-embed-text-v1.5 | 768 | 137M | ~2,000 | 81.2% avg | Good balance of quality and size. Requires task prefix (`search_document:`, `clustering:`, etc). Supports Matryoshka dimensions (768/512/256/128). |
| BGE-large-en-v1.5 | 1024 | 335M | ~800 | Strong | High quality, but large. Produced by BAAI. Good STS performance. |
| E5-large-v2 | 1024 | 335M | ~800 | Strong | Trained on 270M text pairs. E5-small achieved 100% Top-5 accuracy on some benchmarks despite smaller size. |
| GTE-large | 1024 | 335M | ~800 | Strong | From Alibaba DAMO. Competitive with BGE and E5. |

### Short Text Performance

Review comments are typically 1-5 sentences (10-80 tokens). For text this short:

- **all-MiniLM-L6-v2** was optimized for 128-256 token inputs, making it well-suited for short text despite its age.
- **nomic-embed-text** with the `clustering:` prefix is designed to hint the model toward clustering-friendly representations.
- **E5 models** perform surprisingly well on short text; e5-small achieves best-in-class accuracy despite its compact size.
- API models (OpenAI) tend to handle short text well due to their massive training data.

### Recommendation

**Primary: nomic-embed-text-v1.5** (local, free, good quality, supports task prefixes for clustering).
**Fallback: OpenAI text-embedding-3-small** (cheap enough for any scale, higher quality, requires API key).
**Budget/speed: all-MiniLM-L6-v2** (fastest local option, tiny model, good enough for prototyping).

### LiteLLM for Embeddings

LiteLLM supports embeddings via `litellm.embedding()` with the same unified interface it provides for chat completions. Supported providers include OpenAI, Azure, Cohere, Bedrock, HuggingFace, and Vertex AI. It follows the OpenAI `/v1/embeddings` spec, making it easy to swap providers. OpenRouter also exposes an OpenAI-compatible `/v1/embeddings` endpoint supporting text-embedding-3-small/large and Qwen3 embedding models.

Using litellm for embeddings:

```python
import litellm

response = litellm.embedding(
    model="text-embedding-3-small",
    input=["use const instead of let here", "prefer const over let"],
)
embeddings = [item["embedding"] for item in response.data]
```

For local models, sentence-transformers is more appropriate than litellm:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
embeddings = model.encode(
    ["clustering: use const instead of let here",
     "clustering: prefer const over let"],
    normalize_embeddings=True,
)
```

## 2. Clustering Algorithms

### Algorithm Comparison

| Algorithm | Requires k? | Handles noise? | Variable density? | Complexity | Best for |
|-----------|------------|----------------|-------------------|-----------|----------|
| HDBSCAN | No (uses min_cluster_size) | Yes (labels outliers as -1) | Yes | O(n log n) | Unknown cluster count, variable density |
| DBSCAN | No (uses eps + min_samples) | Yes | No (single global density) | O(n log n) | Known density, uniform clusters |
| Agglomerative | Optional (can cut dendrogram) | No (assigns everything) | No | O(n^2 log n) | Exploring granularity, cosine distance matrix |
| K-Means | Yes | No | No | O(n * k * d) | Known cluster count, spherical clusters |

### Recommendation: HDBSCAN

For our use case (unknown number of convention patterns, variable cluster sizes, noisy data):

**HDBSCAN is the clear winner** because:

1. We don't know how many patterns exist in advance.
2. Not every comment represents a convention -- many are one-off feedback. HDBSCAN naturally identifies these as noise (label -1) rather than forcing them into clusters.
3. Some conventions appear frequently (e.g., "use const") while others are rare (e.g., specific error handling patterns). HDBSCAN handles variable-density clusters.
4. It has fewer hyperparameters than DBSCAN. The main one is `min_cluster_size` -- the minimum number of similar comments needed to constitute a "pattern." For convention detection, `min_cluster_size=3` is a sensible default (a pattern seen at least 3 times).

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,       # minimum comments to form a pattern
    min_samples=2,            # conservative core point threshold
    metric="euclidean",       # on UMAP-reduced embeddings
    cluster_selection_method="eom",  # excess of mass (better for varied sizes)
)
labels = clusterer.fit_predict(reduced_embeddings)
```

**Agglomerative clustering** is a good secondary option, particularly because it accepts precomputed cosine similarity matrices directly and lets you explore different granularity levels by cutting the dendrogram at various heights. Useful for interactive exploration.

### HDBSCAN with scikit-learn

As of scikit-learn 1.3+, HDBSCAN is available natively in `sklearn.cluster.HDBSCAN`, so the separate `hdbscan` package is no longer strictly required.

## 3. Dimensionality Reduction

### Why Reduce Dimensions?

Embedding vectors are high-dimensional (384-3072 dims). Clustering algorithms like HDBSCAN struggle in high-dimensional spaces due to the "curse of dimensionality" where distance metrics become less meaningful. Reducing to 5-50 dimensions before clustering significantly improves results.

### UMAP (Recommended)

UMAP (Uniform Manifold Approximation and Projection) is the standard choice for pre-clustering dimensionality reduction:

- **Non-linear**: captures manifold structure that PCA misses.
- **Preserves both local and global structure**: nearby points stay nearby, distant points stay distant.
- **Configurable**: `n_neighbors` controls local vs global balance, `n_components` sets target dimensions.
- **Fast**: much faster than t-SNE, scales to 100k+ points.

```python
import umap

reducer = umap.UMAP(
    n_components=10,     # reduce to 10 dims for clustering (not 2)
    n_neighbors=15,      # default, balances local/global
    min_dist=0.0,        # tight clusters for HDBSCAN
    metric="cosine",     # appropriate for text embeddings
)
reduced = reducer.fit_transform(embeddings)
```

For **visualization** (2D plots), run a separate UMAP with `n_components=2` and `min_dist=0.1`.

### PCA

PCA is linear and fast but misses non-linear structure in text embeddings. However, it's useful as a **preprocessing step before UMAP** when starting from very high dimensions (>500):

```python
from sklearn.decomposition import PCA

# First PCA to reduce 3072 -> 100, then UMAP to reduce 100 -> 10
pca = PCA(n_components=100)
pca_reduced = pca.fit_transform(embeddings)
umap_reduced = reducer.fit_transform(pca_reduced)
```

### t-SNE

t-SNE preserves local structure well but: is slower than UMAP, doesn't preserve global structure, is non-parametric (can't transform new points), and produces 2D/3D only. **Use UMAP instead** for both clustering and visualization.

### Recommendation

PCA (to ~100 dims if starting >500) -> UMAP (to 5-15 dims for clustering). Skip PCA if embeddings are already <500 dimensions (e.g., all-MiniLM-L6-v2 at 384 dims).

## 4. Similarity Thresholds

### Cosine Similarity Ranges for Short Text

Based on research with modern embedding models:

| Cosine Similarity | Interpretation | Example |
|------------------|---------------|---------|
| 0.90+ | Near-identical meaning | "use const" vs "prefer const over let" |
| 0.80-0.90 | Same concept, different wording | "add error handling" vs "this needs a try-catch" |
| 0.70-0.80 | Related topic | "handle the null case" vs "add a null check here" |
| 0.60-0.70 | Loosely related | "fix naming" vs "use camelCase for variables" |
| <0.60 | Different topics | "add tests" vs "fix the CSS" |

### Recommended Threshold for Convention Detection

For identifying "same feedback pattern": **0.75-0.80** as a starting point. This catches comments about the same convention while avoiding false positives from merely related comments.

However, with HDBSCAN we don't use a hard similarity threshold directly -- the algorithm determines clusters based on density. The threshold becomes relevant when:

1. **Post-clustering validation**: check that intra-cluster average similarity is above 0.75.
2. **Assigning new comments to existing clusters**: use 0.75 cosine similarity to the cluster centroid.
3. **Merging clusters**: if two cluster centroids have >0.85 similarity, consider merging.

### Important Caveat

Thresholds vary significantly by embedding model. OpenAI embeddings tend to produce higher similarity scores than sentence-transformers models. Always calibrate thresholds on a sample of your actual data.

## 5. Cluster Quality Evaluation

### Automated Metrics

**Silhouette Score**: Measures how similar each point is to its own cluster vs the nearest other cluster. Ranges from -1 to +1.

- \> 0.7: strong clustering
- \> 0.5: reasonable
- \> 0.25: weak
- Note: skip noise points (HDBSCAN label -1) when computing silhouette.

```python
from sklearn.metrics import silhouette_score

# Exclude noise points
mask = labels != -1
if mask.sum() > 1:
    score = silhouette_score(reduced[mask], labels[mask])
```

**Cluster coherence**: average pairwise cosine similarity within each cluster. Clusters representing real conventions should have coherence > 0.75.

**Noise ratio**: what fraction of comments are labeled as noise. If >80% are noise, `min_cluster_size` might be too high. If <20%, it might be too low.

### Manual Inspection

Sample 3-5 comments from each cluster and verify they represent the same convention. This is essential -- automated metrics can't fully capture whether comments represent the "same feedback."

### Automatic Cluster Labeling with LLM

After clustering, use an LLM to generate a human-readable label for each cluster:

```python
def label_cluster(comments: list[str], llm) -> str:
    sample = comments[:10]  # take up to 10 representative comments
    prompt = f"""These PR review comments were grouped together because they express
similar feedback. What coding convention or pattern do they collectively represent?
Respond with a short label (5-15 words).

Comments:
{chr(10).join(f'- {c}' for c in sample)}"""
    return llm.complete(prompt)
```

This is where the LLM-based approach adds genuine value -- algorithmic clustering finds the groups, but an LLM interprets what the group means in human terms.

## 6. Handling Short Text

PR review comments present specific challenges for embeddings:

### Problem

Short comments like "use const here", "missing error handling", or "nit: naming" lack context. Embeddings for very short text can be noisy because there's little signal to encode.

### Strategies

**1. Task prefix (best for compatible models)**

Models like nomic-embed-text and E5 support task-specific prefixes:

```python
# nomic-embed-text
texts = [f"clustering: {comment}" for comment in comments]

# E5 models
texts = [f"query: {comment}" for comment in comments]
```

**2. Context augmentation with file path**

Append the file path or language context to improve disambiguation:

```python
# "use const here" in a JS file vs a C++ file have different meanings
enriched = f"{comment} [file: {file_path}]"
```

This helps distinguish "use const" in JavaScript (variable declaration) from "use const" in C++ (const correctness), but may over-separate comments that are truly about the same convention.

**3. LLM expansion (expensive but effective)**

Use an LLM to expand terse comments before embedding:

```python
expanded = llm.complete(
    f"Expand this PR review comment into a full sentence explaining "
    f"the coding convention it refers to: '{comment}'"
)
```

This turns "nit: naming" into "The variable naming doesn't follow our team's convention of using camelCase." However, this adds an LLM call per comment and risks hallucination.

**4. Concat with diff context**

If available, concatenate the comment with the code diff it refers to:

```python
enriched = f"Review comment on code `{diff_hunk[:200]}`: {comment}"
```

### Recommendation

Start with **task prefix** (free, improves quality) and **file path context** (cheap, helps disambiguation). Only add LLM expansion if clustering quality is poor on initial results.

## 7. Practical Pipeline

### Full Pipeline

```
Extract comments -> Enrich text -> Embed -> [PCA] -> UMAP -> HDBSCAN -> Label clusters -> Output
```

### Implementation Sketch

```python
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import numpy as np

# 1. Load and enrich comments
comments = load_review_comments()  # list of CommentRecord
texts = [f"clustering: {c.body}" for c in comments]

# 2. Embed
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

# 3. Reduce dimensions
reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine")
reduced = reducer.fit_transform(embeddings)

# 4. Cluster
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, cluster_selection_method="eom")
labels = clusterer.fit_predict(reduced)

# 5. Group and label
clusters = {}
for i, label in enumerate(labels):
    if label == -1:
        continue  # noise
    clusters.setdefault(label, []).append(comments[i])

# 6. Label each cluster with LLM
for label, cluster_comments in clusters.items():
    convention_name = llm_label_cluster([c.body for c in cluster_comments])
    print(f"Convention: {convention_name} ({len(cluster_comments)} occurrences)")
```

### Performance at Scale

| Comments | Embedding (local) | UMAP | HDBSCAN | Total |
|----------|-------------------|------|---------|-------|
| 1,000 | ~5s | <1s | <1s | ~6s |
| 10,000 | ~50s | ~5s | ~2s | ~60s |
| 50,000 | ~4min | ~30s | ~10s | ~5min |
| 100,000 | ~8min | ~1min | ~30s | ~10min |

Embedding is the bottleneck. GPU acceleration (if available) reduces embedding time by 5-10x. API-based embedding can parallelize with batching.

Memory: 10k comments * 768 dims * 4 bytes = ~30MB for embeddings. Not a concern even at 100k.

### Incremental Clustering

For adding new comments without re-clustering everything:

**Approach 1: Nearest-centroid assignment**

Compute centroid of each existing cluster. For new comments, embed and assign to the nearest centroid if similarity > threshold, otherwise label as unassigned. Periodically re-cluster the full dataset.

```python
centroids = {label: np.mean(embeddings[labels == label], axis=0) for label in set(labels) if label != -1}

def assign_new_comment(new_embedding, centroids, threshold=0.75):
    best_label, best_sim = -1, 0
    for label, centroid in centroids.items():
        sim = np.dot(new_embedding, centroid)  # cosine sim (normalized vecs)
        if sim > best_sim:
            best_label, best_sim = label, sim
    return best_label if best_sim >= threshold else -1
```

**Approach 2: IncrementalDBSCAN**

The `incdbscan` package provides an incremental variant of DBSCAN that updates clusters when points are inserted/deleted. However, HDBSCAN has no widely-used incremental variant.

**Approach 3: Periodic re-clustering**

For most gitlore use cases, the comment corpus grows slowly (tens to hundreds per week). Just re-cluster the entire dataset on each analysis run. At 10k comments this takes ~60 seconds, which is acceptable.

**Recommendation**: Use periodic full re-clustering unless the corpus exceeds 50k comments, at which point nearest-centroid assignment with periodic full re-clustering (e.g., weekly) becomes worthwhile.

## 8. Cost Analysis

### Embedding 10,000 Comments

Assumptions: average comment is 30 tokens.

| Model | Location | Cost | Time | Quality |
|-------|----------|------|------|---------|
| all-MiniLM-L6-v2 | Local (CPU) | $0 | ~50s | Good |
| nomic-embed-text-v1.5 | Local (CPU) | $0 | ~2min | Better |
| nomic-embed-text-v1.5 | Local (GPU) | $0 | ~20s | Better |
| text-embedding-3-small | OpenAI API | $0.006 | ~10s | Best |
| text-embedding-3-large | OpenAI API | $0.039 | ~10s | Best+ |
| Cohere embed-v4 | Cohere API | $0.036 | ~10s | Best |

10k comments at 30 tokens each = 300k tokens. Even the most expensive option (text-embedding-3-large) costs under $0.04.

### Local vs API Tradeoff

**Local (sentence-transformers)**:
- Free, private, no rate limits
- Requires Python + PyTorch install (~2GB disk)
- Slower on CPU, competitive with GPU
- Models are smaller (22MB - 1.3GB)
- Good for: privacy-sensitive repos, offline usage, CI pipelines

**API (OpenAI/Cohere)**:
- Negligible cost (<$0.04 per 10k comments)
- No local compute needed
- Fast with batching
- Requires API key and network
- Good for: simplicity, highest quality, when local GPU isn't available

**Recommendation**: Default to local (nomic-embed-text-v1.5) since it's free and quality is sufficient. Offer API option for users who want maximum quality or don't want local model downloads.

## 9. LLM-Based Clustering (Without Embeddings)

### Approach

Instead of embed -> cluster, send batches of comments to an LLM and ask it to group them:

```python
prompt = """Group these PR review comments by the coding convention they enforce.
Return groups as JSON: {"groups": [{"convention": "...", "comment_indices": [...]}]}

Comments:
{numbered_comments}"""
```

### When This Makes Sense

- **Small datasets (<200 comments)**: LLM can process them in a few batches. Cost and quality are acceptable.
- **Exploration phase**: quickly see what kinds of patterns exist before building a full pipeline.
- **Hybrid approach**: use LLM to label and validate clusters found by embedding-based methods.

### When Embedding + Algorithmic Clustering Is Better

- **Scale (>500 comments)**: LLM context windows can't hold all comments at once. Batching introduces inconsistencies across batches (the same convention might get different names in different batches).
- **Consistency**: algorithmic clustering is deterministic. LLM-based grouping varies between runs.
- **Cost**: embedding 10k comments costs $0.006 (text-embedding-3-small). Sending 10k comments to an LLM for grouping would cost $1-5+ depending on model and prompting strategy.
- **Incremental updates**: embeddings are stable -- adding new comments doesn't change existing cluster assignments. LLM-based re-grouping might reorganize everything.

### Hybrid Strategy (Recommended)

Use embedding + HDBSCAN for the actual clustering, then use an LLM for two specific tasks:

1. **Cluster labeling**: given a cluster of similar comments, generate a human-readable convention name.
2. **Cluster validation**: ask the LLM "do these comments all refer to the same coding convention?" as a quality check.

This gets the best of both worlds: scalable, consistent clustering with human-interpretable output.

## 10. Summary of Recommendations

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Embedding model | nomic-embed-text-v1.5 (local) or text-embedding-3-small (API) | Best quality/cost for short text |
| Clustering | HDBSCAN (min_cluster_size=3) | No k needed, handles noise, variable density |
| Dim reduction | UMAP (n_components=10, metric=cosine) | Non-linear, preserves structure, fast |
| Short text handling | Task prefix + optional file path context | Free, meaningful quality improvement |
| Cluster labeling | LLM summarization of representative samples | Human-interpretable output |
| Evaluation | Silhouette score + manual inspection + noise ratio | Complementary automated and manual checks |
| Incremental | Full re-clustering until >50k comments | Simple, fast enough at expected scale |

### Key Dependencies

```
sentence-transformers   # local embedding models
umap-learn              # dimensionality reduction
hdbscan                 # clustering (or sklearn>=1.3)
scikit-learn            # silhouette score, PCA, agglomerative
numpy                   # vector operations
litellm                 # API-based embeddings (optional)
```
