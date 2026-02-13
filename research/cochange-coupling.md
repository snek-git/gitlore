# Co-Change Coupling Analysis: Research Document

## Overview

Co-change coupling (also called logical coupling or temporal coupling) identifies files that tend to change together in version control history, even without explicit code dependencies. This is one of the most actionable signals for AI coding assistants: "when you modify file X, you probably also need to update file Y."

This document covers algorithms, metrics, filtering strategies, graph analysis, and practical Python implementation patterns for detecting co-change coupling from git history.

---

## 1. Co-Occurrence Algorithms

### 1.1 Simple Co-Occurrence Matrix (Same Commit)

The simplest approach: two files are coupled if they appear in the same commit. Build a symmetric co-occurrence matrix where `M[i][j]` counts the number of commits containing both file `i` and file `j`.

```python
from collections import defaultdict
from itertools import combinations

def build_cooccurrence(commits: list[dict]) -> dict[tuple[str, str], int]:
    """
    commits: list of {"hash": str, "files": list[str]}
    Returns: dict mapping (file_a, file_b) -> co-change count
    """
    cooccurrence = defaultdict(int)
    for commit in commits:
        files = sorted(set(commit["files"]))
        for a, b in combinations(files, 2):
            cooccurrence[(a, b)] += 1
    return cooccurrence
```

**Pros**: Simple, fast, no configuration.
**Cons**: Mega-commits (refactors, renames, dependency bumps) create false couplings. Misses couplings spread across related commits in the same logical change.

### 1.2 Temporal Coupling (Time Window / PR-Based)

Instead of requiring files to change in the exact same commit, group changes within a time window or by PR/branch:

- **PR-based grouping**: All commits in the same pull request form one logical changeset. This captures cases where a developer changes `api/auth.ts` in one commit and `middleware/session.ts` in the next commit of the same PR.
- **Time-window grouping**: Group commits by the same author within N hours (CodeScene uses configurable windows). This approximates logical changesets for repos without PR metadata.
- **Ticket-based grouping**: If commit messages reference ticket IDs (e.g., `JIRA-123`), group by ticket.

```python
def group_commits_by_pr(commits, pr_mapping: dict[str, str]) -> dict[str, list[str]]:
    """Group files by PR. pr_mapping: commit_hash -> pr_id"""
    pr_files = defaultdict(set)
    for commit in commits:
        pr_id = pr_mapping.get(commit["hash"])
        if pr_id:
            pr_files[pr_id].update(commit["files"])
    return {pr: sorted(files) for pr, files in pr_files.items()}
```

### 1.3 CodeScene / code-maat Methodology

Adam Tornhill's approach (implemented in [code-maat](https://github.com/adamtornhill/code-maat) and [CodeScene](https://codescene.com)):

**Coupling degree formula**:
```
degree = (shared_revisions / avg_revisions) * 100
```

Where:
- `shared_revisions` = number of commits containing both files
- `avg_revisions` = (revisions_of_A + revisions_of_B) / 2

This normalizes coupling by how frequently each file changes independently. Two files that change together 5 times out of 5 total changes are more tightly coupled than two files that change together 5 times out of 500 total changes.

**CodeScene's three coupling mechanisms**:
1. **Commit-based**: Files modified in the same commit
2. **Developer-based**: Changed by the same programmer within a configurable timeframe
3. **Ticket-based**: Both reference identical ticket IDs in commit messages

**Default thresholds** (from CodeScene docs):
- Minimum 10 revisions per file (filters new/rarely-changed files)
- Minimum 10 shared commits between coupled pairs
- Coupling strength floor of 50%
- Ignore commits touching > 50 files

---

## 2. Association Rule Mining Metrics

Treating commits as "transactions" and files as "items", we can apply classic association rule mining metrics from the Apriori algorithm:

### 2.1 Support

How frequently the file pair appears together across all commits:

```
support(A, B) = count(commits containing both A and B) / total_commits
```

**For gitlore**: Absolute support counts matter more than ratios (since total_commits is not a meaningful denominator for a specific pair). Use **pair support** = count of shared commits.

### 2.2 Confidence

Directional metric — given that file A changed, how often does B also change?

```
confidence(A -> B) = shared_commits(A, B) / total_commits(A)
confidence(B -> A) = shared_commits(A, B) / total_commits(B)
```

Confidence is asymmetric: `auth.ts -> session.ts` may have confidence 0.8 (80% of auth changes also touch session) while `session.ts -> auth.ts` may have confidence 0.4 (session changes sometimes happen independently).

**This asymmetry is valuable for AI assistants** — it tells the direction of the dependency: "if you change auth.ts, you almost certainly need to update session.ts, but not necessarily the other way around."

### 2.3 Lift

Measures whether the co-occurrence is more than random chance:

```
lift(A, B) = support(A, B) / (support(A) * support(B))
```

- `lift > 1`: Files change together more than expected by chance (true coupling)
- `lift = 1`: Independent (co-occurrence is coincidental)
- `lift < 1`: Negative association (files tend NOT to change together)

Lift is the most important metric for distinguishing real coupling from noise. A file that changes in every commit (like `package.json`) will have high raw co-occurrence with everything, but lift close to 1.0 because the co-occurrence is explained by its base rate.

### 2.4 Combined Scoring

```python
import math

def coupling_score(
    shared: int,
    revs_a: int,
    revs_b: int,
    total_commits: int,
) -> dict:
    support_ab = shared / total_commits
    support_a = revs_a / total_commits
    support_b = revs_b / total_commits

    confidence_a_b = shared / revs_a if revs_a > 0 else 0
    confidence_b_a = shared / revs_b if revs_b > 0 else 0
    lift = support_ab / (support_a * support_b) if (support_a * support_b) > 0 else 0

    # code-maat style degree
    avg_revs = (revs_a + revs_b) / 2
    degree = (shared / avg_revs * 100) if avg_revs > 0 else 0

    return {
        "shared_commits": shared,
        "confidence_a_b": round(confidence_a_b, 3),
        "confidence_b_a": round(confidence_b_a, 3),
        "lift": round(lift, 3),
        "degree": round(degree, 1),
        "support": round(support_ab, 6),
    }
```

---

## 3. Graph Analysis with NetworkX

### 3.1 Building the Coupling Graph

```python
import networkx as nx

def build_coupling_graph(
    cooccurrence: dict[tuple[str, str], int],
    file_revisions: dict[str, int],
    min_shared: int = 3,
    min_confidence: float = 0.2,
) -> nx.Graph:
    G = nx.Graph()

    for (a, b), shared in cooccurrence.items():
        revs_a = file_revisions.get(a, 0)
        revs_b = file_revisions.get(b, 0)
        if shared < min_shared:
            continue

        conf_ab = shared / revs_a if revs_a else 0
        conf_ba = shared / revs_b if revs_b else 0
        max_conf = max(conf_ab, conf_ba)

        if max_conf < min_confidence:
            continue

        G.add_edge(a, b, weight=shared, confidence_ab=conf_ab,
                   confidence_ba=conf_ba, max_confidence=max_conf)

    # Add revision counts as node attributes
    for node in G.nodes():
        G.nodes[node]["revisions"] = file_revisions.get(node, 0)

    return G
```

### 3.2 Community Detection for Module Clustering

Communities in the coupling graph reveal implicit modules — groups of files that change together as a unit, even if they live in different directories.

```python
from networkx.algorithms.community import louvain_communities

def detect_modules(G: nx.Graph) -> list[set[str]]:
    """Find implicit modules via Louvain community detection."""
    if len(G.nodes()) == 0:
        return []
    communities = louvain_communities(G, weight="weight", resolution=1.0)
    # Sort by size descending
    return sorted(communities, key=len, reverse=True)
```

**Louvain** is the best default: fast (O(n log n)), works well on weighted graphs, and produces interpretable results. NetworkX also supports:
- **Leiden** (improved Louvain, guarantees connected communities)
- **Girvan-Newman** (edge betweenness, slower but good for small graphs)
- **Label Propagation** (very fast, less deterministic)

### 3.3 Centrality Metrics

Identify core/hub files that couple with many others:

```python
def identify_hub_files(G: nx.Graph, top_n: int = 20) -> list[tuple[str, float]]:
    """Files that are central to the coupling graph."""
    # Weighted degree centrality: sum of edge weights
    centrality = {}
    for node in G.nodes():
        centrality[node] = sum(
            G[node][neighbor]["weight"] for neighbor in G.neighbors(node)
        )
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_n]
```

Betweenness centrality is also useful — files with high betweenness sit between multiple modules and are often integration points:

```python
betweenness = nx.betweenness_centrality(G, weight="weight")
```

### 3.4 Pruning Strategies

Before building the graph, prune noise:

1. **Minimum support threshold**: Require `shared_commits >= N` (3-5 for small repos, 10+ for large repos)
2. **Minimum confidence threshold**: At least one direction must have confidence >= 0.2
3. **Maximum file change frequency**: Files that change in > 30% of all commits are likely noise (lockfiles, configs) — exclude from edges or flag
4. **Ignore specific file patterns**: `*.lock`, `package-lock.json`, `yarn.lock`, `*.min.js`, `CHANGELOG.md`, etc.
5. **Commit size filter**: Skip commits touching > N files (50 is CodeScene's default)

---

## 4. Noise Filtering

### 4.1 Ubiquitous Files (Change With Everything)

Files like `package.json`, `Cargo.lock`, `go.sum`, `.github/workflows/*.yml` change with nearly every feature. They have high raw co-occurrence but provide no useful coupling signal.

**Detection**: Calculate each file's **change frequency ratio**:
```python
def detect_ubiquitous_files(
    file_revisions: dict[str, int],
    total_commits: int,
    threshold: float = 0.15,
) -> set[str]:
    """Files that change in more than `threshold` fraction of all commits."""
    return {
        f for f, revs in file_revisions.items()
        if revs / total_commits > threshold
    }
```

**Alternative (lift-based)**: Instead of excluding ubiquitous files entirely, use lift to automatically down-weight their couplings. A file that changes in 50% of commits will have lift ~1.0 with most other files, so it naturally drops out of high-lift results.

**Recommended approach**: Use a configurable exclusion list + lift filtering. Exclusion list for known noise patterns:

```python
NOISE_PATTERNS = [
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "go.sum",
    "poetry.lock",
    "Pipfile.lock",
    "*.min.js",
    "*.min.css",
    "CHANGELOG*",
    ".gitignore",
]
```

### 4.2 Commit Size Limits

Mega-commits (bulk renames, dependency upgrades, formatter runs) create false couplings between unrelated files.

**Strategy**: Skip commits where `len(files) > max_files_per_commit`. CodeScene uses 50 as the default. For gitlore, a configurable threshold of 30-50 is reasonable.

```python
def filter_commits(commits, max_files: int = 50) -> list[dict]:
    return [c for c in commits if len(c["files"]) <= max_files]
```

Also consider filtering merge commits (which may duplicate file lists from both parents) and bot-authored commits (dependabot, renovate).

### 4.3 Rename Detection

When a file is renamed/moved, its coupling history should transfer to the new path. Git tracks renames with a similarity threshold (default 50%).

**Using PyDriller**:
```python
from pydriller import Repository

for commit in Repository("/path/to/repo").traverse_commits():
    for mod in commit.modified_files:
        if mod.old_path and mod.new_path and mod.old_path != mod.new_path:
            # This is a rename: old_path -> new_path
            # Update coupling data: merge old_path entries into new_path
            pass
```

**Implementation**: Maintain a `rename_map: dict[str, str]` mapping old paths to current paths. Before recording co-occurrences, resolve all file paths through the rename map:

```python
def resolve_path(path: str, rename_map: dict[str, str]) -> str:
    """Follow rename chain to current path."""
    seen = set()
    while path in rename_map and path not in seen:
        seen.add(path)
        path = rename_map[path]
    return path
```

---

## 5. Confidence Scoring & Statistical Significance

### 5.1 When Is a Coupling "Real"?

Not every co-occurrence is meaningful. We need to distinguish genuine couplings from coincidence.

**Minimum thresholds** (practical defaults):
- `shared_commits >= 3`: At least 3 co-changes to be considered
- `min(revs_a, revs_b) >= 5`: Both files must have enough history
- `max(confidence_a_b, confidence_b_a) >= 0.25`: At least 25% directional coupling
- `lift >= 1.5`: Co-occurrence is at least 50% more than random chance

### 5.2 Chi-Squared Test for Independence

For a rigorous statistical test, use a 2x2 contingency table for each file pair:

```
                  B changed    B not changed
A changed         n_AB         n_A - n_AB
A not changed     n_B - n_AB   N - n_A - n_B + n_AB
```

Where:
- `N` = total commits
- `n_A` = commits touching A
- `n_B` = commits touching B
- `n_AB` = commits touching both

```python
from scipy.stats import chi2_contingency
import numpy as np

def is_coupling_significant(
    shared: int, revs_a: int, revs_b: int, total: int, alpha: float = 0.01
) -> tuple[bool, float]:
    """Chi-squared test for independence of file changes."""
    table = np.array([
        [shared, revs_a - shared],
        [revs_b - shared, total - revs_a - revs_b + shared],
    ])
    # Ensure no negative values (can happen with imperfect data)
    table = np.maximum(table, 0)
    if table.sum() == 0 or shared == 0:
        return False, 1.0
    chi2, p_value, dof, expected = chi2_contingency(table, correction=True)
    return p_value < alpha, p_value
```

Use `alpha = 0.01` (not 0.05) because we're testing many pairs — apply Bonferroni correction or use False Discovery Rate (FDR) for multiple testing:

```python
from scipy.stats import false_discovery_control

def filter_significant_pairs(pairs_with_pvalues, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    p_values = [p for _, p in pairs_with_pvalues]
    # scipy >= 1.11 has false_discovery_control
    # For older versions, use statsmodels.stats.multitest
    reject = false_discovery_control(p_values, method='bh')
    return [
        pair for (pair, _), is_sig in zip(pairs_with_pvalues, reject) if is_sig
    ]
```

### 5.3 Practical Scoring for gitlore

For downstream LLM synthesis, a combined score is more useful than raw metrics:

```python
def combined_coupling_strength(
    shared: int,
    revs_a: int,
    revs_b: int,
    total_commits: int,
    lift: float,
) -> float:
    """
    Score 0-1 combining confidence and statistical significance.
    Higher = stronger evidence of true coupling.
    """
    max_conf = max(
        shared / revs_a if revs_a else 0,
        shared / revs_b if revs_b else 0,
    )
    # Penalize low sample sizes with a sigmoid on shared count
    sample_factor = 1 - math.exp(-shared / 5.0)  # approaches 1 as shared grows
    # Lift bonus: reward lift > 1, cap contribution
    lift_factor = min(lift / 3.0, 1.0) if lift > 1 else 0
    return max_conf * 0.5 + sample_factor * 0.25 + lift_factor * 0.25
```

---

## 6. Temporal Decay

### 6.1 Why Decay Matters

Codebase structure changes over time. A coupling that was strong 2 years ago may no longer be relevant if the code has been refactored. Recent co-changes are more predictive of current architecture.

### 6.2 Exponential Decay Function

Weight each co-occurrence by how recent it is:

```python
import math
from datetime import datetime, timezone

def exponential_decay_weight(
    commit_date: datetime,
    reference_date: datetime,
    half_life_days: float = 180,
) -> float:
    """
    Weight decays to 0.5 after half_life_days.
    half_life_days=180 means a co-change from 6 months ago
    counts half as much as one from today.
    """
    age_days = (reference_date - commit_date).total_seconds() / 86400
    if age_days < 0:
        return 1.0
    lambda_ = math.log(2) / half_life_days
    return math.exp(-lambda_ * age_days)
```

**Recommended half-life**: 180-365 days. Shorter half-life (90 days) for fast-moving projects, longer (365 days) for stable codebases. This should be configurable.

### 6.3 Applying Decay to Co-Occurrence Counts

Instead of integer counts, accumulate weighted counts:

```python
def build_weighted_cooccurrence(
    commits: list[dict],
    reference_date: datetime,
    half_life_days: float = 180,
) -> dict[tuple[str, str], float]:
    cooccurrence = defaultdict(float)
    for commit in commits:
        w = exponential_decay_weight(commit["date"], reference_date, half_life_days)
        files = sorted(set(commit["files"]))
        for a, b in combinations(files, 2):
            cooccurrence[(a, b)] += w
    return cooccurrence
```

### 6.4 Decay on Revision Counts Too

For metrics like confidence and lift, also apply decay to individual file revision counts:

```python
def weighted_file_revisions(
    commits: list[dict],
    reference_date: datetime,
    half_life_days: float = 180,
) -> dict[str, float]:
    revisions = defaultdict(float)
    for commit in commits:
        w = exponential_decay_weight(commit["date"], reference_date, half_life_days)
        for f in commit["files"]:
            revisions[f] += w
    return revisions
```

---

## 7. Performance

### 7.1 Scaling Considerations

For a repo with `F` files and `C` commits:
- **Naive co-occurrence**: O(C * F_per_commit^2) to build the matrix. For most commits, F_per_commit is small (< 20), so this is fast even for large repos.
- **The bottleneck is git log parsing**, not the co-occurrence computation. PyDriller traverses commits serially by default.
- **Memory**: The co-occurrence matrix is sparse (most file pairs never co-change). A `defaultdict` or scipy sparse matrix works well.

### 7.2 Optimization Strategies

**1. Use `git log --name-only` directly for speed**:
```python
import subprocess

def fast_git_log(repo_path: str, max_files_per_commit: int = 50) -> list[dict]:
    """Parse git log ~10x faster than PyDriller for co-change analysis."""
    result = subprocess.run(
        ["git", "log", "--pretty=format:%H|%aI", "--name-only"],
        capture_output=True, text=True, cwd=repo_path,
    )
    commits = []
    current = None
    for line in result.stdout.split("\n"):
        if "|" in line and len(line.split("|")) == 2:
            if current and len(current["files"]) <= max_files_per_commit:
                commits.append(current)
            parts = line.split("|")
            current = {
                "hash": parts[0],
                "date": datetime.fromisoformat(parts[1]),
                "files": [],
            }
        elif line.strip() and current is not None:
            current["files"].append(line.strip())
    if current and len(current["files"]) <= max_files_per_commit:
        commits.append(current)
    return commits
```

**2. Sparse matrix for large file sets**:
```python
from scipy.sparse import dok_matrix
import numpy as np

def sparse_cooccurrence(commits, file_index: dict[str, int]) -> dok_matrix:
    """Use sparse matrix when file count > 10,000."""
    n = len(file_index)
    matrix = dok_matrix((n, n), dtype=np.float32)
    for commit in commits:
        indices = sorted(set(file_index[f] for f in commit["files"] if f in file_index))
        for i, a in enumerate(indices):
            for b in indices[i+1:]:
                matrix[a, b] += 1
                matrix[b, a] += 1
    return matrix.tocsr()
```

**3. Incremental updates**: For repos that are re-analyzed periodically, only process new commits since the last analysis. Store the last-analyzed commit hash.

**4. File path interning**: Map file paths to integers early to reduce memory and speed up dictionary lookups.

### 7.3 Benchmarks (Approximate)

| Repo Size | Commits | Files | co-occurrence build time |
|-----------|---------|-------|--------------------------|
| Small     | 1,000   | 200   | < 1 second               |
| Medium    | 10,000  | 2,000 | ~5 seconds               |
| Large     | 50,000  | 10,000| ~30 seconds              |
| Huge      | 100,000+| 50,000| ~2-5 minutes (sparse)    |

Git log parsing dominates. The combinatorial step is fast because average files-per-commit is typically 3-8.

---

## 8. Output Format for LLM Synthesis

### 8.1 Design Goals

The output must be:
1. **Human-readable** — for debugging and validation
2. **LLM-parseable** — structured enough for a frontier model to synthesize into rules
3. **Compact** — only include actionable couplings (no noise)

### 8.2 Recommended Format

```json
{
  "metadata": {
    "repo": "org/project",
    "analyzed_commits": 5432,
    "analysis_date": "2026-02-12",
    "half_life_days": 180,
    "min_shared_commits": 3,
    "min_confidence": 0.25
  },
  "couplings": [
    {
      "file_a": "src/api/auth.ts",
      "file_b": "src/middleware/session.ts",
      "shared_commits": 23,
      "revisions_a": 30,
      "revisions_b": 45,
      "confidence_a_to_b": 0.767,
      "confidence_b_to_a": 0.511,
      "lift": 4.2,
      "strength": 0.85,
      "recent_trend": "stable"
    }
  ],
  "modules": [
    {
      "id": 0,
      "name": "auth-system",
      "files": ["src/api/auth.ts", "src/middleware/session.ts", "src/models/user.ts"],
      "internal_coupling_avg": 0.72
    }
  ],
  "hub_files": [
    {
      "file": "src/db/schema.ts",
      "coupled_file_count": 15,
      "total_coupling_weight": 87.3
    }
  ]
}
```

### 8.3 Natural Language Summary for LLM Context

For direct injection into AI assistant system prompts, generate natural language summaries:

```
## File Coupling Rules

When modifying these files, consider updating their coupled counterparts:

- **src/api/auth.ts** and **src/middleware/session.ts**: Changed together in 23 of 30 commits touching auth.ts (77% confidence). These form the core authentication flow.
- **src/db/schema.ts** is a hub file coupled with 15 other files. Changes here typically require updates to migration files and model definitions.

## Implicit Modules

The following file groups form logical modules based on change patterns:
- Auth system: src/api/auth.ts, src/middleware/session.ts, src/models/user.ts
- Payment flow: src/api/payments.ts, src/services/stripe.ts, src/models/invoice.ts
```

### 8.4 Generating the Summary

```python
def generate_coupling_summary(couplings: list[dict], modules: list[dict]) -> str:
    lines = ["## File Coupling Rules\n"]
    lines.append("When modifying these files, consider updating their coupled counterparts:\n")

    for c in sorted(couplings, key=lambda x: x["strength"], reverse=True)[:20]:
        dominant_dir = "a_to_b" if c["confidence_a_to_b"] >= c["confidence_b_to_a"] else "b_to_a"
        if dominant_dir == "a_to_b":
            src, tgt = c["file_a"], c["file_b"]
            conf = c["confidence_a_to_b"]
            revs = c["revisions_a"]
        else:
            src, tgt = c["file_b"], c["file_a"]
            conf = c["confidence_b_to_a"]
            revs = c["revisions_b"]

        lines.append(
            f"- **{src}** -> **{tgt}**: "
            f"changed together in {c['shared_commits']} of {revs} commits "
            f"touching {src} ({conf:.0%} confidence, lift={c['lift']:.1f})"
        )

    if modules:
        lines.append("\n## Implicit Modules\n")
        for mod in modules[:10]:
            file_list = ", ".join(mod["files"][:5])
            if len(mod["files"]) > 5:
                file_list += f", +{len(mod['files']) - 5} more"
            lines.append(f"- **{mod.get('name', f'Module {mod[\"id\"]}')}**: {file_list}")

    return "\n".join(lines)
```

---

## 9. Complete Pipeline Example

```python
"""
Full co-change coupling pipeline.
Dependencies: networkx, scipy, numpy
"""
from collections import defaultdict
from itertools import combinations
from datetime import datetime, timezone
import math
import subprocess
import json

import networkx as nx
from networkx.algorithms.community import louvain_communities
import numpy as np
from scipy.stats import chi2_contingency


def analyze_coupling(
    repo_path: str,
    max_files_per_commit: int = 50,
    min_shared: int = 3,
    min_confidence: float = 0.25,
    min_lift: float = 1.5,
    half_life_days: float = 180,
    noise_patterns: list[str] | None = None,
) -> dict:
    """Full coupling analysis pipeline."""

    now = datetime.now(timezone.utc)

    # Step 1: Extract commits
    commits = parse_git_log(repo_path, max_files_per_commit, noise_patterns or [])

    # Step 2: Build weighted co-occurrence
    cooccurrence = defaultdict(float)
    file_revisions = defaultdict(float)
    total_commits = len(commits)

    for commit in commits:
        w = exponential_decay_weight(commit["date"], now, half_life_days)
        files = sorted(set(commit["files"]))
        for f in files:
            file_revisions[f] += w
        for a, b in combinations(files, 2):
            cooccurrence[(a, b)] += w

    # Step 3: Compute metrics and filter
    couplings = []
    for (a, b), shared_w in cooccurrence.items():
        revs_a = file_revisions[a]
        revs_b = file_revisions[b]

        conf_ab = shared_w / revs_a if revs_a else 0
        conf_ba = shared_w / revs_b if revs_b else 0
        max_conf = max(conf_ab, conf_ba)

        if max_conf < min_confidence:
            continue
        if shared_w < min_shared:
            continue

        # Lift (using weighted values)
        total_w = sum(
            exponential_decay_weight(c["date"], now, half_life_days)
            for c in commits
        )
        sup_ab = shared_w / total_w if total_w else 0
        sup_a = revs_a / total_w if total_w else 0
        sup_b = revs_b / total_w if total_w else 0
        lift = sup_ab / (sup_a * sup_b) if (sup_a * sup_b) > 0 else 0

        if lift < min_lift:
            continue

        strength = compute_strength(shared_w, max_conf, lift)

        couplings.append({
            "file_a": a,
            "file_b": b,
            "shared_commits": round(shared_w, 2),
            "revisions_a": round(revs_a, 2),
            "revisions_b": round(revs_b, 2),
            "confidence_a_to_b": round(conf_ab, 3),
            "confidence_b_to_a": round(conf_ba, 3),
            "lift": round(lift, 2),
            "strength": round(strength, 3),
        })

    # Step 4: Build graph and detect modules
    G = nx.Graph()
    for c in couplings:
        G.add_edge(c["file_a"], c["file_b"], weight=c["shared_commits"],
                   strength=c["strength"])

    modules = []
    if len(G.nodes()) > 2:
        communities = louvain_communities(G, weight="weight", resolution=1.0)
        for i, community in enumerate(sorted(communities, key=len, reverse=True)):
            modules.append({
                "id": i,
                "files": sorted(community),
            })

    # Step 5: Identify hub files
    hubs = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        total_weight = sum(G[node][n]["weight"] for n in neighbors)
        if len(neighbors) >= 3:
            hubs.append({
                "file": node,
                "coupled_file_count": len(neighbors),
                "total_coupling_weight": round(total_weight, 2),
            })
    hubs.sort(key=lambda x: x["total_coupling_weight"], reverse=True)

    return {
        "metadata": {
            "analyzed_commits": total_commits,
            "analysis_date": now.isoformat(),
            "half_life_days": half_life_days,
            "min_shared_commits": min_shared,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
        },
        "couplings": sorted(couplings, key=lambda x: x["strength"], reverse=True),
        "modules": modules,
        "hub_files": hubs[:20],
    }


def compute_strength(shared_w, max_conf, lift):
    sample_factor = 1 - math.exp(-shared_w / 5.0)
    lift_factor = min(lift / 3.0, 1.0) if lift > 1 else 0
    return max_conf * 0.5 + sample_factor * 0.25 + lift_factor * 0.25


def exponential_decay_weight(commit_date, reference_date, half_life_days):
    age_days = (reference_date - commit_date).total_seconds() / 86400
    if age_days < 0:
        return 1.0
    lambda_ = math.log(2) / half_life_days
    return math.exp(-lambda_ * age_days)


def parse_git_log(repo_path, max_files, noise_patterns):
    import fnmatch
    result = subprocess.run(
        ["git", "log", "--pretty=format:%H|%aI", "--name-only", "--no-merges"],
        capture_output=True, text=True, cwd=repo_path,
    )
    commits = []
    current = None
    for line in result.stdout.split("\n"):
        if "|" in line and len(line.split("|")) == 2:
            if current and 0 < len(current["files"]) <= max_files:
                commits.append(current)
            parts = line.split("|")
            current = {
                "hash": parts[0],
                "date": datetime.fromisoformat(parts[1]),
                "files": [],
            }
        elif line.strip() and current is not None:
            path = line.strip()
            if not any(fnmatch.fnmatch(path, pat) for pat in noise_patterns):
                current["files"].append(path)
    if current and 0 < len(current["files"]) <= max_files:
        commits.append(current)
    return commits
```

---

## 10. Key Design Decisions for gitlore

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Grouping unit | Commit-based (default) + PR-based (when available) | Commits are universal; PRs add signal but require API access |
| Primary metric | Confidence (directional) + Lift | Confidence gives actionable direction; lift filters noise |
| Temporal decay | Exponential, half-life 180 days (configurable) | Balances recency with enough history for signal |
| Noise filtering | Pattern exclusion + commit size cap + lift threshold | Multi-layered approach catches different noise types |
| Min thresholds | 3 shared commits, 0.25 confidence, 1.5 lift | Pragmatic defaults that work across repo sizes |
| Graph analysis | Louvain community detection on weighted coupling graph | Fast, produces interpretable module clusters |
| Output | Structured JSON + natural language summary | JSON for programmatic use; NL for LLM context injection |
| Statistical test | Chi-squared with FDR correction (optional) | For high-rigor analysis; simple thresholds suffice for most cases |
| Rename handling | Build rename map from git diff, resolve before counting | Preserves coupling across file moves |
| Performance target | < 60 seconds for repos up to 50k commits | Direct git log parsing + sparse structures |

---

## References

- Adam Tornhill, *Your Code as a Crime Scene* (2015) and *Software Design X-Rays* (2018)
- [code-maat: logical coupling implementation](https://github.com/adamtornhill/code-maat/blob/master/src/code_maat/analysis/logical_coupling.clj)
- [code-maat: sum of coupling](https://github.com/adamtornhill/code-maat/blob/master/src/code_maat/analysis/sum_of_coupling.clj)
- [CodeScene temporal coupling documentation](https://docs.enterprise.codescene.io/versions/2.4.2/guides/technical/temporal-coupling.html)
- [code-forensics coupling analysis](https://github.com/smontanari/code-forensics/wiki/Coupling-analysis)
- [NetworkX community detection](https://networkx.org/documentation/stable/reference/algorithms/community.html)
- [python-louvain community detection](https://python-louvain.readthedocs.io/en/latest/)
- [Association rule learning (Wikipedia)](https://en.wikipedia.org/wiki/Association_rule_learning)
- [Lift metric (Wikipedia)](https://en.wikipedia.org/wiki/Lift_(data_mining))
- [PyDriller: Python framework for mining software repositories](https://github.com/ishepard/pydriller)
- [Forward Decay: A Practical Time Decay Model](https://dimacs.rutgers.edu/~graham/pubs/papers/fwddecay.pdf)
