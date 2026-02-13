"""Co-change coupling analysis with graph-based module detection."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations

import networkx as nx
from networkx.algorithms.community import louvain_communities

from gitlore.config import AnalysisConfig
from gitlore.extractors.git_log import resolve_path
from gitlore.models import Commit, CouplingPair, HubFile, ImplicitModule
from gitlore.utils.temporal import exponential_decay


def analyze_coupling(
    commits: list[Commit],
    config: AnalysisConfig | None = None,
    reference_date: datetime | None = None,
    rename_map: dict[str, str] | None = None,
) -> tuple[list[CouplingPair], list[ImplicitModule], list[HubFile]]:
    """Analyze co-change coupling between files.

    Builds a weighted co-occurrence matrix, computes association rule metrics
    (confidence, lift, strength), builds a graph, and detects implicit modules
    via Louvain community detection.

    Args:
        commits: List of commits to analyze.
        config: Analysis configuration with coupling thresholds.
        reference_date: Reference point for decay. Defaults to UTC now.
        rename_map: Mapping from old file paths to current paths.

    Returns:
        Tuple of (coupling pairs, implicit modules, hub files).
    """
    if config is None:
        config = AnalysisConfig()
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)
    if rename_map is None:
        rename_map = {}

    half_life = config.half_life_days
    max_files = config.max_files_per_commit
    min_shared = config.min_shared_commits
    min_confidence = config.min_coupling_confidence
    min_lift = config.min_coupling_lift

    # ── Step 1: Build weighted co-occurrence and revision counts ──────────
    cooccurrence: dict[tuple[str, str], float] = defaultdict(float)
    file_revisions: dict[str, float] = defaultdict(float)
    total_weighted = 0.0

    for commit in commits:
        # Resolve all file paths through rename map
        paths = []
        for fc in commit.files:
            resolved = resolve_path(fc.path, rename_map)
            paths.append(resolved)

        # Deduplicate after resolution
        paths = sorted(set(paths))

        # Skip mega-commits
        if len(paths) > max_files:
            continue

        if not paths:
            continue

        w = exponential_decay(commit.author_date, reference_date, half_life)
        total_weighted += w

        for p in paths:
            file_revisions[p] += w

        for a, b in combinations(paths, 2):
            cooccurrence[(a, b)] += w

    # ── Step 2: Compute metrics and filter ────────────────────────────────
    coupling_pairs: list[CouplingPair] = []

    for (a, b), shared_w in cooccurrence.items():
        if shared_w < min_shared:
            continue

        revs_a = file_revisions[a]
        revs_b = file_revisions[b]

        conf_a_to_b = shared_w / revs_a if revs_a > 0 else 0.0
        conf_b_to_a = shared_w / revs_b if revs_b > 0 else 0.0
        max_conf = max(conf_a_to_b, conf_b_to_a)

        if max_conf < min_confidence:
            continue

        # Lift calculation
        if total_weighted > 0:
            sup_ab = shared_w / total_weighted
            sup_a = revs_a / total_weighted
            sup_b = revs_b / total_weighted
            lift = sup_ab / (sup_a * sup_b) if (sup_a * sup_b) > 0 else 0.0
        else:
            lift = 0.0

        if lift < min_lift:
            continue

        # Combined strength score
        strength = _compute_strength(shared_w, max_conf, lift)

        coupling_pairs.append(
            CouplingPair(
                file_a=a,
                file_b=b,
                shared_commits=round(shared_w, 4),
                revisions_a=round(revs_a, 4),
                revisions_b=round(revs_b, 4),
                confidence_a_to_b=round(conf_a_to_b, 4),
                confidence_b_to_a=round(conf_b_to_a, 4),
                lift=round(lift, 4),
                strength=round(strength, 4),
            )
        )

    coupling_pairs.sort(key=lambda cp: cp.strength, reverse=True)

    # ── Step 3: Build graph and detect modules ────────────────────────────
    graph = nx.Graph()
    for cp in coupling_pairs:
        graph.add_edge(
            cp.file_a,
            cp.file_b,
            weight=cp.shared_commits,
            strength=cp.strength,
        )

    modules: list[ImplicitModule] = []
    if len(graph.nodes()) > 2:
        try:
            communities = louvain_communities(graph, weight="weight", resolution=1.0)
            for i, community in enumerate(
                sorted(communities, key=len, reverse=True)
            ):
                # Compute average internal coupling for this community
                internal_strengths: list[float] = []
                community_list = sorted(community)
                for a, b in combinations(community_list, 2):
                    if graph.has_edge(a, b):
                        internal_strengths.append(
                            graph[a][b].get("strength", 0.0)
                        )
                avg_coupling = (
                    sum(internal_strengths) / len(internal_strengths)
                    if internal_strengths
                    else 0.0
                )
                modules.append(
                    ImplicitModule(
                        module_id=i,
                        files=community_list,
                        internal_coupling_avg=round(avg_coupling, 4),
                    )
                )
        except Exception:
            # Louvain can fail on degenerate graphs
            pass

    # ── Step 4: Identify hub files ────────────────────────────────────────
    hub_files: list[HubFile] = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) < 3:
            continue
        total_weight = sum(graph[node][n]["weight"] for n in neighbors)
        hub_files.append(
            HubFile(
                path=node,
                coupled_file_count=len(neighbors),
                total_coupling_weight=round(total_weight, 4),
            )
        )
    hub_files.sort(key=lambda h: h.total_coupling_weight, reverse=True)

    return coupling_pairs, modules, hub_files


def _compute_strength(shared_w: float, max_conf: float, lift: float) -> float:
    """Compute combined coupling strength score in [0, 1]."""
    # Penalize low sample sizes with a sigmoid on shared count
    sample_factor = 1 - math.exp(-shared_w / 5.0)
    # Lift bonus: reward lift > 1, cap contribution
    lift_factor = min(lift / 3.0, 1.0) if lift > 1 else 0.0
    return max_conf * 0.5 + sample_factor * 0.25 + lift_factor * 0.25
