"""Churn hotspot detection with temporal decay weighting."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from gitlore.config import AnalysisConfig
from gitlore.models import ChurnHotspot, ClassifiedCommit, CommitType
from gitlore.utils.temporal import exponential_decay


def analyze_churn(
    classified: list[ClassifiedCommit],
    config: AnalysisConfig | None = None,
    reference_date: datetime | None = None,
) -> list[ChurnHotspot]:
    """Detect churn hotspots from classified commits.

    Computes per-file weighted commit count, relative churn, and fix ratio.
    Results are sorted by score descending.

    Args:
        classified: List of classified commits.
        config: Analysis configuration for half-life.
        reference_date: Reference date for decay calculation. Defaults to UTC now.

    Returns:
        List of ChurnHotspot sorted by score descending.
    """
    if config is None:
        config = AnalysisConfig()
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    half_life = config.half_life_days

    # Per-file accumulation
    file_data: dict[str, _FileStats] = defaultdict(_FileStats)

    for cc in classified:
        commit = cc.commit
        weight = exponential_decay(commit.author_date, reference_date, half_life)
        is_fix = cc.commit_type == CommitType.FIX

        for fc in commit.files:
            stats = file_data[fc.path]
            stats.commit_count += 1
            stats.weighted_commit_count += weight
            stats.lines_added += fc.added or 0
            stats.lines_deleted += fc.deleted or 0
            if is_fix:
                stats.fix_count += 1

    # Build hotspot list
    hotspots: list[ChurnHotspot] = []
    for path, stats in file_data.items():
        if stats.commit_count == 0:
            continue

        total_churn = stats.lines_added + stats.lines_deleted
        churn_ratio = total_churn / stats.commit_count if stats.commit_count > 0 else 0.0
        fix_ratio = stats.fix_count / stats.commit_count

        # Combined score: weighted commits * (1 + fix_ratio) gives extra weight
        # to files that are both frequently changed and frequently fixed
        score = stats.weighted_commit_count * (1.0 + fix_ratio)

        hotspots.append(
            ChurnHotspot(
                path=path,
                commit_count=stats.commit_count,
                weighted_commit_count=round(stats.weighted_commit_count, 4),
                lines_added=stats.lines_added,
                lines_deleted=stats.lines_deleted,
                churn_ratio=round(churn_ratio, 2),
                fix_ratio=round(fix_ratio, 4),
                score=round(score, 4),
            )
        )

    hotspots.sort(key=lambda h: h.score, reverse=True)
    return hotspots


class _FileStats:
    """Internal accumulator for per-file statistics."""

    __slots__ = (
        "commit_count",
        "weighted_commit_count",
        "lines_added",
        "lines_deleted",
        "fix_count",
    )

    def __init__(self) -> None:
        self.commit_count: int = 0
        self.weighted_commit_count: float = 0.0
        self.lines_added: int = 0
        self.lines_deleted: int = 0
        self.fix_count: int = 0
