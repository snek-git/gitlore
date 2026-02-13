"""Temporal decay functions for weighting historical signals."""

from __future__ import annotations

import math
from datetime import datetime, timezone


def exponential_decay(
    commit_date: datetime,
    reference_date: datetime | None = None,
    half_life_days: float = 180,
) -> float:
    """Weight that decays to 0.5 after half_life_days.

    Args:
        commit_date: When the event occurred.
        reference_date: The "now" reference point. Defaults to UTC now.
        half_life_days: Days until weight halves.

    Returns:
        Float in (0, 1].
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)
    # Ensure both are offset-aware for subtraction
    if commit_date.tzinfo is None:
        commit_date = commit_date.replace(tzinfo=timezone.utc)
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)

    age_days = (reference_date - commit_date).total_seconds() / 86400
    if age_days < 0:
        return 1.0
    lambda_ = math.log(2) / half_life_days
    return math.exp(-lambda_ * age_days)
