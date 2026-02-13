"""Fix-after chain detection: commits followed by corrective follow-ups."""

from __future__ import annotations

import re
from datetime import timedelta

from gitlore.models import Commit, FixAfterChain, FixAfterTier

# ── Fix keyword sets per tier ────────────────────────────────────────────────

_FOLLOWUP_KEYWORDS = frozenset({
    "fix",
    "oops",
    "typo",
    "forgot",
    "actually",
    "woops",
    "mistake",
    "correct",
    "amend",
    "patch",
    "broken",
    "missed",
})

_STRONG_FIX_PATTERNS = [
    re.compile(r"fix(?:es|ed)?\s+(?:bug|issue|error|crash|regression)", re.IGNORECASE),
    re.compile(r"\brevert\b", re.IGNORECASE),
    re.compile(r"\bhotfix\b", re.IGNORECASE),
]

# Git's built-in fixup/squash markers
_GIT_FIXUP_RE = re.compile(r"^(?:fixup|squash|amend)!\s+")

# ── Tier thresholds ──────────────────────────────────────────────────────────

_IMMEDIATE_MAX = timedelta(minutes=30)
_FOLLOWUP_MAX = timedelta(hours=4)
_DELAYED_MAX = timedelta(days=7)


def _file_overlap(files_a: list[str], files_b: list[str]) -> bool:
    """Check if there is any file overlap between two commits."""
    return bool(set(files_a) & set(files_b))


def _has_followup_keywords(subject: str) -> bool:
    """Check if subject contains fix-like keywords."""
    lower = subject.lower()
    return any(kw in lower for kw in _FOLLOWUP_KEYWORDS)


def _has_strong_fix_signal(message: str) -> bool:
    """Check for strong fix patterns (for delayed tier)."""
    return any(p.search(message) for p in _STRONG_FIX_PATTERNS)


def _classify_tier(
    original: Commit,
    candidate: Commit,
) -> FixAfterTier | None:
    """Determine if candidate is a fix-after for original, and which tier.

    Returns the tier, or None if no match.
    """
    time_delta = candidate.author_date - original.author_date
    if time_delta <= timedelta():
        return None  # candidate is not after original

    same_author = candidate.author_email == original.author_email
    has_overlap = _file_overlap(original.file_paths, candidate.file_paths)

    if not has_overlap:
        return None

    subject = candidate.subject
    full_msg = f"{subject}\n{candidate.body}"

    # Tier 1: IMMEDIATE - same author, same files, <30 min
    if same_author and time_delta <= _IMMEDIATE_MAX:
        return FixAfterTier.IMMEDIATE

    # Tier 2: FOLLOWUP - same author, same files, <4 hours, fix keywords
    if same_author and time_delta <= _FOLLOWUP_MAX and _has_followup_keywords(subject):
        return FixAfterTier.FOLLOWUP

    # Tier 3: DELAYED - any author, same files, <7 days, strong fix patterns
    if time_delta <= _DELAYED_MAX and _has_strong_fix_signal(full_msg):
        return FixAfterTier.DELAYED

    return None


def detect_fix_after(commits: list[Commit]) -> list[FixAfterChain]:
    """Detect fix-after chains from a list of commits.

    Commits should be sorted by date ascending. For each commit, looks backward
    at recent commits to find potential originals that it fixes.

    Args:
        commits: List of commits sorted by author_date ascending.

    Returns:
        List of FixAfterChain objects.
    """
    # Sort by date to ensure proper ordering
    sorted_commits = sorted(commits, key=lambda c: c.author_date)

    chains: dict[str, FixAfterChain] = {}  # keyed by original hash
    # Track which commits are already identified as fixups to avoid
    # the same fixup being assigned to multiple originals
    used_fixups: set[str] = set()

    for i, commit in enumerate(sorted_commits):
        # Check for git fixup!/squash! markers first
        m = _GIT_FIXUP_RE.match(commit.subject)
        if m:
            target_subject = _GIT_FIXUP_RE.sub("", commit.subject)
            for j in range(i - 1, -1, -1):
                prev = sorted_commits[j]
                if prev.subject == target_subject:
                    _add_to_chain(chains, prev, commit)
                    used_fixups.add(commit.hash)
                    break
            continue

        # Look backward for potential originals
        # Check up to 20 previous commits or until we exceed the max time window
        for j in range(i - 1, max(i - 20, -1), -1):
            prev = sorted_commits[j]

            # Don't use a commit that is already a fixup as an original
            if prev.hash in used_fixups:
                continue

            # Time check: if we're beyond the delayed max, stop looking
            time_delta = commit.author_date - prev.author_date
            if time_delta > _DELAYED_MAX:
                break

            tier = _classify_tier(prev, commit)
            if tier is not None:
                _add_to_chain(chains, prev, commit, tier)
                used_fixups.add(commit.hash)
                break  # one original per fixup

    return list(chains.values())


def _add_to_chain(
    chains: dict[str, FixAfterChain],
    original: Commit,
    fixup: Commit,
    tier: FixAfterTier | None = None,
) -> None:
    """Add a fixup commit to its original's chain."""
    key = original.hash
    if key not in chains:
        chains[key] = FixAfterChain(
            original_hash=original.hash,
            original_subject=original.subject,
            original_author=original.author_name,
            original_date=original.author_date,
            fixup_hashes=[],
            fixup_subjects=[],
            tier=tier or FixAfterTier.IMMEDIATE,
            files=original.file_paths,
            time_span=timedelta(),
        )
    chain = chains[key]
    chain.fixup_hashes.append(fixup.hash)
    chain.fixup_subjects.append(fixup.subject)
    chain.time_span = fixup.author_date - original.author_date
    # Upgrade tier to the weakest (highest) tier in the chain
    if tier is not None and tier.value > chain.tier.value:
        chain.tier = tier
