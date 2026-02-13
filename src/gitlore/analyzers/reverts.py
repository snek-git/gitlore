"""Revert chain detection from commit history."""

from __future__ import annotations

import re

from gitlore.models import Commit, RevertChain

REVERT_SUBJECT_RE = re.compile(r'^Revert "(?P<original_subject>.+)"', re.IGNORECASE)
REVERT_HASH_RE = re.compile(
    r"This reverts commit (?P<hash>[0-9a-f]{7,40})", re.IGNORECASE
)


def _find_reverted_hash(commit: Commit) -> str | None:
    """Extract the reverted commit hash from a revert commit's body."""
    m = REVERT_HASH_RE.search(commit.body)
    if m:
        return m.group("hash")
    # Also check subject for short form: Revert <hash>
    m = re.match(r"^Revert\s+([0-9a-f]{7,40})", commit.subject, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _is_revert_message(commit: Commit) -> bool:
    """Check if a commit message indicates a revert."""
    return bool(
        REVERT_SUBJECT_RE.match(commit.subject)
        or REVERT_HASH_RE.search(commit.body)
    )


def detect_reverts(commits: list[Commit]) -> list[RevertChain]:
    """Detect revert chains from a list of commits.

    Builds chains where a commit is reverted, and that revert may itself
    be reverted (revert-of-revert). Uses both hash matching from the body
    and subject matching to link reverts to their originals.

    Args:
        commits: List of commits, typically in chronological order.

    Returns:
        List of RevertChain objects.
    """
    # Build lookup tables
    hash_to_commit: dict[str, Commit] = {}
    subject_to_hash: dict[str, str] = {}
    for c in commits:
        hash_to_commit[c.hash] = c
        # Also index by short hash prefix for partial matches
        if len(c.hash) >= 7:
            hash_to_commit[c.hash[:7]] = c
        subject_to_hash[c.subject] = c.hash

    chains: dict[str, RevertChain] = {}  # keyed by original hash
    # Track which hashes are part of chains (either original or revert)
    hash_in_chain: dict[str, str] = {}  # hash -> original chain hash

    for commit in commits:
        if not _is_revert_message(commit):
            continue

        # Find the reverted commit hash
        reverted_hash = _find_reverted_hash(commit)

        # If we don't have a hash from the body, try to match by subject
        if not reverted_hash:
            m = REVERT_SUBJECT_RE.match(commit.subject)
            if m:
                original_subject = m.group("original_subject")
                reverted_hash = subject_to_hash.get(original_subject)

        if not reverted_hash:
            continue

        # Resolve short hash to full hash
        reverted_commit = hash_to_commit.get(reverted_hash)
        if reverted_commit:
            reverted_hash = reverted_commit.hash

        # Check if the reverted commit is already part of a chain
        # (this handles revert-of-revert)
        if reverted_hash in hash_in_chain:
            chain_key = hash_in_chain[reverted_hash]
            chains[chain_key].revert_hashes.append(commit.hash)
            hash_in_chain[commit.hash] = chain_key
        elif reverted_hash in chains:
            # The reverted hash is itself the original of an existing chain
            chains[reverted_hash].revert_hashes.append(commit.hash)
            hash_in_chain[commit.hash] = reverted_hash
        else:
            # Start a new chain
            original_commit = hash_to_commit.get(reverted_hash)
            original_subject = (
                original_commit.subject if original_commit else "<unknown>"
            )
            original_author = (
                original_commit.author_name if original_commit else ""
            )
            original_date = (
                original_commit.author_date if original_commit else None
            )
            files = (
                original_commit.file_paths if original_commit else []
            )

            chain = RevertChain(
                original_hash=reverted_hash,
                original_subject=original_subject,
                revert_hashes=[commit.hash],
                original_author=original_author,
                original_date=original_date,
                files=files,
            )
            chains[reverted_hash] = chain
            hash_in_chain[reverted_hash] = reverted_hash
            hash_in_chain[commit.hash] = reverted_hash

    return list(chains.values())
