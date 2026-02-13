"""Incremental analysis checkpoint management."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

CHECKPOINT_FILE = ".gitlore/checkpoint.json"


def load_checkpoint(repo_path: str) -> str | None:
    """Load the last processed commit hash, or None if no checkpoint."""
    path = Path(repo_path) / CHECKPOINT_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("last_commit")


def save_checkpoint(repo_path: str, commit_hash: str) -> None:
    """Save the last processed commit hash."""
    path = Path(repo_path) / CHECKPOINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "last_commit": commit_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
