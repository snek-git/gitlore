"""Load prompt templates from .txt files in this package."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Load a prompt template by name (without .txt extension).

    Reads from src/gitlore/prompts/{name}.txt and caches the result.
    """
    path = _DIR / f"{name}.txt"
    return path.read_text().strip()
