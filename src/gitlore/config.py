"""Configuration loading and defaults."""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    classifier: str = "openrouter/google/gemini-3-flash-preview"
    synthesizer: str = "openrouter/anthropic/claude-sonnet-4-5-20250929"
    embedding: str = "openrouter/openai/text-embedding-3-small"


@dataclass
class AnalysisConfig:
    since_months: int = 12
    half_life_days: float = 180
    min_coupling_confidence: float = 0.25
    min_coupling_lift: float = 1.5
    max_files_per_commit: int = 50
    min_shared_commits: int = 3


@dataclass
class GitHubConfig:
    owner: str = ""
    repo: str = ""
    token: str = ""

    def resolve_token(self) -> str:
        """Get token from config, env var, or gh CLI."""
        if self.token:
            return self.token
        env_token = os.environ.get("GITHUB_TOKEN", "")
        if env_token:
            return env_token
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return ""

    def resolve_owner_repo(self, repo_path: str) -> tuple[str, str]:
        """Get owner/repo from config or git remote origin."""
        if self.owner and self.repo:
            return self.owner, self.repo
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Handle both SSH and HTTPS formats:
                #   git@github.com:owner/repo.git
                #   https://github.com/owner/repo.git
                import re
                m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
                if m:
                    return m.group(1), m.group(2)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return self.owner, self.repo


@dataclass
class OutputConfig:
    formats: list[str] = field(default_factory=lambda: ["report", "html"])


@dataclass
class GitloreConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    repo_path: str = "."

    @classmethod
    def load(cls, path: Path | None = None) -> GitloreConfig:
        """Load config from gitlore.toml, falling back to defaults."""
        if path is None:
            path = Path("gitlore.toml")
        if not path.exists():
            return cls()

        with open(path, "rb") as f:
            raw = tomllib.load(f)

        config = cls()

        if "models" in raw:
            m = raw["models"]
            config.models = ModelConfig(
                classifier=m.get("classifier", config.models.classifier),
                synthesizer=m.get("synthesizer", config.models.synthesizer),
                embedding=m.get("embedding", config.models.embedding),
            )

        if "analysis" in raw:
            a = raw["analysis"]
            config.analysis = AnalysisConfig(
                since_months=a.get("since_months", config.analysis.since_months),
                half_life_days=a.get("half_life_days", config.analysis.half_life_days),
                min_coupling_confidence=a.get(
                    "min_coupling_confidence", config.analysis.min_coupling_confidence
                ),
                min_coupling_lift=a.get("min_coupling_lift", config.analysis.min_coupling_lift),
                max_files_per_commit=a.get(
                    "max_files_per_commit", config.analysis.max_files_per_commit
                ),
                min_shared_commits=a.get(
                    "min_shared_commits", config.analysis.min_shared_commits
                ),
            )

        if "github" in raw:
            g = raw["github"]
            config.github = GitHubConfig(
                owner=g.get("owner", ""),
                repo=g.get("repo", ""),
                token=g.get("token", ""),
            )

        if "output" in raw:
            o = raw["output"]
            config.output = OutputConfig(
                formats=o.get("formats", config.output.formats),
            )

        return config


DEFAULT_CONFIG_TEMPLATE = """\
[models]
classifier = "openrouter/google/gemini-3-flash-preview"
synthesizer = "openrouter/anthropic/claude-sonnet-4-5-20250929"
embedding = "openrouter/openai/text-embedding-3-small"

[analysis]
since_months = 12
half_life_days = 180
min_coupling_confidence = 0.25
min_coupling_lift = 1.5
max_files_per_commit = 50

[github]
# token from GITHUB_TOKEN env or `gh auth token`
owner = ""
repo = ""

[output]
formats = ["report", "html"]
"""
