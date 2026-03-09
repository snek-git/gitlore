"""Configuration loading and defaults."""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    classifier: str = "openrouter/google/gemini-3.1-flash-lite-preview"
    embedding: str = "openrouter/openai/text-embedding-3-small"
    synthesizer: str = "sonnet"


@dataclass
class BuildConfig:
    since_months: int = 12
    half_life_days: float = 180
    min_coupling_confidence: float = 0.25
    min_coupling_lift: float = 1.5
    max_files_per_commit: int = 50
    min_shared_commits: int = 3


@dataclass
class SourcesConfig:
    github: bool = True
    docs: bool = True


@dataclass
class QueryConfig:
    default_format: str = "summary"
    max_items: int = 12
    max_tokens: int = 1200
    semantic: bool = True


@dataclass
class ExportConfig:
    formats: list[str] = field(default_factory=lambda: ["report", "html"])


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
                import re

                match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
                if match:
                    return match.group(1), match.group(2)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return self.owner, self.repo


@dataclass
class GitloreConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    repo_path: str = "."

    @classmethod
    def load(cls, path: Path | None = None) -> GitloreConfig:
        """Load config from gitlore.toml, falling back to defaults."""
        if path is None:
            path = Path("gitlore.toml")
        if not path.exists():
            return cls()

        with open(path, "rb") as file_obj:
            raw = tomllib.load(file_obj)

        config = cls()

        models = raw.get("models", {})
        config.models = ModelConfig(
            classifier=models.get("classifier", config.models.classifier),
            embedding=models.get("embedding", config.models.embedding),
            synthesizer=models.get("synthesizer", config.models.synthesizer),
        )

        build = raw.get("build", {})
        config.build = BuildConfig(
            since_months=build.get("since_months", config.build.since_months),
            half_life_days=build.get("half_life_days", config.build.half_life_days),
            min_coupling_confidence=build.get(
                "min_coupling_confidence",
                config.build.min_coupling_confidence,
            ),
            min_coupling_lift=build.get(
                "min_coupling_lift",
                config.build.min_coupling_lift,
            ),
            max_files_per_commit=build.get(
                "max_files_per_commit",
                config.build.max_files_per_commit,
            ),
            min_shared_commits=build.get(
                "min_shared_commits",
                config.build.min_shared_commits,
            ),
        )

        sources = raw.get("sources", {})
        config.sources = SourcesConfig(
            github=sources.get("github", config.sources.github),
            docs=sources.get("docs", config.sources.docs),
        )

        query = raw.get("query", {})
        config.query = QueryConfig(
            default_format=query.get("default_format", config.query.default_format),
            max_items=query.get("max_items", config.query.max_items),
            max_tokens=query.get("max_tokens", config.query.max_tokens),
            semantic=query.get("semantic", config.query.semantic),
        )

        export = raw.get("export", {})
        config.export = ExportConfig(
            formats=export.get("formats", config.export.formats),
        )

        github = raw.get("github", {})
        config.github = GitHubConfig(
            owner=github.get("owner", ""),
            repo=github.get("repo", ""),
            token=github.get("token", ""),
        )

        return config


DEFAULT_CONFIG_TEMPLATE = """\
[models]
classifier = "openrouter/google/gemini-3.1-flash-lite-preview"
embedding = "openrouter/openai/text-embedding-3-small"
synthesizer = "sonnet"

[build]
since_months = 12
half_life_days = 180
min_coupling_confidence = 0.25
min_coupling_lift = 1.5
max_files_per_commit = 50
min_shared_commits = 3

[sources]
github = true
docs = true

[query]
default_format = "summary"
max_items = 12
max_tokens = 1200
semantic = true

[github]
# token from GITHUB_TOKEN env or `gh auth token`
owner = ""
repo = ""

[export]
formats = ["report", "html"]
"""
