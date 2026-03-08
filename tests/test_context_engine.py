"""Integration tests for the advice-card build/query/export flow."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from gitlore.build import build_index
from gitlore.cli import app
from gitlore.config import GitloreConfig
from gitlore.export import load_export_bundle, write_exports
from gitlore.index import IndexStore
from gitlore.mcp_server import create_mcp_server
from gitlore.query import build_planning_brief, render_planning_brief


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@test.com")
    _git(repo, "config", "user.name", "Test")

    _write(
        repo / "README.md",
        """# Test Repo

## Build & Run

uv sync --all-extras
uv run pytest -q

## Conventions

Use small, focused changes and keep retry tests up to date.
""",
    )
    _write(
        repo / "src" / "client.py",
        "def retry(count: int) -> int:\n    return count\n",
    )
    _write(
        repo / "tests" / "test_client.py",
        "from src.client import retry\n\n\ndef test_retry():\n    assert retry(1) == 1\n",
    )
    _commit(repo, "feat(client): add retry helper")

    _write(
        repo / "src" / "client.py",
        "def retry(count: int) -> int:\n    if count < 0:\n        return 0\n    return count\n",
    )
    _write(
        repo / "tests" / "test_client.py",
        "from src.client import retry\n\n\ndef test_retry():\n    assert retry(1) == 1\n\n\ndef test_retry_negative():\n    assert retry(-1) == 0\n",
    )
    _commit(repo, "fix(client): handle negative retry count")

    _write(
        repo / "src" / "client.py",
        "def retry(count: int, limit: int = 5) -> int:\n    if count < 0:\n        return 0\n    return min(count, limit)\n",
    )
    _write(
        repo / "tests" / "test_client.py",
        "from src.client import retry\n\n\ndef test_retry():\n    assert retry(1) == 1\n\n\ndef test_retry_negative():\n    assert retry(-1) == 0\n\n\ndef test_retry_limit():\n    assert retry(10, limit=3) == 3\n",
    )
    _commit(repo, "refactor(client): add retry limit")

    return repo


def _test_config(repo: Path) -> GitloreConfig:
    config = GitloreConfig()
    config.repo_path = str(repo)
    config.sources.github = False
    config.build.min_shared_commits = 1
    config.build.min_coupling_lift = 1.0
    config.models.synthesizer = ""
    config.models.compressor = ""
    config.models.embedding = ""
    return config


def _write_test_config(repo: Path) -> Path:
    path = repo / "gitlore.toml"
    path.write_text(
        """[models]
classifier = ""
embedding = ""
compressor = ""
synthesizer = ""

[build]
since_months = 12
half_life_days = 180
min_coupling_confidence = 0.25
min_coupling_lift = 1.0
max_files_per_commit = 50
min_shared_commits = 1

[sources]
github = false
docs = true

[query]
default_format = "summary"
max_items = 12
max_tokens = 1200
semantic = false
"""
    )
    return path


def test_build_creates_advice_card_index_and_metadata(tmp_path: Path):
    repo = _make_repo(tmp_path)
    config = _test_config(repo)

    metadata = build_index(config)

    assert metadata.total_commits_analyzed == 3
    assert metadata.card_count > 0
    assert (repo / ".gitlore" / "index.db").exists()

    store = IndexStore(str(repo))
    try:
        assert store.has_index() is True
        loaded = store.get_build_metadata()
        cards = store.load_cards()
    finally:
        store.close()

    assert loaded is not None
    assert loaded.total_commits_analyzed == 3
    assert loaded.source_coverage.docs is True
    assert any("tests/test_client.py" in card.text for card in cards)


def test_planning_brief_returns_notes_and_related_files(tmp_path: Path):
    repo = _make_repo(tmp_path)
    config = _test_config(repo)
    build_index(config)

    brief = build_planning_brief(
        config,
        task="fix retry bug",
        files=["src/client.py"],
    )

    assert brief.notes
    assert any(item.path == "tests/test_client.py" for item in brief.related_files)
    rendered = render_planning_brief(brief, format_name="json")
    payload = json.loads(rendered)
    assert "summary" in payload
    assert payload["notes"]
    assert set(payload["notes"][0]) == {"text", "refs", "priority"}


def test_export_writes_short_guidance_artifacts(tmp_path: Path):
    repo = _make_repo(tmp_path)
    config = _test_config(repo)
    config.export.formats = ["agents_md", "report"]
    build_index(config)

    bundle = load_export_bundle(config)
    written = write_exports(bundle, config)

    assert len(written) == 2
    agents = (repo / "AGENTS.md").read_text()
    report = (repo / "gitlore-report.md").read_text()
    assert "Refs:" in agents or "focused changes" in agents
    assert "Generated by gitlore" in report


def test_cli_and_mcp_use_same_planning_core(tmp_path: Path):
    repo = _make_repo(tmp_path)
    config_path = _write_test_config(repo)
    runner = CliRunner()

    result = runner.invoke(app, ["build", "--repo", str(repo), "--config", str(config_path)])
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "advise",
            "--repo",
            str(repo),
            "--config",
            str(config_path),
            "--task",
            "fix retry bug",
            "--files",
            "src/client.py",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["notes"]

    config = _test_config(repo)
    server = create_mcp_server(config)
    tool_result = asyncio.run(
        server._tool_manager.call_tool(
            "get_planning_brief",
            {
                "task": "fix retry bug",
                "files": ["src/client.py"],
                "format": "json",
            },
        )
    )
    tool_payload = json.loads(tool_result)
    assert tool_payload["notes"]
    assert tool_payload["summary"] == payload["summary"]
