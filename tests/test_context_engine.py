"""Integration tests for the knowledge-note build/query/export flow."""

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
from gitlore.models import KnowledgeNote
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
    config.models.embedding = ""
    return config


def _seed_notes(repo: Path) -> None:
    """Insert test notes directly into the index for query/export tests."""
    store = IndexStore(str(repo))
    from datetime import UTC, datetime

    from gitlore.models import BuildMetadata, FileEdge, SourceCoverage

    notes = [
        KnowledgeNote(
            id="note1",
            text="When editing src/client.py, also update tests/test_client.py. They co-change in 100% of commits.",
            anchors=["src/client.py", "tests/test_client.py"],
            evidence_refs=["coupling: src/client.py::tests/test_client.py"],
            confidence="high",
            search_text="client retry test coupling",
        ),
        KnowledgeNote(
            id="note2",
            text="Use small, focused changes and keep retry tests up to date.",
            anchors=[],
            evidence_refs=["README.md: Conventions"],
            confidence="medium",
            search_text="conventions retry focused changes",
        ),
    ]
    edges = [
        FileEdge(src="src/client.py", dst="tests/test_client.py", edge_type="cochange", score=1.0, reason="co-change (100%)"),
        FileEdge(src="tests/test_client.py", dst="src/client.py", edge_type="cochange", score=1.0, reason="co-change (100%)"),
    ]
    metadata = BuildMetadata(
        repo_path=str(repo),
        built_at=datetime.now(UTC),
        total_commits_analyzed=3,
        note_count=2,
        source_coverage=SourceCoverage(git=True, docs=True),
    )
    store.store_build(notes, edges, metadata)
    store.close()


def _write_test_config(repo: Path) -> Path:
    path = repo / "gitlore.toml"
    path.write_text(
        """[models]
classifier = ""
embedding = ""
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


def test_build_creates_index_and_metadata(tmp_path: Path):
    """Build without agent produces an index with metadata but no notes."""
    repo = _make_repo(tmp_path)
    config = _test_config(repo)

    metadata = build_index(config)

    assert metadata.total_commits_analyzed == 3
    assert metadata.note_count == 0  # no agent = no notes
    assert (repo / ".gitlore" / "index.db").exists()

    store = IndexStore(str(repo))
    try:
        loaded = store.get_build_metadata()
    finally:
        store.close()

    assert loaded is not None
    assert loaded.total_commits_analyzed == 3
    assert loaded.source_coverage.docs is True


def test_empty_build_still_counts_as_index(tmp_path: Path):
    repo = _make_repo(tmp_path)
    config = _test_config(repo)
    build_index(config)

    store = IndexStore(str(repo))
    try:
        assert store.has_index() is True
    finally:
        store.close()

    brief = build_planning_brief(
        config,
        task="fix retry bug",
        files=["src/client.py"],
    )
    assert brief.notes == []
    assert any(item.path == "tests/test_client.py" for item in brief.related_files)

    bundle = load_export_bundle(config)
    assert bundle.notes == []


def test_planning_brief_returns_notes_and_related_files(tmp_path: Path):
    repo = _make_repo(tmp_path)
    config = _test_config(repo)
    build_index(config)
    _seed_notes(repo)

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
    assert set(payload["notes"][0]) == {"text", "refs", "confidence"}


def test_export_writes_guidance_artifacts(tmp_path: Path):
    repo = _make_repo(tmp_path)
    config = _test_config(repo)
    config.export.formats = ["agents_md", "report"]
    build_index(config)
    _seed_notes(repo)

    bundle = load_export_bundle(config)
    written = write_exports(bundle, config)

    assert len(written) == 2
    report = (repo / "gitlore-report.md").read_text()
    assert "Generated by gitlore" in report


def test_cli_and_mcp_use_same_planning_core(tmp_path: Path):
    repo = _make_repo(tmp_path)
    config_path = _write_test_config(repo)

    # Build the index (will be empty without agent)
    runner = CliRunner()
    result = runner.invoke(app, ["build", "--repo", str(repo), "--config", str(config_path)])
    assert result.exit_code == 0

    # Seed notes for query testing
    _seed_notes(repo)

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
