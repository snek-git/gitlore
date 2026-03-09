"""Tests for build-time repository investigation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from gitlore.models import AnalysisResult, ChurnHotspot
from gitlore.synthesis.evidence_tools import create_evidence_tools_server
from gitlore.synthesis.synthesizer import _load_notes_from_file


def _make_analysis() -> AnalysisResult:
    return AnalysisResult(
        hotspots=[
            ChurnHotspot(
                path="src/client.py",
                commit_count=20,
                weighted_commit_count=15.0,
                lines_added=500,
                lines_deleted=200,
                churn_ratio=0.7,
                fix_ratio=0.3,
                score=8.5,
            )
        ],
        total_commits_analyzed=100,
        analysis_date=datetime.now(UTC),
    )


def test_evidence_tools_created():
    """Evidence tools server should be created from analysis."""
    analysis = _make_analysis()
    server = create_evidence_tools_server(analysis, [])
    assert server is not None


def test_load_notes_from_jsonl(tmp_path: Path):
    """Notes should be parsed from a JSONL file."""
    notes_file = tmp_path / "notes.jsonl"
    notes_file.write_text(
        json.dumps({"text": "Test note", "anchors": ["src/foo.py"], "evidence": ["commit abc"], "confidence": "high"}) + "\n"
        + json.dumps({"text": "Another note", "anchors": [], "evidence": [], "confidence": "medium"}) + "\n"
    )

    notes = _load_notes_from_file(notes_file)
    assert len(notes) == 2
    assert notes[0].text == "Test note"
    assert notes[0].confidence == "high"
    assert notes[0].anchors == ["src/foo.py"]
    assert notes[0].evidence_refs == ["commit abc"]
    assert notes[1].text == "Another note"
    assert notes[1].confidence == "medium"


def test_load_notes_skips_bad_lines(tmp_path: Path):
    """Bad JSON lines and empty text should be skipped."""
    notes_file = tmp_path / "notes.jsonl"
    notes_file.write_text(
        "not json\n"
        + json.dumps({"text": "", "anchors": []}) + "\n"
        + json.dumps({"text": "Valid note", "anchors": ["a.py"], "confidence": "low"}) + "\n"
    )

    notes = _load_notes_from_file(notes_file)
    assert len(notes) == 1
    assert notes[0].text == "Valid note"
    assert notes[0].confidence == "low"


def test_load_notes_missing_file(tmp_path: Path):
    """Missing file should return empty list."""
    notes = _load_notes_from_file(tmp_path / "nonexistent.jsonl")
    assert notes == []


def test_run_investigation_returns_empty_without_api_key(monkeypatch):
    """Without an API key, investigation should return empty list."""
    from gitlore.config import GitloreConfig
    from gitlore.synthesis.synthesizer import run_investigation

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = GitloreConfig()
    config.models.synthesizer = "openrouter/anthropic/claude-sonnet-4.6"

    notes = run_investigation(_make_analysis(), [], config, "/tmp/repo")
    assert notes == []


def test_run_investigation_returns_empty_without_model():
    """Without a configured model, investigation should return empty list."""
    from gitlore.config import GitloreConfig
    from gitlore.synthesis.synthesizer import run_investigation

    config = GitloreConfig()
    config.models.synthesizer = ""

    notes = run_investigation(_make_analysis(), [], config, "/tmp/repo")
    assert notes == []
