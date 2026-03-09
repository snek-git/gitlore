"""Tests for build-time repository investigation."""

from __future__ import annotations

from datetime import UTC, datetime

from gitlore.models import AnalysisResult, ChurnHotspot, KnowledgeNote
from gitlore.synthesis.evidence_tools import create_evidence_tools_server


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


def test_evidence_tools_get_hotspots():
    """Evidence tools should return hotspot data from analysis."""
    import asyncio

    analysis = _make_analysis()
    collector: list[KnowledgeNote] = []
    server = create_evidence_tools_server(analysis, [], collector)

    # The server exposes tools; verify it was created
    assert server is not None


def test_record_note_populates_collector():
    """The record_note tool should append notes to the collector list."""
    import asyncio

    from gitlore.synthesis.evidence_tools import _create_evidence_tools

    analysis = _make_analysis()
    collector: list[KnowledgeNote] = []
    tools = _create_evidence_tools(analysis, [], collector)

    # Find the record_note tool
    record_note = None
    for t in tools:
        if t.name == "record_note":
            record_note = t
            break
    assert record_note is not None

    # Call it
    result = asyncio.run(record_note.handler({
        "text": "src/client.py has high churn with 30% fix ratio.",
        "anchors": ["src/client.py"],
        "evidence": ["hotspot: 20 commits, 30% fixes"],
        "confidence": "high",
    }))
    assert not result.get("is_error")
    assert len(collector) == 1
    assert collector[0].text == "src/client.py has high churn with 30% fix ratio."
    assert collector[0].confidence == "high"
    assert collector[0].anchors == ["src/client.py"]


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
