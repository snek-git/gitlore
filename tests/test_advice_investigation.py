"""Tests for build-time advice-card investigation."""

from __future__ import annotations

from gitlore.config import GitloreConfig
from gitlore.models import AdvicePriority, EvidenceRef, InvestigationLead, LeadKind, QueryIntent
from gitlore.synthesis.synthesizer import investigate_leads


def test_investigate_leads_parses_advice_cards(monkeypatch):
    config = GitloreConfig()
    config.models.synthesizer = "openrouter/anthropic/claude-sonnet-4.6"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    lead = InvestigationLead(
        id="lead-1",
        kind=LeadKind.COUPLING,
        title="Coupling: src/client.py <-> tests/test_client.py",
        summary="Changes in src/client.py often require updates in tests/test_client.py.",
        anchors=["src/client.py", "tests/test_client.py"],
        applies_to=[QueryIntent.BUGFIX, QueryIntent.REFACTOR],
        priority=AdvicePriority.HIGH,
        support_count=6,
        confidence=0.8,
        evidence=[
            EvidenceRef(
                source_type="coupling",
                label="co-change",
                ref="6 co-change commits",
                excerpt="src/client.py and tests/test_client.py changed together repeatedly.",
            )
        ],
        search_text="client retry tests",
    )

    async def fake_run_agent(*args, **kwargs):  # type: ignore[no-untyped-def]
        return """
<advice_cards>
  <card priority="high" kind="validation" applies_to="bugfix,refactor">
    <text>Edits in src/client.py usually require updates in tests/test_client.py.</text>
    <anchors>
      <anchor>src/client.py</anchor>
      <anchor>tests/test_client.py</anchor>
    </anchors>
  </card>
</advice_cards>
"""

    monkeypatch.setattr("gitlore.synthesis.synthesizer._run_agent", fake_run_agent)

    cards_by_lead = investigate_leads([lead], config, "/tmp/repo")

    assert list(cards_by_lead) == ["lead-1"]
    [card] = cards_by_lead["lead-1"]
    assert card.text == "Edits in src/client.py usually require updates in tests/test_client.py."
    assert card.kind.value == "validation"
    assert card.priority == AdvicePriority.HIGH
    assert card.anchors == ["src/client.py", "tests/test_client.py"]
    assert [item.ref for item in card.evidence] == ["6 co-change commits"]


def test_investigate_leads_skips_when_no_model(monkeypatch):
    config = GitloreConfig()
    config.models.synthesizer = ""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    lead = InvestigationLead(
        id="lead-1",
        kind=LeadKind.HOTSPOT,
        title="Hotspot: src/client.py",
        summary="src/client.py is a fragile area.",
    )

    assert investigate_leads([lead], config, "/tmp/repo") == {}
