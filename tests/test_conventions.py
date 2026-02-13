"""Tests for convention detection."""

from __future__ import annotations

from datetime import datetime, timezone

from gitlore.analyzers.commit_classifier import classify_commits
from gitlore.analyzers.conventions import analyze_conventions
from gitlore.models import Commit, FileChange


def _make_commit(subject: str, body: str = "") -> Commit:
    return Commit(
        hash="abc1234",
        author_name="Test",
        author_email="test@test.com",
        author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        parents=["p1"],
        subject=subject,
        body=body,
    )


class TestConventionDetection:
    def test_conventional_commits_detected(self):
        commits = [
            _make_commit("feat: add login"),
            _make_commit("fix(auth): handle null token"),
            _make_commit("chore: update deps"),
            _make_commit("refactor(api): simplify handler"),
            _make_commit("test: add unit tests"),
        ]
        classified = classify_commits(commits)
        conv = analyze_conventions(classified)

        assert conv.primary_format == "conventional_commits"
        assert conv.format_adherence == 1.0
        assert "feat" in conv.types_used
        assert "fix" in conv.types_used
        assert "auth" in conv.scopes_used
        assert "api" in conv.scopes_used

    def test_freeform_detected(self):
        commits = [
            _make_commit("Add login page"),
            _make_commit("Fix the broken tests"),
            _make_commit("Update README with examples"),
            _make_commit("Remove unused imports"),
        ]
        classified = classify_commits(commits)
        conv = analyze_conventions(classified)

        assert conv.primary_format == "freeform"

    def test_ticket_prefix_detected(self):
        commits = [
            _make_commit("PROJ-123: add endpoint"),
            _make_commit("PROJ-456: fix auth"),
            _make_commit("PROJ-789: update docs"),
        ]
        classified = classify_commits(commits)
        conv = analyze_conventions(classified)

        assert conv.primary_format == "ticket_prefix"
        assert conv.ticket_format is not None
        assert "PROJ" in conv.ticket_format

    def test_style_signals(self):
        commits = [
            _make_commit("feat: add login"),
            _make_commit("fix: handle error"),
            _make_commit("chore: update deps"),
            _make_commit("docs: update readme"),
        ]
        classified = classify_commits(commits)
        conv = analyze_conventions(classified)

        assert conv.starts_lowercase_rate == 1.0
        assert conv.ends_with_period_rate == 0.0
        assert conv.subject_under_72_rate == 1.0

    def test_empty_input(self):
        conv = analyze_conventions([])
        assert conv.primary_format == "freeform"
        assert conv.format_adherence == 0.0

    def test_detected_rules_generated(self):
        # Build enough conventional commits to trigger rules
        commits = [
            _make_commit("feat: add feature one"),
            _make_commit("fix: handle null case"),
            _make_commit("feat: add feature two"),
            _make_commit("fix(auth): fix token"),
            _make_commit("chore: update deps"),
            _make_commit("test: add unit tests"),
            _make_commit("feat: add feature three"),
            _make_commit("fix: another fix"),
            _make_commit("docs: update readme"),
            _make_commit("feat: add feature four"),
        ]
        classified = classify_commits(commits)
        conv = analyze_conventions(classified)

        assert conv.format_adherence == 1.0
        assert len(conv.detected_rules) > 0
        # Should have a rule about conventional commits
        assert any("conventional commits" in r.lower() for r in conv.detected_rules)

    def test_has_body_rate(self):
        commits = [
            _make_commit("feat: add login", body="Detailed description here"),
            _make_commit("fix: handle error", body="This fixes the issue"),
            _make_commit("chore: update deps"),
        ]
        classified = classify_commits(commits)
        conv = analyze_conventions(classified)

        assert abs(conv.has_body_rate - 2 / 3) < 0.01

    def test_imperative_mood_detection(self):
        commits = [
            _make_commit("feat: add login"),
            _make_commit("fix: handle error"),
            _make_commit("chore: update deps"),
        ]
        classified = classify_commits(commits)
        conv = analyze_conventions(classified)

        # "add", "handle", "update" are all imperative
        assert conv.imperative_mood_rate == 1.0
