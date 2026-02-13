"""Tests for commit classification."""

from __future__ import annotations

from datetime import datetime, timezone

from gitlore.analyzers.commit_classifier import classify_commit, classify_commits
from gitlore.models import Commit, CommitType, FileChange


def _make_commit(
    subject: str,
    body: str = "",
    files: list[FileChange] | None = None,
    parents: list[str] | None = None,
) -> Commit:
    return Commit(
        hash="abc1234",
        author_name="Test",
        author_email="test@test.com",
        author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        parents=parents or ["parent1"],
        subject=subject,
        body=body,
        files=files or [],
    )


class TestConventionalCommits:
    def test_feat(self):
        cc = classify_commit(_make_commit("feat: add login"))
        assert cc.commit_type == CommitType.FEAT
        assert cc.is_conventional
        assert cc.scope is None

    def test_fix_with_scope(self):
        cc = classify_commit(_make_commit("fix(auth): handle null token"))
        assert cc.commit_type == CommitType.FIX
        assert cc.scope == "auth"
        assert cc.is_conventional

    def test_breaking_change_bang(self):
        cc = classify_commit(_make_commit("refactor!: restructure API"))
        assert cc.commit_type == CommitType.REFACTOR
        assert cc.is_breaking
        assert cc.is_conventional

    def test_breaking_change_footer(self):
        cc = classify_commit(
            _make_commit("feat: new auth", body="BREAKING CHANGE: old API removed")
        )
        assert cc.is_breaking
        assert cc.is_conventional

    def test_chore(self):
        cc = classify_commit(_make_commit("chore: update deps"))
        assert cc.commit_type == CommitType.CHORE
        assert cc.is_conventional

    def test_docs(self):
        cc = classify_commit(_make_commit("docs: update README"))
        assert cc.commit_type == CommitType.DOCS

    def test_test(self):
        cc = classify_commit(_make_commit("test: add unit tests"))
        assert cc.commit_type == CommitType.TEST

    def test_ci(self):
        cc = classify_commit(_make_commit("ci: add GitHub Actions"))
        assert cc.commit_type == CommitType.CI

    def test_perf(self):
        cc = classify_commit(_make_commit("perf: optimize query"))
        assert cc.commit_type == CommitType.PERF

    def test_build(self):
        cc = classify_commit(_make_commit("build: update webpack config"))
        assert cc.commit_type == CommitType.BUILD

    def test_style(self):
        cc = classify_commit(_make_commit("style: fix formatting"))
        assert cc.commit_type == CommitType.STYLE

    def test_unknown_type_maps_to_unknown(self):
        cc = classify_commit(_make_commit("yolo: do stuff"))
        assert cc.commit_type == CommitType.UNKNOWN
        assert cc.is_conventional


class TestRevertDetection:
    def test_revert_subject(self):
        cc = classify_commit(
            _make_commit(
                'Revert "feat: add login"',
                body="This reverts commit abc1234.",
            )
        )
        assert cc.commit_type == CommitType.REVERT
        assert not cc.is_conventional

    def test_revert_body_only(self):
        cc = classify_commit(
            _make_commit("undo the thing", body="This reverts commit abc1234.")
        )
        assert cc.commit_type == CommitType.REVERT


class TestTicketExtraction:
    def test_jira_ticket_in_subject(self):
        cc = classify_commit(_make_commit("feat: add login PROJ-123"))
        assert cc.ticket == "PROJ-123"

    def test_github_issue_in_body(self):
        cc = classify_commit(
            _make_commit("fix: handle error", body="fixes #456")
        )
        assert cc.ticket == "#456"

    def test_ticket_prefix_pattern(self):
        cc = classify_commit(_make_commit("PROJ-123: add new endpoint"))
        assert cc.ticket == "PROJ-123"


class TestBracketModule:
    def test_bracket_module(self):
        cc = classify_commit(
            _make_commit(
                "[auth] add login endpoint",
                files=[FileChange("src/auth/login.ts", added=50, deleted=0)],
            )
        )
        assert cc.scope == "auth"
        assert not cc.is_conventional


class TestDiffBasedClassification:
    def test_test_only_files(self):
        files = [
            FileChange("tests/test_auth.py", added=30, deleted=5),
            FileChange("tests/test_login.py", added=20, deleted=0),
        ]
        cc = classify_commit(_make_commit("add tests", files=files))
        assert cc.commit_type == CommitType.TEST

    def test_docs_only_files(self):
        files = [FileChange("docs/guide.md", added=50, deleted=10)]
        cc = classify_commit(_make_commit("update guide", files=files))
        assert cc.commit_type == CommitType.DOCS

    def test_ci_only_files(self):
        files = [FileChange(".github/workflows/ci.yml", added=10, deleted=5)]
        cc = classify_commit(_make_commit("update CI", files=files))
        assert cc.commit_type == CommitType.CI

    def test_config_only_files(self):
        files = [FileChange("pyproject.toml", added=5, deleted=3)]
        cc = classify_commit(_make_commit("update config", files=files))
        assert cc.commit_type == CommitType.CHORE

    def test_refactor_heuristic(self):
        files = [FileChange("src/main.py", added=20, deleted=18)]
        cc = classify_commit(_make_commit("restructure code", files=files))
        assert cc.commit_type == CommitType.REFACTOR

    def test_unknown_fallback(self):
        files = [FileChange("src/main.py", added=100, deleted=0)]
        cc = classify_commit(_make_commit("do stuff", files=files))
        assert cc.commit_type == CommitType.UNKNOWN


class TestMergeCommit:
    def test_merge_commit(self):
        cc = classify_commit(
            _make_commit("Merge branch 'feature'", parents=["p1", "p2"])
        )
        assert cc.commit_type == CommitType.MERGE


class TestClassifyCommits:
    def test_batch_classification(self, sample_commits):
        results = classify_commits(sample_commits)
        assert len(results) == len(sample_commits)
        # First commit is conventional feat
        assert results[0].commit_type == CommitType.FEAT
        assert results[0].scope == "auth"
        # Second is fix
        assert results[1].commit_type == CommitType.FIX
        # Fourth is revert
        assert results[3].commit_type == CommitType.REVERT
        # Fifth is chore
        assert results[4].commit_type == CommitType.CHORE
