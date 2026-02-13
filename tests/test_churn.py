"""Tests for churn hotspot detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gitlore.analyzers.churn import analyze_churn
from gitlore.analyzers.commit_classifier import classify_commits
from gitlore.config import AnalysisConfig
from gitlore.models import Commit, CommitType, FileChange


def _make_commit(
    subject: str,
    files: list[FileChange],
    date: datetime | None = None,
) -> Commit:
    return Commit(
        hash="abc1234",
        author_name="Test",
        author_email="test@test.com",
        author_date=date or datetime(2026, 1, 1, tzinfo=timezone.utc),
        parents=["p1"],
        subject=subject,
        body="",
        files=files,
    )


class TestAnalyzeChurn:
    def test_basic_hotspot_detection(self, sample_commits):
        ref_date = datetime(2026, 1, 10, tzinfo=timezone.utc)
        classified = classify_commits(sample_commits)
        hotspots = analyze_churn(classified, reference_date=ref_date)

        assert len(hotspots) > 0
        # All files should be present
        all_paths = set()
        for c in sample_commits:
            for f in c.files:
                all_paths.add(f.path)
        hotspot_paths = {h.path for h in hotspots}
        assert hotspot_paths == all_paths

    def test_fix_ratio(self, sample_commits):
        ref_date = datetime(2026, 1, 10, tzinfo=timezone.utc)
        classified = classify_commits(sample_commits)
        hotspots = analyze_churn(classified, reference_date=ref_date)

        # src/auth/login.ts appears in feat, fix, and revert commits
        login_hs = next(h for h in hotspots if h.path == "src/auth/login.ts")
        assert login_hs.commit_count == 3  # feat, fix, revert
        # Fix count: only bbb2222 is classified as fix
        assert login_hs.fix_ratio > 0

    def test_score_ordering(self):
        """Files with more commits and fixes should score higher."""
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        commits = [
            _make_commit(
                "feat: add A",
                [FileChange("a.py", added=100, deleted=0)],
                date=base,
            ),
            _make_commit(
                "fix: fix A",
                [FileChange("a.py", added=5, deleted=3)],
                date=base + timedelta(hours=2),
            ),
            _make_commit(
                "fix: fix A again",
                [FileChange("a.py", added=3, deleted=2)],
                date=base + timedelta(hours=4),
            ),
            _make_commit(
                "feat: add B",
                [FileChange("b.py", added=50, deleted=0)],
                date=base,
            ),
        ]
        classified = classify_commits(commits)
        hotspots = analyze_churn(classified, reference_date=ref_date)

        a_hs = next(h for h in hotspots if h.path == "a.py")
        b_hs = next(h for h in hotspots if h.path == "b.py")
        # a.py has more commits and fixes, should score higher
        assert a_hs.score > b_hs.score

    def test_temporal_decay(self):
        """Recent commits should contribute more to weighted count."""
        ref_date = datetime(2026, 7, 1, tzinfo=timezone.utc)

        commits = [
            _make_commit(
                "feat: old change",
                [FileChange("old.py", added=10, deleted=5)],
                date=datetime(2025, 1, 1, tzinfo=timezone.utc),  # 18 months ago
            ),
            _make_commit(
                "feat: new change",
                [FileChange("new.py", added=10, deleted=5)],
                date=datetime(2026, 6, 15, tzinfo=timezone.utc),  # 2 weeks ago
            ),
        ]
        classified = classify_commits(commits)
        hotspots = analyze_churn(classified, reference_date=ref_date)

        old_hs = next(h for h in hotspots if h.path == "old.py")
        new_hs = next(h for h in hotspots if h.path == "new.py")
        # Both have 1 commit, but new should have higher weighted count
        assert new_hs.weighted_commit_count > old_hs.weighted_commit_count

    def test_empty_input(self):
        hotspots = analyze_churn([])
        assert hotspots == []

    def test_churn_ratio(self):
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "feat: add file",
                [FileChange("a.py", added=100, deleted=50)],
                date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        classified = classify_commits(commits)
        hotspots = analyze_churn(classified, reference_date=ref_date)

        a_hs = next(h for h in hotspots if h.path == "a.py")
        # churn_ratio = (100 + 50) / 1 = 150
        assert a_hs.churn_ratio == 150.0
        assert a_hs.lines_added == 100
        assert a_hs.lines_deleted == 50

    def test_binary_files_handled(self):
        ref_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "feat: add image",
                [FileChange("image.png", added=None, deleted=None)],
                date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        classified = classify_commits(commits)
        hotspots = analyze_churn(classified, reference_date=ref_date)

        img_hs = next(h for h in hotspots if h.path == "image.png")
        assert img_hs.lines_added == 0
        assert img_hs.lines_deleted == 0
        assert img_hs.commit_count == 1
