"""Tests for fix-after chain detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gitlore.analyzers.fix_after import detect_fix_after
from gitlore.models import Commit, FileChange, FixAfterTier


def _make_commit(
    hash: str,
    subject: str,
    author_email: str = "alice@test.com",
    date: datetime | None = None,
    files: list[FileChange] | None = None,
) -> Commit:
    return Commit(
        hash=hash,
        author_name="Alice" if "alice" in author_email else "Bob",
        author_email=author_email,
        author_date=date or datetime(2026, 1, 1, tzinfo=timezone.utc),
        parents=["parent1"],
        subject=subject,
        body="",
        files=files or [],
    )


class TestImmediateTier:
    def test_same_author_same_files_within_30_min(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "minor tweak",
                date=base + timedelta(minutes=15),
                files=[FileChange("src/login.py", added=2, deleted=1)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 1
        chain = chains[0]
        assert chain.original_hash == "aaa"
        assert chain.fixup_hashes == ["bbb"]
        assert chain.tier == FixAfterTier.IMMEDIATE

    def test_no_overlap_no_match(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "feat: add signup",
                date=base + timedelta(minutes=15),
                files=[FileChange("src/signup.py", added=50, deleted=0)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 0


class TestFollowupTier:
    def test_same_author_fix_keyword_within_4_hours(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "fix typo in login",
                date=base + timedelta(hours=2),
                files=[FileChange("src/login.py", added=1, deleted=1)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 1
        assert chains[0].tier == FixAfterTier.FOLLOWUP

    def test_no_keyword_no_followup(self):
        """Without fix keywords, 2-hour gap doesn't match followup tier."""
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "add more features to login",
                date=base + timedelta(hours=2),
                files=[FileChange("src/login.py", added=30, deleted=5)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 0

    def test_oops_keyword(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "oops forgot the import",
                date=base + timedelta(hours=1),
                files=[FileChange("src/login.py", added=1, deleted=0)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 1
        assert chains[0].tier == FixAfterTier.FOLLOWUP


class TestDelayedTier:
    def test_different_author_strong_fix_signal(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                author_email="alice@test.com",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "fix bug in login validation",
                author_email="bob@test.com",
                date=base + timedelta(days=3),
                files=[FileChange("src/login.py", added=5, deleted=2)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 1
        assert chains[0].tier == FixAfterTier.DELAYED

    def test_beyond_7_days_no_match(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "fix bug in login",
                author_email="bob@test.com",
                date=base + timedelta(days=10),
                files=[FileChange("src/login.py", added=5, deleted=2)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 0

    def test_hotfix_keyword(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add payment flow",
                author_email="alice@test.com",
                date=base,
                files=[FileChange("src/payment.py", added=100, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "hotfix: payment crash",
                author_email="bob@test.com",
                date=base + timedelta(days=2),
                files=[FileChange("src/payment.py", added=3, deleted=1)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 1
        assert chains[0].tier == FixAfterTier.DELAYED


class TestChainBuilding:
    def test_multiple_fixups_on_one_original(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "quick adjustment",
                date=base + timedelta(minutes=10),
                files=[FileChange("src/login.py", added=2, deleted=1)],
            ),
            _make_commit(
                "ccc",
                "another tweak",
                date=base + timedelta(minutes=20),
                files=[FileChange("src/login.py", added=1, deleted=1)],
            ),
        ]
        chains = detect_fix_after(commits)
        # bbb is fixup of aaa, ccc may be fixup of aaa too
        assert len(chains) >= 1
        main_chain = next(c for c in chains if c.original_hash == "aaa")
        assert len(main_chain.fixup_hashes) >= 1

    def test_git_fixup_marker(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "fixup! feat: add login",
                date=base + timedelta(hours=1),
                files=[FileChange("src/login.py", added=2, deleted=1)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 1
        assert chains[0].original_hash == "aaa"
        assert chains[0].fixup_hashes == ["bbb"]

    def test_time_span_set(self):
        base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        commits = [
            _make_commit(
                "aaa",
                "feat: add login",
                date=base,
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb",
                "quick fix",
                date=base + timedelta(minutes=15),
                files=[FileChange("src/login.py", added=2, deleted=1)],
            ),
        ]
        chains = detect_fix_after(commits)
        assert len(chains) == 1
        assert chains[0].time_span == timedelta(minutes=15)


class TestWithSampleFixture:
    def test_sample_commits(self, sample_commits):
        """The sample fixture has aaa1111 followed by bbb2222 (fix, same author,
        same files, 1 hour later) -- should be detected as fix-after."""
        chains = detect_fix_after(sample_commits)
        # bbb2222 fixes aaa1111 (same author, same files, 1 hour gap, fix keyword)
        fix_chain = next(
            (c for c in chains if c.original_hash == "aaa1111"), None
        )
        assert fix_chain is not None
        assert "bbb2222" in fix_chain.fixup_hashes
