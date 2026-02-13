"""Tests for revert chain detection."""

from __future__ import annotations

from datetime import datetime, timezone

from gitlore.analyzers.reverts import detect_reverts
from gitlore.models import Commit, FileChange


def _make_commit(
    hash: str,
    subject: str,
    body: str = "",
    files: list[FileChange] | None = None,
) -> Commit:
    return Commit(
        hash=hash,
        author_name="Test",
        author_email="test@test.com",
        author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        parents=["parent1"],
        subject=subject,
        body=body,
        files=files or [],
    )


class TestDetectReverts:
    def test_simple_revert(self):
        commits = [
            _make_commit(
                "aaa1111",
                "feat: add login",
                files=[FileChange("src/login.py", added=50, deleted=0)],
            ),
            _make_commit(
                "bbb2222",
                'Revert "feat: add login"',
                body="This reverts commit aaa1111.",
            ),
        ]
        chains = detect_reverts(commits)
        assert len(chains) == 1
        chain = chains[0]
        assert chain.original_hash == "aaa1111"
        assert chain.original_subject == "feat: add login"
        assert chain.revert_hashes == ["bbb2222"]
        assert chain.is_effectively_reverted
        assert chain.depth == 1
        assert chain.original_author == "Test"
        assert chain.files == ["src/login.py"]

    def test_revert_of_revert(self):
        commits = [
            _make_commit("aaa1111", "feat: add login"),
            _make_commit(
                "bbb2222",
                'Revert "feat: add login"',
                body="This reverts commit aaa1111.",
            ),
            _make_commit(
                "ccc3333",
                'Revert "Revert "feat: add login""',
                body="This reverts commit bbb2222.",
            ),
        ]
        chains = detect_reverts(commits)
        assert len(chains) == 1
        chain = chains[0]
        assert chain.original_hash == "aaa1111"
        assert chain.revert_hashes == ["bbb2222", "ccc3333"]
        assert not chain.is_effectively_reverted  # even number of reverts
        assert chain.depth == 2

    def test_no_reverts(self):
        commits = [
            _make_commit("aaa1111", "feat: add login"),
            _make_commit("bbb2222", "fix: handle error"),
        ]
        chains = detect_reverts(commits)
        assert len(chains) == 0

    def test_revert_missing_original(self):
        """Revert of a commit not in our list."""
        commits = [
            _make_commit(
                "bbb2222",
                'Revert "feat: add login"',
                body="This reverts commit deadbeef123.",
            ),
        ]
        chains = detect_reverts(commits)
        assert len(chains) == 1
        chain = chains[0]
        assert chain.original_hash == "deadbeef123"
        assert chain.original_subject == "<unknown>"

    def test_subject_match_fallback(self):
        """When there's no hash in the body, match by subject."""
        commits = [
            _make_commit("aaa1111", "feat: add login"),
            _make_commit("bbb2222", 'Revert "feat: add login"'),
        ]
        chains = detect_reverts(commits)
        assert len(chains) == 1
        assert chains[0].original_hash == "aaa1111"

    def test_with_sample_commits(self, sample_commits):
        """Test with the shared fixture."""
        chains = detect_reverts(sample_commits)
        assert len(chains) == 1
        chain = chains[0]
        assert chain.original_hash == "aaa1111"
        assert "bbb2222" not in chain.revert_hashes  # bbb2222 is a fix, not revert
        assert "ddd4444" in chain.revert_hashes

    def test_multiple_independent_reverts(self):
        commits = [
            _make_commit("aaa1111", "feat: add login"),
            _make_commit("bbb2222", "feat: add signup"),
            _make_commit(
                "ccc3333",
                'Revert "feat: add login"',
                body="This reverts commit aaa1111.",
            ),
            _make_commit(
                "ddd4444",
                'Revert "feat: add signup"',
                body="This reverts commit bbb2222.",
            ),
        ]
        chains = detect_reverts(commits)
        assert len(chains) == 2
        hashes = {c.original_hash for c in chains}
        assert hashes == {"aaa1111", "bbb2222"}
