"""Shared fixtures for gitlore tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gitlore.models import Commit, FileChange


@pytest.fixture
def sample_commits() -> list[Commit]:
    """A small set of commits for testing analyzers."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Commit(
            hash="aaa1111",
            author_name="Alice",
            author_email="alice@test.com",
            author_date=base,
            parents=[],
            subject="feat(auth): add login endpoint",
            body="",
            files=[
                FileChange("src/auth/login.ts", added=100, deleted=0),
                FileChange("src/auth/session.ts", added=50, deleted=0),
                FileChange("tests/auth/test_login.ts", added=30, deleted=0),
            ],
        ),
        Commit(
            hash="bbb2222",
            author_name="Alice",
            author_email="alice@test.com",
            author_date=base.replace(hour=1),
            parents=["aaa1111"],
            subject="fix(auth): handle null token",
            body="",
            files=[
                FileChange("src/auth/login.ts", added=5, deleted=2),
                FileChange("src/auth/session.ts", added=3, deleted=1),
            ],
        ),
        Commit(
            hash="ccc3333",
            author_name="Bob",
            author_email="bob@test.com",
            author_date=base.replace(day=2),
            parents=["bbb2222"],
            subject="feat(api): add user profile endpoint",
            body="",
            files=[
                FileChange("src/api/profile.ts", added=80, deleted=0),
                FileChange("src/models/user.ts", added=20, deleted=5),
            ],
        ),
        Commit(
            hash="ddd4444",
            author_name="Alice",
            author_email="alice@test.com",
            author_date=base.replace(day=3),
            parents=["ccc3333"],
            subject='Revert "feat(auth): add login endpoint"',
            body="This reverts commit aaa1111.",
            files=[
                FileChange("src/auth/login.ts", added=0, deleted=100),
                FileChange("src/auth/session.ts", added=0, deleted=50),
            ],
        ),
        Commit(
            hash="eee5555",
            author_name="Bob",
            author_email="bob@test.com",
            author_date=base.replace(day=4),
            parents=["ddd4444"],
            subject="chore: update dependencies",
            body="",
            files=[
                FileChange("package.json", added=5, deleted=5),
                FileChange("package-lock.json", added=500, deleted=500),
            ],
        ),
    ]
