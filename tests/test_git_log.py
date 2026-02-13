"""Tests for the git log extractor."""

from __future__ import annotations

from gitlore.extractors.git_log import (
    _expand_rename_path,
    _parse_commit_block,
    _parse_numstat_line,
    build_rename_map,
    resolve_path,
)
from gitlore.models import Commit, FileChange


class TestExpandRenamePath:
    def test_simple_rename(self):
        assert _expand_rename_path("{old.py => new.py}", use_old=True) == "old.py"
        assert _expand_rename_path("{old.py => new.py}", use_old=False) == "new.py"

    def test_rename_in_directory(self):
        path = "src/{old.py => new.py}"
        assert _expand_rename_path(path, use_old=True) == "src/old.py"
        assert _expand_rename_path(path, use_old=False) == "src/new.py"

    def test_rename_with_nested_path(self):
        path = "src/{auth/login.py => auth/signin.py}/utils"
        assert _expand_rename_path(path, use_old=True) == "src/auth/login.py/utils"
        assert _expand_rename_path(path, use_old=False) == "src/auth/signin.py/utils"

    def test_directory_rename(self):
        path = "src/{old_dir => new_dir}/file.py"
        assert _expand_rename_path(path, use_old=True) == "src/old_dir/file.py"
        assert _expand_rename_path(path, use_old=False) == "src/new_dir/file.py"


class TestParseNumstatLine:
    def test_normal_line(self):
        fc = _parse_numstat_line("10\t5\tsrc/main.py")
        assert fc is not None
        assert fc.path == "src/main.py"
        assert fc.added == 10
        assert fc.deleted == 5
        assert fc.old_path is None

    def test_binary_file(self):
        fc = _parse_numstat_line("-\t-\timage.png")
        assert fc is not None
        assert fc.path == "image.png"
        assert fc.added is None
        assert fc.deleted is None

    def test_rename_with_braces(self):
        fc = _parse_numstat_line("5\t3\tsrc/{old.py => new.py}")
        assert fc is not None
        assert fc.path == "src/new.py"
        assert fc.old_path == "src/old.py"
        assert fc.added == 5
        assert fc.deleted == 3

    def test_full_rename_without_braces(self):
        fc = _parse_numstat_line("0\t0\told_file.py => new_file.py")
        assert fc is not None
        assert fc.path == "new_file.py"
        assert fc.old_path == "old_file.py"

    def test_invalid_line(self):
        assert _parse_numstat_line("not a numstat line") is None
        assert _parse_numstat_line("") is None


class TestParseCommitBlock:
    def test_basic_commit(self):
        raw = (
            "abc1234\x1fAlice\x1falice@test.com\x1f2026-01-15T10:00:00+00:00"
            "\x1fparent1\x1ffeat: add feature\x1fbody text\n"
            "10\t5\tsrc/main.py\n"
            "3\t0\ttests/test_main.py"
        )
        commit = _parse_commit_block(raw)
        assert commit is not None
        assert commit.hash == "abc1234"
        assert commit.author_name == "Alice"
        assert commit.author_email == "alice@test.com"
        assert commit.subject == "feat: add feature"
        assert commit.body == "body text"
        assert commit.parents == ["parent1"]
        assert len(commit.files) == 2
        assert commit.files[0].path == "src/main.py"
        assert commit.files[0].added == 10
        assert commit.files[0].deleted == 5
        assert commit.files[1].path == "tests/test_main.py"

    def test_commit_no_parents(self):
        raw = (
            "abc1234\x1fAlice\x1falice@test.com\x1f2026-01-15T10:00:00+00:00"
            "\x1f\x1finitial commit\x1f"
        )
        commit = _parse_commit_block(raw)
        assert commit is not None
        assert commit.parents == []

    def test_commit_multiple_parents(self):
        raw = (
            "abc1234\x1fAlice\x1falice@test.com\x1f2026-01-15T10:00:00+00:00"
            "\x1fparent1 parent2\x1fmerge commit\x1f"
        )
        commit = _parse_commit_block(raw)
        assert commit is not None
        assert commit.parents == ["parent1", "parent2"]
        assert commit.is_merge

    def test_empty_block(self):
        assert _parse_commit_block("") is None
        assert _parse_commit_block("no field separators") is None

    def test_commit_with_empty_body(self):
        raw = (
            "abc1234\x1fAlice\x1falice@test.com\x1f2026-01-15T10:00:00+00:00"
            "\x1fparent1\x1ffeat: something\x1f"
        )
        commit = _parse_commit_block(raw)
        assert commit is not None
        assert commit.body == ""


class TestBuildRenameMap:
    def test_simple_rename(self):
        commits = [
            Commit(
                hash="a",
                author_name="A",
                author_email="a@t.com",
                author_date=__import__("datetime").datetime(2026, 1, 1),
                parents=[],
                subject="rename",
                body="",
                files=[FileChange("new.py", old_path="old.py", added=0, deleted=0)],
            )
        ]
        rmap = build_rename_map(commits)
        assert rmap == {"old.py": "new.py"}

    def test_chain_rename(self):
        from datetime import datetime

        commits = [
            Commit(
                hash="a",
                author_name="A",
                author_email="a@t.com",
                author_date=datetime(2026, 1, 1),
                parents=[],
                subject="rename 1",
                body="",
                files=[FileChange("b.py", old_path="a.py", added=0, deleted=0)],
            ),
            Commit(
                hash="b",
                author_name="A",
                author_email="a@t.com",
                author_date=datetime(2026, 1, 2),
                parents=["a"],
                subject="rename 2",
                body="",
                files=[FileChange("c.py", old_path="b.py", added=0, deleted=0)],
            ),
        ]
        rmap = build_rename_map(commits)
        assert rmap["a.py"] == "c.py"
        assert rmap["b.py"] == "c.py"


class TestResolvePath:
    def test_no_rename(self):
        assert resolve_path("file.py", {}) == "file.py"

    def test_single_rename(self):
        assert resolve_path("old.py", {"old.py": "new.py"}) == "new.py"

    def test_chain(self):
        rmap = {"a.py": "b.py", "b.py": "c.py"}
        assert resolve_path("a.py", rmap) == "c.py"

    def test_circular_protection(self):
        rmap = {"a.py": "b.py", "b.py": "a.py"}
        # Should not infinite loop
        result = resolve_path("a.py", rmap)
        assert result in ("a.py", "b.py")
