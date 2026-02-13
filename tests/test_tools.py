"""Tests for git investigation tools used in agentic synthesis."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from gitlore.synthesis.tools import (
    _create_tools,
    _err,
    _git,
    _ok,
    _truncate,
    _validate_path,
    _validate_ref,
    create_git_tools_server,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo with a couple of commits."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )

    # First commit
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("def main():\n    print('hello')\n")
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "."],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "feat: initial commit"],
        capture_output=True,
        check=True,
    )

    # Second commit
    (repo / "src" / "main.py").write_text(
        "def main():\n    print('hello world')\n\ndef helper():\n    pass\n"
    )
    (repo / "src" / "utils.py").write_text("def util():\n    return 42\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "."],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "feat: add helper and utils"],
        capture_output=True,
        check=True,
    )

    return repo


@pytest.fixture
def tools(temp_repo: Path) -> dict[str, Any]:
    """Create tool handlers bound to the temp repo, keyed by name."""
    tool_list = _create_tools(str(temp_repo))
    return {t.name: t.handler for t in tool_list}


def _get_hashes(repo: Path) -> list[str]:
    """Get commit hashes in reverse chronological order."""
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n")


# ── Unit tests: helpers ───────────────────────────────────────────────────────


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_exact_limit_unchanged(self):
        text = "x" * 4000
        assert _truncate(text) == text

    def test_over_limit_truncated(self):
        text = "x" * 5000
        result = _truncate(text)
        assert len(result) < 5000
        assert result.startswith("x" * 4000)
        assert "[... output truncated" in result

    def test_empty_string(self):
        assert _truncate("") == ""


class TestValidatePath:
    def test_normal_path_ok(self):
        _validate_path("src/main.py")

    def test_nested_path_ok(self):
        _validate_path("a/b/c/d.txt")

    def test_dotfile_ok(self):
        _validate_path(".gitignore")

    def test_traversal_rejected(self):
        with pytest.raises(ValueError, match="traversal"):
            _validate_path("../etc/passwd")

    def test_mid_traversal_rejected(self):
        with pytest.raises(ValueError, match="traversal"):
            _validate_path("src/../../etc/passwd")


class TestValidateRef:
    def test_sha_ok(self):
        _validate_ref("abc123def456")

    def test_branch_ok(self):
        _validate_ref("main")

    def test_tag_ok(self):
        _validate_ref("v1.0.0")

    def test_head_ref_ok(self):
        _validate_ref("HEAD~3")

    def test_ref_with_caret(self):
        _validate_ref("HEAD^")

    def test_invalid_ref_rejected(self):
        with pytest.raises(ValueError, match="Invalid git ref"):
            _validate_ref("abc; rm -rf /")

    def test_shell_injection_rejected(self):
        with pytest.raises(ValueError, match="Invalid git ref"):
            _validate_ref("$(whoami)")


class TestResultHelpers:
    def test_ok_returns_text_content(self):
        result = _ok("hello")
        assert result == {"content": [{"type": "text", "text": "hello"}]}

    def test_ok_truncates_long_output(self):
        result = _ok("x" * 5000)
        assert "[... output truncated" in result["content"][0]["text"]

    def test_err_returns_error(self):
        result = _err("something broke")
        assert result["is_error"] is True
        assert result["content"][0]["text"] == "something broke"


# ── Unit tests: _git helper ──────────────────────────────────────────────────


class TestGitCommand:
    def test_runs_git_command(self, temp_repo: Path):
        output = _git(str(temp_repo), "log", "--oneline")
        assert "initial commit" in output

    def test_raises_on_bad_command(self, temp_repo: Path):
        with pytest.raises(subprocess.CalledProcessError):
            _git(str(temp_repo), "log", "--nonexistent-flag-xyz")

    def test_raises_on_nonexistent_repo(self, tmp_path: Path):
        with pytest.raises(subprocess.CalledProcessError):
            _git(str(tmp_path / "nonexistent"), "log")


# ── Integration tests: tool handlers ─────────────────────────────────────────


class TestShowCommit:
    @pytest.mark.asyncio
    async def test_shows_commit_diff(self, temp_repo: Path, tools):
        hashes = _get_hashes(temp_repo)
        result = await tools["show_commit"]({"commit_hash": hashes[0]})
        text = result["content"][0]["text"]
        assert "helper" in text
        assert "utils" in text
        assert "is_error" not in result

    @pytest.mark.asyncio
    async def test_invalid_hash_returns_error(self, tools):
        result = await tools["show_commit"]({"commit_hash": "nonexistent123"})
        assert result["is_error"] is True

    @pytest.mark.asyncio
    async def test_rejects_shell_injection(self, tools):
        result = await tools["show_commit"]({"commit_hash": "$(whoami)"})
        assert result["is_error"] is True
        assert "Invalid commit hash" in result["content"][0]["text"]


class TestFileHistory:
    @pytest.mark.asyncio
    async def test_shows_file_commits(self, tools):
        result = await tools["file_history"]({"path": "src/main.py", "count": 10})
        text = result["content"][0]["text"]
        assert "initial commit" in text
        assert "helper" in text

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_error(self, tools):
        result = await tools["file_history"]({"path": "nonexistent.py", "count": 10})
        assert result["is_error"] is True

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, tools):
        result = await tools["file_history"]({"path": "../etc/passwd", "count": 10})
        assert result["is_error"] is True

    @pytest.mark.asyncio
    async def test_count_capped_at_50(self, tools):
        result = await tools["file_history"]({"path": "src/main.py", "count": 100})
        assert "is_error" not in result


class TestFileContent:
    @pytest.mark.asyncio
    async def test_reads_file_at_head(self, tools):
        result = await tools["file_content"]({"path": "src/main.py"})
        text = result["content"][0]["text"]
        assert "hello world" in text
        assert "helper" in text

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_error(self, tools):
        result = await tools["file_content"]({"path": "nope.txt"})
        assert result["is_error"] is True


class TestBlameRange:
    @pytest.mark.asyncio
    async def test_blames_line_range(self, tools):
        result = await tools["blame_range"](
            {"path": "src/main.py", "start_line": 1, "end_line": 2}
        )
        text = result["content"][0]["text"]
        assert "Test" in text  # author name

    @pytest.mark.asyncio
    async def test_too_large_range_rejected(self, tools):
        result = await tools["blame_range"](
            {"path": "src/main.py", "start_line": 1, "end_line": 300}
        )
        assert result["is_error"] is True

    @pytest.mark.asyncio
    async def test_invalid_range_rejected(self, tools):
        result = await tools["blame_range"](
            {"path": "src/main.py", "start_line": 5, "end_line": 2}
        )
        assert result["is_error"] is True


class TestDiffBetween:
    @pytest.mark.asyncio
    async def test_shows_diff(self, temp_repo: Path, tools):
        hashes = _get_hashes(temp_repo)
        result = await tools["diff_between"](
            {"from_ref": hashes[1], "to_ref": hashes[0], "path": "src/main.py"}
        )
        text = result["content"][0]["text"]
        assert "hello world" in text

    @pytest.mark.asyncio
    async def test_no_diff_returns_message(self, temp_repo: Path, tools):
        hashes = _get_hashes(temp_repo)
        result = await tools["diff_between"](
            {"from_ref": hashes[0], "to_ref": hashes[0], "path": "src/main.py"}
        )
        text = result["content"][0]["text"]
        assert "No differences" in text


class TestListChanges:
    @pytest.mark.asyncio
    async def test_lists_changed_files(self, temp_repo: Path, tools):
        hashes = _get_hashes(temp_repo)
        result = await tools["list_changes"](
            {"from_ref": hashes[1], "to_ref": hashes[0]}
        )
        text = result["content"][0]["text"]
        assert "main.py" in text
        assert "utils.py" in text


class TestRepoTree:
    @pytest.mark.asyncio
    async def test_lists_all_files(self, tools):
        result = await tools["repo_tree"]({})
        text = result["content"][0]["text"]
        assert "src/main.py" in text
        assert "src/utils.py" in text
        assert "README.md" in text


# ── MCP server creation ──────────────────────────────────────────────────────


class TestCreateServer:
    def test_creates_server_with_correct_structure(self, temp_repo: Path):
        server = create_git_tools_server(str(temp_repo))
        assert server["type"] == "sdk"
        assert server["name"] == "git-tools"
        assert "instance" in server

    def test_all_tool_names_registered(self, temp_repo: Path):
        tool_list = _create_tools(str(temp_repo))
        names = {t.name for t in tool_list}
        assert names == {
            "show_commit",
            "file_history",
            "file_content",
            "blame_range",
            "diff_between",
            "list_changes",
            "repo_tree",
        }
