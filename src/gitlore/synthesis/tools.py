"""Git investigation tools for agentic synthesis.

Provides 7 read-only git tools as an in-process MCP server for the Claude Agent SDK.
The agent uses these to investigate patterns discovered by the statistical analysis
before writing synthesis rules.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

_MAX_OUTPUT = 4000
_GIT_TIMEOUT = 10


def _git(repo_path: str, *args: str) -> str:
    """Run a read-only git command and return stdout."""
    result = subprocess.run(
        ["git", "-C", repo_path, *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    return result.stdout


def _truncate(text: str) -> str:
    """Truncate output to _MAX_OUTPUT characters with notice."""
    if len(text) <= _MAX_OUTPUT:
        return text
    return text[:_MAX_OUTPUT] + "\n\n[... output truncated, showing first 4000 chars]"


def _validate_path(path: str) -> None:
    """Reject path traversal attempts."""
    if ".." in path.split("/"):
        raise ValueError(f"Path traversal not allowed: {path}")


def _validate_ref(ref: str) -> None:
    """Validate a git ref (hash, branch, tag) contains only safe characters."""
    if not re.match(r"^[a-zA-Z0-9_./@^~{}\-]+$", ref):
        raise ValueError(f"Invalid git ref: {ref}")


def _ok(text: str) -> dict[str, Any]:
    """Return a success tool result."""
    return {"content": [{"type": "text", "text": _truncate(text)}]}


def _err(message: str) -> dict[str, Any]:
    """Return an error tool result with actionable suggestion."""
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def _create_tools(repo_path: str) -> list:
    """Create all git tool definitions bound to a repository path.

    Returns a list of SdkMcpTool objects. Each has .name, .handler, .description
    attributes. The .handler is the original async function, directly callable
    with a dict of arguments.
    """

    @tool(
        "show_commit",
        "Show the full diff and metadata of a git commit. Use this to investigate "
        "reverts, fix-after chains, or understand what a specific change did. "
        "Returns the commit message, file stats, and patch diff. "
        "Pass a full or abbreviated commit SHA.",
        {"commit_hash": str},
    )
    async def show_commit(args: dict[str, Any]) -> dict[str, Any]:
        try:
            _validate_ref(args["commit_hash"])
            output = _git(
                repo_path, "show", args["commit_hash"],
                "--stat", "--patch", "--no-color",
            )
            return _ok(output)
        except ValueError as e:
            return _err(f"Invalid commit hash: {e}. Use a full or abbreviated SHA.")
        except subprocess.CalledProcessError:
            return _err(
                f"Commit not found: {args['commit_hash']}. Check the hash and try again."
            )
        except subprocess.TimeoutExpired:
            return _err("Command timed out. The diff may be too large.")

    @tool(
        "file_history",
        "Show recent commits that touched a specific file. Use this to understand "
        "why a file churns or what kinds of changes it receives. Returns one-line "
        "commit summaries, following renames. Count defaults to 20, max 50.",
        {"path": str, "count": int},
    )
    async def file_history(args: dict[str, Any]) -> dict[str, Any]:
        try:
            _validate_path(args["path"])
            count = min(args.get("count", 20) or 20, 50)
            output = _git(
                repo_path, "log", "--oneline", "--follow",
                f"-n{count}", "--", args["path"],
            )
            if not output.strip():
                return _err(f"No commits found for {args['path']}. Check the file path.")
            return _ok(output)
        except ValueError as e:
            return _err(f"Invalid path: {e}")
        except subprocess.CalledProcessError:
            return _err(f"Could not get history for {args['path']}. File may not exist.")
        except subprocess.TimeoutExpired:
            return _err("Command timed out.")

    @tool(
        "file_content",
        "Read the current content of a file at HEAD. Use this to understand what a "
        "file does, its structure, and why it might be coupled with other files. "
        "Returns the raw file content as text. All paths are relative to repo root.",
        {"path": str},
    )
    async def file_content(args: dict[str, Any]) -> dict[str, Any]:
        try:
            _validate_path(args["path"])
            output = _git(repo_path, "show", f"HEAD:{args['path']}")
            return _ok(output)
        except ValueError as e:
            return _err(f"Invalid path: {e}")
        except subprocess.CalledProcessError:
            return _err(f"File not found at HEAD: {args['path']}. Check the path.")
        except subprocess.TimeoutExpired:
            return _err("Command timed out. The file may be too large.")

    @tool(
        "blame_range",
        "Show git blame for a specific line range of a file. Use this to see who "
        "last modified each line and when. Helps understand authorship patterns "
        "and recent changes in hotspot regions. Keep ranges under 200 lines.",
        {"path": str, "start_line": int, "end_line": int},
    )
    async def blame_range(args: dict[str, Any]) -> dict[str, Any]:
        try:
            _validate_path(args["path"])
            start = args["start_line"]
            end = args["end_line"]
            if end < start or end - start > 200:
                return _err("Line range too large or invalid. Keep ranges under 200 lines.")
            output = _git(
                repo_path, "blame",
                f"-L{start},{end}", "--", args["path"],
            )
            return _ok(output)
        except ValueError as e:
            return _err(f"Invalid path: {e}")
        except subprocess.CalledProcessError:
            return _err(
                f"Could not blame {args['path']}. Check the file path and line range."
            )
        except subprocess.TimeoutExpired:
            return _err("Command timed out.")

    @tool(
        "diff_between",
        "Show the diff of a specific file between two git refs (commits, tags, or "
        "branches). Use this to see what changed in a file between two points in "
        "time. Returns a unified diff. Refs can be commit SHAs, branch names, or tags.",
        {"from_ref": str, "to_ref": str, "path": str},
    )
    async def diff_between(args: dict[str, Any]) -> dict[str, Any]:
        try:
            _validate_ref(args["from_ref"])
            _validate_ref(args["to_ref"])
            _validate_path(args["path"])
            output = _git(
                repo_path, "diff",
                f"{args['from_ref']}..{args['to_ref']}",
                "--", args["path"],
            )
            if not output.strip():
                return _ok("No differences found between the refs for this file.")
            return _ok(output)
        except ValueError as e:
            return _err(f"Invalid input: {e}")
        except subprocess.CalledProcessError:
            return _err("Could not compute diff. Check the refs and file path.")
        except subprocess.TimeoutExpired:
            return _err("Command timed out. The diff may be too large.")

    @tool(
        "list_changes",
        "List all files changed between two git refs with insertion/deletion stats. "
        "Use this to get an overview of what changed across a range of commits. "
        "Returns a stat summary showing files and line counts.",
        {"from_ref": str, "to_ref": str},
    )
    async def list_changes(args: dict[str, Any]) -> dict[str, Any]:
        try:
            _validate_ref(args["from_ref"])
            _validate_ref(args["to_ref"])
            output = _git(
                repo_path, "diff", "--stat",
                f"{args['from_ref']}..{args['to_ref']}",
            )
            if not output.strip():
                return _ok("No changes between the refs.")
            return _ok(output)
        except ValueError as e:
            return _err(f"Invalid input: {e}")
        except subprocess.CalledProcessError:
            return _err("Could not list changes. Check the refs.")
        except subprocess.TimeoutExpired:
            return _err("Command timed out.")

    @tool(
        "repo_tree",
        "List all files tracked in the repository at HEAD. Use this to understand "
        "the overall directory structure and find correct file paths before using "
        "other tools. Returns a flat list of all tracked file paths.",
        {},
    )
    async def repo_tree(args: dict[str, Any]) -> dict[str, Any]:
        try:
            output = _git(repo_path, "ls-tree", "-r", "--name-only", "HEAD")
            return _ok(output)
        except subprocess.CalledProcessError:
            return _err("Could not list repository tree.")
        except subprocess.TimeoutExpired:
            return _err("Command timed out. The repository may have too many files.")

    return [
        show_commit,
        file_history,
        file_content,
        blame_range,
        diff_between,
        list_changes,
        repo_tree,
    ]


def create_git_tools_server(repo_path: str):
    """Create an MCP server with git investigation tools bound to a repository.

    All tools are read-only git commands with a 10s timeout. Output is capped
    at 4000 characters. The repo_path is captured by closure.
    """
    tools = _create_tools(repo_path)
    return create_sdk_mcp_server(
        name="git-tools",
        version="1.0.0",
        tools=tools,
    )
