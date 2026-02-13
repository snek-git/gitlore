"""Streaming git log parser using subprocess."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from typing import Iterator

from gitlore.models import Commit, FileChange

COMMIT_SEP = "\x1e"  # record separator (ASCII RS)
FIELD_SEP = "\x1f"  # field separator (ASCII US)

# Use %x1f for field separator in git format string to avoid embedded NUL bytes
# which break subprocess argument passing on Linux
GIT_LOG_FORMAT = "%x1f".join(
    [
        "%H",  # hash
        "%an",  # author name
        "%ae",  # author email
        "%aI",  # author date ISO
        "%P",  # parent hashes
        "%s",  # subject
        "%b",  # body
    ]
) + "%x1e"

_RENAME_RE = re.compile(r"\{(.+?) => (.+?)\}")


def _expand_rename_path(path: str, use_old: bool) -> str:
    """Expand git's rename format: 'dir/{old.py => new.py}/sub' -> full path."""

    def replace(m: re.Match[str]) -> str:
        old, new = m.group(1), m.group(2)
        return old if use_old else new

    expanded = _RENAME_RE.sub(replace, path)
    # Clean up double slashes from empty parts like {old => new} at root
    return expanded.replace("//", "/").strip("/")


def _parse_numstat_line(line: str) -> FileChange | None:
    """Parse a numstat line: 'added\\tdeleted\\tpath'."""
    parts = line.split("\t", 2)
    if len(parts) != 3:
        return None

    added_str, deleted_str, path = parts
    added = int(added_str) if added_str != "-" else None
    deleted = int(deleted_str) if deleted_str != "-" else None

    old_path = None
    if " => " in path and "{" in path:
        old_path = _expand_rename_path(path, use_old=True)
        path = _expand_rename_path(path, use_old=False)
    elif " => " in path:
        # Full path rename without braces: old => new
        old_path, path = path.split(" => ", 1)

    return FileChange(path=path, added=added, deleted=deleted, old_path=old_path)


def _parse_commit_block(raw: str) -> Commit | None:
    """Parse a single commit block (header + numstat lines)."""
    lines = raw.split("\n")

    # First non-empty line contains the formatted commit header
    header_line = ""
    numstat_start = 0
    for i, line in enumerate(lines):
        if FIELD_SEP in line:
            header_line = line
            numstat_start = i + 1
            break

    if not header_line:
        return None

    parts = header_line.split(FIELD_SEP)
    if len(parts) < 7:
        return None

    hash_ = parts[0]
    author_name = parts[1]
    author_email = parts[2]
    date_str = parts[3]
    parents_str = parts[4]
    subject = parts[5]
    body = FIELD_SEP.join(parts[6:])  # body may theoretically contain NUL

    commit = Commit(
        hash=hash_,
        author_name=author_name,
        author_email=author_email,
        author_date=datetime.fromisoformat(date_str),
        parents=parents_str.split() if parents_str else [],
        subject=subject,
        body=body.strip(),
    )

    # Remaining lines are numstat: "added\tdeleted\tpath"
    for line in lines[numstat_start:]:
        line = line.strip()
        if not line:
            continue
        fc = _parse_numstat_line(line)
        if fc:
            commit.files.append(fc)

    return commit


def iter_commits(
    repo_path: str,
    since_hash: str | None = None,
    since_months: int | None = None,
    no_merges: bool = True,
) -> Iterator[Commit]:
    """Stream commits from git log with constant memory usage.

    Args:
        repo_path: Path to the git repository.
        since_hash: Only commits after this hash (incremental).
        since_months: Only commits within the last N months.
        no_merges: Skip merge commits.

    Yields:
        Commit objects parsed from git log.
    """
    cmd = [
        "git",
        "-C",
        repo_path,
        "log",
        f"--pretty=format:{GIT_LOG_FORMAT}",
        "--numstat",
        "-M",  # rename detection
    ]
    if no_merges:
        cmd.append("--no-merges")
    if since_months is not None:
        cmd.extend(["--since", f"{since_months} months ago"])
    if since_hash:
        cmd.append(f"{since_hash}..HEAD")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    buffer = ""
    for chunk in iter(lambda: proc.stdout.read(8192), ""):
        buffer += chunk
        while COMMIT_SEP in buffer:
            raw_commit, buffer = buffer.split(COMMIT_SEP, 1)
            raw_commit = raw_commit.strip()
            if not raw_commit:
                continue
            commit = _parse_commit_block(raw_commit)
            if commit:
                yield commit

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        raise RuntimeError(f"git log failed: {stderr}")


def build_rename_map(commits: list[Commit]) -> dict[str, str]:
    """Build a mapping from old paths to current paths from rename data.

    Processes commits in order (oldest first assumed) to build chains:
    if a.py -> b.py then b.py -> c.py, we get a.py -> c.py.
    """
    rename_map: dict[str, str] = {}
    for commit in commits:
        for fc in commit.files:
            if fc.old_path and fc.old_path != fc.path:
                # Follow existing chains: if something pointed to old_path,
                # update it to point to the new path
                for old, current in list(rename_map.items()):
                    if current == fc.old_path:
                        rename_map[old] = fc.path
                rename_map[fc.old_path] = fc.path
    return rename_map


def resolve_path(path: str, rename_map: dict[str, str]) -> str:
    """Follow rename chain to current path."""
    seen: set[str] = set()
    while path in rename_map and path not in seen:
        seen.add(path)
        path = rename_map[path]
    return path
