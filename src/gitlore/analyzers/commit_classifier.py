"""Commit classification using regex cascade and diff-based heuristics."""

from __future__ import annotations

import re

from gitlore.models import ClassifiedCommit, Commit, CommitType

# ── Regex patterns ───────────────────────────────────────────────────────────

CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)"  # type (feat, fix, etc.)
    r"(?:\((?P<scope>[^)]+)\))?"  # optional scope in parens
    r"(?P<breaking>!)?"  # optional breaking change indicator
    r":\s+"  # colon + space separator
    r"(?P<subject>.+)"  # subject/description
)

# Map conventional commit types to our enum
_TYPE_MAP: dict[str, CommitType] = {
    "feat": CommitType.FEAT,
    "fix": CommitType.FIX,
    "refactor": CommitType.REFACTOR,
    "test": CommitType.TEST,
    "tests": CommitType.TEST,
    "docs": CommitType.DOCS,
    "doc": CommitType.DOCS,
    "ci": CommitType.CI,
    "build": CommitType.BUILD,
    "chore": CommitType.CHORE,
    "style": CommitType.STYLE,
    "perf": CommitType.PERF,
    "revert": CommitType.REVERT,
    "merge": CommitType.MERGE,
}

# Ticket references
JIRA_TICKET_RE = re.compile(r"\b(?P<ticket>[A-Z]{2,10}-\d+)\b")
GITHUB_ISSUE_RE = re.compile(
    r"(?:(?:fix(?:es)?|close[sd]?|resolve[sd]?)\s+)?#(?P<issue>\d+)", re.IGNORECASE
)

# Ticket-prefix pattern: "PROJ-123: description" or "PROJ-123 description"
TICKET_PREFIX_RE = re.compile(r"^(?P<ticket>[A-Z]{2,10}-\d+)[:\s]+(?P<subject>.+)")

# Bracket module pattern: "[module] description"
BRACKET_MODULE_RE = re.compile(r"^\[(?P<module>[^\]]+)\]\s*(?P<subject>.+)")

# Gitmoji pattern
GITMOJI_RE = re.compile(r"^(?P<emoji>[\U0001f300-\U0001faff]|:\w+:)\s*(?P<subject>.+)")

# Gitmoji to type mapping (common ones)
_GITMOJI_TYPE_MAP: dict[str, CommitType] = {
    ":sparkles:": CommitType.FEAT,
    ":bug:": CommitType.FIX,
    ":recycle:": CommitType.REFACTOR,
    ":white_check_mark:": CommitType.TEST,
    ":memo:": CommitType.DOCS,
    ":construction_worker:": CommitType.CI,
    ":wrench:": CommitType.CHORE,
    ":art:": CommitType.STYLE,
    ":zap:": CommitType.PERF,
    ":rewind:": CommitType.REVERT,
    ":fire:": CommitType.CHORE,
    ":lipstick:": CommitType.STYLE,
    ":rotating_light:": CommitType.FIX,
    ":rocket:": CommitType.FEAT,
    ":package:": CommitType.BUILD,
    ":heavy_plus_sign:": CommitType.BUILD,
    ":heavy_minus_sign:": CommitType.BUILD,
    ":lock:": CommitType.FIX,
    ":bookmark:": CommitType.CHORE,
}

# Revert patterns
REVERT_SUBJECT_RE = re.compile(r'^Revert ".*"', re.IGNORECASE)
REVERT_BODY_RE = re.compile(r"This reverts commit ([0-9a-f]{7,40})", re.IGNORECASE)

# Breaking change in footer
BREAKING_CHANGE_RE = re.compile(r"^BREAKING[ -]CHANGE:\s*(?P<description>.+)", re.MULTILINE)

# ── Test/doc/ci/config file detection ────────────────────────────────────────

_TEST_INDICATORS = [
    "test_",
    "_test.",
    ".test.",
    "tests/",
    "test/",
    "spec_",
    "_spec.",
    ".spec.",
    "specs/",
    "spec/",
    "__tests__/",
    "testing/",
]

_DOC_EXTENSIONS = {"md", "rst", "txt", "adoc"}
_DOC_DIRS = ("docs/", "doc/", "documentation/")

_CI_PATHS = (
    ".github/workflows/",
    ".gitlab-ci",
    "Jenkinsfile",
    ".circleci/",
    ".travis.yml",
    "azure-pipelines",
    ".buildkite/",
)

_CONFIG_EXTENSIONS = {"toml", "yaml", "yml", "json", "ini", "cfg", "conf"}
_CONFIG_NAMES = {
    ".gitignore",
    ".eslintrc",
    ".prettierrc",
    "Makefile",
    "Dockerfile",
    "docker-compose",
}


def _is_test_file(path: str) -> bool:
    lower = path.lower()
    return any(ind in lower for ind in _TEST_INDICATORS)


def _is_doc_file(path: str) -> bool:
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    return ext in _DOC_EXTENSIONS or any(path.startswith(d) for d in _DOC_DIRS)


def _is_ci_file(path: str) -> bool:
    return any(path.startswith(p) or p in path for p in _CI_PATHS)


def _is_config_file(path: str) -> bool:
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    return ext in _CONFIG_EXTENSIONS or any(n in path for n in _CONFIG_NAMES)


def _classify_from_diff(commit: Commit) -> CommitType:
    """Best-effort classification from file changes when message is ambiguous."""
    paths = commit.file_paths
    if not paths:
        return CommitType.UNKNOWN

    if all(_is_test_file(p) for p in paths):
        return CommitType.TEST
    if all(_is_doc_file(p) for p in paths):
        return CommitType.DOCS
    if all(_is_ci_file(p) for p in paths):
        return CommitType.CI
    if all(_is_config_file(p) for p in paths):
        return CommitType.CHORE

    # Roughly equal additions/deletions suggests refactor
    total_added = sum(f.added or 0 for f in commit.files)
    total_deleted = sum(f.deleted or 0 for f in commit.files)
    if total_added > 5 and total_deleted > 0:
        ratio = min(total_added, total_deleted) / max(total_added, total_deleted)
        if ratio > 0.6:
            return CommitType.REFACTOR

    return CommitType.UNKNOWN


def _extract_ticket(text: str) -> str | None:
    """Extract the first ticket reference from text."""
    m = JIRA_TICKET_RE.search(text)
    if m:
        return m.group("ticket")
    m = GITHUB_ISSUE_RE.search(text)
    if m:
        return f"#{m.group('issue')}"
    return None


def classify_commit(commit: Commit) -> ClassifiedCommit:
    """Classify a single commit using regex cascade and diff heuristics."""
    subject = commit.subject.strip()
    full_message = f"{subject}\n\n{commit.body}" if commit.body else subject

    # Check revert first
    if REVERT_SUBJECT_RE.match(subject) or REVERT_BODY_RE.search(full_message):
        return ClassifiedCommit(
            commit=commit,
            commit_type=CommitType.REVERT,
            is_breaking=False,
            ticket=_extract_ticket(full_message),
            is_conventional=False,
        )

    # Try conventional commits
    m = CONVENTIONAL_COMMIT_RE.match(subject)
    if m:
        type_str = m.group("type").lower()
        commit_type = _TYPE_MAP.get(type_str, CommitType.UNKNOWN)
        is_breaking = bool(m.group("breaking")) or bool(
            BREAKING_CHANGE_RE.search(commit.body)
        )
        return ClassifiedCommit(
            commit=commit,
            commit_type=commit_type,
            scope=m.group("scope"),
            is_breaking=is_breaking,
            ticket=_extract_ticket(full_message),
            is_conventional=True,
        )

    # Try ticket prefix: "PROJ-123: description"
    m = TICKET_PREFIX_RE.match(subject)
    if m:
        ticket = m.group("ticket")
        # Try to infer type from the description after the ticket
        desc = m.group("subject")
        inner = CONVENTIONAL_COMMIT_RE.match(desc)
        if inner:
            type_str = inner.group("type").lower()
            commit_type = _TYPE_MAP.get(type_str, CommitType.UNKNOWN)
            return ClassifiedCommit(
                commit=commit,
                commit_type=commit_type,
                scope=inner.group("scope"),
                is_breaking=bool(inner.group("breaking")),
                ticket=ticket,
                is_conventional=False,
            )
        return ClassifiedCommit(
            commit=commit,
            commit_type=_classify_from_diff(commit),
            ticket=ticket,
            is_conventional=False,
        )

    # Try bracket module: "[module] description"
    m = BRACKET_MODULE_RE.match(subject)
    if m:
        return ClassifiedCommit(
            commit=commit,
            commit_type=_classify_from_diff(commit),
            scope=m.group("module"),
            ticket=_extract_ticket(full_message),
            is_conventional=False,
        )

    # Try gitmoji
    m = GITMOJI_RE.match(subject)
    if m:
        emoji = m.group("emoji")
        commit_type = _GITMOJI_TYPE_MAP.get(emoji, CommitType.UNKNOWN)
        if commit_type == CommitType.UNKNOWN:
            commit_type = _classify_from_diff(commit)
        return ClassifiedCommit(
            commit=commit,
            commit_type=commit_type,
            ticket=_extract_ticket(full_message),
            is_conventional=False,
        )

    # Merge commit
    if commit.is_merge:
        return ClassifiedCommit(
            commit=commit,
            commit_type=CommitType.MERGE,
            ticket=_extract_ticket(full_message),
            is_conventional=False,
        )

    # Fallback: diff-based heuristic
    return ClassifiedCommit(
        commit=commit,
        commit_type=_classify_from_diff(commit),
        ticket=_extract_ticket(full_message),
        is_conventional=False,
    )


def classify_commits(commits: list[Commit]) -> list[ClassifiedCommit]:
    """Classify a list of commits."""
    return [classify_commit(c) for c in commits]
