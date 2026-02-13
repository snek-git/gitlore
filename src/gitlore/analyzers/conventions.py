"""Detect commit message conventions from repository history."""

from __future__ import annotations

import re
from collections import Counter

from gitlore.analyzers.commit_classifier import CONVENTIONAL_COMMIT_RE, JIRA_TICKET_RE
from gitlore.models import ClassifiedCommit, CommitConvention, CommitType

# ── Imperative mood heuristic ────────────────────────────────────────────────

_NON_IMPERATIVE_SUFFIXES = ("ed", "ing", "ied")


def _is_imperative(message: str) -> bool:
    """Rough heuristic: first word doesn't end in -ed, -ing, -ied."""
    words = message.split()
    if not words:
        return False
    first_word = words[0].lower().rstrip(":")
    # Skip if first word is a type prefix like "feat:" -- look at the part after ':'
    if ":" in message:
        after_colon = message.split(":", 1)[1].strip()
        if after_colon:
            first_word = after_colon.split()[0].lower() if after_colon.split() else first_word
    return not any(first_word.endswith(s) for s in _NON_IMPERATIVE_SUFFIXES)


# ── Structural pattern detection ─────────────────────────────────────────────

_TICKET_PATTERN_RE = re.compile(r"[A-Z]{2,10}-\d+")
_ISSUE_PATTERN_RE = re.compile(r"#\d+")


def _extract_structural_pattern(subject: str) -> str:
    """Replace content with structural tokens to identify the message format."""
    if CONVENTIONAL_COMMIT_RE.match(subject):
        return "conventional_commits"
    if re.match(r"^[A-Z]{2,10}-\d+[:\s]", subject):
        return "ticket_prefix"
    if subject.startswith("["):
        return "bracket_module"
    if re.match(r"^[\U0001f300-\U0001faff]|^:\w+:", subject):
        return "gitmoji"
    if re.match(r"^[a-z][a-z0-9/_.-]+:\s+", subject):
        return "kernel_style"
    return "freeform"


def _detect_ticket_format(subjects: list[str]) -> tuple[str | None, float]:
    """Detect the predominant ticket format and its adherence rate."""
    jira_count = sum(1 for s in subjects if JIRA_TICKET_RE.search(s))
    issue_count = sum(1 for s in subjects if _ISSUE_PATTERN_RE.search(s))
    total = len(subjects)
    if total == 0:
        return None, 0.0

    if jira_count >= issue_count and jira_count > 0:
        # Find the most common JIRA prefix
        prefixes: list[str] = []
        for s in subjects:
            m = JIRA_TICKET_RE.search(s)
            if m:
                prefixes.append(m.group("ticket").split("-")[0])
        if prefixes:
            most_common_prefix = Counter(prefixes).most_common(1)[0][0]
            return f"{most_common_prefix}-NNN", jira_count / total
    if issue_count > 0:
        return "#NNN", issue_count / total
    return None, 0.0


def analyze_conventions(classified: list[ClassifiedCommit]) -> CommitConvention:
    """Analyze commit conventions from classified commits."""
    if not classified:
        return CommitConvention(
            primary_format="freeform",
            format_adherence=0.0,
        )

    subjects = [cc.commit.subject for cc in classified]
    total = len(subjects)

    # ── Detect primary format ────────────────────────────────────────────
    pattern_counts = Counter(_extract_structural_pattern(s) for s in subjects)
    primary_format, primary_count = pattern_counts.most_common(1)[0]
    format_adherence = primary_count / total

    # ── Type/scope frequency ─────────────────────────────────────────────
    types_used: dict[str, int] = Counter()
    scopes_used: dict[str, int] = Counter()
    for cc in classified:
        if cc.commit_type != CommitType.UNKNOWN:
            types_used[cc.commit_type.value] += 1
        if cc.scope:
            scopes_used[cc.scope] += 1

    # ── Ticket detection ─────────────────────────────────────────────────
    full_messages = [
        f"{cc.commit.subject}\n{cc.commit.body}" for cc in classified
    ]
    ticket_format, ticket_adherence = _detect_ticket_format(full_messages)

    # ── Message style signals ────────────────────────────────────────────
    imperative_count = sum(1 for s in subjects if _is_imperative(s))
    lengths = [len(s) for s in subjects]
    avg_subject_length = sum(lengths) / total if total else 0.0
    under_72 = sum(1 for length in lengths if length <= 72)
    has_body = sum(1 for cc in classified if cc.commit.body.strip())
    ends_period = sum(1 for s in subjects if s.rstrip().endswith("."))
    starts_lower = sum(1 for s in subjects if s[:1].islower())

    # ── Generate detected rules ──────────────────────────────────────────
    detected_rules: list[str] = []

    if format_adherence > 0.7:
        if primary_format == "conventional_commits":
            type_list = ", ".join(sorted(types_used.keys()))
            detected_rules.append(
                f"Use conventional commits format: <type>(<scope>): <description>. Types used: {type_list}"
            )
            if scopes_used:
                scope_list = ", ".join(
                    s for s, _ in sorted(scopes_used.items(), key=lambda x: -x[1])[:10]
                )
                detected_rules.append(f"Use scopes: {scope_list}")
        elif primary_format == "ticket_prefix":
            detected_rules.append("Prefix commit messages with ticket ID")

    if ticket_adherence > 0.8 and ticket_format:
        detected_rules.append(f"Reference tickets in format: {ticket_format}")

    imperative_rate = imperative_count / total
    if imperative_rate > 0.8:
        detected_rules.append(
            "Use imperative mood in commit messages (e.g., 'add' not 'added')"
        )

    under_72_rate = under_72 / total
    if under_72_rate > 0.9:
        detected_rules.append("Keep commit subject line under 72 characters")

    starts_lower_rate = starts_lower / total
    if starts_lower_rate > 0.8:
        detected_rules.append("Start commit subject with lowercase letter")
    elif starts_lower_rate < 0.2:
        detected_rules.append("Start commit subject with uppercase letter")

    ends_period_rate = ends_period / total
    if ends_period_rate < 0.1:
        detected_rules.append("Do not end commit subject with a period")

    has_body_rate = has_body / total
    if has_body_rate > 0.5:
        detected_rules.append("Include a commit body for non-trivial changes")

    return CommitConvention(
        primary_format=primary_format,
        format_adherence=format_adherence,
        types_used=dict(types_used),
        scopes_used=dict(scopes_used),
        ticket_format=ticket_format,
        ticket_adherence=ticket_adherence,
        imperative_mood_rate=imperative_rate,
        avg_subject_length=avg_subject_length,
        subject_under_72_rate=under_72_rate,
        has_body_rate=has_body_rate,
        ends_with_period_rate=ends_period_rate,
        starts_lowercase_rate=starts_lower_rate,
        detected_rules=detected_rules,
    )
