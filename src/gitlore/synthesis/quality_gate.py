"""Heuristic quality gate for generated rules."""

from __future__ import annotations

import re

from gitlore.models import Rule

# ── Platitude detection ──────────────────────────────────────────────────────

_PLATITUDES = [
    "follow best practices",
    "write clean code",
    "write tests",
    "keep it simple",
    "don't repeat yourself",
    "use meaningful names",
    "keep code clean",
    "write readable code",
    "follow coding standards",
    "use proper error handling",
    "write good documentation",
    "follow the style guide",
    "maintain code quality",
    "use design patterns",
    "keep functions small",
    "avoid code smells",
    "write maintainable code",
    "use version control properly",
    "follow solid principles",
    "write self-documenting code",
]

_IMPERATIVE_VERBS = re.compile(
    r"\b(always|never|must|use|add|update|check|ensure|avoid|include|remove|"
    r"run|require|verify|test|review|request|keep|set|do not|don't|make sure|"
    r"prefer|follow|wrap|handle|validate|call|create|delete|move|rename|apply|"
    r"configure|disable|enable|format|group|import|limit|log|mark|mock|"
    r"override|pass|pin|prefix|put|quote|raise|read|refactor|replace|"
    r"return|revert|schedule|send|separate|skip|split|start|stop|store|"
    r"submit|throw|track|trigger|upgrade|warn|write)\b",
    re.IGNORECASE,
)

_FILE_REFERENCE = re.compile(
    r"(?:"
    r"[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-./]+"  # path with /
    r"|[a-zA-Z0-9_\-]+\.[a-zA-Z]{1,10}"  # file with extension
    r"|[a-zA-Z_][a-zA-Z0-9_]*\(\)"  # function_name()
    r"|`[a-zA-Z_][a-zA-Z0-9_.]*`"  # `identifier`
    r")"
)


def is_actionable(rule: Rule) -> bool:
    """Check if the rule contains an imperative verb."""
    return bool(_IMPERATIVE_VERBS.search(rule.text))


def is_specific(rule: Rule) -> bool:
    """Check if the rule references a file path or function name."""
    return bool(_FILE_REFERENCE.search(rule.text))


def is_concise(rule: Rule) -> bool:
    """Check if the rule is under 200 characters."""
    return len(rule.text) <= 200


def is_not_platitude(rule: Rule) -> bool:
    """Check if the rule is not a known platitude."""
    lower = rule.text.lower()
    return not any(platitude in lower for platitude in _PLATITUDES)


def passes_quality_gate(rule: Rule) -> bool:
    """A rule passes if it satisfies at least 3 of 4 quality checks."""
    checks = [
        is_actionable(rule),
        is_specific(rule),
        is_concise(rule),
        is_not_platitude(rule),
    ]
    return sum(checks) >= 3


def filter_rules(rules: list[Rule]) -> list[Rule]:
    """Apply the quality gate, returning only rules that pass."""
    return [r for r in rules if passes_quality_gate(r)]
