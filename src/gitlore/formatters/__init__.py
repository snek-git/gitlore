"""Output formatters for different AI coding assistant configs."""

from gitlore.formatters.agents_md import format_agents_md
from gitlore.formatters.claude_md import format_claude_md
from gitlore.formatters.copilot_instructions import format_copilot_instructions
from gitlore.formatters.cursor_rules import format_cursor_rules
from gitlore.formatters.report import format_report

__all__ = [
    "format_agents_md",
    "format_claude_md",
    "format_copilot_instructions",
    "format_cursor_rules",
    "format_report",
]
