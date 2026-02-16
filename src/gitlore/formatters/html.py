"""Format synthesis findings as a standalone HTML report."""

from __future__ import annotations

from datetime import date
from html import escape

from gitlore.models import Finding, FindingCategory, FindingSeverity, SynthesisResult

_CATEGORY_ORDER = [
    FindingCategory.FRAGILE_AREA,
    FindingCategory.LANDMINE,
    FindingCategory.ARCHITECTURE,
    FindingCategory.CODE_PATTERN,
    FindingCategory.CONVENTION,
    FindingCategory.TOOLING,
]

_CATEGORY_LABELS = {
    FindingCategory.FRAGILE_AREA: "Fragile Areas",
    FindingCategory.LANDMINE: "Landmines",
    FindingCategory.ARCHITECTURE: "Architecture",
    FindingCategory.CODE_PATTERN: "Code Patterns",
    FindingCategory.CONVENTION: "Conventions",
    FindingCategory.TOOLING: "Tooling",
}

_CATEGORY_ICONS = {
    FindingCategory.FRAGILE_AREA: "!!",
    FindingCategory.LANDMINE: "**",
    FindingCategory.ARCHITECTURE: "##",
    FindingCategory.CODE_PATTERN: "<>",
    FindingCategory.CONVENTION: "~~",
    FindingCategory.TOOLING: "=>",
}

_SEVERITY_COLORS = {
    FindingSeverity.HIGH: ("#ff4d4f", "#2a1215", "#ff4d4f"),
    FindingSeverity.MEDIUM: ("#faad14", "#2b2111", "#faad14"),
    FindingSeverity.LOW: ("#52c41a", "#162312", "#52c41a"),
}

_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    line-height: 1.6;
    padding: 2rem;
    max-width: 960px;
    margin: 0 auto;
}
header {
    border-bottom: 1px solid #21262d;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
header h1 {
    color: #e6edf3;
    font-size: 1.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
header .meta {
    color: #8b949e;
    font-size: 0.875rem;
}
header .meta span { margin-right: 1.5rem; }
.category-section {
    margin-bottom: 2.5rem;
}
.category-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}
.category-header .icon {
    color: #58a6ff;
    font-family: monospace;
    font-weight: 700;
    font-size: 0.875rem;
    background: #161b22;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
}
.category-header h2 {
    color: #e6edf3;
    font-size: 1.25rem;
    font-weight: 600;
}
.category-header .count {
    color: #8b949e;
    font-size: 0.8rem;
}
.finding-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.finding-card:hover {
    border-color: #30363d;
}
.finding-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}
.finding-title h3 {
    color: #e6edf3;
    font-size: 1rem;
    font-weight: 600;
}
.severity-badge {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    letter-spacing: 0.05em;
    white-space: nowrap;
}
.severity-high { color: #ff4d4f; background: #2a1215; border: 1px solid #ff4d4f40; }
.severity-medium { color: #faad14; background: #2b2111; border: 1px solid #faad1440; }
.severity-low { color: #52c41a; background: #162312; border: 1px solid #52c41a40; }
.insight {
    color: #c9d1d9;
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
    white-space: pre-wrap;
}
.files {
    margin-bottom: 0.75rem;
}
.files .label {
    color: #8b949e;
    font-size: 0.8rem;
    margin-bottom: 0.25rem;
}
.files code {
    display: inline-block;
    background: #0d1117;
    color: #79c0ff;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.8rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    margin: 0.15rem 0.25rem 0.15rem 0;
}
.evidence {
    border-top: 1px solid #21262d;
    padding-top: 0.75rem;
    margin-top: 0.5rem;
}
.evidence-item {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
}
.evidence-item .source {
    color: #8b949e;
    font-style: italic;
    white-space: nowrap;
    min-width: fit-content;
}
.evidence-item .text {
    color: #b1bac4;
}
.empty-state {
    text-align: center;
    color: #8b949e;
    padding: 3rem 1rem;
    font-size: 1rem;
}
"""


def _render_finding(f: Finding) -> str:
    """Render a single finding card."""
    sev_class = f"severity-{f.severity.value}"
    parts = [
        '<div class="finding-card">',
        '  <div class="finding-title">',
        f'    <h3>{escape(f.title)}</h3>',
        f'    <span class="severity-badge {sev_class}">{escape(f.severity.value)}</span>',
        "  </div>",
    ]
    if f.insight:
        parts.append(f'  <div class="insight">{escape(f.insight)}</div>')
    if f.files:
        file_codes = " ".join(f"<code>{escape(p)}</code>" for p in f.files)
        parts.append('  <div class="files">')
        parts.append('    <div class="label">Files</div>')
        parts.append(f"    {file_codes}")
        parts.append("  </div>")
    if f.evidence:
        parts.append('  <div class="evidence">')
        for e in f.evidence:
            parts.append('    <div class="evidence-item">')
            parts.append(f'      <span class="source">{escape(e.source)}:</span>')
            parts.append(f'      <span class="text">{escape(e.text)}</span>')
            parts.append("    </div>")
        parts.append("  </div>")
    parts.append("</div>")
    return "\n".join(parts)


def format_html(result: SynthesisResult) -> str:
    """Format a SynthesisResult as a standalone HTML report."""
    today = date.today().isoformat()
    commits = result.analysis.total_commits_analyzed if result.analysis else 0
    review_flag = "with PR review data" if result.has_review_data else "git history only"

    meta_spans = [
        f'<span>Generated {escape(today)}</span>',
        f"<span>{commits} commits analyzed</span>",
        f"<span>{escape(review_flag)}</span>",
    ]

    body_parts: list[str] = []

    # Header
    body_parts.append("<header>")
    body_parts.append("  <h1>gitlore Report</h1>")
    body_parts.append(f'  <div class="meta">{" ".join(meta_spans)}</div>')
    body_parts.append("</header>")

    if not result.findings:
        body_parts.append('<div class="empty-state">No findings to display.</div>')
    else:
        # Group by category
        by_category: dict[FindingCategory, list[Finding]] = {}
        for f in result.findings:
            by_category.setdefault(f.category, []).append(f)

        for cat in _CATEGORY_ORDER:
            findings = by_category.get(cat)
            if not findings:
                continue
            label = _CATEGORY_LABELS[cat]
            icon = _CATEGORY_ICONS[cat]
            body_parts.append('<div class="category-section">')
            body_parts.append('  <div class="category-header">')
            body_parts.append(f'    <span class="icon">{escape(icon)}</span>')
            body_parts.append(f"    <h2>{escape(label)}</h2>")
            body_parts.append(f'    <span class="count">({len(findings)})</span>')
            body_parts.append("  </div>")
            for f in findings:
                body_parts.append(_render_finding(f))
            body_parts.append("</div>")

    body_html = "\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gitlore Report</title>
<style>
{_CSS}</style>
</head>
<body>
{body_html}
</body>
</html>
"""
