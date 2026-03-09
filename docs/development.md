# gitlore Development Guide

This document describes the current technical architecture of `gitlore` in implementation detail.

It is written for contributors working on the build pipeline, retrieval path, MCP integration, and export logic.

## Product Model

`gitlore` has two distinct runtime modes:

1. Build-time analysis
   Expensive, may use live LLM/agent reasoning, writes reusable planning advice into `.gitlore/index.db`.
2. Query-time retrieval
   Cheap, retrieval-only by default, returns a small planning brief for the current task/files/plan.

The system is intentionally asymmetric:

- spend money and reasoning budget once at build time
- keep planning-time MCP calls fast and stable

The central design choice is that the stored retrieval unit is an `AdviceCard`, not a report paragraph and not a raw analyzer metric.

## Entry Points

### CLI

Implemented in [src/gitlore/cli.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/cli.py).

Commands:

- `gitlore build`
  Builds `.gitlore/index.db`.
- `gitlore advise`
  Retrieves a planning brief from the local index.
- `gitlore export`
  Renders stable guidance views such as `AGENTS.md`.
- `gitlore mcp`
  Runs the MCP server over stdio.
- `gitlore init`
  Writes `gitlore.toml`.
- `gitlore auth`
  Checks optional GitHub and `.env`/LLM setup.

### MCP

Implemented in [src/gitlore/mcp_server.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/mcp_server.py).

Primary tool:

- `get_planning_brief(task, files?, diff_path?, tentative_plan?, question?, format?, max_notes?)`

Supporting tools:

- `get_related_files(path, limit?)`
- `get_repo_guidance()`

The MCP path uses the same retrieval core as the CLI. It does not do a live LLM call by default.

## Configuration Model

Implemented in [src/gitlore/config.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/config.py).

Main sections:

- `[models]`
  - `classifier`
  - `embedding`
  - `synthesizer`
- `[build]`
  - `since_months`
  - `half_life_days`
  - `min_coupling_confidence`
  - `min_coupling_lift`
  - `max_files_per_commit`
  - `min_shared_commits`
- `[sources]`
  - `github`
  - `docs`
- `[query]`
  - `default_format`
  - `max_items`
  - `max_tokens`
  - `semantic`
- `[github]`
  - `owner`
  - `repo`
  - `token`
- `[export]`
  - `formats`

Token/config resolution rules:

- GitHub token comes from config, `GITHUB_TOKEN`, or `gh auth token`.
- Repository owner/repo can be autodetected from `origin`.
- `.env` is loaded in CLI startup via `python-dotenv`.

## Data Model

Implemented in [src/gitlore/models.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/models.py).

Important types:

- `InvestigationLead`
  Bounded build-time investigation target.
- `AdviceCard`
  Stored retrieval unit.
- `EvidenceRef`
  Provenance for leads and cards.
- `FileEdge`
  Related-file relationship used during retrieval.
- `PlanningQuery`
  Query-time inputs.
- `PlanningBrief`
  Query-time output.
- `PlanningNote`
  Minimal note shape returned to agents and humans.

The old `KnowledgeFact`-style retrieval model is no longer the product center.

## Build Pipeline

Implemented in [src/gitlore/build.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/build.py).

### Step 1: Deterministic repository mining

The build starts from git history using:

- `iter_commits`
- `classify_commits`
- `analyze_churn`
- `analyze_conventions`
- `analyze_coupling`
- `detect_fix_after`
- `detect_reverts`

Output is collected into `AnalysisResult`.

### Step 2: Optional GitHub review enrichment

If GitHub is enabled and a token is available:

1. fetch merged PR review data with `fetch_review_comments`
2. optionally classify comments with `classify_comments`
3. optionally cluster them with `cluster_comments`

This stage updates `SourceCoverage`:

- `github`
- `classified_reviews`
- `semantic`

The cache DB remains separate from the main advice index:

- `.gitlore/cache.db`

### Step 3: Docs/config extraction

If docs are enabled, the build extracts deterministic snippets from:

- `README*`
- `AGENTS.md`
- `CLAUDE.md`
- `CONTRIBUTING*`
- `docs/**/*.md`
- key config files

Implemented in [src/gitlore/docs.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/docs.py).

Docs are supporting guidance only. They should not dominate planning-time retrieval.

### Step 4: Lead generation

Analyzer output is converted into bounded `InvestigationLead`s.

Lead families:

- hotspots
- hub files
- coupling pairs
- test associations
- fix-after chains
- revert chains
- review clusters or grouped review patterns
- convention/docs guidance

Current build budgets in [src/gitlore/build.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/build.py):

- `MAX_HOTSPOT_LEADS = 20`
- `MAX_COUPLING_LEADS = 20`
- `MAX_REVIEW_LEADS = 15`
- `MAX_HISTORY_LEADS = 10`
- `MAX_GUIDANCE_LEADS = 10`

Each lead includes:

- title
- summary
- anchors
- applies-to intents
- priority
- confidence/support
- evidence
- prompt context

### Step 5: Agentic investigation

Implemented through [src/gitlore/synthesis/synthesizer.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/synthesis/synthesizer.py), specifically `investigate_leads`.

Important behavior:

- one lead is investigated at a time
- existing read-only git tools are reused
- the prompt asks for objective planning advice, not report prose
- the per-lead Claude agent loop is bounded with `max_turns=8`

Prompt file:

- [advice_card_system.txt](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/prompts/advice_card_system.txt)

Expected output:

```xml
<advice_cards>
  <card priority="high" kind="scope" applies_to="bugfix,refactor">
    <text>When editing `src/x.py`, inspect `tests/test_x.py`.</text>
    <anchors>
      <anchor>src/x.py</anchor>
      <anchor>tests/test_x.py</anchor>
    </anchors>
  </card>
</advice_cards>
```

This XML is parsed into `AdviceCard`s.

### Step 6: Deterministic fallback

If build-time synthesis is unavailable or a lead yields no cards, deterministic fallback cards are generated from the lead summary.

This guarantees:

- git-only builds still produce a usable index
- query-time retrieval remains available without live LLM dependencies

### Step 7: Persist build output

The final cards, file edges, and metadata are written to `.gitlore/index.db`.

Implemented in [src/gitlore/index.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/index.py).

## Index Schema

Main tables:

- `advice_cards`
  - `id`
  - `text`
  - `priority`
  - `kind`
  - `applies_to`
  - `anchors`
  - `confidence`
  - `support_count`
  - `search_text`
  - `created_by_build`
- `card_evidence`
  - `card_id`
  - `position`
  - `source_type`
  - `label`
  - `ref`
  - `excerpt`
  - `weight`
- `file_edges`
  - `src`
  - `dst`
  - `edge_type`
  - `score`
  - `reason`
- `build_metadata`
- `card_fts`
  FTS5 table over card text, anchors, and search text

The index is the source of truth for:

- `advise`
- MCP planning brief lookup
- repo guidance exports

## Retrieval Pipeline

Implemented in [src/gitlore/query.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/query.py).

### Step 1: Normalize query input

The query accepts:

- `task`
- `files`
- `diff_path`
- `tentative_plan`
- `question`
- `max_notes`

Intent is inferred heuristically:

- `bugfix`
- `refactor`
- `feature`
- `review`
- `general`

### Step 2: Load local retrieval data

The query loads:

- all cards
- build metadata
- related files from `file_edges`
- FTS matches from `card_fts`

### Step 3: Rank cards

Current scoring factors:

- anchor match
- intent match
- lexical/FTS overlap
- graph match via related files
- card confidence
- support count
- priority

The query is intentionally deterministic and local.

There is no live LLM call in the main retrieval path.

### Step 4: Render planning brief

The result is a `PlanningBrief` with:

- `summary`
- `notes`
- `related_files`
- `source_coverage`
- `build_metadata`

The JSON output is intentionally minimal:

- each note contains only `text`, `refs`, and `priority`

## Export Pipeline

Implemented in [src/gitlore/export.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/export.py).

Exports are secondary views over the advice-card index.

Rules:

- prefer repo-wide convention/review guidance
- include only a small number of stable cards
- keep output short
- do not dump raw docs or large historical examples indiscriminately

Supported formats:

- `agents_md`
- `claude_md`
- `cursor_rules`
- `copilot_instructions`
- `report`
- `html`

## MCP Flow

Implemented in [src/gitlore/mcp_server.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/mcp_server.py).

`FastMCP` exposes:

- `get_planning_brief`
- `get_related_files`
- `get_repo_guidance`

Intended usage by an agent:

1. draft an initial plan
2. call `get_planning_brief`
3. revise the plan using the returned notes
4. begin editing

`gitlore` is a planning-time advisor, not a replacement for planning.

## Existing Git Tools

The read-only git investigation tools used by the build-time agent live in:

- [src/gitlore/synthesis/tools.py](/Users/snek/.codex/worktrees/d994/gitlore/src/gitlore/synthesis/tools.py)

These are the same tools previously used by the synthesis agent and remain the core inspection surface for lead investigation.

## Testing Strategy

The current integration coverage for the new flow lives in:

- [tests/test_context_engine.py](/Users/snek/.codex/worktrees/d994/gitlore/tests/test_context_engine.py)

Covered behavior:

- build creates an advice-card index
- planning brief retrieval returns minimal note objects
- exports render stable guidance
- CLI and MCP use the same planning core

The full suite should pass with:

```bash
.venv/bin/python -m pytest -q
```

## Known Design Constraints

- Query-time retrieval is intentionally simple and deterministic.
- Build-time synthesis is the only place where agentic interpretation should happen.
- Docs are supporting evidence, not the primary retrieval surface.
- Advice quality still depends on real-repo tuning of lead generation and build prompts.
- The current codebase is a clean break from the old report-first pipeline. Planning advice flows only through `build`, `advise`, `export`, and `mcp`.
