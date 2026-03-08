# gitlore

Agent-first context engine for planning. `gitlore` should mine git history, PR review history, and repo docs to surface the few pieces of tribal knowledge that would materially change an agent's plan.

## Product Direction

- Primary interface: MCP tool used by Codex / Claude Code during planning.
- Secondary interface: CLI for building the index, debugging retrieval, and exporting stable repo guidance.
- Core framing: "what objective, repo-specific planning advice would materially improve this plan?"

`gitlore` is not primarily a report generator and should not optimize for broad repo summaries.

## What Counts As Useful

Only surface information that changes one of:

- scope
- implementation approach
- validation plan
- review-readiness

Good examples:

- "When editing `X`, also inspect `Y` and `Z`."
- "This area repeatedly regresses on default / `None` / flag interactions."
- "Maintainers push back on exposing this internal concept publicly."
- "Bugfixes here usually need changelog or docs updates."

Bad examples:

- generic docs excerpts
- broad repo summaries
- praise / merge chatter
- one-off comments without repeated evidence
- generic coding advice

## Planning-Time Use

The intended runtime flow is:

1. Agent drafts a tentative plan.
2. Agent calls `gitlore` during plan mode via MCP.
3. `gitlore` returns a small planning brief: a few high-signal planning notes, not a long dump.
4. Agent revises the plan before editing.

`gitlore` should be a planning-time advisor tool, not a separate ritual the user has to remember before planning.

## Build vs Query

### Build

`build` is the expensive, agentic stage.

- deterministic pre-pass mines git, PR reviews, docs, config
- candidate leads are generated from hotspots, coupling, fix-after chains, reverts, review patterns, and docs
- Claude Agent SDK investigates the top leads with read-only git tools
- output is structured planning knowledge with evidence, not prose report sections

This is where agentic reasoning belongs.

### Query / MCP

Planning-time query should be cheap.

- retrieve relevant planning knowledge for the current task, files, and tentative plan
- return only the few interventions most likely to improve the plan
- avoid live "agent over everything" behavior on each query

## Output Shape

Do not force tribal knowledge into a rigid static taxonomy.

The interface can be stable, but the content inside it should stay flexible and evidence-backed.

The main payload should read like concise, evidence-backed planning advice:

- what you're probably missing
- what tends to go wrong here
- what maintainers usually care about here
- what else you should inspect before coding

If a machine-readable envelope exists, it should stay light. The core value is in a small set of freeform, high-signal planning notes tied to evidence.

## CLI

The CLI is still useful for direct use and debugging:

```bash
uv sync --all-extras
uv run gitlore build
uv run gitlore advise --task "fix flaky option parsing bug" --files src/click/core.py
uv run gitlore export --format agents_md
uv run gitlore mcp
uv run pytest -x -q
uv run ruff check src/ tests/
uv run mypy src/
```

## MCP

The main MCP tool should be planning-oriented.

Desired shape:

- `get_planning_brief(task, files?, diff?, tentative_plan?, question?)`

Supporting tools can exist, but the main product should answer:

- what should I know before I lock this plan?
- what am I likely to miss?
- what should I avoid?
- what should I validate before I’m done?

## Architecture

Important modules:

- `src/gitlore/extractors/` for git and GitHub review extraction
- `src/gitlore/analyzers/` for deterministic lead generation
- `src/gitlore/synthesis/` for agentic investigation with Claude Agent SDK
- `src/gitlore/build.py` for index construction
- `src/gitlore/query.py` for planning-time retrieval
- `src/gitlore/mcp_server.py` for MCP tool exposure
- `src/gitlore/export.py` for stable downstream exports

The product center should be:

```text
history + review memory + docs
  -> build-time investigation
  -> indexed planning knowledge
  -> planning-time MCP retrieval
  -> better agent plans
```

## Conventions

- Build should prefer evidence-backed planning insights over generic rules.
- Agentic synthesis should investigate top leads, not write generic reports.
- Query should optimize for a few strong interventions, not exhaustive retrieval.
- Docs are supporting context, not the main product output.
- Stable exports like `AGENTS.md` and `CLAUDE.md` are secondary views over stronger planning knowledge.
