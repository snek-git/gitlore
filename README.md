# gitlore

`gitlore` builds a local planning-advice index from git history, PR review comments, and repo docs, then retrieves task-specific repository guidance for humans and coding agents.

The product shape is:

```bash
gitlore build
gitlore advise --task "fix flaky retries" --files src/client.py
gitlore export --format agents_md,claude_md
gitlore mcp
```

`build` is the expensive stage. It mines repository history, generates bounded investigation leads, and can use a Claude Agent SDK loop to turn those leads into reusable advice cards.

`advise` and the MCP tools are retrieval-only by default. They read the prebuilt index and return a small planning brief with the most relevant notes for the current task.

## What It Stores

The on-disk source of truth is `.gitlore/index.db`.

The index stores:

- `advice_cards`
  Short planning-oriented notes already written in final usable form.
- `card_evidence`
  Provenance for each card, such as PRs, commits, or coupling signals.
- `file_edges`
  Related-file relationships used to widen planning scope.
- `build_metadata`
  Information about the last successful build.

The retrieval unit is an advice card, not a raw metric and not a report section.

## What Query Returns

`gitlore advise` and the MCP `get_planning_brief` tool return a small planning brief:

- `summary`
- `notes`
  Each note contains only:
  - `text`
  - `refs`
  - `priority`

Example:

```json
{
  "summary": "3 planning notes for this change",
  "notes": [
    {
      "text": "Edits in `src/client.py` usually require updates in `tests/test_client.py`.",
      "refs": ["src/client.py::tests/test_client.py"],
      "priority": "high"
    }
  ]
}
```

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install .
```

For development:

```bash
uv sync --all-extras
```

## Commands

```bash
gitlore init
gitlore build
gitlore advise --task "refactor auth flow" --files src/auth.py
gitlore advise --task "review this patch" --diff /tmp/change.diff --format json
gitlore export --format report,html
gitlore mcp
gitlore auth
```

## MCP Tools

Primary tool:

- `get_planning_brief(task, files?, diff_path?, tentative_plan?, question?, format?, max_notes?)`

Supporting tools:

- `get_related_files(path, limit?)`
- `get_repo_guidance()`

The intended agent workflow is:

1. draft a tentative plan
2. call `get_planning_brief`
3. revise the plan using the returned notes
4. begin editing

## Configuration

Configure the tool with `gitlore.toml`:

- `[models]`
  Optional model strings for review classification, embeddings, and build-time synthesis.
- `[build]`
  Lookback and coupling thresholds.
- `[sources]`
  Enable or disable GitHub/docs ingestion.
- `[query]`
  Retrieval defaults.
- `[github]`
  Owner/repo and token override.
- `[export]`
  Artifact formats.

GitHub and LLM enrichment are optional.

- With no GitHub token, build still works from local git + docs.
- With no build-time synthesis model or no `OPENROUTER_API_KEY`, build falls back to deterministic advice cards derived from the leads.
- Query-time retrieval does not require live LLM calls by default.

## Outputs

- `.gitlore/index.db`
  Source of truth for retrieval.
- `gitlore advise`
  Emits `summary` or `json`.
- `gitlore export`
  Writes stable views such as `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/gitlore.mdc`, `.github/copilot-instructions.md`, `gitlore-report.md`, and `gitlore-report.html`.

## Developer Docs

For the full technical flow, data model, prompt flow, index schema, retrieval behavior, and testing strategy, see [docs/development.md](/Users/snek/.codex/worktrees/d994/gitlore/docs/development.md).
