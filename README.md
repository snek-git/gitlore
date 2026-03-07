# gitlore

gitlore builds a local knowledge index from git history, PR review comments, and repo docs, then retrieves task-specific tribal knowledge for humans and coding agents.

The core workflow is:

```bash
gitlore build
gitlore context --task "fix flaky retries" --files src/client.py
gitlore export --format agents_md,claude_md
gitlore mcp
```

## How it works

`gitlore build` mines repository history into typed facts:

- stable rules from conventions and docs
- situational guidance from hotspots, coupling, hub files, and review themes
- historical examples from reverts and fix-after chains
- test associations and related files from co-change data

Those facts are stored in `.gitlore/index.db`. `gitlore context` ranks them against a task, file list, and optional diff, then returns a bounded context bundle. No live LLM call is required by default.

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
gitlore context --task "refactor auth flow" --files src/auth.py
gitlore context --task "review this patch" --diff /tmp/change.diff --format prompt
gitlore export --format report,html
gitlore mcp
gitlore auth
```

## Configuration

Configure the tool with `gitlore.toml`:

- `[models]` for optional review classification, embeddings, and compression
- `[build]` for lookback and coupling thresholds
- `[sources]` to enable or disable GitHub/docs ingestion
- `[query]` for retrieval defaults
- `[github]` for owner/repo and token override
- `[export]` for artifact formats

GitHub and LLM enrichment are optional. Build still works with local git + docs only.

## Outputs

- `.gitlore/index.db` is the source of truth for retrieval
- `gitlore context` emits `summary`, `prompt`, or `json`
- `gitlore export` writes stable views such as `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/gitlore.mdc`, `.github/copilot-instructions.md`, `gitlore-report.md`, and `gitlore-report.html`
