# gitlore

gitlore mines your git history and PR reviews to produce a knowledge report about your codebase. It reads commit patterns, file coupling, churn hotspots, reverts, and fix-after chains from git log, then pulls PR review comments from GitHub and classifies them by theme. The output is a report documenting what your team's actual practices and pain points are, grounded in evidence from the repo itself.

## How it works

The tool runs two branches of analysis. Branch A is purely local: it parses git log output to find hotspot files, co-change coupling between files, revert chains, fix-after patterns, and commit conventions. No network or LLM calls needed. Branch B fetches merged PR review comments from GitHub's GraphQL API, classifies each one with an LLM, embeds them, and clusters similar comments together to surface recurring themes.

Both branches feed into an agentic synthesis step where an LLM investigates the patterns using read-only git tools (blame, file history, diffs) and writes the final report.

## Usage

```
uv sync --all-extras
uv run gitlore init              # create gitlore.toml
uv run gitlore analyze            # full pipeline
uv run gitlore analyze --git-only # skip GitHub/LLM, local analysis only
uv run gitlore analyze --dry-run  # preview without writing files
uv run gitlore analyze --no-cache # skip cache reads
```

Or install globally (`pip install uv` if you don't have it):

```
uv tool install .
gitlore analyze
```

Configure models, GitHub repo, and output formats in `gitlore.toml`. Needs an `OPENROUTER_API_KEY` in `.env` (or `~/.config/gitlore/.env`) and GitHub auth via `gh` CLI or `GITHUB_TOKEN`.

## Output

Default output is `gitlore-report.md` and `gitlore-report.html`. Can also generate AI assistant configs (`claude_md`, `agents_md`, `cursor_rules`, `copilot_instructions`) via the `[output]` section in `gitlore.toml`.
