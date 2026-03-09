# gitlore

Extract tribal knowledge from any git repository. gitlore mines commit history, PR reviews, and repo docs to produce the kind of knowledge that experienced contributors carry in their heads -- then makes it available to agents and developers via MCP, CLI, and stable exports.

## What it does

```bash
gitlore build                                    # investigate the repo
gitlore advise --task "fix option parsing" --files src/click/core.py
gitlore mcp                                      # serve knowledge via MCP
gitlore export --format agents_md,claude_md       # export to AGENTS.md etc
```

`build` runs an agent that investigates the repository using precomputed analysis (churn, coupling, fix-after chains, PR review patterns) and read-only git tools. It writes knowledge notes grounded in evidence.

`advise` and the MCP tools retrieve relevant notes for a given task using semantic similarity, full-text search, and file-graph expansion.

## What it produces

```json
{
  "summary": "5 notes for this change",
  "notes": [
    {
      "text": "Checking falsey defaults like `if value` fails for values of 0, False, or empty string. Always use explicit None checks.",
      "refs": ["PR #1970 comment from davidism"],
      "confidence": "high"
    }
  ]
}
```

Notes are specific to the repository, grounded in commits/PRs/review history, and capture things you'd only learn from working in the codebase over time.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install .
```

For development:

```bash
uv sync --all-extras
```

## Configuration

Create `gitlore.toml` (or run `gitlore init`):

```toml
[models]
classifier = "openrouter/google/gemini-3.1-flash-lite-preview"
embedding = "openrouter/openai/text-embedding-3-small"
synthesizer = "sonnet"   # uses Claude subscription auth

[sources]
github = true
docs = true
```

Requirements:
- `synthesizer = "sonnet"` uses your Claude Code subscription. No API key needed.
- `OPENROUTER_API_KEY` in `.env` for the classifier and embedding models.
- GitHub token via `GITHUB_TOKEN` env or `gh auth login` for PR review data.

All enrichment is optional. Without GitHub, build works from local git + docs. Without a synthesizer model, build produces the analysis but no knowledge notes.

## Commands

```bash
gitlore init                                      # create gitlore.toml
gitlore build                                     # build the knowledge index
gitlore build --repo ~/dev/click                  # build for a specific repo
gitlore advise --task "refactor auth" --files src/auth.py
gitlore advise --task "review this" --diff /tmp/change.diff --format json
gitlore export --format report,html,agents_md
gitlore mcp                                       # run MCP server over stdio
gitlore auth                                      # check available credentials
```

## MCP tools

- `get_planning_brief(task, files?, diff_path?, tentative_plan?, question?, format?, max_notes?)` -- retrieve relevant knowledge notes
- `get_related_files(path, limit?)` -- find structurally related files via coupling graph
- `get_repo_guidance()` -- get repo-wide knowledge notes

## How it works

1. **Extract**: mine git history (commits, diffs, numstat) and GitHub PR reviews (comments, threads, review state)
2. **Analyze**: detect churn hotspots, file coupling, fix-after chains, revert chains, commit conventions
3. **Classify & cluster**: LLM-classify review comments by category, embed and cluster them into themes, summarize each theme
4. **Investigate**: run a single Claude agent session with evidence query tools + git tools. The agent explores freely and writes knowledge notes as it goes
5. **Embed**: embed notes with text-embedding-3-small for semantic retrieval
6. **Store**: persist notes, file edges, and metadata in `.gitlore/index.db`
7. **Retrieve**: match notes to queries via anchor matching, FTS, semantic similarity, and coupling graph expansion

## Outputs

- `.gitlore/index.db` -- SQLite index (source of truth)
- `.gitlore/notes.jsonl` -- raw notes from investigation
- `gitlore advise` -- retrieval as summary or JSON
- `gitlore export` -- stable views: `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/gitlore.mdc`, `.github/copilot-instructions.md`, `gitlore-report.md`, `gitlore-report.html`

## Developer docs

See [docs/development.md](docs/development.md) for the full technical architecture.
