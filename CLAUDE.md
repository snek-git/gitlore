# gitlore

Context engineering tool that extracts tribal knowledge from git history and PR reviews. gitlore builds the context that makes agents and developers work like experienced contributors -- surfacing what the code won't tell you by itself.

## Vision

Agents are capable but lack context. They don't know how work gets done in a specific repo -- the workflows, the reviewer preferences, the approaches that have been tried and rejected, the files that always need updating together. gitlore is a context engineering tool that closes this gap by extracting tribal knowledge from repository history and making it retrievable at the right moment.

The goal is not code analysis or bug prevention. It's making the consumer -- whether an agent or a human -- behave like someone who's been on the team for six months. Write code that fits. Make PRs that don't get 15 review comments. Know the unwritten rules.

All product decisions should serve this: build the right context from history, deliver it when it matters. If a feature doesn't help someone produce work that looks like it came from an experienced contributor, it doesn't belong.

## What Counts As Useful

The knowledge gitlore extracts should be things you can only learn from history, reviews, and working in the codebase over time. Not things a linter catches. Not things obvious from reading a file.

Good:
- "PRs that touch the public API need a CHANGELOG entry. No CI enforces this, but the maintainer requests it every time."
- "Don't refactor variable names unless you're rewriting the line. The maintainer pushes back on this in every PR."
- "The retry logic was rewritten and reverted twice. The issue both times was timeout=0 + max_retries=None."
- "Features land behind feature flags first. Direct additions to the main code path get pushed back."
- "Options flow through a non-obvious path: decorator -> Parameter.__init__ -> type_cast_value -> Context.invoke."

Bad:
- "This file has high churn." (So what?)
- "Use super() without arguments." (Generic Python, not repo-specific.)
- "Imports are sorted." (Linter handles it.)
- "The codebase uses pytest." (Obvious.)

Code style is only worth noting if a reviewer would flag it -- things a linter won't catch but a human will.

## Architecture

```
git history + PR reviews + docs
  -> deterministic extraction + analysis (fast, minutes)
  -> evidence stored in memory as AnalysisResult
  -> single agent session with evidence tools + git tools
  -> agent investigates freely, writes notes to .gitlore/notes.jsonl
  -> notes embedded and stored in .gitlore/index.db
  -> retrieval via MCP, CLI, or export
```

The deterministic analysis pipeline (extractors, analyzers) is the agent's toolkit, not its input feed. The agent decides what to investigate, what matters, and what to skip.

### Key modules

- `src/gitlore/extractors/` -- git log parsing, GitHub PR comment fetching
- `src/gitlore/analyzers/` -- churn hotspots, coupling, fix-after chains, reverts, conventions
- `src/gitlore/classifiers/` -- LLM classification of review comments
- `src/gitlore/clustering/` -- semantic clustering of review comments with HDBSCAN + LLM summarization
- `src/gitlore/synthesis/` -- build-time agent session (Claude Agent SDK), evidence query tools, git tools
- `src/gitlore/build.py` -- build orchestration
- `src/gitlore/query.py` -- retrieval with FTS + semantic + anchor matching
- `src/gitlore/index.py` -- SQLite index store
- `src/gitlore/mcp_server.py` -- MCP tool exposure
- `src/gitlore/export.py` -- AGENTS.md, CLAUDE.md, cursor rules, etc.

### Build-time agent

The agent receives evidence query tools (precomputed analysis) and git tools (read-only repo access). It investigates the repository freely -- no turn limits, no pre-digested leads, no flashcard-style one-question-at-a-time investigation. It forms its own mental model and writes notes via the Write tool to `.gitlore/notes.jsonl`.

The system prompt prioritizes: review patterns > architectural knowledge > failure patterns > workflows > maintainer preferences > code style (only if reviewers would flag it).

### Retrieval

Query-time retrieval is cheap and local:
- Anchor matching (file path overlap)
- FTS (full-text search via SQLite FTS5)
- Semantic similarity (cosine on embedded notes, when embeddings available)
- Graph expansion (coupling edges for related files)
- Confidence weighting

No live LLM call at query time.

## CLI

```bash
uv sync --all-extras
uv run gitlore build                          # build the knowledge index
uv run gitlore advise --task "fix X" --files src/x.py  # retrieve relevant notes
uv run gitlore export --format agents_md      # export to AGENTS.md etc
uv run gitlore mcp                            # run MCP server
uv run pytest -x -q
uv run ruff check src/ tests/
uv run mypy src/
```

## Configuration

Default config uses Claude subscription auth for the investigation agent (`synthesizer = "sonnet"`) and OpenRouter for classification/embedding. Set `OPENROUTER_API_KEY` in `.env` for the classifier and embeddings. GitHub token comes from `GITHUB_TOKEN` env or `gh auth login`.

## Conventions

- The agent is the core intelligence. The deterministic pipeline is its toolkit.
- Knowledge notes should be specific, evidence-backed, and repo-specific.
- Query-time retrieval is deterministic and local. No live LLM calls.
- Exports are secondary views over the knowledge index.
- The product is tribal knowledge, not metrics, reports, or code analysis.
