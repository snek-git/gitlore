# gitlore Development Guide

Technical architecture reference for contributors.

## Product Model

gitlore is a context engineering tool. It extracts tribal knowledge from repository history and makes it retrievable at the moment someone needs it -- during planning, editing, or review. The output is not analysis or metrics. It's the context that makes an agent or developer behave like an experienced contributor.

Two runtime modes:

1. **Build**: expensive, agentic. Mines history, runs an investigation agent, writes knowledge notes.
2. **Query**: cheap, local. Retrieves relevant notes for a task via MCP or CLI.

## Data Flow

```
git log + GitHub PR reviews + repo docs
  |
  v
Extractors (git_log.py, github_comments.py)
  |
  v
Analyzers (churn, coupling, fix_after, reverts, conventions)
  + Classifiers (comment_classifier.py)
  + Clustering (semantic.py: embed -> HDBSCAN -> LLM summarize)
  |
  v
AnalysisResult (in memory)
  |
  v
Evidence tools (MCP server querying AnalysisResult)
  + Git tools (read-only repo access)
  |
  v
Single agent session (Claude Agent SDK)
  -> writes .gitlore/notes.jsonl via Write tool
  |
  v
Embed notes (text-embedding-3-small)
  |
  v
Store in .gitlore/index.db (SQLite)
  |
  v
Retrieve via query.py (FTS + semantic + anchor + graph)
  |
  v
Serve via MCP / CLI / export
```

## Entry Points

### CLI (`src/gitlore/cli.py`)

- `gitlore build` -- builds `.gitlore/index.db`
- `gitlore advise` -- retrieves a planning brief
- `gitlore export` -- renders stable exports
- `gitlore mcp` -- runs MCP server over stdio
- `gitlore init` -- writes `gitlore.toml`
- `gitlore auth` -- checks credentials

### MCP (`src/gitlore/mcp_server.py`)

- `get_planning_brief(task, files?, ...)` -- primary retrieval tool
- `get_related_files(path, limit?)` -- coupling graph lookup
- `get_repo_guidance()` -- repo-wide notes

## Build Pipeline (`src/gitlore/build.py`)

### Step 1: Extract and analyze

Deterministic. Runs extractors and analyzers on git history:
- Churn hotspots (`analyzers/churn.py`)
- File coupling via co-change (`analyzers/coupling.py`)
- Fix-after chains (`analyzers/fix_after.py`)
- Revert chains (`analyzers/reverts.py`)
- Commit conventions (`analyzers/conventions.py`)
- Commit classification (`analyzers/commit_classifier.py`)

Output: `AnalysisResult` with all signals.

### Step 2: GitHub review enrichment (optional)

If GitHub is enabled:
1. Fetch merged PR review comments via GraphQL
2. Classify comments with LLM (categories: bug, architecture, convention, security, performance, etc.)
3. Embed comments and cluster with HDBSCAN
4. Summarize each cluster with LLM (full context: comment bodies, diff hunks, thread replies)

### Step 3: Doc/config extraction (optional)

Extract snippets from README, CONTRIBUTING, CI configs, pyproject.toml, docs/, etc.

### Step 4: Agent investigation

One Claude Agent SDK session. The agent receives:
- **Evidence tools** (`synthesis/evidence_tools.py`): query precomputed analysis (hotspots, coupling, fix-after chains, review patterns, review comments, doc snippets, conventions, modules)
- **Git tools** (`synthesis/tools.py`): read-only repo access (show_commit, file_history, file_content, blame_range, diff_between, list_changes, repo_tree)
- **Write tool**: allowed only for `.gitlore/notes.jsonl`

The agent investigates freely -- no turn limits, no pre-structured leads. It writes JSONL notes as it discovers things. Each note has: text, anchors, evidence refs, confidence.

System prompt (`prompts/investigation_system.txt`) prioritizes:
1. Review patterns (what do reviewers push back on?)
2. Architectural knowledge (how do subsystems connect?)
3. Failure patterns (what's been reverted or fixed repeatedly?)
4. Workflows and process (how do changes land?)
5. Maintainer preferences (what does the approver care about?)
6. Code style (only if reviewers would flag it)

### Step 5: Embed notes

Notes are embedded with `text-embedding-3-small` for semantic retrieval. Stored as BLOB in SQLite.

### Step 6: Store

Notes, file edges, and build metadata written to `.gitlore/index.db`.

## Index Schema (`src/gitlore/index.py`)

```sql
knowledge_notes (id, text, anchors, evidence_refs, confidence, search_text, created_at, embedding)
file_edges (src, dst, edge_type, score, reason)
build_metadata (key, value)
note_fts (note_id, text, anchors, search_text)  -- FTS5
```

## Data Model (`src/gitlore/models.py`)

Core types:
- `KnowledgeNote` -- stored retrieval unit (text, anchors, evidence_refs, confidence, embedding)
- `AnalysisResult` -- aggregated analyzer output (hotspots, coupling, fix-after, reverts, etc.)
- `PlanningQuery` -- query inputs (task, files, diff, plan, question)
- `PlanningBrief` -- query output (summary, notes, related files)
- `PlanningNote` -- minimal note returned to consumers (text, refs, confidence)
- `FileEdge` -- coupling relationship for scope expansion
- `ExportBundle` -- notes selected for export

Analyzer models (unchanged from extraction):
- `Commit`, `ClassifiedCommit`, `ChurnHotspot`, `CouplingPair`, `FixAfterChain`, `RevertChain`, `CommitConvention`, `ReviewComment`, `ClassifiedComment`, `CommentCluster`

## Retrieval (`src/gitlore/query.py`)

Ranking signals:
- **Anchor match** (0.30): direct file path overlap between query files and note anchors
- **Lexical/FTS** (0.20): token overlap and SQLite FTS5 BM25 scoring
- **Semantic** (0.20): cosine similarity between query embedding and note embedding
- **Graph match** (0.12): coupling edge expansion from query files to note anchors
- **Confidence** (0.10): note confidence (high=1.0, medium=0.6, low=0.3)
- **Relevance boost** (0.08): any-signal-present bonus

When embeddings aren't available, weights rebalance to anchor=0.40, lexical=0.25, graph=0.15, confidence=0.10, boost=0.10.

No live LLM call at query time.

## Configuration (`src/gitlore/config.py`)

```toml
[models]
classifier = "openrouter/google/gemini-3.1-flash-lite-preview"
embedding = "openrouter/openai/text-embedding-3-small"
synthesizer = "sonnet"  # "sonnet", "opus", or "openrouter/..." for OpenRouter

[build]
since_months = 12
half_life_days = 180
min_coupling_confidence = 0.25
min_coupling_lift = 1.5

[sources]
github = true
docs = true

[query]
semantic = true

[github]
# auto-detected from git remote, or override:
owner = ""
repo = ""
```

Auth resolution:
- `synthesizer = "sonnet"` uses Claude Code subscription (no API key needed)
- `synthesizer = "openrouter/..."` requires `OPENROUTER_API_KEY`
- GitHub token: config > `GITHUB_TOKEN` env > `gh auth token`

## Testing

```bash
uv run pytest -x -q       # 221 tests
uv run ruff check src/ tests/
uv run mypy src/
```

Key test files:
- `test_context_engine.py` -- integration: build + query + export + MCP
- `test_advice_investigation.py` -- evidence tools, note parsing, investigation guards
- `test_churn.py`, `test_coupling.py`, `test_fix_after.py`, `test_reverts.py` -- analyzer unit tests
- `test_comment_classifier.py` -- LLM classification
- `test_semantic_clustering.py` -- embedding + HDBSCAN clustering
- `test_git_log.py`, `test_github_comments.py` -- extractor tests
