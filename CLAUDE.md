# gitlore

CLI tool that mines git history and PR review comments to generate AI coding assistant config files (CLAUDE.md, AGENTS.md, .cursor/rules/*.mdc, etc.) encoding a team's actual tribal knowledge.

## Build & Run

```bash
uv sync --all-extras          # install all deps including dev
uv run gitlore analyze        # full pipeline
uv run gitlore analyze --git-only --dry-run  # git-only, preview
uv run gitlore init           # create gitlore.toml
uv run gitlore auth           # check GitHub auth
uv run pytest -x -q           # run tests (241 tests)
uv run ruff check src/ tests/ # lint
uv run mypy src/              # type check
```

## Environment

- `OPENROUTER_API_KEY` in `.env` (loaded via python-dotenv)
- `GITHUB_TOKEN` env var or `gh auth token` for PR comment extraction
- Python 3.13+, uv as package manager

## Architecture

Two-branch pipeline merging at synthesis:

```
Branch A (git-only, no LLM, no network):
  git log subprocess → commit classification (regex) → churn/hotspots
  → revert detection → fix-after detection → co-change coupling (networkx)
  → convention extraction

Branch B (GitHub + LLM):
  GraphQL PR comments (httpx) → LLM classification (litellm)
  → semantic clustering (HDBSCAN) → LLM cluster labeling

Both → AnalysisResult → XML → agentic synthesis (Claude Agent SDK + git tools) → quality gate → formatters
```

### Key modules

| Module | Path | LLM? | Description |
|--------|------|------|-------------|
| Git extractor | `src/gitlore/extractors/git_log.py` | No | Streaming `git log` subprocess parser |
| GitHub extractor | `src/gitlore/extractors/github_comments.py` | No | GraphQL PR review thread extraction |
| Commit classifier | `src/gitlore/analyzers/commit_classifier.py` | No | Regex-based conventional commit parsing + diff heuristics |
| Conventions | `src/gitlore/analyzers/conventions.py` | No | Statistical commit convention detection |
| Churn | `src/gitlore/analyzers/churn.py` | No | Hotspot detection with exponential temporal decay |
| Reverts | `src/gitlore/analyzers/reverts.py` | No | Revert chain detection from commit messages |
| Fix-after | `src/gitlore/analyzers/fix_after.py` | No | 3-tier fix-after pattern detection (30min/4hr/7day) |
| Coupling | `src/gitlore/analyzers/coupling.py` | No | Co-change coupling with confidence/lift metrics, networkx Louvain |
| Comment classifier | `src/gitlore/classifiers/comment_classifier.py` | Yes | LLM classification of PR comments into 8 categories |
| Clustering | `src/gitlore/clustering/semantic.py` | Yes | Embedding via litellm API + HDBSCAN + LLM labeling |
| Synthesizer | `src/gitlore/synthesis/synthesizer.py` | Yes | Agentic loop via Claude Agent SDK — investigates patterns with git tools, then writes rules |
| Git tools | `src/gitlore/synthesis/tools.py` | No | 7 read-only git tools (show_commit, file_history, file_content, blame_range, diff_between, list_changes, repo_tree) as in-process MCP server |
| Quality gate | `src/gitlore/synthesis/quality_gate.py` | No | Heuristic rule validation (actionable, specific, concise) |
| Formatters | `src/gitlore/formatters/` | No | CLAUDE.md, AGENTS.md, .cursor/rules, copilot-instructions |
| Pipeline | `src/gitlore/pipeline.py` | - | Orchestrator wiring both branches |
| Config | `src/gitlore/config.py` | - | TOML config loading with `GitloreConfig` dataclass |
| Models | `src/gitlore/models.py` | - | All shared dataclasses (Commit, Rule, AnalysisResult, etc.) |

## LLM Integration

- Comment classification and embeddings go through `src/gitlore/utils/llm.py` (litellm)
- Synthesis uses `claude-agent-sdk` — runs an agentic loop where the LLM gets analysis XML + 7 git investigation tools, investigates top patterns, then writes rules
- Models configurable via `gitlore.toml` `[models]` section
- Default classifier: `openrouter/openai/gpt-oss-120b` (cheap/fast)
- Default synthesizer: `openrouter/google/gemini-3-flash-preview`
- Default embedding: `openrouter/openai/text-embedding-3-small`
- XML is the input format for synthesis prompts
- JSON mode for classification output

## Config

`gitlore.toml` at repo root. See `gitlore.toml.example`. Key sections:
- `[models]` — litellm model strings for classifier/synthesizer/embedding
- `[analysis]` — lookback period, coupling thresholds, decay half-life
- `[github]` — owner/repo for PR comment extraction
- `[output]` — which formats to generate

## Data Flow

All types defined in `models.py`:
- `Commit` / `FileChange` → extractors
- `ClassifiedCommit` / `CommitType` → commit_classifier
- `ChurnHotspot`, `RevertChain`, `FixAfterChain` → analyzers
- `CouplingPair`, `ImplicitModule`, `HubFile` → coupling analyzer
- `CommitConvention` → conventions analyzer
- `ReviewComment` → github extractor
- `ClassifiedComment` / `CommentCategory` → comment classifier
- `CommentCluster` → semantic clustering
- `AnalysisResult` → aggregation of all analysis (input to synthesis)
- `SynthesisResult` / `Rule` → synthesis output (input to formatters)

## Testing

- 241 tests, all in `tests/`
- Synthesis agent mocked via `unittest.mock.patch` on `claude_agent_sdk.query`
- LLM calls mocked via `unittest.mock.patch` on `litellm.acompletion`
- GitHub API mocked via `respx` (httpx mock)
- Embeddings mocked by returning synthetic numpy arrays
- Git tools tested directly against temp repos (no mocking needed)
- Run with `uv run pytest -x -q`

## Research

8 detailed research docs in `research/` covering implementation decisions:
- `git-log-extraction.md` — why raw subprocess over PyDriller
- `pr-comment-extraction.md` — GraphQL vs REST, isResolved field
- `cochange-coupling.md` — confidence/lift metrics, Louvain community detection
- `churn-revert-detection.md` — temporal decay, SZZ algorithm, 3-tier fix-after
- `comment-classification.md` — model comparison, few-shot prompts, cost analysis
- `commit-classification.md` — regex vs LLM (regex wins for 90%+ cases)
- `semantic-clustering.md` — HDBSCAN, embedding models, cluster labeling
- `rule-synthesis.md` — XML input, quality criteria, multi-format output

## Conventions

- No PyTorch / sentence-transformers / UMAP — all embeddings via litellm API
- Analyzers are pure Python with no LLM calls (deterministic, fast, testable)
- LLM calls in: comment_classifier (litellm), semantic clustering labels (litellm), synthesizer (claude-agent-sdk)
- Synthesis is an agentic loop — the LLM investigates patterns via git tools before writing rules
- Git tools in `tools.py` are read-only, 10s timeout, 4000 char output cap, input-validated
- Formatters are deterministic — synthesis output goes through quality gate, then template formatting
- All async LLM calls use `litellm.acompletion` with semaphore concurrency control
