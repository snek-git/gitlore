# Rule Synthesis: Turning Pattern Data into AI Coding Assistant Rules

Research document for gitlore's synthesis stage -- the LLM-powered step that converts structured pattern data (co-change coupling, churn hotspots, reverts, clustered PR feedback, commit conventions) into actionable natural language rules for CLAUDE.md, .cursorrules, and AGENTS.md files.

**Priority: quality > cost > latency.**

---

## 1. Model Comparison for Rule Writing Quality

Rule synthesis is fundamentally a **concise technical writing** task, not a reasoning or math task. The model needs to:
- Interpret structured data (JSON/XML of patterns)
- Decide which patterns are worth surfacing as rules
- Write clear, opinionated, actionable instructions

### Model Selection: Configurable via litellm

**Model choice is a configuration concern, not an architecture decision.** The model landscape shifts monthly. gitlore exposes the synthesizer model as a config option, using litellm/OpenRouter for provider abstraction.

Rule synthesis is a **concise technical writing** task. The model must produce imperative, specific, non-obvious instructions — not verbose explanations. Quality of the written output is the primary selection criterion.

**Current strong options for synthesis (SOTA tier):**

| Model | Notes |
|---|---|
| **Claude Opus 4.6** | Excellent instruction-following, structured-to-natural-language |
| **GPT 5.2/5.3 Codex** | Strong at code-aware rule generation |
| **GLM 5** | Worth benchmarking for rule quality |
| **Kimi K2.5** | Worth benchmarking for rule quality |

**Budget fallback options:**

| Model | Notes |
|---|---|
| **DeepSeek V3.2** | Terse, factual output; adequate for straightforward patterns |
| **Gemini 3 Flash** | Fast, cheap, good enough for simple repos |

*Specific models will be benchmarked against real repo data. The prompting strategies, quality criteria, and pipeline design in this document are model-agnostic.*

**Cost is a non-issue for this stage** regardless of model. Synthesis is 1-5 LLM calls per repo with ~5K input and ~2K output tokens each. Even the most expensive SOTA models cost under $0.50/repo.

### Integration: litellm

Use litellm as the LLM gateway with fallback routing:

```python
import litellm

# Model loaded from config — user can swap to any litellm-supported model
response = litellm.completion(
    model=config.models.synthesizer,  # e.g. "anthropic/claude-opus-4-6"
    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": pattern_data}],
    temperature=0.3,  # Low temp for consistent rule writing
)
```

Configure fallback chains in config: SOTA model -> budget fallback (e.g., Opus 4.6 -> DeepSeek V3.2).

---

## 2. Prompt Engineering for Rule Synthesis

### Input Format: XML

**Use XML tags for pattern data input to all models.** XML provides explicit boundaries between pattern categories, which prevents cross-contamination. Research confirms XML outperforms JSON and Markdown for structured LLM input across model families — XML's hierarchical nesting and named tags make it unambiguous for models to parse, while JSON brackets and commas are easily confused in long inputs. XML is also more token-efficient than JSON for nested data (no repeated key quoting).

```xml
<patterns>
  <category name="co-change-coupling">
    <pattern confidence="0.92" occurrences="47" last_seen="2026-01-15">
      <files>src/auth/login.ts, src/auth/session.ts</files>
      <description>These files change together in 92% of commits touching either file</description>
    </pattern>
  </category>
  <category name="churn-hotspots">
    <pattern confidence="0.85" occurrences="23" last_seen="2026-02-01">
      <file>src/api/routes.ts</file>
      <churn_score>8.7</churn_score>
      <description>High edit frequency with frequent reverts, suggesting instability</description>
    </pattern>
  </category>
  <category name="commit-conventions">
    <pattern confidence="0.95" occurrences="312">
      <convention>conventional commits with scope (feat(auth): ...)</convention>
    </pattern>
  </category>
</patterns>
```

XML is the single input format for all models — no need for model-specific formatting logic.

### System Prompt Design

The system prompt is the single most important lever for output quality. Key elements:

```
You are a senior engineer writing configuration rules for an AI coding assistant.
Your rules will be read by an AI (Claude, Cursor, Codex) working on this codebase.

Write rules that are:
- Imperative ("Always update X when changing Y", not "X and Y tend to change together")
- Specific (reference actual file paths, function names, patterns)
- Non-obvious (skip anything the AI would figure out by reading the code)
- Evidence-backed (include a brief reason or confidence indicator)

Do NOT:
- Write rules that are vague platitudes ("follow best practices", "write clean code")
- Hedge or use passive voice ("It is recommended that..." -> "Always...")
- State obvious things ("run tests before committing")
- Include rules with low confidence unless they concern safety/security

Each rule should be 1-2 sentences. Group rules by category.
```

### Few-Shot Examples

Include 2-3 examples of good rules and 2-3 examples of bad rules directly in the system prompt. This is the highest-ROI prompt engineering technique for this task.

**Good rule examples:**
```
- Always update src/auth/session.ts when modifying src/auth/login.ts -- these files are tightly coupled and 92% of PRs that miss one get reverted.
- src/api/routes.ts is a churn hotspot (8.7 churn score). Changes here get reverted 3x more than average. Add extra test coverage and request review from @backend-team.
- Use conventional commits with scope: feat(auth): description. The repo enforces this in CI.
```

**Bad rule examples (with annotations):**
```
BAD: "The authentication module is complex." (not actionable -- what should the AI do?)
BAD: "It is recommended to consider updating session management when login changes." (hedging, passive)
BAD: "Always write tests." (obvious, not specific to this codebase)
BAD: "Files in src/api/ sometimes change together." (vague -- which files? how often?)
```

### Chain-of-Thought: No

Research from Wharton (2025) shows CoT provides marginal benefit for non-reasoning tasks like summarization and synthesis, while increasing latency 20-80% and token costs. Rule writing is a writing/synthesis task, not a multi-step reasoning problem.

**Exception:** If the model needs to **filter** patterns (deciding which of 200 patterns are worth surfacing), a brief `<thinking>` step can help it reason about importance. But this should be a separate filtering call, not mixed into the writing call.

---

## 3. Rule Quality Criteria

Every generated rule should pass these checks (usable as an LLM-as-judge rubric or human review checklist):

| Criterion | Definition | Test |
|---|---|---|
| **Actionable** | The AI assistant can follow it without ambiguity | Does it contain a verb in imperative form? |
| **Specific** | References concrete files, patterns, or conventions | Does it name at least one file, function, or tool? |
| **Opinionated** | Tells the AI what to do, not what was observed | Does it use "always/never/must" rather than "tends to/was observed"? |
| **Concise** | One rule = one sentence or short paragraph | Is it under 50 words? |
| **Evidence-backed** | Includes confidence or rationale | Does it mention frequency, percentage, or consequence? |
| **Non-obvious** | The AI wouldn't figure this out from code alone | Would an AI reading the code independently discover this? If yes, skip it. |
| **Current** | Based on recent patterns, not ancient history | Is the supporting evidence from the last 6-12 months? |

### Quality Gate

Implement a simple quality gate: after generation, run each rule through the criteria checklist (can be done with a cheap model or even regex heuristics). Rules that fail 2+ criteria get dropped.

---

## 4. Confidence Scoring

### Include Confidence -- But Implicitly

Explicit confidence scores (e.g., "[confidence: 0.87]") add clutter and AI assistants don't usefully interpret numeric confidence. Instead, **bake confidence into rule language**:

- **High confidence (>0.85):** "Always update X when changing Y" (imperative, no hedging)
- **Medium confidence (0.6-0.85):** "When changing X, check if Y needs updating -- they change together in ~70% of cases"
- **Low confidence (<0.6):** Exclude by default. Exception: safety-critical patterns get included with explicit uncertainty: "src/crypto/keys.ts has had security-related reverts. Have changes reviewed even if CI passes."

### Determining Confidence

Confidence is computed from the pattern data before it reaches the LLM:

```python
def compute_confidence(pattern):
    score = 0.0
    score += min(pattern.occurrences / 30, 0.4)    # More data points = higher confidence (cap at 0.4)
    score += min(pattern.recency_days / 180, 0.3)   # Recent patterns weighted higher (cap at 0.3)
    score += pattern.consistency * 0.3               # How consistent the pattern is (0-1, cap at 0.3)
    return score
```

Pass the confidence score to the LLM as part of the pattern data. Let the system prompt instruct the LLM on how to translate confidence levels into language strength.

---

## 5. Evidence Citations

### Recommendation: Minimal Inline Citations

Heavy citations (commit hashes, PR links) clutter CLAUDE.md and provide marginal value -- the AI assistant won't click links. But some evidence adds credibility for human readers who review the generated file.

**Format:** Brief parenthetical at the end of a rule.

```
Always update src/auth/session.ts when modifying src/auth/login.ts -- they are
tightly coupled. (47 co-changes in last 6 months, 92% coupling rate)
```

**Do not include:**
- Full commit SHAs (noise; they'll be outdated quickly)
- PR links (CLAUDE.md isn't a web page)
- Detailed statistical breakdowns

**Optional:** Generate a separate `GITLORE_EVIDENCE.md` file with full citations that CLAUDE.md rules can reference. This keeps the rules file clean while providing an audit trail.

```
## In CLAUDE.md:
Always update session.ts when modifying login.ts. (see GITLORE_EVIDENCE.md#coupling-1)

## In GITLORE_EVIDENCE.md:
### coupling-1
Files: src/auth/login.ts, src/auth/session.ts
Coupling rate: 92% (47/51 co-changes)
Recent commits: abc1234, def5678, ...
Last analyzed: 2026-02-10
```

---

## 6. Context Window Management

A large repo might produce hundreds of patterns. At ~100 tokens per pattern, 500 patterns = 50k tokens of input alone. Strategy:

### Pre-LLM Filtering (Deterministic)

Before the LLM sees anything, filter patterns programmatically:

1. **Confidence threshold:** Drop patterns below 0.5 confidence
2. **Recency filter:** Deprioritize patterns with no supporting evidence in the last 6 months
3. **Deduplication:** Merge overlapping patterns (e.g., A<->B coupling and B<->A coupling)
4. **Top-N per category:** Keep the top 20-30 patterns per category, ranked by confidence * impact

This typically reduces hundreds of patterns to 50-100 high-signal ones, fitting comfortably in a single prompt.

### Chunked Synthesis (If Still Too Large)

If filtered patterns still exceed ~80k tokens:

1. **Chunk by category:** Send each category (coupling, churn, conventions, review feedback) as a separate LLM call
2. **Merge pass:** A final LLM call receives all category outputs and:
   - Removes redundancies across categories
   - Orders rules by importance
   - Ensures consistent voice/style

Cost of merge pass: minimal (input is just the generated rules, ~2-4k tokens).

### Avoiding Redundant Rules

Include in the system prompt: "If two patterns lead to the same actionable advice, combine them into a single rule citing both patterns. Never state the same advice twice."

---

## 7. Output Structure

### Recommended CLAUDE.md Sections

```markdown
<!-- Generated by gitlore on 2026-02-10. Do not edit manually -- changes will be overwritten. -->
<!-- Source: github.com/org/repo, analyzed 1,247 commits from 2025-08 to 2026-02 -->

## Architecture & File Coupling

[Rules about which files must change together, module boundaries, dependency patterns]

## Hotspots & Stability

[Rules about high-churn files, frequently reverted areas, fragile code]

## Code Review Patterns

[Rules derived from clustered PR review feedback -- common reviewer complaints, recurring issues]

## Commit & PR Conventions

[Rules about commit message format, branch naming, PR structure]

## Testing Patterns

[Rules about test coverage expectations, files that need extra testing]
```

### Ordering

Within each section, order rules by **confidence * impact** (highest first). The AI assistant reads top-down and earlier rules get more attention weight.

### Section Boundaries

Use standard Markdown `##` headers. Do not use custom delimiters or YAML frontmatter -- CLAUDE.md and AGENTS.md are plain Markdown and tools expect standard structure.

### Metadata Header

Include a machine-readable comment at the top with generation metadata (date, repo, commit range). This supports incremental updates and lets users know the rules are auto-generated.

---

## 8. Multi-Format Output

### Target Formats

| Format | Tool | Notes |
|---|---|---|
| `CLAUDE.md` | Claude Code | Markdown. Read at session start. Keep concise. |
| `.cursorrules` | Cursor | Markdown/plaintext. Similar to CLAUDE.md but placed in project root. |
| `AGENTS.md` | OpenAI Codex, Google Jules, Cursor, etc. | Markdown. Open standard. Supports subdirectory placement. |
| `.github/copilot-instructions.md` | GitHub Copilot | Markdown. |

### Recommendation: LLM Generates Content, Code Handles Formatting

The differences between formats are **structural, not semantic**. The same rule ("always update session.ts when modifying login.ts") should appear in all formats. Only the wrapper changes.

**Pipeline:**
1. LLM generates a single canonical set of rules as structured Markdown (the "content")
2. Deterministic Python code wraps the content into each target format:
   - CLAUDE.md: Add gitlore header comment, output as-is
   - .cursorrules: Same content, different filename
   - AGENTS.md: Same content, potentially add AGENTS.md-specific metadata sections
   - copilot-instructions.md: Same content, place in .github/

This avoids paying for multiple LLM calls that produce nearly identical output, and ensures consistency across formats.

**Exception:** If a format has meaningfully different constraints (e.g., token limits, special syntax), generate format-specific content. As of 2026, all major formats accept plain Markdown, so this isn't needed yet.

---

## 9. Iterative Refinement

### Self-Review Pass: Yes, Conditionally

A second LLM call reviewing the generated rules improves quality measurably, but the ROI depends on the primary model:

| Primary Model Tier | Review Pass Worth It? | Recommended Reviewer |
|---|---|---|
| Budget (DeepSeek V3.2, Gemini 3 Flash) | **Yes** -- catches vague/obvious rules | SOTA model |
| SOTA (Opus 4.6, GPT 5.x, GLM 5, Kimi K2.5) | **No** -- quality is already high | Skip |

### Review Prompt

```
Review these AI coding assistant rules for quality. For each rule, check:
1. Is it actionable? (imperative verb, clear instruction)
2. Is it specific? (names files, functions, or patterns)
3. Is it non-obvious? (wouldn't be apparent from reading the code)
4. Is it concise? (under 50 words)

Remove or rewrite rules that fail 2+ checks. Do not add new rules.
Output only the final revised rule set.
```

### Cost of Review Pass

At Sonnet 4.5 Batch API pricing: ~$0.02 per review call (input: ~2k tokens of rules, output: ~2k tokens). Negligible.

### Human-in-the-Loop

**Recommended for v1:** Generate rules, present to user for review/edit, then write to file. The `--interactive` flag could show rules in a TUI with accept/reject/edit per rule.

**Later:** As confidence in generation quality increases, default to auto-write with a `--review` flag for manual inspection.

---

## 10. Implementation Recommendations

### Synthesis Pipeline Summary

```
Raw Patterns (JSON)
    |
    v
[Deterministic Filtering]  -- confidence threshold, recency, dedup, top-N
    |
    v
[Format as XML]  -- structure patterns for LLM input
    |
    v
[LLM: Generate Rules]  -- Sonnet 4.5, low temperature, few-shot prompt
    |
    v
[Optional: LLM Review Pass]  -- if primary model is budget tier
    |
    v
[Deterministic Quality Gate]  -- regex/heuristic checks on output
    |
    v
[Format for Target]  -- CLAUDE.md / .cursorrules / AGENTS.md
    |
    v
Output File
```

### Key Design Decisions

1. **Model:** Configurable via litellm. SOTA tier (Opus 4.6, GPT 5.x Codex, GLM 5, Kimi K2.5) or budget tier (DeepSeek V3.2, Gemini 3 Flash). Default TBD via benchmarking.
2. **Input format:** XML tags (all models — XML provides explicit boundaries, prevents cross-contamination between pattern categories, and is consistently well-parsed across model families)
3. **Temperature:** 0.2-0.3 (consistent output, slight variation for naturalness)
4. **Few-shot:** 3 good examples, 3 bad examples in system prompt
5. **Chain-of-thought:** No (not a reasoning task)
6. **Confidence:** Implicit in language strength, computed deterministically before LLM
7. **Citations:** Brief inline parentheticals, optional full evidence file
8. **Context management:** Deterministic pre-filtering, chunked synthesis if needed
9. **Multi-format:** Single LLM call, deterministic formatting per target
10. **Review pass:** Conditional on primary model quality
11. **LLM gateway:** litellm for provider abstraction and fallback routing

### Cost Estimates (Per Repository Synthesis)

| Scenario | Calls | Est. Cost Range |
|---|---|---|
| Standard (1 synthesis call) | 1 | $0.002 - $0.08 |
| With review pass | 2 | $0.01 - $0.15 |
| Large repo (chunked) | 5-7 | $0.05 - $0.50 |

*Cost depends on model choice. Budget models (DeepSeek V3.2) at the low end, SOTA (Opus 4.6) at the high end. All scenarios well under $1/repo.*

**Cost is a non-issue for this stage.** Optimize for rule quality, not synthesis cost.

---

## 11. Future: Workflow Synthesis & Agent Config Management Platform

Beyond generating static config files, gitlore's natural evolution is toward a **platform for managing AI agent configurations** across teams and repos:

- **Workflow synthesis**: Generate not just rules but complete agent workflows — multi-step instructions, tool configurations, and task-specific prompts derived from how the team actually works (PR review flows, deployment checklists, incident response patterns observed in git history)
- **Config drift detection**: Monitor whether generated rules are still accurate as the repo evolves. Alert when patterns change significantly enough to warrant re-generation
- **Multi-repo rule aggregation**: For orgs with many repos, synthesize org-wide conventions from cross-repo patterns and push consistent base rules to all repos
- **Agent config versioning**: Track changes to generated configs over time, diff them, roll back if a config update degrades AI assistant performance
- **Config dashboard**: Web UI for reviewing, editing, and approving generated rules before they land in the repo. Audit trail of what patterns produced what rules

This is the path from "CLI tool" to "platform" — but it's post-MVP. The CLI needs to prove value first.
