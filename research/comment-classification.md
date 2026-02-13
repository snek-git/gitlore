# PR Comment Classification with Open Source LLMs

Research for classifying PR review comments into actionable categories using open source models via litellm/OpenRouter.

## Target Categories

| Category | Actionable | Example |
|----------|-----------|---------|
| Bug/correctness | Yes | "this will break when X is null" |
| Architecture/design | Yes | "this should be in the service layer, not the controller" |
| Convention violation | Yes | "we use camelCase here", "always add error handling" |
| Security concern | Yes | "this is vulnerable to injection" |
| Performance | Yes | "this will be O(n^2), use a map instead" |
| Nitpick/style | Yes | "missing semicolon", "rename this variable" |
| Question/discussion | No | "why did you choose this approach?" |
| Praise/approval | No | "LGTM", "nice refactor!" |

---

## 1. Model Comparison for Classification Tasks

### Model Selection: Configurable via litellm

**Model choice is a configuration concern, not an architecture decision.** The model landscape shifts monthly — hardcoding a specific model is wrong. gitlore exposes the classifier model as a config option with sensible defaults, using litellm/OpenRouter for provider abstraction.

All models below support JSON mode and work with the classification prompts in this document.

**Current strong options for classification (cheap/fast tier):**

| Model | Notes |
|-------|-------|
| **DeepSeek V3.2** | Strong cost/quality, proven for structured tasks |
| **Grok 4.1 Fast** | Fast inference, good for batching |
| **GPT-o-s-120B** | 120B-class quality at competitive pricing |
| **Gemini 3 Flash** | Google's fast tier, consistently good at structured tasks |

*Prices fluctuate by provider and change frequently. Check OpenRouter for current pricing.*

**Specific models will be benchmarked against real repo data** once the pipeline is functional. The prompting strategies, batch processing, and JSON mode patterns in this document are model-agnostic and apply regardless of which model is selected.

### Classification Quality Assessment

For a text classification task like ours (8 categories, short inputs ~50-150 tokens):

- **Cost is negligible regardless of model choice.** Even expensive models cost under $2 for 10K comments. The cheap tier costs pennies.
- **70B+ class models handle ambiguous cases better** (convention vs nitpick, multi-label) but for a constrained 8-label task, most modern models perform well.
- **Avoid reasoning models** (e.g., DeepSeek R1 full, o1-style) for classification — chain-of-thought inflates output tokens for zero benefit on a label-picking task.

### Default Configuration

The default classifier model should be whichever cheap/fast model benchmarks best at the time of release. The config looks like:

```toml
[models]
classifier = "openrouter/deepseek/v3.2"  # or grok-4.1-fast, gpt-o-s-120b, gemini-3-flash
```

Users can swap to any litellm-supported model string. The pipeline doesn't care which model is behind the interface.

---

## 2. Prompt Engineering for Classification

### System Prompt Design

```
You are a PR review comment classifier. Given a code review comment, classify it into one or more categories. Respond with JSON only.

Categories:
- bug: Bug or correctness issue (will cause incorrect behavior, crashes, or data loss)
- architecture: Architecture or design concern (wrong layer, missing abstraction, coupling)
- convention: Convention violation (naming, formatting, project-specific patterns)
- security: Security vulnerability (injection, auth bypass, data exposure)
- performance: Performance issue (algorithmic complexity, unnecessary allocations, N+1)
- nitpick: Minor style issue (typos, whitespace, variable naming preference)
- question: Question or discussion (not directly actionable)
- praise: Approval or positive feedback (LGTM, looks good, nice work)
```

### Zero-Shot Prompt

```json
{
  "system": "<system prompt above>",
  "user": "Classify this PR review comment:\n\n\"{comment}\"\n\nRespond with JSON: {\"categories\": [\"<category>\", ...], \"confidence\": <0.0-1.0>}"
}
```

Zero-shot works well for clear-cut cases (praise, obvious bugs) but struggles with:
- Distinguishing `convention` vs `nitpick` (both about style, differ in scope)
- Distinguishing `architecture` vs `convention` (design patterns can be conventions)
- Multi-label edge cases

### Few-Shot Prompt (Recommended)

Including 1-2 examples per category significantly improves boundary cases. Research shows few-shot can approach fine-tuned model accuracy (95% vs 96.2% F1 with just 10 examples in one study).

```
Classify the following PR review comment. A comment may belong to multiple categories.

Examples:
- "This will NPE if user is null" -> {"categories": ["bug"], "confidence": 0.95}
- "Move this logic to the service layer, the controller shouldn't handle business rules" -> {"categories": ["architecture"], "confidence": 0.90}
- "We always use camelCase for method names in this project" -> {"categories": ["convention"], "confidence": 0.92}
- "This is vulnerable to SQL injection, use parameterized queries" -> {"categories": ["security", "bug"], "confidence": 0.95}
- "This is O(n^2), use a Set for lookups instead" -> {"categories": ["performance"], "confidence": 0.93}
- "Typo: 'recieve' should be 'receive'" -> {"categories": ["nitpick"], "confidence": 0.97}
- "Why did you choose Redis over Memcached here?" -> {"categories": ["question"], "confidence": 0.88}
- "LGTM, nice refactor!" -> {"categories": ["praise"], "confidence": 0.98}

Now classify:
"{comment}"

Respond ONLY with JSON: {"categories": [...], "confidence": <float>}
```

### Structured Output (JSON Mode)

Use litellm's `response_format={"type": "json_object"}` parameter to force JSON output. Most models on OpenRouter support this. This eliminates parsing failures from markdown wrapping or extra text.

For additional reliability, validate the response with a simple schema check:

```python
import json

def parse_classification(response_text: str) -> dict:
    result = json.loads(response_text)
    valid_categories = {"bug", "architecture", "convention", "security",
                        "performance", "nitpick", "question", "praise"}
    result["categories"] = [c for c in result["categories"] if c in valid_categories]
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    return result
```

### Multi-Label Classification

Comments often span categories. The prompt explicitly allows multiple labels:
- "This SQL is vulnerable to injection and will also crash on null input" -> `["security", "bug"]`
- "Rename this and move it to the utils module" -> `["nitpick", "architecture"]`

The few-shot examples should include multi-label cases to teach the model this is acceptable.

### Handling Ambiguous Comments

For comments where the model returns low confidence (<0.6):
1. Accept the classification but flag it for potential review
2. Optionally re-classify with a larger model (fallback escalation)
3. For clustering/synthesis purposes, low-confidence classifications are acceptable -- the downstream clustering step will group similar comments regardless

---

## 3. Batch Processing Strategies

### Token Estimation

Typical PR review comment: 20-80 tokens (median ~40 tokens).

Per-comment request:
- System prompt: ~150 tokens (zero-shot) / ~350 tokens (few-shot)
- User message (comment + instructions): ~80 tokens
- Output: ~30 tokens (JSON response)
- **Total per comment: ~260 tokens input, ~30 tokens output (zero-shot)**
- **Total per comment: ~460 tokens input, ~30 tokens output (few-shot)**

### Strategy A: Single Comment Per Call (Recommended for MVP)

Simplest approach. One API call per comment.

```python
import asyncio
import litellm

async def classify_comment(comment: str) -> dict:
    response = await litellm.acompletion(
        model="openrouter/deepseek/deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Classify: "{comment}"\n\nJSON only:'}
        ],
        response_format={"type": "json_object"},
        max_tokens=100,
        temperature=0.0,
    )
    return parse_classification(response.choices[0].message.content)
```

**Pros:** Simple, each comment gets full attention, easy retry on failure.
**Cons:** More API calls, slightly higher total tokens (repeated system prompt).

### Strategy B: Batching Multiple Comments Per Call

Send 5-10 comments in a single call to amortize the system prompt cost.

```python
async def classify_batch(comments: list[str]) -> list[dict]:
    numbered = "\n".join(f"{i+1}. \"{c}\"" for i, c in enumerate(comments))
    response = await litellm.acompletion(
        model="openrouter/deepseek/deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify each comment:\n{numbered}\n\nRespond with JSON array:"}
        ],
        response_format={"type": "json_object"},
        max_tokens=50 * len(comments),
        temperature=0.0,
    )
    return parse_batch_classification(response.choices[0].message.content)
```

**Pros:** ~40% fewer input tokens (system prompt amortized), fewer API calls.
**Cons:** If one comment in the batch causes a malformed response, the whole batch fails. Harder to retry individual failures. Models occasionally skip items or merge adjacent items.

**Recommendation:** Start with Strategy A for reliability. Move to batching (5 comments/call) once the pipeline is stable and cost optimization matters.

### Async Processing with Concurrency Control

```python
import asyncio
from asyncio import Semaphore

async def classify_all(comments: list[str], max_concurrent: int = 20) -> list[dict]:
    semaphore = Semaphore(max_concurrent)

    async def classify_with_limit(comment: str) -> dict:
        async with semaphore:
            return await classify_comment(comment)

    tasks = [classify_with_limit(c) for c in comments]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### OpenRouter Rate Limits

OpenRouter rate limits vary by account tier and model. Typical limits:
- Free tier: 10-20 requests/minute
- Paid tier: 200+ requests/minute (model-dependent)
- Credits-based: generally higher limits

Use litellm's built-in retry with exponential backoff:

```python
litellm.num_retries = 3
litellm.retry_after = 1  # seconds
```

### Retry and Fallback Strategy

```python
MODELS = [
    "openrouter/deepseek/deepseek-r1-distill-llama-70b",  # primary
    "openrouter/meta-llama/llama-3.3-70b-instruct",        # fallback 1
    "openrouter/meta-llama/llama-3.1-8b-instruct",         # fallback 2 (cheap)
]

async def classify_with_fallback(comment: str) -> dict:
    for model in MODELS:
        try:
            return await classify_comment(comment, model=model)
        except Exception:
            continue
    return {"categories": ["question"], "confidence": 0.0}  # safe default
```

---

## 4. Cost Analysis

### Per-Comment Token Estimates

| Approach | Input Tokens | Output Tokens |
|----------|-------------|---------------|
| Zero-shot (single) | ~260 | ~30 |
| Few-shot (single) | ~460 | ~30 |
| Few-shot (batch of 5) | ~560 total (~112 per comment) | ~150 total (~30 per comment) |

### Cost Estimate

Using few-shot single-comment approach (~460 input + ~30 output tokens per comment):

At current cheap-tier pricing, classification costs are negligible regardless of model:

| Scale | Est. Cost Range |
|-------|----------------|
| Small repo (500 comments) | < $0.01 |
| Medium repo (5K comments) | $0.05 - $0.10 |
| Large repo (50K comments) | $0.50 - $2.00 |
| Monorepo (200K comments) | $2.00 - $5.00 |

*Exact cost depends on model choice and current OpenRouter pricing. Cheap-tier models (DeepSeek V3.2, Gemini 3 Flash, Grok 4.1 Fast) cluster at the low end of these ranges.*

**Cost is a non-issue for this stage.** Even at the high end, classifying 200K comments costs less than a coffee. Optimize for quality, not cost.

---

## 5. Alternative: Fine-Tuning a Small Model

### Is Fine-Tuning Worth It?

For an 8B model fine-tuned on labeled PR review comments:

**Potential benefits:**
- A fine-tuned 8B model can match or exceed zero-shot 70B performance (research confirms fine-tuned 4B can match 70B zero-shot)
- Lower latency (smaller model, faster inference)
- Could run locally or on cheap GPU instances

**Requirements:**
- ~500-2000 labeled examples (covering all 8 categories with edge cases)
- LoRA/QLoRA fine-tuning: ~2-4 hours on a single A100/H100
- Labeled data creation: Classify ~200 comments manually, then use a large model to label ~2000 more with human review

**Hosting costs:**
- Self-hosted 8B model on a T4 GPU: ~$0.35/hour (cloud) -- process millions of comments per hour
- API cost for the same volume via OpenRouter: $0.11-$0.17 per 10K comments

**Verdict for MVP: Do not fine-tune.**
- OpenRouter API costs are negligible ($0.17 per 10K comments with 70B model)
- Fine-tuning requires labeled data that does not yet exist
- Self-hosting adds operational complexity for no meaningful cost savings
- Fine-tuning makes sense only if: (a) processing >1M comments/month consistently, (b) need sub-100ms latency, or (c) need offline/airgapped operation

Fine-tuning becomes worthwhile in a later phase if gitlore accumulates enough labeled data from user feedback/corrections.

---

## 6. Alternative: No LLM at All

### Keyword/Regex Heuristics

A heuristic classifier could handle the clearest cases:

```python
import re

PATTERNS = {
    "praise": [
        r"\bLGTM\b", r"\blooks?\s+good\b", r"\bnice\b", r"\bgreat\b",
        r"\b(ship\s+it|approved?)\b", r"^\+1$", r"^:?\+1:?$",
    ],
    "nitpick": [
        r"\bnit(pick)?\b", r"\btypo\b", r"\bwhitespace\b", r"\bspelling\b",
        r"\bmissing\s+(semicolon|comma|period)\b",
    ],
    "security": [
        r"\b(sql\s+)?injection\b", r"\bXSS\b", r"\bCSRF\b", r"\bvulnerab\w+\b",
        r"\bunsaniti[sz]ed\b", r"\bauth(entication|orization)\s+(bypass|missing)\b",
    ],
    "performance": [
        r"\bO\(n[²2]\)\b", r"\bO\(n\s*\*\s*n\)\b", r"\bN\+1\b",
        r"\b(slow|expensiv|inefficien)\w*\b",
    ],
    "bug": [
        r"\b(null|nil|undefined)\s*(pointer|ref|check|exception)\b",
        r"\b(crash|break|fail|NPE|segfault)\w*\b",
        r"\boff[- ]?by[- ]?one\b", r"\brace\s+condition\b",
    ],
    "question": [
        r"^(why|how|what|when|where|should|could|would)\s",
        r"\?$",
    ],
}
```

### Heuristic Accuracy Estimate

| Category | Heuristic Coverage | Notes |
|----------|-------------------|-------|
| Praise | ~90% | Very keyword-driven, few false positives |
| Nitpick | ~60% | "nit:" prefix common, but many nitpicks lack keywords |
| Security | ~70% | Specific vocabulary, but LLM needed for subtle cases |
| Performance | ~50% | O(n^2) easy to match, "consider caching" is harder |
| Bug | ~40% | Keyword overlap with other categories is high |
| Architecture | ~20% | Too varied in language for keywords |
| Convention | ~15% | Project-specific, almost impossible with generic rules |
| Question | ~70% | Question marks help, but rhetorical questions confuse |

**Overall: Heuristics alone cover ~40-50% of comments reliably.**

### Hybrid Approach (Recommended if Cost Were a Concern)

1. Run heuristic classifier first
2. If a heuristic matches with high confidence (multiple keyword hits), accept it
3. Send ambiguous/unmatched comments to the LLM

This could reduce LLM calls by 30-50%. But given that 10K LLM classifications cost $0.17 with R1 Distill, the complexity is not justified for MVP.

**Verdict: Use LLM for all classifications. Heuristics add code complexity for negligible savings.**

One useful heuristic to keep: pre-filter obvious praise/LGTM comments (regex on "LGTM", "+1", "looks good") before sending to LLM. These are ~10-20% of comments in typical repos and are trivially identifiable. This is a single `if` check, not a classification system.

---

## 7. Recommendations

### Model

**Model selection is configurable via litellm/OpenRouter.** Default will be determined by benchmarking against real repo data. Current candidates for the cheap/fast classification tier:

- DeepSeek V3.2
- Grok 4.1 Fast
- GPT-o-s-120B
- Gemini 3 Flash

All support JSON mode and structured output. Use litellm's fallback chain for resilience:

### Prompt Template

Use the few-shot prompt from Section 2 with JSON mode enabled. Include one example per category (8 examples total), with at least one multi-label example.

### Batch Strategy

1. Process comments individually (one per API call) for reliability
2. Use `asyncio.gather` with a semaphore (20 concurrent requests) for throughput
3. Set `temperature=0.0` for deterministic classification
4. Set `max_tokens=100` to cap output cost
5. Retry up to 3 times with exponential backoff, then fall back to next model

### Expected Cost

### Implementation Priority

1. **MVP:** Few-shot classification with configurable model (default TBD via benchmarking), single comment per call, async with concurrency limit
2. **Later:** Batch 5 comments per call if throughput is a bottleneck
3. **Later:** Accumulate corrections/labels for potential fine-tuning
4. **Skip:** Heuristic pre-filtering (not worth the complexity at these prices), reasoning models (overkill for classification)

### litellm Integration

```python
import litellm

# Configure once
litellm.num_retries = 3
litellm.request_timeout = 30

# Model loaded from config — user can swap to any litellm-supported model
# Examples: "openrouter/deepseek/v3.2", "openrouter/x-ai/grok-4.1-fast",
#           "openrouter/google/gemini-3-flash", "openrouter/openai/gpt-o-s-120b"
MODEL = config.models.classifier
```

OpenRouter handles load balancing across providers automatically. litellm handles retries, response format enforcement, and provider abstraction.
