# Churn Hotspot, Revert, and Fix-After Pattern Detection

Research document for gitlore: detecting signals of "things that went wrong" from git history to generate "don't do X" rules for AI coding assistants.

---

## 1. Churn Hotspot Detection

### 1.1 What Are Hotspots?

Hotspots are files (or modules) where high change frequency overlaps with high complexity. The core insight from Adam Tornhill's *Your Code as a Crime Scene* (and CodeScene): complex code that rarely changes is low risk; simple code that changes often is normal development; **complex code that changes often is where defects concentrate**.

### 1.2 Change Frequency Metrics

**Simple commit count:**
```python
# Count commits touching each file
from collections import Counter
file_commits = Counter()
for commit in repo.traverse_commits():
    for mod in commit.modified_files:
        file_commits[mod.new_path] += 1
```

CodeScene's default hotspot criterion is commit frequency per file -- research shows change frequency alone is the single most predictive metric for quality issues.

**Weighted by diff size (relative code churn):**

Nagappan & Ball (ICSE 2005) demonstrated that *relative* churn measures predict defect density far better than absolute counts. Key metrics:

- `churn_ratio = (lines_added + lines_removed) / file_size` -- normalizes by file size
- `churn_count = lines_added + lines_removed` -- raw volume of change
- `net_churn = lines_added - lines_removed` -- growth/shrinkage signal

Their Windows Server 2003 study achieved 89% accuracy discriminating fault-prone vs non-fault-prone binaries using relative churn.

**Recommendation for gitlore:** Use both commit count AND relative churn. Commit count identifies frequently-touched files; relative churn identifies files undergoing heavy rewrites. The intersection is the strongest signal.

### 1.3 Complexity Proxy

CodeScene uses **lines of code (LOC)** as a language-neutral complexity proxy. This is intentionally simple -- LOC correlates well enough with actual complexity and requires no language-specific parsing.

For gitlore's purposes, LOC is sufficient. The hotspot formula is essentially:

```
hotspot_score(file) = change_frequency(file) * complexity(file)
```

Where `complexity(file)` can be LOC at HEAD, and `change_frequency` is a temporally-weighted commit count (see Section 4).

### 1.4 Normalizing by File Age

New files naturally have high recent churn (they were just created). To avoid false positives:

```python
def normalized_churn(file_path, commits, file_creation_date):
    age_days = (now - file_creation_date).days
    if age_days < 30:  # grace period for new files
        return 0
    raw_churn = sum(c.churn for c in commits if c.touches(file_path))
    return raw_churn / age_days  # churn rate per day
```

Alternatively, exclude files younger than N days from hotspot analysis entirely (CodeScene uses configurable temporal windows, typically 2-3 months minimum).

### 1.5 Distinguishing "Actively Developed" from "Problematic" Churn

Not all churn is bad. Key distinguishing signals:

| Signal | Active Development | Problematic Churn |
|--------|-------------------|-------------------|
| Commit messages | Feature keywords ("add", "implement", "refactor") | Fix keywords ("fix", "revert", "hotfix", "patch") |
| Author diversity | Multiple authors (team collaboration) | Single author fixing repeatedly |
| Change coupling | Changes isolated to feature area | Changes ripple across many modules |
| Temporal pattern | Steady cadence | Bursts of rapid changes |
| Revert presence | No reverts | Reverts present in history |

**For gitlore:** Combine churn score with fix-commit ratio. If >30% of commits to a file contain fix-like keywords, it's likely problematic rather than active development.

---

## 2. Revert Detection

### 2.1 Explicit `git revert` Commits

The easiest case. Git revert creates commits with standardized messages:

```
Revert "<original commit message>"

This reverts commit <hash>.
```

Detection pattern:
```python
import re

REVERT_PATTERNS = [
    re.compile(r'^Revert ".*"', re.IGNORECASE),
    re.compile(r'This reverts commit ([0-9a-f]{7,40})', re.IGNORECASE),
    re.compile(r'^Revert\s+([0-9a-f]{7,40})', re.IGNORECASE),
]

def detect_explicit_revert(commit_msg: str) -> str | None:
    """Returns the reverted commit hash if this is a revert, else None."""
    for pattern in REVERT_PATTERNS:
        match = pattern.search(commit_msg)
        if match and match.groups():
            return match.group(1)
    return None
```

### 2.2 Manual Reverts (Diff Inversion)

When developers manually undo changes without `git revert`, detection requires diff comparison. A manual revert of commit A by commit B means B's diff is approximately the inverse of A's diff (additions become deletions and vice versa).

**Algorithm:**
```python
def is_inverse_diff(diff_a, diff_b, threshold=0.8):
    """Check if diff_b approximately inverts diff_a.

    Compare added lines in A with removed lines in B and vice versa.
    """
    a_added = set(diff_a.added_lines)
    a_removed = set(diff_a.removed_lines)
    b_added = set(diff_b.added_lines)
    b_removed = set(diff_b.removed_lines)

    # B should remove what A added, and add what A removed
    if not a_added and not a_removed:
        return False

    forward_match = len(a_added & b_removed) / max(len(a_added), 1)
    reverse_match = len(a_removed & b_added) / max(len(a_removed), 1)

    similarity = (forward_match + reverse_match) / 2
    return similarity >= threshold
```

**Challenges:**
- Must compare diffs on the same files
- Whitespace and formatting changes create noise
- Need to normalize line content (strip whitespace, ignore blank lines)
- Computationally expensive: O(n^2) if comparing all commit pairs

**Optimization:** Only compare commits touching the same files, within a reasonable time window (e.g., 30 days). Most manual reverts happen within days of the original commit.

### 2.3 Partial Reverts

Developers sometimes revert only part of a commit. Detection strategy:

```python
def partial_revert_score(diff_a, diff_b):
    """Score how much of diff_a is undone by diff_b. Returns 0.0-1.0."""
    a_added = set(diff_a.added_lines)
    b_removed = set(diff_b.removed_lines)

    if not a_added:
        return 0.0

    overlap = len(a_added & b_removed)
    return overlap / len(a_added)
```

Consider a partial revert significant if >30% of a commit's additions are removed by a subsequent commit on the same files.

### 2.4 Revert-of-Revert Chains

When commit C reverts commit B which reverted commit A, the net effect is re-applying A. This indicates indecision or a rushed initial revert.

```python
def detect_revert_chains(commits):
    """Detect chains of reverts. Returns list of chain tuples."""
    revert_map = {}  # reverted_hash -> reverting_commit
    chains = []

    for commit in commits:
        reverted = detect_explicit_revert(commit.msg)
        if reverted:
            revert_map[reverted] = commit.hash
            # Check if this commit reverts a revert
            if reverted in revert_map.values():
                original = [k for k, v in revert_map.items() if v == reverted][0]
                chains.append((original, reverted, commit.hash))

    return chains
```

**For gitlore:** Revert chains are strong signals. They indicate the original change was contentious or problematic -- good fodder for "be careful with X" rules.

### 2.5 False Positive Mitigation

Not all reverts indicate mistakes:
- **Feature flag teardown:** Reverting a feature flag after launch is normal
- **Release branch management:** Reverts on release branches to defer features
- **Experiment cleanup:** A/B test code removal

Mitigation strategies:
- Check if the revert commit message contains "release", "deploy", "feature flag", "experiment"
- Reverts on non-main branches (especially release branches) are less likely to be mistakes
- Reverts followed by a re-implementation with changes are more meaningful than clean reverts

---

## 3. Fix-After Patterns

### 3.1 Quick Remedy Commits (Academic Research)

The most rigorous study on this topic is "Quick remedy commits and their impact on mining software repositories" (Empirical Software Engineering, 2021). Key findings:

**Detection heuristic (two filters):**

1. **Temporal filter:** Commits by the same author within 5 minutes of each other (based on first-quartile analysis of inter-commit intervals)
2. **Lexical filter:** Commit messages containing patterns like `(former|last|prev|previous) commit`

From 1,041,397 candidate commits, 1,577 matched both criteria. Manual validation of 500 samples found 458 true positives (91.6% precision).

**Taxonomy of quick remedy commits:**
- Missing code changes (most common) -- forgot to include a file, incomplete rename
- Bug fixes for just-introduced bugs
- Code refactoring/cleanup of previous commit
- Build/config fixes after code changes

### 3.2 Broader Fix-After Heuristics

For gitlore's purposes, we want a wider net than 5 minutes. Recommended multi-tier approach:

**Tier 1 -- High confidence (minutes):**
```python
def is_immediate_fix(commit_a, commit_b):
    """Same author, same files, within 30 minutes."""
    same_author = commit_a.author.email == commit_b.author.email
    time_delta = (commit_b.author_date - commit_a.author_date).total_seconds()
    same_files = set(commit_a.files) & set(commit_b.files)
    return same_author and 0 < time_delta < 1800 and len(same_files) > 0
```

**Tier 2 -- Medium confidence (hours):**
```python
def is_followup_fix(commit_a, commit_b):
    """Same author, same files, within 4 hours, fix-like message."""
    FIX_KEYWORDS = {'fix', 'oops', 'typo', 'forgot', 'actually', 'woops',
                    'mistake', 'correct', 'amend', 'patch', 'broken', 'missed'}
    same_author = commit_a.author.email == commit_b.author.email
    time_delta = (commit_b.author_date - commit_a.author_date).total_seconds()
    same_files = set(commit_a.files) & set(commit_b.files)
    msg_lower = commit_b.msg.lower()
    has_fix_keyword = any(kw in msg_lower for kw in FIX_KEYWORDS)
    return same_author and 0 < time_delta < 14400 and len(same_files) > 0 and has_fix_keyword
```

**Tier 3 -- Lower confidence (days, any author):**
```python
def is_bug_fix_for(commit_a, commit_b, max_days=7):
    """Different author may fix same files within a week. Requires strong message signal."""
    STRONG_FIX_PATTERNS = [
        re.compile(r'fix(es|ed)?\s+(bug|issue|error|crash|regression)', re.I),
        re.compile(r'revert', re.I),
        re.compile(r'hotfix', re.I),
    ]
    time_delta = (commit_b.author_date - commit_a.author_date).days
    same_files = set(commit_a.files) & set(commit_b.files)
    has_strong_signal = any(p.search(commit_b.msg) for p in STRONG_FIX_PATTERNS)
    return 0 < time_delta <= max_days and len(same_files) > 0 and has_strong_signal
```

### 3.3 Commit Message Signal Words

Ranked by specificity (higher = more confident it's fixing a mistake):

| Confidence | Keywords/Patterns |
|-----------|-------------------|
| Very High | "revert", "oops", "woops", "my bad" |
| High | "hotfix", "fix bug", "fix crash", "fix regression", "broken" |
| Medium | "fix", "patch", "correct", "typo", "forgot", "missed", "actually" |
| Low | "update", "adjust", "tweak", "change" |
| Noise | "refactor", "cleanup", "improve", "optimize" |

Also useful: `git commit --fixup` creates messages starting with `fixup!` -- these explicitly reference the commit they fix.

### 3.4 SZZ Algorithm (Bug-Introducing Commit Detection)

The SZZ algorithm (Sliwerski, Zimmermann, Zeller 2005) works backward from fix commits to identify the commit that introduced the bug:

**Algorithm:**
1. Identify a bug-fixing commit (from message keywords or issue tracker links)
2. Extract the lines modified (deleted/changed) in the fix
3. Use `git blame` on the pre-fix version to find which commit last touched those lines
4. That commit is the bug-introducing commit

**SZZ Variants and their performance (Linux kernel study, 76K commits):**

| Variant | Precision | Recall | F1 | Key Improvement |
|---------|-----------|--------|-----|-----------------|
| B-SZZ | 0.42 | 0.58 | 0.49 | Basic implementation |
| AG-SZZ | 0.42 | 0.56 | 0.48 | Filters blank lines, comments |
| MA-SZZ | 0.40 | 0.55 | 0.46 | Filters meta-changes (merges, properties) |
| L-SZZ | 0.57 | 0.43 | 0.49 | Line-mapping improvements |
| R-SZZ | 0.59 | 0.45 | 0.51 | Refactoring-aware |
| RA-SZZ | 0.97* | 0.36* | -- | Integrates RefDiff/RefactoringMiner |

*RA-SZZ trades recall for very high precision.

**Ghost commit problem:** 17.47% of bug-fixing commits are "ghost commits" where the fix doesn't touch the same lines as the bug introduction (e.g., the bug was a missing null check -- the fix adds lines, the introduction deleted no lines). SZZ fundamentally cannot handle these cases.

**Recommendation for gitlore:** Full SZZ is overkill and has mediocre accuracy. Instead, use a simplified approach:
1. Identify fix commits by message keywords
2. Use `git blame` on modified lines to find the previous author/commit
3. If the blamed commit is recent (< 30 days) and by a different author, flag it as a potential bug-introducing commit
4. Capture both the fix and introduction context for LLM synthesis

### 3.5 Distinguishing Iterative Development from Mistake Correction

This is the hardest problem. Heuristics:

**Signals of iterative development (not mistakes):**
- Commit messages describe new functionality ("add", "implement", "extend")
- Changes are additive (mostly additions, few deletions of recent code)
- Part of a feature branch with a clear PR/merge
- Follows a consistent cadence (daily commits on same feature)

**Signals of mistake correction:**
- Commit messages reference fixing something ("fix", "oops", "revert")
- Changes are subtractive or corrective (deleting/modifying recently added code)
- Very short time gap (minutes to hours)
- Small, targeted changes (few lines) to recently-added code
- Commit affects build/test configuration after a code change

**Composite scoring:**
```python
def mistake_probability(commit_pair):
    """Score 0.0-1.0 indicating likelihood this is a mistake correction."""
    score = 0.0
    time_gap_hours = time_delta(commit_pair).total_seconds() / 3600

    # Time proximity (closer = more likely a fix)
    if time_gap_hours < 0.5: score += 0.3
    elif time_gap_hours < 4: score += 0.2
    elif time_gap_hours < 24: score += 0.1

    # Message signals
    if has_fix_keywords(commit_pair.fix.msg): score += 0.3
    if has_oops_keywords(commit_pair.fix.msg): score += 0.2

    # Same author (self-fix)
    if same_author(commit_pair): score += 0.1

    # Small change size (targeted fix)
    if commit_pair.fix.lines_changed < 10: score += 0.1

    return min(score, 1.0)
```

---

## 4. Temporal Decay Functions

### 4.1 Why Decay Matters

A revert from 3 years ago is less relevant than one from last month. Code conventions evolve, teams change, architectures shift. Temporal weighting ensures recent signals dominate.

### 4.2 Decay Function Options

**Exponential decay (recommended):**
```python
import math

def exponential_weight(age_days: float, half_life_days: float = 180) -> float:
    """Weight decays by half every half_life_days."""
    return math.exp(-0.693 * age_days / half_life_days)
    # Equivalent: 2 ** (-age_days / half_life_days)
```

Properties: smooth, continuous, mathematically well-behaved. Most natural model for "relevance fading over time." Used in most information retrieval systems.

**Linear decay:**
```python
def linear_weight(age_days: float, max_age_days: float = 365) -> float:
    """Weight decreases linearly to 0 at max_age."""
    return max(0, 1 - age_days / max_age_days)
```

Properties: simpler to reason about, hard cutoff at max_age. Risk: abrupt transition at boundary.

**Step function:**
```python
def step_weight(age_days: float) -> float:
    """Recent/old binary classification."""
    if age_days < 90: return 1.0    # last quarter
    if age_days < 365: return 0.5   # last year
    return 0.1                       # older
```

Properties: easiest to explain to users, but loses granularity.

### 4.3 Half-Life Selection

Research and practice suggest different half-lives for different contexts:

| Signal Type | Recommended Half-Life | Rationale |
|------------|----------------------|-----------|
| Code conventions | 6-12 months | Conventions evolve slowly |
| Churn hotspots | 3-6 months | Recent activity matters most |
| Reverts | 6 months | Lessons from reverts stay relevant longer |
| Fix-after pairs | 3 months | Quick fixes are most relevant when recent |
| Bug-introducing patterns | 9-12 months | Architectural lessons persist |

**For gitlore:** Start with a 6-month half-life as default across all signal types. This balances recency bias against losing historical context. Make it configurable.

### 4.4 Differential Decay by Signal Type

Reverts and bug patterns should decay more slowly than simple churn metrics because:
1. They encode **lessons** ("don't do X") rather than **activity** ("file Y is busy")
2. The lesson remains valid even after the code has stabilized
3. They're rarer, so losing signals is more costly

Recommended: Use a `decay_multiplier` per signal type:
```python
DECAY_MULTIPLIERS = {
    'churn': 1.0,        # base half-life (e.g., 6 months)
    'revert': 1.5,       # 9-month effective half-life
    'fix_after': 0.75,   # 4.5-month effective half-life
    'bug_intro': 2.0,    # 12-month effective half-life
}

def weighted_score(signal_type, age_days, base_half_life=180):
    effective_half_life = base_half_life * DECAY_MULTIPLIERS[signal_type]
    return exponential_weight(age_days, effective_half_life)
```

---

## 5. Evidence Extraction

For each detected pattern, capture sufficient context for downstream LLM synthesis to generate actionable rules.

### 5.1 Data Model

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class SignalType(Enum):
    CHURN_HOTSPOT = "churn_hotspot"
    REVERT = "revert"
    FIX_AFTER = "fix_after"
    BUG_INTRODUCING = "bug_introducing"

@dataclass
class DetectedSignal:
    signal_type: SignalType
    confidence: float              # 0.0-1.0
    temporal_weight: float         # decay-adjusted weight

    # Primary commit
    commit_hash: str
    commit_message: str
    author: str
    timestamp: datetime
    files_changed: list[str]

    # Related commit (for pairs like revert/original, fix/broken)
    related_commit_hash: str | None = None
    related_commit_message: str | None = None
    related_author: str | None = None
    related_timestamp: datetime | None = None

    # Diff context
    diff_summary: str | None = None  # human-readable summary of what changed
    lines_added: int = 0
    lines_removed: int = 0

    # Metadata
    tags: list[str] = field(default_factory=list)  # e.g., ["same-author", "within-1-hour"]
```

### 5.2 What to Capture Per Signal Type

**Churn hotspots:**
- File path, current LOC, total commit count, weighted churn score
- Top 5 most recent commit messages touching the file
- Author distribution (number of distinct authors)
- Change coupling partners (files that change together)

**Reverts:**
- Original commit: hash, message, author, timestamp, full file list
- Revert commit: hash, message, author, timestamp
- Time between original and revert
- Whether it was a `git revert` or manual revert
- Whether the original was subsequently re-applied (revert-of-revert)

**Fix-after pairs:**
- Both commits: hash, message, author, timestamp
- Time gap between them
- Which files were touched by both
- Tier classification (immediate/followup/delayed)
- The fix commit's diff (what was actually corrected)

**Bug-introducing commits:**
- The fix commit and its message (describes the bug)
- The blamed/introducing commit and its message
- The specific lines that were blamed
- Time between introduction and fix

### 5.3 Context for LLM Synthesis

The downstream LLM needs enough context to generate rules like:
> "When modifying authentication middleware, always update both the token validator and the session handler together -- these have been a source of bugs (3 fix commits in 6 months)."

To enable this, each signal should include:
1. **What happened** (commit messages from both sides of the pair)
2. **Where it happened** (file paths, ideally function/class names from diff headers)
3. **How often** (count of similar patterns on same files)
4. **How recently** (temporal weight)
5. **Aggregate patterns** (cluster signals by file or directory)

---

## 6. Practical Implementation

### 6.1 Architecture Overview

```
git log / PyDriller
       |
       v
  Raw Commits (hash, msg, author, date, files, diffs)
       |
       v
  +-----------+  +-----------+  +-----------+
  | Churn     |  | Revert    |  | Fix-After |
  | Analyzer  |  | Detector  |  | Detector  |
  +-----------+  +-----------+  +-----------+
       |              |              |
       v              v              v
  DetectedSignal[]  (unified signal format)
       |
       v
  Temporal Weighting & Scoring
       |
       v
  Aggregation & Clustering by file/directory
       |
       v
  Evidence Documents (for LLM synthesis)
```

### 6.2 Performance Considerations

**Git traversal is the bottleneck.** PyDriller processes ~1 commit per 800ms including diff parsing. For a repo with 10,000 commits, that's ~2.2 hours.

**Optimizations:**

1. **Single-pass traversal:** Extract all signals in one pass through commit history. Don't traverse separately for churn, reverts, and fix-after.

```python
from pydriller import Repository

def extract_all_signals(repo_path, since=None):
    churn_data = defaultdict(list)
    recent_commits = deque(maxlen=1000)  # sliding window for fix-after
    revert_candidates = []

    for commit in Repository(repo_path, since=since).traverse_commits():
        # Churn: accumulate per-file stats
        for mod in commit.modified_files:
            churn_data[mod.new_path or mod.old_path].append({
                'hash': commit.hash,
                'date': commit.author_date,
                'added': mod.added_lines,
                'removed': mod.deleted_lines,
            })

        # Reverts: check message
        reverted = detect_explicit_revert(commit.msg)
        if reverted:
            revert_candidates.append((commit, reverted))

        # Fix-after: compare against recent commits
        for prev in recent_commits:
            if is_immediate_fix(prev, commit) or is_followup_fix(prev, commit):
                yield make_fix_after_signal(prev, commit)

        recent_commits.append(commit)
```

2. **Use `git log` directly for metadata, PyDriller only for diffs.** Parsing `git log --format` is orders of magnitude faster than full PyDriller traversal. Use PyDriller selectively for commits that need diff analysis.

```python
import subprocess

def fast_commit_list(repo_path, since=None):
    """Get commit metadata via git log (fast) without diffs."""
    cmd = ['git', '-C', repo_path, 'log',
           '--format=%H%x00%an%x00%ae%x00%aI%x00%s',
           '--name-only']
    if since:
        cmd.append(f'--since={since}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse the output into commit objects
    # ...
```

3. **Time-bound the analysis.** For most use cases, only the last 12-24 months matter (given temporal decay). Use `--since` to limit traversal.

4. **Cache intermediate results.** Store per-file churn data so re-analysis with different parameters doesn't require re-traversal.

### 6.3 Manual Revert Detection (Expensive Path)

Manual revert detection requires comparing diffs between commit pairs, which is O(n^2) in the worst case. Practical approach:

```python
def detect_manual_reverts(repo_path, time_window_days=30):
    """Only compare commits touching the same files within time window."""
    file_commits = defaultdict(list)  # file -> [(commit, diff)]

    for commit in Repository(repo_path).traverse_commits():
        for mod in commit.modified_files:
            path = mod.new_path or mod.old_path
            file_commits[path].append((commit, mod))

    # For each file, compare commits within time window
    for path, entries in file_commits.items():
        for i, (c1, d1) in enumerate(entries):
            for j in range(i+1, len(entries)):
                c2, d2 = entries[j]
                gap = (c2.author_date - c1.author_date).days
                if gap > time_window_days:
                    break
                if is_inverse_diff(d1, d2, threshold=0.7):
                    yield ManualRevertSignal(original=c1, reverter=c2, file=path)
```

### 6.4 Putting It Together

```python
def analyze_repository(repo_path, config=None):
    """Main entry point. Returns scored, weighted signals."""
    config = config or AnalysisConfig()
    signals = []

    # Phase 1: Fast metadata scan
    commits = fast_commit_list(repo_path, since=config.since)

    # Phase 2: Detect reverts (message-based, fast)
    revert_signals = detect_explicit_reverts(commits)
    signals.extend(revert_signals)

    # Phase 3: Detect fix-after patterns (metadata-based, fast)
    fix_signals = detect_fix_after_patterns(commits)
    signals.extend(fix_signals)

    # Phase 4: Compute churn (needs line counts -- use git log --numstat)
    churn_signals = compute_churn_hotspots(repo_path, config)
    signals.extend(churn_signals)

    # Phase 5: Manual revert detection (expensive, optional)
    if config.detect_manual_reverts:
        manual_reverts = detect_manual_reverts(repo_path, config.revert_window_days)
        signals.extend(manual_reverts)

    # Phase 6: Apply temporal weighting
    now = datetime.now(timezone.utc)
    for signal in signals:
        age_days = (now - signal.timestamp).days
        signal.temporal_weight = weighted_score(
            signal.signal_type.value, age_days, config.base_half_life
        )

    # Phase 7: Aggregate by file/directory
    return aggregate_signals(signals)
```

---

## References

- Tornhill, A. *Your Code as a Crime Scene* (Pragmatic Bookshelf, 2015). Hotspot methodology, change coupling.
- Tornhill, A. *Software Design X-Rays* (Pragmatic Bookshelf, 2018). Deeper analysis patterns.
- Nagappan, N. & Ball, T. "Use of Relative Code Churn Measures to Predict System Defect Density" (ICSE 2005). Relative churn as defect predictor.
- Sliwerski, J., Zimmermann, T., & Zeller, A. "When Do Changes Induce Fixes?" (MSR 2005). Original SZZ algorithm.
- Borg, M. et al. "SZZ Unleashed: An Open Implementation of the SZZ Algorithm" (2019). Open-source SZZ implementation.
- Rosa, G. et al. "Evaluating SZZ Implementations: An Empirical Study on the Linux Kernel" (2023). SZZ variant comparison, ghost commit analysis.
- Aladics, T. et al. "Quick remedy commits and their impact on mining software repositories" (Empirical Software Engineering, 2021). Fix-after detection heuristics.
- CodeScene documentation: https://docs.enterprise.codescene.io/
- Code Maat: https://github.com/adamtornhill/code-maat
- PyDriller: https://github.com/ishepard/pydriller
