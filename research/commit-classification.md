# Commit Pattern Classification Research

Research on classifying commits by type, detecting conventions, identifying revert/fix-after chains, and extracting team conventions from commit history.

---

## 1. Conventional Commits Parsing

### The Specification

The conventional commits spec (v1.0.0) defines this structure:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `ci`, `build`, `style`, `perf` (only `feat` and `fix` are mandated; the rest come from the Angular convention via `@commitlint/config-conventional`).

Breaking changes are indicated by `!` before the colon or a `BREAKING CHANGE:` footer.

### Core Regex Pattern

The canonical regex for parsing a conventional commit subject line:

```python
CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)"       # type (feat, fix, etc.)
    r"(?:\((?P<scope>[^)]+)\))?"   # optional scope in parens
    r"(?P<breaking>!)?"            # optional breaking change indicator
    r":\s+"                        # colon + space separator
    r"(?P<subject>.+)"             # subject/description
)
```

This handles: `feat: add login`, `fix(auth): handle token expiry`, `refactor!: restructure API`.

### Extracting Additional Components

**Breaking changes in footer:**
```python
BREAKING_CHANGE_RE = re.compile(
    r"^BREAKING[ -]CHANGE:\s*(?P<description>.+)", re.MULTILINE
)
```

**Ticket references** (multiple formats exist in the wild):
```python
# JIRA-style: PROJ-123, ABC-4567
JIRA_TICKET_RE = re.compile(r"\b(?P<ticket>[A-Z]{2,10}-\d+)\b")

# GitHub-style: #123, fixes #456, closes #789
GITHUB_ISSUE_RE = re.compile(
    r"(?:(?:fix(?:es)?|close[sd]?|resolve[sd]?)\s+)?#(?P<issue>\d+)", re.IGNORECASE
)

# GitLab MR-style: !123
GITLAB_MR_RE = re.compile(r"!(?P<mr>\d+)")
```

**Footer parsing** (git trailer format):
```python
# "Token: value" or "Token #value"
FOOTER_RE = re.compile(
    r"^(?P<token>[A-Za-z-]+)(?::\s|\ #)(?P<value>.+)", re.MULTILINE
)
```

### Existing Libraries

**python-semantic-release** (`semantic_release.commit_parser.conventional`):
- `ConventionalCommitParser` class with full parsing pipeline
- Extracts: type, scope, subject, breaking changes, linked issues, linked MRs
- Handles squashed merge commits (splits them into individual messages for parsing)
- Has `is_merge_commit()` detection
- Configurable via `ConventionalCommitParserOptions` (custom type-to-bump mappings, allowed types, etc.)
- Also has `EmojiCommitParser` for gitmoji-style commits
- **Verdict**: Mature and well-tested, but heavy dependency. The parsing logic is straightforward enough to reimplement with regex. The real value is in the `ParsedMessageResult` data structure design, which we should mirror.

**commitizen** (`commitizen.cz.conventional_commits`):
- `commit_parser` regex is configurable in `pyproject.toml`/`.cz.yaml`/`.cz.json`
- Default: `r"^(?P<change_type>feat|fix|refactor|perf|BREAKING CHANGE)(?:\((?P<scope>[^()\r\n]*)\))?(?P<breaking>!)?:\s(?P<message>.*)?"`
- Supports custom `change_type_map`, `bump_pattern`, `bump_map`
- Can detect project's commitizen config and use it to understand their conventions
- **Verdict**: The custom parser config is a goldmine for convention detection. If a repo has `.cz.toml`/`pyproject.toml` with commitizen config, we can read it directly to understand their commit conventions.

**Recommendation**: Don't depend on either library at runtime. Instead, reimplement the regex parsing (it's ~20 lines) and read commitizen/commitlint config files when present to discover project-specific conventions.

### Non-Standard But Consistent Patterns

Many teams use consistent patterns that aren't conventional commits:

```python
# Pattern: "[module] description"
BRACKET_MODULE_RE = re.compile(r"^\[(?P<module>[^\]]+)\]\s*(?P<subject>.+)")

# Pattern: "JIRA-123: description" or "JIRA-123 description"
TICKET_PREFIX_RE = re.compile(r"^(?P<ticket>[A-Z]{2,10}-\d+)[:\s]+(?P<subject>.+)")

# Pattern: "module: description" (Linux kernel style)
KERNEL_STYLE_RE = re.compile(r"^(?P<module>[a-z][a-z0-9/_.-]+):\s+(?P<subject>.+)")

# Pattern: "emoji description" (gitmoji)
GITMOJI_RE = re.compile(
    r"^(?P<emoji>[\U0001F300-\U0001FAFF]|:\w+:)\s*(?P<subject>.+)"
)

# Pattern: "#123 description" (issue-first)
ISSUE_FIRST_RE = re.compile(r"^#(?P<issue>\d+)\s+(?P<subject>.+)")
```

**Strategy**: Try each pattern against a sample of commits. The pattern that matches the highest percentage is likely the team's convention. Multiple patterns can coexist (e.g., JIRA ticket + conventional commit type).

---

## 2. Detecting Implicit Conventions

### The Problem

Many teams have no formal commit convention but develop organic patterns. We need to detect these statistically.

### Prefix Frequency Analysis

```python
def extract_first_word(message: str) -> str:
    """Get first word, normalized."""
    return message.split()[0].lower().rstrip(":") if message.split() else ""

def analyze_prefixes(messages: list[str]) -> dict[str, float]:
    """Return prefix -> frequency mapping."""
    from collections import Counter
    prefixes = Counter(extract_first_word(m) for m in messages)
    total = len(messages)
    return {prefix: count / total for prefix, count in prefixes.most_common(20)}
```

If `fix`, `add`, `update`, `remove` collectively account for >60% of first words, the team has an implicit verb-first convention even without formal conventional commits.

### N-gram Analysis for Structural Patterns

```python
from collections import Counter
import re

def extract_structural_pattern(message: str) -> str:
    """Replace content with structural tokens."""
    # Replace specific content with type tokens
    pattern = message.split("\n")[0]  # first line only
    pattern = re.sub(r'[A-Z]{2,10}-\d+', '<TICKET>', pattern)
    pattern = re.sub(r'#\d+', '<ISSUE>', pattern)
    pattern = re.sub(r'\([^)]+\)', '(<SCOPE>)', pattern)
    # Detect structural markers
    if pattern.startswith('['):
        return "[<MODULE>] <desc>"
    if re.match(r'^[a-z]+:', pattern):
        return "<type>: <desc>"
    if re.match(r'^[A-Z]{2,10}-\d+', pattern):
        return "<TICKET>: <desc>"
    return "<freeform>"

def detect_dominant_pattern(messages: list[str]) -> tuple[str, float]:
    """Find the most common structural pattern and its prevalence."""
    patterns = Counter(extract_structural_pattern(m) for m in messages)
    most_common, count = patterns.most_common(1)[0]
    return most_common, count / len(messages)
```

### Character-Level Structural Signals

Count messages that:
- Start with a capital letter vs lowercase (convention signal)
- End with a period vs no period
- Contain a colon in the first 30 characters (type separator)
- Start with a verb vs noun (imperative mood check)
- Are under 50 / 72 / unlimited characters (line length discipline)

```python
def structural_signals(messages: list[str]) -> dict[str, float]:
    n = len(messages)
    return {
        "starts_lowercase": sum(1 for m in messages if m[0:1].islower()) / n,
        "starts_uppercase": sum(1 for m in messages if m[0:1].isupper()) / n,
        "ends_with_period": sum(1 for m in messages if m.rstrip().endswith(".")) / n,
        "has_colon_prefix": sum(1 for m in messages if ":" in m[:30]) / n,
        "under_50_chars": sum(1 for m in messages if len(m.split("\n")[0]) <= 50) / n,
        "under_72_chars": sum(1 for m in messages if len(m.split("\n")[0]) <= 72) / n,
        "has_body": sum(1 for m in messages if "\n\n" in m) / n,
        "imperative_mood": sum(1 for m in messages if _is_imperative(m)) / n,
    }

def _is_imperative(message: str) -> bool:
    """Rough heuristic: first word doesn't end in -ed, -ing, -s."""
    first_word = message.split()[0].lower().rstrip(":") if message.split() else ""
    return not any(first_word.endswith(s) for s in ("ed", "ing", "ied"))
```

### When to Use LLM vs Regex

**Regex handles (estimated 90-95% of cases):**
- Conventional commits and their variants
- Ticket-prefixed messages
- Bracket-module patterns
- Gitmoji
- Verb-first patterns
- Structural signals (casing, punctuation, length)

**LLM needed for (~5-10% edge cases):**
- Classifying freeform messages that don't follow any structural pattern
- Determining if a commit is a feature vs bugfix from the description alone (e.g., "handle edge case where user has no email" -- fix or feat?)
- Detecting sarcasm/frustration commits vs genuine descriptions ("just make it work" vs "implement retry logic")
- Understanding domain-specific abbreviations

**Verdict**: Use regex/heuristics for convention detection and structural classification. Reserve LLM for semantic classification of freeform messages only when needed downstream (e.g., for generating rules about what types of changes the team makes).

---

## 3. Commit Type Classification (Beyond Message Parsing)

### Diff-Based Heuristics

When the commit message doesn't clearly indicate the type, the diff content itself is highly informative.

```python
from dataclasses import dataclass

@dataclass
class DiffProfile:
    """Characterize a commit by what it changed."""
    files_changed: list[str]
    additions: int
    deletions: int

    @property
    def file_extensions(self) -> set[str]:
        return {f.rsplit(".", 1)[-1] for f in self.files_changed if "." in f}

    @property
    def is_test_only(self) -> bool:
        return all(self._is_test_file(f) for f in self.files_changed)

    @property
    def is_docs_only(self) -> bool:
        doc_exts = {"md", "rst", "txt", "adoc"}
        doc_dirs = {"docs/", "doc/", "documentation/"}
        return all(
            f.rsplit(".", 1)[-1] in doc_exts or
            any(f.startswith(d) for d in doc_dirs)
            for f in self.files_changed
        )

    @property
    def is_config_only(self) -> bool:
        config_patterns = {
            "toml", "yaml", "yml", "json", "ini", "cfg", "conf",
            ".gitignore", ".eslintrc", ".prettierrc", "Makefile",
            "Dockerfile", "docker-compose",
        }
        return all(
            f.rsplit(".", 1)[-1] in config_patterns or
            any(p in f for p in config_patterns)
            for f in self.files_changed
        )

    @property
    def is_ci_only(self) -> bool:
        ci_paths = {
            ".github/workflows/", ".gitlab-ci", "Jenkinsfile",
            ".circleci/", ".travis.yml", "azure-pipelines",
            ".buildkite/",
        }
        return all(
            any(f.startswith(p) or p in f for p in ci_paths)
            for f in self.files_changed
        )

    @property
    def likely_refactor(self) -> bool:
        """Roughly equal additions and deletions, same file set."""
        if self.additions == 0 or self.deletions == 0:
            return False
        ratio = min(self.additions, self.deletions) / max(self.additions, self.deletions)
        return ratio > 0.6 and self.additions > 5

    @staticmethod
    def _is_test_file(path: str) -> bool:
        test_indicators = [
            "test_", "_test.", ".test.", "tests/", "test/",
            "spec_", "_spec.", ".spec.", "specs/", "spec/",
            "__tests__/", "testing/",
        ]
        return any(ind in path.lower() for ind in test_indicators)

    @property
    def has_test_files(self) -> bool:
        return any(self._is_test_file(f) for f in self.files_changed)

    def classify(self) -> str:
        """Best-effort classification from diff profile alone."""
        if self.is_test_only:
            return "test"
        if self.is_docs_only:
            return "docs"
        if self.is_ci_only:
            return "ci"
        if self.is_config_only:
            return "chore"
        if self.likely_refactor:
            return "refactor"
        if self.additions > 0 and self.deletions == 0:
            return "feat"  # pure additions are likely new features
        return "unknown"
```

### Additional Heuristic Signals

| Signal | Interpretation |
|--------|---------------|
| Only `package.json` / `Cargo.lock` / `go.sum` changed | Dependency update (`deps` / `chore`) |
| Only migration files changed | Database migration (`db` / `migration`) |
| Large deletion count, zero additions | Removal / cleanup |
| Single file, <5 line change | Likely a fix or typo correction |
| New file(s) only, no modifications | New feature or generated code |
| `.env.example`, `README.md` changed alongside code | Documentation update with feature |
| Test file added alongside source file | Feature with tests (good practice signal) |

### Detecting Merge Commits, Squash Merges, Cherry-Picks

```python
def classify_merge_type(commit) -> str:
    """Classify merge/special commit types.

    commit: object with .parents (list of parent hashes), .message (str)
    """
    # Standard merge commit: has 2+ parents
    if len(commit.parents) > 1:
        return "merge"

    msg = commit.message.strip()

    # Squash merge: single parent but message pattern
    # GitHub: "Title (#123)\n\n* commit 1\n* commit 2\n..."
    # GitLab: "Title\n\nSee merge request group/project!123"
    if re.search(r"\(#\d+\)$", msg.split("\n")[0]):
        # Could be squash merge or just a PR reference
        if msg.count("\n* ") >= 2:
            return "squash_merge"
    if "See merge request" in msg:
        return "squash_merge"

    # Cherry-pick: message contains cherry-pick reference
    if "(cherry picked from commit" in msg:
        return "cherry_pick"

    # Auto-merge patterns
    if msg.startswith("Merge branch ") or msg.startswith("Merge pull request "):
        return "merge"
    if msg.startswith("Merge remote-tracking branch"):
        return "merge"

    return "regular"
```

---

## 4. Revert Chain Detection

### Parsing Reverts

Git generates revert messages in a standard format:

```python
REVERT_RE = re.compile(
    r'^Revert "(?P<original_subject>.+)"', re.MULTILINE
)

REVERT_HASH_RE = re.compile(
    r"This reverts commit (?P<hash>[0-9a-f]{7,40})", re.MULTILINE
)
```

### Building Revert Chains

```python
from dataclasses import dataclass, field

@dataclass
class RevertChain:
    """A chain of commits and their reverts.

    Example: A -> revert(A) -> revert(revert(A))
    Net effect: A is reapplied (odd number of reverts = reverted, even = restored)
    """
    original_hash: str
    original_subject: str
    reverts: list[str] = field(default_factory=list)  # ordered list of revert hashes

    @property
    def is_effectively_reverted(self) -> bool:
        """True if the original commit's effect is undone (odd # of reverts)."""
        return len(self.reverts) % 2 == 1

    @property
    def depth(self) -> int:
        return len(self.reverts)

def build_revert_chains(commits) -> list[RevertChain]:
    """Build revert chains from a list of commits.

    commits: iterable of objects with .hash, .message attributes
    """
    chains = {}          # original_hash -> RevertChain
    revert_map = {}      # reverted_hash -> reverting_commit_hash
    hash_to_subject = {} # hash -> subject line

    for commit in commits:
        msg = commit.message
        subject = msg.split("\n")[0]
        hash_to_subject[commit.hash] = subject

        # Check if this commit reverts another
        hash_match = REVERT_HASH_RE.search(msg)
        if hash_match:
            reverted_hash = hash_match.group("hash")
            revert_map[reverted_hash] = commit.hash

            # Find the root of the chain
            root = reverted_hash
            while root in revert_map and revert_map[root] != commit.hash:
                # Walk backwards -- but this doesn't work for forward chains
                break

            # Check if the reverted commit is itself a revert in an existing chain
            found_chain = False
            for chain in chains.values():
                if reverted_hash in chain.reverts or reverted_hash == chain.original_hash:
                    chain.reverts.append(commit.hash)
                    found_chain = True
                    break

            if not found_chain:
                # Start a new chain
                chains[reverted_hash] = RevertChain(
                    original_hash=reverted_hash,
                    original_subject=hash_to_subject.get(reverted_hash, "<unknown>"),
                    reverts=[commit.hash],
                )

    return list(chains.values())
```

### Edge Cases

- **Partial reverts**: Not all reverts undo the full commit. The `-n` / `--no-commit` flag stages the revert without committing. These won't appear as revert commits.
- **Manual reverts**: Developer manually undoes changes without using `git revert`. These are nearly impossible to detect without diffing, and even then it's ambiguous.
- **Revert of a merge commit**: `git revert -m 1 <merge-hash>` reverts a merge. The message still follows the `Revert "..."` pattern.
- **Missing original**: The reverted commit hash might not be in the analyzed range. We should flag these as "unresolved reverts."

### What Reverts Tell Us

- **High revert rate** (>5% of commits): Signals instability, insufficient testing, or pressure to ship
- **Quick reverts** (within hours): Usually indicates broken deployments -- CI/CD convention issue
- **Revert chains of depth >1**: Indicates confusion or contentious changes
- **Reverts concentrated on specific files/authors**: Signals risk areas or knowledge gaps

---

## 5. Fix-After Chain Detection

### Definition

A "fix-after chain" is a sequence of commits where the author makes a commit, then follows up with one or more corrective commits. This is different from iterative development -- it signals mistakes that could have been caught before committing.

### Detection Heuristics

```python
from datetime import timedelta

@dataclass
class FixAfterChain:
    original_hash: str
    original_subject: str
    fixups: list[str]  # hashes of follow-up fix commits
    author: str
    time_span: timedelta

    @property
    def chain_length(self) -> int:
        return 1 + len(self.fixups)

# Message signals that indicate a fix-after commit
FIXUP_SIGNALS = re.compile(
    r"^(?:"
    r"fix(?:up)?[\s:]|"        # "fix", "fixup", "fix:", "fix "
    r"oops|"                    # "oops"
    r"typo|"                    # "typo"
    r"forgot|"                  # "forgot to add..."
    r"actually|"                # "actually do the thing"
    r"wip|"                     # "WIP"
    r"address|"                 # "address review comments"
    r"nit|"                     # "nit"
    r"lint|"                    # "lint"
    r"format|"                  # "formatting"
    r"missing|"                 # "missing file"
    r"whoops|"                  # "whoops"
    r"revert.*and|"             # "revert X and redo properly"
    r"amend|"                   # "amend previous"
    r"follow.?up"               # "follow-up", "followup"
    r")",
    re.IGNORECASE
)

# Git's built-in fixup/squash markers
GIT_FIXUP_RE = re.compile(r"^(?:fixup|squash|amend)!\s+")

def detect_fix_after_chains(
    commits,
    max_gap: timedelta = timedelta(hours=4),
    min_file_overlap: float = 0.3,
) -> list[FixAfterChain]:
    """Detect fix-after chains.

    A fix-after is identified when:
    1. Same author
    2. Overlapping files (>30% Jaccard similarity)
    3. Within time window (default 4 hours)
    4. Message signals OR high file overlap with small diff

    commits: sorted by date ascending, with .hash, .message, .author,
             .date, .files_changed attributes
    """
    chains = {}  # original_hash -> FixAfterChain
    commit_list = list(commits)

    for i, commit in enumerate(commit_list):
        msg = commit.message.split("\n")[0]

        # Check for git fixup!/squash! markers (these explicitly reference their target)
        if GIT_FIXUP_RE.match(msg):
            target_subject = GIT_FIXUP_RE.sub("", msg)
            # Find the target commit by subject
            for j in range(i - 1, -1, -1):
                prev_subject = commit_list[j].message.split("\n")[0]
                if prev_subject == target_subject:
                    _add_to_chain(chains, commit_list[j], commit)
                    break
            continue

        # Check message signals
        has_signal = bool(FIXUP_SIGNALS.search(msg))

        # Look backwards for potential parent commits
        for j in range(i - 1, max(i - 20, -1), -1):
            prev = commit_list[j]

            # Must be same author
            if prev.author != commit.author:
                continue

            # Must be within time window
            if commit.date - prev.date > max_gap:
                break  # sorted by date, so no point looking further back

            # Check file overlap
            overlap = _file_overlap(prev.files_changed, commit.files_changed)

            if has_signal and overlap > 0:
                # Message signal + any file overlap = fix-after
                _add_to_chain(chains, prev, commit)
                break
            elif overlap >= min_file_overlap and _is_small_change(commit):
                # High file overlap + small change = likely fix-after
                _add_to_chain(chains, prev, commit)
                break

    return list(chains.values())

def _file_overlap(files_a: list[str], files_b: list[str]) -> float:
    """Jaccard similarity of two file sets."""
    set_a, set_b = set(files_a), set(files_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def _is_small_change(commit) -> bool:
    """Heuristic: small diff is likely a fix."""
    return (commit.additions + commit.deletions) < 20

def _add_to_chain(chains, parent, fixup):
    """Add a fixup commit to its parent's chain."""
    key = parent.hash
    if key not in chains:
        chains[key] = FixAfterChain(
            original_hash=parent.hash,
            original_subject=parent.message.split("\n")[0],
            fixups=[],
            author=parent.author,
            time_span=timedelta(),
        )
    chains[key].fixups.append(fixup.hash)
    chains[key].time_span = fixup.date - parent.date
```

### Distinguishing Iterative Development from Mistakes

| Signal | Iterative Development | Mistake Correction |
|--------|----------------------|-------------------|
| Time gap | Hours to days | Minutes to hours |
| Diff size of follow-up | Substantial (>50 lines) | Small (<20 lines) |
| Message quality | Descriptive ("add error handling") | Terse ("fix", "oops", "wip") |
| File overlap | Partial (extending the work) | High (same files, same areas) |
| Tests included | Often adds tests | Rarely touches tests |
| `fixup!`/`squash!` prefix | Intentional (will be squashed) | Reactive |

### What Fix-After Chains Tell Us

- **High fix-after rate** (>15% of commits are follow-ups): Team might benefit from pre-commit hooks, better linting, or code review before merge
- **Frequent `fixup!`/`squash!` usage**: Team knows about interactive rebase -- sophisticated git workflow
- **Long chains** (3+ fix-ups for one commit): Signals the original commit was significantly incomplete
- **Concentrated by author**: Might indicate a team member needs more support or is working under pressure
- **Concentrated by time of day**: Late-night fix-after commits suggest burnout

---

## 6. Convention Extraction

### Summary Format

The goal is to produce a structured summary of detected conventions that downstream synthesis can use to generate CLAUDE.md / .cursorrules / etc.

```python
@dataclass
class CommitConventionReport:
    """Summary of commit conventions detected in a repository."""

    # Convention identification
    primary_format: str  # "conventional_commits", "ticket_prefix", "freeform", etc.
    format_adherence: float  # 0.0 to 1.0 -- what % of commits follow the primary format

    # Conventional commits specifics (if applicable)
    types_used: dict[str, int]  # type -> count (e.g., {"feat": 234, "fix": 189})
    scopes_used: dict[str, int]  # scope -> count (e.g., {"api": 45, "auth": 32})
    breaking_change_count: int

    # Ticket references
    ticket_format: str | None  # "PROJ-NNN", "#NNN", None
    ticket_adherence: float  # what % of commits include a ticket reference

    # Message style
    imperative_mood_rate: float
    average_subject_length: float
    subject_under_72_chars_rate: float
    has_body_rate: float
    ends_with_period_rate: float
    starts_lowercase_rate: float

    # Health indicators
    revert_rate: float
    fix_after_rate: float
    merge_commit_rate: float
    squash_merge_rate: float

    # Recommendations for AI config generation
    detected_rules: list[str]
    # e.g.:
    # "Use conventional commits format: <type>(<scope>): <description>"
    # "Always reference JIRA ticket in format PROJ-NNN"
    # "Keep commit subject under 72 characters"
    # "Use imperative mood in commit messages"
    # "Include scope from: api, auth, db, ui, cli"

def generate_convention_summary(report: CommitConventionReport) -> list[str]:
    """Generate human-readable convention rules from the report."""
    rules = []

    if report.format_adherence > 0.7:
        if report.primary_format == "conventional_commits":
            types = ", ".join(sorted(report.types_used.keys()))
            rules.append(f"Use conventional commits format. Types used: {types}")
            if report.scopes_used:
                scopes = ", ".join(
                    s for s, _ in sorted(
                        report.scopes_used.items(), key=lambda x: -x[1]
                    )[:10]
                )
                rules.append(f"Use scopes: {scopes}")

    if report.ticket_adherence > 0.8 and report.ticket_format:
        rules.append(f"Reference tickets in format: {report.ticket_format}")

    if report.imperative_mood_rate > 0.8:
        rules.append("Use imperative mood in commit messages (e.g., 'add' not 'added')")

    if report.subject_under_72_chars_rate > 0.9:
        rules.append("Keep commit subject line under 72 characters")

    if report.starts_lowercase_rate > 0.8:
        rules.append("Start commit subject with lowercase letter")
    elif report.starts_lowercase_rate < 0.2:
        rules.append("Start commit subject with uppercase letter")

    if report.ends_with_period_rate < 0.1:
        rules.append("Do not end commit subject with a period")

    if report.has_body_rate > 0.5:
        rules.append("Include a commit body for non-trivial changes")

    return rules
```

### Output for Downstream Synthesis

The convention report should be serialized as JSON or a structured dict for the synthesis stage:

```json
{
  "commit_conventions": {
    "format": "conventional_commits",
    "adherence": 0.87,
    "types": ["feat", "fix", "chore", "refactor", "test", "docs", "ci"],
    "scopes": ["api", "auth", "db", "ui", "cli"],
    "ticket_format": "PROJ-NNN",
    "ticket_adherence": 0.94,
    "style_rules": [
      "Use conventional commits: <type>(<scope>): <description>",
      "Reference JIRA ticket: PROJ-NNN",
      "Keep subject under 72 characters",
      "Use imperative mood",
      "Start with lowercase"
    ],
    "health": {
      "revert_rate": 0.02,
      "fix_after_rate": 0.08,
      "merge_commit_rate": 0.15,
      "squash_merge_rate": 0.05
    }
  }
}
```

---

## 7. Recommendation: LLM vs Pure Regex/Heuristics

### Verdict: Regex/heuristics first, LLM only for freeform semantic classification

**For commit parsing and convention detection (this stage): No LLM needed.**

The structure of commit messages is mechanical enough that regex handles it reliably:

| Task | Approach | Confidence |
|------|----------|------------|
| Parse conventional commits | Regex | 99%+ accuracy |
| Parse ticket references | Regex | 99%+ accuracy |
| Detect non-standard patterns | Regex + frequency analysis | 95%+ accuracy |
| Classify by diff content | File path heuristics | 90%+ accuracy |
| Detect reverts | Regex on message + parent analysis | 99%+ accuracy |
| Detect fix-after chains | Time + author + file overlap heuristics | 85%+ accuracy |
| Detect merge/squash/cherry-pick | Parent count + message patterns | 95%+ accuracy |
| Extract structural conventions | Statistical analysis | 90%+ accuracy |

**Where an LLM adds value (downstream, not this stage):**
- Classifying the *intent* of freeform commits that match no pattern (is "handle edge case" a fix or feat?)
- Summarizing conventions into natural-language rules for the output config file
- Generating the actual CLAUDE.md / .cursorrules content from structured convention data

**Performance argument**: A typical repo has 1,000-100,000 commits. Running regex over all of them takes milliseconds. Running even a small LLM would take minutes and cost money. The regex approach is the only practical option at scale.

**Accuracy argument**: Conventional commits are a formal grammar. Regex is the correct tool for parsing formal grammars. Using an LLM here would be like using a neural network to add two numbers -- technically possible but wasteful and less reliable.

### Recommended Architecture

```
git log data
    |
    v
[Stage 1: Regex Parsing]  -- parse messages, classify by pattern
    |
    v
[Stage 2: Diff Analysis]  -- classify by file changes when message is ambiguous
    |
    v
[Stage 3: Chain Detection]  -- build revert chains, fix-after chains
    |
    v
[Stage 4: Statistical Summary]  -- aggregate into convention report
    |
    v
[Stage 5: Rule Synthesis]  -- LLM converts structured report into config rules
```

Only Stage 5 uses an LLM, and it operates on the structured summary (~1KB of JSON), not on raw commits.

---

## Appendix: Config File Detection

Before analyzing commits, check for existing convention configuration:

| File | Convention System |
|------|------------------|
| `.commitlintrc.*` / `commitlint.config.*` | commitlint |
| `pyproject.toml` → `[tool.commitizen]` | commitizen |
| `.cz.toml` / `.cz.json` / `.cz.yaml` | commitizen |
| `.czrc` | commitizen |
| `.releaserc.*` / `release.config.*` | semantic-release |
| `.changelogrc` | conventional-changelog |
| `.github/workflows/*.yml` containing `semantic-release` | semantic-release in CI |

If any of these exist, we can extract the team's *intended* convention directly. Then we compare actual commit adherence to the intended convention -- the gap between intent and practice is itself a valuable signal.

## Appendix: Complete Conventional Commit Regex

A production-grade regex for the full conventional commit format:

```python
FULL_CONVENTIONAL_COMMIT = re.compile(
    r"^"
    r"(?P<type>[a-z]+)"                        # type (lowercase)
    r"(?:\((?P<scope>[a-zA-Z0-9_./-]+)\))?"    # optional (scope)
    r"(?P<breaking>!)?"                         # optional ! for breaking
    r":\s"                                      # colon + space
    r"(?P<subject>[^\n]+)"                      # subject line
    r"(?:\n\n(?P<body>(?:(?!\n\n[A-Za-z-]+(?::\s|\s#)).)+))?"  # optional body
    r"(?:\n\n(?P<footers>(?:[A-Za-z-]+(?::\s|\s#).+\n?)*))?"  # optional footers
    r"$",
    re.DOTALL
)
```
