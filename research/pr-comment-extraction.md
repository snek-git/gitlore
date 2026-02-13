# PR Review Comment Extraction: Research

## 1. Python GitHub API Libraries

### PyGithub
- **Status**: Mature, widely used (6k+ GitHub stars), actively maintained
- **Approach**: Object-oriented wrappers around REST API v3
- **Type safety**: Partial — typed stubs exist but not fully complete
- **Async**: No native async support (open issue since 2020, no resolution)
- **Pagination**: Automatic via `PaginatedList` — lazy-loads pages transparently
- **Ergonomics**: `repo.get_pull(31).get_comments()` — clean, Pythonic
- **Limitations**: REST-only (no GraphQL), cannot query `reviewThreads.isResolved` since that's GraphQL-only

### ghapi (fastai/AnswerDotAI)
- **Status**: Maintained by Answer.AI, auto-generated from OpenAPI spec — always up-to-date
- **Approach**: Thin wrapper, 35kB total. Methods map 1:1 to REST endpoints
- **Type safety**: Dynamically generated, limited IDE support
- **Async**: No native async support
- **Pagination**: Built-in `paged()` helper for automatic pagination
- **Ergonomics**: `api.pulls.list_review_comments(owner, repo, pull_number=123)`
- **Limitations**: REST-only like PyGithub. Lightweight but less discoverable API surface

### GitHubKit
- **Status**: Modern SDK by yanyongyu, actively developed
- **Approach**: All-batteries-included — REST, GraphQL, webhooks, OAuth
- **Type safety**: Fully typed with Pydantic models, excellent IDE support
- **Async**: Native sync AND async support (first-class)
- **Pagination**: Built-in pagination for both REST and GraphQL
- **Extras**: HTTP caching, auto-retry, lazy loading, REST API versioning
- **Ergonomics**: Modern Python patterns, context managers

### Raw GraphQL (requests/httpx/gql)
- **Approach**: Write GraphQL queries directly against `https://api.github.com/graphql`
- **Advantage**: Access to `reviewThreads`, `isResolved`, thread context — data unavailable via REST
- **Disadvantage**: No type safety, manual pagination, more boilerplate
- **Best for**: When you need GraphQL-only features (resolution status, batched queries)

### Recommendation

**GitHubKit** is the best choice for gitlore:
- Native async enables concurrent fetching of multiple PRs
- GraphQL support gives access to `reviewThreads.isResolved` (critical for our use case)
- REST fallback for endpoints where REST is simpler
- Fully typed — better maintainability
- Built-in pagination, caching, retry — less boilerplate

If we want to minimize dependencies, raw `httpx` + hand-written GraphQL queries is a viable alternative. The GraphQL queries we need are well-defined and won't change often.


## 2. GraphQL vs REST for Review Comments

### Why GraphQL is superior for this use case

The REST API exposes review comments via:
- `GET /repos/{owner}/{repo}/pulls/{pull_number}/comments` — inline review comments
- `GET /repos/{owner}/{repo}/pulls/{pull_number}/reviews` — review summaries
- `GET /repos/{owner}/{repo}/pulls/{pull_number}/reviews/{review_id}/comments` — comments per review
- `GET /repos/{owner}/{repo}/issues/{issue_number}/comments` — general PR conversation comments

**Problem**: REST cannot tell you whether a review thread was resolved. The `isResolved` field exists only in GraphQL's `PullRequestReviewThread` type.

### Key GraphQL query for our use case

```graphql
query($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(first: 50, states: [MERGED, CLOSED], after: $cursor) {
      nodes {
        number
        title
        state
        mergedAt
        author { login }
        reviewThreads(first: 100) {
          nodes {
            isResolved
            path
            line
            startLine
            diffSide
            comments(first: 50) {
              nodes {
                body
                author { login }
                createdAt
                diffHunk
                path
                position
                originalPosition
              }
            }
          }
        }
        reviews(first: 20) {
          nodes {
            state
            author { login }
            body
            submittedAt
          }
        }
        commits(last: 1) {
          nodes {
            commit { oid }
          }
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
```

This single query fetches PRs with their review threads, resolution status, comments with diff context, review verdicts, and pagination info — data that would require 4+ REST calls per PR.


## 3. Rate Limiting Strategies

### Limits overview

| Auth method | REST | GraphQL |
|---|---|---|
| Personal Access Token | 5,000 req/hr | 5,000 points/hr |
| GitHub App installation | 5,000 req/hr (+ bonuses) | 5,000 points/hr |
| OAuth app | 5,000 req/hr | 5,000 points/hr |
| GITHUB_TOKEN (Actions) | 1,000 req/hr | 1,000 points/hr |
| Unauthenticated | 60 req/hr | N/A |

### GraphQL point calculation

Formula: `(total_nodes_requested) / 100`, rounded up, minimum 1 point per call.

Example for our query (50 PRs, 100 threads each, 50 comments each):
- PRs: 1
- Threads: 50 * 100 = 5,000
- Comments: 5,000 * 50 = 250,000
- Total: (1 + 5,000 + 250,000) / 100 = **2,551 points**

This is too expensive. We need to reduce nesting:

**Optimized approach**: Fetch PRs in pages of 25, with 30 threads and 10 comments each:
- (1 + 25*30 + 750*10) / 100 = **83 points per page**
- 5,000 / 83 = ~60 pages/hr = **1,500 PRs/hr**

### Conditional requests (REST only)

- Use `If-None-Match` with ETags: 304 responses don't count against rate limit
- Use `If-Modified-Since` for time-based caching
- Store ETags/timestamps in local cache between runs
- **Not applicable to GraphQL** — GraphQL has no conditional request support

### Secondary rate limits

- REST: No more than 100 concurrent requests; no more than 900 points/minute for REST
- GraphQL: No more than 2,000 points/minute; mutations cost 5 points, queries cost 1
- Respect `Retry-After` header on 429 responses
- Use exponential backoff with jitter

### Practical strategies

1. **GraphQL batching**: Fetch 25 PRs per query with nested threads/comments
2. **Two-pass approach**: First pass fetches PR list with basic metadata (cheap), second pass fetches review details for PRs that have reviews
3. **Incremental sync**: Store last-fetched timestamp, only process PRs updated since then
4. **Sleep between pages**: Add 1-2s delay between pagination requests to stay under secondary limits
5. **Progress persistence**: Save cursor position so interrupted runs can resume


## 4. Metadata to Capture Per Review Comment

### Core fields

| Field | Source | Why |
|---|---|---|
| `comment.body` | GraphQL/REST | The actual review feedback text |
| `thread.path` | GraphQL | File the comment is on |
| `thread.line` / `thread.startLine` | GraphQL | Line number(s) in the diff |
| `comment.diffHunk` | GraphQL/REST | Surrounding diff context |
| `thread.isResolved` | GraphQL only | Whether the feedback was addressed |
| `comment.author.login` | GraphQL/REST | Who gave the feedback |
| `review.state` | GraphQL/REST | APPROVED, CHANGES_REQUESTED, COMMENTED |
| `comment.createdAt` | GraphQL/REST | Timestamp for ordering |
| `pr.number` / `pr.title` | GraphQL/REST | PR context |
| `pr.mergedAt` | GraphQL/REST | Whether PR was merged (feedback was accepted) |
| `pr.author.login` | GraphQL/REST | Who wrote the code being reviewed |

### Derived/enriched fields

| Field | How to compute |
|---|---|
| `thread_length` | Count of comments in thread (back-and-forth indicates contention or importance) |
| `reviewer_frequency` | Count how often this reviewer leaves similar comments across PRs |
| `file_extension` | Extract from `path` — useful for language-specific rules |
| `directory` | Extract from `path` — useful for module-specific conventions |
| `time_to_resolve` | `resolved_at - created_at` if available (proxy via subsequent commit timestamps) |
| `comment_led_to_change` | See section 5 below |

### Thread context

Capturing the full thread (not just the first comment) is critical. The first comment may say "this looks wrong" while the reply clarifies "ah, we do it this way because X" — that reply contains the institutional knowledge.


## 5. Detecting "Comment Led to Code Change"

This is the hardest problem and there is no perfect solution. GitHub does not natively link review comments to the commits that address them.

### Approach 1: Thread resolution as proxy (simplest, recommended)

If `thread.isResolved == true` and the PR was merged, the comment was almost certainly addressed. This is the strongest signal available.

**Limitations**: Some teams don't use the resolve feature. Some comments get resolved without code changes (e.g., "good point, but we'll do it in a follow-up").

### Approach 2: Timeline-based heuristic

1. Fetch the PR's commit timeline
2. For each review comment, find commits pushed AFTER the comment's `createdAt`
3. Check if any of those subsequent commits touched the same file (`path`)
4. If the same file was modified after the comment and the thread was resolved, high confidence the comment led to a change

```python
def comment_likely_addressed(comment, subsequent_commits):
    """Heuristic: comment was addressed if a later commit touched the same file."""
    for commit in subsequent_commits:
        if commit.timestamp > comment.created_at:
            if comment.path in commit.files_changed:
                return True
    return False
```

### Approach 3: Diff-level correlation (most precise, most expensive)

1. Get the diff hunk from the review comment
2. Get the diff of subsequent commits in the same PR
3. Check if the subsequent commit modified lines near the commented lines
4. This requires fetching per-commit diffs — expensive in API calls

### Approach 4: LLM-assisted classification (for high-value analysis)

Feed the comment text + subsequent diff to an LLM and ask: "Did this code change address this review comment?" This is the most accurate but requires LLM inference per comment.

### Recommended strategy for gitlore

Use Approach 1 (resolution status) as the primary signal, with Approach 2 (timeline heuristic) as a secondary signal for repos that don't use resolution. This captures 80%+ of cases with minimal API cost.


## 6. GitLab & Bitbucket API Equivalents

### GitLab

**Endpoint**: `GET /projects/:id/merge_requests/:iid/discussions`

**Key differences from GitHub**:
- Discussions are first-class objects (not nested under reviews)
- Each discussion has `notes[]` (equivalent to comments)
- Notes have `resolvable` (boolean) and `resolved` (boolean) fields — available via REST, unlike GitHub
- `resolved_by` and `resolved_at` fields are available
- Diff notes include `position` object with file path, line numbers, diff refs
- Individual notes can be resolved (not just threads)
- Pagination: offset-based, 20 per page default, configurable up to 100

**Python library**: `python-gitlab` — mature, well-maintained, supports discussions/notes API

**Advantage over GitHub**: Resolution status available via REST (no GraphQL needed).

### Bitbucket

**Endpoint**: `GET /2.0/repositories/{workspace}/{repo}/pullrequests/{id}/comments`

**Key differences**:
- Comments returned in flat list with `parent` field for threading
- Inline comments have `inline` object with `path`, `from`, `to` line numbers
- Activity log: `GET /2.0/repositories/{workspace}/{repo}/pullrequests/{id}/activity` includes comments + approvals + updates
- No native resolution/resolved status on comments (as of 2025)
- Pagination: cursor-based with `next` URL in response

**Python library**: `atlassian-python-api` — covers Bitbucket Server and Cloud

**Limitation**: No resolution tracking means we rely entirely on heuristics (timeline-based, Approach 2 above) to detect whether comments were addressed.

### Multi-platform abstraction

All three platforms share the same core concept: inline comments on specific file/line in a diff. A common internal model would look like:

```python
@dataclass
class ReviewComment:
    platform: str           # "github" | "gitlab" | "bitbucket"
    pr_number: int
    file_path: str
    line_number: int | None
    body: str
    author: str
    created_at: datetime
    is_resolved: bool | None  # None for Bitbucket
    diff_context: str
    thread_comments: list[str]
    review_verdict: str | None  # "approved", "changes_requested", etc.
```


## 7. Authentication Patterns for CLI Tools

### Option 1: Personal Access Token (simplest)

User creates a fine-grained PAT at github.com/settings/tokens with `repo` scope (or fine-grained: `pull_requests:read`). CLI reads from environment variable or config file.

```python
# Simple PAT auth
from githubkit import GitHub
gh = GitHub(os.environ["GITHUB_TOKEN"])
```

**Pros**: Simplest to implement, users understand it
**Cons**: User must manually create token, tokens can be overly broad

### Option 2: OAuth Device Flow (best UX for CLI)

The device flow is designed for headless applications. Steps:

1. CLI requests a device code from `POST https://github.com/login/device/code` with a registered OAuth App's client ID
2. CLI displays: "Visit https://github.com/login/device and enter code: ABCD-1234"
3. CLI polls `POST https://github.com/login/oauth/access_token` until user completes auth
4. Token is stored locally (e.g., `~/.config/gitlore/token`)

**Pros**: Best UX — user never copies tokens. Scoped permissions. Refresh possible.
**Cons**: Requires registering an OAuth App on GitHub. More complex implementation.

### Option 3: GitHub App Installation Token

For organization-wide use. A GitHub App is installed on the org, and gitlore authenticates as the app to get an installation token.

**Pros**: Fine-grained permissions, higher rate limits for large orgs
**Cons**: Overkill for individual use, requires App registration and installation

### Recommendation for gitlore

1. **MVP**: Support PAT via `GITHUB_TOKEN` environment variable (zero effort)
2. **V1**: Add OAuth device flow for better UX (users just run `gitlore auth` and follow the prompt)
3. **Future**: GitHub App for enterprise/org deployments with higher rate limits

Also support `gh auth token` — if the user has GitHub CLI installed, we can read their existing token:
```python
import subprocess
token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
```


## 8. Throughput Estimates

### How many PRs per hour?

**GraphQL (optimized query, 25 PRs per page, moderate nesting)**:
- ~83 points per page = ~60 pages/hr = **1,500 PRs/hr**
- Each PR includes up to 30 review threads with 10 comments each

**REST (PyGithub/ghapi)**:
- List PRs: 1 request per 100 PRs
- Per PR: 1 req for comments + 1 req for reviews = 2 requests minimum
- 5,000 / 2 = **2,500 PRs/hr** (but without resolution status)
- With review comments per review: 3+ requests/PR = **~1,600 PRs/hr**

**For a repo with 1,000 PRs**: ~40 minutes with GraphQL, ~25 minutes with REST (but REST misses resolution data).

### Strategies for large repos (5,000+ PRs)

1. **Filter by date**: Only process PRs from the last N months. Recent conventions matter more than old ones.
2. **Filter by state**: Only MERGED PRs — closed-without-merge PRs are noise.
3. **Incremental processing**: Store watermark, only fetch new/updated PRs on re-runs.
4. **Parallel with GitHub App**: Multiple installations can each have independent rate limits.
5. **Two-pass**: First pass gets PR metadata (cheap), second pass fetches review details only for PRs that have `review_comments > 0`.


## 9. Practical Code Patterns

### Pattern 1: GraphQL with httpx (minimal dependencies)

```python
import httpx
from typing import AsyncIterator

QUERY = """
query($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(first: 25, states: [MERGED], after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        title
        mergedAt
        author { login }
        reviewThreads(first: 30) {
          nodes {
            isResolved
            path
            line
            comments(first: 10) {
              nodes {
                body
                author { login }
                createdAt
                diffHunk
              }
            }
          }
        }
        reviews(first: 10) {
          nodes { state author { login } body submittedAt }
        }
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
"""

async def fetch_pr_reviews(token: str, owner: str, repo: str) -> AsyncIterator[dict]:
    """Fetch all merged PRs with review threads via GraphQL."""
    headers = {"Authorization": f"bearer {token}"}
    cursor = None

    async with httpx.AsyncClient() as client:
        while True:
            variables = {"owner": owner, "repo": repo, "cursor": cursor}
            resp = await client.post(
                "https://api.github.com/graphql",
                json={"query": QUERY, "variables": variables},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()["data"]["repository"]["pullRequests"]

            for pr in data["nodes"]:
                yield pr

            if not data["pageInfo"]["hasNextPage"]:
                break
            cursor = data["pageInfo"]["endCursor"]
```

### Pattern 2: GitHubKit with async

```python
from githubkit import GitHub, TokenAuthStrategy

async def fetch_with_githubkit(token: str, owner: str, repo: str):
    gh = GitHub(TokenAuthStrategy(token))

    # REST: list PRs
    prs = await gh.rest.pulls.async_list(owner, repo, state="closed", per_page=100)

    # GraphQL: get review threads with resolution status
    query = """
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $number) {
          reviewThreads(first: 100) {
            nodes {
              isResolved
              path
              line
              comments(first: 50) {
                nodes { body author { login } diffHunk }
              }
            }
          }
        }
      }
    }
    """
    for pr in prs.parsed_data:
        result = await gh.graphql(query, variables={
            "owner": owner, "repo": repo, "number": pr.number
        })
        yield pr, result
```

### Pattern 3: Rate-limit-aware fetching

```python
import asyncio
import time

class RateLimiter:
    """Simple token bucket rate limiter for GitHub API."""

    def __init__(self, points_per_hour: int = 5000):
        self.points_per_hour = points_per_hour
        self.points_used = 0
        self.window_start = time.monotonic()

    async def acquire(self, cost: int = 1):
        elapsed = time.monotonic() - self.window_start
        if elapsed >= 3600:
            self.points_used = 0
            self.window_start = time.monotonic()

        if self.points_used + cost > self.points_per_hour:
            sleep_time = 3600 - elapsed
            await asyncio.sleep(sleep_time)
            self.points_used = 0
            self.window_start = time.monotonic()

        self.points_used += cost
```

### Pattern 4: Extracting actionable review comments

```python
def extract_actionable_comments(pr_data: dict) -> list[dict]:
    """Filter review threads to find actionable institutional knowledge."""
    actionable = []

    for thread in pr_data.get("reviewThreads", {}).get("nodes", []):
        comments = thread.get("comments", {}).get("nodes", [])
        if not comments:
            continue

        first_comment = comments[0]
        body = first_comment.get("body", "")

        # Skip trivial comments
        if len(body) < 20:
            continue
        # Skip pure approval comments
        if body.strip().lower() in ("lgtm", "lgtm!", "looks good", "ship it", "+1"):
            continue

        actionable.append({
            "file_path": thread.get("path"),
            "line": thread.get("line"),
            "body": body,
            "author": first_comment.get("author", {}).get("login"),
            "diff_context": first_comment.get("diffHunk"),
            "is_resolved": thread.get("isResolved"),
            "thread_length": len(comments),
            "full_thread": [c.get("body") for c in comments],
        })

    return actionable
```


## 10. Summary of Recommendations

| Decision | Recommendation | Rationale |
|---|---|---|
| Primary API | GraphQL | `reviewThreads.isResolved` is critical, fewer API calls per PR |
| Python library | GitHubKit or raw httpx+GraphQL | Async, typed, GraphQL support |
| Auth (MVP) | PAT via `GITHUB_TOKEN` env var | Zero implementation cost |
| Auth (v1) | OAuth device flow | Best CLI UX |
| Rate limiting | 25 PRs/page, sleep between pages, cursor persistence | ~1,500 PRs/hr is sufficient |
| Comment-led-to-change | `isResolved` as primary signal | Highest signal-to-noise, lowest cost |
| Multi-platform | Common `ReviewComment` dataclass | Abstract over GitHub/GitLab/Bitbucket differences |
| Large repos | Date filter + incremental sync + two-pass | Keeps under rate limits |
