"""Fetch PR review comments from GitHub via GraphQL."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone

import httpx

from gitlore.models import ReviewComment

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

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
          nodes {
            state
            author { login }
            body
            submittedAt
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
"""

# Comments matching these patterns are filtered out as trivial praise.
_PRAISE_PATTERNS = re.compile(
    r"^\s*("
    r"lgtm!?"
    r"|looks?\s+good(\s+to\s+me)?!?"
    r"|\+1"
    r"|:?\+1:?"
    r"|ship\s+it!?"
    r"|approved?!?"
    r"|nice!?"
    r"|great!?"
    r")\s*$",
    re.IGNORECASE,
)

MIN_COMMENT_LENGTH = 20


def _is_trivial(body: str) -> bool:
    """Return True if a comment is too short or pure praise."""
    stripped = body.strip()
    if len(stripped) < MIN_COMMENT_LENGTH:
        return True
    if _PRAISE_PATTERNS.match(stripped):
        return True
    return False


def _parse_datetime(value: str | None) -> datetime:
    """Parse an ISO 8601 datetime string from GitHub."""
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    # GitHub returns ISO 8601: "2025-01-15T10:30:00Z"
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _extract_comments_from_pr(pr_node: dict) -> list[ReviewComment]:
    """Extract ReviewComment objects from a single PR GraphQL node."""
    pr_number = pr_node["number"]
    comments: list[ReviewComment] = []

    # Extract from review threads (inline comments)
    for thread in pr_node.get("reviewThreads", {}).get("nodes", []) or []:
        thread_comments = thread.get("comments", {}).get("nodes", []) or []
        if not thread_comments:
            continue

        first = thread_comments[0]
        body = first.get("body", "")

        if _is_trivial(body):
            continue

        author_node = first.get("author") or {}
        comments.append(
            ReviewComment(
                pr_number=pr_number,
                file_path=thread.get("path"),
                line=thread.get("line"),
                body=body,
                author=author_node.get("login", "ghost"),
                created_at=_parse_datetime(first.get("createdAt")),
                is_resolved=thread.get("isResolved"),
                diff_context=first.get("diffHunk"),
                thread_comments=[c.get("body", "") for c in thread_comments[1:]],
            )
        )

    # Extract from top-level reviews (review body comments, e.g. CHANGES_REQUESTED summary)
    for review in pr_node.get("reviews", {}).get("nodes", []) or []:
        body = review.get("body", "")
        if not body or _is_trivial(body):
            continue

        author_node = review.get("author") or {}
        comments.append(
            ReviewComment(
                pr_number=pr_number,
                file_path=None,
                line=None,
                body=body,
                author=author_node.get("login", "ghost"),
                created_at=_parse_datetime(review.get("submittedAt")),
                is_resolved=None,
                diff_context=None,
                thread_comments=[],
                review_state=review.get("state"),
            )
        )

    return comments


async def fetch_review_comments(
    token: str,
    owner: str,
    repo: str,
    *,
    max_pages: int = 40,
    page_delay: float = 1.0,
) -> list[ReviewComment]:
    """Fetch merged PR review comments via GitHub GraphQL API.

    Args:
        token: GitHub bearer token.
        owner: Repository owner (user or org).
        repo: Repository name.
        max_pages: Maximum number of pages to fetch (25 PRs/page).
        page_delay: Seconds to sleep between pages to respect secondary rate limits.

    Returns:
        List of ReviewComment objects, filtered of trivial/praise comments.
    """
    headers = {
        "Authorization": f"bearer {token}",
        "Content-Type": "application/json",
    }
    all_comments: list[ReviewComment] = []
    cursor: str | None = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for page in range(max_pages):
            variables: dict[str, str | None] = {
                "owner": owner,
                "repo": repo,
                "cursor": cursor,
            }

            response = await client.post(
                GRAPHQL_URL,
                json={"query": QUERY, "variables": variables},
                headers=headers,
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "60"))
                logger.warning("Rate limited, sleeping %ds", retry_after)
                await asyncio.sleep(retry_after)
                # Retry the same page
                response = await client.post(
                    GRAPHQL_URL,
                    json={"query": QUERY, "variables": variables},
                    headers=headers,
                )

            response.raise_for_status()
            data = response.json()

            # Check for GraphQL errors
            if "errors" in data:
                error_msg = data["errors"][0].get("message", "Unknown GraphQL error")
                raise RuntimeError(f"GraphQL error: {error_msg}")

            repo_data = data["data"]["repository"]["pullRequests"]
            pr_nodes = repo_data["nodes"]

            for pr_node in pr_nodes:
                all_comments.extend(_extract_comments_from_pr(pr_node))

            page_info = repo_data["pageInfo"]
            if not page_info["hasNextPage"]:
                break

            cursor = page_info["endCursor"]

            # Sleep between pages to respect secondary rate limits
            if page < max_pages - 1 and page_info["hasNextPage"]:
                await asyncio.sleep(page_delay)

    logger.info("Fetched %d review comments from %s/%s", len(all_comments), owner, repo)
    return all_comments
