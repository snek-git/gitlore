"""Tests for GitHub PR comment extraction."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest
import respx

from gitlore.extractors.github_comments import (
    GRAPHQL_URL,
    _extract_comments_from_pr,
    _is_trivial,
    fetch_review_comments,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_pr_node(
    number: int = 1,
    threads: list[dict] | None = None,
    reviews: list[dict] | None = None,
) -> dict:
    """Build a minimal PR GraphQL node for testing."""
    return {
        "number": number,
        "title": f"PR #{number}",
        "mergedAt": "2026-01-15T10:00:00Z",
        "author": {"login": "alice"},
        "reviewThreads": {"nodes": threads or []},
        "reviews": {"nodes": reviews or []},
    }


def _make_thread(
    body: str = "This function should handle the null case properly",
    author: str = "bob",
    path: str = "src/auth/login.ts",
    line: int = 42,
    is_resolved: bool = True,
    extra_comments: list[str] | None = None,
) -> dict:
    """Build a minimal review thread node."""
    comments = [
        {
            "body": body,
            "author": {"login": author},
            "createdAt": "2026-01-15T11:00:00Z",
            "diffHunk": "@@ -10,5 +10,7 @@\n some code",
        }
    ]
    for extra in extra_comments or []:
        comments.append(
            {
                "body": extra,
                "author": {"login": "carol"},
                "createdAt": "2026-01-15T12:00:00Z",
                "diffHunk": None,
            }
        )
    return {
        "isResolved": is_resolved,
        "path": path,
        "line": line,
        "comments": {"nodes": comments},
    }


def _make_review(
    body: str = "Please address the null handling issues before merging",
    author: str = "bob",
    state: str = "CHANGES_REQUESTED",
) -> dict:
    """Build a minimal review node."""
    return {
        "state": state,
        "author": {"login": author},
        "body": body,
        "submittedAt": "2026-01-15T13:00:00Z",
    }


def _graphql_response(
    pr_nodes: list[dict],
    has_next_page: bool = False,
    end_cursor: str | None = None,
) -> dict:
    """Wrap PR nodes in a full GraphQL response structure."""
    return {
        "data": {
            "repository": {
                "pullRequests": {
                    "nodes": pr_nodes,
                    "pageInfo": {
                        "hasNextPage": has_next_page,
                        "endCursor": end_cursor,
                    },
                }
            }
        }
    }


# ── Unit tests for filtering ─────────────────────────────────────────────────


class TestIsTrivial:
    def test_short_comment_is_trivial(self):
        assert _is_trivial("ok") is True
        assert _is_trivial("fix this") is True
        assert _is_trivial("nit") is True

    def test_lgtm_variants_are_trivial(self):
        assert _is_trivial("LGTM") is True
        assert _is_trivial("lgtm!") is True
        assert _is_trivial("  LGTM  ") is True

    def test_praise_variants_are_trivial(self):
        assert _is_trivial("looks good") is True
        assert _is_trivial("Looks good to me!") is True
        assert _is_trivial("+1") is True
        assert _is_trivial("ship it") is True

    def test_substantive_comment_is_not_trivial(self):
        assert _is_trivial("This function should handle the null case properly") is False
        assert _is_trivial("Consider using a map instead of a list for O(1) lookups") is False

    def test_empty_is_trivial(self):
        assert _is_trivial("") is True
        assert _is_trivial("   ") is True


# ── Unit tests for PR node extraction ─────────────────────────────────────────


class TestExtractCommentsFromPR:
    def test_extracts_thread_comments(self):
        pr = _make_pr_node(
            number=42,
            threads=[_make_thread(body="This needs proper error handling for edge cases")],
        )
        comments = _extract_comments_from_pr(pr)
        assert len(comments) == 1
        assert comments[0].pr_number == 42
        assert comments[0].file_path == "src/auth/login.ts"
        assert comments[0].line == 42
        assert comments[0].author == "bob"
        assert comments[0].is_resolved is True
        assert "error handling" in comments[0].body

    def test_extracts_review_body_comments(self):
        pr = _make_pr_node(
            number=10,
            reviews=[_make_review(body="Please address the null handling issues before merging")],
        )
        comments = _extract_comments_from_pr(pr)
        assert len(comments) == 1
        assert comments[0].review_state == "CHANGES_REQUESTED"
        assert comments[0].file_path is None

    def test_filters_trivial_thread_comments(self):
        pr = _make_pr_node(
            threads=[
                _make_thread(body="LGTM"),
                _make_thread(body="+1"),
                _make_thread(body="This needs proper error handling for edge cases"),
            ],
        )
        comments = _extract_comments_from_pr(pr)
        assert len(comments) == 1
        assert "error handling" in comments[0].body

    def test_captures_thread_replies(self):
        pr = _make_pr_node(
            threads=[
                _make_thread(
                    body="This should use dependency injection instead of direct instantiation",
                    extra_comments=["Good point, I'll refactor this"],
                ),
            ],
        )
        comments = _extract_comments_from_pr(pr)
        assert len(comments) == 1
        assert len(comments[0].thread_comments) == 1
        assert "refactor" in comments[0].thread_comments[0]

    def test_handles_null_author(self):
        pr = _make_pr_node(
            threads=[
                {
                    "isResolved": False,
                    "path": "src/main.ts",
                    "line": 1,
                    "comments": {
                        "nodes": [
                            {
                                "body": "This could cause memory leaks under heavy load",
                                "author": None,
                                "createdAt": "2026-01-15T11:00:00Z",
                                "diffHunk": None,
                            }
                        ]
                    },
                }
            ],
        )
        comments = _extract_comments_from_pr(pr)
        assert len(comments) == 1
        assert comments[0].author == "ghost"

    def test_empty_threads_and_reviews(self):
        pr = _make_pr_node(threads=[], reviews=[])
        comments = _extract_comments_from_pr(pr)
        assert comments == []

    def test_filters_trivial_review_body(self):
        pr = _make_pr_node(reviews=[_make_review(body="LGTM")])
        comments = _extract_comments_from_pr(pr)
        assert comments == []


# ── Integration tests with mocked httpx (respx) ──────────────────────────────


class TestFetchReviewComments:
    @respx.mock
    async def test_single_page(self):
        pr = _make_pr_node(
            number=1,
            threads=[_make_thread(body="This needs proper error handling for all code paths")],
        )
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=_graphql_response([pr]))
        )

        comments = await fetch_review_comments("fake-token", "owner", "repo")
        assert len(comments) == 1
        assert comments[0].pr_number == 1

    @respx.mock
    async def test_pagination(self):
        pr1 = _make_pr_node(
            number=1,
            threads=[_make_thread(body="Use dependency injection rather than direct instantiation")],
        )
        pr2 = _make_pr_node(
            number=2,
            threads=[_make_thread(body="This should be moved to the service layer instead")],
        )

        # First page has next
        route = respx.post(GRAPHQL_URL)
        route.side_effect = [
            httpx.Response(
                200,
                json=_graphql_response([pr1], has_next_page=True, end_cursor="cursor1"),
            ),
            httpx.Response(200, json=_graphql_response([pr2])),
        ]

        comments = await fetch_review_comments(
            "fake-token", "owner", "repo", page_delay=0.0
        )
        assert len(comments) == 2
        assert {c.pr_number for c in comments} == {1, 2}

    @respx.mock
    async def test_rate_limit_retry(self):
        pr = _make_pr_node(
            number=1,
            threads=[_make_thread(body="Add validation for user input before processing")],
        )

        route = respx.post(GRAPHQL_URL)
        route.side_effect = [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json=_graphql_response([pr])),
        ]

        comments = await fetch_review_comments(
            "fake-token", "owner", "repo", page_delay=0.0
        )
        assert len(comments) == 1

    @respx.mock
    async def test_graphql_error_raises(self):
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={"errors": [{"message": "Not Found"}]},
            )
        )

        with pytest.raises(RuntimeError, match="GraphQL error"):
            await fetch_review_comments("fake-token", "owner", "repo")

    @respx.mock
    async def test_empty_repo(self):
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=_graphql_response([]))
        )

        comments = await fetch_review_comments("fake-token", "owner", "repo")
        assert comments == []

    @respx.mock
    async def test_multiple_threads_per_pr(self):
        pr = _make_pr_node(
            number=5,
            threads=[
                _make_thread(body="This function lacks proper error handling for edge cases"),
                _make_thread(body="Consider using a map instead of a list here for performance"),
                _make_thread(body="LGTM"),  # should be filtered
            ],
        )
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=_graphql_response([pr]))
        )

        comments = await fetch_review_comments("fake-token", "owner", "repo")
        assert len(comments) == 2

    @respx.mock
    async def test_auth_header_sent(self):
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=_graphql_response([]))
        )

        await fetch_review_comments("my-secret-token", "owner", "repo")

        request = respx.calls[0].request
        assert request.headers["Authorization"] == "bearer my-secret-token"

    @respx.mock
    async def test_datetime_parsing(self):
        pr = _make_pr_node(
            number=1,
            threads=[_make_thread(body="This should validate the input before processing data")],
        )
        respx.post(GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json=_graphql_response([pr]))
        )

        comments = await fetch_review_comments("fake-token", "owner", "repo")
        assert comments[0].created_at == datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
