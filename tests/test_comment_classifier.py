"""Tests for LLM-based comment classification."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from gitlore.classifiers.comment_classifier import (
    _parse_classification,
    classify_comments,
)
from gitlore.models import ClassifiedComment, CommentCategory, ReviewComment


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_comment(body: str, pr_number: int = 1) -> ReviewComment:
    return ReviewComment(
        pr_number=pr_number,
        file_path="src/main.ts",
        line=10,
        body=body,
        author="reviewer",
        created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
    )


# ── Unit tests for response parsing ──────────────────────────────────────────


class TestParseClassification:
    def test_single_category(self):
        cats, conf = _parse_classification('{"categories": ["bug"], "confidence": 0.95}')
        assert cats == [CommentCategory.BUG]
        assert conf == pytest.approx(0.95)

    def test_multiple_categories(self):
        cats, conf = _parse_classification(
            '{"categories": ["security", "bug"], "confidence": 0.9}'
        )
        assert CommentCategory.SECURITY in cats
        assert CommentCategory.BUG in cats
        assert len(cats) == 2

    def test_all_categories(self):
        for cat in CommentCategory:
            cats, _ = _parse_classification(
                f'{{"categories": ["{cat.value}"], "confidence": 0.8}}'
            )
            assert cats == [cat]

    def test_invalid_category_filtered(self):
        cats, _ = _parse_classification(
            '{"categories": ["bug", "invalid_cat", "nitpick"], "confidence": 0.8}'
        )
        assert cats == [CommentCategory.BUG, CommentCategory.NITPICK]

    def test_all_invalid_defaults_to_question(self):
        cats, _ = _parse_classification(
            '{"categories": ["not_a_category"], "confidence": 0.8}'
        )
        assert cats == [CommentCategory.QUESTION]

    def test_empty_categories_defaults_to_question(self):
        cats, _ = _parse_classification('{"categories": [], "confidence": 0.5}')
        assert cats == [CommentCategory.QUESTION]

    def test_missing_categories_defaults_to_question(self):
        cats, _ = _parse_classification('{"confidence": 0.5}')
        assert cats == [CommentCategory.QUESTION]

    def test_confidence_clamped_to_range(self):
        _, conf = _parse_classification('{"categories": ["bug"], "confidence": 1.5}')
        assert conf == 1.0
        _, conf = _parse_classification('{"categories": ["bug"], "confidence": -0.3}')
        assert conf == 0.0

    def test_missing_confidence_defaults(self):
        _, conf = _parse_classification('{"categories": ["bug"]}')
        assert conf == 0.5

    def test_invalid_json_returns_defaults(self):
        cats, conf = _parse_classification("not json at all")
        assert cats == [CommentCategory.QUESTION]
        assert conf == 0.0

    def test_json_with_extra_whitespace(self):
        cats, conf = _parse_classification(
            '  { "categories" : [ "performance" ] , "confidence" : 0.7 }  '
        )
        assert cats == [CommentCategory.PERFORMANCE]
        assert conf == pytest.approx(0.7)

    def test_case_insensitive_categories(self):
        cats, _ = _parse_classification('{"categories": ["BUG", "Security"], "confidence": 0.8}')
        assert CommentCategory.BUG in cats
        assert CommentCategory.SECURITY in cats


# ── Integration tests with mocked LLM ────────────────────────────────────────


class TestClassifyComments:
    @patch("gitlore.classifiers.comment_classifier.complete", new_callable=AsyncMock)
    async def test_classify_single_comment(self, mock_complete: AsyncMock):
        mock_complete.return_value = '{"categories": ["bug"], "confidence": 0.95}'

        comments = [_make_comment("This will NPE if user is null")]
        result = await classify_comments(comments, "test-model")

        assert len(result) == 1
        assert result[0].categories == [CommentCategory.BUG]
        assert result[0].confidence == pytest.approx(0.95)
        assert result[0].comment is comments[0]

    @patch("gitlore.classifiers.comment_classifier.complete", new_callable=AsyncMock)
    async def test_classify_multiple_comments(self, mock_complete: AsyncMock):
        mock_complete.side_effect = [
            '{"categories": ["bug"], "confidence": 0.9}',
            '{"categories": ["architecture"], "confidence": 0.85}',
            '{"categories": ["praise"], "confidence": 0.98}',
        ]

        comments = [
            _make_comment("This will crash on null input"),
            _make_comment("Move this to the service layer"),
            _make_comment("Nice refactor, looks clean!"),
        ]
        result = await classify_comments(comments, "test-model")

        assert len(result) == 3
        assert result[0].categories == [CommentCategory.BUG]
        assert result[1].categories == [CommentCategory.ARCHITECTURE]
        assert result[2].categories == [CommentCategory.PRAISE]

    @patch("gitlore.classifiers.comment_classifier.complete", new_callable=AsyncMock)
    async def test_classify_empty_list(self, mock_complete: AsyncMock):
        result = await classify_comments([], "test-model")
        assert result == []
        mock_complete.assert_not_called()

    @patch("gitlore.classifiers.comment_classifier.complete", new_callable=AsyncMock)
    async def test_llm_failure_returns_default(self, mock_complete: AsyncMock):
        mock_complete.side_effect = RuntimeError("API error")

        comments = [_make_comment("Some comment")]
        result = await classify_comments(comments, "test-model")

        assert len(result) == 1
        assert result[0].categories == [CommentCategory.QUESTION]
        assert result[0].confidence == 0.0

    @patch("gitlore.classifiers.comment_classifier.complete", new_callable=AsyncMock)
    async def test_partial_failure(self, mock_complete: AsyncMock):
        mock_complete.side_effect = [
            '{"categories": ["bug"], "confidence": 0.9}',
            RuntimeError("API timeout"),
            '{"categories": ["nitpick"], "confidence": 0.85}',
        ]

        comments = [
            _make_comment("Bug here"),
            _make_comment("Timeout here"),
            _make_comment("Nitpick here"),
        ]
        result = await classify_comments(comments, "test-model")

        assert len(result) == 3
        assert result[0].categories == [CommentCategory.BUG]
        assert result[1].categories == [CommentCategory.QUESTION]  # fallback
        assert result[2].categories == [CommentCategory.NITPICK]

    @patch("gitlore.classifiers.comment_classifier.complete", new_callable=AsyncMock)
    async def test_multi_label_classification(self, mock_complete: AsyncMock):
        mock_complete.return_value = '{"categories": ["security", "bug"], "confidence": 0.95}'

        comments = [_make_comment("SQL injection vulnerability that will crash on null")]
        result = await classify_comments(comments, "test-model")

        assert len(result) == 1
        assert CommentCategory.SECURITY in result[0].categories
        assert CommentCategory.BUG in result[0].categories

    @patch("gitlore.classifiers.comment_classifier.complete", new_callable=AsyncMock)
    async def test_llm_called_with_correct_params(self, mock_complete: AsyncMock):
        mock_complete.return_value = '{"categories": ["bug"], "confidence": 0.9}'

        comments = [_make_comment("This will NPE")]
        await classify_comments(comments, "my-model")

        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args
        assert call_kwargs.kwargs["model"] == "my-model"
        assert call_kwargs.kwargs["temperature"] == 0.0
        assert call_kwargs.kwargs["max_tokens"] == 100
        assert call_kwargs.kwargs["json_mode"] is True
        assert "This will NPE" in call_kwargs.kwargs["user"]

    @patch("gitlore.classifiers.comment_classifier.complete", new_callable=AsyncMock)
    async def test_concurrency_limit(self, mock_complete: AsyncMock):
        """Verify semaphore limits concurrent calls."""
        call_count = 0
        max_concurrent = 0
        current_concurrent = 0

        async def tracked_complete(**kwargs):
            nonlocal call_count, max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            call_count += 1
            # Small delay to allow overlap
            import asyncio
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            return '{"categories": ["nitpick"], "confidence": 0.8}'

        mock_complete.side_effect = tracked_complete

        comments = [_make_comment(f"Comment {i} with enough length to pass filter") for i in range(10)]
        result = await classify_comments(comments, "test-model", max_concurrent=3)

        assert len(result) == 10
        assert call_count == 10
        assert max_concurrent <= 3
