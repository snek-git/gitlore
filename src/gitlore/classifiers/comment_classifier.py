"""LLM-based classification of PR review comments."""

from __future__ import annotations

import asyncio
import json
import logging

from gitlore.models import ClassifiedComment, CommentCategory, ReviewComment
from gitlore.prompts import load as load_prompt
from gitlore.utils.llm import complete

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {c.value for c in CommentCategory}


def _parse_classification(raw: str) -> tuple[list[CommentCategory], float]:
    """Parse and validate the LLM JSON response into categories and confidence."""
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse classification JSON: %.100s", raw)
        return [CommentCategory.QUESTION], 0.0

    raw_categories = result.get("categories", [])
    if not isinstance(raw_categories, list):
        raw_categories = [raw_categories]

    categories = []
    for cat in raw_categories:
        if isinstance(cat, str) and cat.lower() in VALID_CATEGORIES:
            categories.append(CommentCategory(cat.lower()))

    if not categories:
        categories = [CommentCategory.QUESTION]

    confidence = result.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.5

    return categories, confidence


async def _classify_one(
    comment: ReviewComment,
    model: str,
    semaphore: asyncio.Semaphore,
) -> ClassifiedComment:
    """Classify a single review comment with concurrency control."""
    async with semaphore:
        user_msg = load_prompt("classifier_user").format(
            few_shot=load_prompt("classifier_few_shot"),
            comment=comment.body,
        )
        raw = await complete(
            model=model,
            system=load_prompt("classifier_system"),
            user=user_msg,
            temperature=0.0,
            max_tokens=100,
            json_mode=True,
        )
        categories, confidence = _parse_classification(raw)
        return ClassifiedComment(
            comment=comment,
            categories=categories,
            confidence=confidence,
        )


async def classify_comments(
    comments: list[ReviewComment],
    model: str,
    *,
    max_concurrent: int = 20,
) -> list[ClassifiedComment]:
    """Classify a list of review comments using an LLM.

    Args:
        comments: Review comments to classify.
        model: LiteLLM model identifier (e.g. "openrouter/deepseek/deepseek-chat").
        max_concurrent: Maximum concurrent LLM requests.

    Returns:
        List of ClassifiedComment with assigned categories and confidence scores.
    """
    if not comments:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [_classify_one(c, model, semaphore) for c in comments]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    classified: list[ClassifiedComment] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Classification failed for comment %d: %s", i, result)
            classified.append(
                ClassifiedComment(
                    comment=comments[i],
                    categories=[CommentCategory.QUESTION],
                    confidence=0.0,
                )
            )
        else:
            classified.append(result)

    return classified
