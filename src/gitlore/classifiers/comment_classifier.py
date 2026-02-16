"""LLM-based classification of PR review comments."""

from __future__ import annotations

import asyncio
import logging
import re

from gitlore.models import ClassifiedComment, CommentCategory, ReviewComment
from gitlore.prompts import load as load_prompt
from gitlore.utils.llm import complete

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {c.value for c in CommentCategory}

_CATEGORY_RE = re.compile(r"<category>\s*(\w+)\s*</category>")
_CONFIDENCE_RE = re.compile(r"<confidence>\s*(-?[\d.]+)\s*</confidence>")


def _parse_classification(raw: str) -> tuple[list[CommentCategory], float]:
    """Parse the LLM XML response into categories and confidence.

    Extracts <category> and <confidence> tags from the response. Resilient to
    truncation — if the response is cut off after some <category> tags but before
    <confidence>, the categories are still captured.
    """
    text = raw.strip()
    if not text:
        logger.warning("Empty classification response")
        return [CommentCategory.QUESTION], 0.0

    raw_categories = _CATEGORY_RE.findall(text)
    categories = []
    for cat in raw_categories:
        if cat.lower() in VALID_CATEGORIES:
            categories.append(CommentCategory(cat.lower()))

    if not categories:
        logger.warning("No categories found in classification: %.100s", text)
        return [CommentCategory.QUESTION], 0.0

    conf_match = _CONFIDENCE_RE.search(text)
    if conf_match:
        try:
            confidence = max(0.0, min(1.0, float(conf_match.group(1))))
        except ValueError:
            confidence = 0.5
    else:
        confidence = 0.5

    return categories, confidence


async def _classify_one(
    comment: ReviewComment,
    model: str,
    semaphore: asyncio.Semaphore,
) -> ClassifiedComment:
    """Classify a single review comment with concurrency control."""
    async with semaphore:
        user_msg = load_prompt("classifier_user").format(comment=comment.body)
        raw = await complete(
            model=model,
            system=load_prompt("classifier_system"),
            user=user_msg,
            temperature=0.0,
            max_tokens=100,
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
