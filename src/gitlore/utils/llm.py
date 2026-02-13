"""LiteLLM wrapper for completion and embedding calls."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm

log = logging.getLogger("gitlore.llm")

# Defaults
litellm.num_retries = 3
litellm.request_timeout = 60


async def complete(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """Async LLM completion, returns the content string."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    log.debug(
        "LLM CALL → model=%s temp=%.1f max_tokens=%d json_mode=%s\n"
        "── SYSTEM ──\n%s\n"
        "── USER ──\n%s",
        model, temperature, max_tokens, json_mode, system, user,
    )

    response = await litellm.acompletion(**kwargs)
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    log.debug(
        "LLM RESPONSE ← model=%s tokens=%s\n"
        "── CONTENT ──\n%s",
        model,
        f"in={usage.prompt_tokens} out={usage.completion_tokens}" if usage else "n/a",
        content,
    )

    return content


def complete_sync(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """Synchronous wrapper around async complete."""
    return asyncio.run(
        complete(model, system, user, temperature, max_tokens, json_mode)
    )


async def embed(model: str, texts: list[str]) -> list[list[float]]:
    """Get embeddings via litellm (for API-based models)."""
    log.debug("EMBED CALL → model=%s texts=%d", model, len(texts))
    response = await litellm.aembedding(model=model, input=texts)
    usage = getattr(response, "usage", None)
    log.debug(
        "EMBED RESPONSE ← model=%s tokens=%s vectors=%d",
        model,
        f"in={usage.prompt_tokens}" if usage else "n/a",
        len(response.data),
    )
    return [item["embedding"] for item in response.data]
