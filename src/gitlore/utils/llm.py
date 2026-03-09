"""LiteLLM wrapper for completion and embedding calls."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm

log = logging.getLogger("gitlore.llm")

# Suppress litellm's noisy stderr spam
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM Proxy").setLevel(logging.CRITICAL)

litellm.num_retries = 3
litellm.request_timeout = 60

# Also suppress Python-level RuntimeWarnings from litellm's async internals
import warnings
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


async def complete(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
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
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await litellm.acompletion(**kwargs)
    content = response.choices[0].message.content or ""
    return content


def complete_sync(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Synchronous wrapper around async complete."""
    return asyncio.run(complete(model, system, user, temperature, json_mode))


async def embed(model: str, texts: list[str]) -> list[list[float]]:
    """Get embeddings via litellm."""
    response = await litellm.aembedding(model=model, input=texts)
    return [item["embedding"] for item in response.data]
