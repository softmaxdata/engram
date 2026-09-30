"""Abstract LLM adapter interface and LiteLLM-based universal implementation."""

from __future__ import annotations

import abc
import json
import logging
from typing import Any

from engram.core.exceptions import LLMAdapterError

logger = logging.getLogger(__name__)


class LLMAdapter(abc.ABC):
    """Abstract interface for LLM calls used by ingestion and materialization."""

    @abc.abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str | None = None,
        model: str | None = None,
    ) -> str:
        """Generate a completion from the LLM.

        `model` overrides the adapter's default for this call only. Used by the
        validity gate and re-extraction to route to a different model than the
        canonical Reflector.
        """

    @abc.abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""


class LiteLLMAdapter(LLMAdapter):
    """Universal LLM adapter using LiteLLM to support any provider."""

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str | None = None,
        model: str | None = None,
    ) -> str:
        import litellm

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM completion failed: %s", exc)
            raise LLMAdapterError(f"Completion failed: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.embedding_model,
            "input": [text],
        }
        if self.embedding_api_key:
            kwargs["api_key"] = self.embedding_api_key

        try:
            response = await litellm.aembedding(**kwargs)
            return response.data[0]["embedding"]
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise LLMAdapterError(f"Embedding failed: {exc}") from exc
