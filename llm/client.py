"""
Async OpenAI client with rate limiting (semaphore), retries, and token tracking.
"""

import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from config.settings import get_settings


class LLMClient:
    """Async OpenAI client with concurrency limit."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_concurrent: Optional[int] = None,
    ):
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model or settings.openai_agent_model
        self._semaphore = asyncio.Semaphore(
            max_concurrent or settings.max_concurrent_llm_calls
        )
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._call_count = 0
        self._session_call_count = 0
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def session_call_count(self) -> int:
        return self._session_call_count

    @property
    def session_prompt_tokens(self) -> int:
        return self._session_prompt_tokens

    @property
    def session_completion_tokens(self) -> int:
        return self._session_completion_tokens

    def reset_survey_stats(self) -> None:
        """Reset per-survey counters. Call before each survey to track LLM usage."""
        self._session_call_count = 0
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Send chat completion with rate limiting. Returns content of first choice.
        """
        async with self._semaphore:
            return await self._chat_impl(messages, model, temperature, max_tokens)

    async def _chat_impl(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        model = model or self._model
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = response.usage
        if usage:
            pt = usage.prompt_tokens or 0
            ct = usage.completion_tokens or 0
            self._total_prompt_tokens += pt
            self._total_completion_tokens += ct
            self._session_prompt_tokens += pt
            self._session_completion_tokens += ct
        self._call_count += 1
        self._session_call_count += 1
        content = response.choices[0].message.content
        return content or ""

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Single completion from prompt; optional system message."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)


# Module-level default client (lazy)
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
