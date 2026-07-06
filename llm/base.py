"""Abstractions for structured LLM completions.

Defining a narrow :class:`StructuredLLM` protocol (rather than importing the
concrete Groq client everywhere) keeps the grading and narrative logic testable:
tests pass a fake implementation, production passes
:class:`~llm.groq_client.GroqClient`.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

__all__ = ["StructuredLLM", "LLMError"]

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM call fails or its output can't be parsed/validated."""


class StructuredLLM(Protocol):
    """Anything that can turn a prompt pair into a validated Pydantic model."""

    async def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        """Return an instance of ``response_model`` parsed from the LLM response.

        Implementations must raise :class:`LLMError` if a valid instance cannot
        be produced within ``max_retries`` attempts.
        """
        ...
