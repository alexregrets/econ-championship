"""Async Groq client producing validated structured output.

Wraps Groq's OpenAI-compatible chat-completions endpoint. The model is asked for
a JSON object; the response is parsed and validated against a caller-supplied
Pydantic model, retrying on transport errors and on malformed/invalid JSON.
"""

from __future__ import annotations

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from llm.base import LLMError

__all__ = ["GroqClient"]

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClient:
    """Minimal async wrapper over Groq chat completions with structured output.

    Parameters
    ----------
    api_key:
        Groq API key.
    model:
        Model id to use. Defaults to ``llama-3.3-70b-versatile``.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        *,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = _BASE_URL

    async def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        """Call the model and return a validated ``response_model`` instance.

        The model is invoked in JSON mode. The system prompt is augmented with the
        expected JSON schema so the model knows the exact shape to emit. On a
        transport error or a parse/validation failure the call is retried up to
        ``max_retries`` times; the last failure is wrapped in :class:`LLMError`.

        Parameters
        ----------
        system_prompt, user_prompt:
            The two halves of the conversation.
        response_model:
            Pydantic model the JSON response must conform to.
        temperature:
            Sampling temperature; default is low (0.1) for deterministic grading.
        max_retries:
            Number of attempts before giving up.

        Raises
        ------
        LLMError
            If no valid response is obtained within ``max_retries`` attempts.
        """
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        full_system = (
            f"{system_prompt}\n\n"
            "Respond with a single JSON object that strictly conforms to this "
            f"JSON Schema:\n{schema}\n"
            "Do not include any text outside the JSON object."
        )
        payload = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for _attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    return response_model.model_validate_json(content)
                except (
                    httpx.HTTPError,
                    KeyError,
                    json.JSONDecodeError,
                    ValidationError,
                ) as exc:
                    last_error = exc
                    continue

        raise LLMError(
            f"failed to obtain valid {response_model.__name__} after "
            f"{max_retries} attempts: {last_error}"
        ) from last_error
