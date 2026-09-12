"""Async Groq client producing validated structured output.

Wraps Groq's OpenAI-compatible chat-completions endpoint. The model is asked for
a JSON object; the response is parsed and validated against a caller-supplied
Pydantic model, retrying on transport errors and on malformed/invalid JSON.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from llm.base import LLMError

__all__ = ["GroqClient"]

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = "openai/gpt-oss-120b"
_BASE_URL = "https://api.groq.com/openai/v1"
# Пауза перед повтором после 429/5xx, секунд, если сервер не назвал свою
# (``Retry-After``). Удваивается с каждой попыткой. Ключ Groq общий с ботами
# на VPS, лимит в минуту делится между ними — на 12.09 семь подряд запросов
# грейдинга упёрлись в 429 на второй команде.
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 30.0


class GroqClient:
    """Minimal async wrapper over Groq chat completions with structured output.

    Parameters
    ----------
    api_key:
        Groq API key.
    model:
        Model id to use. Defaults to ``openai/gpt-oss-120b`` —
        ``llama-3.3-70b-versatile`` снята с Groq (404 на 12.09.2026); та же
        модель, что у ``uni_mail_bot`` (решение 21.08.2026).
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        *,
        timeout: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = _BASE_URL
        self._sleep = sleep  # подменяется в тестах, чтобы не ждать по-настоящему

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float | None:
        """Сколько ждать перед повтором; ``None`` — повторять сразу.

        429 и 5xx — временные, ждём (``Retry-After`` сервера, иначе
        экспонента). Прочие HTTP-ошибки и невалидный JSON повторяются без
        паузы: там дело не в нагрузке.
        """
        if response is None or response.status_code not in (429, 500, 502, 503, 504):
            return None
        header = response.headers.get("retry-after")
        if header:
            try:
                return min(float(header), _BACKOFF_MAX_SECONDS)
            except ValueError:
                pass
        return float(min(_BACKOFF_BASE_SECONDS * 2.0**attempt, _BACKOFF_MAX_SECONDS))

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
            for attempt in range(max_retries):
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
                    failed = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
                    delay = self._retry_delay(failed, attempt)
                    if delay is not None and attempt + 1 < max_retries:
                        await self._sleep(delay)
                    continue

        raise LLMError(
            f"failed to obtain valid {response_model.__name__} after "
            f"{max_retries} attempts: {last_error}"
        ) from last_error
