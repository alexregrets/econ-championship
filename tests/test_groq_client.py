"""Клиент Groq: backoff на 429/5xx, без сети (httpx.MockTransport не нужен —
подменяем ``client.post`` через monkeypatch на уровне httpx.AsyncClient)."""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from llm.base import LLMError
from llm.groq_client import GroqClient


class _Out(BaseModel):
    ok: bool


def _response(status: int, body: str = "", headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status, text=body, headers=headers or {}, request=request)


def _good_body() -> str:
    return '{"choices":[{"message":{"content":"{\\"ok\\": true}"}}]}'


async def _run(
    monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]
) -> tuple[_Out, list[float]]:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    queue = list(responses)

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        return queue.pop(0)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    client = GroqClient("key", sleep=fake_sleep)
    out = await client.structured_completion("s", "u", _Out, max_retries=4)
    return out, slept


async def test_retries_429_with_backoff_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    out, slept = await _run(
        monkeypatch, [_response(429), _response(429), _response(200, _good_body())]
    )
    assert out.ok is True
    assert slept == [2.0, 4.0]  # экспонента от базы 2 с


async def test_honours_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    out, slept = await _run(
        monkeypatch,
        [_response(429, headers={"retry-after": "7"}), _response(200, _good_body())],
    )
    assert out.ok is True
    assert slept == [7.0]


async def test_bad_json_retries_without_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    out, slept = await _run(monkeypatch, [_response(200, "not json"), _response(200, _good_body())])
    assert out.ok is True
    assert slept == []


async def test_404_is_not_retried_with_delay_and_finally_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(LLMError, match="404"):
        await _run(monkeypatch, [_response(404)] * 4)


def test_retry_delay_caps_at_max() -> None:
    assert GroqClient._retry_delay(_response(503), attempt=10) == 30.0
    assert GroqClient._retry_delay(_response(429, headers={"retry-after": "900"}), 0) == 30.0
    assert GroqClient._retry_delay(_response(400), 0) is None
    assert GroqClient._retry_delay(None, 0) is None
