"""Unit tests for :mod:`translation_dubbing_skill.providers.translation.llm`.

Covers the happy path plus the four error-mapping paths exercised by the
adaptive scheduler:

    * HTTP 429                 → :class:`RateLimitError`
    * HTTP 413 / context overflow → :class:`PayloadTooLargeError`
    * ``httpx.TimeoutException`` → :class:`TransientError`
    * Malformed JSON response    → :class:`TransientError`

Uses :class:`httpx.MockTransport` to intercept HTTP calls so the tests
run fully offline.
"""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from translation_dubbing_skill.models import ProviderConfig, SubtitleEntry
from translation_dubbing_skill.providers.registry import default_registry
from translation_dubbing_skill.providers.translation.llm import (
    LLMTranslationProvider,
)
from translation_dubbing_skill.scheduler.signals import (
    PayloadTooLargeError,
    RateLimitError,
    TransientError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    handler: Callable[[httpx.Request], httpx.Response],
) -> LLMTranslationProvider:
    """Create an initialized ``LLMTranslationProvider`` with a mock transport."""
    provider = LLMTranslationProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://api.llm.test/v1/chat/completions",
            credential="secret-token",
            extra={"model_name": "gpt-test"},
        )
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _entries(*texts: str) -> list[SubtitleEntry]:
    return [
        SubtitleEntry(index=i + 1, start_ms=i * 1000, end_ms=(i + 1) * 1000, text=t)
        for i, t in enumerate(texts)
    ]


def _chat_response(items: list[dict]) -> httpx.Response:
    """Build a chat-completions-shaped 200 response carrying ``items``."""
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": json.dumps(items, ensure_ascii=False)}}
            ]
        },
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_llm_provider_is_registered_on_import() -> None:
    """The ``@register`` decorator registers the provider at import time."""
    assert "llm" in default_registry.list("translation")


def test_llm_provider_class_metadata() -> None:
    assert LLMTranslationProvider.provider_type == "llm"
    assert LLMTranslationProvider.supports_batch is True
    assert LLMTranslationProvider.payload_unit == "tokens"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialize_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        LLMTranslationProvider().initialize(
            ProviderConfig(endpoint="", credential="k", extra={"model_name": "m"})
        )


def test_initialize_requires_credential() -> None:
    with pytest.raises(ValueError):
        LLMTranslationProvider().initialize(
            ProviderConfig(
                endpoint="https://x", credential="", extra={"model_name": "m"}
            )
        )


def test_initialize_requires_model_name() -> None:
    with pytest.raises(ValueError):
        LLMTranslationProvider().initialize(
            ProviderConfig(endpoint="https://x", credential="k")
        )


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_size_of_estimates_tokens() -> None:
    provider = LLMTranslationProvider()
    # ceil(len / 2)
    assert provider.size_of("") == 0
    assert provider.size_of("abcd") == 2
    assert provider.size_of("abcde") == 3


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_translate_batch_happy_path_uses_authorization_header() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return _chat_response(
            [
                {"id": 1, "translation": "你好"},
                {"id": 2, "translation": "世界"},
            ]
        )

    provider = _make_provider(handler)
    out = await provider.translate_batch(_entries("hello", "world"))

    assert [e.text for e in out] == ["你好", "世界"]
    assert [e.index for e in out] == [1, 2]
    assert [e.start_ms for e in out] == [0, 1000]
    assert [e.end_ms for e in out] == [1000, 2000]
    assert captured["url"] == "https://api.llm.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret-token"
    assert captured["body"]["model"] == "gpt-test"

    await provider.aclose()


@pytest.mark.asyncio
async def test_translate_batch_accepts_bare_array_response() -> None:
    """A provider that returns a raw JSON array (not chat-wrapped) works."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"id": 1, "translation": "你好"},
            ],
        )

    provider = _make_provider(handler)
    out = await provider.translate_batch(_entries("hello"))
    assert out[0].text == "你好"
    await provider.aclose()


@pytest.mark.asyncio
async def test_translate_batch_empty_returns_empty() -> None:
    """Empty input skips the HTTP call entirely."""
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=[])

    provider = _make_provider(handler)
    out = await provider.translate_batch([])
    assert out == []
    assert calls["n"] == 0
    await provider.aclose()


@pytest.mark.asyncio
async def test_translate_alias_delegates_to_translate_batch() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _chat_response([{"id": 1, "translation": "你好"}])

    provider = _make_provider(handler)
    out = await provider.translate(_entries("hello"))
    assert out[0].text == "你好"
    await provider.aclose()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_429_maps_to_rate_limit_error_with_retry_after() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "7"}, json={"error": "rate_limit"}
        )

    provider = _make_provider(handler)
    with pytest.raises(RateLimitError) as exc_info:
        await provider.translate_batch(_entries("hello"))
    assert exc_info.value.retry_after == 7.0
    assert exc_info.value.context["status_code"] == 429
    await provider.aclose()


@pytest.mark.asyncio
async def test_http_413_maps_to_payload_too_large() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="Payload Too Large")

    provider = _make_provider(handler)
    with pytest.raises(PayloadTooLargeError):
        await provider.translate_batch(_entries("x" * 1000))
    await provider.aclose()


@pytest.mark.asyncio
async def test_context_length_exceeded_body_maps_to_payload_too_large() -> None:
    """Even with a 400, a context-overflow body is recognized."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "This model's maximum context length is 8192.",
                    "code": "context_length_exceeded",
                }
            },
        )

    provider = _make_provider(handler)
    with pytest.raises(PayloadTooLargeError):
        await provider.translate_batch(_entries("very long input"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_timeout_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_malformed_json_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_malformed_content_string_maps_to_transient_error() -> None:
    """Content field that isn't a JSON array → transient."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "garbage"}}]}
        )

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_wrong_number_of_items_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _chat_response([{"id": 1, "translation": "你好"}])

    provider = _make_provider(handler)
    with pytest.raises(TransientError) as exc_info:
        await provider.translate_batch(_entries("hello", "world"))
    assert exc_info.value.context["expected"] == 2
    assert exc_info.value.context["actual"] == 1
    await provider.aclose()


@pytest.mark.asyncio
async def test_id_mismatch_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _chat_response([{"id": 99, "translation": "你好"}])

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_5xx_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_rate_limit_business_code_without_429_status() -> None:
    """Some providers return 400 with a ``rate_limit`` body."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"code": "rate_limit", "message": "rate limit"}}
        )

    provider = _make_provider(handler)
    with pytest.raises(RateLimitError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()
