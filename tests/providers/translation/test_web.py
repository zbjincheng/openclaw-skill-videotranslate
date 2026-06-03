"""Unit tests for :mod:`translation_dubbing_skill.providers.translation.web`.

Covers the happy path and each error-mapping path exercised by the
adaptive scheduler:

    * HTTP 429                → :class:`RateLimitError`
    * HTTP 413 / text-too-long → :class:`PayloadTooLargeError`
    * ``httpx.TimeoutException`` → :class:`TransientError`
    * Malformed JSON response   → :class:`TransientError`

Uses :class:`httpx.MockTransport` so the tests run fully offline.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from translation_dubbing_skill.models import ProviderConfig, SubtitleEntry
from translation_dubbing_skill.providers.registry import default_registry
from translation_dubbing_skill.providers.translation.web import (
    WebTranslationProvider,
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
    *,
    language_pair: str = "en-zh",
) -> WebTranslationProvider:
    provider = WebTranslationProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://api.web.test/translate",
            credential="secret-token",
            extra={"language_pair": language_pair},
        )
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _entries(*texts: str) -> list[SubtitleEntry]:
    return [
        SubtitleEntry(index=i + 1, start_ms=i * 1000, end_ms=(i + 1) * 1000, text=t)
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_web_provider_is_registered_on_import() -> None:
    assert "web" in default_registry.list("translation")


def test_web_provider_class_metadata() -> None:
    assert WebTranslationProvider.provider_type == "web"
    assert WebTranslationProvider.supports_batch is True
    assert WebTranslationProvider.payload_unit == "chars"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialize_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        WebTranslationProvider().initialize(
            ProviderConfig(endpoint="", credential="k")
        )


def test_initialize_requires_credential() -> None:
    with pytest.raises(ValueError):
        WebTranslationProvider().initialize(
            ProviderConfig(endpoint="https://x", credential="")
        )


def test_initialize_parses_language_pair() -> None:
    provider = WebTranslationProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://x",
            credential="k",
            extra={"language_pair": "en-ja"},
        )
    )
    assert provider.source_language == "en"
    assert provider.target_language == "ja"


def test_initialize_defaults_language_pair_when_missing() -> None:
    provider = WebTranslationProvider()
    provider.initialize(ProviderConfig(endpoint="https://x", credential="k"))
    assert provider.source_language == "en"
    assert provider.target_language == "zh"


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_size_of_is_character_count() -> None:
    provider = WebTranslationProvider()
    assert provider.size_of("") == 0
    assert provider.size_of("hello") == 5
    assert provider.size_of("你好") == 2


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_translate_batch_happy_path_sends_expected_payload() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured.append(
            {
                "authorization": request.headers.get("authorization"),
                "body": _json.loads(request.content),
            }
        )
        return httpx.Response(200, json={"translatedText": "你好"})

    provider = _make_provider(handler)
    out = await provider.translate_batch(_entries("hello"))

    assert out[0].text == "你好"
    assert out[0].index == 1
    assert out[0].start_ms == 0
    assert out[0].end_ms == 1000
    assert captured[0]["authorization"] == "Bearer secret-token"
    assert captured[0]["body"] == {"q": "hello", "source": "en", "target": "zh"}

    await provider.aclose()


@pytest.mark.asyncio
async def test_translate_batch_iterates_all_entries() -> None:
    """The batch wrapper issues one request per entry, preserving order."""
    responses = iter(["你好", "世界", "再见"])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"translatedText": next(responses)})

    provider = _make_provider(handler)
    out = await provider.translate_batch(_entries("hello", "world", "bye"))

    assert [e.text for e in out] == ["你好", "世界", "再见"]
    assert [e.index for e in out] == [1, 2, 3]
    await provider.aclose()


@pytest.mark.asyncio
async def test_empty_entry_text_skips_http_call() -> None:
    """Whitespace-only entries are returned as empty without an HTTP call."""
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"translatedText": "x"})

    provider = _make_provider(handler)
    out = await provider.translate_batch(
        [
            SubtitleEntry(index=1, start_ms=0, end_ms=1000, text="   "),
            SubtitleEntry(index=2, start_ms=1000, end_ms=2000, text=""),
        ]
    )
    assert calls["n"] == 0
    assert [e.text for e in out] == ["", ""]
    await provider.aclose()


@pytest.mark.asyncio
async def test_translate_batch_accepts_google_v2_shape() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": {"translations": [{"translatedText": "你好"}]}},
        )

    provider = _make_provider(handler)
    out = await provider.translate_batch(_entries("hello"))
    assert out[0].text == "你好"
    await provider.aclose()


@pytest.mark.asyncio
async def test_translate_batch_empty_returns_empty() -> None:
    """Empty list short-circuits without network calls."""
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"translatedText": "x"})

    provider = _make_provider(handler)
    assert await provider.translate_batch([]) == []
    assert calls["n"] == 0
    await provider.aclose()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_429_maps_to_rate_limit_error_with_retry_after() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "3"}, json={"error": "rate_limit"}
        )

    provider = _make_provider(handler)
    with pytest.raises(RateLimitError) as exc_info:
        await provider.translate_batch(_entries("hello"))
    assert exc_info.value.retry_after == 3.0
    await provider.aclose()


@pytest.mark.asyncio
async def test_http_413_maps_to_payload_too_large() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="Payload Too Large")

    provider = _make_provider(handler)
    with pytest.raises(PayloadTooLargeError):
        await provider.translate_batch(_entries("x" * 10_000))
    await provider.aclose()


@pytest.mark.asyncio
async def test_text_too_long_business_error_maps_to_payload_too_large() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "text too long"})

    provider = _make_provider(handler)
    with pytest.raises(PayloadTooLargeError):
        await provider.translate_batch(_entries("hello"))
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
        return httpx.Response(200, content=b"not json")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_response_missing_known_fields_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_5xx_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.translate_batch(_entries("hello"))
    await provider.aclose()
