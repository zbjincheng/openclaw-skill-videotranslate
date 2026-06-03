"""Unit tests for :mod:`translation_dubbing_skill.providers.tts.web`.

Covers the happy path and each error-mapping path exercised by the
adaptive scheduler:

    * HTTP 429                → :class:`RateLimitError`
    * HTTP 413 / char-limit    → :class:`PayloadTooLargeError`
    * ``httpx.TimeoutException`` → :class:`TransientError`
    * Malformed JSON / missing audio → :class:`TransientError`

Uses :class:`httpx.MockTransport` so the tests run fully offline.
"""

from __future__ import annotations

import base64
import json
from typing import Callable

import httpx
import pytest

from translation_dubbing_skill.models import ProviderConfig
from translation_dubbing_skill.providers.registry import default_registry
from translation_dubbing_skill.providers.tts.web import WebTTSProvider
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
) -> WebTTSProvider:
    provider = WebTTSProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://api.web-tts.test/synth",
            credential="secret-token",
            extra={"default_voice": "zh-CN-XiaoxiaoNeural"},
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    return provider


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Registration / metadata
# ---------------------------------------------------------------------------


def test_web_tts_provider_is_registered_on_import() -> None:
    assert "web" in default_registry.list("tts")


def test_web_tts_provider_class_metadata() -> None:
    assert WebTTSProvider.provider_type == "web"
    assert WebTTSProvider.supports_batch is False
    assert WebTTSProvider.payload_unit == "chars"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialize_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        WebTTSProvider().initialize(
            ProviderConfig(endpoint="", credential="k")
        )


def test_initialize_requires_credential() -> None:
    with pytest.raises(ValueError):
        WebTTSProvider().initialize(
            ProviderConfig(endpoint="https://x", credential="")
        )


def test_initialize_captures_default_voice() -> None:
    provider = WebTTSProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://x",
            credential="k",
            extra={"default_voice": "voice-123"},
        )
    )
    assert provider.default_voice == "voice-123"


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_size_of_is_character_count() -> None:
    provider = WebTTSProvider()
    assert provider.size_of("") == 0
    assert provider.size_of("hello") == 5
    assert provider.size_of("你好") == 2


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synth_happy_path_sends_expected_payload() -> None:
    audio_payload = b"OPUS-BYTES"
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "audio_base64": _b64(audio_payload),
                "duration_ms": 750,
            },
        )

    provider = _make_provider(handler)
    audio, duration = await provider.synth("你好", "zh-CN-XiaoxiaoNeural")

    assert audio == audio_payload
    assert duration == 750
    assert captured["url"] == "https://api.web-tts.test/synth"
    assert captured["authorization"] == "Bearer secret-token"
    assert captured["body"] == {"text": "你好", "voice": "zh-CN-XiaoxiaoNeural"}
    await provider.aclose()


@pytest.mark.asyncio
async def test_synth_accepts_raw_audio_body() -> None:
    audio_payload = b"raw-wav-bytes"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=audio_payload,
            headers={
                "Content-Type": "audio/wav",
                "X-Audio-Duration-Ms": "321",
            },
        )

    provider = _make_provider(handler)
    audio, duration = await provider.synth("hi", "v1")
    assert audio == audio_payload
    assert duration == 321
    await provider.aclose()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_429_maps_to_rate_limit_error_with_retry_after() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "4"}, json={"error": "rate_limit"}
        )

    provider = _make_provider(handler)
    with pytest.raises(RateLimitError) as exc_info:
        await provider.synth("hello", "v1")
    assert exc_info.value.retry_after == 4.0
    await provider.aclose()


@pytest.mark.asyncio
async def test_http_413_maps_to_payload_too_large() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="Payload Too Large")

    provider = _make_provider(handler)
    with pytest.raises(PayloadTooLargeError):
        await provider.synth("x" * 10_000, "v1")
    await provider.aclose()


@pytest.mark.asyncio
async def test_text_too_long_business_body_maps_to_payload_too_large() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "text too long"})

    provider = _make_provider(handler)
    with pytest.raises(PayloadTooLargeError):
        await provider.synth("hello", "v1")
    await provider.aclose()


@pytest.mark.asyncio
async def test_timeout_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hello", "v1")
    await provider.aclose()


@pytest.mark.asyncio
async def test_malformed_json_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hello", "v1")
    await provider.aclose()


@pytest.mark.asyncio
async def test_missing_audio_base64_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"duration_ms": 100})

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hi", "v1")
    await provider.aclose()


@pytest.mark.asyncio
async def test_negative_duration_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"audio_base64": _b64(b"X"), "duration_ms": -10},
        )

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hi", "v1")
    await provider.aclose()


@pytest.mark.asyncio
async def test_5xx_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hi", "v1")
    await provider.aclose()
