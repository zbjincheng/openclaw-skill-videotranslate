"""Unit tests for :mod:`translation_dubbing_skill.providers.tts.llm`.

Covers the happy path and each error-mapping path exercised by the
adaptive scheduler:

    * HTTP 429                → :class:`RateLimitError`
    * HTTP 413 / context overflow → :class:`PayloadTooLargeError`
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
from translation_dubbing_skill.providers.tts.llm import LLMTTSProvider
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
) -> LLMTTSProvider:
    provider = LLMTTSProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://api.llm.test/v1/audio/speech",
            credential="secret-token",
            extra={"model_name": "tts-1", "default_voice": "alloy"},
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


def test_llm_tts_provider_is_registered_on_import() -> None:
    assert "llm" in default_registry.list("tts")


def test_llm_tts_provider_class_metadata() -> None:
    assert LLMTTSProvider.provider_type == "llm"
    assert LLMTTSProvider.supports_batch is True
    assert LLMTTSProvider.payload_unit == "tokens"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialize_requires_endpoint() -> None:
    with pytest.raises(ValueError):
        LLMTTSProvider().initialize(
            ProviderConfig(
                endpoint="",
                credential="k",
                extra={"model_name": "m"},
            )
        )


def test_initialize_requires_credential() -> None:
    with pytest.raises(ValueError):
        LLMTTSProvider().initialize(
            ProviderConfig(
                endpoint="https://x",
                credential="",
                extra={"model_name": "m"},
            )
        )


def test_initialize_requires_model_name() -> None:
    with pytest.raises(ValueError):
        LLMTTSProvider().initialize(
            ProviderConfig(endpoint="https://x", credential="k")
        )


def test_initialize_captures_default_voice() -> None:
    provider = LLMTTSProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://x",
            credential="k",
            extra={"model_name": "m", "default_voice": "alloy"},
        )
    )
    assert provider.default_voice == "alloy"


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_size_of_estimates_tokens() -> None:
    provider = LLMTTSProvider()
    assert provider.size_of("") == 0
    assert provider.size_of("abcd") == 2
    assert provider.size_of("abcde") == 3


# ---------------------------------------------------------------------------
# Happy path — synth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synth_happy_path_returns_bytes_and_duration() -> None:
    audio_payload = b"PCM-DATA-HERE"
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "audio_base64": _b64(audio_payload),
                "duration_ms": 1234,
            },
        )

    provider = _make_provider(handler)
    audio, duration = await provider.synth("你好", "alloy")

    assert audio == audio_payload
    assert duration == 1234
    assert captured["authorization"] == "Bearer secret-token"
    assert captured["body"]["model"] == "tts-1"
    assert captured["body"]["voice"] == "alloy"
    assert captured["body"]["inputs"] == ["你好"]

    await provider.aclose()


@pytest.mark.asyncio
async def test_synth_accepts_raw_audio_body() -> None:
    audio_payload = b"RAW-OPUS-BYTES"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=audio_payload,
            headers={
                "Content-Type": "audio/mpeg",
                "X-Audio-Duration-Ms": "500",
            },
        )

    provider = _make_provider(handler)
    audio, duration = await provider.synth("hi", "alloy")
    assert audio == audio_payload
    assert duration == 500
    await provider.aclose()


# ---------------------------------------------------------------------------
# Happy path — synth_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synth_batch_returns_pairs_in_order() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"audio_base64": _b64(b"A"), "duration_ms": 100},
                {"audio_base64": _b64(b"BB"), "duration_ms": 200},
                {"audio_base64": _b64(b"CCC"), "duration_ms": 300},
            ],
        )

    provider = _make_provider(handler)
    out = await provider.synth_batch(["x", "yy", "zzz"], "alloy")
    assert out == [(b"A", 100), (b"BB", 200), (b"CCC", 300)]
    await provider.aclose()


@pytest.mark.asyncio
async def test_synth_batch_empty_returns_empty() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=[])

    provider = _make_provider(handler)
    assert await provider.synth_batch([], "alloy") == []
    assert calls["n"] == 0
    await provider.aclose()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_429_maps_to_rate_limit_error_with_retry_after() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "5"}, json={"error": "rate_limit"}
        )

    provider = _make_provider(handler)
    with pytest.raises(RateLimitError) as exc_info:
        await provider.synth("hello", "alloy")
    assert exc_info.value.retry_after == 5.0
    assert exc_info.value.context["status_code"] == 429
    await provider.aclose()


@pytest.mark.asyncio
async def test_http_413_maps_to_payload_too_large() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="Payload Too Large")

    provider = _make_provider(handler)
    with pytest.raises(PayloadTooLargeError):
        await provider.synth("x" * 5000, "alloy")
    await provider.aclose()


@pytest.mark.asyncio
async def test_input_too_long_body_maps_to_payload_too_large() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "input is too long"}}
        )

    provider = _make_provider(handler)
    with pytest.raises(PayloadTooLargeError):
        await provider.synth("hi", "alloy")
    await provider.aclose()


@pytest.mark.asyncio
async def test_timeout_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hi", "alloy")
    await provider.aclose()


@pytest.mark.asyncio
async def test_malformed_json_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json-at-all")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hi", "alloy")
    await provider.aclose()


@pytest.mark.asyncio
async def test_wrong_item_count_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"audio_base64": _b64(b"X"), "duration_ms": 10}],
        )

    provider = _make_provider(handler)
    with pytest.raises(TransientError) as exc_info:
        await provider.synth_batch(["a", "b"], "alloy")
    assert exc_info.value.context["expected"] == 2
    assert exc_info.value.context["actual"] == 1
    await provider.aclose()


@pytest.mark.asyncio
async def test_missing_audio_base64_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"duration_ms": 100})

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hi", "alloy")
    await provider.aclose()


@pytest.mark.asyncio
async def test_negative_duration_ms_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"audio_base64": _b64(b"X"), "duration_ms": -5},
        )

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hi", "alloy")
    await provider.aclose()


@pytest.mark.asyncio
async def test_5xx_maps_to_transient_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    provider = _make_provider(handler)
    with pytest.raises(TransientError):
        await provider.synth("hi", "alloy")
    await provider.aclose()
