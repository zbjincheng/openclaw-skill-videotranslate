"""Unit tests for :mod:`translation_dubbing_skill.providers.tts.protocol`.

Verifies that :class:`TTSProvider` behaves as a ``runtime_checkable``
structural Protocol: a trivial class that supplies all required attributes
and coroutine methods is recognised by ``isinstance``, while missing
members are rejected.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from translation_dubbing_skill.models import ProviderConfig
from translation_dubbing_skill.providers.tts.protocol import (
    TTSProvider,
    default_size_of_for,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeTTSProvider:
    """Minimal structural implementation of :class:`TTSProvider`."""

    provider_type: ClassVar[str] = "fake"
    supports_batch: ClassVar[bool] = False
    payload_unit: ClassVar[str] = "chars"

    def __init__(self) -> None:
        self.config: ProviderConfig | None = None
        self.calls: list[tuple[str, str]] = []

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config

    def size_of(self, text: str) -> int:
        return default_size_of_for(self.payload_unit)(text)  # type: ignore[arg-type]

    async def synth(self, text: str, voice_id: str) -> tuple[bytes, int]:
        self.calls.append((text, voice_id))
        return (text.encode("utf-8"), len(text) * 10)

    async def synth_batch(
        self,
        texts: list[str],
        voice_id: str,
    ) -> list[tuple[bytes, int]]:
        return [await self.synth(t, voice_id) for t in texts]


class _MissingMembers:
    """Implements only ``provider_type`` — should not satisfy the Protocol."""

    provider_type: ClassVar[str] = "broken"


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_protocol_is_runtime_checkable() -> None:
    """A compliant class passes ``isinstance`` against the Protocol."""
    assert isinstance(_FakeTTSProvider(), TTSProvider)


def test_non_compliant_class_rejected_by_isinstance() -> None:
    """A class missing protocol members is rejected."""
    assert not isinstance(_MissingMembers(), TTSProvider)


def test_protocol_defaults_supports_batch_to_false() -> None:
    """Per design, TTSProvider defaults to single-shot (``supports_batch=False``)."""
    assert TTSProvider.supports_batch is False


# ---------------------------------------------------------------------------
# Default ``size_of`` factory behaviour
# ---------------------------------------------------------------------------


def test_default_size_of_for_chars_returns_char_count() -> None:
    sizer = default_size_of_for("chars")
    assert sizer("hello") == 5
    assert sizer("你好") == 2


def test_default_size_of_for_tokens_returns_token_estimate() -> None:
    sizer = default_size_of_for("tokens")
    assert sizer("") == 0
    assert sizer("abcd") == 2


# ---------------------------------------------------------------------------
# Fake provider end-to-end behaviour
# ---------------------------------------------------------------------------


def test_fake_provider_initialize_stores_config() -> None:
    provider = _FakeTTSProvider()
    cfg = ProviderConfig(endpoint="https://x", credential="k")
    provider.initialize(cfg)
    assert provider.config is cfg


def test_fake_provider_size_of_uses_payload_unit() -> None:
    provider = _FakeTTSProvider()
    # payload_unit = "chars" -> len(text)
    assert provider.size_of("hello") == 5


@pytest.mark.asyncio
async def test_fake_provider_synth_returns_bytes_and_duration() -> None:
    provider = _FakeTTSProvider()
    audio, duration_ms = await provider.synth("你好", "voice-1")
    assert isinstance(audio, bytes)
    assert isinstance(duration_ms, int)
    assert duration_ms >= 0


@pytest.mark.asyncio
async def test_fake_provider_synth_batch_preserves_order_and_length() -> None:
    provider = _FakeTTSProvider()
    texts = ["one", "two", "three"]
    out = await provider.synth_batch(texts, "voice-1")
    assert len(out) == len(texts)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in out)
    # Order preserved: each audio payload decodes back to the original text.
    assert [audio.decode("utf-8") for audio, _ in out] == texts


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_tts_package_exports_protocol() -> None:
    from translation_dubbing_skill.providers import tts

    assert "TTSProvider" in tts.__all__
    assert tts.TTSProvider is TTSProvider
