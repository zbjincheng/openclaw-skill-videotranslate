"""Unit tests for :mod:`translation_dubbing_skill.providers.translation.protocol`.

Verifies that :class:`TranslationProvider` behaves as a ``runtime_checkable``
structural Protocol: a trivial class that supplies all required attributes
and coroutine methods is recognised by ``isinstance``, while missing
members are rejected.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from translation_dubbing_skill.models import ProviderConfig, SubtitleEntry
from translation_dubbing_skill.providers.translation.protocol import (
    TranslationProvider,
    default_size_of_for,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeTranslationProvider:
    """Minimal structural implementation of :class:`TranslationProvider`."""

    provider_type: ClassVar[str] = "fake"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "tokens"

    def __init__(self) -> None:
        self.config: ProviderConfig | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config

    def size_of(self, text: str) -> int:
        return default_size_of_for(self.payload_unit)(text)  # type: ignore[arg-type]

    async def translate_batch(
        self,
        entries: list[SubtitleEntry],
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        return list(entries)

    async def translate(
        self,
        entries: list[SubtitleEntry],
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        return await self.translate_batch(entries, target_language)


class _MissingMembers:
    """Implements only ``provider_type`` — should not satisfy the Protocol."""

    provider_type: ClassVar[str] = "broken"


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_protocol_is_runtime_checkable() -> None:
    """A compliant class passes ``isinstance`` against the Protocol."""
    assert isinstance(_FakeTranslationProvider(), TranslationProvider)


def test_non_compliant_class_rejected_by_isinstance() -> None:
    """A class missing protocol members is rejected."""
    assert not isinstance(_MissingMembers(), TranslationProvider)


def test_protocol_declares_expected_class_attributes() -> None:
    """Class-level metadata is declared on the Protocol itself."""
    # supports_batch has a default of True per the design.
    assert TranslationProvider.supports_batch is True


# ---------------------------------------------------------------------------
# Default ``size_of`` factory behaviour
# ---------------------------------------------------------------------------


def test_default_size_of_for_chars_returns_char_count() -> None:
    sizer = default_size_of_for("chars")
    assert sizer("hello") == 5
    assert sizer("你好") == 2


def test_default_size_of_for_tokens_returns_token_estimate() -> None:
    sizer = default_size_of_for("tokens")
    # ceil(len / 2)
    assert sizer("") == 0
    assert sizer("abcd") == 2
    assert sizer("abcde") == 3


# ---------------------------------------------------------------------------
# Fake provider end-to-end behaviour
# ---------------------------------------------------------------------------


def test_fake_provider_initialize_stores_config() -> None:
    provider = _FakeTranslationProvider()
    cfg = ProviderConfig(endpoint="https://x", credential="k")
    provider.initialize(cfg)
    assert provider.config is cfg


def test_fake_provider_size_of_uses_payload_unit() -> None:
    provider = _FakeTranslationProvider()
    # payload_unit = "tokens" -> ceil(len / 2)
    assert provider.size_of("abcd") == 2


@pytest.mark.asyncio
async def test_fake_provider_translate_batch_preserves_entries() -> None:
    provider = _FakeTranslationProvider()
    entries = [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="hi"),
        SubtitleEntry(index=2, start_ms=1_000, end_ms=2_000, text="world"),
    ]
    output = await provider.translate_batch(entries)
    assert output == entries


@pytest.mark.asyncio
async def test_fake_provider_translate_alias_delegates() -> None:
    provider = _FakeTranslationProvider()
    entries = [SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="hi")]
    output = await provider.translate(entries)
    assert output == entries


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_translation_package_exports_protocol() -> None:
    from translation_dubbing_skill.providers import translation

    assert "TranslationProvider" in translation.__all__
    assert translation.TranslationProvider is TranslationProvider
