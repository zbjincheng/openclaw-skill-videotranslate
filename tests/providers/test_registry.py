"""Unit tests for :mod:`translation_dubbing_skill.providers.registry`.

Covers:

- Registration via :meth:`ProviderRegistry.register` for both kinds
  (``"translation"`` and ``"tts"``).
- Instantiation via :meth:`ProviderRegistry.create` which calls
  ``initialize(config)`` on the new instance.
- Listing via :meth:`ProviderRegistry.list` (returns sorted copies).
- Error path: :class:`ProviderNotRegisteredError` surfaces with
  ``requested_type`` and the currently registered identifiers when an
  unknown ``provider_type`` is requested.
- The module-level ``@register`` decorator registers to
  ``default_registry`` and returns the class unchanged.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from translation_dubbing_skill.errors import ProviderNotRegisteredError
from translation_dubbing_skill.models import ProviderConfig
from translation_dubbing_skill.providers import (
    ProviderRegistry,
    default_registry,
    register,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeTranslationProvider:
    """Minimal provider double used throughout the registry tests."""

    provider_type: ClassVar[str] = "fake"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    def __init__(self) -> None:
        self.config: ProviderConfig | None = None
        self.initialized: bool = False

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config
        self.initialized = True


class _FakeTTSProvider:
    provider_type: ClassVar[str] = "fake-tts"
    supports_batch: ClassVar[bool] = False
    payload_unit: ClassVar[str] = "chars"

    def __init__(self) -> None:
        self.config: ProviderConfig | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config


# ---------------------------------------------------------------------------
# register + create
# ---------------------------------------------------------------------------


def test_register_and_create_translation_provider_returns_initialized_instance() -> None:
    registry = ProviderRegistry()
    registry.register("translation", "fake", _FakeTranslationProvider)

    cfg = ProviderConfig(endpoint="https://x", credential="k")
    instance = registry.create("translation", "fake", cfg)

    assert isinstance(instance, _FakeTranslationProvider)
    assert instance.initialized is True
    assert instance.config is cfg


def test_register_and_create_tts_provider_isolated_from_translation() -> None:
    registry = ProviderRegistry()
    registry.register("tts", "fake-tts", _FakeTTSProvider)

    # Same identifier in the "translation" kind must not exist.
    with pytest.raises(ProviderNotRegisteredError):
        registry.create(
            "translation",
            "fake-tts",
            ProviderConfig(endpoint="x", credential="k"),
        )

    instance = registry.create(
        "tts",
        "fake-tts",
        ProviderConfig(endpoint="x", credential="k"),
    )
    assert isinstance(instance, _FakeTTSProvider)


def test_register_rejects_empty_provider_type() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ValueError):
        registry.register("translation", "", _FakeTranslationProvider)


def test_register_rejects_invalid_kind() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ValueError):
        registry.register("bogus", "fake", _FakeTranslationProvider)  # type: ignore[arg-type]


def test_register_overwrite_replaces_previous_class() -> None:
    registry = ProviderRegistry()

    class _Other:
        def __init__(self) -> None:
            self.initialized = False

        def initialize(self, config: ProviderConfig) -> None:
            self.initialized = True

    registry.register("translation", "fake", _FakeTranslationProvider)
    registry.register("translation", "fake", _Other)

    instance = registry.create(
        "translation",
        "fake",
        ProviderConfig(endpoint="x", credential="k"),
    )
    assert isinstance(instance, _Other)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_returns_sorted_registered_identifiers() -> None:
    registry = ProviderRegistry()
    registry.register("translation", "web", _FakeTranslationProvider)
    registry.register("translation", "llm", _FakeTranslationProvider)
    registry.register("translation", "fake", _FakeTranslationProvider)

    assert registry.list("translation") == ["fake", "llm", "web"]


def test_list_empty_kind_returns_empty_list() -> None:
    registry = ProviderRegistry()
    assert registry.list("translation") == []
    assert registry.list("tts") == []


def test_list_returns_fresh_copy_each_call() -> None:
    registry = ProviderRegistry()
    registry.register("translation", "fake", _FakeTranslationProvider)

    first = registry.list("translation")
    first.append("mutated")

    assert registry.list("translation") == ["fake"]


def test_list_rejects_invalid_kind() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ValueError):
        registry.list("bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ProviderNotRegisteredError shape
# ---------------------------------------------------------------------------


def test_create_unknown_translation_type_raises_with_context() -> None:
    registry = ProviderRegistry()
    registry.register("translation", "llm", _FakeTranslationProvider)
    registry.register("translation", "web", _FakeTranslationProvider)

    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        registry.create(
            "translation",
            "ghost",
            ProviderConfig(endpoint="x", credential="k"),
        )

    err = excinfo.value
    assert err.context["requested_type"] == "ghost"
    assert err.context["registered_types"] == ["llm", "web"]
    assert err.context["kind"] == "translation"
    assert err.stage == "translating"
    assert err.code == "provider_not_registered"


def test_create_unknown_tts_type_uses_tts_stage() -> None:
    registry = ProviderRegistry()
    registry.register("tts", "llm", _FakeTTSProvider)

    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        registry.create(
            "tts",
            "ghost",
            ProviderConfig(endpoint="x", credential="k"),
        )

    err = excinfo.value
    assert err.stage == "tts"
    assert err.context["registered_types"] == ["llm"]


def test_create_rejects_invalid_kind() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ValueError):
        registry.create("bogus", "fake", ProviderConfig(endpoint="x", credential="k"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# @register decorator + default_registry singleton
# ---------------------------------------------------------------------------


def test_register_decorator_adds_to_default_registry_and_returns_class() -> None:
    # Use a unique provider_type to avoid cross-test contamination.
    unique_type = "decorator-test-only"

    @register(kind="translation", provider_type=unique_type)
    class _DecoratedProvider:
        provider_type: ClassVar[str] = unique_type
        supports_batch: ClassVar[bool] = True
        payload_unit: ClassVar[str] = "chars"

        def __init__(self) -> None:
            self.config: ProviderConfig | None = None

        def initialize(self, config: ProviderConfig) -> None:
            self.config = config

    try:
        assert unique_type in default_registry.list("translation")
        # Decorator must return the class unchanged.
        assert _DecoratedProvider.__name__ == "_DecoratedProvider"

        instance = default_registry.create(
            "translation",
            unique_type,
            ProviderConfig(endpoint="x", credential="k"),
        )
        assert isinstance(instance, _DecoratedProvider)
    finally:
        # Clean up so subsequent tests don't see this registration.
        default_registry._classes["translation"].pop(unique_type, None)


def test_register_decorator_supports_tts_kind() -> None:
    unique_type = "decorator-tts-only"

    @register(kind="tts", provider_type=unique_type)
    class _DecoratedTTS:
        provider_type: ClassVar[str] = unique_type
        supports_batch: ClassVar[bool] = False
        payload_unit: ClassVar[str] = "chars"

        def __init__(self) -> None:
            pass

        def initialize(self, config: ProviderConfig) -> None:
            pass

    try:
        assert unique_type in default_registry.list("tts")
    finally:
        default_registry._classes["tts"].pop(unique_type, None)


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_providers_package_exports_registry_symbols() -> None:
    from translation_dubbing_skill import providers

    assert "ProviderRegistry" in providers.__all__
    assert "default_registry" in providers.__all__
    assert "register" in providers.__all__
    assert providers.default_registry is default_registry
