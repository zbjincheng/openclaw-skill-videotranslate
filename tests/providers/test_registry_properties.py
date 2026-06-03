"""Property-based tests for the provider registry.

Task 5.4 — Property 10: unregistered provider error.

Validates: Requirements 5.7, 6.8

For any ``provider_type`` string NOT registered on a
:class:`ProviderRegistry`, the translation coordinator and the TTS
coordinator — both of which ultimately call ``registry.create(kind,
provider_type, config)`` to materialize their provider — must raise
:class:`ProviderNotRegisteredError`. The error must also expose the set
of currently-registered provider-type identifiers so the caller (and the
error message derived from it) can list them back to the user.

The property here exercises the registry directly, since the coordinator
contract for "loading the provider" is precisely "invoke
``registry.create``"; adding an extra indirection layer would only
re-test the registry.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from hypothesis import assume, given, strategies as st

from translation_dubbing_skill.errors import ProviderNotRegisteredError
from translation_dubbing_skill.models import ProviderConfig
from translation_dubbing_skill.providers import ProviderRegistry


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubProvider:
    """Minimal provider double satisfying the registry's instantiation path.

    Both the translation and TTS registry paths merely need a class with a
    zero-argument ``__init__`` and an ``initialize(config)`` method. This
    stub is intentionally generic so the same class can back either kind.
    """

    provider_type: ClassVar[str] = "stub"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    def __init__(self) -> None:
        self.config: ProviderConfig | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.config = config


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Seed each registry with a small, known set of providers. These are the
# identifiers that must appear in ``registered_types`` when an unknown
# type is requested.
_REGISTERED_TRANSLATION_TYPES: tuple[str, ...] = ("llm", "web", "fake")
_REGISTERED_TTS_TYPES: tuple[str, ...] = ("llm", "web")

_ALL_REGISTERED: frozenset[str] = frozenset(
    _REGISTERED_TRANSLATION_TYPES + _REGISTERED_TTS_TYPES
)

# Provider-type strings: non-empty printable text. ``provider_type`` is a
# user-facing identifier, so empty strings are rejected upstream by the
# registry itself (that path is covered by the unit tests). Here we focus
# on "syntactically valid but not registered" inputs.
_provider_type_strategy = (
    st.text(
        alphabet=st.characters(
            min_codepoint=0x20,
            max_codepoint=0x7E,
            blacklist_categories=("Cs",),
        ),
        min_size=1,
        max_size=32,
    )
    .filter(lambda s: s not in _ALL_REGISTERED)
)

_endpoint_strategy = st.text(min_size=1, max_size=64).filter(lambda s: s.strip() != "")
_credential_strategy = st.text(min_size=1, max_size=64).filter(lambda s: s.strip() != "")


def _make_seeded_registry() -> ProviderRegistry:
    """Build a fresh registry populated with the canonical provider set."""
    registry = ProviderRegistry()
    for provider_type in _REGISTERED_TRANSLATION_TYPES:
        registry.register("translation", provider_type, _StubProvider)
    for provider_type in _REGISTERED_TTS_TYPES:
        registry.register("tts", provider_type, _StubProvider)
    return registry


# ---------------------------------------------------------------------------
# Property 10
# ---------------------------------------------------------------------------


@given(
    provider_type=_provider_type_strategy,
    endpoint=_endpoint_strategy,
    credential=_credential_strategy,
)
def test_property_10_translation_coordinator_raises_for_unregistered_type(
    provider_type: str,
    endpoint: str,
    credential: str,
) -> None:
    """Translation path: unknown ``provider_type`` raises with context.

    Validates: Requirements 5.7, 6.8
    """
    # Belt-and-suspenders — the filter above already excludes these, but
    # Hypothesis shrinking can be surprising.
    assume(provider_type not in _ALL_REGISTERED)

    registry = _make_seeded_registry()
    config = ProviderConfig(endpoint=endpoint, credential=credential)

    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        registry.create("translation", provider_type, config)

    err = excinfo.value

    # Context carries the requested (missing) type verbatim.
    assert err.context["requested_type"] == provider_type

    # Context lists exactly the seeded translation registrations, sorted.
    assert err.context["registered_types"] == sorted(_REGISTERED_TRANSLATION_TYPES)

    # The error surfaces from the translating stage (mirrors the
    # coordinator stage) and its reason mentions the offending identifier
    # — repr-quoted so special characters survive round-tripping to logs.
    assert err.stage == "translating"
    assert repr(provider_type) in err.reason


@given(
    provider_type=_provider_type_strategy,
    endpoint=_endpoint_strategy,
    credential=_credential_strategy,
)
def test_property_10_tts_coordinator_raises_for_unregistered_type(
    provider_type: str,
    endpoint: str,
    credential: str,
) -> None:
    """TTS path: unknown ``provider_type`` raises with context.

    Validates: Requirements 5.7, 6.8
    """
    assume(provider_type not in _ALL_REGISTERED)

    registry = _make_seeded_registry()
    config = ProviderConfig(endpoint=endpoint, credential=credential)

    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        registry.create("tts", provider_type, config)

    err = excinfo.value

    assert err.context["requested_type"] == provider_type
    assert err.context["registered_types"] == sorted(_REGISTERED_TTS_TYPES)
    assert err.stage == "tts"
    assert repr(provider_type) in err.reason


@given(
    provider_type=_provider_type_strategy,
    endpoint=_endpoint_strategy,
    credential=_credential_strategy,
)
def test_property_10_registered_types_listed_match_registry_list(
    provider_type: str,
    endpoint: str,
    credential: str,
) -> None:
    """``registered_types`` in the error matches ``registry.list(kind)``.

    This nails down the "错误消息列出已注册的类型标识" clause of Property 10:
    the list in the error's context must be the authoritative view of
    what's currently registered, for both kinds.

    Validates: Requirements 5.7, 6.8
    """
    assume(provider_type not in _ALL_REGISTERED)

    registry = _make_seeded_registry()
    config = ProviderConfig(endpoint=endpoint, credential=credential)

    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        registry.create("translation", provider_type, config)
    assert excinfo.value.context["registered_types"] == registry.list("translation")

    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        registry.create("tts", provider_type, config)
    assert excinfo.value.context["registered_types"] == registry.list("tts")
