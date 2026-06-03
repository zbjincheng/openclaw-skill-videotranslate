"""Property-based tests for the unified ``SkillError`` hierarchy.

Task 2.5 — Property 24: error object structure completeness.

Validates: Requirements 10.5

For any ``SkillError`` subclass and for any valid ``(reason, context)``
input, the resulting instance must expose:

* ``stage``: a non-empty ``str`` drawn from the closed set of valid
  pipeline stages.
* ``code``: a non-empty ``str`` (machine-readable error code).
* ``reason``: a non-empty ``str`` (human-readable summary).
* ``context``: a ``dict`` (may be empty, but must never be ``None`` or
  any other type).

These four invariants together are what R10.5 requires from the unified
error model: any subsystem error transported back to the caller carries a
non-empty stage / reason and a dict-shaped context. The property walks
every concrete subclass via Hypothesis' ``sampled_from`` over the registry
exported by ``translation_dubbing_skill.errors``.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, strategies as st

from translation_dubbing_skill.errors import (
    InvalidTimestampError,
    ManifestParamMissingError,
    NoEnglishSubtitleError,
    OriginalAudioExtractionError,
    ProviderContractViolationError,
    ProviderNotRegisteredError,
    ProviderUnavailableError,
    SkillError,
    SubtitleFileInaccessibleError,
    SubtitleParseError,
    TranslationError,
    TTSError,
    UnsupportedProcessingModeError,
    UnsupportedProviderTypeError,
    UnsupportedSubtitleFormatError,
    UnsupportedVideoFormatError,
    VideoDecodeError,
    VideoFileInaccessibleError,
)

# All 17 concrete ``SkillError`` subclasses. Any new subclass added to the
# hierarchy must also be registered here so the property covers it.
_ERROR_CLASSES: tuple[type[SkillError], ...] = (
    ManifestParamMissingError,
    UnsupportedProcessingModeError,
    UnsupportedProviderTypeError,
    VideoFileInaccessibleError,
    SubtitleFileInaccessibleError,
    UnsupportedVideoFormatError,
    UnsupportedSubtitleFormatError,
    SubtitleParseError,
    InvalidTimestampError,
    NoEnglishSubtitleError,
    ProviderNotRegisteredError,
    ProviderUnavailableError,
    TranslationError,
    TTSError,
    ProviderContractViolationError,
    VideoDecodeError,
    OriginalAudioExtractionError,
)

# The closed set of valid ``stage`` values as defined by the error model.
_VALID_STAGES: frozenset[str] = frozenset(
    {"input", "parsing", "translating", "tts", "aligning", "muxing"}
)


# Reasons are constrained to non-empty text: whitespace-only strings would
# technically satisfy ``len(reason) > 0`` but R10.5 talks about
# human-readable summaries, so we require at least one non-whitespace char.
_reason_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")

# Context values: arbitrary JSON-ish scalars / containers. We keep the
# structure shallow to keep example generation fast while still exercising
# nested dicts and lists (which the implementation also needs to handle
# when serializing via ``to_dict``).
_context_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=50),
)
_context_value: st.SearchStrategy[Any] = st.recursive(
    _context_scalar,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=5),
    ),
    max_leaves=10,
)
_context_strategy = st.dictionaries(
    st.text(min_size=1, max_size=20),
    _context_value,
    max_size=6,
)


@given(
    cls=st.sampled_from(_ERROR_CLASSES),
    reason=_reason_strategy,
    context=_context_strategy,
)
def test_property_24_error_object_structure_is_complete(
    cls: type[SkillError],
    reason: str,
    context: dict[str, Any],
) -> None:
    """Property 24 — every ``SkillError`` instance has a complete shape.

    Validates: Requirements 10.5
    """
    err = cls(reason, context=context)

    # stage is a non-empty string drawn from the valid stages set.
    assert isinstance(err.stage, str)
    assert err.stage != ""
    assert err.stage in _VALID_STAGES

    # code is a non-empty string.
    assert isinstance(err.code, str)
    assert err.code != ""

    # reason is a non-empty string and matches the caller-supplied value.
    assert isinstance(err.reason, str)
    assert err.reason != ""
    assert err.reason == reason

    # context is always a dict, never None or any other type.
    assert isinstance(err.context, dict)
    assert err.context == context


@given(cls=st.sampled_from(_ERROR_CLASSES), reason=_reason_strategy)
def test_property_24_error_object_structure_with_default_context(
    cls: type[SkillError], reason: str
) -> None:
    """Same invariants hold when ``context`` is omitted entirely.

    Validates: Requirements 10.5
    """
    err = cls(reason)

    assert isinstance(err.stage, str) and err.stage in _VALID_STAGES
    assert isinstance(err.code, str) and err.code != ""
    assert isinstance(err.reason, str) and err.reason == reason
    assert isinstance(err.context, dict)
    assert err.context == {}
