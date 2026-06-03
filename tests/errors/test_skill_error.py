"""Unit tests for the unified error hierarchy.

Covers:
- ``SkillError`` base-class shape (stage/code/reason/context) and defaults.
- Every subclass has the design-mandated default ``stage`` and ``code``.
- ``to_dict`` returns the canonical 4-key payload with sensitive keys
  redacted recursively regardless of key case.
- Stage validation rejects out-of-range values.
- Public API is re-exported from the package ``__init__``.
"""

from __future__ import annotations

import pytest

from translation_dubbing_skill import errors as errors_module
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


# ---------------------------------------------------------------------------
# Base class semantics
# ---------------------------------------------------------------------------


def test_skill_error_inherits_from_exception() -> None:
    """``SkillError`` is a normal Python exception."""
    assert issubclass(SkillError, Exception)


def test_skill_error_uses_subclass_defaults() -> None:
    """When ``stage``/``code`` are omitted, subclass defaults apply."""
    err = SubtitleParseError("bad line", context={"line_number": 3})
    assert err.stage == "parsing"
    assert err.code == "subtitle_parse_error"
    assert err.reason == "bad line"
    assert err.context == {"line_number": 3}
    # Exception message is the reason
    assert str(err) == "bad line"


def test_skill_error_context_defaults_to_empty_dict() -> None:
    """An omitted ``context`` becomes an empty dict, never ``None``."""
    err = TranslationError("boom")
    assert err.context == {}
    assert isinstance(err.context, dict)


def test_skill_error_context_is_copied_not_aliased() -> None:
    """Mutating the original dict does not leak into the error."""
    ctx = {"entry_index": 7}
    err = TranslationError("boom", context=ctx)
    ctx["entry_index"] = 999
    assert err.context == {"entry_index": 7}


def test_skill_error_allows_stage_and_code_override() -> None:
    """Callers can override the class defaults when a subclass is reused."""
    # e.g. ProviderNotRegisteredError surfaces from the TTS engine
    err = ProviderNotRegisteredError(
        "no such tts provider",
        context={"requested_type": "x", "registered_types": ["llm"]},
        stage="tts",
        code="provider_not_registered",
    )
    assert err.stage == "tts"
    assert err.code == "provider_not_registered"


def test_skill_error_rejects_invalid_stage() -> None:
    """An unknown stage raises ``ValueError`` at construction time."""
    with pytest.raises(ValueError, match="invalid stage"):
        SubtitleParseError("x", stage="frobnicate")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Subclass defaults (R1.10–R1.12, R2.4–R2.7, R5.7–R5.9, R6.8–R6.10,
# R7.6, R9.14–R9.15, R10.1–R10.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "expected_stage", "expected_code"),
    [
        (ManifestParamMissingError, "input", "manifest_param_missing"),
        (UnsupportedProcessingModeError, "input", "unsupported_processing_mode"),
        (UnsupportedProviderTypeError, "input", "unsupported_provider_type"),
        (VideoFileInaccessibleError, "input", "video_file_inaccessible"),
        (SubtitleFileInaccessibleError, "input", "subtitle_file_inaccessible"),
        (UnsupportedVideoFormatError, "input", "unsupported_video_format"),
        (UnsupportedSubtitleFormatError, "input", "unsupported_subtitle_format"),
        (SubtitleParseError, "parsing", "subtitle_parse_error"),
        (InvalidTimestampError, "parsing", "invalid_timestamp"),
        (NoEnglishSubtitleError, "parsing", "no_english_subtitle"),
        (ProviderNotRegisteredError, "translating", "provider_not_registered"),
        (ProviderUnavailableError, "translating", "provider_unavailable"),
        (TranslationError, "translating", "translation_error"),
        (TTSError, "tts", "tts_error"),
        (ProviderContractViolationError, "translating", "provider_contract_violation"),
        (VideoDecodeError, "muxing", "video_decode_error"),
        (OriginalAudioExtractionError, "muxing", "original_audio_extraction_error"),
    ],
)
def test_subclass_has_expected_defaults(
    cls: type[SkillError], expected_stage: str, expected_code: str
) -> None:
    """Each subclass ships the design-mandated default stage and code."""
    err = cls("something went wrong")
    assert err.stage == expected_stage
    assert err.code == expected_code
    assert err.reason == "something went wrong"
    assert isinstance(err, SkillError)


# ---------------------------------------------------------------------------
# to_dict serialization and redaction
# ---------------------------------------------------------------------------


def test_to_dict_has_canonical_keys() -> None:
    """``to_dict`` exposes exactly code/stage/reason/context."""
    err = TranslationError(
        "failure",
        context={"entry_index": 2, "provider_type": "llm"},
    )
    payload = err.to_dict()
    assert set(payload.keys()) == {"code", "stage", "reason", "context"}
    assert payload == {
        "code": "translation_error",
        "stage": "translating",
        "reason": "failure",
        "context": {"entry_index": 2, "provider_type": "llm"},
    }


def test_to_dict_redacts_credential_variants() -> None:
    """Sensitive keys are redacted irrespective of case."""
    err = ProviderUnavailableError(
        "auth failed",
        context={
            "provider_type": "web",
            "credential": "super-secret",
            "API_KEY": "sk-abc",
            "Authorization": "Bearer xyz",
            "endpoint": "https://api.example.com",
        },
    )
    payload = err.to_dict()
    ctx = payload["context"]
    assert ctx["credential"] == "***"
    assert ctx["API_KEY"] == "***"
    assert ctx["Authorization"] == "***"
    # Non-sensitive keys are preserved verbatim
    assert ctx["provider_type"] == "web"
    assert ctx["endpoint"] == "https://api.example.com"


def test_to_dict_redacts_nested_structures() -> None:
    """Redaction walks into nested dicts and list/tuple values."""
    err = TranslationError(
        "upstream",
        context={
            "extra": {
                "credential": "sk-nested",
                "model": "gpt-x",
                "headers": {"Authorization": "Bearer y"},
            },
            "retries": [
                {"api_key": "leak-1", "status": 429},
                {"api_key": "leak-2", "status": 503},
            ],
        },
    )
    payload = err.to_dict()
    extra = payload["context"]["extra"]
    assert extra["credential"] == "***"
    assert extra["headers"]["Authorization"] == "***"
    assert extra["model"] == "gpt-x"
    retries = payload["context"]["retries"]
    assert retries[0] == {"api_key": "***", "status": 429}
    assert retries[1] == {"api_key": "***", "status": 503}


def test_to_dict_does_not_mutate_original_context() -> None:
    """Serialization is non-destructive; the original context is untouched."""
    err = ProviderUnavailableError(
        "x", context={"credential": "secret", "endpoint": "https://e"}
    )
    err.to_dict()
    assert err.context == {"credential": "secret", "endpoint": "https://e"}


def test_to_dict_context_is_always_a_dict() -> None:
    """Even with no context provided, ``context`` is an empty dict."""
    err = NoEnglishSubtitleError("no eng stream")
    payload = err.to_dict()
    assert payload["context"] == {}
    assert isinstance(payload["context"], dict)


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------


def test_module_exports_expected_public_api() -> None:
    """All promised subclasses are re-exported from ``errors``."""
    expected = {
        "SkillError",
        "SkillErrorStage",
        "ManifestParamMissingError",
        "UnsupportedProcessingModeError",
        "UnsupportedProviderTypeError",
        "VideoFileInaccessibleError",
        "SubtitleFileInaccessibleError",
        "UnsupportedVideoFormatError",
        "UnsupportedSubtitleFormatError",
        "SubtitleParseError",
        "InvalidTimestampError",
        "NoEnglishSubtitleError",
        "ProviderNotRegisteredError",
        "ProviderUnavailableError",
        "TranslationError",
        "TTSError",
        "ProviderContractViolationError",
        "VideoDecodeError",
        "OriginalAudioExtractionError",
    }
    assert expected.issubset(set(errors_module.__all__))
    for name in expected:
        assert hasattr(errors_module, name)
