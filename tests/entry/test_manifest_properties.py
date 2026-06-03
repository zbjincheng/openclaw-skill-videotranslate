"""Property-based tests for :func:`parse_manifest`.

Covers Tasks 11.3, 11.4, 11.5, 11.6, 11.7 and corresponding design
properties P11, P22, P23, P30, P31.

Each test follows the design convention:

    # Feature: video-subtitle-translation-dubbing,
    # Property {N}: {property_text}

and is tagged with ``**Validates: Requirements X.Y**`` in the docstring.

The fixtures share a tiny helper that builds a base manifest from a
tmp-path fixture, mirroring the example-based tests. The video / subtitle
files created are just marker bytes — :func:`parse_manifest` does not open
them, it only checks ``os.access`` and the file extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import assume, given, strategies as st

from translation_dubbing_skill.entry.manifest import (
    ALLOWED_PROCESSING_MODES,
    ALLOWED_PROVIDER_TYPES,
    DEFAULT_SUPPORTED_VIDEO_FORMATS,
    parse_manifest,
)
from translation_dubbing_skill.errors import (
    ManifestParamMissingError,
    UnsupportedProcessingModeError,
    UnsupportedProviderTypeError,
    UnsupportedSubtitleFormatError,
    UnsupportedVideoFormatError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_params(tmp_path: Path) -> dict[str, Any]:
    """Return a minimal valid manifest dict backed by real files.

    The fixture creates an ``.mp4`` marker file so the accessibility
    check passes; individual tests mutate this dict to exercise a single
    failure path at a time.
    """
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake-video")
    return {
        "video_path": str(video),
        "target_language": "zh-CN",
        "processing_mode": "subtitle_and_dubbing",
        "translation_provider": "llm",
        "translation_endpoint": "https://example.com/translate",
        "translation_credential": "tkey",
        "tts_provider": "web",
        "tts_endpoint": "https://example.com/tts",
        "tts_credential": "vkey",
    }


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# A non-empty printable ASCII string. Good enough as a "provider type
# identifier" stand-in — the schema only accepts ``{llm, web}`` so any
# other printable value should be rejected.
_printable_ascii_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0x7E,
        blacklist_categories=("Cs",),
    ),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "")

# Alphanumeric-ish stems used when synthesizing fake file paths. Avoid
# path separators and dots — we add the extension ourselves.
_path_stem = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=16,
)

# Common "definitely-not-video" extensions to seed the generator with so
# Hypothesis explores the intended boundary (unsupported extension ⇒
# error) quickly; it will also generate random strings around them.
_non_video_ext = st.one_of(
    st.sampled_from(["avi", "flv", "wmv", "ts", "mpeg", "mpg", "ogv", "3gp"]),
    _path_stem,
).filter(
    lambda s: s.lower().lstrip(".") not in set(DEFAULT_SUPPORTED_VIDEO_FORMATS)
    and s != ""
)

_non_subtitle_ext = st.one_of(
    st.sampled_from(["ass", "ssa", "sub", "txt", "sbv", "idx"]),
    _path_stem,
).filter(lambda s: s.lower().lstrip(".") not in {"srt", "vtt"} and s != "")

_invalid_mode_text = _printable_ascii_text.filter(
    lambda s: s not in ALLOWED_PROCESSING_MODES
)

_invalid_provider_text = _printable_ascii_text.filter(
    lambda s: s not in ALLOWED_PROVIDER_TYPES
)


# ---------------------------------------------------------------------------
# Property 11 — unsupported provider type (Task 11.3)
# ---------------------------------------------------------------------------


# Feature: video-subtitle-translation-dubbing,
# Property 11: unsupported provider type error.
@given(bad_type=_invalid_provider_text, kind=st.sampled_from(["translation", "tts"]))
def test_property_11_unsupported_provider_type_for_translation_and_tts(
    bad_type: str,
    kind: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Property 11 — any provider_type outside the enum ``{llm, web}`` raises.

    Validates: Requirements 1.12

    Exercised for both the ``translation_provider`` and ``tts_provider``
    fields (the latter only matters under ``subtitle_and_dubbing`` since
    ``subtitle_only`` ignores TTS fields entirely per R1.7).
    """
    tmp_path = tmp_path_factory.mktemp("p11")
    params = _base_params(tmp_path)

    if kind == "translation":
        params["translation_provider"] = bad_type
    else:  # "tts"
        params["tts_provider"] = bad_type

    with pytest.raises(UnsupportedProviderTypeError) as exc_info:
        parse_manifest(params)

    ctx = exc_info.value.context
    assert ctx["requested_type"] == bad_type
    # The error message and context MUST list the allowed values for the
    # current kind so the caller can tell the user what's accepted
    # (R1.12). Translation is ``{llm, web}``; TTS additionally accepts
    # ``edge`` for Microsoft's free Read-Aloud service.
    if kind == "translation":
        expected_allowed = ["llm", "web"]
    else:
        expected_allowed = ["llm", "web", "edge", "minimax"]
    assert ctx["allowed_types"] == expected_allowed
    assert ctx["kind"] == kind
    assert "llm" in str(exc_info.value) and "web" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Property 22 — unsupported video format (Task 11.4)
# ---------------------------------------------------------------------------


# Feature: video-subtitle-translation-dubbing,
# Property 22: unsupported video format error.
@given(stem=_path_stem, ext=_non_video_ext)
def test_property_22_unsupported_video_format_error(
    stem: str,
    ext: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Property 22 — any extension outside ``supported_video_formats`` raises.

    Validates: Requirements 10.3

    The error's context must carry the supported format list so the
    caller can display it to the end user. The test synthesizes a path
    that does NOT exist on disk — the format check fires before the
    accessibility check, so this is sufficient to observe the intended
    error.
    """
    # Filter guard: Hypothesis' filter already removes colliding exts,
    # but double-check the concrete value because ``_non_video_ext``
    # leans on a ``_path_stem`` shrinker that can occasionally produce a
    # supported extension after filtering is applied.
    assume(ext.lower().lstrip(".") not in set(DEFAULT_SUPPORTED_VIDEO_FORMATS))
    assume(ext != "")

    tmp_path = tmp_path_factory.mktemp("p22")
    params = _base_params(tmp_path)
    # Use a path that doesn't exist so we're only observing the format
    # check; the format check fires first by design.
    params["video_path"] = str(tmp_path / f"{stem}.{ext}")

    with pytest.raises(UnsupportedVideoFormatError) as exc_info:
        parse_manifest(params)

    ctx = exc_info.value.context
    assert ctx["given_ext"] == ext.lower().lstrip(".")
    assert set(ctx["supported"]) == set(DEFAULT_SUPPORTED_VIDEO_FORMATS)


# ---------------------------------------------------------------------------
# Property 23 — unsupported subtitle format (Task 11.5)
# ---------------------------------------------------------------------------


# Feature: video-subtitle-translation-dubbing,
# Property 23: unsupported subtitle format error.
@given(stem=_path_stem, ext=_non_subtitle_ext)
def test_property_23_unsupported_subtitle_format_error(
    stem: str,
    ext: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Property 23 — any subtitle extension other than ``.srt``/``.vtt`` raises.

    Validates: Requirements 10.4
    """
    assume(ext.lower().lstrip(".") not in {"srt", "vtt"})
    assume(ext != "")

    tmp_path = tmp_path_factory.mktemp("p23")
    params = _base_params(tmp_path)
    params["subtitle_path"] = str(tmp_path / f"{stem}.{ext}")

    with pytest.raises(UnsupportedSubtitleFormatError) as exc_info:
        parse_manifest(params)

    ctx = exc_info.value.context
    assert ctx["given_ext"] == ext.lower().lstrip(".")


# ---------------------------------------------------------------------------
# Property 30 — TTS fields required under subtitle_and_dubbing (Task 11.6)
# ---------------------------------------------------------------------------


_TTS_REQUIRED_FIELDS: tuple[str, ...] = (
    "tts_provider",
    "tts_endpoint",
    "tts_credential",
)


# Feature: video-subtitle-translation-dubbing,
# Property 30: subtitle_and_dubbing mode requires TTS fields.
@given(
    # Any non-empty subset of the three TTS fields to drop / blank.
    dropped=st.lists(
        st.sampled_from(_TTS_REQUIRED_FIELDS), min_size=1, max_size=3, unique=True
    ),
    # How to make the field "missing" for each index — drop it entirely
    # vs. set it to ``None`` vs. set to an empty / whitespace string.
    blank_strategies=st.lists(
        st.sampled_from(["delete", "none", "empty", "whitespace"]),
        min_size=3,
        max_size=3,
    ),
)
def test_property_30_tts_fields_required_under_dubbing(
    dropped: list[str],
    blank_strategies: list[str],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Property 30 — any missing/blank TTS field under dubbing raises.

    Validates: Requirements 1.6, 1.10

    For each TTS field we pick either to delete it from the manifest or
    to blank it (None / empty string / whitespace). All of these must
    surface as :class:`ManifestParamMissingError` with the offending
    field listed in ``context.missing_fields``.
    """
    tmp_path = tmp_path_factory.mktemp("p30")
    params = _base_params(tmp_path)
    assert params["processing_mode"] == "subtitle_and_dubbing"

    # Apply the "make this field missing" transformation for each
    # dropped name, picking the per-field strategy by position.
    blanks = {
        field: strategy
        for field, strategy in zip(_TTS_REQUIRED_FIELDS, blank_strategies)
    }
    for field in dropped:
        strategy = blanks[field]
        if strategy == "delete":
            params.pop(field, None)
        elif strategy == "none":
            params[field] = None
        elif strategy == "empty":
            params[field] = ""
        else:  # "whitespace"
            params[field] = "   \t"

    with pytest.raises(ManifestParamMissingError) as exc_info:
        parse_manifest(params)

    missing_in_ctx = exc_info.value.context["missing_fields"]
    # The missing_fields list must at least cover the fields we blanked.
    # It may list more if we blanked multiple — implementation rolls up
    # all blanks together under a single error.
    for field in dropped:
        assert field in missing_in_ctx, (
            f"expected {field!r} in missing_fields but got {missing_in_ctx!r}"
        )
    # processing_mode is preserved in context for downstream routing.
    assert exc_info.value.context["processing_mode"] == "subtitle_and_dubbing"


# ---------------------------------------------------------------------------
# Property 31 — invalid processing mode (Task 11.7)
# ---------------------------------------------------------------------------


# Feature: video-subtitle-translation-dubbing,
# Property 31: illegal processing_mode value triggers error.
@given(bad_mode=_invalid_mode_text)
def test_property_31_illegal_processing_mode(
    bad_mode: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Property 31 — any ``processing_mode`` outside the enum raises.

    Validates: Requirements 1.11

    The raised :class:`UnsupportedProcessingModeError` must expose both
    the requested mode and the full set of allowed modes so the caller
    can surface them to the end user.
    """
    tmp_path = tmp_path_factory.mktemp("p31")
    params = _base_params(tmp_path)
    params["processing_mode"] = bad_mode

    with pytest.raises(UnsupportedProcessingModeError) as exc_info:
        parse_manifest(params)

    ctx = exc_info.value.context
    assert ctx["requested_mode"] == bad_mode
    assert ctx["allowed_modes"] == ["subtitle_only", "subtitle_and_dubbing"]
