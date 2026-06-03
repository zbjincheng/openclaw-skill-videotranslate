"""Example-based tests for :func:`parse_manifest`.

Complements the property-based tests 11.3–11.7 with hand-crafted cases
that exercise the success path and each explicit failure mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from translation_dubbing_skill.entry.manifest import (
    DEFAULT_SUPPORTED_VIDEO_FORMATS,
    ManifestParams,
    parse_manifest,
)
from translation_dubbing_skill.errors import (
    ManifestParamMissingError,
    SubtitleFileInaccessibleError,
    UnsupportedProcessingModeError,
    UnsupportedProviderTypeError,
    UnsupportedSubtitleFormatError,
    UnsupportedVideoFormatError,
    VideoFileInaccessibleError,
)
from translation_dubbing_skill.models import ProcessingMode


def _base_params(tmp_path: Path) -> dict[str, Any]:
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


def test_parse_manifest_happy_path(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    parsed = parse_manifest(params)

    assert isinstance(parsed, ManifestParams)
    assert parsed.processing_mode is ProcessingMode.SUBTITLE_AND_DUBBING
    assert parsed.translation_provider == "llm"
    assert parsed.tts_provider == "web"
    assert parsed.tts_rate_limit is not None
    # Defaults populated from the schema.
    assert parsed.translation_rate_limit.batch_size_initial == 20
    assert parsed.tts_rate_limit.supports_batch is False
    assert parsed.supported_video_formats == list(DEFAULT_SUPPORTED_VIDEO_FORMATS)


def test_parse_manifest_default_processing_mode(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    del params["processing_mode"]

    parsed = parse_manifest(params)
    assert parsed.processing_mode is ProcessingMode.SUBTITLE_AND_DUBBING


def test_parse_manifest_subtitle_only_ignores_tts(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    params["processing_mode"] = "subtitle_only"
    # Keep TTS fields populated — they should be ignored.
    params["voice_id"] = "nova"

    parsed = parse_manifest(params)
    assert parsed.processing_mode is ProcessingMode.SUBTITLE_ONLY
    assert parsed.tts_provider is None
    assert parsed.tts_endpoint is None
    assert parsed.tts_credential is None
    assert parsed.tts_rate_limit is None
    assert parsed.voice_id is None


def test_parse_manifest_missing_tts_under_dubbing(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    del params["tts_provider"]
    del params["tts_endpoint"]

    with pytest.raises(ManifestParamMissingError) as exc_info:
        parse_manifest(params)
    missing = exc_info.value.context["missing_fields"]
    assert "tts_provider" in missing
    assert "tts_endpoint" in missing


def test_parse_manifest_invalid_processing_mode(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    params["processing_mode"] = "mystery"

    with pytest.raises(UnsupportedProcessingModeError) as exc_info:
        parse_manifest(params)
    ctx = exc_info.value.context
    assert ctx["requested_mode"] == "mystery"
    assert ctx["allowed_modes"] == ["subtitle_only", "subtitle_and_dubbing"]


def test_parse_manifest_invalid_provider_type(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    params["translation_provider"] = "unknown_thing"

    with pytest.raises(UnsupportedProviderTypeError) as exc_info:
        parse_manifest(params)
    ctx = exc_info.value.context
    assert ctx["requested_type"] == "unknown_thing"
    assert ctx["allowed_types"] == ["llm", "web"]
    assert ctx["kind"] == "translation"


def test_parse_manifest_unsupported_video_format(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    params = _base_params(tmp_path)
    params["video_path"] = str(video)

    with pytest.raises(UnsupportedVideoFormatError) as exc_info:
        parse_manifest(params)
    assert exc_info.value.context["given_ext"] == "avi"
    assert set(exc_info.value.context["supported"]) == set(DEFAULT_SUPPORTED_VIDEO_FORMATS)


def test_parse_manifest_unsupported_subtitle_format(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    bad = tmp_path / "sub.ass"
    bad.write_text("whatever", encoding="utf-8")
    params["subtitle_path"] = str(bad)

    with pytest.raises(UnsupportedSubtitleFormatError):
        parse_manifest(params)


def test_parse_manifest_video_inaccessible(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    params["video_path"] = str(tmp_path / "missing.mp4")

    with pytest.raises(VideoFileInaccessibleError):
        parse_manifest(params)


def test_parse_manifest_subtitle_inaccessible(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    params["subtitle_path"] = str(tmp_path / "missing.srt")

    with pytest.raises(SubtitleFileInaccessibleError):
        parse_manifest(params)


def test_parse_manifest_merges_rate_limit_partial_override(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    params["translation_rate_limit"] = {"batch_size_initial": 5, "batch_size_min": 1, "batch_size_max": 10}

    parsed = parse_manifest(params)
    # Overridden field.
    assert parsed.translation_rate_limit.batch_size_initial == 5
    # Non-overridden field retains its default.
    assert parsed.translation_rate_limit.payload_unit == "tokens"


def test_parse_manifest_rate_limit_invalid_triple(tmp_path: Path) -> None:
    params = _base_params(tmp_path)
    params["translation_rate_limit"] = {
        "batch_size_initial": 100,
        "batch_size_min": 1,
        "batch_size_max": 10,  # initial > max
    }
    with pytest.raises(ManifestParamMissingError):
        parse_manifest(params)
