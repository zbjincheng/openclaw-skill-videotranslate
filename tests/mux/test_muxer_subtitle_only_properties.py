"""Property-based tests for ``VideoMuxer.mux_subtitle_only``.

Covers properties P27 – P29 from the design document, plus the
``subtitle_only`` half of P18 (video codec parameters preserved):

* **P27** — output stream structure for ``subtitle_only`` (R9.8, R9.10)
* **P28** — audio language tag & default flag (R9.9, R9.12)
* **P29** — subtitle language tags & default flags (R9.10, R9.11, R9.12)
* **P18 (subtitle_only leg)** — video codec parameters preserved (R9.13)

Like :mod:`tests.mux.test_muxer_full_properties` these tests require a
real ffmpeg/ffprobe toolchain, and are skipped wholesale by the package
``conftest`` when the binaries are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from translation_dubbing_skill.mux.ffprobe import probe_streams
from translation_dubbing_skill.mux.muxer import VideoMuxer

_subtitle_line_strategy = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs", "Cc"),
        min_codepoint=0x20,
        max_codepoint=0x4E00 + 500,
    ),
    min_size=1,
    max_size=40,
)


def _render_srt(line_a: str, line_b: str) -> str:
    return (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        f"{line_a}\n"
        "\n"
        "2\n"
        "00:00:00,500 --> 00:00:01,000\n"
        f"{line_b}\n"
    )


def _run_mux_subtitle_only(
    sample_video: Path,
    tmp_path: Path,
    zh_text: str,
    en_text: str,
) -> dict:
    zh_srt = tmp_path / "zh.srt"
    en_srt = tmp_path / "en.srt"
    zh_srt.write_text(_render_srt(zh_text, zh_text[::-1]), encoding="utf-8")
    en_srt.write_text(_render_srt(en_text, en_text[::-1]), encoding="utf-8")
    out = tmp_path / "out.mkv"

    VideoMuxer().mux_subtitle_only(
        input_video=sample_video,
        zh_subtitle=zh_srt,
        en_subtitle=en_srt,
        output_path=out,
    )
    return probe_streams(out)


def _streams_by_type(probe: dict) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"video": [], "audio": [], "subtitle": []}
    for stream in probe["streams"]:
        bucket = buckets.setdefault(stream.get("codec_type", "other"), [])
        bucket.append(stream)
    return buckets


# ---------------------------------------------------------------------------
# P18 (subtitle_only leg) — codec parameters preserved (R9.13)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(zh_text=_subtitle_line_strategy, en_text=_subtitle_line_strategy)
# Feature: video-subtitle-translation-dubbing, Property 18:
# output video codec parameters preserved (subtitle_only leg).
def test_p18_subtitle_only_preserves_video_codec_parameters(
    sample_video: Path,
    tmp_path_factory: pytest.TempPathFactory,
    zh_text: str,
    en_text: str,
) -> None:
    """Validates: Requirements 9.13"""
    input_probe = probe_streams(sample_video)
    input_video = next(
        s for s in input_probe["streams"] if s.get("codec_type") == "video"
    )

    tmp_path = tmp_path_factory.mktemp("p18-so")
    out_probe = _run_mux_subtitle_only(sample_video, tmp_path, zh_text, en_text)
    output_video = next(
        s for s in out_probe["streams"] if s.get("codec_type") == "video"
    )

    for field in ("codec_name", "width", "height", "avg_frame_rate"):
        assert output_video.get(field) == input_video.get(field), (
            f"video {field} differs: in={input_video.get(field)!r} "
            f"out={output_video.get(field)!r}"
        )


# ---------------------------------------------------------------------------
# P27 — subtitle_only stream structure (R9.8, R9.10)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(zh_text=_subtitle_line_strategy, en_text=_subtitle_line_strategy)
# Feature: video-subtitle-translation-dubbing, Property 27:
# subtitle_only mode output video stream structure.
def test_p27_subtitle_only_stream_counts(
    sample_video: Path,
    tmp_path_factory: pytest.TempPathFactory,
    zh_text: str,
    en_text: str,
) -> None:
    """Validates: Requirements 9.8, 9.10"""
    tmp_path = tmp_path_factory.mktemp("p27")
    probe = _run_mux_subtitle_only(sample_video, tmp_path, zh_text, en_text)

    streams = _streams_by_type(probe)
    assert len(streams["video"]) == 1, "expected exactly one video stream"
    assert len(streams["audio"]) == 1, "expected exactly one audio stream (English)"
    assert len(streams["subtitle"]) == 2, "expected exactly two subtitle streams"


# ---------------------------------------------------------------------------
# P28 — subtitle_only audio language + default flag (R9.9, R9.12)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(zh_text=_subtitle_line_strategy, en_text=_subtitle_line_strategy)
# Feature: video-subtitle-translation-dubbing, Property 28:
# subtitle_only mode output audio language and default flag.
def test_p28_subtitle_only_audio_lang_and_default(
    sample_video: Path,
    tmp_path_factory: pytest.TempPathFactory,
    zh_text: str,
    en_text: str,
) -> None:
    """Validates: Requirements 9.9, 9.12"""
    tmp_path = tmp_path_factory.mktemp("p28")
    probe = _run_mux_subtitle_only(sample_video, tmp_path, zh_text, en_text)
    streams = _streams_by_type(probe)

    assert len(streams["audio"]) == 1
    audio = streams["audio"][0]
    assert audio.get("tags", {}).get("language") == "eng"
    assert audio.get("disposition", {}).get("default") == 1


# ---------------------------------------------------------------------------
# P29 — subtitle_only subtitle tracks (R9.10, R9.11, R9.12)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(zh_text=_subtitle_line_strategy, en_text=_subtitle_line_strategy)
# Feature: video-subtitle-translation-dubbing, Property 29:
# subtitle_only mode output subtitle tracks — language and default flag.
def test_p29_subtitle_only_subtitle_tracks(
    sample_video: Path,
    tmp_path_factory: pytest.TempPathFactory,
    zh_text: str,
    en_text: str,
) -> None:
    """Validates: Requirements 9.10, 9.11, 9.12"""
    tmp_path = tmp_path_factory.mktemp("p29")
    probe = _run_mux_subtitle_only(sample_video, tmp_path, zh_text, en_text)
    streams = _streams_by_type(probe)

    # Exactly two subtitle streams, languages are a one-to-one mapping
    # onto {zho, eng}.
    assert len(streams["subtitle"]) == 2
    langs = {s.get("tags", {}).get("language") for s in streams["subtitle"]}
    assert langs == {"zho", "eng"}

    # Chinese subtitle is default; English is non-default.
    by_lang = {
        s.get("tags", {}).get("language"): s for s in streams["subtitle"]
    }
    assert by_lang["zho"].get("disposition", {}).get("default") == 1
    assert by_lang["eng"].get("disposition", {}).get("default") == 0
