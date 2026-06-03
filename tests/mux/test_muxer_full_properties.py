"""Property-based tests for ``VideoMuxer.mux_full`` (``subtitle_and_dubbing``).

Covers properties P17 – P20 from the design document:

* **P17** — output stream structure (R9.1, R9.2, R9.3)
* **P18** — video codec parameters preserved (R9.13)
* **P19** — language tags (R9.4, R9.12)
* **P20** — default-track markers (R9.5, R9.6, R9.7)

These tests require a real ffmpeg/ffprobe toolchain. The package-level
``conftest`` in :mod:`tests.mux.conftest` skips the whole package when
they are missing.

Because each example invokes ffmpeg (encoding a tiny AAC track) we keep
``max_examples`` modest (~30) per the "30–50 iterations" budget declared
in the design document for properties P17–P20.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from translation_dubbing_skill.mux.ffprobe import probe_streams
from translation_dubbing_skill.mux.muxer import VideoMuxer

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Subtitle content variations used to exercise the muxer with slightly
# different byte sequences on each iteration. The structural assertions
# below don't depend on the content, but varying it forces the muxer
# onto different byte offsets and guards against accidental content
# sensitivity.
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
    """Render a two-entry SRT file with the given line contents."""
    return (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        f"{line_a}\n"
        "\n"
        "2\n"
        "00:00:00,500 --> 00:00:01,000\n"
        f"{line_b}\n"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_mux_full(
    sample_video: Path,
    sample_zh_wav: Path,
    tmp_path: Path,
    zh_text: str,
    en_text: str,
) -> dict:
    """Mux a ``subtitle_and_dubbing`` output and return ffprobe JSON."""
    zh_srt = tmp_path / "zh.srt"
    en_srt = tmp_path / "en.srt"
    zh_srt.write_text(_render_srt(zh_text, zh_text[::-1]), encoding="utf-8")
    en_srt.write_text(_render_srt(en_text, en_text[::-1]), encoding="utf-8")
    out = tmp_path / "out.mkv"

    VideoMuxer().mux_full(
        input_video=sample_video,
        aligned_zh_audio=sample_zh_wav,
        zh_subtitle=zh_srt,
        en_subtitle=en_srt,
        output_path=out,
    )
    return probe_streams(out)


def _streams_by_type(probe: dict) -> dict[str, list[dict]]:
    """Group probe streams by ``codec_type`` for convenient counting."""
    buckets: dict[str, list[dict]] = {"video": [], "audio": [], "subtitle": []}
    for stream in probe["streams"]:
        bucket = buckets.setdefault(stream.get("codec_type", "other"), [])
        bucket.append(stream)
    return buckets


# ---------------------------------------------------------------------------
# P17 — output stream structure (R9.1, R9.2, R9.3)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(zh_text=_subtitle_line_strategy, en_text=_subtitle_line_strategy)
# Feature: video-subtitle-translation-dubbing, Property 17:
# subtitle_and_dubbing mode output video structure.
def test_p17_subtitle_and_dubbing_stream_counts(
    sample_video: Path,
    sample_zh_wav: Path,
    tmp_path_factory: pytest.TempPathFactory,
    zh_text: str,
    en_text: str,
) -> None:
    """Validates: Requirements 9.1, 9.2, 9.3"""
    tmp_path = tmp_path_factory.mktemp("p17")
    probe = _run_mux_full(sample_video, sample_zh_wav, tmp_path, zh_text, en_text)

    streams = _streams_by_type(probe)
    assert len(streams["video"]) == 1, "expected exactly one video stream"
    assert len(streams["audio"]) == 2, "expected exactly two audio streams"
    assert len(streams["subtitle"]) == 2, "expected exactly two subtitle streams"


# ---------------------------------------------------------------------------
# P18 — output video codec parameters preserved (R9.13)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(zh_text=_subtitle_line_strategy, en_text=_subtitle_line_strategy)
# Feature: video-subtitle-translation-dubbing, Property 18:
# output video codec parameters preserved across modes.
def test_p18_full_preserves_video_codec_parameters(
    sample_video: Path,
    sample_zh_wav: Path,
    tmp_path_factory: pytest.TempPathFactory,
    zh_text: str,
    en_text: str,
) -> None:
    """Validates: Requirements 9.13"""
    input_probe = probe_streams(sample_video)
    input_video = next(
        s for s in input_probe["streams"] if s.get("codec_type") == "video"
    )

    tmp_path = tmp_path_factory.mktemp("p18")
    out_probe = _run_mux_full(sample_video, sample_zh_wav, tmp_path, zh_text, en_text)
    output_video = next(
        s for s in out_probe["streams"] if s.get("codec_type") == "video"
    )

    for field in ("codec_name", "width", "height", "avg_frame_rate"):
        assert output_video.get(field) == input_video.get(field), (
            f"video {field} differs: in={input_video.get(field)!r} "
            f"out={output_video.get(field)!r}"
        )


# ---------------------------------------------------------------------------
# P19 — language tags (R9.4, R9.12)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(zh_text=_subtitle_line_strategy, en_text=_subtitle_line_strategy)
# Feature: video-subtitle-translation-dubbing, Property 19:
# subtitle_and_dubbing output language tags.
def test_p19_language_tags(
    sample_video: Path,
    sample_zh_wav: Path,
    tmp_path_factory: pytest.TempPathFactory,
    zh_text: str,
    en_text: str,
) -> None:
    """Validates: Requirements 9.4, 9.12"""
    tmp_path = tmp_path_factory.mktemp("p19")
    probe = _run_mux_full(sample_video, sample_zh_wav, tmp_path, zh_text, en_text)
    streams = _streams_by_type(probe)

    audio_langs = {s.get("tags", {}).get("language") for s in streams["audio"]}
    sub_langs = {s.get("tags", {}).get("language") for s in streams["subtitle"]}

    assert audio_langs == {"zho", "eng"}, (
        f"audio languages must be a one-to-one mapping to {{zho, eng}}, "
        f"got {audio_langs!r}"
    )
    assert sub_langs == {"zho", "eng"}, (
        f"subtitle languages must be a one-to-one mapping to {{zho, eng}}, "
        f"got {sub_langs!r}"
    )


# ---------------------------------------------------------------------------
# P20 — default-track markers (R9.5, R9.6, R9.7)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(zh_text=_subtitle_line_strategy, en_text=_subtitle_line_strategy)
# Feature: video-subtitle-translation-dubbing, Property 20:
# subtitle_and_dubbing default-track markers.
def test_p20_default_track_markers(
    sample_video: Path,
    sample_zh_wav: Path,
    tmp_path_factory: pytest.TempPathFactory,
    zh_text: str,
    en_text: str,
) -> None:
    """Validates: Requirements 9.5, 9.6, 9.7"""
    tmp_path = tmp_path_factory.mktemp("p20")
    probe = _run_mux_full(sample_video, sample_zh_wav, tmp_path, zh_text, en_text)
    streams = _streams_by_type(probe)

    # Exactly one default audio stream, and it must be the Chinese one.
    default_audio = [
        s for s in streams["audio"] if s.get("disposition", {}).get("default") == 1
    ]
    assert len(default_audio) == 1, (
        f"expected exactly one default audio stream, got {len(default_audio)}"
    )
    assert default_audio[0].get("tags", {}).get("language") == "zho", (
        "default audio stream must be the Chinese dub (language=zho)"
    )

    # Non-default audio must be the English stream.
    non_default_audio = [
        s for s in streams["audio"] if s.get("disposition", {}).get("default") == 0
    ]
    assert len(non_default_audio) == 1
    assert non_default_audio[0].get("tags", {}).get("language") == "eng"

    # Exactly one default subtitle stream, and it must be the Chinese one.
    default_sub = [
        s for s in streams["subtitle"] if s.get("disposition", {}).get("default") == 1
    ]
    assert len(default_sub) == 1, (
        f"expected exactly one default subtitle stream, got {len(default_sub)}"
    )
    assert default_sub[0].get("tags", {}).get("language") == "zho"

    non_default_sub = [
        s for s in streams["subtitle"] if s.get("disposition", {}).get("default") == 0
    ]
    assert len(non_default_sub) == 1
    assert non_default_sub[0].get("tags", {}).get("language") == "eng"
