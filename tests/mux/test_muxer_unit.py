"""Unit tests for :class:`VideoMuxer` using an injected runner.

These tests don't invoke real ffmpeg — they stub the runner to return a
pre-canned :class:`subprocess.CompletedProcess`. They cover:

* Command assembly for both ``mux_full`` and ``mux_subtitle_only``.
* Error classification (``VideoDecodeError`` / ``OriginalAudioExtractionError``).
* Per-call runner override.

Unlike the property tests in the sibling modules, these tests do NOT
require ffmpeg/ffprobe, so they run on every environment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from translation_dubbing_skill.errors import (
    OriginalAudioExtractionError,
    VideoDecodeError,
)
from translation_dubbing_skill.mux.muxer import VideoMuxer


def _make_fake_runner(
    *,
    returncode: int = 0,
    stderr: bytes = b"",
    output_contents: bytes = b"FAKE_MKV",
    captured: list[Sequence[str]] | None = None,
):
    """Return a runner callable that fabricates a ``.mkv``-like output file.

    The runner records every invocation into ``captured`` (when given) so
    tests can inspect the assembled argv. The output path argument (last
    entry in the argv list) may be rendered with a ``file:`` prefix —
    we strip that before writing because ``Path`` doesn't understand
    ffmpeg's protocol-URI form.
    """

    def runner(cmd: Sequence[str]) -> "subprocess.CompletedProcess[bytes]":
        if captured is not None:
            captured.append(list(cmd))
        # Last argv element is the output path by convention. ffmpeg's
        # file: protocol prefix (added by the production code to avoid
        # mis-parsing filenames with ``[``, ``%`` etc.) is stripped
        # here so the stub can open the real path.
        raw_output = cmd[-1]
        if isinstance(raw_output, str) and raw_output.startswith("file:"):
            raw_output = raw_output[len("file:") :]
        out_path = Path(raw_output)
        out_path.write_bytes(output_contents)
        return subprocess.CompletedProcess(
            args=list(cmd), returncode=returncode, stdout=b"", stderr=stderr
        )

    return runner


# ---------------------------------------------------------------------------
# mux_full command assembly
# ---------------------------------------------------------------------------


def test_mux_full_assembles_expected_command(tmp_path: Path) -> None:
    captured: list[Sequence[str]] = []
    muxer = VideoMuxer(runner=_make_fake_runner(captured=captured))

    inputs = {
        "input_video": tmp_path / "in.mp4",
        "aligned_zh_audio": tmp_path / "zh.wav",
        "zh_subtitle": tmp_path / "zh.srt",
        "en_subtitle": tmp_path / "en.srt",
        "output_path": tmp_path / "out.mkv",
    }
    for path in inputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".mkv":
            path.write_bytes(b"stub")

    result = muxer.mux_full(**inputs)

    assert result == inputs["output_path"]
    assert result.exists(), "runner stub should have created the output file"
    assert len(captured) == 1
    cmd = captured[0]

    # Stream mapping in the expected order.
    map_args = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-map"]
    assert map_args == ["0:v:0", "1:a:0", "0:a:0", "2:s:0", "3:s:0"]

    # Chinese audio is re-encoded to AAC; English audio is copied.
    assert "-c:a:0" in cmd and cmd[cmd.index("-c:a:0") + 1] == "aac"
    assert "-c:a:1" in cmd and cmd[cmd.index("-c:a:1") + 1] == "copy"

    # Chinese is default on both the audio and subtitle side.
    assert "-disposition:a:0" in cmd
    assert cmd[cmd.index("-disposition:a:0") + 1] == "default"
    assert "-disposition:s:0" in cmd
    assert cmd[cmd.index("-disposition:s:0") + 1] == "default"


# ---------------------------------------------------------------------------
# mux_subtitle_only command assembly
# ---------------------------------------------------------------------------


def test_mux_subtitle_only_assembles_expected_command(tmp_path: Path) -> None:
    captured: list[Sequence[str]] = []
    muxer = VideoMuxer(runner=_make_fake_runner(captured=captured))

    inputs = {
        "input_video": tmp_path / "in.mp4",
        "zh_subtitle": tmp_path / "zh.srt",
        "en_subtitle": tmp_path / "en.srt",
        "output_path": tmp_path / "out.mkv",
    }
    for path in inputs.values():
        if path.suffix != ".mkv":
            path.write_bytes(b"stub")

    result = muxer.mux_subtitle_only(**inputs)

    assert result == inputs["output_path"]
    assert result.exists()
    cmd = captured[0]

    # No second audio input; all streams come from the input video or
    # the subtitle files.
    map_args = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-map"]
    assert map_args == ["0:v:0", "0:a:0", "1:s:0", "2:s:0"]

    # Video + audio are copied; no AAC re-encode.
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-c:a:0" not in cmd  # no per-index audio codec args

    # English audio is default; Chinese subtitle is default.
    assert cmd[cmd.index("-disposition:a:0") + 1] == "default"
    assert cmd[cmd.index("-disposition:s:0") + 1] == "default"
    assert cmd[cmd.index("-disposition:s:1") + 1] == "0"


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def test_video_decode_failure_raises_video_decode_error(tmp_path: Path) -> None:
    runner = _make_fake_runner(
        returncode=1,
        stderr=b"Invalid data found when processing input\n",
    )
    muxer = VideoMuxer(runner=runner)
    with pytest.raises(VideoDecodeError) as info:
        muxer.mux_subtitle_only(
            input_video=tmp_path / "in.mp4",
            zh_subtitle=tmp_path / "zh.srt",
            en_subtitle=tmp_path / "en.srt",
            output_path=tmp_path / "out.mkv",
        )
    assert info.value.stage == "muxing"
    assert info.value.context["returncode"] == 1


def test_missing_audio_raises_original_audio_extraction_error(tmp_path: Path) -> None:
    runner = _make_fake_runner(
        returncode=1,
        stderr=b"Stream map '0:a:0' matches no streams. Does not contain any stream\n",
    )
    muxer = VideoMuxer(runner=runner)
    with pytest.raises(OriginalAudioExtractionError):
        muxer.mux_full(
            input_video=tmp_path / "in.mp4",
            aligned_zh_audio=tmp_path / "zh.wav",
            zh_subtitle=tmp_path / "zh.srt",
            en_subtitle=tmp_path / "en.srt",
            output_path=tmp_path / "out.mkv",
        )


def test_unknown_failure_raises_runtime_error(tmp_path: Path) -> None:
    runner = _make_fake_runner(returncode=1, stderr=b"something weird\n")
    muxer = VideoMuxer(runner=runner)
    with pytest.raises(RuntimeError):
        muxer.mux_subtitle_only(
            input_video=tmp_path / "in.mp4",
            zh_subtitle=tmp_path / "zh.srt",
            en_subtitle=tmp_path / "en.srt",
            output_path=tmp_path / "out.mkv",
        )


# ---------------------------------------------------------------------------
# Per-call runner override
# ---------------------------------------------------------------------------


def test_per_call_runner_override(tmp_path: Path) -> None:
    instance_calls: list[Sequence[str]] = []
    override_calls: list[Sequence[str]] = []

    muxer = VideoMuxer(runner=_make_fake_runner(captured=instance_calls))
    override = _make_fake_runner(captured=override_calls)

    muxer.mux_subtitle_only(
        input_video=tmp_path / "in.mp4",
        zh_subtitle=tmp_path / "zh.srt",
        en_subtitle=tmp_path / "en.srt",
        output_path=tmp_path / "out.mkv",
        runner=override,
    )
    assert len(instance_calls) == 0
    assert len(override_calls) == 1
