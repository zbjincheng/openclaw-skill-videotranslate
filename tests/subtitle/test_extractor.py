"""Unit tests for :func:`extract_from_video`.

Covers requirements R2.6 (extract embedded English subtitle track) and
R2.7 (raise ``NoEnglishSubtitleError`` when none is available).

All tests mock :func:`subprocess.run` so no real ``ffmpeg`` invocation is
performed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from translation_dubbing_skill.errors import NoEnglishSubtitleError
from translation_dubbing_skill.subtitle import extract_from_video


def _completed(
    *,
    returncode: int,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> subprocess.CompletedProcess[bytes]:
    """Build a :class:`subprocess.CompletedProcess` for the mocked call."""
    return subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_extract_returns_srt_text_and_format() -> None:
    """A successful ffmpeg run returns ``(decoded_text, "srt")``."""
    srt_bytes = (
        b"1\n"
        b"00:00:01,000 --> 00:00:02,500\n"
        b"Hello, world!\n"
    )
    with patch("translation_dubbing_skill.subtitle.extractor.subprocess.run") as run:
        run.return_value = _completed(returncode=0, stdout=srt_bytes)
        text, fmt = extract_from_video(Path("sample.mkv"))

    assert fmt == "srt"
    assert "Hello, world!" in text
    assert text.startswith("1\n")


def test_extract_invokes_ffmpeg_with_english_subtitle_map() -> None:
    """The command passed to ``subprocess.run`` must select eng subs."""
    with patch("translation_dubbing_skill.subtitle.extractor.subprocess.run") as run:
        run.return_value = _completed(returncode=0, stdout=b"WEBVTT\n\n")
        extract_from_video(Path("sample.mkv"))

    assert run.call_count == 1
    args, kwargs = run.call_args
    cmd = args[0]
    # Critical flags: eng subtitle map, srt codec, pipe stdout.
    assert cmd[0] == "ffmpeg"
    assert "-map" in cmd
    assert "0:s:m:language:eng" in cmd
    assert "-c:s" in cmd and "srt" in cmd
    assert cmd[-1] == "-"
    # Must capture stderr/stdout and not block on stdin.
    assert kwargs.get("capture_output") is True
    assert "-nostdin" in cmd


def test_extract_decodes_utf8_output() -> None:
    """Non-ASCII subtitle text round-trips through UTF-8 decode."""
    payload = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "你好，世界 🌍\n"
    ).encode("utf-8")
    with patch("translation_dubbing_skill.subtitle.extractor.subprocess.run") as run:
        run.return_value = _completed(returncode=0, stdout=payload)
        text, fmt = extract_from_video(Path("sample.mkv"))

    assert "你好，世界 🌍" in text
    assert fmt == "srt"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_extract_raises_when_ffmpeg_returns_nonzero() -> None:
    """A non-zero exit code maps to ``NoEnglishSubtitleError``."""
    with patch("translation_dubbing_skill.subtitle.extractor.subprocess.run") as run:
        run.return_value = _completed(
            returncode=1,
            stdout=b"",
            stderr=b"Stream map '0:s:m:language:eng' matches no streams.\n",
        )
        with pytest.raises(NoEnglishSubtitleError) as excinfo:
            extract_from_video(Path("sample.mkv"))

    err = excinfo.value
    assert err.stage == "parsing"
    assert err.code == "no_english_subtitle"
    assert err.context["video_path"] == "sample.mkv"
    assert err.context["returncode"] == 1
    assert "matches no streams" in err.context["stderr_tail"]


def test_extract_raises_when_stdout_empty_even_if_returncode_zero() -> None:
    """An empty stdout is treated as "nothing extracted"."""
    with patch("translation_dubbing_skill.subtitle.extractor.subprocess.run") as run:
        run.return_value = _completed(returncode=0, stdout=b"", stderr=b"")
        with pytest.raises(NoEnglishSubtitleError):
            extract_from_video(Path("sample.mkv"))


def test_extract_raises_when_ffmpeg_not_installed() -> None:
    """``FileNotFoundError`` from ``subprocess.run`` is wrapped."""
    with patch("translation_dubbing_skill.subtitle.extractor.subprocess.run") as run:
        run.side_effect = FileNotFoundError(2, "No such file or directory", "ffmpeg")
        with pytest.raises(NoEnglishSubtitleError) as excinfo:
            extract_from_video(Path("sample.mkv"))

    err = excinfo.value
    assert err.context["video_path"] == "sample.mkv"
    assert "ffmpeg" in err.reason


def test_extract_raises_on_timeout() -> None:
    """A subprocess timeout maps to ``NoEnglishSubtitleError``."""
    def _raise_timeout(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1.0)

    with patch(
        "translation_dubbing_skill.subtitle.extractor.subprocess.run",
        side_effect=_raise_timeout,
    ):
        with pytest.raises(NoEnglishSubtitleError) as excinfo:
            extract_from_video(Path("sample.mkv"))

    assert "timed out" in excinfo.value.reason
    assert excinfo.value.context["video_path"] == "sample.mkv"
