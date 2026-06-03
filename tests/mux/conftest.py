"""Shared fixtures for muxer property tests.

Many of the properties under ``tests/mux`` assert on the *structure* of
a real ``.mkv`` file — number of streams, language tags, default
dispositions, codec parameters. Building those assertions requires:

1. A small sample ``.mkv`` to act as the input video. We generate one
   lazily at session scope via ``ffmpeg -f lavfi`` so the test suite
   does not need to ship binary artifacts.
2. Small sample subtitle and audio files.
3. A helper that runs :class:`VideoMuxer` and probes the output.

If ``ffmpeg`` / ``ffprobe`` are not available on ``PATH`` the entire
``tests/mux/`` package is skipped, keeping the suite green on minimal
environments (R-agnostic — the design explicitly permits skipping here).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Skip every test in this directory if the ffmpeg toolchain is missing.
# Using ``pytest.skip(..., allow_module_level=True)`` inside ``conftest``
# propagates to all collected tests.
if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:  # pragma: no cover
    pytest.skip(
        "ffmpeg/ffprobe not available; skipping muxer property tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Sample asset builders
# ---------------------------------------------------------------------------


def _build_sample_video(path: Path, duration_s: int = 1) -> None:
    """Write a tiny h.264 + AAC MP4 at ``path``.

    Uses ``lavfi`` sources so nothing hits the filesystem beyond the
    output file. Keeping the clip at 32×24 @ 1 fps gives us a ~1 KB
    video whose mux-time is dominated by process startup, not encoding.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f", "lavfi", "-i", f"color=c=black:s=32x24:r=1:d={duration_s}",
        "-f", "lavfi", "-i", f"anullsrc=r=8000:cl=mono:d={duration_s}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:  # pragma: no cover
        raise RuntimeError(
            "failed to build sample video: "
            + result.stderr.decode("utf-8", errors="replace")
        )


def _write_sample_srt(path: Path, language_hint: str) -> None:
    """Write a minimal 2-entry SRT file.

    ``language_hint`` only influences the subtitle text so human readers
    can distinguish the Chinese and English subtitle files during
    debugging — muxer behaviour does not depend on content.
    """
    content = (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        f"{language_hint} line 1\n"
        "\n"
        "2\n"
        "00:00:00,500 --> 00:00:01,000\n"
        f"{language_hint} line 2\n"
    )
    path.write_text(content, encoding="utf-8")


def _build_silent_wav(path: Path, duration_s: int = 1) -> None:
    """Write a short, low-level tone WAV for use as the aligned Chinese audio.

    A real dub clip has audio content; using ``anullsrc`` here produces a
    pure-silence WAV, which :func:`VideoMuxer.mux_full` then hands to
    the AAC encoder after ``loudnorm``. Loudnorm on pure silence yields
    NaN/Inf and AAC refuses to encode it. A gentle sine tone keeps the
    property tests fast and deterministic while exercising the
    loudnorm → AAC path end-to-end.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={duration_s}",
        "-af", "volume=0.2",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:  # pragma: no cover
        raise RuntimeError(
            "failed to build sample wav: "
            + result.stderr.decode("utf-8", errors="replace")
        )


# ---------------------------------------------------------------------------
# Session-scoped inputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a session-cached tiny MP4 sample video."""
    path = tmp_path_factory.mktemp("mux-inputs") / "sample.mp4"
    _build_sample_video(path)
    return path


@pytest.fixture(scope="session")
def sample_zh_srt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mux-inputs") / "zh.srt"
    _write_sample_srt(path, "中文")
    return path


@pytest.fixture(scope="session")
def sample_en_srt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mux-inputs") / "en.srt"
    _write_sample_srt(path, "English")
    return path


@pytest.fixture(scope="session")
def sample_zh_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mux-inputs") / "zh.wav"
    _build_silent_wav(path)
    return path


# ---------------------------------------------------------------------------
# Per-test output path
# ---------------------------------------------------------------------------


@pytest.fixture
def output_mkv(tmp_path: Path) -> Path:
    """Return a per-test ``.mkv`` output path."""
    return tmp_path / "out.mkv"
