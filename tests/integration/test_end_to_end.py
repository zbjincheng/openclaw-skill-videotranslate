"""End-to-end integration tests for :func:`translation_dubbing_skill.run` (tasks 14.4 / 14.5).

The skill's entry point exposes every external collaborator as a keyword
argument: parser, translator, tts engine, aligner, muxer, duration
probe, and subtitle reader/extractor. Both tests below exploit that
hook to run the *real* orchestration glue (mode dispatch, subtitle
serialization, progress reporting) while substituting stubs for every
component that would otherwise require ``ffmpeg`` or a live HTTP
provider.

- **Task 14.4** — ``subtitle_and_dubbing`` end-to-end: asserts the
  returned ``output_video_path`` exists, the ``output_subtitle_path``
  is UTF-8 simplified Chinese, and the TTS+aligner+mux_full path was
  taken exactly once.

- **Task 14.5** — ``subtitle_only`` end-to-end with garbage/missing
  ``tts_*`` fields: asserts execution doesn't fail on the invalid TTS
  params (R1.7, P26) and the subtitle_only path is taken.

The fixtures live entirely on the filesystem under ``tmp_path`` so
parallel test runs don't interfere.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from translation_dubbing_skill import run
from translation_dubbing_skill.entry.manifest import ManifestParams
from translation_dubbing_skill.models import (
    AudioClip,
    ProcessingMode,
    ProviderConfig,
    SubtitleEntry,
)
from translation_dubbing_skill.progress.reporter import InMemoryReporter
from translation_dubbing_skill.scheduler.config import ProviderRateLimitConfig


# ---------------------------------------------------------------------------
# SRT fixture + helpers
# ---------------------------------------------------------------------------


def _format_ts(ms: int) -> str:
    """Format milliseconds as ``HH:MM:SS,mmm`` for SRT output."""
    hours, r = divmod(ms, 3_600_000)
    minutes, r = divmod(r, 60_000)
    seconds, millis = divmod(r, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _write_sample_srt(path: Path) -> None:
    """Write a small English SRT file covering a few seconds of video."""
    blocks: list[str] = []
    samples = [
        (1, 1_000, 3_000, "Hello, world!"),
        (2, 3_500, 6_000, "This is a subtitle."),
        (3, 6_500, 9_000, "Have a nice day."),
    ]
    for idx, start_ms, end_ms, text in samples:
        blocks.append(
            f"{idx}\n{_format_ts(start_ms)} --> {_format_ts(end_ms)}\n{text}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _default_rate_limit() -> ProviderRateLimitConfig:
    """Tiny-but-valid rate-limit config for the integration tests."""
    return ProviderRateLimitConfig(
        batch_size_initial=4,
        batch_size_min=1,
        batch_size_max=8,
        payload_size_initial=1_000,
        payload_size_min=10,
        payload_size_max=10_000,
        payload_unit="chars",
        concurrency_initial=2,
        concurrency_min=1,
        concurrency_max=4,
        max_retries=2,
        backoff_base_ms=1,
        backoff_jitter_ms=0,
        probe_up_every_n_success=100,
        supports_batch=True,
    )


# ---------------------------------------------------------------------------
# Stub collaborators (no real network, no real ffmpeg)
# ---------------------------------------------------------------------------


@dataclass
class _MockTranslator:
    """Translator stand-in that produces a Chinese translation per entry.

    Matches the signature :class:`Translator.translate` exposes. The
    output preserves structure (index / timestamps); only the text is
    replaced with a deterministic Chinese string so contract checks
    in downstream code still pass.
    """

    calls: int = 0

    async def translate(
        self,
        entries: list[SubtitleEntry],
        provider_type: str,
        config: ProviderConfig,
        rate_limit_config: ProviderRateLimitConfig,
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        self.calls += 1
        out: list[SubtitleEntry] = []
        for e in entries:
            if e.text.strip() == "":
                text = ""
            else:
                text = f"中文翻译{e.index}"
            out.append(
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text=text,
                )
            )
        return out


@dataclass
class _MockTTSEngine:
    """TTS engine stand-in that returns one audio clip per non-empty entry."""

    calls: int = 0
    voices_seen: list[str | None] = field(default_factory=list)

    async def synthesize(
        self,
        entries: list[SubtitleEntry],
        voice_id: str | None,
        provider_type: str,
        config: ProviderConfig,
        rate_limit_config: ProviderRateLimitConfig,
    ) -> list[AudioClip]:
        self.calls += 1
        self.voices_seen.append(voice_id)
        clips: list[AudioClip] = []
        for e in entries:
            if not e.text.strip():
                continue
            clips.append(
                AudioClip(
                    entry_index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    audio=b"",
                    duration_ms=max(0, e.end_ms - e.start_ms),
                )
            )
        return clips


@dataclass
class _MockAligner:
    """Aligner stand-in that writes a dummy WAV and returns its path."""

    calls: int = 0
    output_path: Path | None = None

    def align(self, clips: Any, video_duration_ms: int) -> Path:
        self.calls += 1
        assert self.output_path is not None
        # Minimal RIFF header so file identity isn't silently empty.
        # The muxer stub never reads it, but a non-zero payload is
        # friendlier to any future debugging.
        self.output_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        return self.output_path


@dataclass
class _MockMuxer:
    """Muxer stand-in that records its call path and writes a fake .mkv."""

    full_calls: int = 0
    subtitle_only_calls: int = 0
    last_zh_subtitle: Path | None = None
    last_en_subtitle: Path | None = None
    last_aligned_audio: Path | None = None

    def mux_full(
        self,
        *,
        input_video: Path,
        aligned_zh_audio: Path,
        zh_subtitle: Path,
        en_subtitle: Path,
        output_path: Path,
    ) -> Path:
        self.full_calls += 1
        self.last_zh_subtitle = zh_subtitle
        self.last_en_subtitle = en_subtitle
        self.last_aligned_audio = aligned_zh_audio
        # Fake Matroska header so downstream assertions can sanity-check
        # file existence + minimum size. Real ffmpeg would produce a
        # binary; for the test we only care that the path is populated.
        output_path.write_bytes(b"\x1a\x45\xdf\xa3")
        return output_path

    def mux_subtitle_only(
        self,
        *,
        input_video: Path,
        zh_subtitle: Path,
        en_subtitle: Path,
        output_path: Path,
    ) -> Path:
        self.subtitle_only_calls += 1
        self.last_zh_subtitle = zh_subtitle
        self.last_en_subtitle = en_subtitle
        output_path.write_bytes(b"\x1a\x45\xdf\xa3")
        return output_path


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------


def _build_dubbing_manifest(
    tmp_root: Path,
    *,
    subtitle_path: Path,
    video_path: Path,
) -> ManifestParams:
    """Construct a valid ``subtitle_and_dubbing`` ManifestParams."""
    return ManifestParams(
        video_path=video_path,
        subtitle_path=subtitle_path,
        target_language="zh-CN",
        processing_mode=ProcessingMode.SUBTITLE_AND_DUBBING,
        voice_id="voice-a",
        translation_provider="llm",
        translation_endpoint="https://example.invalid/translate",
        translation_credential="tkey",
        translation_extra={},
        translation_rate_limit=_default_rate_limit(),
        tts_provider="llm",
        tts_endpoint="https://example.invalid/tts",
        tts_credential="vkey",
        tts_extra={"default_voice": "voice-a"},
        tts_rate_limit=_default_rate_limit(),
        supported_video_formats=["mkv", "mp4"],
    )


def _build_subtitle_only_manifest_with_bad_tts(
    tmp_root: Path,
    *,
    subtitle_path: Path,
    video_path: Path,
) -> ManifestParams:
    """Construct a ``subtitle_only`` ManifestParams whose tts_* fields are junk.

    P26 / R1.7 — subtitle_only mode ignores tts_provider / tts_endpoint /
    tts_credential / voice_id. Construct them as invalid on purpose so
    the test proves execution is unaffected.
    """
    return ManifestParams(
        video_path=video_path,
        subtitle_path=subtitle_path,
        target_language="zh-CN",
        processing_mode=ProcessingMode.SUBTITLE_ONLY,
        # Noise values: a mix of missing, empty and obviously wrong.
        voice_id=None,
        translation_provider="llm",
        translation_endpoint="https://example.invalid/translate",
        translation_credential="tkey",
        translation_extra={},
        translation_rate_limit=_default_rate_limit(),
        tts_provider=None,
        tts_endpoint="",
        tts_credential="!@#$bogus",
        tts_extra={},
        tts_rate_limit=None,
        supported_video_formats=["mkv", "mp4"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_subtitle_and_dubbing_end_to_end_writes_video_and_chinese_subtitle(
    tmp_path: Path,
) -> None:
    """Task 14.4: full pipeline produces an existing video + UTF-8 Chinese SRT.

    - ``output_video_path`` must exist on disk after ``run`` returns.
    - ``output_subtitle_path`` must be a UTF-8 SRT containing simplified
      Chinese characters (the :class:`_MockTranslator` emits ``中文翻译N``
      per entry).
    - The mux_full code path must be taken exactly once; mux_subtitle_only
      must not be called.
    - TTS engine and aligner must each be invoked exactly once.
    """
    # --- Fixture files ------------------------------------------------
    subtitle_path = tmp_path / "sample.srt"
    _write_sample_srt(subtitle_path)
    video_path = tmp_path / "sample.mkv"
    video_path.write_bytes(b"")  # muxer stub never reads it
    output_dir = tmp_path / "output"

    # --- Stubs --------------------------------------------------------
    translator = _MockTranslator()
    tts_engine = _MockTTSEngine()
    aligner = _MockAligner(output_path=tmp_path / "aligned.wav")
    muxer = _MockMuxer()
    reporter = InMemoryReporter()

    params = _build_dubbing_manifest(
        tmp_path,
        subtitle_path=subtitle_path,
        video_path=video_path,
    )

    # --- Run ----------------------------------------------------------
    result = asyncio.run(
        run(
            params,
            reporter=reporter,
            translator=translator,  # type: ignore[arg-type]
            tts_engine=tts_engine,  # type: ignore[arg-type]
            aligner=aligner,  # type: ignore[arg-type]
            muxer=muxer,  # type: ignore[arg-type]
            duration_probe=lambda _p: 10_000,
            output_dir_factory=lambda: output_dir,
        )
    )

    # --- Output video assertions -------------------------------------
    assert result.output_video_path.exists(), (
        f"output video missing: {result.output_video_path}"
    )
    assert result.output_video_path.stat().st_size > 0

    # --- Output subtitle assertions ----------------------------------
    assert result.output_subtitle_path.exists(), (
        f"output subtitle missing: {result.output_subtitle_path}"
    )
    # Decoding with utf-8 strictly proves the file is valid UTF-8.
    zh_content = result.output_subtitle_path.read_text(encoding="utf-8")
    # The mock translator emits ``中文翻译<index>`` for every non-empty
    # source entry, so we expect at least one CJK ideograph per entry.
    assert "中文翻译" in zh_content, (
        "translated subtitle should contain simplified Chinese text; "
        f"got: {zh_content!r}"
    )
    # Ensure every input entry (3) produced a Chinese line.
    for idx in (1, 2, 3):
        assert f"中文翻译{idx}" in zh_content, (
            f"expected Chinese line for entry {idx}, got:\n{zh_content}"
        )

    # --- Dispatch assertions -----------------------------------------
    assert muxer.full_calls == 1, (
        f"mux_full should be called once, got {muxer.full_calls}"
    )
    assert muxer.subtitle_only_calls == 0, (
        "subtitle_and_dubbing must not call mux_subtitle_only"
    )
    assert tts_engine.calls == 1
    assert aligner.calls == 1
    assert translator.calls == 1
    # voice_id was passed through from the manifest.
    assert tts_engine.voices_seen == ["voice-a"]

    # --- Progress assertions -----------------------------------------
    stages = [e.stage for e in reporter.events]
    # The dubbing path emits: parsing → translating → tts → muxing → done.
    assert "parsing" in stages
    assert "translating" in stages
    assert "tts" in stages, "dubbing mode must emit stage=tts (R11.3)"
    assert "muxing" in stages
    assert stages[-1] == "done"


def test_subtitle_only_end_to_end_tolerates_invalid_tts_params(
    tmp_path: Path,
) -> None:
    """Task 14.5: subtitle_only ignores bogus tts_* fields (P26, R1.7).

    The manifest supplies ``tts_provider=None``, an empty
    ``tts_endpoint``, a garbage ``tts_credential`` and
    ``tts_rate_limit=None`` — the execution must complete successfully
    because the skill never consults the TTS engine or the aligner
    under :data:`ProcessingMode.SUBTITLE_ONLY`.
    """
    subtitle_path = tmp_path / "sample.srt"
    _write_sample_srt(subtitle_path)
    video_path = tmp_path / "sample.mkv"
    video_path.write_bytes(b"")
    output_dir = tmp_path / "output"

    translator = _MockTranslator()
    tts_engine = _MockTTSEngine()
    aligner = _MockAligner(output_path=tmp_path / "aligned.wav")
    muxer = _MockMuxer()
    reporter = InMemoryReporter()

    params = _build_subtitle_only_manifest_with_bad_tts(
        tmp_path,
        subtitle_path=subtitle_path,
        video_path=video_path,
    )

    result = asyncio.run(
        run(
            params,
            reporter=reporter,
            translator=translator,  # type: ignore[arg-type]
            tts_engine=tts_engine,  # type: ignore[arg-type]
            aligner=aligner,  # type: ignore[arg-type]
            muxer=muxer,  # type: ignore[arg-type]
            # duration_probe should not be called in subtitle_only mode;
            # the lambda raises to prove that.
            duration_probe=lambda _p: pytest.fail(
                "duration_probe must not be called in subtitle_only mode"
            ),
            output_dir_factory=lambda: output_dir,
        )
    )

    # --- Output assertions -------------------------------------------
    assert result.output_video_path.exists(), (
        f"output video missing: {result.output_video_path}"
    )
    assert result.output_subtitle_path.exists(), (
        f"output subtitle missing: {result.output_subtitle_path}"
    )
    # UTF-8 decode must succeed on the Chinese subtitle output.
    zh_content = result.output_subtitle_path.read_text(encoding="utf-8")
    assert "中文翻译" in zh_content

    # --- Dispatch assertions -----------------------------------------
    assert muxer.subtitle_only_calls == 1
    assert muxer.full_calls == 0, (
        "subtitle_only must not call mux_full"
    )
    # TTS engine and aligner must not be invoked at all (R1.7, P26).
    assert tts_engine.calls == 0
    assert aligner.calls == 0

    # --- Progress assertions (P32) -----------------------------------
    stages = [e.stage for e in reporter.events]
    assert "tts" not in stages, (
        "subtitle_only must not emit stage=tts (R11.4, P32); "
        f"saw stages: {stages}"
    )
    assert stages[-1] == "done"
