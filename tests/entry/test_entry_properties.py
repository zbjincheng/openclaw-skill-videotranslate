"""Property tests for the skill entry orchestration (tasks 12.2–12.4).

These tests exercise :func:`translation_dubbing_skill.entry.run` end-to-end
with injected stubs so that neither ``ffmpeg`` nor any real provider is
involved. Each test pins a specific property from the design's
P26 / P32 / P33 specification:

- **P26** — ``subtitle_only`` mode must not instantiate or call the TTS
  engine or the audio aligner (R1.7, R6.7, R8.7).
- **P32** — ``subtitle_only`` mode must never emit a ``stage="tts"``
  progress event (R11.4).
- **P33** — ``subtitle_and_dubbing`` mode, given at least one non-empty
  subtitle entry, MUST emit at least one ``stage="tts"`` progress event
  (R11.3).

The shared helpers below construct a hermetic fixture: a temp SRT file on
disk, a fake video path whose existence is satisfied by writing an empty
file with a supported suffix, a counting TTS stub, a counting aligner
stub, a muxer stub that writes a dummy output, a translator stub that
returns Chinese translations, and a duration probe that returns a
constant. The stubs carry ``.calls`` / ``.events`` counters so the
properties can assert exact invocation counts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st

from translation_dubbing_skill.align.aligner import AudioAligner
from translation_dubbing_skill.entry.entry import run
from translation_dubbing_skill.entry.manifest import ManifestParams
from translation_dubbing_skill.models import (
    AudioClip,
    ProcessingMode,
    ProgressEvent,
    ProviderConfig,
    SubtitleEntry,
)
from translation_dubbing_skill.mux.muxer import VideoMuxer
from translation_dubbing_skill.progress.reporter import InMemoryReporter
from translation_dubbing_skill.providers.registry import ProviderRegistry
from translation_dubbing_skill.scheduler.config import ProviderRateLimitConfig
from translation_dubbing_skill.translation.translator import Translator
from translation_dubbing_skill.tts.engine import TTSEngine


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _CountingTTSEngine:
    """TTS engine stand-in that records every ``synthesize`` call.

    ``calls`` counts invocations; ``received_entries`` captures the
    subtitle lists so a property can also assert *what* was passed in.
    The class quacks like :class:`TTSEngine` (awaitable ``synthesize``).
    """

    calls: int = 0
    received_entries: list[list[SubtitleEntry]] = field(default_factory=list)
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
        self.received_entries.append(list(entries))
        self.voices_seen.append(voice_id)
        # Produce a trivial clip per non-empty entry so the aligner
        # contract (non-empty list → non-empty clips) is respected.
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
class _CountingAligner:
    """Audio aligner stand-in that records every ``align`` call."""

    calls: int = 0
    #: Path the stub writes to and returns; set by the fixture builder.
    output_path: Path | None = None

    def align(self, clips: Any, video_duration_ms: int) -> Path:
        self.calls += 1
        assert self.output_path is not None, "output_path must be set by the fixture"
        # Write something so the path exists; the muxer stub does not
        # actually read it, but this keeps the stub honest.
        self.output_path.write_bytes(b"")
        return self.output_path


@dataclass
class _StubTranslator:
    """Translator stand-in that maps every entry's text to a CJK string.

    The real :class:`Translator` would route through the adaptive
    scheduler and a registered provider; for an entry-point property
    test we only need a deterministic, async-compatible translation
    that preserves the entry structure.
    """

    async def translate(
        self,
        entries: list[SubtitleEntry],
        provider_type: str,
        config: ProviderConfig,
        rate_limit_config: ProviderRateLimitConfig,
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        translated: list[SubtitleEntry] = []
        for e in entries:
            if e.text.strip() == "":
                text = ""
            else:
                text = "翻译" + str(e.index)
            translated.append(
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text=text,
                )
            )
        return translated


@dataclass
class _StubMuxer:
    """Muxer stand-in that writes an empty file at ``output_path``."""

    full_calls: int = 0
    subtitle_only_calls: int = 0

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
        output_path.write_bytes(b"")
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
        output_path.write_bytes(b"")
        return output_path


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


# Bound entry counts small — the entry-point test does not exercise
# scheduler adaptivity; a handful of entries is enough to cover the
# dispatch and progress paths.
_MIN_ENTRIES = 1
_MAX_ENTRIES = 5


@st.composite
def _non_empty_entry_text(draw: st.DrawFn) -> str:
    """Generate non-whitespace subtitle text suitable for translation."""
    # ASCII + some unicode; strip trailing spaces so the leading
    # ``strip`` check in the TTS engine and translator sees non-empty.
    body = draw(
        st.text(
            alphabet=st.characters(
                min_codepoint=0x20,
                max_codepoint=0x7E,
                blacklist_categories=("Cs",),
            ),
            min_size=1,
            max_size=30,
        )
    )
    # Force at least one non-whitespace character.
    if not body.strip():
        body = "hello" + body
    return body


@st.composite
def _subtitle_entries(
    draw: st.DrawFn,
    *,
    require_non_empty: bool = False,
) -> list[SubtitleEntry]:
    """Build a list of monotonically non-overlapping subtitle entries."""
    count = draw(st.integers(min_value=_MIN_ENTRIES, max_value=_MAX_ENTRIES))
    entries: list[SubtitleEntry] = []
    cursor = 0
    for i in range(count):
        start = cursor + draw(st.integers(min_value=0, max_value=50))
        duration = draw(st.integers(min_value=50, max_value=500))
        end = start + duration
        # Decide whether to produce an empty entry (mimicking a silent
        # caption line). In ``require_non_empty=True`` mode, force at
        # least one entry to be non-empty; the simplest way is to
        # always produce non-empty text.
        if require_non_empty:
            text = draw(_non_empty_entry_text())
        else:
            text = draw(
                st.one_of(
                    st.just(""),
                    st.just("   "),  # whitespace-only
                    _non_empty_entry_text(),
                )
            )
        entries.append(
            SubtitleEntry(
                index=i + 1,
                start_ms=start,
                end_ms=end,
                text=text,
            )
        )
        cursor = end + draw(st.integers(min_value=0, max_value=20))
    if require_non_empty:
        # Guarantee at least one entry with non-empty text even if the
        # strategy produced blanks upstream (defensive).
        if not any(e.text.strip() for e in entries):
            first = entries[0]
            entries[0] = SubtitleEntry(
                index=first.index,
                start_ms=first.start_ms,
                end_ms=first.end_ms,
                text="hello world",
            )
    return entries


@st.composite
def _tts_manifest_noise(draw: st.DrawFn) -> dict[str, Any]:
    """Noise values for tts_* fields in subtitle_only mode (P26).

    P26 requires that the skill ignores these fields entirely, including
    missing, empty, garbage, or obviously-invalid values.
    """
    noise_values = st.one_of(
        st.none(),
        st.just(""),
        st.text(min_size=0, max_size=30),
        st.just("  "),
        st.just("!@#$%^&"),
    )
    return {
        "tts_provider": draw(noise_values),
        "tts_endpoint": draw(noise_values),
        "tts_credential": draw(noise_values),
        "voice_id": draw(noise_values),
    }


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_srt(path: Path, entries: list[SubtitleEntry]) -> None:
    """Write ``entries`` to ``path`` as a minimal SRT file."""
    blocks: list[str] = []
    for e in entries:
        start = _format_ts(e.start_ms)
        end = _format_ts(e.end_ms)
        text = e.text if e.text else ""
        blocks.append(f"{e.index}\n{start} --> {end}\n{text}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _format_ts(ms: int) -> str:
    hours, r = divmod(ms, 3_600_000)
    minutes, r = divmod(r, 60_000)
    seconds, millis = divmod(r, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _default_rate_limit() -> ProviderRateLimitConfig:
    """Small but valid rate-limit config for tests."""
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


def _build_manifest(
    *,
    tmp_root: Path,
    entries: list[SubtitleEntry],
    mode: ProcessingMode,
    tts_noise: dict[str, Any] | None = None,
) -> ManifestParams:
    """Construct a :class:`ManifestParams` referring to on-disk fixture files.

    The video path is a zero-byte ``.mkv`` — the muxer stub never reads
    from it, so the content doesn't matter as long as it exists.
    """
    video_path = tmp_root / "input.mkv"
    video_path.write_bytes(b"")
    subtitle_path = tmp_root / "input.srt"
    _write_srt(subtitle_path, entries)

    # Always set translation_* to sane values — these are required in
    # both modes. For ``subtitle_only`` the tts_* fields are ignored by
    # the skill regardless of what they hold (R1.7, P26). We still need
    # ``tts_rate_limit=None`` (default) because parse_manifest does not
    # populate it in SUBTITLE_ONLY mode.
    return ManifestParams(
        video_path=video_path,
        subtitle_path=subtitle_path,
        target_language="zh-CN",
        processing_mode=mode,
        voice_id=(tts_noise or {}).get("voice_id") if mode is ProcessingMode.SUBTITLE_ONLY else None,
        translation_provider="llm",
        translation_endpoint="https://example.invalid/translate",
        translation_credential="secret",
        translation_extra={},
        translation_rate_limit=_default_rate_limit(),
        tts_provider=(tts_noise or {}).get("tts_provider") if mode is ProcessingMode.SUBTITLE_ONLY else None,
        tts_endpoint=(tts_noise or {}).get("tts_endpoint") if mode is ProcessingMode.SUBTITLE_ONLY else None,
        tts_credential=(tts_noise or {}).get("tts_credential") if mode is ProcessingMode.SUBTITLE_ONLY else None,
        tts_extra={},
        tts_rate_limit=None if mode is ProcessingMode.SUBTITLE_ONLY else _default_rate_limit(),
        supported_video_formats=["mkv", "mp4"],
    )


def _build_dubbing_manifest(
    *,
    tmp_root: Path,
    entries: list[SubtitleEntry],
) -> ManifestParams:
    """Construct a :class:`ManifestParams` for the dubbing path.

    All tts_* fields are populated with valid placeholders — none of
    them hit the wire because the ``TTSEngine`` is replaced by a stub.
    """
    video_path = tmp_root / "input.mkv"
    video_path.write_bytes(b"")
    subtitle_path = tmp_root / "input.srt"
    _write_srt(subtitle_path, entries)

    return ManifestParams(
        video_path=video_path,
        subtitle_path=subtitle_path,
        target_language="zh-CN",
        processing_mode=ProcessingMode.SUBTITLE_AND_DUBBING,
        voice_id="voice-a",
        translation_provider="llm",
        translation_endpoint="https://example.invalid/translate",
        translation_credential="secret",
        translation_extra={},
        translation_rate_limit=_default_rate_limit(),
        tts_provider="llm",
        tts_endpoint="https://example.invalid/tts",
        tts_credential="tts-secret",
        tts_extra={},
        tts_rate_limit=_default_rate_limit(),
        supported_video_formats=["mkv", "mp4"],
    )


# ---------------------------------------------------------------------------
# P26 — subtitle_only suppresses TTS + aligner (task 12.2)
# ---------------------------------------------------------------------------


@given(
    entries=_subtitle_entries(),
    tts_noise=_tts_manifest_noise(),
)
@settings(
    max_examples=30,
    deadline=1000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_subtitle_only_suppresses_tts_and_aligner(
    tmp_path_factory,
    entries,
    tts_noise,
):
    """**Validates: Requirements 1.7, 6.7, 8.7**

    P26: Under ``processing_mode=subtitle_only`` the skill MUST NOT
    invoke the TTS engine or the audio aligner regardless of what the
    tts_* manifest fields hold.
    """
    tmp_root = tmp_path_factory.mktemp("p26")
    params = _build_manifest(
        tmp_root=tmp_root,
        entries=entries,
        mode=ProcessingMode.SUBTITLE_ONLY,
        tts_noise=tts_noise,
    )

    tts_stub = _CountingTTSEngine()
    aligner_stub = _CountingAligner(output_path=tmp_root / "aligned.wav")
    muxer_stub = _StubMuxer()
    translator_stub = _StubTranslator()

    asyncio.run(
        run(
            params,
            reporter=None,
            translator=translator_stub,  # type: ignore[arg-type]
            tts_engine=tts_stub,  # type: ignore[arg-type]
            aligner=aligner_stub,  # type: ignore[arg-type]
            muxer=muxer_stub,  # type: ignore[arg-type]
            duration_probe=lambda _p: 10_000,
            output_dir_factory=lambda: tmp_root / "out",
        )
    )

    assert tts_stub.calls == 0, (
        "subtitle_only must not invoke the TTS engine "
        f"(saw {tts_stub.calls} calls)"
    )
    assert aligner_stub.calls == 0, (
        "subtitle_only must not invoke the audio aligner "
        f"(saw {aligner_stub.calls} calls)"
    )
    assert muxer_stub.full_calls == 0, "subtitle_only must not call mux_full"
    assert muxer_stub.subtitle_only_calls == 1, (
        "subtitle_only must call mux_subtitle_only exactly once "
        f"(saw {muxer_stub.subtitle_only_calls})"
    )


# ---------------------------------------------------------------------------
# P32 — subtitle_only never emits stage="tts" (task 12.3)
# ---------------------------------------------------------------------------


@given(entries=_subtitle_entries())
@settings(
    max_examples=30,
    deadline=1000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_subtitle_only_never_emits_tts_stage(
    tmp_path_factory,
    entries,
):
    """**Validates: Requirement 11.4**

    P32: Under ``processing_mode=subtitle_only`` the progress event
    sequence MUST NOT contain any event with ``stage="tts"``.
    """
    tmp_root = tmp_path_factory.mktemp("p32")
    params = _build_manifest(
        tmp_root=tmp_root,
        entries=entries,
        mode=ProcessingMode.SUBTITLE_ONLY,
    )
    reporter = InMemoryReporter()

    asyncio.run(
        run(
            params,
            reporter=reporter,
            translator=_StubTranslator(),  # type: ignore[arg-type]
            muxer=_StubMuxer(),  # type: ignore[arg-type]
            duration_probe=lambda _p: 10_000,
            output_dir_factory=lambda: tmp_root / "out",
        )
    )

    tts_events = [e for e in reporter.events if e.stage == "tts"]
    assert tts_events == [], (
        "subtitle_only must emit zero stage=tts events "
        f"(saw {len(tts_events)}: {tts_events})"
    )


# ---------------------------------------------------------------------------
# P33 — subtitle_and_dubbing emits at least one stage="tts" (task 12.4)
# ---------------------------------------------------------------------------


@given(entries=_subtitle_entries(require_non_empty=True))
@settings(
    max_examples=30,
    deadline=1000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_subtitle_and_dubbing_emits_tts_stage(
    tmp_path_factory,
    entries,
):
    """**Validates: Requirement 11.3**

    P33: Under ``processing_mode=subtitle_and_dubbing`` with at least
    one non-empty subtitle entry, the progress event sequence MUST
    include at least one event with ``stage="tts"``.
    """
    tmp_root = tmp_path_factory.mktemp("p33")
    params = _build_dubbing_manifest(tmp_root=tmp_root, entries=entries)
    reporter = InMemoryReporter()

    asyncio.run(
        run(
            params,
            reporter=reporter,
            translator=_StubTranslator(),  # type: ignore[arg-type]
            tts_engine=_CountingTTSEngine(),  # type: ignore[arg-type]
            aligner=_CountingAligner(output_path=tmp_root / "aligned.wav"),  # type: ignore[arg-type]
            muxer=_StubMuxer(),  # type: ignore[arg-type]
            duration_probe=lambda _p: 60_000,
            output_dir_factory=lambda: tmp_root / "out",
        )
    )

    tts_events = [e for e in reporter.events if e.stage == "tts"]
    assert len(tts_events) >= 1, (
        "subtitle_and_dubbing with non-empty entries must emit at "
        f"least one stage=tts event (saw {len(tts_events)})"
    )
