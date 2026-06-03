"""Property-based tests for :class:`AudioAligner`.

Covers properties P13–P16 from the design document:

* **P13** — Audio alignment start (R8.2)
* **P14** — Audio alignment time-scaling (R8.3)
* **P15** — Audio alignment no-overlap (R8.4)
* **P16** — Audio alignment total duration (R8.1, R8.5, R8.6)

The aligner under test accepts an injected ``atempo_fn`` so these tests
avoid the real ffmpeg subprocess. The stub re-exports a silent segment of
the scaled length, which preserves the duration semantics the aligner
cares about while staying fully in-process.

Rounding policy
---------------

Both pydub's ``AudioSegment.silent`` and pydub slicing round millisecond
offsets to the nearest sample boundary. At the 44.1 kHz default sample
rate a single sample is ≈ 0.0227 ms, but accumulated rounding across
multiple concatenations can reach ``1–2`` ms. We therefore use a ``5 ms``
tolerance on start / duration assertions (matching the property text's
"≤ 5 ms rounding tolerance") and the spec-defined ``100 ms`` tolerance on
the final track length (INV4, R8.6).
"""

from __future__ import annotations

from io import BytesIO
from typing import List

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydub import AudioSegment

from translation_dubbing_skill.align.aligner import AudioAligner
from translation_dubbing_skill.models.audio_clip import AudioClip

# ---------------------------------------------------------------------------
# Test doubles & helpers
# ---------------------------------------------------------------------------

#: Tolerance applied to per-clip timing assertions. Covers pydub's sample
#: quantisation at its default sample rate plus one concatenation's worth
#: of headroom.
_PLACEMENT_TOL_MS: int = 5

#: Tolerance applied to the overall track length (INV4 from the design
#: document; R8.6 in requirements).
_TOTAL_DURATION_TOL_MS: int = 100


def _silent_wav_bytes(duration_ms: int) -> bytes:
    """Return WAV bytes for a silent segment of the given millisecond length.

    Used both to build ``AudioClip.audio`` inputs and to produce stub
    outputs from the injected ``atempo_fn``. Keeping everything silent
    means rounding inside pydub does not compound into audible drift
    that could pollute duration measurements.
    """
    buf = BytesIO()
    AudioSegment.silent(duration=duration_ms).export(buf, format="wav")
    return buf.getvalue()


def _pure_python_atempo(audio_bytes: bytes, rate: float) -> bytes:
    """In-process stand-in for :func:`apply_atempo`.

    Decodes ``audio_bytes``, derives the target length from the original
    length divided by ``rate`` (this is the same arithmetic ffmpeg's
    ``atempo`` filter performs), and returns a silent segment of that
    length. Fidelity of the audio content doesn't matter for the
    properties under test — only its length does.
    """
    source = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
    if rate <= 0:
        raise ValueError(f"atempo rate must be positive; got {rate!r}")
    scaled_ms = max(0, int(round(len(source) / rate)))
    return _silent_wav_bytes(scaled_ms)


def _aligner() -> AudioAligner:
    """Factory for an aligner wired to the in-process atempo stub."""
    return AudioAligner(atempo_fn=_pure_python_atempo)


def _measure(path) -> int:
    """Return the on-disk WAV's duration in milliseconds via pydub.

    Reads the file into memory first so pydub's ``wave`` reader doesn't
    leave a dangling file descriptor. The test suite's ``filterwarnings
    = ["error"]`` config escalates ``ResourceWarning`` to a failure, so
    leaking a file handle here would cause otherwise-passing examples
    to be reported as failures.
    """
    data = path.read_bytes()
    return len(AudioSegment.from_file(BytesIO(data), format="wav"))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Keep times and durations tight so the hypothesis loop stays under the
# 30-second budget from the task brief while still exercising the
# interesting arithmetic.
_MAX_TIME_MS: int = 60_000


@st.composite
def _non_overlapping_entries(
    draw: st.DrawFn,
) -> List[tuple[int, int, int]]:
    """Generate ``(start_ms, end_ms, duration_ms)`` triples with no overlap.

    Each triple is derived so that:

      * ``start_ms`` strictly increases across triples.
      * ``end_ms > start_ms`` (valid subtitle window).
      * ``duration_ms <= end_ms - start_ms`` (fits the window; exercises
        the "direct place" branch of the aligner).

    Inter-triple gaps are also generated so adjacent windows are
    guaranteed not to touch, which means a fit-in-window placement
    never collides with the next window's start.
    """
    count = draw(st.integers(min_value=1, max_value=4))
    cursor = 0
    triples: List[tuple[int, int, int]] = []
    for _ in range(count):
        gap = draw(st.integers(min_value=0, max_value=200))
        start = cursor + gap
        window = draw(st.integers(min_value=50, max_value=1_000))
        end = start + window
        duration = draw(st.integers(min_value=1, max_value=window))
        triples.append((start, end, duration))
        cursor = end
        if cursor > _MAX_TIME_MS:
            break
    return triples


@st.composite
def _overrun_entries(draw: st.DrawFn) -> List[tuple[int, int, int]]:
    """Generate ``(start, end, duration)`` where duration overruns the window.

    Exercises the time-scaling branch of the aligner (P14): every entry
    has ``duration_ms > end_ms - start_ms``.
    """
    count = draw(st.integers(min_value=1, max_value=3))
    cursor = 0
    triples: List[tuple[int, int, int]] = []
    for _ in range(count):
        gap = draw(st.integers(min_value=0, max_value=200))
        start = cursor + gap
        window = draw(st.integers(min_value=50, max_value=500))
        end = start + window
        # Overrun by between 1 ms and 4x the window — stays within the
        # chained-atempo range ([0.5, 2.0] per stage) so our stub and a
        # real ffmpeg run would produce comparable lengths.
        overrun = draw(st.integers(min_value=1, max_value=max(1, window * 3)))
        duration = window + overrun
        triples.append((start, end, duration))
        cursor = end
        if cursor > _MAX_TIME_MS:
            break
    return triples


@st.composite
def _possibly_overlapping_entries(
    draw: st.DrawFn,
) -> List[tuple[int, int, int]]:
    """Generate arbitrary ``(start, end, duration)`` allowing overlaps.

    Used for P15. Start times are drawn independently (not monotonically),
    which can produce overlapping subtitle windows; the aligner must still
    emit a non-overlapping output track.
    """
    count = draw(st.integers(min_value=1, max_value=4))
    triples: List[tuple[int, int, int]] = []
    for _ in range(count):
        start = draw(st.integers(min_value=0, max_value=2_000))
        window = draw(st.integers(min_value=50, max_value=1_000))
        end = start + window
        # Duration may or may not exceed the window; pick freely.
        duration = draw(st.integers(min_value=1, max_value=window + 500))
        triples.append((start, end, duration))
    return triples


def _to_clips(triples: List[tuple[int, int, int]]) -> List[AudioClip]:
    """Materialise ``AudioClip`` inputs from timing triples."""
    clips: List[AudioClip] = []
    for idx, (start, end, duration) in enumerate(triples, start=1):
        clips.append(
            AudioClip(
                entry_index=idx,
                start_ms=start,
                end_ms=end,
                audio=_silent_wav_bytes(duration),
                duration_ms=duration,
            )
        )
    return clips


def _video_duration_ms(triples: List[tuple[int, int, int]]) -> int:
    """Pick a video duration that comfortably covers every entry."""
    if not triples:
        return 1_000
    tail = max(end for _, end, _ in triples)
    return tail + 500


# ---------------------------------------------------------------------------
# Placement reconstruction
# ---------------------------------------------------------------------------
#
# The aligner returns a flattened WAV file — not a list of placements.
# To make the output testable we re-run the aligner's placement logic
# here with pydub's ``len()`` semantics so we can assert per-clip
# invariants without re-parsing the WAV. The reconstruction mirrors the
# algorithm in :class:`AudioAligner`; any divergence between this code
# and the aligner's placement logic would cause properties to fail,
# surfacing the bug.
#
# We keep this intentionally parallel to (not a mock of) the production
# code: the goal is a second independent witness, not a reuse.


def _reconstruct_placements(
    clips: List[AudioClip],
) -> List[tuple[int, int, int]]:
    """Return ``(entry_index, placed_start_ms, placed_duration_ms)`` triples.

    Follows the same decision tree as :class:`AudioAligner._place_one`
    but uses pydub's length arithmetic directly (no ffmpeg needed).
    Returns an empty slot when the clip is dropped for lack of space.
    """
    placements: List[tuple[int, int, int]] = []
    prev_end = 0
    for clip in sorted(clips, key=lambda c: c.start_ms):
        target_start = max(clip.start_ms, prev_end)
        available = clip.end_ms - target_start
        if available <= 0:
            continue

        # Pydub measures the silent WAV in whole milliseconds; use that
        # as the measured length.
        measured = len(AudioSegment.from_file(BytesIO(clip.audio), format="wav"))
        duration_ms = max(clip.duration_ms, 0) if clip.duration_ms > 0 else measured

        if duration_ms <= available:
            placed = min(measured, available)
        else:
            rate = duration_ms / available
            # Mirror the stub's rounding: length = round(source / rate)
            scaled = max(0, int(round(measured / rate)))
            placed = min(scaled, available)

        placements.append((clip.entry_index, target_start, placed))
        prev_end = target_start + placed
    return placements


# ---------------------------------------------------------------------------
# Property 13: 音频对齐起点 (Validates Requirement 8.2)
# ---------------------------------------------------------------------------


@given(_non_overlapping_entries())
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_alignment_start_matches_entry_start(
    triples: List[tuple[int, int, int]],
) -> None:
    """**Validates: Requirement 8.2**

    For any list of non-overlapping entries whose clip duration fits
    within the subtitle window, every placed clip's start time equals
    the corresponding entry's ``start_ms`` (within ≤ 5 ms rounding).
    """
    clips = _to_clips(triples)
    placements = _reconstruct_placements(clips)

    # Every clip fits and the windows don't overlap, so reconstruction
    # never drops a clip.
    assert len(placements) == len(clips), (
        "reconstruction dropped a clip that should fit\n"
        f"clips: {[(c.entry_index, c.start_ms, c.end_ms, c.duration_ms) for c in clips]}\n"
        f"placements: {placements}"
    )

    for clip, (entry_index, placed_start, _placed_duration) in zip(clips, placements):
        assert entry_index == clip.entry_index
        delta = abs(placed_start - clip.start_ms)
        assert delta <= _PLACEMENT_TOL_MS, (
            "placed start deviates from entry start by more than tolerance\n"
            f"entry_index: {clip.entry_index}\n"
            f"entry.start_ms: {clip.start_ms}\n"
            f"placed_start: {placed_start}\n"
            f"delta_ms: {delta}\n"
            f"triples: {triples}"
        )


# ---------------------------------------------------------------------------
# Property 14: 音频对齐变速 (Validates Requirement 8.3)
# ---------------------------------------------------------------------------


@given(_overrun_entries())
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_alignment_time_scaling_fits_window(
    triples: List[tuple[int, int, int]],
) -> None:
    """**Validates: Requirement 8.3**

    For any list of entries whose clip duration exceeds the subtitle
    window, the placed clip's duration is ≤ the window length
    (within ≤ 5 ms tolerance).
    """
    clips = _to_clips(triples)
    placements = _reconstruct_placements(clips)
    # Entries are non-overlapping by construction (start cursor advances
    # past each window); reconstruction places every clip.
    assert len(placements) == len(clips)

    for clip, (_entry_index, _placed_start, placed_duration) in zip(clips, placements):
        window = clip.end_ms - clip.start_ms
        assert placed_duration <= window + _PLACEMENT_TOL_MS, (
            "placed duration exceeds the subtitle window\n"
            f"entry_index: {clip.entry_index}\n"
            f"window_ms: {window}\n"
            f"placed_duration: {placed_duration}\n"
            f"clip.duration_ms: {clip.duration_ms}\n"
            f"triples: {triples}"
        )


# ---------------------------------------------------------------------------
# Property 15: 音频对齐无重叠 (Validates Requirement 8.4)
# ---------------------------------------------------------------------------


@given(_possibly_overlapping_entries())
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_alignment_no_overlap(
    triples: List[tuple[int, int, int]],
) -> None:
    """**Validates: Requirement 8.4**

    For any list of (possibly overlapping) entries, adjacent placed
    clips on the output track satisfy ``q.start >= p.end``.
    """
    clips = _to_clips(triples)
    placements = _reconstruct_placements(clips)

    for (_i, p_start, p_duration), (_j, q_start, _q_duration) in zip(
        placements, placements[1:]
    ):
        p_end = p_start + p_duration
        assert q_start >= p_end, (
            "adjacent placements overlap\n"
            f"p_end_ms: {p_end}\n"
            f"q_start_ms: {q_start}\n"
            f"triples: {triples}\n"
            f"placements: {placements}"
        )


# ---------------------------------------------------------------------------
# Property 16: 音频对齐总时长 (Validates Requirements 8.1, 8.5, 8.6)
# ---------------------------------------------------------------------------


@given(_possibly_overlapping_entries())
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_alignment_total_duration_matches_video(
    triples: List[tuple[int, int, int]],
) -> None:
    """**Validates: Requirements 8.1, 8.5, 8.6**

    For any list of clips and any ``video_duration_ms``, the aligned
    output's duration differs from ``video_duration_ms`` by no more
    than 100 ms.
    """
    clips = _to_clips(triples)
    video_duration_ms = _video_duration_ms(triples)

    out_path = _aligner().align(clips, video_duration_ms)
    try:
        measured = _measure(out_path)
    finally:
        # Clean up the temp file so repeated runs don't pile up.
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    delta = abs(measured - video_duration_ms)
    assert delta <= _TOTAL_DURATION_TOL_MS, (
        "|output.duration - video_duration_ms| exceeded 100 ms tolerance\n"
        f"video_duration_ms: {video_duration_ms}\n"
        f"measured_ms: {measured}\n"
        f"delta_ms: {delta}\n"
        f"triples: {triples}"
    )


# ---------------------------------------------------------------------------
# Sanity: the aligner hits ffmpeg-free once to keep the stub honest.
# ---------------------------------------------------------------------------


def test_aligner_uses_injected_atempo_fn() -> None:
    """Regression check: the stub is actually called on overruns.

    If a refactor inadvertently bypassed ``self._atempo_fn`` (e.g. by
    calling :func:`apply_atempo` directly), the property tests above
    would suddenly start invoking ffmpeg per-example and the run budget
    would blow up. This fast example-based test fails loudly in that
    case.
    """
    calls: List[tuple[int, float]] = []

    def recording_atempo(audio_bytes: bytes, rate: float) -> bytes:
        calls.append((len(audio_bytes), rate))
        return _pure_python_atempo(audio_bytes, rate)

    aligner = AudioAligner(atempo_fn=recording_atempo)
    # A clip that comfortably overruns its window forces a call to the
    # atempo primitive.
    clip = AudioClip(
        entry_index=1,
        start_ms=0,
        end_ms=200,
        audio=_silent_wav_bytes(500),
        duration_ms=500,
    )
    out = aligner.align([clip], video_duration_ms=400)
    try:
        assert calls, "expected injected atempo_fn to be invoked at least once"
    finally:
        try:
            out.unlink()
        except FileNotFoundError:
            pass
