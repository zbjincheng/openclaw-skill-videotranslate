"""Property-based tests for the TTS layer (P8, P9, P21, P25 — tts part).

Covers tasks 7.4, 7.5, 7.6 and 7.7 of the spec. Each block below
implements one property from ``design.md``:

- P8  — 语音合成提供方返回形状 (R6.2, R7.3)
- P9  — 语音合成条目映射与顺序 (R6.1, R7.4)
- P21 — voice_id 透传 (R6.5)
- P25 — 翻译与语音合成进度事件完整性（语音合成部分, R11.3）

P8 is parametrised over both ``LLMTTSProvider`` and ``WebTTSProvider``;
network traffic is stubbed via :class:`httpx.MockTransport` so the tests
run fully offline. P9/P21/P25 drive :class:`TTSEngine` with a simple
in-memory mock provider that records every call so the property can
observe call order, per-call ``voice_id``, and the reporter's event
stream.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable, ClassVar

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from translation_dubbing_skill.models import (
    AudioClip,
    ProgressEvent,
    ProviderConfig,
    SubtitleEntry,
)
from translation_dubbing_skill.providers import ProviderRegistry
from translation_dubbing_skill.providers.tts.llm import LLMTTSProvider
from translation_dubbing_skill.providers.tts.web import WebTTSProvider
from translation_dubbing_skill.scheduler import ProviderRateLimitConfig
from translation_dubbing_skill.tts import TTSEngine


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_rate_limit_config(**overrides: Any) -> ProviderRateLimitConfig:
    """Return a permissive config (large caps, no probing) with overrides."""
    base: dict[str, Any] = dict(
        batch_size_initial=4,
        batch_size_min=1,
        batch_size_max=16,
        payload_size_initial=10_000,
        payload_size_min=10,
        payload_size_max=100_000,
        payload_unit="chars",
        concurrency_initial=2,
        concurrency_min=1,
        concurrency_max=8,
        max_retries=1,
        backoff_base_ms=1,
        backoff_jitter_ms=0,
        probe_up_every_n_success=10_000,
        supports_batch=True,
    )
    base.update(overrides)
    return ProviderRateLimitConfig(**base)


class _InMemoryReporter:
    """Captures :class:`ProgressEvent` objects for later assertion."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def report(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Hypothesis generators
# ---------------------------------------------------------------------------


# Common-CJK Unicode block — every character here is non-whitespace and
# treated as "non-empty simplified Chinese" by the coordinator's semantic
# checks. Using a restricted block keeps the generator fast and avoids
# surrogate-pair complications.
_CJK_CHARS = st.characters(
    min_codepoint=0x4E00,
    max_codepoint=0x9FFF,
)

_NONEMPTY_CHINESE_TEXT = st.text(
    alphabet=_CJK_CHARS,
    min_size=1,
    max_size=12,
)

# ASCII / digit generator used for voice_id — keeps URLs and HTTP headers
# happy while still exercising a reasonable slice of the input space.
_VOICE_ID = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=16,
)

# Generator allowing both empty/whitespace and non-empty Chinese text so
# P9 can observe the engine's skip-empty behaviour under a realistic mix.
_MIXED_CHINESE_TEXT = st.one_of(
    st.sampled_from(["", " ", "  ", "\t", "\n", "\t \n"]),
    _NONEMPTY_CHINESE_TEXT,
)


@st.composite
def _subtitle_entries(
    draw: st.DrawFn,
    *,
    min_size: int = 1,
    max_size: int = 6,
    text_strategy: st.SearchStrategy[str] = _MIXED_CHINESE_TEXT,
) -> list[SubtitleEntry]:
    """Draw a list of non-overlapping subtitle entries with monotonic times.

    Each entry's index is 1-based and strictly increasing; start/end
    timestamps are positive and ``end_ms > start_ms`` is enforced.
    """
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    entries: list[SubtitleEntry] = []
    cursor = 0
    for i in range(count):
        start = cursor + draw(st.integers(min_value=0, max_value=100))
        duration = draw(st.integers(min_value=1, max_value=1_000))
        end = start + duration
        text = draw(text_strategy)
        entries.append(
            SubtitleEntry(
                index=i + 1,
                start_ms=start,
                end_ms=end,
                text=text,
            )
        )
        cursor = end
    return entries


# ---------------------------------------------------------------------------
# Mock transports for real providers (P8)
# ---------------------------------------------------------------------------


def _llm_tts_handler(request: httpx.Request) -> httpx.Response:
    """Return one ``(audio_base64, duration_ms)`` pair per input text.

    Mirrors the shape ``LLMTTSProvider`` expects: a JSON array of
    ``{"audio_base64": ..., "duration_ms": ...}`` entries. The audio is
    deterministically derived from the input text so the property can
    additionally assert provider-level fidelity if needed; the duration
    is non-negative (``>= 0``) as required by the contract.
    """
    body = json.loads(request.content)
    inputs = body.get("inputs", [])
    items = [
        {
            "audio_base64": _b64(f"audio-{text}".encode("utf-8")),
            "duration_ms": max(0, len(text) * 10),
        }
        for text in inputs
    ]
    return httpx.Response(200, json=items)


def _web_tts_handler(request: httpx.Request) -> httpx.Response:
    """Return a single ``{audio_base64, duration_ms}`` JSON object.

    ``WebTTSProvider`` is single-shot so each call yields exactly one
    response. Duration is non-negative.
    """
    body = json.loads(request.content)
    text = body.get("text", "")
    return httpx.Response(
        200,
        json={
            "audio_base64": _b64(f"audio-{text}".encode("utf-8")),
            "duration_ms": max(0, len(text) * 10),
        },
    )


def _make_llm_tts_provider() -> LLMTTSProvider:
    provider = LLMTTSProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://api.llm.test/v1/audio/speech",
            credential="secret-token",
            extra={"model_name": "tts-1", "default_voice": "alloy"},
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_llm_tts_handler)
    )
    return provider


def _make_web_tts_provider() -> WebTTSProvider:
    provider = WebTTSProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://api.web-tts.test/synth",
            credential="secret-token",
            extra={"default_voice": "zh-CN-XiaoxiaoNeural"},
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_web_tts_handler)
    )
    return provider


_PROVIDER_FACTORIES: list[tuple[str, Callable[[], Any]]] = [
    ("llm", _make_llm_tts_provider),
    ("web", _make_web_tts_provider),
]


# ---------------------------------------------------------------------------
# Property 8 — 语音合成提供方返回形状 (task 7.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_label, make_provider",
    _PROVIDER_FACTORIES,
    ids=[label for label, _ in _PROVIDER_FACTORIES],
)
@given(text=_NONEMPTY_CHINESE_TEXT, voice_id=_VOICE_ID)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_property_8_tts_provider_return_shape(
    provider_label: str,
    make_provider: Callable[[], Any],
    text: str,
    voice_id: str,
) -> None:
    """**Validates: Requirements 6.2, 7.3**

    For any registered TTS provider and any non-empty Chinese text +
    legal ``voice_id``, :meth:`synth` (or :meth:`synth_batch` for
    batch-capable providers) SHALL return a ``(bytes, int)`` pair with
    ``duration_ms >= 0``.
    """
    provider = make_provider()
    try:
        # Always exercise the single-shot path — it is part of the
        # protocol for every provider regardless of batching support.
        result = await provider.synth(text, voice_id)

        assert isinstance(result, tuple), (
            f"{provider_label}: expected tuple, got {type(result).__name__}"
        )
        assert len(result) == 2, (
            f"{provider_label}: expected 2-tuple, got {len(result)}-tuple"
        )
        audio, duration_ms = result
        assert isinstance(audio, bytes), (
            f"{provider_label}: expected bytes audio, got "
            f"{type(audio).__name__}"
        )
        assert isinstance(duration_ms, int) and not isinstance(
            duration_ms, bool
        ), (
            f"{provider_label}: expected int duration_ms, got "
            f"{type(duration_ms).__name__}"
        )
        assert duration_ms >= 0, (
            f"{provider_label}: duration_ms must be non-negative, got "
            f"{duration_ms}"
        )

        # When the provider advertises batching, exercise synth_batch
        # too — the contract says both paths return the same shape.
        if getattr(provider, "supports_batch", False):
            batch_out = await provider.synth_batch([text], voice_id)
            assert isinstance(batch_out, list) and len(batch_out) == 1
            b_audio, b_duration = batch_out[0]
            assert isinstance(b_audio, bytes)
            assert isinstance(b_duration, int) and not isinstance(
                b_duration, bool
            )
            assert b_duration >= 0
    finally:
        await provider.aclose()


# ---------------------------------------------------------------------------
# Mock TTSProvider for P9 / P21 / P25
# ---------------------------------------------------------------------------


class _RecordingTTSProvider:
    """In-memory mock provider that records every call.

    Records per invocation:
      - ``synth_calls``: list of ``(text, voice_id)`` for single-shot calls.
      - ``batch_calls``: list of ``(texts, voice_id)`` for batched calls.

    The class-level ``_last`` reference is pinned by each test's
    :class:`ProviderRegistry` so the assertions can reach the instance
    actually used by :class:`TTSEngine` (the registry builds one
    instance per ``synthesize`` call).
    """

    provider_type: ClassVar[str] = "recording"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    # Class-level pointer so tests can inspect the most recently built
    # instance without threading one through the registry.
    _last: ClassVar["_RecordingTTSProvider | None"] = None

    def __init__(self) -> None:
        self.synth_calls: list[tuple[str, str]] = []
        self.batch_calls: list[tuple[list[str], str]] = []
        type(self)._last = self

    def initialize(self, config: ProviderConfig) -> None:
        pass

    def size_of(self, text: str) -> int:
        return len(text)

    async def synth(self, text: str, voice_id: str) -> tuple[bytes, int]:
        self.synth_calls.append((text, voice_id))
        return (text.encode("utf-8"), max(0, len(text) * 10))

    async def synth_batch(
        self,
        texts: list[str],
        voice_id: str,
    ) -> list[tuple[bytes, int]]:
        self.batch_calls.append((list(texts), voice_id))
        return [
            (t.encode("utf-8"), max(0, len(t) * 10)) for t in texts
        ]


def _build_recording_registry(
    *,
    supports_batch: bool = True,
) -> tuple[ProviderRegistry, type]:
    """Build a fresh registry bound to a :class:`_RecordingTTSProvider`.

    Each test gets its own ``cls`` so the class-level ``_last`` pointer
    is not shared across Hypothesis examples running concurrently.
    """
    cls = type(
        "_RecordingTTSProviderInstance",
        (_RecordingTTSProvider,),
        {"supports_batch": supports_batch, "_last": None},
    )
    registry = ProviderRegistry()
    registry.register("tts", "recording", cls)
    return registry, cls


# ---------------------------------------------------------------------------
# Property 9 — 语音合成条目映射与顺序 (task 7.5)
# ---------------------------------------------------------------------------


def _is_blank(text: str) -> bool:
    return not text.strip()


@given(
    entries=_subtitle_entries(min_size=1, max_size=6),
    supports_batch=st.booleans(),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_property_9_tts_entry_mapping_and_order(
    entries: list[SubtitleEntry],
    supports_batch: bool,
) -> None:
    """**Validates: Requirements 6.1, 7.4**

    For any list of subtitle entries (including a mix of empty and
    non-empty text), :meth:`TTSEngine.synthesize` SHALL return:

    - ``len(clips) == number of non-empty entries``
    - ``clip.entry_index`` monotonically strictly increasing across the
      result list (because input indices are monotonically increasing
      and empty entries are skipped)
    - every ``clip.entry_index`` equals some non-empty ``entries[j].index``
    """
    registry, _ = _build_recording_registry(supports_batch=supports_batch)
    engine = TTSEngine(registry)

    non_empty = [e for e in entries if not _is_blank(e.text)]

    config = ProviderConfig(
        endpoint="https://stub",
        credential="stub",
        extra={"default_voice": "v-default"},
    )

    clips = await engine.synthesize(
        entries=entries,
        voice_id="v-test",
        provider_type="recording",
        config=config,
        rate_limit_config=_make_rate_limit_config(
            supports_batch=supports_batch,
        ),
    )

    # Length equals count of non-empty entries.
    assert len(clips) == len(non_empty), (
        f"expected {len(non_empty)} clips for non-empty entries, "
        f"got {len(clips)}"
    )

    # Every clip is an AudioClip.
    assert all(isinstance(c, AudioClip) for c in clips)

    # entry_index values match the non-empty input indices, in order.
    expected_indices = [e.index for e in non_empty]
    actual_indices = [c.entry_index for c in clips]
    assert actual_indices == expected_indices, (
        f"entry_index sequence mismatch: expected {expected_indices}, "
        f"got {actual_indices}"
    )

    # Monotonic strictly increasing (follows from input indices being 1..N).
    for a, b in zip(actual_indices, actual_indices[1:]):
        assert b > a, f"entry_index regressed: {a} -> {b}"

    # Every emitted entry_index corresponds to some non-empty entry.
    valid_indices = {e.index for e in non_empty}
    for idx in actual_indices:
        assert idx in valid_indices, (
            f"clip.entry_index {idx} has no matching non-empty entry"
        )


# ---------------------------------------------------------------------------
# Property 21 — voice_id 透传 (task 7.6)
# ---------------------------------------------------------------------------


@given(
    entries=_subtitle_entries(
        min_size=1,
        max_size=6,
        text_strategy=_NONEMPTY_CHINESE_TEXT,  # guarantee ≥1 non-empty entry
    ),
    voice_id=_VOICE_ID,
    supports_batch=st.booleans(),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_property_21_voice_id_passthrough(
    entries: list[SubtitleEntry],
    voice_id: str,
    supports_batch: bool,
) -> None:
    """**Validates: Requirement 6.5**

    The ``voice_id`` supplied to :meth:`TTSEngine.synthesize` SHALL be
    passed verbatim to every provider invocation (``synth`` or
    ``synth_batch``).
    """
    registry, cls = _build_recording_registry(supports_batch=supports_batch)
    engine = TTSEngine(registry)

    config = ProviderConfig(
        endpoint="https://stub",
        credential="stub",
        extra={"default_voice": "v-default-should-not-be-used"},
    )

    clips = await engine.synthesize(
        entries=entries,
        voice_id=voice_id,
        provider_type="recording",
        config=config,
        rate_limit_config=_make_rate_limit_config(
            supports_batch=supports_batch,
        ),
    )

    assert len(clips) >= 1, "test requires at least one non-empty entry"

    provider = cls._last
    assert provider is not None, "recording provider was never instantiated"

    observed_voices: list[str] = []
    if supports_batch:
        assert provider.batch_calls, (
            "batch-capable provider should have been called via synth_batch"
        )
        observed_voices.extend(v for _, v in provider.batch_calls)
    else:
        assert provider.synth_calls, (
            "single-shot provider should have been called via synth"
        )
        observed_voices.extend(v for _, v in provider.synth_calls)

    assert observed_voices, "no provider calls captured"
    assert all(v == voice_id for v in observed_voices), (
        f"voice_id not passed through verbatim: "
        f"expected all=={voice_id!r}, got {observed_voices!r}"
    )


# ---------------------------------------------------------------------------
# Property 25 — 语音合成阶段进度事件完整性 (task 7.7)
# ---------------------------------------------------------------------------


@given(
    entries=_subtitle_entries(
        min_size=1,
        max_size=8,
        text_strategy=_MIXED_CHINESE_TEXT,
    ),
    batch_size=st.integers(min_value=1, max_value=5),
    supports_batch=st.booleans(),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_property_25_tts_progress_event_completeness(
    entries: list[SubtitleEntry],
    batch_size: int,
    supports_batch: bool,
) -> None:
    """**Validates: Requirement 11.3**

    For any legal list of entries (under ``subtitle_and_dubbing``-mode
    semantics, where ``TTSEngine`` is actually exercised), the engine's
    progress events SHALL satisfy:

    - every event's ``stage == "tts"``
    - ``completed`` is monotonically non-decreasing
    - final ``completed`` equals the count of non-empty entries
    - every event reports ``total == count of non-empty entries``

    Blank-only inputs are a degenerate corner case the engine short-
    circuits without emitting events; the property skips that case
    rather than encoding a separate assertion.
    """
    non_empty = [e for e in entries if not _is_blank(e.text)]
    if not non_empty:
        # Degenerate: engine returns [] without touching the provider
        # or the reporter. Nothing to assert about progress events.
        return

    registry, _ = _build_recording_registry(supports_batch=supports_batch)
    reporter = _InMemoryReporter()
    engine = TTSEngine(registry, reporter=reporter)

    config = ProviderConfig(
        endpoint="https://stub",
        credential="stub",
        extra={"default_voice": "v-default"},
    )

    clips = await engine.synthesize(
        entries=entries,
        voice_id="v-test",
        provider_type="recording",
        config=config,
        rate_limit_config=_make_rate_limit_config(
            batch_size_initial=batch_size,
            batch_size_max=max(batch_size, 1),
            supports_batch=supports_batch,
        ),
    )
    assert len(clips) == len(non_empty)

    tts_events = [e for e in reporter.events if e.stage == "tts"]
    assert tts_events, "expected at least one tts progress event"

    total = len(non_empty)
    completeds: list[int] = []
    for ev in tts_events:
        assert ev.stage == "tts"
        assert ev.total == total, (
            f"expected total={total} on every tts event, got {ev.total}"
        )
        assert ev.completed is not None
        completeds.append(ev.completed)

    # Monotonic non-decreasing.
    for a, b in zip(completeds, completeds[1:]):
        assert b >= a, f"completed regressed: {a} -> {b}"

    # Final value reaches the count of non-empty entries.
    assert completeds[-1] == total, (
        f"final completed {completeds[-1]} != total {total}"
    )
