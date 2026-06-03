"""Unit tests for :class:`translation_dubbing_skill.tts.TTSEngine`.

Covers the behaviours task 7.1 promises to its callers:

- Empty / all-whitespace inputs short-circuit without producing clips.
- Whitespace-only entries are skipped and never sent to the provider.
- Non-empty entries flow through the adaptive scheduler; output order
  matches the original entries' order.
- Batched and non-batched providers both work (the scheduler clamps to
  ``batch_size=1`` for the latter).
- ``voice_id`` defaults to ``config.extra["default_voice"]`` when the
  explicit argument is missing.
- Contract violations (wrong tuple shape, non-bytes audio, negative
  ``duration_ms``) raise :class:`ProviderContractViolationError` with
  ``stage="tts"``.
- :class:`ProviderNotRegisteredError` propagates verbatim; other
  initialize-time failures wrap into :class:`ProviderUnavailableError`
  (``phase="initialize"``, ``stage="tts"``).
- Scheduler retry-budget exhaustion wraps into :class:`TTSError` with
  context carrying entry indices, provider type, and reason.
- Progress events are reported per batch, monotonically non-decreasing,
  and terminate at the count of non-empty entries.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from translation_dubbing_skill.errors import (
    ProviderContractViolationError,
    ProviderNotRegisteredError,
    ProviderUnavailableError,
    TTSError,
)
from translation_dubbing_skill.models import (
    AudioClip,
    ProgressEvent,
    ProviderConfig,
    SubtitleEntry,
)
from translation_dubbing_skill.providers import ProviderRegistry
from translation_dubbing_skill.scheduler import (
    ProviderRateLimitConfig,
    RateLimitError,
)
from translation_dubbing_skill.tts import TTSEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rate_limit_config(**overrides: Any) -> ProviderRateLimitConfig:
    base: dict[str, Any] = dict(
        batch_size_initial=4,
        batch_size_min=1,
        batch_size_max=10,
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
    base.update(overrides)
    return ProviderRateLimitConfig(**base)


def _entry(index: int, start: int, end: int, text: str) -> SubtitleEntry:
    return SubtitleEntry(index=index, start_ms=start, end_ms=end, text=text)


class _InMemoryReporter:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def report(self, event: ProgressEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Test doubles — providers
# ---------------------------------------------------------------------------


class _BatchProvider:
    """A batch-capable provider that returns deterministic bytes/durations."""

    provider_type: ClassVar[str] = "batch"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    def __init__(self) -> None:
        self.initialized_config: ProviderConfig | None = None
        self.received_batches: list[list[str]] = []
        self.received_voices: list[str] = []

    def initialize(self, config: ProviderConfig) -> None:
        self.initialized_config = config

    def size_of(self, text: str) -> int:
        return len(text)

    async def synth(
        self, text: str, voice_id: str
    ) -> tuple[bytes, int]:  # pragma: no cover - unused
        raise AssertionError("batch provider should go through synth_batch")

    async def synth_batch(
        self,
        texts: list[str],
        voice_id: str,
    ) -> list[tuple[bytes, int]]:
        self.received_batches.append(list(texts))
        self.received_voices.append(voice_id)
        return [(t.encode("utf-8"), max(0, len(t) * 10)) for t in texts]


class _SingleProvider:
    """A non-batch provider that fulfils only :meth:`synth`."""

    provider_type: ClassVar[str] = "single"
    supports_batch: ClassVar[bool] = False
    payload_unit: ClassVar[str] = "chars"

    def __init__(self) -> None:
        self.received_calls: list[tuple[str, str]] = []

    def initialize(self, config: ProviderConfig) -> None:
        pass

    def size_of(self, text: str) -> int:
        return len(text)

    async def synth(self, text: str, voice_id: str) -> tuple[bytes, int]:
        self.received_calls.append((text, voice_id))
        return (text.encode("utf-8"), max(0, len(text) * 10))

    async def synth_batch(  # pragma: no cover - unused
        self, texts: list[str], voice_id: str
    ) -> list[tuple[bytes, int]]:
        raise AssertionError("single-shot provider should not be batched")


class _FailingInitProvider:
    provider_type: ClassVar[str] = "failing-init"
    supports_batch: ClassVar[bool] = False
    payload_unit: ClassVar[str] = "chars"

    def initialize(self, config: ProviderConfig) -> None:
        raise RuntimeError("boom: cannot reach tts endpoint")

    def size_of(self, text: str) -> int:  # pragma: no cover - unreached
        return len(text)

    async def synth(  # pragma: no cover - unreached
        self, text: str, voice_id: str
    ) -> tuple[bytes, int]:
        raise AssertionError("unreachable")


class _AlwaysRateLimitedProvider:
    provider_type: ClassVar[str] = "always-429"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    def initialize(self, config: ProviderConfig) -> None:
        pass

    def size_of(self, text: str) -> int:
        return len(text)

    async def synth_batch(
        self, texts: list[str], voice_id: str
    ) -> list[tuple[bytes, int]]:
        raise RateLimitError("throttled", retry_after=0.0)

    async def synth(  # pragma: no cover - unused
        self, text: str, voice_id: str
    ) -> tuple[bytes, int]:
        raise RateLimitError("throttled", retry_after=0.0)


class _ContractBreakingProvider:
    """Return-shape / type / duration violations.

    The ``mode`` class attribute is set before the test calls
    :meth:`TTSEngine.synthesize` so each instance honours the chosen
    violation strategy.
    """

    provider_type: ClassVar[str] = "violator"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"
    mode: ClassVar[str] = "shape"

    def initialize(self, config: ProviderConfig) -> None:
        pass

    def size_of(self, text: str) -> int:
        return len(text)

    async def synth_batch(
        self, texts: list[str], voice_id: str
    ) -> list[Any]:
        if self.mode == "shape":
            # Return single scalar instead of (bytes, int) tuple.
            return [b"audio" for _ in texts]
        if self.mode == "audio_type":
            return [("not-bytes", 10) for _ in texts]
        if self.mode == "duration_type":
            return [(b"audio", "ten") for _ in texts]
        if self.mode == "negative_duration":
            return [(b"audio", -1) for _ in texts]
        raise AssertionError(f"unknown mode: {self.mode}")

    async def synth(  # pragma: no cover - unused in these tests
        self, text: str, voice_id: str
    ) -> tuple[bytes, int]:
        raise AssertionError("unused")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_batch() -> ProviderRegistry:
    r = ProviderRegistry()
    r.register("tts", "batch", _BatchProvider)
    return r


@pytest.fixture
def registry_single() -> ProviderRegistry:
    r = ProviderRegistry()
    r.register("tts", "single", _SingleProvider)
    return r


@pytest.fixture
def config() -> ProviderConfig:
    return ProviderConfig(
        endpoint="https://example",
        credential="secret",
        extra={"default_voice": "zh-CN-XiaoxiaoNeural"},
    )


# ---------------------------------------------------------------------------
# Empty / fast paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_entries_returns_empty_without_provider(
    registry_batch: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    engine = TTSEngine(registry_batch)
    out = await engine.synthesize(
        entries=[],
        voice_id="v1",
        provider_type="batch",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )
    assert out == []


@pytest.mark.asyncio
async def test_all_whitespace_entries_produce_no_clips(
    registry_batch: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    engine = TTSEngine(registry_batch)
    entries = [
        _entry(1, 0, 1_000, ""),
        _entry(2, 1_000, 2_000, "   "),
        _entry(3, 2_000, 3_000, "\t\n"),
    ]
    out = await engine.synthesize(
        entries=entries,
        voice_id="v1",
        provider_type="batch",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )
    assert out == []


@pytest.mark.asyncio
async def test_whitespace_entries_are_not_sent_to_provider(
    registry_batch: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    engine = TTSEngine(registry_batch)
    instances: list[_BatchProvider] = []

    class _Tracked(_BatchProvider):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

    registry_batch.register("tts", "batch", _Tracked)

    entries = [
        _entry(1, 0, 1_000, ""),
        _entry(2, 1_000, 2_000, "你好"),
        _entry(3, 2_000, 3_000, "  "),
        _entry(4, 3_000, 4_000, "世界"),
    ]
    clips = await engine.synthesize(
        entries=entries,
        voice_id="v1",
        provider_type="batch",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )

    assert len(instances) == 1
    sent_texts = [t for batch in instances[0].received_batches for t in batch]
    assert sent_texts == ["你好", "世界"]
    # And clips are the two non-empty entries in original order.
    assert [c.entry_index for c in clips] == [2, 4]
    assert [c.start_ms for c in clips] == [1_000, 3_000]
    assert [c.end_ms for c in clips] == [2_000, 4_000]


# ---------------------------------------------------------------------------
# Success path: order + batching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_provider_preserves_order_and_content(
    registry_batch: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    engine = TTSEngine(registry_batch)
    entries = [
        _entry(1, 0, 1_000, "你好"),
        _entry(2, 1_000, 2_000, "世界"),
        _entry(3, 2_000, 3_000, "再见"),
    ]
    clips = await engine.synthesize(
        entries=entries,
        voice_id="v1",
        provider_type="batch",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )

    assert len(clips) == 3
    assert [c.entry_index for c in clips] == [1, 2, 3]
    assert [c.audio for c in clips] == [
        "你好".encode("utf-8"),
        "世界".encode("utf-8"),
        "再见".encode("utf-8"),
    ]
    assert all(c.duration_ms >= 0 for c in clips)


@pytest.mark.asyncio
async def test_non_batch_provider_calls_synth_once_per_entry(
    registry_single: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    engine = TTSEngine(registry_single)
    instances: list[_SingleProvider] = []

    class _Tracked(_SingleProvider):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

    registry_single.register("tts", "single", _Tracked)

    entries = [
        _entry(1, 0, 1_000, "你好"),
        _entry(2, 1_000, 2_000, "世界"),
    ]
    clips = await engine.synthesize(
        entries=entries,
        voice_id="v-single",
        provider_type="single",
        config=config,
        rate_limit_config=_make_rate_limit_config(supports_batch=False),
    )

    assert len(instances) == 1
    # The scheduler should have broken inputs into singleton batches;
    # synth is called exactly once per non-empty entry.
    assert len(instances[0].received_calls) == 2
    assert [t for t, _ in instances[0].received_calls] == ["你好", "世界"]
    # voice_id threaded through every call.
    assert {v for _, v in instances[0].received_calls} == {"v-single"}
    # Clips in original order.
    assert [c.entry_index for c in clips] == [1, 2]


# ---------------------------------------------------------------------------
# voice_id defaulting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_voice_used_when_voice_id_not_provided(
    registry_batch: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    engine = TTSEngine(registry_batch)
    instances: list[_BatchProvider] = []

    class _Tracked(_BatchProvider):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

    registry_batch.register("tts", "batch", _Tracked)

    await engine.synthesize(
        entries=[_entry(1, 0, 1_000, "你好")],
        voice_id=None,
        provider_type="batch",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )

    assert instances[0].received_voices == ["zh-CN-XiaoxiaoNeural"]


@pytest.mark.asyncio
async def test_missing_voice_id_and_default_raises_value_error(
    registry_batch: ProviderRegistry,
) -> None:
    engine = TTSEngine(registry_batch)
    with pytest.raises(ValueError, match="voice_id"):
        await engine.synthesize(
            entries=[_entry(1, 0, 1_000, "你好")],
            voice_id=None,
            provider_type="batch",
            config=ProviderConfig(endpoint="https://x", credential="k"),
            rate_limit_config=_make_rate_limit_config(),
        )


@pytest.mark.asyncio
async def test_explicit_voice_id_beats_default(
    registry_batch: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    engine = TTSEngine(registry_batch)
    instances: list[_BatchProvider] = []

    class _Tracked(_BatchProvider):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

    registry_batch.register("tts", "batch", _Tracked)

    await engine.synthesize(
        entries=[_entry(1, 0, 1_000, "你好")],
        voice_id="override-voice",
        provider_type="batch",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )

    assert instances[0].received_voices == ["override-voice"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_not_registered_propagates_with_tts_stage(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    engine = TTSEngine(registry)
    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        await engine.synthesize(
            entries=[_entry(1, 0, 1_000, "你好")],
            voice_id="v",
            provider_type="ghost",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )
    assert excinfo.value.stage == "tts"
    assert excinfo.value.context["requested_type"] == "ghost"
    assert excinfo.value.context["kind"] == "tts"


@pytest.mark.asyncio
async def test_initialize_failure_wraps_as_provider_unavailable(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    registry.register("tts", "failing-init", _FailingInitProvider)
    engine = TTSEngine(registry)

    with pytest.raises(ProviderUnavailableError) as excinfo:
        await engine.synthesize(
            entries=[_entry(1, 0, 1_000, "你好")],
            voice_id="v",
            provider_type="failing-init",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )
    err = excinfo.value
    assert err.stage == "tts"
    assert err.context["provider_type"] == "failing-init"
    assert err.context["phase"] == "initialize"
    assert "boom" in err.context["reason"]


@pytest.mark.asyncio
async def test_scheduler_exhaustion_wraps_as_tts_error(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    registry.register("tts", "always-429", _AlwaysRateLimitedProvider)
    engine = TTSEngine(registry)

    with pytest.raises(TTSError) as excinfo:
        await engine.synthesize(
            entries=[
                _entry(1, 0, 1_000, ""),
                _entry(2, 1_000, 2_000, "你好"),
                _entry(3, 2_000, 3_000, "世界"),
            ],
            voice_id="v",
            provider_type="always-429",
            config=config,
            rate_limit_config=_make_rate_limit_config(
                max_retries=1,
                batch_size_initial=2,
                batch_size_max=2,
            ),
        )
    err = excinfo.value
    assert err.stage == "tts"
    assert err.context["provider_type"] == "always-429"
    # Indices are positions within the non-empty list (0 and 1).
    assert err.context["entry_indices"]


# ---------------------------------------------------------------------------
# Contract violations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["shape", "audio_type", "duration_type", "negative_duration"],
)
async def test_return_shape_violation_raises_contract_error(
    mode: str,
    config: ProviderConfig,
) -> None:
    _ContractBreakingProvider.mode = mode
    registry = ProviderRegistry()
    registry.register("tts", "violator", _ContractBreakingProvider)
    engine = TTSEngine(registry)

    with pytest.raises(ProviderContractViolationError) as excinfo:
        await engine.synthesize(
            entries=[_entry(1, 0, 1_000, "你好")],
            voice_id="v",
            provider_type="violator",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )
    assert excinfo.value.stage == "tts"
    assert excinfo.value.context["provider_type"] == "violator"
    assert "violated_clause" in excinfo.value.context


# ---------------------------------------------------------------------------
# Progress events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_events_monotonic_and_terminate_at_total(
    registry_batch: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    reporter = _InMemoryReporter()
    engine = TTSEngine(registry_batch, reporter=reporter)
    entries = [
        _entry(1, 0, 1_000, ""),  # skipped
        _entry(2, 1_000, 2_000, "你好"),
        _entry(3, 2_000, 3_000, "世界"),
        _entry(4, 3_000, 4_000, "再见"),
    ]
    clips = await engine.synthesize(
        entries=entries,
        voice_id="v",
        provider_type="batch",
        config=config,
        rate_limit_config=_make_rate_limit_config(
            batch_size_initial=2, batch_size_max=2
        ),
    )
    assert len(clips) == 3

    tts_events = [e for e in reporter.events if e.stage == "tts"]
    assert tts_events, "expected at least one TTS progress event"
    # completed is monotonic non-decreasing.
    completed_values = [e.completed for e in tts_events]
    assert completed_values == sorted(completed_values)
    # Total always equals the count of non-empty entries.
    assert all(e.total == 3 for e in tts_events)
    # Final event reaches total.
    assert tts_events[-1].completed == 3


@pytest.mark.asyncio
async def test_no_reporter_means_no_events_emitted(
    registry_batch: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    # Just ensure it runs cleanly with reporter=None.
    engine = TTSEngine(registry_batch, reporter=None)
    out = await engine.synthesize(
        entries=[_entry(1, 0, 1_000, "你好")],
        voice_id="v",
        provider_type="batch",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )
    assert len(out) == 1
    assert isinstance(out[0], AudioClip)
