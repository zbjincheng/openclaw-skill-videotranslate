"""Unit tests for :class:`translation_dubbing_skill.translation.Translator`.

Covers the behaviours task 6.1 promises to its callers:

- Empty entries short-circuit without touching the provider.
- Whitespace-only entries are fast-pathed to empty strings and never
  leave the coordinator.
- Non-empty entries flow through the adaptive scheduler and the
  output merges back in the original order.
- Structural contract violations (length / index / timestamp) and
  semantic contract violations (empty-for-nonempty, non-Chinese)
  raise :class:`ProviderContractViolationError`.
- :class:`ProviderNotRegisteredError` propagates verbatim; other
  initialize-time failures wrap into :class:`ProviderUnavailableError`.
- Scheduler retry-budget exhaustion wraps into
  :class:`TranslationError` with the original entry indices + provider
  metadata.
- Progress is reported per batch, monotonically non-decreasing, and
  terminates at ``len(entries)``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from translation_dubbing_skill.errors import (
    ProviderContractViolationError,
    ProviderNotRegisteredError,
    ProviderUnavailableError,
    TranslationError,
)
from translation_dubbing_skill.models import (
    ProgressEvent,
    ProviderConfig,
    SubtitleEntry,
)
from translation_dubbing_skill.providers import ProviderRegistry
from translation_dubbing_skill.scheduler import (
    ProviderRateLimitConfig,
    RateLimitError,
)
from translation_dubbing_skill.translation import Translator


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
    """Records :class:`ProgressEvent` objects for later assertion."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def report(self, event: ProgressEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SuccessProvider:
    """Provider that translates each entry to a fixed Chinese prefix."""

    provider_type: ClassVar[str] = "success"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    def __init__(self) -> None:
        self.received_batches: list[list[SubtitleEntry]] = []
        self.initialized_config: ProviderConfig | None = None

    def initialize(self, config: ProviderConfig) -> None:
        self.initialized_config = config

    def size_of(self, text: str) -> int:
        return len(text)

    async def translate_batch(
        self,
        entries: list[SubtitleEntry],
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        self.received_batches.append(list(entries))
        return [
            SubtitleEntry(
                index=e.index,
                start_ms=e.start_ms,
                end_ms=e.end_ms,
                text=f"中文{e.index}",
            )
            for e in entries
        ]


class _FailingInitProvider:
    provider_type: ClassVar[str] = "failing-init"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    def initialize(self, config: ProviderConfig) -> None:
        raise RuntimeError("cannot reach endpoint")

    def size_of(self, text: str) -> int:  # pragma: no cover - unreached
        return len(text)

    async def translate_batch(  # pragma: no cover - unreached
        self,
        entries: list[SubtitleEntry],
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        raise AssertionError("provider was not expected to be called")


class _ContractBreakingProvider:
    """Provider producing configurable contract violations.

    ``mode`` is a class attribute so tests can set it before
    instantiation (the registry constructs the provider internally
    via its zero-arg constructor).
    """

    provider_type: ClassVar[str] = "violator"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"
    mode: ClassVar[str] = "length"

    def initialize(self, config: ProviderConfig) -> None:
        pass

    def size_of(self, text: str) -> int:
        return len(text)

    async def translate_batch(
        self,
        entries: list[SubtitleEntry],
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        if self.mode == "length":
            # Drop the last entry → scheduler catches length mismatch
            # and raises SchedulerBatchFailure, which the translator
            # maps into TranslationError. But the coordinator's OWN
            # length check fires first when the scheduler is not
            # involved. For the single-batch case here we return one
            # FEWER entry to exercise the scheduler's length check.
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text="中文",
                )
                for e in entries[:-1]
            ]
        if self.mode == "index":
            return [
                SubtitleEntry(
                    index=e.index + 100,  # wrong index
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text="中文",
                )
                for e in entries
            ]
        if self.mode == "timestamp":
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms + 1,  # shifted timestamps
                    end_ms=e.end_ms,
                    text="中文",
                )
                for e in entries
            ]
        if self.mode == "empty_output":
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text="",  # empty output for non-empty input
                )
                for e in entries
            ]
        if self.mode == "non_chinese":
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text="latin text only",
                )
                for e in entries
            ]
        raise AssertionError(f"unknown mode: {self.mode}")


class _AlwaysRateLimitedProvider:
    """Provider that perpetually signals rate-limit, forcing retry exhaustion."""

    provider_type: ClassVar[str] = "always-429"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    def initialize(self, config: ProviderConfig) -> None:
        pass

    def size_of(self, text: str) -> int:
        return len(text)

    async def translate_batch(
        self,
        entries: list[SubtitleEntry],
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        raise RateLimitError("throttled", retry_after=0.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_with_success(monkeypatch: pytest.MonkeyPatch) -> ProviderRegistry:
    """Return a fresh registry holding only ``_SuccessProvider``."""
    registry = ProviderRegistry()
    registry.register("translation", "success", _SuccessProvider)
    return registry


@pytest.fixture
def config() -> ProviderConfig:
    return ProviderConfig(endpoint="https://example", credential="secret")


# ---------------------------------------------------------------------------
# Empty / whitespace fast paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_entries_returns_empty_without_touching_provider(
    registry_with_success: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    translator = Translator(registry_with_success)
    out = await translator.translate(
        entries=[],
        provider_type="success",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )
    assert out == []


@pytest.mark.asyncio
async def test_all_whitespace_entries_skipped_and_returned_as_empty(
    registry_with_success: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    translator = Translator(registry_with_success)
    entries = [
        _entry(1, 0, 1_000, ""),
        _entry(2, 1_000, 2_000, "   "),
        _entry(3, 2_000, 3_000, "\t\n"),
    ]

    out = await translator.translate(
        entries=entries,
        provider_type="success",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )

    # Timestamps and indices preserved; all text emptied.
    assert len(out) == 3
    for i, (inp, res) in enumerate(zip(entries, out)):
        assert res.index == inp.index
        assert res.start_ms == inp.start_ms
        assert res.end_ms == inp.end_ms
        assert res.text == ""


@pytest.mark.asyncio
async def test_whitespace_entries_are_not_sent_to_provider(
    registry_with_success: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    translator = Translator(registry_with_success)
    # Register a fresh instance so we can inspect received_batches.
    provider_instances: list[_SuccessProvider] = []
    original_success = _SuccessProvider

    class _Tracked(original_success):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            provider_instances.append(self)

    registry_with_success.register("translation", "success", _Tracked)

    entries = [
        _entry(1, 0, 1_000, ""),
        _entry(2, 1_000, 2_000, "hello"),
        _entry(3, 2_000, 3_000, "  "),
        _entry(4, 3_000, 4_000, "world"),
    ]

    await translator.translate(
        entries=entries,
        provider_type="success",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )

    assert len(provider_instances) == 1
    received_texts: list[str] = [
        e.text
        for batch in provider_instances[0].received_batches
        for e in batch
    ]
    assert received_texts == ["hello", "world"]


# ---------------------------------------------------------------------------
# Success path: structure + order + merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_preserves_order_and_timestamps(
    registry_with_success: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    translator = Translator(registry_with_success)
    entries = [
        _entry(1, 0, 1_000, "hello"),
        _entry(2, 1_000, 2_000, ""),
        _entry(3, 2_000, 3_000, "world"),
        _entry(4, 3_000, 4_000, "foo"),
    ]

    out = await translator.translate(
        entries=entries,
        provider_type="success",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )

    assert len(out) == 4
    # Index / timestamp preservation across all positions.
    for inp, res in zip(entries, out):
        assert res.index == inp.index
        assert res.start_ms == inp.start_ms
        assert res.end_ms == inp.end_ms
    # Empty-input stays empty; non-empty become the stubbed Chinese.
    assert out[0].text == "中文1"
    assert out[1].text == ""
    assert out[2].text == "中文3"
    assert out[3].text == "中文4"


@pytest.mark.asyncio
async def test_success_passes_initialize_config_through(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    instances: list[_SuccessProvider] = []

    class _CapturingSuccess(_SuccessProvider):
        def __init__(self) -> None:
            super().__init__()
            instances.append(self)

    registry.register("translation", "success", _CapturingSuccess)
    translator = Translator(registry)

    await translator.translate(
        entries=[_entry(1, 0, 1_000, "hi")],
        provider_type="success",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )

    assert len(instances) == 1
    assert instances[0].initialized_config is config


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_not_registered_propagates_unchanged(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    translator = Translator(registry)

    with pytest.raises(ProviderNotRegisteredError) as excinfo:
        await translator.translate(
            entries=[_entry(1, 0, 1_000, "hi")],
            provider_type="ghost",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )

    err = excinfo.value
    assert err.context["requested_type"] == "ghost"
    assert err.context["kind"] == "translation"
    assert err.stage == "translating"


@pytest.mark.asyncio
async def test_initialize_failure_wraps_as_provider_unavailable(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    registry.register("translation", "failing-init", _FailingInitProvider)
    translator = Translator(registry)

    with pytest.raises(ProviderUnavailableError) as excinfo:
        await translator.translate(
            entries=[_entry(1, 0, 1_000, "hi")],
            provider_type="failing-init",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )

    err = excinfo.value
    assert err.stage == "translating"
    assert err.context["provider_type"] == "failing-init"
    assert err.context["phase"] == "initialize"
    assert "cannot reach endpoint" in err.context["reason"]


@pytest.mark.asyncio
async def test_scheduler_retry_exhaustion_wraps_as_translation_error(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    registry.register("translation", "always-429", _AlwaysRateLimitedProvider)
    translator = Translator(registry)

    entries = [
        _entry(1, 0, 1_000, ""),
        _entry(2, 1_000, 2_000, "hello"),
        _entry(3, 2_000, 3_000, "world"),
    ]

    with pytest.raises(TranslationError) as excinfo:
        await translator.translate(
            entries=entries,
            provider_type="always-429",
            config=config,
            rate_limit_config=_make_rate_limit_config(
                max_retries=1,  # keep the test fast
                batch_size_initial=2,
                batch_size_max=2,
            ),
        )

    err = excinfo.value
    assert err.stage == "translating"
    assert err.context["provider_type"] == "always-429"
    # The failing batch carried positions 1 and 2 in the ORIGINAL
    # entries list (whitespace entry #1 was filtered out before
    # scheduling, so non-empty positions [1, 2] map back to [1, 2]).
    assert set(err.context["entry_indices"]).issubset({1, 2})
    assert err.context["entry_indices"]  # non-empty
    assert "provider_reason" in err.context


# ---------------------------------------------------------------------------
# Contract violations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_violation_length_mismatch(
    config: ProviderConfig,
) -> None:
    # The scheduler itself detects len(outputs) != len(batch) and
    # raises SchedulerBatchFailure, which the translator wraps into
    # TranslationError. This is the P12-adjacent case for length.
    registry = ProviderRegistry()
    registry.register("translation", "violator", _ContractBreakingProvider)
    translator = Translator(registry)

    # Pre-set the mode on the class before instantiation.
    _ContractBreakingProvider.mode = "length"  # type: ignore[misc]

    entries = [
        _entry(1, 0, 1_000, "a"),
        _entry(2, 1_000, 2_000, "b"),
    ]

    with pytest.raises(TranslationError):
        await translator.translate(
            entries=entries,
            provider_type="violator",
            config=config,
            rate_limit_config=_make_rate_limit_config(
                max_retries=0,
                batch_size_initial=2,
                batch_size_max=2,
            ),
        )


@pytest.mark.asyncio
async def test_contract_violation_index_mismatch(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    registry.register("translation", "violator", _ContractBreakingProvider)
    translator = Translator(registry)
    _ContractBreakingProvider.mode = "index"  # type: ignore[misc]

    entries = [_entry(1, 0, 1_000, "a")]
    with pytest.raises(ProviderContractViolationError) as excinfo:
        await translator.translate(
            entries=entries,
            provider_type="violator",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )
    err = excinfo.value
    assert err.context["violated_clause"] == "index_mismatch"
    assert err.context["provider_type"] == "violator"


@pytest.mark.asyncio
async def test_contract_violation_timestamp_mismatch(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    registry.register("translation", "violator", _ContractBreakingProvider)
    translator = Translator(registry)
    _ContractBreakingProvider.mode = "timestamp"  # type: ignore[misc]

    entries = [_entry(1, 0, 1_000, "a")]
    with pytest.raises(ProviderContractViolationError) as excinfo:
        await translator.translate(
            entries=entries,
            provider_type="violator",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )
    assert excinfo.value.context["violated_clause"] == "start_ms_mismatch"


@pytest.mark.asyncio
async def test_contract_violation_empty_output_for_nonempty_input(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    registry.register("translation", "violator", _ContractBreakingProvider)
    translator = Translator(registry)
    _ContractBreakingProvider.mode = "empty_output"  # type: ignore[misc]

    entries = [_entry(1, 0, 1_000, "hello")]
    with pytest.raises(ProviderContractViolationError) as excinfo:
        await translator.translate(
            entries=entries,
            provider_type="violator",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )
    assert (
        excinfo.value.context["violated_clause"]
        == "empty_translation_for_nonempty_input"
    )


@pytest.mark.asyncio
async def test_contract_violation_non_chinese_output(
    config: ProviderConfig,
) -> None:
    registry = ProviderRegistry()
    registry.register("translation", "violator", _ContractBreakingProvider)
    translator = Translator(registry)
    _ContractBreakingProvider.mode = "non_chinese"  # type: ignore[misc]

    entries = [_entry(1, 0, 1_000, "hello")]
    with pytest.raises(ProviderContractViolationError) as excinfo:
        await translator.translate(
            entries=entries,
            provider_type="violator",
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )
    assert (
        excinfo.value.context["violated_clause"]
        == "non_simplified_chinese_output"
    )


# ---------------------------------------------------------------------------
# Progress reporting (R11.2, R12.16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_is_monotonic_and_terminates_at_total(
    registry_with_success: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    reporter = _InMemoryReporter()
    translator = Translator(registry_with_success, reporter=reporter)

    entries = [
        _entry(i, i * 1_000, (i + 1) * 1_000, f"line {i}")
        for i in range(1, 7)
    ]

    out = await translator.translate(
        entries=entries,
        provider_type="success",
        config=config,
        rate_limit_config=_make_rate_limit_config(
            batch_size_initial=2, batch_size_max=2
        ),
    )

    assert len(out) == 6
    assert reporter.events, "expected at least one progress event"

    completeds = [ev.completed for ev in reporter.events]
    # Monotonic non-decreasing.
    for a, b in zip(completeds, completeds[1:]):
        assert a is not None and b is not None
        assert b >= a
    # Every event reports the same total (len(entries)).
    for ev in reporter.events:
        assert ev.total == 6
        assert ev.stage == "translating"
    # Final event pins completed at total.
    assert reporter.events[-1].completed == 6


@pytest.mark.asyncio
async def test_progress_not_called_when_reporter_is_none(
    registry_with_success: ProviderRegistry,
    config: ProviderConfig,
) -> None:
    # The no-reporter path must not crash and must still produce correct output.
    translator = Translator(registry_with_success, reporter=None)
    out = await translator.translate(
        entries=[_entry(1, 0, 1_000, "hi")],
        provider_type="success",
        config=config,
        rate_limit_config=_make_rate_limit_config(),
    )
    assert len(out) == 1
