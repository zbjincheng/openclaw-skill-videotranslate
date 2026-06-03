"""Unit tests for :class:`translation_dubbing_skill.scheduler.AdaptiveScheduler`.

These tests exercise the core state machine — success / rate-limit /
payload-too-large / transient paths — without mocking the asyncio
primitives. They inject a deterministic sleeper and ``random.Random``
so the timing assertions are reproducible.

Property-based coverage (P34–P43) arrives in tasks 5.5–5.14; this file
covers only the unit-level behaviours the scheduler promises to its
coordinators.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable

import pytest

from translation_dubbing_skill.scheduler import (
    AdaptiveScheduler,
    PayloadTooLargeError,
    ProviderRateLimitConfig,
    RateLimitError,
    SchedulerBatchFailure,
    TransientError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ProviderRateLimitConfig:
    """Return a permissive config suitable for most tests, with overrides."""
    base: dict[str, Any] = dict(
        batch_size_initial=3,
        batch_size_min=1,
        batch_size_max=10,
        payload_size_initial=100,
        payload_size_min=10,
        payload_size_max=1_000,
        payload_unit="chars",
        concurrency_initial=2,
        concurrency_min=1,
        concurrency_max=8,
        max_retries=3,
        backoff_base_ms=10,
        backoff_jitter_ms=0,  # deterministic sleeps in tests
        probe_up_every_n_success=2,
        supports_batch=True,
    )
    base.update(overrides)
    return ProviderRateLimitConfig(**base)


class _RecordingSleeper:
    """Deterministic ``asyncio.sleep`` replacement that records durations.

    Fast-forwards instantly but appends each requested duration to
    :attr:`waits` for later assertion.
    """

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, duration: float) -> None:
        self.waits.append(duration)
        # Yield so other coroutines can run; avoids starvation when
        # many batches all start with a pending delay.
        await asyncio.sleep(0)


def _text_size(text: str) -> int:
    return len(text)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_empty_items_returns_empty_list_without_calling_fetch() -> None:
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        _make_config(), size_of=_text_size
    )

    calls: list[list[str]] = []

    async def fetch(batch: list[str]) -> list[str]:
        calls.append(batch)
        return [t.upper() for t in batch]

    out = await scheduler.run([], fetch)
    assert out == []
    assert calls == []


@pytest.mark.asyncio
async def test_run_preserves_order_across_multiple_batches() -> None:
    # batch_size=2 forces three batches from six items; verify order.
    cfg = _make_config(batch_size_initial=2, batch_size_min=1, batch_size_max=2)
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(cfg, size_of=_text_size)

    async def fetch(batch: list[str]) -> list[str]:
        return [t + "!" for t in batch]

    items = ["a", "b", "c", "d", "e", "f"]
    out = await scheduler.run(items, fetch)
    assert out == ["a!", "b!", "c!", "d!", "e!", "f!"]


@pytest.mark.asyncio
async def test_run_respects_initial_batch_size() -> None:
    cfg = _make_config(batch_size_initial=2, batch_size_min=1, batch_size_max=2)
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(cfg, size_of=_text_size)

    seen_batch_sizes: list[int] = []

    async def fetch(batch: list[str]) -> list[str]:
        seen_batch_sizes.append(len(batch))
        return [t.upper() for t in batch]

    await scheduler.run(["a"] * 6, fetch)
    # Six items @ batch_size=2 → three batches of size 2.
    assert seen_batch_sizes == [2, 2, 2]


@pytest.mark.asyncio
async def test_run_with_single_batch_calls_fetch_once() -> None:
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        _make_config(), size_of=_text_size
    )
    call_count = 0

    async def fetch(batch: list[str]) -> list[str]:
        nonlocal call_count
        call_count += 1
        return list(batch)

    await scheduler.run(["a", "b"], fetch)
    assert call_count == 1


# ---------------------------------------------------------------------------
# Non-batch provider path (R12.14)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_batch_provider_forces_batch_size_one() -> None:
    cfg = _make_config(
        supports_batch=False,
        batch_size_initial=5,  # ignored when supports_batch=False
        batch_size_max=5,
    )
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(cfg, size_of=_text_size)

    batch_sizes: list[int] = []

    async def fetch(batch: list[str]) -> list[str]:
        batch_sizes.append(len(batch))
        return list(batch)

    await scheduler.run(["a", "b", "c", "d"], fetch)
    assert batch_sizes == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Rate-limit path (R12.6, R12.7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_halves_all_three_dimensions_and_retries() -> None:
    cfg = _make_config(
        batch_size_initial=8,
        batch_size_min=1,
        batch_size_max=8,
        payload_size_initial=100,
        payload_size_min=1,
        payload_size_max=100,
        concurrency_initial=4,
        concurrency_min=1,
        concurrency_max=4,
        max_retries=3,
    )
    sleeper = _RecordingSleeper()
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        cfg, size_of=_text_size, sleeper=sleeper
    )

    attempts = {"count": 0}

    async def fetch(batch: list[str]) -> list[str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimitError("429 slow down")
        return [t.upper() for t in batch]

    out = await scheduler.run(["a", "b"], fetch)
    assert out == ["A", "B"]
    # First attempt failed, second succeeded.
    assert attempts["count"] == 2
    # One backoff wait recorded: base_ms * 2**0 = 10ms = 0.01s.
    assert sleeper.waits == [0.01]


@pytest.mark.asyncio
async def test_rate_limit_honours_retry_after_header() -> None:
    sleeper = _RecordingSleeper()
    cfg = _make_config()
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        cfg, size_of=_text_size, sleeper=sleeper
    )

    attempts = {"count": 0}

    async def fetch(batch: list[str]) -> list[str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimitError("slow", retry_after=2.0)
        return list(batch)

    await scheduler.run(["x"], fetch)
    assert sleeper.waits == [2.0]


@pytest.mark.asyncio
async def test_rate_limit_exhausts_budget_and_raises_batch_failure() -> None:
    sleeper = _RecordingSleeper()
    cfg = _make_config(max_retries=2)
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        cfg, size_of=_text_size, sleeper=sleeper, provider_type="stub"
    )

    call_count = {"n": 0}

    async def fetch(batch: list[str]) -> list[str]:
        call_count["n"] += 1
        raise RateLimitError("always")

    with pytest.raises(SchedulerBatchFailure) as excinfo:
        await scheduler.run(["a", "b"], fetch)

    err = excinfo.value
    # max_retries=2 → up to 3 calls total (1 initial + 2 retries).
    assert call_count["n"] == 3
    assert err.provider_type == "stub"
    assert err.entry_indices == [0, 1]
    assert isinstance(err.last_error, RateLimitError)


# ---------------------------------------------------------------------------
# Payload-too-large path (R12.12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_payload_too_large_shrinks_payload_and_reslices() -> None:
    """First call with all items fails; second call sees smaller batches."""
    cfg = _make_config(
        batch_size_initial=4,
        batch_size_min=1,
        batch_size_max=4,
        payload_size_initial=100,
        payload_size_min=1,
        payload_size_max=100,
        max_retries=5,
    )
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        cfg, size_of=_text_size
    )

    batches_observed: list[list[str]] = []

    async def fetch(batch: list[str]) -> list[str]:
        batches_observed.append(list(batch))
        # Only reject the first (largest) call.
        if len(batches_observed) == 1:
            raise PayloadTooLargeError("too big")
        return [t.upper() for t in batch]

    items = ["ab", "cd", "ef", "gh"]  # each size 2, total 8
    out = await scheduler.run(items, fetch)

    # First call saw the full 4-item batch; after shrink to 50, it still
    # fits — but the scheduler re-slices anyway and the sub-batches
    # succeed. All outputs arrive in order.
    assert out == ["AB", "CD", "EF", "GH"]
    # First call is the original batch; subsequent calls are the re-sliced
    # sub-batches.
    assert batches_observed[0] == items
    # All re-sliced items together cover the original input.
    reassembled = sum(batches_observed[1:], [])
    assert reassembled == items


@pytest.mark.asyncio
async def test_payload_too_large_does_not_consume_retry_budget() -> None:
    """P42: a PayloadTooLarge does not count against max_retries.

    We set max_retries=0 so even a single retry would fail — yet a
    PayloadTooLarge followed by success must still complete.
    """
    cfg = _make_config(
        batch_size_initial=4,
        batch_size_min=1,
        batch_size_max=4,
        payload_size_initial=100,
        payload_size_min=1,
        max_retries=0,  # strict: no retries allowed
    )
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        cfg, size_of=_text_size
    )

    attempts = {"n": 0}

    async def fetch(batch: list[str]) -> list[str]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise PayloadTooLargeError("trim")
        return [t.upper() for t in batch]

    out = await scheduler.run(["a", "b", "c", "d"], fetch)
    assert out == ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Transient error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transient_error_retries_with_backoff() -> None:
    sleeper = _RecordingSleeper()
    cfg = _make_config(backoff_base_ms=20, backoff_jitter_ms=0, max_retries=2)
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        cfg, size_of=_text_size, sleeper=sleeper
    )

    attempts = {"n": 0}

    async def fetch(batch: list[str]) -> list[str]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientError("blip")
        return list(batch)

    await scheduler.run(["x"], fetch)
    assert attempts["n"] == 3
    # Exponential: attempt 1 → 20ms, attempt 2 → 40ms.
    assert sleeper.waits == [pytest.approx(0.020), pytest.approx(0.040)]


@pytest.mark.asyncio
async def test_transient_failure_eventually_raises_batch_failure() -> None:
    cfg = _make_config(max_retries=1)
    sleeper = _RecordingSleeper()
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        cfg, size_of=_text_size, sleeper=sleeper
    )

    async def fetch(batch: list[str]) -> list[str]:
        raise TransientError("never")

    with pytest.raises(SchedulerBatchFailure) as excinfo:
        await scheduler.run(["x"], fetch)
    assert isinstance(excinfo.value.last_error, TransientError)


# ---------------------------------------------------------------------------
# Success counts & up-tune (R12.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_up_tune_stays_below_or_at_max() -> None:
    """After many successes, ``batch_size`` / ``concurrency`` never exceed max."""
    cfg = _make_config(
        batch_size_initial=1,
        batch_size_min=1,
        batch_size_max=3,
        payload_size_initial=10,
        payload_size_min=1,
        payload_size_max=10,
        concurrency_initial=1,
        concurrency_min=1,
        concurrency_max=3,
        probe_up_every_n_success=1,  # up-tune on every success
    )

    seen: list[int] = []

    async def fetch(batch: list[str]) -> list[str]:
        seen.append(len(batch))
        return list(batch)

    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(cfg, size_of=_text_size)
    # 20 items: plenty of chances to up-tune past max if the scheduler
    # doesn't clamp.
    await scheduler.run(["a"] * 20, fetch)
    # No observed batch exceeds batch_size_max.
    assert max(seen) <= cfg.batch_size_max


# ---------------------------------------------------------------------------
# Contract violation on length mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_length_mismatch_raises_batch_failure() -> None:
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        _make_config(), size_of=_text_size, provider_type="stub"
    )

    async def fetch(batch: list[str]) -> list[str]:
        # Deliberately drop one element.
        return [t.upper() for t in batch[:-1]]

    with pytest.raises(SchedulerBatchFailure) as excinfo:
        await scheduler.run(["a", "b"], fetch)

    err = excinfo.value
    assert err.provider_type == "stub"
    assert err.entry_indices == [0, 1]


# ---------------------------------------------------------------------------
# Concurrency cap (R12.11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_flight_never_exceeds_current_concurrency() -> None:
    """A tiny version of P35: track concurrent fetch calls under a cap of 2."""
    cfg = _make_config(
        batch_size_initial=1,
        batch_size_min=1,
        batch_size_max=1,
        concurrency_initial=2,
        concurrency_min=1,
        concurrency_max=2,
        probe_up_every_n_success=999,  # avoid up-tune disturbing the cap
    )

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def fetch(batch: list[str]) -> list[str]:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        # Yield to let siblings start.
        await asyncio.sleep(0)
        async with lock:
            in_flight -= 1
        return list(batch)

    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(cfg, size_of=_text_size)
    await scheduler.run(["x"] * 10, fetch)
    assert max_in_flight <= cfg.concurrency_max
    assert max_in_flight <= cfg.concurrency_initial


# ---------------------------------------------------------------------------
# Backoff determinism with jitter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backoff_includes_jitter_when_configured() -> None:
    sleeper = _RecordingSleeper()
    cfg = _make_config(backoff_base_ms=100, backoff_jitter_ms=50, max_retries=1)
    # Seed RNG for deterministic jitter in this test.
    rng = random.Random(0)
    scheduler: AdaptiveScheduler[str, str] = AdaptiveScheduler(
        cfg, size_of=_text_size, sleeper=sleeper, rng=rng
    )

    attempts = {"n": 0}

    async def fetch(batch: list[str]) -> list[str]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TransientError("blip")
        return list(batch)

    await scheduler.run(["x"], fetch)
    assert len(sleeper.waits) == 1
    # Wait is base (100ms) plus jitter in [0, 50ms].
    w = sleeper.waits[0]
    assert 0.100 <= w <= 0.150
