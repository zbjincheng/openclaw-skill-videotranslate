"""Property-based tests for :class:`AdaptiveScheduler` (P34–P43).

Implements tasks 5.5 through 5.14 of the spec: Hypothesis properties that
exercise the three-dimensional AIMD scheduler against the requirements
R12.1, R12.2, R12.5, R12.6, R12.7, R12.8, R12.9, R12.10, R12.11, R12.12,
R12.13, R12.14, R12.16.

The tests share a deterministic ``_RecordingSleeper`` and seeded
``random.Random`` so timing assertions are reproducible. A permissive
:func:`_make_config` helper builds a :class:`ProviderRateLimitConfig`
with sensible defaults; each property overrides only the knobs it cares
about.
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from translation_dubbing_skill.scheduler import (
    AdaptiveScheduler,
    PayloadTooLargeError,
    ProviderRateLimitConfig,
    RateLimitError,
    SchedulerBatchFailure,
    TransientError,
    make_batches,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> ProviderRateLimitConfig:
    """Return a permissive config with overrides applied on top."""
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
        backoff_jitter_ms=0,
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
        await asyncio.sleep(0)


def _text_size(text: str) -> int:
    """``size_of`` used throughout: count characters."""
    return len(text)


def _run_async(coro: Any) -> Any:
    """Run a coroutine on a fresh event loop and close it cleanly.

    Hypothesis drives these tests synchronously so every example needs
    its own event loop. Using :func:`asyncio.run` is tempting but
    leaves a stale-loop reference around long enough for pytest's
    ``filterwarnings=error`` to turn into a :class:`ResourceWarning`
    when a later test collects the garbage. The explicit create-and-
    close pattern below keeps every example self-contained.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Property 34 — Batch request count upper bound (task 5.5)
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=40),
    batch_size_initial=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_34_batch_request_count_upper_bound(
    n: int, batch_size_initial: int
) -> None:
    """**Validates: Requirements 12.1, 12.2, 12.10**

    Batch request count upper bound:

    - Without rate limiting, a counting stub is invoked at most
      ``ceil(n / batch_size_initial)`` times.
    - Under persistent rate limiting, the total call count does not
      exceed ``ceil(n / batch_size_min) + max_retries``.
    """
    items = [f"item-{i}" for i in range(n)]

    async def _run_unlimited() -> int:
        cfg = _make_config(
            batch_size_initial=batch_size_initial,
            batch_size_min=1,
            batch_size_max=max(batch_size_initial, 1),
            probe_up_every_n_success=10_000,  # disable up-tune
            payload_size_initial=10_000,
            payload_size_max=10_000,
        )
        calls = 0

        async def fetch(batch: list[str]) -> list[str]:
            nonlocal calls
            calls += 1
            return list(batch)

        sched: AdaptiveScheduler[str, str] = AdaptiveScheduler(
            cfg, size_of=_text_size
        )
        await sched.run(items, fetch)
        return calls

    unlimited_calls = _run_async(_run_unlimited())
    expected_max_unlimited = math.ceil(n / batch_size_initial)
    assert unlimited_calls <= expected_max_unlimited, (
        f"unlimited path issued {unlimited_calls} calls for n={n}, "
        f"batch_size_initial={batch_size_initial}; expected "
        f"<= ceil(n/batch_size_initial)={expected_max_unlimited}"
    )

    # Rate-limited scenario: the first batch the scheduler issues
    # rate-limits ``max_retries`` times before succeeding; every other
    # batch succeeds on its first attempt. The total call count must
    # satisfy ``R <= ceil(n / batch_size_min) + max_retries`` (the loose
    # upper bound stated by the spec).
    max_retries = 2

    async def _run_limited() -> int:
        cfg = _make_config(
            batch_size_initial=batch_size_initial,
            batch_size_min=1,
            batch_size_max=max(batch_size_initial, 1),
            concurrency_initial=1,
            concurrency_min=1,
            concurrency_max=2,
            max_retries=max_retries,
            backoff_base_ms=1,
            backoff_jitter_ms=0,
            probe_up_every_n_success=10_000,
            payload_size_initial=10_000,
            payload_size_max=10_000,
        )
        calls = 0
        sleeper = _RecordingSleeper()
        first_batch_key: dict[str, int] = {}

        async def fetch(batch: list[str]) -> list[str]:
            nonlocal calls
            calls += 1
            key = batch[0]
            seen = first_batch_key.get(key, 0)
            first_batch_key[key] = seen + 1
            # Only the very first original batch (smallest key) rate-limits
            # up to max_retries times before succeeding.
            if key == items[0] and seen < max_retries:
                raise RateLimitError("slow")
            return list(batch)

        sched: AdaptiveScheduler[str, str] = AdaptiveScheduler(
            cfg, size_of=_text_size, sleeper=sleeper
        )
        await sched.run(items, fetch)
        return calls

    limited_calls = _run_async(_run_limited())
    batch_size_min = 1
    expected_max_limited = math.ceil(n / batch_size_min) + max_retries
    assert limited_calls <= expected_max_limited, (
        f"limited path issued {limited_calls} calls; expected "
        f"<= ceil(n/batch_size_min) + max_retries = {expected_max_limited}"
    )



# ---------------------------------------------------------------------------
# Property 35 — Concurrency cap (task 5.6)
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=2, max_value=20),
    concurrency_initial=st.integers(min_value=1, max_value=4),
    concurrency_max=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_35_concurrency_never_exceeds_cap_for_any_provider_kind(
    n: int, concurrency_initial: int, concurrency_max: int
) -> None:
    """**Validates: Requirement 12.11**

    Concurrency cap applies uniformly to every provider kind:

    - A "slow LLM" config (large payloads, long simulated latency) must
      never exceed ``concurrency_max`` in-flight batches.
    - A "fast web" config (small payloads, minimal latency) must obey
      the same bound.

    At any instant, ``in_flight <= current_concurrency`` where
    ``current_concurrency <= concurrency_max``.
    """
    assume(concurrency_initial <= concurrency_max)

    items = [f"x-{i}" for i in range(n)]

    async def _run(supports_batch: bool, sleep_each: float) -> int:
        cfg = _make_config(
            batch_size_initial=1,
            batch_size_min=1,
            batch_size_max=1,
            concurrency_initial=concurrency_initial,
            concurrency_min=1,
            concurrency_max=concurrency_max,
            probe_up_every_n_success=10_000,
            supports_batch=supports_batch,
            payload_size_initial=1_000,
            payload_size_max=1_000,
            payload_size_min=1,
        )
        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def fetch(batch: list[str]) -> list[str]:
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                if in_flight > max_in_flight:
                    max_in_flight = in_flight
            # Yield control so siblings can start and inflight can spike.
            await asyncio.sleep(sleep_each)
            async with lock:
                in_flight -= 1
            return list(batch)

        sched: AdaptiveScheduler[str, str] = AdaptiveScheduler(
            cfg, size_of=_text_size
        )
        await sched.run(items, fetch)
        return max_in_flight

    # Slow LLM: long awaits inside fetch exaggerate overlap.
    max_llm = _run_async(_run(supports_batch=True, sleep_each=0.001))
    # Fast web-like service: minimal sleep.
    max_web = _run_async(_run(supports_batch=False, sleep_each=0))

    assert max_llm <= concurrency_max, (
        f"slow-LLM path saw max_in_flight={max_llm} > "
        f"concurrency_max={concurrency_max}"
    )
    assert max_web <= concurrency_max, (
        f"fast-web path saw max_in_flight={max_web} > "
        f"concurrency_max={concurrency_max}"
    )
    # Tighter: both should also respect the *initial* cap because
    # probe_up_every_n_success is huge (no up-tune).
    assert max_llm <= concurrency_initial
    assert max_web <= concurrency_initial



# ---------------------------------------------------------------------------
# Property 36 — Rate-limit feedback halves all three dimensions (task 5.7)
# ---------------------------------------------------------------------------


class _StateRecordingScheduler(AdaptiveScheduler[str, str]):
    """White-box subclass that captures AIMD state after every down-tune.

    The scheduler deliberately keeps its tuning state internal; this
    minimal observer records ``(batch_size, payload_size, concurrency)``
    snapshots after every rate-limit (or payload-too-large) triggered
    adjustment so the property tests can compare "before" vs "after"
    across a single run.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.before_snapshots: list[tuple[int, int, int]] = []
        self.after_snapshots: list[tuple[int, int, int]] = []

    async def _down_tune_all(self, state: Any) -> None:  # type: ignore[override]
        self.before_snapshots.append(
            (state.batch_size, state.payload_size, state.concurrency)
        )
        await super()._down_tune_all(state)
        self.after_snapshots.append(
            (state.batch_size, state.payload_size, state.concurrency)
        )


@given(
    batch_size_initial=st.integers(min_value=2, max_value=16),
    payload_size_initial=st.integers(min_value=4, max_value=200),
    concurrency_initial=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_36_rate_limit_halves_all_three_dimensions(
    batch_size_initial: int,
    payload_size_initial: int,
    concurrency_initial: int,
) -> None:
    """**Validates: Requirement 12.6**

    A rate-limit signal triggers multiplicative down-tune on every
    dimension, each floored at its ``*_min``::

        batch_size_new     <= max(batch_size_before  // 2, batch_size_min)
        payload_size_new   <= max(payload_size_before // 2, payload_size_min)
        concurrency_new    <= max(concurrency_before  // 2, concurrency_min)
    """
    batch_size_min = 1
    payload_size_min = 1
    concurrency_min = 1

    cfg = _make_config(
        batch_size_initial=batch_size_initial,
        batch_size_min=batch_size_min,
        batch_size_max=batch_size_initial,
        payload_size_initial=payload_size_initial,
        payload_size_min=payload_size_min,
        payload_size_max=payload_size_initial,
        concurrency_initial=concurrency_initial,
        concurrency_min=concurrency_min,
        concurrency_max=concurrency_initial,
        max_retries=3,
        backoff_base_ms=1,
        backoff_jitter_ms=0,
        probe_up_every_n_success=10_000,
    )

    async def _run() -> _StateRecordingScheduler:
        attempts = {"count": 0}

        async def fetch(batch: list[str]) -> list[str]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RateLimitError("slow")
            return list(batch)

        sleeper = _RecordingSleeper()
        sched = _StateRecordingScheduler(
            cfg, size_of=_text_size, sleeper=sleeper
        )
        await sched.run(["a", "b"], fetch)
        return sched

    sched = _run_async(_run())
    assert sched.before_snapshots, "expected at least one down-tune event"
    before = sched.before_snapshots[0]
    after = sched.after_snapshots[0]

    bs_before, pl_before, co_before = before
    bs_after, pl_after, co_after = after

    assert bs_after <= max(bs_before // 2, batch_size_min), (
        f"batch_size {bs_before} -> {bs_after} violates halving bound"
    )
    assert pl_after <= max(pl_before // 2, payload_size_min), (
        f"payload_size {pl_before} -> {pl_after} violates halving bound"
    )
    assert co_after <= max(co_before // 2, concurrency_min), (
        f"concurrency {co_before} -> {co_after} violates halving bound"
    )



# ---------------------------------------------------------------------------
# Property 37 — Retry-After wait compliance (task 5.8)
# ---------------------------------------------------------------------------


@given(
    retry_after_seconds=st.floats(
        min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_37_retry_after_header_is_honoured(
    retry_after_seconds: float,
) -> None:
    """**Validates: Requirement 12.7**

    When the upstream advertises ``Retry-After: k`` the scheduler waits
    at least ``k`` seconds before retrying. (A ``k == 0`` header may
    legitimately skip the sleep entirely.) Without ``Retry-After`` the
    wait falls in the exponential backoff + jitter window — covered by
    the companion property test below.
    """
    cfg = _make_config(
        max_retries=2,
        backoff_base_ms=50,
        backoff_jitter_ms=0,
    )
    sleeper = _RecordingSleeper()

    async def _run() -> list[float]:
        attempts = {"n": 0}

        async def fetch(batch: list[str]) -> list[str]:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RateLimitError("slow", retry_after=retry_after_seconds)
            return list(batch)

        sched: AdaptiveScheduler[str, str] = AdaptiveScheduler(
            cfg, size_of=_text_size, sleeper=sleeper
        )
        await sched.run(["x"], fetch)
        return sleeper.waits

    waits = _run_async(_run())
    # The scheduler may skip the sleep entirely when Retry-After is 0.
    total_wait = sum(waits)
    assert total_wait >= retry_after_seconds, (
        f"total wait {total_wait}s < advertised Retry-After "
        f"{retry_after_seconds}s"
    )


@given(
    backoff_base_ms=st.integers(min_value=1, max_value=200),
    backoff_jitter_ms=st.integers(min_value=0, max_value=100),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_37b_backoff_without_retry_after_falls_in_window(
    backoff_base_ms: int, backoff_jitter_ms: int, seed: int
) -> None:
    """**Validates: Requirement 12.7**

    With no ``Retry-After`` the wait sits in the exponential backoff +
    jitter window.
    """
    cfg = _make_config(
        max_retries=1,
        backoff_base_ms=backoff_base_ms,
        backoff_jitter_ms=backoff_jitter_ms,
    )
    sleeper = _RecordingSleeper()
    rng = random.Random(seed)

    async def _run() -> list[float]:
        attempts = {"n": 0}

        async def fetch(batch: list[str]) -> list[str]:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RateLimitError("slow")  # no retry-after
            return list(batch)

        sched: AdaptiveScheduler[str, str] = AdaptiveScheduler(
            cfg, size_of=_text_size, sleeper=sleeper, rng=rng
        )
        await sched.run(["x"], fetch)
        return sleeper.waits

    waits = _run_async(_run())
    assert len(waits) == 1
    actual = waits[0]
    lower = backoff_base_ms / 1000.0  # attempt=1 → 2**0 * base
    upper = (backoff_base_ms + backoff_jitter_ms) / 1000.0
    assert lower <= actual <= upper, (
        f"wait {actual}s not in [{lower}, {upper}] for base={backoff_base_ms}ms, "
        f"jitter={backoff_jitter_ms}ms"
    )



# ---------------------------------------------------------------------------
# Property 38 — Max retries bound (task 5.9)
# ---------------------------------------------------------------------------


@given(
    max_retries=st.integers(min_value=0, max_value=5),
    n=st.integers(min_value=1, max_value=6),
    use_transient=st.booleans(),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_38_max_retries_bounds_total_calls(
    max_retries: int, n: int, use_transient: bool
) -> None:
    """**Validates: Requirements 12.8, 12.9**

    A provider that never recovers (persistent 429 or persistent
    transient failure) consumes at most ``max_retries + 1`` calls per
    batch. The scheduler raises :class:`SchedulerBatchFailure` whose
    ``entry_indices`` cover exactly the failing batch's items and whose
    ``provider_type`` matches the construction-time label.
    """
    # One batch of n items so the budget maths are simple.
    cfg = _make_config(
        batch_size_initial=n,
        batch_size_min=1,
        batch_size_max=n,
        max_retries=max_retries,
        backoff_base_ms=1,
        backoff_jitter_ms=0,
    )

    async def _run() -> tuple[int, SchedulerBatchFailure]:
        calls = 0

        async def fetch(batch: list[str]) -> list[str]:
            nonlocal calls
            calls += 1
            if use_transient:
                raise TransientError("5xx")
            raise RateLimitError("always")

        sleeper = _RecordingSleeper()
        sched: AdaptiveScheduler[str, str] = AdaptiveScheduler(
            cfg,
            size_of=_text_size,
            sleeper=sleeper,
            provider_type="stub",
        )
        with pytest.raises(SchedulerBatchFailure) as excinfo:
            await sched.run([f"i-{i}" for i in range(n)], fetch)
        return calls, excinfo.value

    calls, err = _run_async(_run())
    assert calls <= max_retries + 1, (
        f"scheduler issued {calls} calls; allowed at most "
        f"max_retries+1 = {max_retries + 1}"
    )
    assert err.provider_type == "stub"
    assert err.entry_indices == list(range(n))
    # Last error propagates the right type.
    if use_transient:
        assert isinstance(err.last_error, TransientError)
    else:
        assert isinstance(err.last_error, RateLimitError)



# ---------------------------------------------------------------------------
# Property 39 — Up-tune never exceeds per-dimension maximum (task 5.10)
# ---------------------------------------------------------------------------


class _PeakRecordingScheduler(AdaptiveScheduler[str, str]):
    """Subclass that records peak ``(batch_size, payload_size, concurrency)``.

    Captures the highest value each dimension ever reaches, including the
    final state after all up-tunes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.peak_batch: int = 0
        self.peak_payload: int = 0
        self.peak_concurrency: int = 0

    async def _record_success(self, state: Any) -> None:  # type: ignore[override]
        await super()._record_success(state)
        if state.batch_size > self.peak_batch:
            self.peak_batch = state.batch_size
        if state.payload_size > self.peak_payload:
            self.peak_payload = state.payload_size
        if state.concurrency > self.peak_concurrency:
            self.peak_concurrency = state.concurrency


@given(
    n=st.integers(min_value=5, max_value=40),
    batch_size_max=st.integers(min_value=2, max_value=6),
    payload_size_max=st.integers(min_value=8, max_value=80),
    concurrency_max=st.integers(min_value=2, max_value=6),
    probe_step=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_39_up_tune_stays_within_maxima(
    n: int,
    batch_size_max: int,
    payload_size_max: int,
    concurrency_max: int,
    probe_step: int,
) -> None:
    """**Validates: Requirement 12.5**

    After sustained successes — more than ``probe_up_every_n_success``
    consecutive wins — every dimension stays at or below its maximum.
    """
    cfg = _make_config(
        batch_size_initial=1,
        batch_size_min=1,
        batch_size_max=batch_size_max,
        payload_size_initial=1,
        payload_size_min=1,
        payload_size_max=payload_size_max,
        concurrency_initial=1,
        concurrency_min=1,
        concurrency_max=concurrency_max,
        probe_up_every_n_success=probe_step,
    )

    async def _run() -> _PeakRecordingScheduler:
        async def fetch(batch: list[str]) -> list[str]:
            return list(batch)

        sched = _PeakRecordingScheduler(cfg, size_of=_text_size)
        # Use many items so the scheduler sees enough successes to
        # fully up-tune beyond every maximum if it were ever going to.
        await sched.run([f"x-{i}" for i in range(n)], fetch)
        return sched

    sched = _run_async(_run())
    assert sched.peak_batch <= batch_size_max, (
        f"peak batch_size {sched.peak_batch} exceeds max {batch_size_max}"
    )
    assert sched.peak_payload <= payload_size_max, (
        f"peak payload_size {sched.peak_payload} exceeds max "
        f"{payload_size_max}"
    )
    assert sched.peak_concurrency <= concurrency_max, (
        f"peak concurrency {sched.peak_concurrency} exceeds max "
        f"{concurrency_max}"
    )



# ---------------------------------------------------------------------------
# Property 40 — Non-batch providers are forced to batch size 1 (task 5.11)
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=20),
    batch_size_initial=st.integers(min_value=1, max_value=10),
    concurrency_initial=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_40_non_batch_provider_uses_batch_size_one(
    n: int, batch_size_initial: int, concurrency_initial: int
) -> None:
    """**Validates: Requirement 12.14**

    When the provider reports ``supports_batch=False`` the scheduler
    clamps batch size to 1 for every request, regardless of the
    ``batch_size_initial`` the caller provided. Payload size and
    concurrency remain tunable.
    """
    cfg = _make_config(
        supports_batch=False,
        batch_size_initial=batch_size_initial,
        batch_size_min=1,
        batch_size_max=batch_size_initial,
        concurrency_initial=concurrency_initial,
        concurrency_min=1,
        concurrency_max=concurrency_initial,
        payload_size_initial=50,
        payload_size_min=1,
        payload_size_max=50,
        probe_up_every_n_success=10_000,
    )

    observed_sizes: list[int] = []

    async def _run() -> None:
        async def fetch(batch: list[str]) -> list[str]:
            observed_sizes.append(len(batch))
            return list(batch)

        sched: AdaptiveScheduler[str, str] = AdaptiveScheduler(
            cfg, size_of=_text_size
        )
        await sched.run([f"x-{i}" for i in range(n)], fetch)

    _run_async(_run())
    assert observed_sizes, "expected the scheduler to invoke fetch"
    assert all(size == 1 for size in observed_sizes), (
        f"non-batch provider saw batches with sizes {observed_sizes}; "
        "every call must use batch_size=1"
    )
    # We should see exactly n calls (one per item).
    assert len(observed_sizes) == n



# ---------------------------------------------------------------------------
# Property 41 — Progress monotonicity under backoff (task 5.12)
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=15),
    seed=st.integers(min_value=0, max_value=1_000),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_41_progress_monotonic_under_mixed_failures(
    n: int, seed: int
) -> None:
    """**Validates: Requirement 12.16**

    Under intermittent ``RateLimitError`` / ``PayloadTooLargeError``
    followed by success, the scheduler still returns outputs aligned
    1:1 with the input list and in the original order. The "progress
    never regresses" guarantee at the scheduler level manifests as:

    - ``len(outputs) == len(items)``;
    - ``outputs[i]`` corresponds to ``items[i]`` for every i (via the
      stub's deterministic transform);
    - no item is dropped or duplicated even after re-slices and retries.
    """
    cfg = _make_config(
        batch_size_initial=4,
        batch_size_min=1,
        batch_size_max=4,
        payload_size_initial=100,
        payload_size_min=1,
        payload_size_max=100,
        max_retries=5,
        backoff_base_ms=1,
        backoff_jitter_ms=0,
    )
    rng = random.Random(seed)
    items = [f"i{i}" for i in range(n)]

    # For each unique batch content, decide up front how it should
    # misbehave on its FIRST encounter (and only that first one). After
    # the first rejection the batch always succeeds — this guarantees
    # the scheduler's retry budget is never exhausted while still
    # exercising both feedback paths.
    fail_modes = ["ok", "rl", "pl"]

    async def _run() -> list[str]:
        first_outcome: dict[str, str] = {}

        async def fetch(batch: list[str]) -> list[str]:
            key = "|".join(batch)
            if key not in first_outcome:
                first_outcome[key] = rng.choice(fail_modes)
                outcome = first_outcome[key]
                if outcome == "rl":
                    # Flip to ok so the next encounter succeeds.
                    first_outcome[key] = "done"
                    raise RateLimitError("slow")
                if outcome == "pl":
                    first_outcome[key] = "done"
                    raise PayloadTooLargeError("big")
                # outcome == "ok" → fall through to success path
                first_outcome[key] = "done"
            return [t.upper() for t in batch]

        sleeper = _RecordingSleeper()
        sched: AdaptiveScheduler[str, str] = AdaptiveScheduler(
            cfg, size_of=_text_size, sleeper=sleeper
        )
        return await sched.run(items, fetch)

    outputs = _run_async(_run())
    assert len(outputs) == len(items), (
        f"len(outputs)={len(outputs)} != len(items)={len(items)}"
    )
    assert outputs == [t.upper() for t in items], (
        "outputs do not preserve 1:1 correspondence with inputs"
    )



# ---------------------------------------------------------------------------
# Property 42 — Payload-too-large triggers down-tune + reslice (task 5.13)
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=2, max_value=12),
    payload_multiplier=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_42_payload_too_large_shrinks_and_reslices_without_retry_budget(
    n: int, payload_multiplier: int
) -> None:
    """**Validates: Requirement 12.12**

    When a batch raises :class:`PayloadTooLargeError`:

    1. ``payload_size`` halves, floored at ``payload_size_min``.
    2. The batch is re-sliced so every resulting sub-batch satisfies
       ``sum(size_of(x)) <= payload_size_new``.
    3. The re-slice does NOT consume ``max_retries`` — we prove this by
       running with ``max_retries=0`` and still completing successfully.
    """
    # Items of fixed size so the arithmetic is tractable. Pick a payload
    # budget large enough that the initial ``make_batches`` produces a
    # single batch; that way ``batches_seen[1:]`` consists exclusively
    # of post-reslice sub-batches.
    items = ["ab"] * n  # size 2 each
    total_volume = 2 * n
    payload_size_initial = total_volume * payload_multiplier
    payload_size_min = 1

    before_payload: list[int] = []
    after_payload: list[int] = []

    class _PayloadObserver(AdaptiveScheduler[str, str]):
        async def _handle_payload_too_large(  # type: ignore[override]
            self, batch: Any, exc: Any, state: Any, pending_out: Any
        ) -> None:
            before_payload.append(state.payload_size)
            await super()._handle_payload_too_large(batch, exc, state, pending_out)
            after_payload.append(state.payload_size)

    cfg = _make_config(
        batch_size_initial=n,
        batch_size_min=1,
        batch_size_max=n,
        payload_size_initial=payload_size_initial,
        payload_size_min=payload_size_min,
        payload_size_max=payload_size_initial,
        max_retries=0,  # budget must not be consumed by the reslice
        backoff_base_ms=1,
        backoff_jitter_ms=0,
        probe_up_every_n_success=10_000,
    )

    batches_seen: list[list[str]] = []

    async def _run() -> list[str]:
        first_done = {"done": False}

        async def fetch(batch: list[str]) -> list[str]:
            batches_seen.append(list(batch))
            if not first_done["done"]:
                first_done["done"] = True
                raise PayloadTooLargeError("big")
            return [t.upper() for t in batch]

        sched = _PayloadObserver(cfg, size_of=_text_size)
        return await sched.run(items, fetch)

    outputs = _run_async(_run())
    assert outputs == ["AB"] * n, (
        "payload-too-large path must still produce correct outputs"
    )
    # Down-tune: payload halved (or floored).
    assert before_payload, "expected a payload down-tune event"
    assert after_payload[0] <= max(
        before_payload[0] // 2, payload_size_min
    )

    # Every post-reslice batch fits under the new payload_size. Since
    # the initial make_batches produced a single batch (all items),
    # ``batches_seen[0]`` is that offending batch and ``batches_seen[1:]``
    # is entirely post-reslice.
    assert batches_seen[0] == items, (
        f"expected the first call to receive the whole input; got "
        f"{batches_seen[0]}"
    )
    new_payload = after_payload[0]
    for batch in batches_seen[1:]:
        volume = sum(_text_size(t) for t in batch)
        assert volume <= new_payload, (
            f"reslice batch {batch} has volume {volume} > "
            f"new payload_size {new_payload}"
        )



# ---------------------------------------------------------------------------
# Property 43 — make_batches dual-dimension invariants (task 5.14)
# ---------------------------------------------------------------------------


@given(
    items=st.lists(
        st.text(alphabet=st.characters(whitelist_categories=("L", "N", "P", "Zs")), min_size=0, max_size=6),
        min_size=0,
        max_size=30,
    ),
    current_batch_size=st.integers(min_value=1, max_value=8),
    current_payload_size=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=200, deadline=None)
def test_property_43_make_batches_dual_dimension_invariants(
    items: list[str],
    current_batch_size: int,
    current_payload_size: int,
) -> None:
    """**Validates: Requirement 12.13**

    Pure-function invariants for :func:`make_batches`:

    1. Every batch honours ``len(batch) <= current_batch_size``.
    2. Every batch honours
       ``sum(size_of(x) for x in batch) <= current_payload_size``,
       EXCEPT a singleton whose sole item exceeds the payload cap.
    3. Any oversized item (``size_of(x) > current_payload_size``) is
       emitted as its own batch.
    4. Concatenating the batches in order reproduces ``items`` exactly.
    """
    batches = make_batches(
        items, current_batch_size, current_payload_size, _text_size
    )

    # Order and count preservation.
    flattened: list[str] = []
    for batch in batches:
        flattened.extend(batch)
    assert flattened == items, (
        "make_batches must preserve item order and count; got "
        f"{flattened!r} vs {items!r}"
    )

    for batch in batches:
        assert len(batch) <= current_batch_size, (
            f"batch {batch} exceeds current_batch_size={current_batch_size}"
        )
        volume = sum(_text_size(x) for x in batch)
        if len(batch) == 1 and _text_size(batch[0]) > current_payload_size:
            # Oversized singleton: allowed to exceed the payload cap.
            continue
        assert volume <= current_payload_size, (
            f"batch {batch} has volume {volume} > "
            f"current_payload_size={current_payload_size}"
        )

    # Oversized singletons must be isolated.
    for item in items:
        if _text_size(item) > current_payload_size:
            assert [item] in batches, (
                f"oversized item {item!r} (size "
                f"{_text_size(item)} > {current_payload_size}) must "
                "stand alone as a singleton batch"
            )
