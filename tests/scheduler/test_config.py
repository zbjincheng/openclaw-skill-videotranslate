"""Unit tests for :mod:`translation_dubbing_skill.scheduler.config`.

Validates :class:`ProviderRateLimitConfig.__post_init__` enforces the
design's invariants:

- ``min <= initial <= max`` for each of the three triples.
- ``payload_unit`` is one of ``{"chars", "tokens"}``.
- ``max_retries >= 0``, ``backoff_base_ms > 0``,
  ``backoff_jitter_ms >= 0``, ``probe_up_every_n_success > 0``.
- All positive integer fields are at least 1.
"""

from __future__ import annotations

import pytest

from translation_dubbing_skill.scheduler.config import ProviderRateLimitConfig


def _valid_kwargs(**overrides: object) -> dict[str, object]:
    """Return a valid config kwargs dict with optional per-test overrides."""
    base: dict[str, object] = dict(
        batch_size_initial=10,
        batch_size_min=1,
        batch_size_max=50,
        payload_size_initial=1000,
        payload_size_min=100,
        payload_size_max=10_000,
        payload_unit="chars",
        concurrency_initial=2,
        concurrency_min=1,
        concurrency_max=8,
        max_retries=5,
        backoff_base_ms=500,
        backoff_jitter_ms=300,
        probe_up_every_n_success=10,
        supports_batch=True,
    )
    base.update(overrides)
    return base


def test_valid_config_constructs_successfully() -> None:
    cfg = ProviderRateLimitConfig(**_valid_kwargs())
    assert cfg.payload_unit == "chars"
    assert cfg.supports_batch is True


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        (dict(batch_size_initial=0), "batch_size_initial must be >= 1"),
        (dict(batch_size_min=0), "batch_size_min must be >= 1"),
        (dict(batch_size_max=0), "batch_size_max must be >= 1"),
        (dict(batch_size_min=5, batch_size_initial=3, batch_size_max=10), "batch_size requires min <= initial <= max"),
        (dict(batch_size_initial=100, batch_size_max=50), "batch_size requires min <= initial <= max"),
        (dict(payload_size_initial=0), "payload_size_initial must be >= 1"),
        (dict(payload_size_min=200, payload_size_initial=100), "payload_size requires min <= initial <= max"),
        (dict(concurrency_min=0), "concurrency_min must be >= 1"),
        (dict(concurrency_max=1, concurrency_initial=2), "concurrency requires min <= initial <= max"),
    ],
)
def test_config_rejects_broken_triples(overrides: dict, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        ProviderRateLimitConfig(**_valid_kwargs(**overrides))


def test_config_rejects_unknown_payload_unit() -> None:
    with pytest.raises(ValueError, match="payload_unit"):
        ProviderRateLimitConfig(**_valid_kwargs(payload_unit="bytes"))


def test_config_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        ProviderRateLimitConfig(**_valid_kwargs(max_retries=-1))


def test_config_accepts_zero_max_retries() -> None:
    # max_retries == 0 means "first attempt only, no retries".
    cfg = ProviderRateLimitConfig(**_valid_kwargs(max_retries=0))
    assert cfg.max_retries == 0


def test_config_rejects_zero_backoff_base() -> None:
    with pytest.raises(ValueError, match="backoff_base_ms"):
        ProviderRateLimitConfig(**_valid_kwargs(backoff_base_ms=0))


def test_config_accepts_zero_jitter() -> None:
    cfg = ProviderRateLimitConfig(**_valid_kwargs(backoff_jitter_ms=0))
    assert cfg.backoff_jitter_ms == 0


def test_config_rejects_negative_jitter() -> None:
    with pytest.raises(ValueError, match="backoff_jitter_ms"):
        ProviderRateLimitConfig(**_valid_kwargs(backoff_jitter_ms=-1))


def test_config_rejects_zero_probe_step() -> None:
    with pytest.raises(ValueError, match="probe_up_every_n_success"):
        ProviderRateLimitConfig(**_valid_kwargs(probe_up_every_n_success=0))


def test_config_accepts_tokens_unit_and_non_batch_provider() -> None:
    cfg = ProviderRateLimitConfig(
        **_valid_kwargs(payload_unit="tokens", supports_batch=False)
    )
    assert cfg.payload_unit == "tokens"
    assert cfg.supports_batch is False
