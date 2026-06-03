"""Unit tests for :mod:`translation_dubbing_skill.scheduler.signals`.

Covers the three signal exceptions, the detection helpers, and their
interaction with ``httpx.HTTPStatusError`` forwarded verbatim by a
provider.
"""

from __future__ import annotations

import httpx
import pytest

from translation_dubbing_skill.scheduler.signals import (
    PayloadTooLargeError,
    RateLimitError,
    SchedulerSignalError,
    TransientError,
    is_payload_too_large,
    is_rate_limited,
    retry_after_of,
)


# ---------------------------------------------------------------------------
# Signal exception shape
# ---------------------------------------------------------------------------


def test_signal_carries_reason_and_defaults_context_to_empty_dict() -> None:
    exc = RateLimitError("slow down")
    assert exc.reason == "slow down"
    assert exc.retry_after is None
    assert exc.context == {}
    # args[0] must be the reason so standard Python tooling (str(),
    # traceback) shows the message.
    assert str(exc) == "slow down"


def test_signal_context_is_copied_not_aliased() -> None:
    source = {"upstream_code": 429}
    exc = RateLimitError("x", context=source)
    source["upstream_code"] = 999
    assert exc.context == {"upstream_code": 429}


def test_signal_rejects_negative_retry_after() -> None:
    with pytest.raises(ValueError):
        RateLimitError("x", retry_after=-1.0)


def test_all_three_signals_share_base_class() -> None:
    assert issubclass(RateLimitError, SchedulerSignalError)
    assert issubclass(PayloadTooLargeError, SchedulerSignalError)
    assert issubclass(TransientError, SchedulerSignalError)


# ---------------------------------------------------------------------------
# is_rate_limited
# ---------------------------------------------------------------------------


def test_is_rate_limited_true_for_direct_rate_limit_error() -> None:
    assert is_rate_limited(RateLimitError("nope")) is True


def test_is_rate_limited_false_for_other_signals() -> None:
    assert is_rate_limited(PayloadTooLargeError("big")) is False
    assert is_rate_limited(TransientError("boom")) is False


def _make_http_error(status_code: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/x")
    response = httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=request,
    )
    return httpx.HTTPStatusError("upstream", request=request, response=response)


def test_is_rate_limited_recognises_http_429() -> None:
    assert is_rate_limited(_make_http_error(429)) is True


def test_is_rate_limited_walks_cause_chain() -> None:
    try:
        try:
            raise _make_http_error(429)
        except httpx.HTTPStatusError as http_err:
            raise TransientError("wrapped") from http_err
    except TransientError as wrapper:
        # Wrapper looks transient but the underlying cause is 429.
        assert is_rate_limited(wrapper) is True


def test_is_rate_limited_false_for_unrelated_error() -> None:
    assert is_rate_limited(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# is_payload_too_large
# ---------------------------------------------------------------------------


def test_is_payload_too_large_true_for_direct_payload_error() -> None:
    assert is_payload_too_large(PayloadTooLargeError("too big")) is True


def test_is_payload_too_large_recognises_http_413() -> None:
    assert is_payload_too_large(_make_http_error(413)) is True


def test_is_payload_too_large_matches_context_window_hint() -> None:
    err = TransientError("context_length_exceeded: 8193 > 8192")
    assert is_payload_too_large(err) is True


def test_is_payload_too_large_does_not_match_arbitrary_text() -> None:
    # Hint match only applies to scheduler signal exceptions to avoid
    # false positives from unrelated error messages.
    assert is_payload_too_large(ValueError("input is too long")) is False


def test_is_payload_too_large_and_rate_limited_are_disjoint_for_simple_cases() -> None:
    rl = RateLimitError("slow down")
    pl = PayloadTooLargeError("huge")
    assert is_rate_limited(rl) and not is_payload_too_large(rl)
    assert is_payload_too_large(pl) and not is_rate_limited(pl)


# ---------------------------------------------------------------------------
# retry_after_of
# ---------------------------------------------------------------------------


def test_retry_after_of_reads_signal_attribute() -> None:
    exc = RateLimitError("slow", retry_after=2.5)
    assert retry_after_of(exc) == 2.5


def test_retry_after_of_returns_none_when_absent() -> None:
    assert retry_after_of(RateLimitError("x")) is None
    assert retry_after_of(ValueError("unrelated")) is None


def test_retry_after_of_reads_http_header() -> None:
    exc = _make_http_error(429, headers={"Retry-After": "3"})
    assert retry_after_of(exc) == 3.0


def test_retry_after_of_ignores_malformed_header() -> None:
    exc = _make_http_error(429, headers={"Retry-After": "Tue, 01 Jan 2030 00:00:00 GMT"})
    assert retry_after_of(exc) is None


def test_retry_after_of_prefers_signal_over_header() -> None:
    try:
        try:
            raise _make_http_error(429, headers={"Retry-After": "10"})
        except httpx.HTTPStatusError as http_err:
            raise RateLimitError("wrapped", retry_after=0.5) from http_err
    except RateLimitError as wrapper:
        assert retry_after_of(wrapper) == 0.5
