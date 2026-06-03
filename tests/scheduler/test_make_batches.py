"""Unit tests for :func:`translation_dubbing_skill.scheduler.make_batches`.

Covers the pure batching function that enforces both the entry-count
cap and the text-volume cap simultaneously (R12.13). Exhaustive
property-based coverage (P43) lives in task 5.14; these tests pin down
the specific behaviours the scheduler relies on.
"""

from __future__ import annotations

import pytest

from translation_dubbing_skill.scheduler import make_batches


def _size_of_len(text: str) -> int:
    return len(text)


def test_empty_input_returns_empty_batches() -> None:
    assert make_batches([], 10, 100, _size_of_len) == []


def test_single_item_under_both_caps_returns_single_batch() -> None:
    batches = make_batches(["hi"], 10, 100, _size_of_len)
    assert batches == [["hi"]]


def test_batch_size_cap_honoured() -> None:
    items = ["a"] * 5
    batches = make_batches(items, current_batch_size=2, current_payload_size=100, size_of=_size_of_len)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert sum((b for b in batches), []) == items


def test_payload_size_cap_honoured() -> None:
    # Each item is size 3; payload cap is 5 → two items per batch would
    # be 6 (too large), so one item per batch.
    items = ["abc"] * 4
    batches = make_batches(items, current_batch_size=10, current_payload_size=5, size_of=_size_of_len)
    assert [len(b) for b in batches] == [1, 1, 1, 1]


def test_both_caps_interact() -> None:
    # batch_size=3, payload_size=10; items of size 4 each → only 2 per batch fit
    # (8 <= 10, but 12 > 10).
    items = ["abcd"] * 5
    batches = make_batches(items, current_batch_size=3, current_payload_size=10, size_of=_size_of_len)
    assert [len(b) for b in batches] == [2, 2, 1]


def test_oversized_singleton_isolated() -> None:
    items = ["short", "way_too_big_for_cap", "short"]
    batches = make_batches(items, current_batch_size=10, current_payload_size=6, size_of=_size_of_len)
    # "short" (5) + "way_too_big_for_cap" (size 19 > 6): the big one
    # must stand alone; "short" items group together where possible.
    assert batches[0] == ["short"]
    assert batches[1] == ["way_too_big_for_cap"]
    assert batches[2] == ["short"]


def test_oversized_singleton_preserves_total_order() -> None:
    items = [
        "a",      # size 1
        "bbbbb",  # size 5, oversized (cap 3)
        "c",
        "d",
        "eeeeee", # size 6, oversized
        "f",
    ]
    batches = make_batches(items, current_batch_size=10, current_payload_size=3, size_of=_size_of_len)
    assert sum(batches, []) == items


def test_order_always_preserved() -> None:
    items = list("abcdefghij")
    batches = make_batches(items, current_batch_size=3, current_payload_size=100, size_of=_size_of_len)
    assert sum(batches, []) == items


def test_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        make_batches(["a"], 0, 10, _size_of_len)


def test_rejects_non_positive_payload_size() -> None:
    with pytest.raises(ValueError):
        make_batches(["a"], 10, 0, _size_of_len)


def test_rejects_negative_size_callback() -> None:
    with pytest.raises(ValueError):
        make_batches(["a"], 10, 10, lambda _x: -1)


def test_size_zero_items_fit_until_batch_size_cap() -> None:
    # Items with size 0 never bump payload; only batch_size gates them.
    items = [""] * 7
    batches = make_batches(items, current_batch_size=3, current_payload_size=1, size_of=_size_of_len)
    assert [len(b) for b in batches] == [3, 3, 1]
