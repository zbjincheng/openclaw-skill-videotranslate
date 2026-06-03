"""Unit tests for :mod:`translation_dubbing_skill.scheduler.sizing`.

The scheduler-level sizing module re-exports the authoritative helpers
from :mod:`translation_dubbing_skill.providers.sizing` and adds
:func:`size_of_for_unit` for coordinator convenience. These tests only
cover the scheduler-specific surface; exhaustive coverage of the
underlying helpers lives in ``tests/providers/test_sizing.py``.
"""

from __future__ import annotations

import pytest

from translation_dubbing_skill.scheduler.sizing import (
    size_of_chars,
    size_of_for_unit,
    size_of_tokens,
)


def test_size_of_for_unit_chars_returns_len() -> None:
    sizer = size_of_for_unit("chars")
    assert sizer("hello") == size_of_chars("hello") == 5


def test_size_of_for_unit_tokens_returns_heuristic() -> None:
    sizer = size_of_for_unit("tokens")
    assert sizer("hello") == size_of_tokens("hello")


def test_size_of_for_unit_rejects_unknown_unit() -> None:
    sizer = size_of_for_unit("bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload_unit"):
        sizer("x")
