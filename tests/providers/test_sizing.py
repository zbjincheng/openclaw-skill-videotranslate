"""Unit tests for the shared provider text-size measurement helpers.

Covers :mod:`translation_dubbing_skill.providers.sizing` — the default
``size_of`` implementations dispatched from the provider protocols'
``payload_unit``.
"""

from __future__ import annotations

import pytest

from translation_dubbing_skill.providers.sizing import (
    default_size_of,
    size_of_chars,
    size_of_tokens,
)


# ---------------------------------------------------------------------------
# size_of_chars
# ---------------------------------------------------------------------------


def test_size_of_chars_empty_string_is_zero() -> None:
    assert size_of_chars("") == 0


def test_size_of_chars_ascii_counts_code_points() -> None:
    assert size_of_chars("hello") == 5


def test_size_of_chars_unicode_counts_code_points_not_bytes() -> None:
    # "你好" is 2 code points but 6 UTF-8 bytes; we count code points.
    assert size_of_chars("你好") == 2


def test_size_of_chars_handles_emoji() -> None:
    # 🌏 is a single code point.
    assert size_of_chars("🌏") == 1


# ---------------------------------------------------------------------------
# size_of_tokens
# ---------------------------------------------------------------------------


def test_size_of_tokens_empty_is_zero() -> None:
    assert size_of_tokens("") == 0


def test_size_of_tokens_uses_ceiling_heuristic() -> None:
    # ceil(len / 2)
    assert size_of_tokens("a") == 1
    assert size_of_tokens("ab") == 1
    assert size_of_tokens("abc") == 2
    assert size_of_tokens("abcd") == 2
    assert size_of_tokens("abcde") == 3


def test_size_of_tokens_unicode() -> None:
    # 4 code points -> 2 tokens under the heuristic.
    assert size_of_tokens("你好世界") == 2


# ---------------------------------------------------------------------------
# default_size_of dispatch
# ---------------------------------------------------------------------------


def test_default_size_of_chars_matches_helper() -> None:
    text = "hello 世界"
    assert default_size_of(text, "chars") == size_of_chars(text)


def test_default_size_of_tokens_matches_helper() -> None:
    text = "hello 世界"
    assert default_size_of(text, "tokens") == size_of_tokens(text)


def test_default_size_of_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError, match="payload_unit"):
        default_size_of("x", "bytes")  # type: ignore[arg-type]
