"""Unit tests for ``SubtitleEntry`` and its equivalence helpers.

Covers requirements R2.3 (subtitle entry shape) and R4.3 (text equivalence
used by the SRT/VTT round-trip properties).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from translation_dubbing_skill.models import (
    SubtitleEntry,
    entries_equivalent,
    normalize_text,
)
from translation_dubbing_skill.models import subtitle_entry as se_module


# ---------------------------------------------------------------------------
# SubtitleEntry shape / behaviour
# ---------------------------------------------------------------------------


def test_subtitle_entry_is_frozen_dataclass() -> None:
    """SubtitleEntry is a frozen dataclass with the expected fields."""
    assert is_dataclass(SubtitleEntry)
    field_names = [f.name for f in fields(SubtitleEntry)]
    assert field_names == ["index", "start_ms", "end_ms", "text"]


def test_subtitle_entry_field_types() -> None:
    """Field annotations follow the design spec."""
    annotations = {f.name: f.type for f in fields(SubtitleEntry)}
    assert annotations == {
        "index": "int",
        "start_ms": "int",
        "end_ms": "int",
        "text": "str",
    }


def test_subtitle_entry_instance_stores_values() -> None:
    """Constructing an entry preserves provided values."""
    entry = SubtitleEntry(index=1, start_ms=0, end_ms=1_500, text="Hello")
    assert entry.index == 1
    assert entry.start_ms == 0
    assert entry.end_ms == 1_500
    assert entry.text == "Hello"


def test_subtitle_entry_is_immutable() -> None:
    """Frozen dataclass rejects attribute mutation."""
    entry = SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="Hi")
    with pytest.raises(FrozenInstanceError):
        entry.text = "Changed"  # type: ignore[misc]


def test_subtitle_entry_equality_and_hash() -> None:
    """Value-based equality + hashability from ``frozen=True``."""
    a = SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="A")
    b = SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="A")
    c = SubtitleEntry(index=2, start_ms=0, end_ms=1_000, text="A")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    # Usable inside sets/dicts.
    assert {a, b, c} == {a, c}


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_converts_crlf_to_lf() -> None:
    """CRLF line endings are normalized to LF."""
    assert normalize_text("a\r\nb\r\nc") == "a\nb\nc"


def test_normalize_text_converts_lone_cr_to_lf() -> None:
    """Standalone CR characters also become LF."""
    assert normalize_text("a\rb\rc") == "a\nb\nc"


def test_normalize_text_handles_mixed_line_endings() -> None:
    """CRLF, CR, and LF may be freely mixed."""
    assert normalize_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_normalize_text_strips_trailing_whitespace_per_line() -> None:
    """Trailing spaces and tabs are stripped from each line."""
    assert normalize_text("hello   \nworld\t\n") == "hello\nworld\n"


def test_normalize_text_preserves_leading_whitespace() -> None:
    """Only *trailing* whitespace is touched; leading whitespace survives."""
    assert normalize_text("  indented\n\ttabbed  ") == "  indented\n\ttabbed"


def test_normalize_text_preserves_blank_interior_lines() -> None:
    """Blank interior lines are kept so line counts remain stable."""
    assert normalize_text("a\n\nb") == "a\n\nb"


def test_normalize_text_empty_string() -> None:
    """Empty input maps to empty output without error."""
    assert normalize_text("") == ""


def test_normalize_text_whitespace_only() -> None:
    """Lines containing only whitespace collapse to empty lines."""
    assert normalize_text("   \n\t\t\n") == "\n\n"


def test_normalize_text_unicode_preserved() -> None:
    """Non-ASCII content (CJK, emoji) passes through untouched."""
    assert normalize_text("你好   \r\n🌏\t") == "你好\n🌏"


def test_normalize_text_is_idempotent() -> None:
    """Applying ``normalize_text`` twice yields the same result as once."""
    raw = "line one   \r\nline two\t\r\n\r\n末尾  "
    once = normalize_text(raw)
    twice = normalize_text(once)
    assert once == twice


# ---------------------------------------------------------------------------
# entries_equivalent
# ---------------------------------------------------------------------------


def _entry(
    index: int = 1,
    start_ms: int = 0,
    end_ms: int = 1_000,
    text: str = "hi",
) -> SubtitleEntry:
    return SubtitleEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text)


def test_entries_equivalent_true_for_identical_lists() -> None:
    """Identical entry lists are equivalent."""
    a = [_entry(1, 0, 1_000, "hi"), _entry(2, 1_000, 2_000, "world")]
    b = [_entry(1, 0, 1_000, "hi"), _entry(2, 1_000, 2_000, "world")]
    assert entries_equivalent(a, b) is True


def test_entries_equivalent_both_empty() -> None:
    """Two empty sequences are trivially equivalent."""
    assert entries_equivalent([], []) is True


def test_entries_equivalent_false_on_length_mismatch() -> None:
    """Different lengths are never equivalent."""
    a = [_entry(1, 0, 1_000, "hi")]
    b = [_entry(1, 0, 1_000, "hi"), _entry(2, 1_000, 2_000, "world")]
    assert entries_equivalent(a, b) is False


def test_entries_equivalent_false_on_index_mismatch() -> None:
    """Different ``index`` breaks equivalence."""
    a = [_entry(index=1)]
    b = [_entry(index=2)]
    assert entries_equivalent(a, b) is False


def test_entries_equivalent_false_on_start_mismatch() -> None:
    """Different ``start_ms`` breaks equivalence."""
    a = [_entry(start_ms=0)]
    b = [_entry(start_ms=100)]
    assert entries_equivalent(a, b) is False


def test_entries_equivalent_false_on_end_mismatch() -> None:
    """Different ``end_ms`` breaks equivalence."""
    a = [_entry(end_ms=1_000)]
    b = [_entry(end_ms=1_500)]
    assert entries_equivalent(a, b) is False


def test_entries_equivalent_ignores_crlf_vs_lf() -> None:
    """Text differing only in line-ending style is considered equivalent."""
    a = [_entry(text="line one\r\nline two")]
    b = [_entry(text="line one\nline two")]
    assert entries_equivalent(a, b) is True


def test_entries_equivalent_ignores_trailing_whitespace() -> None:
    """Text differing only in trailing whitespace is considered equivalent."""
    a = [_entry(text="hello   \nworld\t")]
    b = [_entry(text="hello\nworld")]
    assert entries_equivalent(a, b) is True


def test_entries_equivalent_detects_real_text_difference() -> None:
    """Substantive text differences are still caught."""
    a = [_entry(text="hello")]
    b = [_entry(text="hola")]
    assert entries_equivalent(a, b) is False


def test_entries_equivalent_accepts_tuples() -> None:
    """Helper works with any ``Sequence[SubtitleEntry]``, not just lists."""
    a = (_entry(1), _entry(2, start_ms=1_000, end_ms=2_000, text="x"))
    b = [_entry(1), _entry(2, start_ms=1_000, end_ms=2_000, text="x")]
    assert entries_equivalent(a, b) is True


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exports_public_api() -> None:
    """Public API is re-exported from the models package."""
    assert set(se_module.__all__) == {
        "SubtitleEntry",
        "normalize_text",
        "entries_equivalent",
    }
    from translation_dubbing_skill import models as models_pkg

    for name in ("SubtitleEntry", "normalize_text", "entries_equivalent"):
        assert name in models_pkg.__all__
        assert hasattr(models_pkg, name)
