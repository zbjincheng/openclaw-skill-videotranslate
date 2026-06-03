"""Unit tests for :class:`SubtitleSerializer`.

Covers requirements R3.1 (to SRT), R3.2 (to VTT), R3.3 (index/timestamps/text
preserved), R3.4 (UTF-8 file output). Round-trip property tests live in
the companion ``test_serializer_properties.py`` module (tasks 3.4–3.7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from translation_dubbing_skill.models import SubtitleEntry
from translation_dubbing_skill.subtitle import SubtitleParser, SubtitleSerializer


@pytest.fixture
def serializer() -> SubtitleSerializer:
    return SubtitleSerializer()


@pytest.fixture
def parser() -> SubtitleParser:
    return SubtitleParser()


# ---------------------------------------------------------------------------
# SRT output (R3.1, R3.3)
# ---------------------------------------------------------------------------


def test_to_srt_single_entry_matches_normal_form(
    serializer: SubtitleSerializer,
) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=1_000, end_ms=2_500, text="Hello, world!"),
    ]
    assert serializer.to_srt(entries) == (
        "1\n"
        "00:00:01,000 --> 00:00:02,500\n"
        "Hello, world!\n"
    )


def test_to_srt_multiple_entries_separated_by_blank_line(
    serializer: SubtitleSerializer,
) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=1_000, end_ms=2_000, text="First"),
        SubtitleEntry(index=2, start_ms=3_000, end_ms=4_000, text="Second"),
    ]
    assert serializer.to_srt(entries) == (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "First\n"
        "\n"
        "2\n"
        "00:00:03,000 --> 00:00:04,000\n"
        "Second\n"
    )


def test_to_srt_preserves_original_index(serializer: SubtitleSerializer) -> None:
    """R3.3: entry.index passes through verbatim (not renumbered)."""
    entries = [
        SubtitleEntry(index=42, start_ms=0, end_ms=1_000, text="a"),
        SubtitleEntry(index=100, start_ms=2_000, end_ms=3_000, text="b"),
    ]
    output = serializer.to_srt(entries)
    assert output.startswith("42\n")
    assert "\n100\n" in output


def test_to_srt_multiline_text_preserved(serializer: SubtitleSerializer) -> None:
    entries = [
        SubtitleEntry(
            index=1, start_ms=0, end_ms=1_000, text="Line one\nLine two\nLine three"
        ),
    ]
    assert serializer.to_srt(entries) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Line one\n"
        "Line two\n"
        "Line three\n"
    )


def test_to_srt_strips_trailing_whitespace_from_text_lines(
    serializer: SubtitleSerializer,
) -> None:
    """Normal Form: trailing whitespace (including tabs) is removed per line."""
    entries = [
        SubtitleEntry(
            index=1,
            start_ms=0,
            end_ms=1_000,
            text="hello  \nworld\t\n  leading kept",
        ),
    ]
    output = serializer.to_srt(entries)
    assert "hello\nworld\n  leading kept\n" in output


def test_to_srt_normalizes_crlf_in_text(serializer: SubtitleSerializer) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="a\r\nb\rc"),
    ]
    output = serializer.to_srt(entries)
    # CRLF and lone CR both collapse to LF; result uses LF-only Normal Form.
    assert "\r" not in output
    assert "a\nb\nc" in output


def test_to_srt_formats_hours_minutes_seconds_milliseconds(
    serializer: SubtitleSerializer,
) -> None:
    one_hour_two_min_three_sec_four_ms = (
        1 * 3_600_000 + 2 * 60_000 + 3 * 1_000 + 4
    )
    entries = [
        SubtitleEntry(
            index=1,
            start_ms=0,
            end_ms=one_hour_two_min_three_sec_four_ms,
            text="x",
        ),
    ]
    assert "00:00:00,000 --> 01:02:03,004" in serializer.to_srt(entries)


def test_to_srt_empty_list_returns_empty_string(
    serializer: SubtitleSerializer,
) -> None:
    assert serializer.to_srt([]) == ""


# ---------------------------------------------------------------------------
# VTT output (R3.2, R3.3)
# ---------------------------------------------------------------------------


def test_to_vtt_single_entry_with_header(serializer: SubtitleSerializer) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=1_000, end_ms=2_500, text="Hello, world!"),
    ]
    assert serializer.to_vtt(entries) == (
        "WEBVTT\n"
        "\n"
        "1\n"
        "00:00:01.000 --> 00:00:02.500\n"
        "Hello, world!\n"
    )


def test_to_vtt_uses_dot_as_millisecond_separator(
    serializer: SubtitleSerializer,
) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=1_234, end_ms=5_678, text="t"),
    ]
    output = serializer.to_vtt(entries)
    assert "00:00:01.234 --> 00:00:05.678" in output
    # Ensure no SRT-style comma separator leaks into the VTT output.
    assert ",234" not in output
    assert ",678" not in output


def test_to_vtt_multiple_entries_separated_by_blank_line(
    serializer: SubtitleSerializer,
) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="a"),
        SubtitleEntry(index=2, start_ms=1_500, end_ms=2_500, text="b"),
    ]
    assert serializer.to_vtt(entries) == (
        "WEBVTT\n"
        "\n"
        "1\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "a\n"
        "\n"
        "2\n"
        "00:00:01.500 --> 00:00:02.500\n"
        "b\n"
    )


def test_to_vtt_empty_list_emits_header_only(
    serializer: SubtitleSerializer,
) -> None:
    assert serializer.to_vtt([]) == "WEBVTT\n"


# ---------------------------------------------------------------------------
# Parser round-trip (R4.1, R4.2, R4.3) — simple smoke checks
# ---------------------------------------------------------------------------


def test_srt_round_trip_simple(
    parser: SubtitleParser, serializer: SubtitleSerializer
) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="hi"),
        SubtitleEntry(index=2, start_ms=2_500, end_ms=3_750, text="你好\nworld"),
    ]
    assert parser.parse_srt(serializer.to_srt(entries)) == entries


def test_vtt_round_trip_simple(
    parser: SubtitleParser, serializer: SubtitleSerializer
) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="hi"),
        SubtitleEntry(index=2, start_ms=2_500, end_ms=3_750, text="你好\nworld"),
    ]
    assert parser.parse_vtt(serializer.to_vtt(entries)) == entries


# ---------------------------------------------------------------------------
# write_file (R3.4)
# ---------------------------------------------------------------------------


def test_write_file_writes_utf8_bytes(
    serializer: SubtitleSerializer, tmp_path: Path
) -> None:
    entries = [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="你好 🌏"),
    ]
    content = serializer.to_srt(entries)
    target = tmp_path / "out.srt"
    serializer.write_file(target, content)

    # Round-trip the bytes: decode as UTF-8 and confirm the stored text
    # matches what we wrote.
    raw = target.read_bytes()
    assert raw.decode("utf-8") == content
    assert "你好 🌏".encode("utf-8") in raw


def test_write_file_creates_parent_directories(
    serializer: SubtitleSerializer, tmp_path: Path
) -> None:
    target = tmp_path / "nested" / "dir" / "out.vtt"
    serializer.write_file(target, "WEBVTT\n")
    assert target.read_text(encoding="utf-8") == "WEBVTT\n"


def test_write_file_preserves_lf_line_endings_on_any_platform(
    serializer: SubtitleSerializer, tmp_path: Path
) -> None:
    """`newline=""` must disable universal-newlines translation."""
    entries = [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="a"),
        SubtitleEntry(index=2, start_ms=2_000, end_ms=3_000, text="b"),
    ]
    content = serializer.to_srt(entries)
    target = tmp_path / "out.srt"
    serializer.write_file(target, content)
    raw = target.read_bytes()
    assert b"\r\n" not in raw
