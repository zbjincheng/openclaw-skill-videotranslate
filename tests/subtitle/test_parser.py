"""Unit tests for :class:`SubtitleParser`.

Covers requirements R2.1 (SRT parse), R2.2 (VTT parse), R2.3 (entry shape),
R2.4 (syntax errors with ``line_number``), R2.5 (start > end rejected with
``entry_index``).
"""

from __future__ import annotations

import pytest

from translation_dubbing_skill.errors import (
    InvalidTimestampError,
    SubtitleParseError,
)
from translation_dubbing_skill.models import SubtitleEntry
from translation_dubbing_skill.subtitle import SubtitleParser


@pytest.fixture
def parser() -> SubtitleParser:
    return SubtitleParser()


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------


def test_parse_srt_single_entry(parser: SubtitleParser) -> None:
    """A minimal single-block SRT parses into one entry."""
    text = (
        "1\n"
        "00:00:01,000 --> 00:00:02,500\n"
        "Hello, world!\n"
    )
    entries = parser.parse_srt(text)
    assert entries == [
        SubtitleEntry(index=1, start_ms=1_000, end_ms=2_500, text="Hello, world!")
    ]


def test_parse_srt_multiple_entries_with_multiline_text(
    parser: SubtitleParser,
) -> None:
    """Multiple blocks, multi-line text, blank-line separation."""
    text = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "First line\n"
        "Second line\n"
        "\n"
        "2\n"
        "00:00:03,000 --> 00:00:04,000\n"
        "Second cue\n"
    )
    entries = parser.parse_srt(text)
    assert entries == [
        SubtitleEntry(
            index=1, start_ms=1_000, end_ms=2_000, text="First line\nSecond line"
        ),
        SubtitleEntry(index=2, start_ms=3_000, end_ms=4_000, text="Second cue"),
    ]


def test_parse_srt_crlf_line_endings(parser: SubtitleParser) -> None:
    """CRLF line endings are handled transparently."""
    text = (
        "1\r\n"
        "00:00:00,000 --> 00:00:01,000\r\n"
        "Hi\r\n"
        "\r\n"
        "2\r\n"
        "00:00:01,500 --> 00:00:02,500\r\n"
        "Second\r\n"
    )
    entries = parser.parse_srt(text)
    assert [e.index for e in entries] == [1, 2]
    assert entries[0].text == "Hi"
    assert entries[1].text == "Second"


def test_parse_srt_strips_utf8_bom(parser: SubtitleParser) -> None:
    """A leading BOM is stripped before parsing."""
    text = (
        "\ufeff1\n"
        "00:00:00,100 --> 00:00:00,900\n"
        "Hi\n"
    )
    entries = parser.parse_srt(text)
    assert entries[0] == SubtitleEntry(
        index=1, start_ms=100, end_ms=900, text="Hi"
    )


def test_parse_srt_empty_text_block_is_allowed(parser: SubtitleParser) -> None:
    """A cue with no text lines parses as an empty-text entry."""
    text = "1\n00:00:00,000 --> 00:00:01,000\n\n"
    entries = parser.parse_srt(text)
    assert entries == [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="")
    ]


def test_parse_srt_empty_input_yields_empty_list(parser: SubtitleParser) -> None:
    assert parser.parse_srt("") == []
    assert parser.parse_srt("\n\n\n") == []


def test_parse_srt_preserves_original_indices(parser: SubtitleParser) -> None:
    """Non-sequential indices in the file are preserved verbatim."""
    text = (
        "7\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "A\n"
        "\n"
        "42\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "B\n"
    )
    entries = parser.parse_srt(text)
    assert [e.index for e in entries] == [7, 42]


def test_parse_srt_hour_rollover(parser: SubtitleParser) -> None:
    """Timestamps past the one-hour mark parse correctly."""
    text = (
        "1\n"
        "01:02:03,456 --> 01:02:04,000\n"
        "T\n"
    )
    entries = parser.parse_srt(text)
    expected_start = ((1 * 60 + 2) * 60 + 3) * 1000 + 456
    expected_end = ((1 * 60 + 2) * 60 + 4) * 1000
    assert entries[0].start_ms == expected_start
    assert entries[0].end_ms == expected_end


def test_parse_srt_rejects_non_integer_index(parser: SubtitleParser) -> None:
    """The index line must be an integer; failure reports the exact line."""
    text = (
        "not-a-number\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "T\n"
    )
    with pytest.raises(SubtitleParseError) as exc:
        parser.parse_srt(text)
    assert exc.value.context["line_number"] == 1


def test_parse_srt_rejects_missing_timestamp(parser: SubtitleParser) -> None:
    """An index at the very end of the input without a timestamp is rejected."""
    # Pure "1" (no trailing newline) leaves the timestamp slot entirely
    # missing; the parser points at the index line.
    with pytest.raises(SubtitleParseError) as exc:
        parser.parse_srt("1")
    assert exc.value.context["line_number"] == 1


def test_parse_srt_rejects_malformed_timestamp_line(
    parser: SubtitleParser,
) -> None:
    """Timestamp line missing the arrow triggers a syntax error."""
    text = (
        "1\n"
        "00:00:00,000 to 00:00:01,000\n"
        "T\n"
    )
    with pytest.raises(SubtitleParseError) as exc:
        parser.parse_srt(text)
    assert exc.value.context["line_number"] == 2


def test_parse_srt_rejects_vtt_style_timestamp(parser: SubtitleParser) -> None:
    """SRT parser rejects the ``.mmm`` VTT-style separator."""
    text = (
        "1\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "T\n"
    )
    with pytest.raises(SubtitleParseError) as exc:
        parser.parse_srt(text)
    assert exc.value.context["line_number"] == 2


def test_parse_srt_rejects_start_after_end(parser: SubtitleParser) -> None:
    """``start_ms > end_ms`` raises ``InvalidTimestampError`` with entry index."""
    text = (
        "1\n"
        "00:00:05,000 --> 00:00:02,000\n"
        "Bad\n"
    )
    with pytest.raises(InvalidTimestampError) as exc:
        parser.parse_srt(text)
    assert exc.value.context["entry_index"] == 1


def test_parse_srt_reports_correct_entry_index_for_second_block(
    parser: SubtitleParser,
) -> None:
    """When the second block has an invalid timestamp, ``entry_index == 2``."""
    text = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "OK\n"
        "\n"
        "2\n"
        "00:00:05,000 --> 00:00:02,000\n"
        "Bad\n"
    )
    with pytest.raises(InvalidTimestampError) as exc:
        parser.parse_srt(text)
    assert exc.value.context["entry_index"] == 2


def test_parse_srt_allows_start_equal_to_end(parser: SubtitleParser) -> None:
    """Instantaneous cues (``start == end``) are not rejected."""
    text = (
        "1\n"
        "00:00:01,000 --> 00:00:01,000\n"
        "Flash\n"
    )
    entries = parser.parse_srt(text)
    assert entries[0].start_ms == entries[0].end_ms == 1_000


def test_parse_srt_leading_blank_lines_are_tolerated(
    parser: SubtitleParser,
) -> None:
    """Blank lines before the first block are skipped."""
    text = (
        "\n\n"
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Hi\n"
    )
    entries = parser.parse_srt(text)
    assert len(entries) == 1 and entries[0].text == "Hi"


# ---------------------------------------------------------------------------
# VTT
# ---------------------------------------------------------------------------


def test_parse_vtt_single_entry(parser: SubtitleParser) -> None:
    """A minimal VTT file parses into one entry with a sequential index."""
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:02.500\n"
        "Hello, world!\n"
    )
    entries = parser.parse_vtt(text)
    assert entries == [
        SubtitleEntry(index=1, start_ms=1_000, end_ms=2_500, text="Hello, world!")
    ]


def test_parse_vtt_multiple_entries_with_identifiers(
    parser: SubtitleParser,
) -> None:
    """VTT identifier lines are consumed; indices are sequential."""
    text = (
        "WEBVTT\n"
        "\n"
        "cue-1\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "First\n"
        "\n"
        "cue-two\n"
        "00:00:03.000 --> 00:00:04.000\n"
        "Second\n"
        "Second line\n"
    )
    entries = parser.parse_vtt(text)
    assert entries == [
        SubtitleEntry(index=1, start_ms=1_000, end_ms=2_000, text="First"),
        SubtitleEntry(
            index=2, start_ms=3_000, end_ms=4_000, text="Second\nSecond line"
        ),
    ]


def test_parse_vtt_assigns_sequential_indices(parser: SubtitleParser) -> None:
    """Sequential indices are assigned regardless of cue identifiers."""
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "A\n"
        "\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "B\n"
        "\n"
        "00:00:02.000 --> 00:00:03.000\n"
        "C\n"
    )
    entries = parser.parse_vtt(text)
    assert [e.index for e in entries] == [1, 2, 3]


def test_parse_vtt_header_with_description(parser: SubtitleParser) -> None:
    """VTT allows free-form description after ``WEBVTT``."""
    text = (
        "WEBVTT - Sample caption file\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Hi\n"
    )
    entries = parser.parse_vtt(text)
    assert len(entries) == 1


def test_parse_vtt_skips_note_block(parser: SubtitleParser) -> None:
    """``NOTE`` blocks are skipped without raising or producing entries."""
    text = (
        "WEBVTT\n"
        "\n"
        "NOTE This is a comment.\n"
        "Spanning multiple lines.\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Real cue\n"
    )
    entries = parser.parse_vtt(text)
    assert entries == [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="Real cue")
    ]


def test_parse_vtt_crlf_line_endings(parser: SubtitleParser) -> None:
    text = (
        "WEBVTT\r\n"
        "\r\n"
        "00:00:00.000 --> 00:00:01.000\r\n"
        "Hi\r\n"
    )
    entries = parser.parse_vtt(text)
    assert len(entries) == 1 and entries[0].text == "Hi"


def test_parse_vtt_short_timestamp_form(parser: SubtitleParser) -> None:
    """VTT tolerates the short ``MM:SS.mmm`` form."""
    text = (
        "WEBVTT\n"
        "\n"
        "01:02.500 --> 01:05.000\n"
        "Short form\n"
    )
    entries = parser.parse_vtt(text)
    expected_start = (0 * 3600 + 1 * 60 + 2) * 1000 + 500
    expected_end = (0 * 3600 + 1 * 60 + 5) * 1000
    assert entries[0].start_ms == expected_start
    assert entries[0].end_ms == expected_end


def test_parse_vtt_ignores_cue_settings(parser: SubtitleParser) -> None:
    """Trailing cue settings on the timestamp line are ignored."""
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000 align:start line:50%\n"
        "Styled\n"
    )
    entries = parser.parse_vtt(text)
    assert entries == [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="Styled")
    ]


def test_parse_vtt_empty_text_block_is_allowed(parser: SubtitleParser) -> None:
    """Cues with no text lines produce empty-text entries."""
    text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n\n"
    entries = parser.parse_vtt(text)
    assert entries == [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="")
    ]


def test_parse_vtt_requires_webvtt_header(parser: SubtitleParser) -> None:
    """Missing or misspelled header is a syntax error on line 1."""
    text = (
        "00:00:00.000 --> 00:00:01.000\n"
        "Hi\n"
    )
    with pytest.raises(SubtitleParseError) as exc:
        parser.parse_vtt(text)
    assert exc.value.context["line_number"] == 1


def test_parse_vtt_empty_input_raises(parser: SubtitleParser) -> None:
    with pytest.raises(SubtitleParseError):
        parser.parse_vtt("")


def test_parse_vtt_rejects_srt_style_timestamp(parser: SubtitleParser) -> None:
    """VTT parser rejects the ``,mmm`` SRT-style separator."""
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "T\n"
    )
    with pytest.raises(SubtitleParseError) as exc:
        parser.parse_vtt(text)
    # line 3 is the bad timestamp
    assert exc.value.context["line_number"] == 3


def test_parse_vtt_rejects_start_after_end(parser: SubtitleParser) -> None:
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:05.000 --> 00:00:02.000\n"
        "Bad\n"
    )
    with pytest.raises(InvalidTimestampError) as exc:
        parser.parse_vtt(text)
    assert exc.value.context["entry_index"] == 1


def test_parse_vtt_reports_correct_entry_index_for_later_block(
    parser: SubtitleParser,
) -> None:
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "OK\n"
        "\n"
        "00:00:05.000 --> 00:00:02.000\n"
        "Bad\n"
    )
    with pytest.raises(InvalidTimestampError) as exc:
        parser.parse_vtt(text)
    assert exc.value.context["entry_index"] == 2


def test_parse_vtt_identifier_without_timestamp_is_error(
    parser: SubtitleParser,
) -> None:
    """An identifier line followed by a blank line is a syntax error."""
    text = (
        "WEBVTT\n"
        "\n"
        "lonely-identifier\n"
        "\n"
    )
    with pytest.raises(SubtitleParseError):
        parser.parse_vtt(text)


# ---------------------------------------------------------------------------
# parse_auto
# ---------------------------------------------------------------------------


def test_parse_auto_sniffs_vtt(parser: SubtitleParser) -> None:
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Hi\n"
    )
    assert parser.parse_auto(text) == [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="Hi")
    ]


def test_parse_auto_sniffs_srt(parser: SubtitleParser) -> None:
    text = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Hi\n"
    )
    assert parser.parse_auto(text) == [
        SubtitleEntry(index=1, start_ms=0, end_ms=1_000, text="Hi")
    ]


def test_parse_auto_honors_hint_srt(parser: SubtitleParser) -> None:
    """An explicit ``srt`` hint bypasses sniffing."""
    text = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Hi\n"
    )
    entries = parser.parse_auto(text, hint_format="srt")
    assert entries[0].start_ms == 0


def test_parse_auto_honors_hint_vtt(parser: SubtitleParser) -> None:
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Hi\n"
    )
    entries = parser.parse_auto(text, hint_format="vtt")
    assert entries[0].end_ms == 1_000


def test_parse_auto_hint_srt_on_vtt_fails(parser: SubtitleParser) -> None:
    """Forcing ``srt`` on a VTT file surfaces a syntax error."""
    text = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Hi\n"
    )
    with pytest.raises(SubtitleParseError):
        parser.parse_auto(text, hint_format="srt")


def test_parse_auto_rejects_unknown_hint(parser: SubtitleParser) -> None:
    with pytest.raises(ValueError):
        parser.parse_auto("anything", hint_format="ssa")  # type: ignore[arg-type]


def test_parse_auto_handles_leading_bom(parser: SubtitleParser) -> None:
    """BOM does not confuse the format sniffer."""
    text = (
        "\ufeffWEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "Hi\n"
    )
    entries = parser.parse_auto(text)
    assert len(entries) == 1
