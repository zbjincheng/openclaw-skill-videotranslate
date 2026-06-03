"""Property-based tests for :class:`SubtitleParser` error paths.

This module hosts property tests that exercise the parser's rejection
behaviour. Round-trip equivalence properties live alongside the
serializer in :mod:`tests.subtitle.test_serializer_properties`; this
file focuses on *detection* properties where the parser must raise a
specific error at a specific position.

## Property 4: 非法时间戳检测

    For any SRT or VTT text where exactly one subtitle entry has
    ``start_ms > end_ms`` and every earlier entry is well-formed,
    :meth:`SubtitleParser.parse_srt` / :meth:`SubtitleParser.parse_vtt`
    raises :class:`InvalidTimestampError` whose ``context["entry_index"]``
    equals the 1-based ordinal position of the offending entry as
    counted in the parser's output list.

    This matches the parser's reporting contract: ``entry_index`` is the
    position the entry *would have* occupied in the returned list
    (``len(entries) + 1`` at the moment of detection), not the SRT cue
    index embedded in the file or the WebVTT cue identifier.

**Validates: Requirement 2.5**
"""

from __future__ import annotations

from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from translation_dubbing_skill.errors import (
    InvalidTimestampError,
    SubtitleParseError,
)
from translation_dubbing_skill.models import SubtitleEntry
from translation_dubbing_skill.subtitle import SubtitleParser, SubtitleSerializer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Cap times well under 24h so HH fits two digits — matches the serializer
# formatting domain (``{hours:02d}``).
_MAX_MS: int = 23 * 3_600_000 + 59 * 60_000 + 59 * 1_000 + 999


def _safe_text_strategy() -> st.SearchStrategy[str]:
    """Generate a single-line subtitle text that cannot confuse the block parser.

    Constraints:
      - No newlines (so the cue spans exactly one text line and we don't
        need to worry about the "blank line inside text" hazard).
      - No control characters other than ordinary printable ASCII /
        common Unicode letters — keeps examples focused on the timestamp
        validation path rather than parser tokenisation edge cases.
      - Not an all-digit string, so that the SRT parser cannot mistake
        the text for a subsequent cue's index line.

    A small alphabet keeps shrunk counterexamples readable.
    """
    return (
        st.text(
            alphabet=st.characters(
                min_codepoint=0x41,  # 'A'
                max_codepoint=0x7A,  # 'z'
                blacklist_categories=("Cs",),
            ),
            min_size=1,
            max_size=8,
        )
        # ``isalpha`` is stricter than we need but guarantees no digit
        # collisions with SRT index lines; letters-only is fine for the
        # property being exercised.
        .filter(str.isalpha)
    )


def _valid_entry_strategy(index: int) -> st.SearchStrategy[SubtitleEntry]:
    """A subtitle entry with ``start_ms <= end_ms`` (well-formed)."""
    return st.builds(
        lambda times, text: SubtitleEntry(
            index=index,
            start_ms=min(times),
            end_ms=max(times),
            text=text,
        ),
        times=st.tuples(
            st.integers(min_value=0, max_value=_MAX_MS),
            st.integers(min_value=0, max_value=_MAX_MS),
        ),
        text=_safe_text_strategy(),
    )


def _invalid_entry_strategy(index: int) -> st.SearchStrategy[SubtitleEntry]:
    """A subtitle entry with ``start_ms > end_ms`` (ill-formed by R2.5).

    We draw two distinct millisecond values and assign the *larger* one
    to ``start_ms`` and the *smaller* one to ``end_ms`` so the
    ``>`` relation is strict, not merely ``>=``.
    """
    return st.builds(
        lambda times, text: SubtitleEntry(
            index=index,
            start_ms=max(times),
            end_ms=min(times),
            text=text,
        ),
        # ``unique=True`` guarantees ``max > min`` strictly.
        times=st.lists(
            st.integers(min_value=0, max_value=_MAX_MS),
            min_size=2,
            max_size=2,
            unique=True,
        ),
        text=_safe_text_strategy(),
    )


@st.composite
def entries_with_one_invalid_at_known_position(
    draw: st.DrawFn,
) -> tuple[List[SubtitleEntry], int]:
    """Generate an entries list where position ``k`` (1-based) is invalid.

    All entries before position ``k`` are well-formed; entries after
    position ``k`` are irrelevant to the property since the parser
    raises on the *first* bad entry it encounters, but we still
    generate them (as well-formed) to exercise the "bad cue in the
    middle of a file" path too.

    Returns:
        A tuple ``(entries, k)`` where ``k`` is the 1-based position
        of the single invalid entry in ``entries``.
    """
    total = draw(st.integers(min_value=1, max_value=5))
    bad_position = draw(st.integers(min_value=1, max_value=total))

    entries: List[SubtitleEntry] = []
    for pos in range(1, total + 1):
        # Use ``pos`` as the SRT index so the file's own ordinals are
        # sequential; the parser reports ``entry_index`` based on output
        # list position, not this file-embedded index, so the property
        # holds regardless.
        if pos == bad_position:
            entries.append(draw(_invalid_entry_strategy(index=pos)))
        else:
            entries.append(draw(_valid_entry_strategy(index=pos)))
    return entries, bad_position


# ---------------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------------
#
# The serializer's :meth:`to_srt` / :meth:`to_vtt` happily format entries
# whose ``start_ms > end_ms`` — :class:`SubtitleEntry` does not validate
# ordering at construction time, and the serializer only rejects negative
# millisecond values. That makes the serializer a convenient "text
# constructor" for this property: we can build a syntactically well-formed
# SRT/VTT document whose timestamp arithmetic is deliberately inverted at
# a known cue, then hand the text back to the parser.


def _serialize_as_srt(entries: List[SubtitleEntry]) -> str:
    return SubtitleSerializer().to_srt(entries)


def _serialize_as_vtt(entries: List[SubtitleEntry]) -> str:
    return SubtitleSerializer().to_vtt(entries)


# ---------------------------------------------------------------------------
# Property 4: SRT path
# ---------------------------------------------------------------------------


@given(entries_with_one_invalid_at_known_position())
@settings(max_examples=200)
def test_parse_srt_raises_invalid_timestamp_at_entry_index(
    case: tuple[List[SubtitleEntry], int],
) -> None:
    """**Validates: Requirement 2.5**

    For SRT text containing an entry with ``start_ms > end_ms`` at
    1-based position ``k`` (all earlier entries well-formed), the
    parser raises :class:`InvalidTimestampError` with
    ``context["entry_index"] == k``.
    """
    entries, bad_position = case
    text = _serialize_as_srt(entries)

    with pytest.raises(InvalidTimestampError) as exc_info:
        SubtitleParser().parse_srt(text)

    # The parser's contract is "entry_index is the 1-based ordinal in
    # the output list the offender would have occupied", which — since
    # every earlier entry is well-formed and parses successfully — is
    # exactly ``bad_position``.
    assert exc_info.value.context["entry_index"] == bad_position, (
        "expected entry_index == bad_position\n"
        f"bad_position: {bad_position}\n"
        f"entries: {entries!r}\n"
        f"serialized: {text!r}\n"
        f"context: {exc_info.value.context!r}"
    )


# ---------------------------------------------------------------------------
# Property 4: VTT path
# ---------------------------------------------------------------------------


@given(entries_with_one_invalid_at_known_position())
@settings(max_examples=200)
def test_parse_vtt_raises_invalid_timestamp_at_entry_index(
    case: tuple[List[SubtitleEntry], int],
) -> None:
    """**Validates: Requirement 2.5**

    Same as :func:`test_parse_srt_raises_invalid_timestamp_at_entry_index`
    but exercises the VTT parser path. The VTT parser assigns
    sequential 1-based ``entry_index`` values regardless of any cue
    identifier emitted by the serializer, so the property holds with
    the same ``bad_position`` value.
    """
    entries, bad_position = case
    text = _serialize_as_vtt(entries)

    with pytest.raises(InvalidTimestampError) as exc_info:
        SubtitleParser().parse_vtt(text)

    assert exc_info.value.context["entry_index"] == bad_position, (
        "expected entry_index == bad_position\n"
        f"bad_position: {bad_position}\n"
        f"entries: {entries!r}\n"
        f"serialized: {text!r}\n"
        f"context: {exc_info.value.context!r}"
    )


# ---------------------------------------------------------------------------
# Property 5: 非法字幕文本检测
# ---------------------------------------------------------------------------
#
# For any SRT text obtained by corrupting a well-formed document in one of
# the following ways, :meth:`SubtitleParser.parse_srt` raises
# :class:`SubtitleParseError` whose ``context["line_number"]`` lies in the
# closed interval ``[1, total_lines]`` where ``total_lines`` counts the
# lines of the corrupted text (using the parser's ``\n`` splitter).
#
# The corruption mutators exercise four structurally distinct failure
# modes explicitly called out by R2.4:
#
#   * ``break-timestamp-separator`` — rewrites ``-->`` so the timestamp
#     line no longer contains the arrow token.
#   * ``drop-index`` — deletes a cue's numeric index line, shifting the
#     timestamp line into the slot where the parser expects an integer.
#   * ``invalid-timestamp-format`` — replaces the timestamp line with
#     free-form garbage that still contains ``-->`` so the failure point
#     is the timestamp token, not the arrow.
#   * ``non-numeric-index`` — overwrites the index line with a token that
#     is not parseable as an integer.
#
# **Validates: Requirement 2.4**


# Map from cue position (1-based) to the line number at which its index
# line and timestamp line live in the serializer's normal form.
#
# The SRT serializer emits blocks in the shape::
#
#   <index>\n<timestamp>\n<text...>\n\n
#
# For ``_safe_text_strategy`` texts (single line, never empty) each cue
# therefore occupies exactly four lines in the unified-LF split: index,
# timestamp, text, trailing blank. The final cue also ends with a blank
# line because the serializer terminates the file with ``\n``. Given a
# 1-based ``pos``, the index line sits at ``4 * (pos - 1) + 1`` and the
# timestamp line at the next line.


_SRT_ARROW = "-->"


def _srt_index_line_number(pos: int) -> int:
    """1-based line number of cue ``pos``'s index line in the normal form."""
    return 4 * (pos - 1) + 1


def _srt_timestamp_line_number(pos: int) -> int:
    """1-based line number of cue ``pos``'s timestamp line in the normal form."""
    return _srt_index_line_number(pos) + 1


@st.composite
def _well_formed_srt_entries(draw: st.DrawFn) -> List[SubtitleEntry]:
    """Generate a non-empty list of well-formed SRT entries.

    Text fields are drawn from :func:`_safe_text_strategy` — single-line,
    letters-only, non-empty — so the serializer's output has a
    predictable "4 lines per cue" shape and cue text cannot be mistaken
    for an index line by the parser.
    """
    total = draw(st.integers(min_value=1, max_value=4))
    entries: List[SubtitleEntry] = []
    for pos in range(1, total + 1):
        entries.append(draw(_valid_entry_strategy(index=pos)))
    return entries


# Replacement tokens used by the corruption mutators. Each is chosen so
# the resulting line cannot be interpreted the way the parser would
# otherwise interpret it.

# Arrows that are *not* ``-->``. Keeping them free of the literal
# substring is the only requirement — the parser looks for ``-->``
# verbatim.
_BROKEN_ARROWS: Final[tuple[str, ...]] = ("-->>", "->", "==>", "to", "-", "")

# Timestamp line replacements that still contain ``-->`` so the failure
# lands on the timestamp parse step rather than the arrow-missing step.
_GARBAGE_TIMESTAMP_LINES: Final[tuple[str, ...]] = (
    "garbage --> alsogarbage",
    "not-a-time --> also-not-a-time",
    "aa:bb:cc,ddd --> ee:ff:gg,hhh",
    "00:00:XX,000 --> 00:00:01,000",
    "99:99:99,999 --> 00:00:01,000",
)

# Non-numeric, non-blank replacements for the index line. A leading
# alphabetic character guarantees ``int()`` rejects the token and that
# ``.strip()`` does not collapse it to an empty string (which the parser
# treats as a block separator instead of a malformed index).
_NON_NUMERIC_INDICES: Final[tuple[str, ...]] = (
    "abc",
    "one",
    "X",
    "cue-1",
    "1.5",  # ``int('1.5')`` raises — dots are not tolerated by ``int``
    "1a",
)


def _corrupt_break_timestamp_separator(
    lines: list[str], pos: int
) -> tuple[list[str], int]:
    """Replace the ``-->`` arrow in cue ``pos``'s timestamp line.

    Returns the mutated line list and the 1-based line number where the
    parser is expected to fault (the timestamp line itself).
    """
    target = _srt_timestamp_line_number(pos) - 1  # 0-based index
    original = lines[target]
    # Substitute the first occurrence of the arrow with a broken token
    # drawn from the fixture. If several replacements are viable we pick
    # the first deterministically — the property does not depend on
    # which broken token we chose.
    replacement = _BROKEN_ARROWS[pos % len(_BROKEN_ARROWS)]
    mutated = original.replace(_SRT_ARROW, replacement, 1)
    new_lines = list(lines)
    new_lines[target] = mutated
    return new_lines, target + 1


def _corrupt_drop_index(
    lines: list[str], pos: int
) -> tuple[list[str], int]:
    """Delete cue ``pos``'s index line.

    After deletion the parser encounters the timestamp line where it
    expected an integer index; the reported ``line_number`` is the line
    the timestamp now sits on (i.e. the old index line's slot).
    """
    target = _srt_index_line_number(pos) - 1  # 0-based
    new_lines = lines[:target] + lines[target + 1 :]
    # The timestamp line has moved up into the former index slot.
    return new_lines, target + 1


def _corrupt_invalid_timestamp_format(
    lines: list[str], pos: int
) -> tuple[list[str], int]:
    """Replace cue ``pos``'s timestamp line with garbage containing ``-->``."""
    target = _srt_timestamp_line_number(pos) - 1
    replacement = _GARBAGE_TIMESTAMP_LINES[
        pos % len(_GARBAGE_TIMESTAMP_LINES)
    ]
    new_lines = list(lines)
    new_lines[target] = replacement
    return new_lines, target + 1


def _corrupt_non_numeric_index(
    lines: list[str], pos: int
) -> tuple[list[str], int]:
    """Replace cue ``pos``'s index line with a non-numeric token."""
    target = _srt_index_line_number(pos) - 1
    replacement = _NON_NUMERIC_INDICES[pos % len(_NON_NUMERIC_INDICES)]
    new_lines = list(lines)
    new_lines[target] = replacement
    return new_lines, target + 1


# Registry of mutators. The strategy picks one per example. The mutator
# signature is ``(lines, pos) -> (new_lines, expected_fault_line)``; the
# "expected fault line" is retained in case a future tightening wants to
# assert exact positions, but the property only asserts the interval.
_MUTATORS: Final[tuple[str, ...]] = (
    "break-timestamp-separator",
    "drop-index",
    "invalid-timestamp-format",
    "non-numeric-index",
)

_MUTATOR_FNS = {
    "break-timestamp-separator": _corrupt_break_timestamp_separator,
    "drop-index": _corrupt_drop_index,
    "invalid-timestamp-format": _corrupt_invalid_timestamp_format,
    "non-numeric-index": _corrupt_non_numeric_index,
}


@st.composite
def malformed_srt_text(draw: st.DrawFn) -> str:
    """Generate an SRT document corrupted by exactly one R2.4 mutator."""
    entries = draw(_well_formed_srt_entries())
    text = _serialize_as_srt(entries)
    # Count lines with the same convention the parser uses so the
    # property's ``total_lines`` is comparable to the parser's reported
    # ``line_number``.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    mutator_name = draw(st.sampled_from(_MUTATORS))
    pos = draw(st.integers(min_value=1, max_value=len(entries)))
    mutated_lines, _ = _MUTATOR_FNS[mutator_name](lines, pos)
    return "\n".join(mutated_lines)


@given(malformed_srt_text())
@settings(max_examples=200)
def test_parse_srt_raises_subtitle_parse_error_with_valid_line_number(
    text: str,
) -> None:
    """**Validates: Requirement 2.4**

    For any SRT text produced by a single-site structural corruption
    (broken arrow, dropped index, garbage timestamp, or non-numeric
    index), the parser raises :class:`SubtitleParseError` with
    ``1 <= context["line_number"] <= total_lines``, where
    ``total_lines`` is the line count of the corrupted text under the
    parser's own line splitter.
    """
    # Use the parser's line-splitting convention so the bound we check
    # against matches the basis on which the parser assigns line numbers.
    total_lines = len(
        text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )

    with pytest.raises(SubtitleParseError) as exc_info:
        SubtitleParser().parse_srt(text)

    line_number = exc_info.value.context.get("line_number")
    assert isinstance(line_number, int), (
        "SubtitleParseError.context must carry an integer 'line_number'\n"
        f"text: {text!r}\n"
        f"context: {exc_info.value.context!r}"
    )
    assert 1 <= line_number <= total_lines, (
        "line_number must fall within [1, total_lines]\n"
        f"line_number: {line_number}\n"
        f"total_lines: {total_lines}\n"
        f"text: {text!r}\n"
        f"context: {exc_info.value.context!r}"
    )
