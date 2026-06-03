"""Unit tests for :class:`AudioClip`.

Covers requirement R6.2 at the data-model layer:
- Frozen dataclass with the design-mandated fields.
- Field values round-trip through construction.
- Instances are immutable and hashable.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from translation_dubbing_skill.models import AudioClip
from translation_dubbing_skill.models import audio_clip as ac_module


def test_audio_clip_is_frozen_dataclass_with_expected_fields() -> None:
    """AudioClip is a frozen dataclass with the design-mandated fields."""
    assert is_dataclass(AudioClip)
    field_names = [f.name for f in fields(AudioClip)]
    assert field_names == [
        "entry_index",
        "start_ms",
        "end_ms",
        "audio",
        "duration_ms",
    ]


def test_audio_clip_stores_values() -> None:
    """Constructing with explicit values preserves them verbatim."""
    clip = AudioClip(
        entry_index=3,
        start_ms=1_000,
        end_ms=2_500,
        audio=b"\x00\x01\x02",
        duration_ms=1_450,
    )
    assert clip.entry_index == 3
    assert clip.start_ms == 1_000
    assert clip.end_ms == 2_500
    assert clip.audio == b"\x00\x01\x02"
    assert clip.duration_ms == 1_450


def test_audio_clip_is_immutable() -> None:
    """Frozen dataclass rejects attribute mutation."""
    clip = AudioClip(
        entry_index=1, start_ms=0, end_ms=1_000, audio=b"", duration_ms=0
    )
    with pytest.raises(FrozenInstanceError):
        clip.duration_ms = 500  # type: ignore[misc]


def test_audio_clip_equality_and_hash() -> None:
    """Value-based equality + hashability from ``frozen=True``."""
    a = AudioClip(
        entry_index=1, start_ms=0, end_ms=1_000, audio=b"abc", duration_ms=500
    )
    b = AudioClip(
        entry_index=1, start_ms=0, end_ms=1_000, audio=b"abc", duration_ms=500
    )
    c = AudioClip(
        entry_index=2, start_ms=0, end_ms=1_000, audio=b"abc", duration_ms=500
    )
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert {a, b, c} == {a, c}


def test_audio_clip_supports_empty_audio() -> None:
    """Empty audio bytes and zero duration are valid values."""
    clip = AudioClip(
        entry_index=0, start_ms=0, end_ms=0, audio=b"", duration_ms=0
    )
    assert clip.audio == b""
    assert clip.duration_ms == 0


def test_module_exports_public_api() -> None:
    """Public API is re-exported from the models package."""
    assert set(ac_module.__all__) == {"AudioClip"}
    from translation_dubbing_skill import models as models_pkg

    assert "AudioClip" in models_pkg.__all__
    assert hasattr(models_pkg, "AudioClip")
