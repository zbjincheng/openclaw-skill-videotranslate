"""Unit tests for the ``ProcessingMode`` enum and its default constant.

Covers requirement R1.3 at the data-model layer:
- Enum values match the manifest-facing strings.
- ``ProcessingMode`` is a ``str``-subclassed ``Enum`` (for manifest/JSON interop).
- ``DEFAULT_PROCESSING_MODE`` equals ``ProcessingMode.SUBTITLE_AND_DUBBING``.
"""

from __future__ import annotations

from enum import Enum

from translation_dubbing_skill.models import (
    DEFAULT_PROCESSING_MODE,
    ProcessingMode,
)
from translation_dubbing_skill.models import processing_mode as pm_module


def test_processing_mode_is_str_enum() -> None:
    """ProcessingMode is both an ``Enum`` and a ``str`` subclass."""
    assert issubclass(ProcessingMode, Enum)
    assert issubclass(ProcessingMode, str)


def test_processing_mode_enum_values() -> None:
    """Enum members expose the manifest-facing string values."""
    assert ProcessingMode.SUBTITLE_ONLY.value == "subtitle_only"
    assert ProcessingMode.SUBTITLE_AND_DUBBING.value == "subtitle_and_dubbing"


def test_processing_mode_only_two_members() -> None:
    """Exactly the two expected modes are defined; no extras drift in."""
    assert {m.value for m in ProcessingMode} == {
        "subtitle_only",
        "subtitle_and_dubbing",
    }


def test_processing_mode_str_interop() -> None:
    """String equality works thanks to the ``str`` mixin."""
    assert ProcessingMode.SUBTITLE_ONLY == "subtitle_only"
    assert ProcessingMode.SUBTITLE_AND_DUBBING == "subtitle_and_dubbing"


def test_processing_mode_from_value_roundtrip() -> None:
    """``ProcessingMode(value)`` recovers the enum member."""
    assert ProcessingMode("subtitle_only") is ProcessingMode.SUBTITLE_ONLY
    assert (
        ProcessingMode("subtitle_and_dubbing")
        is ProcessingMode.SUBTITLE_AND_DUBBING
    )


def test_default_processing_mode_is_subtitle_and_dubbing() -> None:
    """Default mode matches the design spec (R1.3)."""
    assert DEFAULT_PROCESSING_MODE is ProcessingMode.SUBTITLE_AND_DUBBING
    assert pm_module.DEFAULT_PROCESSING_MODE is ProcessingMode.SUBTITLE_AND_DUBBING
