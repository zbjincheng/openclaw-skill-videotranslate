"""Unit tests for :class:`ProgressEvent`.

Covers requirement R11.1 at the data-model layer:
- Frozen dataclass with the design-mandated fields.
- ``stage`` is constrained to the five valid lifecycle stages
  (parsing / translating / tts / muxing / done).
- ``completed`` / ``total`` / ``extra`` default to ``None``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import get_args

import pytest

from translation_dubbing_skill.models import ProgressEvent, ProgressStage
from translation_dubbing_skill.models import progress_event as pe_module


def test_progress_event_is_frozen_dataclass_with_expected_fields() -> None:
    """ProgressEvent is a frozen dataclass with the design-mandated fields."""
    assert is_dataclass(ProgressEvent)
    field_names = [f.name for f in fields(ProgressEvent)]
    assert field_names == ["stage", "message", "completed", "total", "extra"]


def test_progress_stage_literal_matches_design() -> None:
    """ProgressStage enumerates exactly the five design stages."""
    assert set(get_args(ProgressStage)) == {
        "parsing",
        "translating",
        "tts",
        "muxing",
        "done",
    }


def test_progress_event_minimal_construction() -> None:
    """Optional fields default to ``None`` when omitted."""
    event = ProgressEvent(stage="parsing", message="Parsing subtitles")
    assert event.stage == "parsing"
    assert event.message == "Parsing subtitles"
    assert event.completed is None
    assert event.total is None
    assert event.extra is None


def test_progress_event_full_construction() -> None:
    """All fields round-trip through construction."""
    event = ProgressEvent(
        stage="translating",
        message="Translated 5/10",
        completed=5,
        total=10,
        extra={"batch_id": 2},
    )
    assert event.stage == "translating"
    assert event.message == "Translated 5/10"
    assert event.completed == 5
    assert event.total == 10
    assert event.extra == {"batch_id": 2}


@pytest.mark.parametrize(
    "stage", ["parsing", "translating", "tts", "muxing", "done"]
)
def test_progress_event_accepts_each_valid_stage(stage: str) -> None:
    """Every documented stage is accepted."""
    event = ProgressEvent(stage=stage, message="x")  # type: ignore[arg-type]
    assert event.stage == stage


def test_progress_event_done_stage_carries_extra_paths() -> None:
    """The terminal ``done`` event typically carries output paths in ``extra``."""
    event = ProgressEvent(
        stage="done",
        message="Completed",
        extra={
            "output_video_path": "/tmp/out.mkv",
            "output_subtitle_path": "/tmp/out.zh.srt",
        },
    )
    assert event.stage == "done"
    assert event.extra is not None
    assert event.extra["output_video_path"] == "/tmp/out.mkv"


def test_progress_event_is_immutable() -> None:
    """Frozen dataclass rejects attribute mutation."""
    event = ProgressEvent(stage="parsing", message="x")
    with pytest.raises(FrozenInstanceError):
        event.message = "y"  # type: ignore[misc]


def test_progress_event_equality_and_hash() -> None:
    """Value-based equality + hashability from ``frozen=True``."""
    a = ProgressEvent(stage="translating", message="m", completed=1, total=2)
    b = ProgressEvent(stage="translating", message="m", completed=1, total=2)
    c = ProgressEvent(stage="translating", message="m", completed=2, total=2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_module_exports_public_api() -> None:
    """Public API is re-exported from the models package."""
    assert set(pe_module.__all__) == {"ProgressEvent", "ProgressStage"}
    from translation_dubbing_skill import models as models_pkg

    for name in ("ProgressEvent", "ProgressStage"):
        assert name in models_pkg.__all__
        assert hasattr(models_pkg, name)
