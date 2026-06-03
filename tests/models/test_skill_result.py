"""Unit tests for :class:`SkillResult`.

Covers requirement R9.16 at the data-model layer:
- Frozen dataclass with ``output_video_path`` and ``output_subtitle_path``.
- Both fields are :class:`pathlib.Path` instances.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest

from translation_dubbing_skill.models import SkillResult
from translation_dubbing_skill.models import skill_result as sr_module


def test_skill_result_is_frozen_dataclass_with_expected_fields() -> None:
    """SkillResult is a frozen dataclass with the design-mandated fields."""
    assert is_dataclass(SkillResult)
    field_names = [f.name for f in fields(SkillResult)]
    assert field_names == ["output_video_path", "output_subtitle_path"]


def test_skill_result_stores_values() -> None:
    """Constructing with explicit paths preserves them verbatim."""
    video = Path("/tmp/out.mkv")
    subtitle = Path("/tmp/out.zh.srt")
    result = SkillResult(output_video_path=video, output_subtitle_path=subtitle)
    assert result.output_video_path == video
    assert result.output_subtitle_path == subtitle


def test_skill_result_is_immutable() -> None:
    """Frozen dataclass rejects attribute mutation."""
    result = SkillResult(
        output_video_path=Path("/a.mkv"),
        output_subtitle_path=Path("/a.srt"),
    )
    with pytest.raises(FrozenInstanceError):
        result.output_video_path = Path("/b.mkv")  # type: ignore[misc]


def test_skill_result_equality_and_hash() -> None:
    """Value-based equality + hashability from ``frozen=True``."""
    a = SkillResult(Path("/a.mkv"), Path("/a.srt"))
    b = SkillResult(Path("/a.mkv"), Path("/a.srt"))
    c = SkillResult(Path("/c.mkv"), Path("/a.srt"))
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_module_exports_public_api() -> None:
    """Public API is re-exported from the models package."""
    assert set(sr_module.__all__) == {"SkillResult"}
    from translation_dubbing_skill import models as models_pkg

    assert "SkillResult" in models_pkg.__all__
    assert hasattr(models_pkg, "SkillResult")
