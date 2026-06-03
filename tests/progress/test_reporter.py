"""Unit tests for the :mod:`translation_dubbing_skill.progress.reporter`.

Covers Task 11.1 — R11.1.

The reporter is a thin dispatcher, so these tests focus on:
* ``ProgressReporter.report`` forwards events to its callback in order;
* the callback receives the exact ``ProgressEvent`` instance passed in;
* :class:`InMemoryReporter` accumulates events into ``self.events``.
"""

from __future__ import annotations

from translation_dubbing_skill.models import ProgressEvent
from translation_dubbing_skill.progress import InMemoryReporter, ProgressReporter


def test_progress_reporter_forwards_event_to_callback() -> None:
    captured: list[ProgressEvent] = []

    reporter = ProgressReporter(captured.append)
    event = ProgressEvent(stage="parsing", message="parsing subtitles")

    reporter.report(event)

    assert captured == [event]
    assert captured[0] is event


def test_progress_reporter_preserves_call_order() -> None:
    captured: list[ProgressEvent] = []
    reporter = ProgressReporter(captured.append)

    events = [
        ProgressEvent(stage="parsing", message="parsing"),
        ProgressEvent(stage="translating", message="translating", completed=0, total=3),
        ProgressEvent(stage="translating", message="translating", completed=3, total=3),
        ProgressEvent(stage="muxing", message="muxing"),
        ProgressEvent(stage="done", message="done", extra={"video": "/out.mkv"}),
    ]
    for event in events:
        reporter.report(event)

    assert captured == events


def test_in_memory_reporter_records_all_events() -> None:
    reporter = InMemoryReporter()

    assert reporter.events == []

    e1 = ProgressEvent(stage="parsing", message="a")
    e2 = ProgressEvent(stage="translating", message="b", completed=1, total=2)
    reporter.report(e1)
    reporter.report(e2)

    assert reporter.events == [e1, e2]
