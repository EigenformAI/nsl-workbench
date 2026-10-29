"""Scrape ``[nsl-event]`` lines from container stderr and replay them
through the recorder so native-workflow profiles surface the same
``workflow_step_enter`` / ``workflow_step_exit`` shapes as the host-side
``WorkflowDriver``.
"""

from __future__ import annotations

import threading
from pathlib import Path

from src.harness.container import _replay_nsl_events, _scrape_nsl_events
from src.harness.recorder import EventRecorder


def _recorder(tmp_path: Path) -> EventRecorder:
    return EventRecorder(
        episode_id="ep-1",
        output_path=tmp_path / "events.jsonl",
        cancel_event=threading.Event(),
    )


def test_scrape_extracts_workflow_events_in_order() -> None:
    log = (
        "noise before\n"
        "[nsl-event] "
        '{"event": "workflow_step_enter", "step": "explore", '
        '"input_kind": "string", "context_mode": "inherit"}\n'
        "more noise\n"
        "[nsl-event] "
        '{"event": "workflow_step_exit", "step": "explore", '
        '"outcome": "ok", "message_count": 4, "duration_s": 1.2}\n'
        "[nsl-event] "
        '{"event": "workflow_finished", "message_count": 4, '
        '"recovery_fired": false, "steps_visited": ["explore"]}\n'
    )
    events = _scrape_nsl_events(log)
    assert [e["event"] for e in events] == [
        "workflow_step_enter",
        "workflow_step_exit",
        "workflow_finished",
    ]
    assert events[0]["context_mode"] == "inherit"
    assert events[1]["outcome"] == "ok"


def test_scrape_skips_malformed_lines() -> None:
    log = (
        "[nsl-event] not json at all\n"
        "[nsl-event] {\"not an event\": true}\n"
        "[nsl-event] {\"event\": \"workflow_finished\", \"steps_visited\": []}\n"
        "[nsl-event] \n"
    )
    events = _scrape_nsl_events(log)
    assert len(events) == 1
    assert events[0]["event"] == "workflow_finished"


def test_scrape_tolerates_docker_log_prefixes() -> None:
    # docker logs sometimes prepend a timestamp; the matcher is positional
    # within each line so any leading characters are fine.
    log = "2026-05-14T15:09:00Z [nsl-event] {\"event\": \"workflow_step_exit\", \"step\": \"a\"}\n"
    events = _scrape_nsl_events(log)
    assert events == [{"event": "workflow_step_exit", "step": "a"}]


def test_replay_emits_decisions_and_sets_label(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _replay_nsl_events(
        [
            {
                "event": "workflow_step_enter",
                "step": "explore",
                "context_mode": "inherit",
            },
            {
                "event": "workflow_step_exit",
                "step": "explore",
                "outcome": "ok",
                "message_count": 3,
            },
            {
                "event": "workflow_finished",
                "recovery_fired": False,
                "steps_visited": ["explore"],
            },
        ],
        recorder,
    )

    assert recorder.snapshot_labels()["last_workflow_step"] == "explore"
    lines = [
        line for line in (tmp_path / "events.jsonl").read_text().splitlines() if line
    ]
    assert any('"workflow_step_enter"' in line for line in lines)
    assert any('"workflow_step_exit"' in line for line in lines)
    assert any('"workflow_finished"' in line for line in lines)


def test_replay_updates_label_to_last_step(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _replay_nsl_events(
        [
            {"event": "workflow_step_enter", "step": "explore"},
            {"event": "workflow_step_exit", "step": "explore", "outcome": "ok"},
            {"event": "workflow_step_enter", "step": "execute"},
            {
                "event": "workflow_step_exit",
                "step": "execute",
                "outcome": "error",
                "exc_type": "RuntimeError",
            },
        ],
        recorder,
    )
    assert recorder.snapshot_labels()["last_workflow_step"] == "execute"


def test_replay_unknown_event_logs_as_state(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    _replay_nsl_events(
        [{"event": "future_extension", "data": 42}],
        recorder,
    )
    text = (tmp_path / "events.jsonl").read_text()
    assert '"nsl_event"' in text
    assert '"future_extension"' in text
