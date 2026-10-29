import threading
from pathlib import Path

import pytest

from src.harness.recorder import EventRecorder


@pytest.fixture
def recorder(tmp_path: Path) -> EventRecorder:
    return EventRecorder(
        episode_id="test-ep",
        output_path=tmp_path / "test_events.jsonl",
    )


def test_bump_counter_and_snapshot(recorder: EventRecorder) -> None:
    recorder.bump_counter("tool_calls")
    recorder.bump_counter("tool_calls")
    recorder.bump_counter("tool_calls")

    assert recorder.snapshot_counters() == {"tool_calls": 3}


def test_set_label_and_snapshot(recorder: EventRecorder) -> None:
    recorder.set_label("last_tool", "analyzer")

    assert recorder.snapshot_labels()["last_tool"] == "analyzer"


def test_snapshot_counters_is_copy(recorder: EventRecorder) -> None:
    recorder.bump_counter("x")

    snap = recorder.snapshot_counters()
    snap["x"] = 999

    assert recorder.snapshot_counters()["x"] == 1


def test_snapshot_labels_is_copy(recorder: EventRecorder) -> None:
    recorder.set_label("k", "v")

    snap = recorder.snapshot_labels()
    snap["k"] = "mutated"

    assert recorder.snapshot_labels()["k"] == "v"


def test_bump_counter_thread_safe(recorder: EventRecorder) -> None:
    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait()
        for _ in range(100):
            recorder.bump_counter("c")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert recorder.snapshot_counters()["c"] == 1000


def test_snapshot_counters_empty_by_default(recorder: EventRecorder) -> None:
    assert recorder.snapshot_counters() == {}


def test_snapshot_labels_empty_by_default(recorder: EventRecorder) -> None:
    assert recorder.snapshot_labels() == {}
