"""TracedGenner honors cancel_event before issuing inference calls."""

from __future__ import annotations

import threading
from pathlib import Path

from result import Err, Ok

from src.genner.Base import Genner
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.observability.types import InferenceResult


class _StubGenner(Genner):
    def __init__(self) -> None:
        super().__init__(identifier="stub")
        self.call_count = 0

    def plist_completion(self, messages):
        self.call_count += 1
        return Ok(InferenceResult(content="resp", usage=None))

    @staticmethod
    def get_usage_info(response):  # type: ignore[override]
        return None


def test_cancel_event_short_circuits_call(tmp_path: Path) -> None:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    inner = _StubGenner()
    cancel = threading.Event()
    cancel.set()

    traced = TracedGenner(
        inner=inner,
        recorder=recorder,
        cancel_event=cancel,
        episode_id="ep-1",
    )

    result = traced.plist_completion([{"role": "user", "content": "x"}], phase="orchestrator")
    assert isinstance(result, Err)
    assert inner.call_count == 0
    # One inference record emitted with success=False for the cancelled call.
    assert len(recorder.inference_records) == 1
    assert recorder.inference_records[0].success is False


def test_non_cancelled_call_goes_through(tmp_path: Path) -> None:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    inner = _StubGenner()
    cancel = threading.Event()

    traced = TracedGenner(
        inner=inner,
        recorder=recorder,
        cancel_event=cancel,
        episode_id="ep-1",
    )
    result = traced.plist_completion([{"role": "user", "content": "x"}], phase="orchestrator")
    assert isinstance(result, Ok)
    assert inner.call_count == 1
    assert recorder.inference_records[0].success is True
    assert recorder.inference_records[0].phase == "orchestrator"
