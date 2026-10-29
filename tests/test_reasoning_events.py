from __future__ import annotations

from datetime import datetime
import threading

from result import Ok

from src.harness.recorder import EventRecorder, TrajectoryRecord, _record_to_jsonable
from src.harness.training_row_adapter import record_to_row
from src.harness.traced_genner import TracedGenner
from src.observability.types import InferenceResult


def test_reasoning_is_persisted_in_events_but_excluded_from_sft_rows() -> None:
    record = TrajectoryRecord(
        episode_id="ep-0",
        phase="code",
        messages=[{"role": "user", "content": "question"}],
        response="public answer",
        usage=None,
        timestamp=datetime.now().isoformat(),
        success=True,
        reasoning_content="private chain of thought",
    )

    event = _record_to_jsonable(record)
    row = record_to_row(record, run_id="run", version="v1")

    assert event["reasoning_content"] == "private chain of thought"
    assert row["raw_response"] == "public answer"
    assert "private chain of thought" not in str(row)


def test_traced_genner_keeps_reasoning_in_replay_but_not_recorded_response(
    tmp_path,
) -> None:
    class Inner:
        def plist_completion(self, _messages, **_kwargs):
            return Ok(
                InferenceResult(
                    content="<think>reasoning</think>answer",
                    reasoning_content="reasoning",
                )
            )

    recorder = EventRecorder("ep-0", tmp_path / "events.jsonl")
    genner = TracedGenner(
        Inner(), recorder, threading.Event(), episode_id="ep-0"
    )

    result = genner.plist_completion(
        [{"role": "user", "content": "question"}], phase="code"
    ).unwrap()

    assert result.content == "<think>reasoning</think>answer"
    assert recorder.inference_records[0].response == "answer"
    assert recorder.inference_records[0].reasoning_content == "reasoning"
