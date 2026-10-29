"""``EventRecorder.capability_pairs()`` — profile-facing reconstruction.

Profiles rebuild ``(invocation, result)`` tuples from recorder state
without reaching into event categories. The bridge emits
``mcp_capability_call`` / ``mcp_capability_result`` events with shared
``correlation_id``; this method joins them.

Contract:

- Returns pairs in chronological order.
- Unpaired events raise ``HarnessError`` — an unpaired event is a bridge
  bug, not a tolerated state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.harness.base import HarnessError
from src.harness.recorder import EventRecorder
from src.task.types import CapabilityInvocation, CapabilityResult


def _record_pair(
    recorder: EventRecorder,
    cid: str,
    cap: str,
    *,
    input: dict | None = None,
    output: dict | None = None,
    success: bool = True,
) -> None:
    recorder.log_action(
        category="mcp_capability_call",
        payload={
            "correlation_id": cid,
            "capability": cap,
            "invocation": {"name": cap, "input": input or {}},
        },
    )
    recorder.log_observation(
        category="mcp_capability_result",
        payload={
            "correlation_id": cid,
            "capability": cap,
            "result": {
                "name": cap,
                "output": output or {},
                "success": success,
                "error": None,
            },
        },
    )


def test_pairs_returned_in_chronological_order(tmp_path: Path) -> None:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "e.jsonl")
    _record_pair(recorder, "c1", "analyzer", input={"i": 1})
    _record_pair(recorder, "c2", "exploiter", input={"i": 2})

    pairs = recorder.capability_pairs()
    assert len(pairs) == 2
    assert isinstance(pairs[0][0], CapabilityInvocation)
    assert isinstance(pairs[0][1], CapabilityResult)
    assert pairs[0][0].name == "analyzer"
    assert pairs[0][0].input == {"i": 1}
    assert pairs[1][0].name == "exploiter"


def test_pairs_join_when_observations_interleave(tmp_path: Path) -> None:
    """Concurrent tool calls: two actions logged back-to-back, then two
    observations in completion order. Pairing is by correlation_id, not
    arrival order."""
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "e.jsonl")
    # action A
    recorder.log_action(
        category="mcp_capability_call",
        payload={
            "correlation_id": "A",
            "capability": "analyzer",
            "invocation": {"name": "analyzer", "input": {"k": "a"}},
        },
    )
    # action B
    recorder.log_action(
        category="mcp_capability_call",
        payload={
            "correlation_id": "B",
            "capability": "analyzer",
            "invocation": {"name": "analyzer", "input": {"k": "b"}},
        },
    )
    # observation B finishes first
    recorder.log_observation(
        category="mcp_capability_result",
        payload={
            "correlation_id": "B",
            "capability": "analyzer",
            "result": {
                "name": "analyzer",
                "output": {"v": "b"},
                "success": True,
                "error": None,
            },
        },
    )
    # observation A finishes second
    recorder.log_observation(
        category="mcp_capability_result",
        payload={
            "correlation_id": "A",
            "capability": "analyzer",
            "result": {
                "name": "analyzer",
                "output": {"v": "a"},
                "success": True,
                "error": None,
            },
        },
    )

    pairs = recorder.capability_pairs()
    # Pair invariant: invocation's input matches its paired result's output.
    by_input = {p[0].input.get("k"): p[1].output.get("v") for p in pairs}
    assert by_input == {"a": "a", "b": "b"}


def test_unpaired_event_raises_harness_error(tmp_path: Path) -> None:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "e.jsonl")
    recorder.log_action(
        category="mcp_capability_call",
        payload={
            "correlation_id": "orphan",
            "capability": "analyzer",
            "invocation": {"name": "analyzer", "input": {}},
        },
    )
    # No paired observation — this is a bridge bug.
    with pytest.raises(HarnessError):
        recorder.capability_pairs()


def test_empty_recorder_returns_empty_list(tmp_path: Path) -> None:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "e.jsonl")
    assert recorder.capability_pairs() == []
