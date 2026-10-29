"""Truncation observability contract.

Two regressions this guards:

1. The ``tool_output`` observation must be emitted AFTER truncation, with a
   ``truncated`` flag and byte counts that reflect what the agent actually
   saw. Earlier code emitted the event before truncation, which lied about
   both the byte count and whether truncation occurred.

2. If the full-output artifact cannot be written to disk, the truncation
   marker in the agent-visible text MUST NOT claim it was persisted. The
   prior behavior pointed debugging at a nonexistent file.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.harness.orchestrator_modes import OrchestratorModeHarness, _EpisodeState
from src.harness.recorder import EventRecorder


def _make_ctx(tmp_path: Path) -> SimpleNamespace:
    recorder = EventRecorder(
        episode_id="ep-trunc", output_path=tmp_path / "events.jsonl"
    )
    config = SimpleNamespace(train_data_save_folder=str(tmp_path))
    return SimpleNamespace(recorder=recorder, config=config, metrics=None)


def test_truncate_execution_result_tags_truncated_flag(tmp_path: Path) -> None:
    harness = OrchestratorModeHarness({})
    state = _EpisodeState(episode_id="ep-trunc")
    ctx = _make_ctx(tmp_path)

    exec_result = {
        "success": True,
        "stdout": "A" * 500,
        "stderr": "",
        "return_code": 0,
        "executed_code": "",
    }
    out = harness._truncate_execution_result(ctx, state, "explorer", exec_result, 100)
    assert out["truncated"] is True
    assert len(out["stdout"]) < 500

    harness._emit_tool_output_observation(ctx, "explorer", out)
    obs = [e for e in ctx.recorder.events if e.kind == "observation"]
    assert len(obs) == 1
    payload = obs[0].payload
    assert payload["truncated"] is True
    assert payload["stdout_bytes"] == len(out["stdout"])


def test_truncate_does_not_set_flag_when_under_cap(tmp_path: Path) -> None:
    harness = OrchestratorModeHarness({})
    state = _EpisodeState(episode_id="ep-trunc")
    ctx = _make_ctx(tmp_path)

    exec_result = {
        "success": True,
        "stdout": "hello",
        "stderr": "",
        "return_code": 0,
        "executed_code": "",
    }
    out = harness._truncate_execution_result(ctx, state, "explorer", exec_result, 100)
    assert out["truncated"] is False
    assert out["stdout"] == "hello"


def test_truncation_marker_does_not_lie_on_artifact_write_failure(
    tmp_path: Path,
) -> None:
    harness = OrchestratorModeHarness({})
    state = _EpisodeState(episode_id="ep-trunc")
    ctx = _make_ctx(tmp_path)

    # Make train_data_save_folder read-only so the artifact write OSErrors.
    save_dir = tmp_path / "readonly"
    save_dir.mkdir()
    save_dir.chmod(0o500)
    ctx.config = SimpleNamespace(train_data_save_folder=str(save_dir))

    try:
        truncated_text, was_truncated = harness._truncate_tool_output(
            ctx, state, "explorer", "stdout", "X" * 500, 100
        )
    finally:
        save_dir.chmod(0o700)

    assert was_truncated is True
    assert "full output at" not in truncated_text
    assert "not persisted" in truncated_text

    warnings = [e for e in ctx.recorder.events if e.kind == "warning"]
    categories = {e.category for e in warnings}
    assert "tool_output_artifact_failed" in categories
    trunc_events = [e for e in warnings if e.category == "tool_output_truncated"]
    assert len(trunc_events) == 1
    assert trunc_events[0].payload["artifact_written"] is False
    assert "artifact_path" not in trunc_events[0].payload


@pytest.mark.parametrize(
    "max_chars,input_len,expected",
    [(100, 50, False), (100, 100, False), (100, 101, True), (0, 10, False)],
)
def test_truncation_boundary(
    tmp_path: Path, max_chars: int, input_len: int, expected: bool
) -> None:
    harness = OrchestratorModeHarness({})
    state = _EpisodeState(episode_id="ep-trunc")
    ctx = _make_ctx(tmp_path)
    _, truncated = harness._truncate_tool_output(
        ctx, state, "explorer", "stdout", "X" * input_len, max_chars
    )
    assert truncated is expected
