"""Coverage for the two non-trivial termination branches in
OrchestratorModeHarness: context overflow and repetition collapse.

Both branches were previously uncovered — a behavioral regression in
either would not have been caught by existing tests.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from result import Err, Ok

from src.genner.Base import (
    CONTEXT_OVERFLOW_PREFIX,
    INFERENCE_TIMEOUT_PREFIX,
    InferenceTimeoutError,
)
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad
from src.harness.orchestrator_modes import OrchestratorModeHarness
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import Capability, TaskPromptSpec, Variation


class _StubTask:
    metric_name = "dummy"
    metric_unit = ""
    name = "stub"
    description = "stub"

    def parse_response(self, raw_response, *, invoked_capability=None):
        return []

    def execute_capability(self, invocation, containers, variation, ctx):
        from src.task.types import CapabilityResult

        return CapabilityResult(name=invocation.name, output={}, success=True)


def _make_ctx(
    tmp_path: Path,
    inner,
    *,
    repetition_config: dict | None = None,
) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep", output_path=tmp_path / "events.jsonl")
    traced = TracedGenner(
        inner=inner,
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep",
    )
    # Modes are owned by config, capabilities are real MCP tools.
    settings = {
        "max_harness_iterations": 5,
        "scratchpad_max_chars": 5000,
        "tool_output_max_chars": 0,
        "orchestrator_prompt": "{scratchpad_content}\n{budget_remaining}/{total_budget}",
        "modes": {
            "explorer": {
                "prompt": "{instruction}\n{scratchpad_content}",
                "timeout_s": 30,
                "writes_scratchpad": True,
                "scratchpad_label": "Results",
            }
        },
    }
    if repetition_config is not None:
        settings["repetition_guard"] = repetition_config

    return HarnessContext(
        episode_id="ep",
        genner=traced,
        task=_StubTask(),  # type: ignore[arg-type]
        variation=Variation(name="v", description="d"),
        prompt_spec=TaskPromptSpec(
            system_instruction="sys",
            capabilities=[Capability(name="run_python", description="exec")],
        ),
        episode_context={},
        containers=[],
        agent_container=None,  # type: ignore[arg-type]
        host_cache_folder=tmp_path,
        config=HarnessConfigView(
            harness_settings=settings,
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,  # type: ignore[arg-type]
        recorder=recorder,
        cancel_event=threading.Event(),
        harness_session={"scratchpad": CrossEpisodeScratchpad(max_chars=5000)},
    )


def test_context_overflow_terminates_with_correct_category(tmp_path: Path) -> None:
    inner = MagicMock()
    # First orchestrator call returns a context-overflow error.
    inner.plist_completion.return_value = Err(
        f"{CONTEXT_OVERFLOW_PREFIX} prompt exceeded limit"
    )
    inner.collector = None
    ctx = _make_ctx(tmp_path, inner)

    transcript = OrchestratorModeHarness({}).run_episode(ctx=ctx)
    assert transcript.termination_category == "context_overflow"
    assert "exceeded" in transcript.termination_reason

    warning_cats = {e.category for e in ctx.recorder.events if e.kind == "warning"}
    assert "context_overflow" in warning_cats


def test_inference_timeout_propagates_out_of_orchestrator_harness(
    tmp_path: Path,
) -> None:
    inner = MagicMock()
    inner.plist_completion.return_value = Err(
        f"{INFERENCE_TIMEOUT_PREFIX} APITimeoutError: request timed out"
    )
    inner.collector = None
    ctx = _make_ctx(tmp_path, inner)

    with pytest.raises(InferenceTimeoutError, match="request timed out"):
        OrchestratorModeHarness({}).run_episode(ctx=ctx)


def test_inference_timeout_propagates_out_of_delegated_turn(tmp_path: Path) -> None:
    inner = MagicMock()
    orchestrator_result = MagicMock(
        content="MODE: explorer\nINSTRUCTION: inspect\nREASONING: test",
        usage=None,
    )
    inner.plist_completion.side_effect = [
        Ok(orchestrator_result),
        Err(f"{INFERENCE_TIMEOUT_PREFIX} delegated request timed out"),
    ]
    inner.collector = None
    ctx = _make_ctx(tmp_path, inner)

    with pytest.raises(InferenceTimeoutError, match="delegated request timed out"):
        OrchestratorModeHarness({}).run_episode(ctx=ctx)


def test_repetition_collapse_terminates_with_correct_category(
    tmp_path: Path,
) -> None:
    # Orchestrator emits a response with repeated multi-line paragraphs —
    # the detector operates on double-newline-separated blocks.
    para = (
        "This is a long-enough paragraph that the detector treats as a unit "
        "for near-duplicate comparison against its neighbors in the window."
    )
    repeated = "MODE: explorer\nINSTRUCTION: go\nREASONING: tick\n\n" + "\n\n".join(
        [para] * 8
    )
    inner = MagicMock()

    def _complete(messages):
        phase = next(
            (m.get("meta", {}).get("phase") for m in messages if m.get("meta")),
            "orchestrator",
        )
        result = MagicMock()
        result.content = repeated if phase == "orchestrator" else "Results: ok"
        result.usage = None
        return Ok(result)

    inner.plist_completion.side_effect = _complete
    inner.collector = None

    # Aggressive detector config: small threshold and short window so
    # repetition is detected deterministically.
    ctx = _make_ctx(
        tmp_path,
        inner,
        repetition_config={
            "min_paragraphs": 3,
            "similarity_threshold": 0.80,
            "min_paragraph_chars": 50,
            "window_size": 4,
            "first_hit_action": "warn_only",
            "second_hit_action": "end_episode",
        },
    )

    transcript = OrchestratorModeHarness({}).run_episode(ctx=ctx)
    assert transcript.termination_category == "repetition_collapse"
    warning_cats = {e.category for e in ctx.recorder.events if e.kind == "warning"}
    assert "repetition_collapse" in warning_cats
