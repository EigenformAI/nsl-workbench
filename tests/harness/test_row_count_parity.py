"""Inference-record count parity vs legacy prompt_responses.

The orchestrator-modes harness emits exactly 2 inference records per step
(orchestrator + delegated capability turn) — matching the legacy prompt-
pair emission rate. Verified on a budget-exhausted run with mocked Genner.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from result import Ok

from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad
from src.harness.orchestrator_modes import OrchestratorModeHarness
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.harness.training_row_adapter import records_to_rows
from src.task.types import (
    Capability,
    TaskPromptSpec,
    Variation,
)


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


def _make_inner_genner():
    inner = MagicMock()

    def _complete(messages):
        # Look at phase meta to decide what to echo back.
        phase = next(
            (m.get("meta", {}).get("phase") for m in messages if m.get("meta")),
            "orchestrator",
        )
        if phase == "orchestrator":
            content = "MODE: explorer\nINSTRUCTION: inspect\nREASONING: tick"
        else:
            content = "Results: nothing"
        result = MagicMock()
        result.content = content
        result.usage = None
        return Ok(result)

    inner.plist_completion.side_effect = _complete
    inner.collector = None
    return inner


def test_budget_of_n_yields_2n_inference_records(tmp_path: Path) -> None:
    budget = 3
    recorder = EventRecorder(
        episode_id="ep-parity", output_path=tmp_path / "events.jsonl"
    )
    inner = _make_inner_genner()
    traced = TracedGenner(
        inner=inner,
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-parity",
    )

    task = _StubTask()
    variation = Variation(name="v1", description="t")
    prompt_spec = TaskPromptSpec(
        system_instruction="sys",
        capabilities=[Capability(name="run_python", description="exec")],
    )

    # Modes are a harness concern — declared in config, not on capabilities.
    config_view = HarnessConfigView(
        harness_settings={
            "max_harness_iterations": budget,
            "scratchpad_max_chars": 1000,
            "tool_output_max_chars": 0,
            "orchestrator_prompt": (
                "{scratchpad_content}\n{budget_remaining}/{total_budget}"
            ),
            "modes": {
                "explorer": {
                    "prompt": "{instruction}\n{scratchpad_content}",
                    "timeout_s": 30,
                    "writes_scratchpad": True,
                    "scratchpad_label": "Results",
                }
            },
        },
        model_name="test",
        train_data_save_folder=str(tmp_path),
        code_host_cache_path=str(tmp_path),
    )

    ctx = HarnessContext(
        episode_id="ep-parity",
        genner=traced,
        task=task,  # type: ignore[arg-type]
        variation=variation,
        prompt_spec=prompt_spec,
        episode_context={},
        containers=[],
        agent_container=None,  # type: ignore[arg-type]
        host_cache_folder=tmp_path,
        config=config_view,
        metrics=None,  # type: ignore[arg-type]
        recorder=recorder,
        cancel_event=threading.Event(),
        harness_session={"scratchpad": CrossEpisodeScratchpad(max_chars=1000)},
    )

    harness = OrchestratorModeHarness(harness_config={})
    transcript = harness.run_episode(ctx=ctx)

    # Exactly 2 inference records per step × budget steps.
    assert len(recorder.inference_records) == 2 * budget
    phases = [r.phase for r in recorder.inference_records]
    assert phases.count("orchestrator") == budget
    assert phases.count("explorer") == budget

    # Transcript reports the step count correctly.
    assert transcript.extra["harness_iterations_count"] == budget
    assert transcript.termination_category == "budget_exhausted"

    # Adapter parity: one training row per inference record, same
    # phase-to-interaction_type mapping.
    rows = records_to_rows(recorder.inference_records, run_id="rp", version="v")
    assert len(rows) == len(recorder.inference_records) == 2 * budget
    row_interaction_types = [r["interaction_type"] for r in rows]
    assert row_interaction_types == phases
    # Legacy prompt_responses shape that generate_training_data expects is
    # derived 1:1 from rows.
    legacy_shape = [
        {
            "prompt": r["prompt"],
            "raw_response": r["raw_response"],
            "interaction_type": r["interaction_type"],
            "success": r["success"],
        }
        for r in rows
    ]
    assert len(legacy_shape) == 2 * budget
