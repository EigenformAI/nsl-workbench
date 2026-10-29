from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from result import Ok

from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.orchestrator_modes import OrchestratorModeHarness
from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import (
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    TaskPromptSpec,
    Variation,
)


class _Task:
    metric_name = "metric"
    metric_unit = ""
    name = "stub"
    description = "stub"

    def __init__(self, *, parsed: list[CapabilityInvocation] | None = None) -> None:
        self.calls: list[CapabilityInvocation] = []
        self._parsed = parsed or []

    def parse_response(self, raw_response, *, invoked_capability=None):
        return list(self._parsed)

    def execute_capability(self, invocation, containers, variation, ctx):
        self.calls.append(invocation)
        if invocation.name == "run_python":
            return CapabilityResult(
                name=invocation.name,
                output={"stdout": "ran", "stderr": "", "return_code": 0},
                success=True,
            )
        return CapabilityResult(name=invocation.name, output=invocation.input)


def _make_inner(response_by_phase: dict[str, str]) -> MagicMock:
    inner = MagicMock()

    def _complete(messages):
        phase = next(
            (m.get("meta", {}).get("phase") for m in messages if m.get("meta")),
            "orchestrator",
        )
        result = MagicMock()
        result.content = response_by_phase.get(phase, "Results: ok")
        result.usage = None
        return Ok(result)

    inner.plist_completion.side_effect = _complete
    inner.collector = None
    return inner


def _ctx(
    tmp_path: Path,
    *,
    task: _Task,
    response_by_phase: dict[str, str],
    modes: dict,
) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    traced = TracedGenner(
        inner=_make_inner(response_by_phase),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    return HarnessContext(
        episode_id="ep-1",
        genner=traced,
        task=task,  # type: ignore[arg-type]
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
            harness_settings={
                "max_harness_iterations": 1,
                "scratchpad_max_chars": 5000,
                "tool_output_max_chars": 0,
                "orchestrator_prompt": "{scratchpad_content}\n{budget_remaining}/{total_budget}",
                "modes": modes,
            },
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,  # type: ignore[arg-type]
        recorder=recorder,
        cancel_event=threading.Event(),
        harness_session={"scratchpad": CrossEpisodeScratchpad(max_chars=5000)},
    )


def test_code_exec_mode_dispatches_through_bridge_events(tmp_path: Path) -> None:
    task = _Task()
    ctx = _ctx(
        tmp_path,
        task=task,
        response_by_phase={
            "orchestrator": "MODE: analyzer\nINSTRUCTION: go\nREASONING: x",
            "analyzer": "Findings: ok\n```python\nprint('hi')\n```",
        },
        modes={
            "analyzer": {
                "prompt": "{instruction}\n{scratchpad_content}",
                "runs_code": True,
                "code_capability": "run_python",
                "writes_scratchpad": True,
                "scratchpad_label": "Findings",
            }
        },
    )

    OrchestratorModeHarness({}).run_episode(ctx=ctx)

    pairs = ctx.recorder.capability_pairs()
    assert len(pairs) == 1
    assert pairs[0][0].name == "run_python"
    assert pairs[0][1].output["stdout"] == "ran"
    correlated = [
        e
        for e in ctx.recorder.events
        if e.category in {"mcp_capability_call", "mcp_capability_result"}
    ]
    assert [e.kind for e in correlated] == ["action", "observation"]
    assert correlated[0].payload["correlation_id"] == correlated[1].payload[
        "correlation_id"
    ]


def test_parse_response_invocation_dispatches_through_bridge_events(
    tmp_path: Path,
) -> None:
    parsed = [CapabilityInvocation(name="metric_report", input={"metric": 3.0})]
    task = _Task(parsed=parsed)
    ctx = _ctx(
        tmp_path,
        task=task,
        response_by_phase={
            "orchestrator": "MODE: explorer\nINSTRUCTION: go\nREASONING: x",
            "explorer": "Results: metric=3",
        },
        modes={
            "explorer": {
                "prompt": "{instruction}\n{scratchpad_content}",
                "writes_scratchpad": True,
                "scratchpad_label": "Results",
                "publishes_metric": True,
            }
        },
    )

    OrchestratorModeHarness({}).run_episode(ctx=ctx)

    pairs = ctx.recorder.capability_pairs()
    assert len(pairs) == 1
    assert pairs[0][0].name == "metric_report"
    assert pairs[0][0].input == {"metric": 3.0}
    assert pairs[0][1].output == {"metric": 3.0}
