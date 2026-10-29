from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from src.harness.container import ContainerHarness
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.orchestrator_modes import OrchestratorModeHarness, _EpisodeState
from src.harness.orchestrator_modes.harness import _ActionBudget
from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad
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


def _ctx(tmp_path: Path, *, harness_session: dict | None = None) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    traced = TracedGenner(
        inner=MagicMock(),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    return HarnessContext(
        episode_id="ep-1",
        genner=traced,
        task=_StubTask(),  # type: ignore[arg-type]
        variation=Variation(name="v", description="d"),
        prompt_spec=TaskPromptSpec(
            system_instruction="sys",
            capabilities=[Capability(name="analyzer", description="read")],
        ),
        episode_context={},
        containers=[],
        agent_container=None,  # type: ignore[arg-type]
        host_cache_folder=tmp_path,
        config=HarnessConfigView(
            harness_settings={
                "orchestrator_prompt": "prompt {scratchpad_content}",
                "scratchpad_max_chars": 5000,
            },
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,
        recorder=recorder,
        cancel_event=threading.Event(),
        docker_client=MagicMock(),
        harness_session=harness_session or {},
    )


def test_orchestrator_telemetry_reports_step_and_budget(tmp_path: Path) -> None:
    harness = OrchestratorModeHarness({})
    harness._episode_state = _EpisodeState(episode_id="ep-1", current_step=3)
    harness._episode_budget = _ActionBudget(total=5)
    harness._episode_budget.remaining = 2

    assert harness.telemetry() == {"step": "3", "budget_left": "2"}
    assert harness.telemetry_columns() == ["step", "budget_left"]


def test_orchestrator_failure_extras_include_scratchpad_and_mode_history(
    tmp_path: Path,
) -> None:
    scratchpad = CrossEpisodeScratchpad(max_chars=5000)
    scratchpad.append("remember this", episode_id="ep-1")
    harness = OrchestratorModeHarness({})
    harness._ctx = _ctx(tmp_path, harness_session={"scratchpad": scratchpad})
    harness._episode_state = _EpisodeState(
        episode_id="ep-1",
        mode_history=[{"mode": "planner"}],
    )

    extras = harness.failure_extras()

    assert "remember this" in extras["scratchpad_final"]
    assert extras["mode_history"] == [{"mode": "planner"}]


def test_orchestrator_failure_extras_tolerate_missing_scratchpad(
    tmp_path: Path,
) -> None:
    harness = OrchestratorModeHarness({})
    harness._ctx = _ctx(tmp_path, harness_session={})
    harness._episode_state = _EpisodeState(
        episode_id="ep-1",
        mode_history=[{"mode": "planner"}],
    )

    assert harness.failure_extras() == {
        "scratchpad_final": "",
        "mode_history": [{"mode": "planner"}],
    }


def _container_harness() -> ContainerHarness:
    return ContainerHarness(
        harness_config={
            "profile": "ms_agent",
            "profile_config": {"model": "test-model"},
            "image": "img",
            "max_wall_seconds": 5,
        }
    )


def test_container_telemetry_and_failure_extras_empty_before_run() -> None:
    harness = _container_harness()

    assert harness.telemetry() == {}
    assert harness.failure_extras() == {}


def test_container_telemetry_merges_counters_and_labels(tmp_path: Path) -> None:
    harness = _container_harness()
    harness._ctx = _ctx(tmp_path)
    harness._ctx.recorder.bump_counter("turns")
    harness._ctx.recorder.bump_counter("tool_calls")
    harness._ctx.recorder.bump_counter("tool_calling_no_calls")
    harness._ctx.recorder.set_label("last_tool", "analyzer")

    assert harness.telemetry() == {
        "turns": "1",
        "last_tool": "analyzer",
        "tool_calling_no_calls": "1",
    }
    assert harness.failure_extras() == {"telemetry": harness.telemetry()}
