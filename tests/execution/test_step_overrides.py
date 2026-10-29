from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.execution import BackendRuntime
from src.execution.episode import EpisodeRequest, run_episode
from src.harness.transcript import HarnessTranscript
from src.task.types import (
    BudgetConstraints,
    Capability,
    EpisodeArtifacts,
    EpisodeConstraints,
    StepConstraints,
    SuccessConstraints,
    TaskPromptSpec,
    TaskReward,
    Variation,
    Workflow,
    WorkflowStep,
)
from src.typing.config import AppConfig

from tests.execution.test_run_episode_outcome import _FakeRecorder, _FakeTracedGenner


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name="claude",
        code_host_cache_path=str(tmp_path / "cache"),
        container_ids=["container-a"],
        train_data_save_folder=str(tmp_path / "train-data"),
        harness={
            "name": "orchestrator_modes",
            "orchestrator_modes": {
                "orchestrator_prompt": "prompt {scratchpad_content}",
            },
        },
        observability={"enabled": False},
    )


def _request() -> EpisodeRequest:
    return EpisodeRequest(
        episode_id="ep-step-overrides",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
    )


class _WorkflowTask:
    agent_service_name = "agent"
    metric_unit = "kb"

    def __init__(self, constraints: EpisodeConstraints) -> None:
        self._constraints = constraints

    def prompt_spec(self, variation, episode_context):
        return TaskPromptSpec(
            system_instruction="sys",
            capabilities=[Capability(name="run_python", description="run")],
        )

    def workflow(self, variation, episode_context):
        return Workflow(
            steps=(
                WorkflowStep(
                    name="plan_cleanup",
                    prompt="plan",
                    inherit_all_capabilities=False,
                    capabilities=(),
                ),
            )
        )

    def episode_constraints(self, variation, episode_context):
        return self._constraints

    def measure_initial_state(self, containers, episode_context, *, private_context=None):
        return {}

    def finalize_episode(
        self,
        containers,
        initial_state,
        episode_context,
        artifacts,
        *,
        private_context=None,
        finalization_context=None,
    ):
        return TaskReward(value=1.0, success=True, breakdown={"reward": "ok"})


def test_finalization_uses_terminating_step_success_constraints(tmp_path: Path) -> None:
    task = _WorkflowTask(
        EpisodeConstraints(
            budgets=BudgetConstraints(max_task_tool_calls=5),
            success=SuccessConstraints(min_task_tool_calls_for_success=1),
            step_overrides={
                "plan_cleanup": StepConstraints(
                    budgets=BudgetConstraints(max_task_tool_calls=0),
                    success=SuccessConstraints(min_task_tool_calls_for_success=0),
                )
            },
        )
    )
    runtime = BackendRuntime(
        config=_make_config(tmp_path),
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    transcript = HarnessTranscript(
        artifacts=EpisodeArtifacts(),
        llm_turns=0,
        termination_reason="done",
        termination_category="success",
        extra={"last_workflow_step": "plan_cleanup"},
    )
    harness = MagicMock(run_workflow=MagicMock(return_value=transcript))

    with (
        patch("src.execution.episode.resolve_event_recorder_class", return_value=_FakeRecorder),
        patch("src.execution.episode.resolve_traced_genner_class", return_value=_FakeTracedGenner),
        patch("src.execution.episode.construct_harness", return_value=harness),
        patch("src.execution.episode.records_to_rows", return_value=[]),
    ):
        outcome = run_episode(runtime, _request())

    assert outcome.tool_calls_count == 0
    assert outcome.success is True
    assert outcome.trajectory["extra"]["last_workflow_step"] == "plan_cleanup"


def test_unknown_step_override_key_fails_before_episode_launch(tmp_path: Path) -> None:
    task = _WorkflowTask(
        EpisodeConstraints(
            step_overrides={
                "missing": StepConstraints(
                    budgets=BudgetConstraints(max_task_tool_calls=0),
                )
            }
        )
    )
    runtime = BackendRuntime(
        config=_make_config(tmp_path),
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    harness = MagicMock()

    with (
        patch("src.execution.episode.resolve_event_recorder_class", return_value=_FakeRecorder),
        patch("src.execution.episode.resolve_traced_genner_class", return_value=_FakeTracedGenner),
        patch("src.execution.episode.construct_harness", return_value=harness),
        pytest.raises(ValueError, match="unknown step_overrides"),
    ):
        run_episode(runtime, _request())

    harness.run_workflow.assert_not_called()
    harness.run_episode.assert_not_called()
