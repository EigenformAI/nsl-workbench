from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.execution import BackendRuntime
from src.execution.episode import EpisodeRequest, run_episode
from src.harness.transcript import HarnessTranscript
from src.task.types import (
    EpisodeArtifacts,
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


def test_run_episode_dispatches_workflow_when_task_declares_one(tmp_path: Path) -> None:
    workflow = Workflow(steps=(WorkflowStep(name="solve", prompt="solve"),))

    class _Task:
        agent_service_name = "agent"
        metric_unit = "kb"

        def prompt_spec(self, variation, episode_context):
            return TaskPromptSpec(system_instruction="sys", capabilities=[])

        def workflow(self, variation, episode_context):
            return workflow

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
            return TaskReward(value=0.0, success=False)

    task = _Task()
    runtime = BackendRuntime(
        config=_make_config(tmp_path),
        run_id="run-123",
        task=task,
        genner=MagicMock(collector=None),
        docker_client=MagicMock(),
        metrics=None,
    )
    request = EpisodeRequest(
        episode_id="ep-workflow",
        containers=[MagicMock(id="container-a")],
        container_manager=MagicMock(),
        agent_container=MagicMock(id="agent-container"),
        variation=Variation(name="var-1", description="variation"),
        episode_context={},
    )
    transcript = HarnessTranscript(
        artifacts=EpisodeArtifacts(),
        llm_turns=1,
        termination_reason="workflow done",
        termination_category="success",
        extra={},
    )
    harness = MagicMock()
    harness.run_workflow.return_value = transcript

    with (
        patch("src.execution.episode.resolve_event_recorder_class", return_value=_FakeRecorder),
        patch("src.execution.episode.resolve_traced_genner_class", return_value=_FakeTracedGenner),
        patch("src.execution.episode.construct_harness", return_value=harness),
        patch("src.execution.episode.records_to_rows", return_value=[]),
    ):
        run_episode(runtime, request)

    harness.run_workflow.assert_called_once()
    assert harness.run_workflow.call_args.args[0] is workflow
    harness.run_episode.assert_not_called()
