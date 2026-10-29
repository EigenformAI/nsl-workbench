from __future__ import annotations

from pathlib import Path

import pytest

from src.task.base import TaskSpec
from src.task.types import (
    Capability,
    EpisodeArtifacts,
    PopulationOutcome,
    TaskPromptSpec,
    TaskReward,
    Variation,
    Workflow,
    WorkflowStep,
)


class _WorkflowTask(TaskSpec[dict[str, object]]):
    name = "workflow-task"
    description = "workflow task"
    metric_name = "score"
    metric_unit = "points"
    higher_is_better = True
    docker_compose_dir = "docker/stub"
    agent_service_name = "agent"

    def __init__(self, task_config: dict[str, object] | None = None) -> None:
        pass

    def list_variations(self) -> list[Variation]:
        return [Variation(name="v1", description="one")]

    def populate(self, containers, variation) -> PopulationOutcome:
        return PopulationOutcome(results=[])

    def prompt_spec(self, variation, episode_context) -> TaskPromptSpec:
        return TaskPromptSpec(
            system_instruction="system",
            capabilities=[Capability(name="run_python", description="run code")],
        )

    def workflow(self, variation, episode_context) -> Workflow | None:
        return Workflow(steps=(WorkflowStep(name="solve", prompt="solve"),))

    def measure_initial_state(
        self, containers, episode_context, *, private_context=None
    ) -> dict[str, object]:
        return {}

    def compute_reward(self, initial, final, artifacts: EpisodeArtifacts) -> TaskReward:
        return TaskReward(value=0.0, success=False)


def _patch_compose(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services:\n  agent:\n    image: alpine\n")
    monkeypatch.setattr("src.task.base.resolve_compose_file", lambda _dir: compose_file)
    monkeypatch.setattr("src.task.base.compose_services", lambda _file: ["agent"])


def test_task_validate_accepts_well_formed_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_compose(monkeypatch, tmp_path)

    _WorkflowTask({}).validate()


def test_task_validate_rejects_unknown_workflow_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_compose(monkeypatch, tmp_path)

    class BadTask(_WorkflowTask):
        def workflow(self, variation, episode_context) -> Workflow | None:
            return Workflow(
                steps=(
                    WorkflowStep(
                        name="solve",
                        prompt="solve",
                        inherit_all_capabilities=False,
                        capabilities=("missing",),
                    ),
                )
            )

    with pytest.raises(ValueError, match="unknown capabilities.*missing"):
        BadTask({}).validate()


def test_task_validate_checks_each_variation_for_dynamic_capabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_compose(monkeypatch, tmp_path)

    class DynamicTask(_WorkflowTask):
        def list_variations(self) -> list[Variation]:
            return [Variation(name="v1", description="one"), Variation("v2", "two")]

        def prompt_spec(self, variation, episode_context) -> TaskPromptSpec:
            cap_name = "run_python" if variation.name == "v1" else "run_shell"
            return TaskPromptSpec(
                system_instruction="system",
                capabilities=[Capability(name=cap_name, description="run")],
            )

        def workflow(self, variation, episode_context) -> Workflow | None:
            return Workflow(
                steps=(
                    WorkflowStep(
                        name="solve",
                        prompt="solve",
                        inherit_all_capabilities=False,
                        capabilities=("run_python",),
                    ),
                )
            )

    with pytest.raises(ValueError, match="variation 'v2'.*run_python"):
        DynamicTask({}).validate()
