from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.task.base import TaskSpec
from src.task.types import (
    Capability,
    PopulationOutcome,
    TaskPromptSpec,
    TaskReward,
    Variation,
)


class DummyTask(TaskSpec[dict[str, str]]):
    name = "dummy"
    description = "dummy"
    metric_name = "score"
    metric_unit = "points"
    higher_is_better = True
    agent_service_name = "does-not-exist"

    def __init__(self, compose_dir: Path) -> None:
        self._compose_dir = compose_dir

    @property
    def docker_compose_dir(self) -> str:
        return str(self._compose_dir)

    def list_variations(self) -> list[Variation]:
        return [Variation(name="v1", description="variation")]

    def populate(
        self,
        containers,
        variation,
    ) -> PopulationOutcome:
        return PopulationOutcome(results=[])

    def prompt_spec(
        self,
        variation: Variation,
        episode_context: dict[str, object],
    ) -> TaskPromptSpec:
        return TaskPromptSpec(
            system_instruction="system",
            capabilities=[
                Capability(
                    name="explorer",
                    description="test",
                    annotations={
                        "orchestrator_modes": {
                            "writes_scratchpad": True,
                            "scratchpad_label": "Results",
                        }
                    },
                )
            ],
        )

    def measure_initial_state(
        self,
        containers,
        episode_context,
        *,
        private_context=None,
    ) -> dict[str, str]:
        return {}

    def compute_reward(self, initial, final, artifacts) -> TaskReward:
        return TaskReward(value=0.0, success=False, breakdown={})


def test_validate_rejects_agent_service_missing_from_compose(tmp_path: Path) -> None:
    compose_dir = tmp_path / "compose"
    compose_dir.mkdir()
    (compose_dir / "docker-compose.yml").write_text(
        "services:\n  service-a:\n    image: alpine\n"
    )
    task = DummyTask(compose_dir)

    with patch(
        "subprocess.run",
        return_value=SimpleNamespace(stdout="service-a\n", stderr="", returncode=0),
    ):
        with pytest.raises(ValueError, match="does-not-exist"):
            task.validate()
