"""Stub task implementations for loader / ABC / harness tests."""

from dataclasses import dataclass
from typing import Any

from src.task.base import TaskSpec
from src.task.types import (
    Capability,
    PopulationOutcome,
    PopulationResult,
    TaskPromptSpec,
    TaskReward,
    Variation,
)


@dataclass
class StubState:
    value: float = 0.0


class ValidStubTask(TaskSpec[StubState]):
    """A fully valid task for loader tests."""

    name = "valid-stub"
    description = "A valid stub for testing the loader"
    metric_name = "stub_metric"
    metric_unit = "units"
    higher_is_better = True
    docker_compose_dir = "docker/stub"
    agent_service_name = "agent"

    def __init__(self, task_config: dict[str, Any]):
        self._custom_key = task_config.get("custom_key", "default")

    def list_variations(self) -> list[Variation]:
        return [Variation(name="v1", description="test variation")]

    def populate(self, containers, variation) -> PopulationOutcome:
        return PopulationOutcome(
            results=[
                PopulationResult(
                    container_id="c1",
                    variation_name=variation.name,
                    description="ok",
                    success=True,
                )
            ]
        )

    def prompt_spec(self, variation, episode_context) -> TaskPromptSpec:
        # Phase 2: prompts live in harness config, not on TaskPromptSpec.
        # Stub declares one capability opted into the scratchpad via
        # the orchestrator_modes annotation so OrchestratorModeHarness
        # can run against it without failing the "no scratchpad writer"
        # guard.
        return TaskPromptSpec(
            system_instruction="Test system prompt",
            capabilities=[
                Capability(
                    name="investigator",
                    description="Run diagnostics",
                    runs_code=True,
                    annotations={
                        "orchestrator_modes": {
                            "writes_scratchpad": True,
                            "scratchpad_label": "Findings",
                        },
                    },
                ),
            ],
        )

    def measure_initial_state(
        self, containers, episode_context, *, private_context=None
    ) -> StubState:
        return StubState(1.0)

    def compute_reward(self, initial, final, artifacts) -> TaskReward:
        return TaskReward(value=final.value - initial.value, success=True)


class NotATask:
    """Not a TaskSpec subclass."""

    pass


class IncompleteTask(TaskSpec[StubState]):
    """Missing abstract method implementations — will fail at instantiation."""

    name = "incomplete"
    description = "Incomplete task"
    metric_name = "m"
    metric_unit = "u"
    higher_is_better = True
    docker_compose_dir = "docker/stub"
    agent_service_name = "agent"

    def __init__(self, task_config: dict[str, Any]):
        pass

    def list_variations(self) -> list[Variation]:
        return [Variation(name="v1", description="test")]


class ZeroVariationTask(ValidStubTask):
    """Returns zero variations — validate() should reject."""

    def list_variations(self) -> list[Variation]:
        return []


NOT_A_CLASS_CONSTANT = 42
"""A module-level constant — not a class at all."""


# Phase 2: prompt templates migrated from TaskPromptSpec.extras to typed
# ``[harness.orchestrator_modes]`` config. Placeholder-validation stubs
# that previously lived here target a validator that no longer exists;
# the equivalent check now belongs at the AppConfig layer against
# OrchestratorModesConfig. Tracking as follow-up cleanup.
