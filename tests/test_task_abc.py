"""Tests for TaskSpec ABC enforcement and behavior."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from src.task.base import EpisodeClassification, TaskSpec
from src.task.types import (
    Capability,
    CapabilityExecutionContext,
    CapabilityInvocation,
    EpisodeArtifacts,
    PopulationOutcome,
    PopulationResult,
    TaskPromptSpec,
    TaskReward,
    Variation,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal complete stub for testing
# ---------------------------------------------------------------------------


@dataclass
class StubState:
    value: float = 0.0


def _stub_prompt_spec() -> TaskPromptSpec:
    return TaskPromptSpec(
        system_instruction="Test system prompt",
        capabilities=[
            Capability(name="investigator", description="Run diagnostics"),
        ],
    )


class StubTask(TaskSpec["StubState"]):
    name = "stub-task"
    description = "A stub task for testing"
    metric_name = "stub_metric"
    metric_unit = "units"
    higher_is_better = True
    docker_compose_dir = "docker/stub"
    agent_service_name = "agent"

    def __init__(self, task_config: dict[str, Any]):
        pass

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
        return _stub_prompt_spec()

    def measure_initial_state(
        self, containers, episode_context, *, private_context=None
    ) -> StubState:
        return StubState(1.0)

    def compute_reward(self, initial, final, artifacts) -> TaskReward:
        return TaskReward(
            value=final.value - initial.value,
            success=True,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultMethods:
    def test_optional_task_defaults(self):
        task = StubTask({})
        task.reset([])
        assert task.verify_population([], Variation("v1", "test"), {}) is True
        assert task.parse_response("some response") == []
        assert (
            task.parse_response("some response", invoked_capability="investigator")
            == []
        )
        assert task.workflow(Variation("v1", "test"), {}) is None

    def test_execute_capability_default_echoes_input(self):
        task = StubTask({})
        inv = CapabilityInvocation(name="explorer", input={"k": "v"})
        result = task.execute_capability(
            inv,
            [],
            Variation("v1", "test"),
            CapabilityExecutionContext("ep", None, {}),
        )
        assert result.name == "explorer"
        assert result.success is True
        assert result.output == {"k": "v"}

    def test_measure_final_state_delegates_to_initial(self):
        task = StubTask({})
        artifacts = EpisodeArtifacts()
        state = task.measure_final_state([], {}, artifacts)
        assert isinstance(state, StubState)
        assert state.value == 1.0

    def test_finalize_episode_calls_measure_and_compute(self):
        task = StubTask({})
        initial = StubState(1.0)
        artifacts = EpisodeArtifacts()
        reward = task.finalize_episode([], initial, {}, artifacts)
        assert isinstance(reward, TaskReward)
        assert reward.value == 0.0
        assert reward.success is True

    def test_saturation_defaults_are_noop(self):
        task = StubTask({})

        assert task.saturation_enabled() is False
        assert task.classify_episode(
            SimpleNamespace(
                task_breakdown={"bootstrap_active": True, "admitted": True}
            )
        ) == EpisodeClassification()
        assert task.training_data_observability([]) == {}
        task.on_generation_rollover("test", None)
        assert task.generation_checkpoint_state() is None
        task.load_generation_checkpoint_state({"anything": "ignored"})


class TestPrivateContext:
    def test_verify_population_accepts_private_context(self):
        task = StubTask({})
        result = task.verify_population(
            [],
            Variation("v1", "test"),
            {},
            private_context={"secret": "x"},
        )
        assert result is True

    def test_finalize_episode_threads_private_context_to_measure_final(self):
        received = {}

        class SpyTask(StubTask):
            def measure_final_state(
                self,
                containers,
                episode_context,
                artifacts,
                *,
                private_context=None,
            ):
                received["private_context"] = private_context
                return StubState(2.0)

        task = SpyTask({})
        ctx = {"admin_key": "secret123"}
        task.finalize_episode(
            [],
            StubState(1.0),
            {},
            EpisodeArtifacts(),
            private_context=ctx,
        )
        assert received["private_context"] is ctx


class TestValidation:
    def test_validate_rejects_zero_variations(self):
        class ZeroVarTask(StubTask):
            def list_variations(self):
                return []

        task = ZeroVarTask({})
        with pytest.raises(ValueError, match="zero variations"):
            task.validate()

    def test_validate_rejects_zero_capabilities(self):
        class NoCapsTask(StubTask):
            def prompt_spec(self, variation, episode_context):
                return TaskPromptSpec(
                    system_instruction="test",
                    capabilities=[],
                )

        task = NoCapsTask({})
        with pytest.raises(ValueError, match="zero capabilities"):
            task.validate()

    def test_validate_rejects_duplicate_capability_names(self):
        class DupTask(StubTask):
            def prompt_spec(self, variation, episode_context):
                return TaskPromptSpec(
                    system_instruction="test",
                    capabilities=[
                        Capability(name="explorer", description="a"),
                        Capability(name="explorer", description="b"),
                    ],
                )

        task = DupTask({})
        with pytest.raises(ValueError, match="duplicate capability names"):
            task.validate()

    def test_validate_rejects_reserved_capability_name(self):
        class ReservedTask(StubTask):
            def prompt_spec(self, variation, episode_context):
                return TaskPromptSpec(
                    system_instruction="test",
                    capabilities=[
                        Capability(name="orchestrator", description="reserved"),
                    ],
                )

        task = ReservedTask({})
        with pytest.raises(ValueError, match="framework-reserved"):
            task.validate()
