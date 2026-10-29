"""Tests for the task loader (load_task and placeholder validation)."""

import pytest

from src.task.base import TaskSpec
from src.task.loader import load_task
from src.task.types import (  # noqa: F401
    EpisodeArtifacts,
    PopulationOutcome,
    PopulationResult,
    TaskPromptSpec,
    TaskReward,
    Variation,
)


class TestLoadTask:
    def test_loads_valid_task_by_dotted_path(self):
        """load_task resolves a dotted path and returns a TaskSpec instance."""
        # Use the stub in this test file's companion module
        task = load_task(
            "tests.fixtures.stub_task.ValidStubTask",
            task_config={"custom_key": "custom_value"},
        )
        assert isinstance(task, TaskSpec)
        assert task.name == "valid-stub"

    def test_passes_task_config_to_init(self):
        task = load_task(
            "tests.fixtures.stub_task.ValidStubTask",
            task_config={"custom_key": "hello"},
        )
        assert task._custom_key == "hello"

    def test_applies_empty_config_when_none(self):
        task = load_task(
            "tests.fixtures.stub_task.ValidStubTask",
            task_config=None,
        )
        assert task._custom_key == "default"

    def test_rejects_nonexistent_module(self):
        with pytest.raises(ImportError):
            load_task("nonexistent.module.Task")

    def test_rejects_nonexistent_class(self):
        with pytest.raises(AttributeError):
            load_task("tests.fixtures.stub_task.NonexistentTask")

    def test_rejects_non_taskspec_class(self):
        with pytest.raises(TypeError, match="must be a subclass of TaskSpec"):
            load_task("tests.fixtures.stub_task.NotATask")

    def test_rejects_non_class_attribute(self):
        """A non-class attribute triggers the clear error, not issubclass TypeError."""
        with pytest.raises(TypeError, match="must be a subclass of TaskSpec"):
            load_task("tests.fixtures.stub_task.NOT_A_CLASS_CONSTANT")

    def test_rejects_incomplete_taskspec(self):
        """A TaskSpec with missing abstract methods raises TypeError."""
        with pytest.raises(TypeError):
            load_task("tests.fixtures.stub_task.IncompleteTask")

    def test_calls_validate(self):
        """load_task calls validate(), which catches zero variations."""
        with pytest.raises(ValueError, match="zero variations"):
            load_task("tests.fixtures.stub_task.ZeroVariationTask")


# Phase 2: prompt placeholder validation moved from the task loader to
# the harness-config layer (prompts live on [harness.orchestrator_modes]
# now, not on TaskPromptSpec). Re-introduce as config-layer tests when
# that validation is implemented. The old placeholder-validation tests
# targeted a code path that no longer exists and were removed here.
