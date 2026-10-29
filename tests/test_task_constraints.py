"""Contract tests for task-owned episode constraint value objects."""

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from src.task.types import (
    BudgetConstraints,
    EpisodeConstraints,
    NoToolReplyPolicy,
    StepConstraints,
    SuccessConstraints,
)


def test_constraint_graph_is_immutable() -> None:
    budget = BudgetConstraints(
        max_task_tool_calls=10,
        max_task_tool_calls_by_name={"cap": 3},
    )
    policy = NoToolReplyPolicy(retry=True, max_retries=2)
    success = SuccessConstraints()
    step = StepConstraints(success=SuccessConstraints(min_task_tool_calls_for_success=0))
    episode = EpisodeConstraints(step_overrides={"plan_cleanup": step})

    assert isinstance(budget.max_task_tool_calls_by_name, MappingProxyType)
    assert isinstance(episode.step_overrides, MappingProxyType)
    with pytest.raises(FrozenInstanceError):
        budget.max_task_tool_calls = 20  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.retry = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        success.min_task_tool_calls_for_success = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        episode.budgets = BudgetConstraints()  # type: ignore[misc]
    with pytest.raises(TypeError):
        budget.max_task_tool_calls_by_name["cap"] = 4  # type: ignore[index]
    with pytest.raises(TypeError):
        episode.step_overrides["other"] = step  # type: ignore[index]
