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


def test_constraint_defaults_contract() -> None:
    bc = BudgetConstraints()
    policy = NoToolReplyPolicy()
    success = SuccessConstraints()
    step = StepConstraints()
    ec = EpisodeConstraints()

    assert bc.max_task_tool_calls is None
    assert bc.max_llm_turns is None
    assert bc.max_task_tool_calls_by_name == {}
    assert policy.retry is False
    assert policy.max_retries == 0
    assert policy.retry_instruction is None
    assert success.min_task_tool_calls_for_success == 1
    assert success.terminal_capability_for_success is None
    assert isinstance(step.budgets, BudgetConstraints)
    assert isinstance(step.no_tool_reply, NoToolReplyPolicy)
    assert isinstance(step.success, SuccessConstraints)
    assert isinstance(ec.budgets, BudgetConstraints)
    assert isinstance(ec.no_tool_reply, NoToolReplyPolicy)
    assert isinstance(ec.success, SuccessConstraints)
    assert ec.step_overrides == {}


def test_configured_nested_constraint_graph_contract() -> None:
    step = StepConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=3),
        no_tool_reply=NoToolReplyPolicy(
            retry=True,
            max_retries=2,
            retry_instruction="Call a tool.",
        ),
        success=SuccessConstraints(
            min_task_tool_calls_for_success=0,
            terminal_capability_for_success="report_metric",
        ),
    )
    ec = EpisodeConstraints(
        budgets=BudgetConstraints(
            max_task_tool_calls=30,
            max_task_tool_calls_by_name={"deploy_attack_sol": 8},
            max_llm_turns=40,
        ),
        success=SuccessConstraints(
            min_task_tool_calls_for_success=1,
            terminal_capability_for_success=None,
        ),
        step_overrides={"plan_cleanup": step},
    )

    assert ec.budgets.max_task_tool_calls == 30
    assert ec.budgets.max_task_tool_calls_by_name["deploy_attack_sol"] == 8
    assert ec.budgets.max_llm_turns == 40
    assert ec.success.min_task_tool_calls_for_success == 1
    override = ec.step_overrides["plan_cleanup"]
    assert override.budgets.max_task_tool_calls == 3
    assert override.no_tool_reply.retry is True
    assert override.no_tool_reply.max_retries == 2
    assert override.no_tool_reply.retry_instruction == "Call a tool."
    assert override.success.min_task_tool_calls_for_success == 0
    assert override.success.terminal_capability_for_success == "report_metric"


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
