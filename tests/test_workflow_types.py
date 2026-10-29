from __future__ import annotations

import pytest

from src.task.types import Workflow, WorkflowStep


def test_workflow_entry_step_explicit_and_implicit() -> None:
    implicit = Workflow(
        steps=(
            WorkflowStep(name="plan", prompt="plan", next_steps=("act",)),
            WorkflowStep(name="act", prompt="act"),
        )
    )
    assert implicit.entry_step.name == "plan"

    explicit = Workflow(
        steps=(
            WorkflowStep(name="recover", prompt="recover"),
            WorkflowStep(name="plan", prompt="plan", is_entry=True),
        )
    )
    assert explicit.entry_step.name == "plan"


def test_workflow_topological_order_detects_isolated_cycle() -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(name="a", prompt="a", next_steps=("b",)),
            WorkflowStep(name="b", prompt="b", next_steps=("a",)),
        )
    )

    with pytest.raises(ValueError, match="cycle"):
        workflow.topological_order()


def test_workflow_validate_accepts_error_only_recovery() -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(name="try", prompt="try", on_error="recover"),
            WorkflowStep(name="recover", prompt="recover"),
        )
    )

    workflow.validate(set())


@pytest.mark.parametrize(
    "steps",
    [
        (
            WorkflowStep(name="root", prompt="root", is_entry=True),
            WorkflowStep(name="a", prompt="a", on_error="b"),
            WorkflowStep(name="b", prompt="b", on_error="a"),
        ),
        (
            WorkflowStep(
                name="root", prompt="root", next_steps=("act",), is_entry=True
            ),
            WorkflowStep(name="act", prompt="act", on_error="recover"),
            WorkflowStep(name="recover", prompt="recover", next_steps=("finish",)),
            WorkflowStep(name="finish", prompt="finish", on_error="act"),
        ),
    ],
    ids=["disconnected-error-cycle", "reachable-mixed-cycle"],
)
def test_workflow_validate_rejects_all_edge_cycles(
    steps: tuple[WorkflowStep, ...],
) -> None:
    workflow = Workflow(steps=steps)

    with pytest.raises(ValueError, match="cycle"):
        workflow.validate(set())


def test_workflow_validate_accepts_empty_explicit_allowlist() -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(
                name="summarize",
                prompt="summarize",
                inherit_all_capabilities=False,
                capabilities=(),
            ),
        )
    )

    workflow.validate({"run_python"})


def test_workflow_validate_rejects_inherit_all_with_named_capabilities() -> None:
    workflow = Workflow(
        steps=(WorkflowStep(name="bad", prompt="bad", capabilities=("run_python",)),)
    )

    with pytest.raises(ValueError, match="inherit_all_capabilities"):
        workflow.validate({"run_python"})


def test_workflow_validate_rejects_fan_in_and_fan_out() -> None:
    fan_in = Workflow(
        steps=(
            WorkflowStep(name="a", prompt="a", next_steps=("c",), is_entry=True),
            WorkflowStep(name="b", prompt="b", next_steps=("c",)),
            WorkflowStep(name="c", prompt="c"),
        )
    )
    with pytest.raises(ValueError, match="fan-in"):
        fan_in.validate(set())

    fan_out = Workflow(
        steps=(
            WorkflowStep(name="a", prompt="a", next_steps=("b", "c")),
            WorkflowStep(name="b", prompt="b"),
            WorkflowStep(name="c", prompt="c"),
        )
    )
    with pytest.raises(ValueError, match="fan-out"):
        fan_out.validate(set())
