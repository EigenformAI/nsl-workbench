from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import (
    BudgetConstraints,
    Capability,
    EpisodeConstraints,
    NoToolReplyPolicy,
    StepConstraints,
    SuccessConstraints,
    TaskPromptSpec,
    Variation,
    Workflow,
    WorkflowStep,
)


def _ctx(tmp_path: Path) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    traced = TracedGenner(
        inner=MagicMock(),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    return HarnessContext(
        episode_id="ep-1",
        genner=traced,
        task=MagicMock(),
        variation=Variation(name="v", description="d"),
        prompt_spec=TaskPromptSpec(
            system_instruction="sys",
            capabilities=[
                Capability(name="alpha", description="a"),
                Capability(name="beta", description="b"),
            ],
        ),
        episode_context={},
        containers=[],
        agent_container=None,  # type: ignore[arg-type]
        host_cache_folder=tmp_path,
        config=HarnessConfigView(
            harness_settings={},
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,
        recorder=recorder,
        cancel_event=threading.Event(),
    )


def test_harness_context_workflow_defaults_to_none(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert ctx.workflow is None


def test_harness_context_can_project_capability_allowlist(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    workflow = Workflow(steps=(WorkflowStep(name="s", prompt="s"),))

    projected = ctx.with_capability_allowlist({"beta"}, workflow=workflow)

    assert [cap.name for cap in projected.prompt_spec.capabilities] == ["beta"]
    assert projected.capabilities_view == ("beta",)
    assert projected.workflow is workflow
    assert [cap.name for cap in ctx.prompt_spec.capabilities] == ["alpha", "beta"]


def test_harness_context_projects_step_constraints_after_capability_allowlist(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=10),
        no_tool_reply=NoToolReplyPolicy(retry=True, max_retries=2),
        success=SuccessConstraints(min_task_tool_calls_for_success=1),
        step_overrides={
            "plan": StepConstraints(
                budgets=BudgetConstraints(max_task_tool_calls=0),
                no_tool_reply=NoToolReplyPolicy(retry=True, max_retries=2),
                success=SuccessConstraints(min_task_tool_calls_for_success=0),
            )
        },
    )

    projected = ctx.with_capability_allowlist(set()).with_step_constraints("plan")

    assert projected.constraints is not None
    assert projected.constraints.budgets.max_task_tool_calls == 0
    assert projected.constraints.success.min_task_tool_calls_for_success == 0
    assert projected.constraints.no_tool_reply.retry is False
