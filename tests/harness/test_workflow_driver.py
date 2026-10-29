from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.harness.base import HarnessSpec
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.harness.transcript import HarnessTranscript, StepTranscript
from src.harness.workflow_driver import StepFailure, WorkflowDriver
from src.harness.budget import BudgetLedger, ToolCallRequest
from src.task.types import (
    BudgetConstraints,
    Capability,
    EpisodeArtifacts,
    EpisodeConstraints,
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


def _workflow() -> Workflow:
    return Workflow(
        steps=(
            WorkflowStep(name="plan", prompt="plan", next_steps=("act",)),
            WorkflowStep(
                name="act",
                prompt="act",
                inherit_all_capabilities=False,
                capabilities=("beta",),
                context_mode="isolated",
            ),
        )
    )


class _CollapsingHarness(HarnessSpec):
    def __init__(self) -> None:
        super().__init__({})
        self.run_episode_called = False

    def run_episode(self, *, ctx: HarnessContext) -> HarnessTranscript:
        self.run_episode_called = True
        return HarnessTranscript(
            artifacts=EpisodeArtifacts(final_response="single"),
            llm_turns=1,
            termination_reason="collapsed",
            termination_category="success",
        )


def test_workflow_driver_collapses_when_harness_has_no_step_api(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    harness = _CollapsingHarness()

    with patch("src.harness.workflow_driver.logger.warning") as warning:
        transcript = WorkflowDriver(harness).run(_workflow(), ctx)

    assert harness.run_episode_called is True
    assert transcript.artifacts.final_response == "single"
    assert warning.called
    events = ctx.recorder.events
    assert events[-1].category == "workflow_collapsed"
    assert events[-1].payload["step_count"] == 2


class _SteppedHarness(HarnessSpec):
    def __init__(self, *, fail_step: str | None = None) -> None:
        super().__init__({})
        self.fail_step = fail_step
        self.calls: list[dict[str, object]] = []

    def run_episode(self, *, ctx: HarnessContext) -> HarnessTranscript:
        raise AssertionError("run_episode should not be used")

    def run_step(
        self,
        step: WorkflowStep,
        prompt_spec: TaskPromptSpec,
        ctx: HarnessContext,
        *,
        fresh_conversation: bool,
        is_first: bool,
        is_last: bool,
    ) -> StepTranscript:
        self.calls.append(
            {
                "step": step.name,
                "prompt": step.prompt,
                "capabilities": ctx.capabilities_view,
                "fresh_conversation": fresh_conversation,
                "is_first": is_first,
                "is_last": is_last,
            }
        )
        if step.name == self.fail_step:
            raise StepFailure(f"{step.name} failed")
        return StepTranscript(
            artifacts=EpisodeArtifacts(final_response=step.name),
            llm_turns=1,
            termination_reason=f"{step.name} ok",
            termination_category="success",
        )


def test_workflow_driver_dispatches_steps_and_projects_capabilities(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    harness = _SteppedHarness()

    transcript = WorkflowDriver(harness).run(_workflow(), ctx)

    assert [call["step"] for call in harness.calls] == ["plan", "act"]
    assert harness.calls[0]["capabilities"] == ("alpha", "beta")
    assert harness.calls[1]["capabilities"] == ("beta",)
    assert harness.calls[0]["is_first"] is True
    assert harness.calls[1]["is_last"] is True
    assert harness.calls[1]["fresh_conversation"] is True
    assert transcript.llm_turns == 2
    assert transcript.artifacts.final_response == "act"
    assert [event.category for event in ctx.recorder.events] == [
        "workflow_step_enter",
        "workflow_step_exit",
        "workflow_step_enter",
        "workflow_step_exit",
    ]
    assert transcript.extra["last_workflow_step"] == "act"


def test_workflow_driver_routes_step_failure_to_on_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    workflow = Workflow(
        steps=(
            WorkflowStep(name="try", prompt="try", on_error="recover", is_entry=True),
            WorkflowStep(name="recover", prompt="recover"),
        )
    )
    harness = _SteppedHarness(fail_step="try")

    transcript = WorkflowDriver(harness).run(workflow, ctx)

    assert [call["step"] for call in harness.calls] == ["try", "recover"]
    assert transcript.artifacts.final_response == "recover"
    exits = [
        event for event in ctx.recorder.events if event.category == "workflow_step_exit"
    ]
    assert [event.payload["outcome"] for event in exits] == ["error", "ok"]


class _BudgetHarness(HarnessSpec):
    def __init__(self) -> None:
        super().__init__({})
        self.limits: list[int | None] = []

    def run_episode(self, *, ctx: HarnessContext) -> HarnessTranscript:
        raise AssertionError("run_episode should not be used")

    def run_step(
        self,
        step: WorkflowStep,
        prompt_spec: TaskPromptSpec,
        ctx: HarnessContext,
        *,
        fresh_conversation: bool,
        is_first: bool,
        is_last: bool,
    ) -> StepTranscript:
        assert ctx.budget_ledger is not None
        self.limits.append(ctx.budget_ledger.snapshot().task_tool_calls_limit)
        if step.name == "first":
            assert ctx.budget_ledger.try_consume_tool_calls(
                [ToolCallRequest("alpha"), ToolCallRequest("alpha")]
            ).accepted
        if step.name == "second":
            assert ctx.constraints is not None
            assert ctx.constraints.success.min_task_tool_calls_for_success == 0
            result = ctx.budget_ledger.try_consume_tool_calls(
                [ToolCallRequest("alpha")]
            )
            assert result.accepted is False
            assert result.rejection_reason == "task_tool_calls"
        return StepTranscript(
            artifacts=EpisodeArtifacts(final_response=step.name),
            llm_turns=1,
            termination_reason=f"{step.name} ok",
            termination_category="success",
        )


def test_workflow_driver_pushes_step_budget_overrides_and_unwinds(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.budget_ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=5))
    ctx.constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=5),
        step_overrides={
            "second": StepConstraints(
                budgets=BudgetConstraints(max_task_tool_calls=2),
                success=SuccessConstraints(min_task_tool_calls_for_success=0),
            )
        },
    )
    workflow = Workflow(
        steps=(
            WorkflowStep(name="first", prompt="first", next_steps=("second",)),
            WorkflowStep(name="second", prompt="second"),
        )
    )
    harness = _BudgetHarness()

    transcript = WorkflowDriver(harness).run(workflow, ctx)

    assert harness.limits == [5, 2]
    assert ctx.budget_ledger.active_step_name() is None
    assert ctx.budget_ledger.exhausted() is not None
    assert transcript.extra["last_workflow_step"] == "second"


class _GlobalBudgetLatchHarness(_SteppedHarness):
    def run_step(
        self,
        step: WorkflowStep,
        prompt_spec: TaskPromptSpec,
        ctx: HarnessContext,
        *,
        fresh_conversation: bool,
        is_first: bool,
        is_last: bool,
    ) -> StepTranscript:
        transcript = super().run_step(
            step,
            prompt_spec,
            ctx,
            fresh_conversation=fresh_conversation,
            is_first=is_first,
            is_last=is_last,
        )
        if step.name == "plan":
            assert ctx.budget_ledger is not None
            assert ctx.budget_ledger.try_consume_tool_calls(
                [ToolCallRequest("alpha")]
            ).accepted
            rejection = ctx.budget_ledger.try_consume_tool_calls(
                [ToolCallRequest("alpha")]
            )
            assert rejection.rejection_reason == "task_tool_calls"
        return transcript


def test_global_tool_budget_latch_stops_before_next_step(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.budget_ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=1))
    harness = _GlobalBudgetLatchHarness()

    transcript = WorkflowDriver(harness).run(_workflow(), ctx)

    assert [call["step"] for call in harness.calls] == ["plan"]
    assert ctx.budget_ledger.exhausted() is not None
    assert transcript.extra["last_workflow_step"] == "plan"


def test_workflow_driver_unwinds_step_budget_when_push_logging_fails(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.budget_ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=5))
    ctx.constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=5),
        step_overrides={
            "only": StepConstraints(
                budgets=BudgetConstraints(max_task_tool_calls=2),
            )
        },
    )
    workflow = Workflow(steps=(WorkflowStep(name="only", prompt="only"),))

    with patch.object(
        ctx.recorder,
        "log_state",
        side_effect=RuntimeError("recorder failed"),
    ):
        with pytest.raises(RuntimeError, match="recorder failed"):
            WorkflowDriver(_BudgetHarness()).run(workflow, ctx)

    assert ctx.budget_ledger.active_step_name() is None


def test_workflow_driver_raises_when_on_error_repeats_step(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    workflow = Workflow(
        steps=(
            WorkflowStep(name="try", prompt="try", next_steps=("fail",)),
            WorkflowStep(name="fail", prompt="fail", on_error="try"),
        )
    )

    with pytest.raises(ValueError, match="workflow contains a cycle"):
        WorkflowDriver(_SteppedHarness(fail_step="fail")).run(workflow, ctx)
