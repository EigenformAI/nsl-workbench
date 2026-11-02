"""Native task workflows share a decision budget and enforce phase constraints."""

from __future__ import annotations

import threading
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from result import Err, Ok

from src.framework.capability_bridge import CapabilityMcpBridge
from src.genner.Base import (
    CONTEXT_OVERFLOW_PREFIX,
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
    InferenceTimeoutError,
)
from src.harness.base import HarnessError
from src.harness.budget import BudgetLedger
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.orchestrator_modes import OrchestratorModeHarness
from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.observability.types import InferenceResult
from src.task.base import TaskEnvironmentError
from src.task.types import (
    BudgetConstraints,
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeConstraints,
    NoToolReplyPolicy,
    StepConstraints,
    SuccessConstraints,
    TaskPromptSpec,
    Variation,
    Workflow,
    WorkflowStep,
)


class _Task:
    def __init__(self, *, report_success=True, error=None):
        self.calls = []
        self.report_success = report_success
        self.error = error

    def parse_response(self, raw_response, *, invoked_capability=None):
        if "REPORT" in raw_response:
            return [CapabilityInvocation(name="report_result")]
        if raw_response == "TOOL":
            return [CapabilityInvocation(name="run_python")]
        return []

    def execute_capability(self, invocation, containers, variation, ctx):
        self.calls.append((invocation, ctx))
        if self.error is not None:
            raise self.error
        return CapabilityResult(
            name=invocation.name,
            success=self.report_success if invocation.name == "report_result" else True,
            output={"stdout": "ran", "stderr": "", "return_code": 0},
        )


def _workflow():
    return Workflow(
        steps=(
            WorkflowStep(
                name="explore",
                prompt="EXPLORE_OBJECTIVE",
                inherit_all_capabilities=False,
                capabilities=("run_python",),
                next_steps=("plan",),
            ),
            WorkflowStep(
                name="plan",
                prompt="PLAN_OBJECTIVE",
                inherit_all_capabilities=False,
                next_steps=("execute",),
            ),
            WorkflowStep(
                name="execute",
                prompt="EXECUTE_OBJECTIVE",
                inherit_all_capabilities=False,
                capabilities=("run_python", "report_result"),
                terminator_capabilities=("report_result",),
            ),
        )
    )


def _ctx(tmp_path, *, respond=None, max_iters=12, task=None, repetition_guard=None):
    recorder = EventRecorder(episode_id="ep", output_path=tmp_path / "events.jsonl")
    cancel_event = threading.Event()
    constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=10),
        no_tool_reply=NoToolReplyPolicy(retry=True, max_retries=2),
        step_overrides={
            "plan": StepConstraints(
                budgets=BudgetConstraints(max_task_tool_calls=0),
                no_tool_reply=NoToolReplyPolicy(retry=True, max_retries=2),
                success=SuccessConstraints(min_task_tool_calls_for_success=0),
            ),
            "execute": StepConstraints(
                budgets=BudgetConstraints(max_task_tool_calls=5),
            ),
        },
    )
    ledger = BudgetLedger(constraints.budgets)
    inner = MagicMock(collector=None)

    def complete(messages):
        meta = messages[0]["meta"]
        if respond is None:
            content = "DECISION: ACT" if meta["phase"] == "orchestrator" else "working"
        else:
            content = respond(meta["phase"], meta["workflow_step"])
        return content if isinstance(content, Err) else Ok(InferenceResult(content))

    inner.plist_completion.side_effect = complete
    genner = TracedGenner(
        inner=inner,
        recorder=recorder,
        cancel_event=cancel_event,
        episode_id="ep",
        budget_ledger=ledger,
    )
    return HarnessContext(
        episode_id="ep",
        genner=genner,
        task=task or _Task(),
        variation=Variation(name="v", description="d"),
        prompt_spec=TaskPromptSpec(
            system_instruction="sys",
            environment_context="ENVIRONMENT_CONTEXT",
            capabilities=[
                Capability(name="run_python", description="exec", runs_code=True),
                Capability(name="report_result", description="report"),
            ],
        ),
        episode_context={},
        containers=[],
        agent_container=None,
        host_cache_folder=tmp_path,
        config=HarnessConfigView(
            harness_settings={
                "max_harness_iterations": max_iters,
                "tool_output_max_chars": 0,
                "repetition_guard": repetition_guard or {},
                # Invalid flat-loop settings must not affect native workflows.
                "modes": {},
                "orchestrator_prompt": "UNUSED_CONFIG_PROMPT",
            },
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,
        recorder=recorder,
        cancel_event=cancel_event,
        constraints=constraints,
        budget_ledger=ledger,
    )


def _phases(ctx):
    return [e.payload["step"] for e in ctx.recorder.events if e.category == "phase_start"]


def _calls(ctx):
    return [call.args[0][0]["meta"] for call in ctx.genner.inner.plist_completion.call_args_list]


def test_task_prompts_tool_free_plan_and_successful_terminator(tmp_path, monkeypatch):
    def respond(phase, step):
        if phase == "orchestrator":
            return "DECISION: ADVANCE" if step == "explore" else "DECISION: ACT"
        return "REPORT" if step == "plan" else "```python\nprint(1)\n```\nREPORT"

    ctx = _ctx(tmp_path, respond=respond)
    harness = OrchestratorModeHarness({})
    projected = []
    worker = harness._run_workflow_worker

    def observe_worker(**kwargs):
        projected.append(kwargs["ctx"])
        return worker(**kwargs)

    monkeypatch.setattr(harness, "_run_workflow_worker", observe_worker)

    transcript = harness.run_workflow(_workflow(), ctx)

    assert _phases(ctx) == ["explore", "plan", "execute"]
    assert transcript.termination_category == "success"
    assert transcript.extra["harness_iterations_count"] == 3
    assert [inv.name for inv, _ in ctx.task.calls] == ["run_python", "report_result"]
    assert all(call_ctx.workflow_step == "execute" for _, call_ctx in ctx.task.calls)
    assert transcript.artifacts.capability_results[0].success is False
    assert "not permitted" in transcript.artifacts.capability_results[0].error
    assert ctx.budget_ledger.snapshot().task_tool_calls_used == 2
    assert ctx.budget_ledger.active_step_name() is None
    assert ctx.budget_ledger.exhausted() is None  # Forbidden plan call never reserved.
    assert projected[0].capabilities_view == ()
    assert projected[0].constraints.success.min_task_tool_calls_for_success == 0
    assert projected[0].constraints.no_tool_reply.retry is False
    assert projected[1].constraints.budgets.max_task_tool_calls == 5
    assert all(phase_ctx.budget_ledger is ctx.budget_ledger for phase_ctx in projected)
    prompts = "\n".join(
        call.args[0][-1]["content"] for call in ctx.genner.inner.plist_completion.call_args_list
    )
    assert all(f"{name}_OBJECTIVE" in prompts for name in ("EXPLORE", "PLAN", "EXECUTE"))
    assert "UNUSED_CONFIG_PROMPT" not in prompts
    assert all(
        "ENVIRONMENT_CONTEXT" in call.args[0][0]["content"]
        for call in ctx.genner.inner.plist_completion.call_args_list
    )


@pytest.mark.parametrize("max_iters,reservation", [(1, 1), (8, 2), (12, 2)])
def test_shared_decision_budget_reserves_final_phase(tmp_path, max_iters, reservation):
    ctx = _ctx(tmp_path, max_iters=max_iters)

    transcript = OrchestratorModeHarness({}).run_workflow(_workflow(), ctx)

    assert _phases(ctx) == ["explore"] * (max_iters - reservation) + ["execute"] * reservation
    assert sum(call["phase"] == "orchestrator" for call in _calls(ctx)) == max_iters
    assert sum(call["phase"] != "orchestrator" for call in _calls(ctx)) == max_iters
    assert transcript.termination_category == "budget_exhausted"
    assert transcript.extra["harness_iterations_count"] == max_iters
    force = next(e for e in ctx.recorder.events if e.category == "workflow_force_advance")
    assert force.payload["budget_remaining"] == reservation
    assert ctx.budget_ledger.active_step_name() is None


def test_unparseable_decision_acts_and_failed_terminator_does_not_finish(tmp_path):
    ctx = _ctx(
        tmp_path,
        max_iters=3,
        task=_Task(report_success=False),
        respond=lambda phase, step: "unparseable" if phase == "orchestrator" else "REPORT",
    )
    workflow = Workflow(steps=(_workflow().steps[-1],))

    transcript = OrchestratorModeHarness({}).run_workflow(workflow, ctx)

    assert _phases(ctx) == ["execute"] * 3
    assert len(ctx.task.calls) == 3
    assert not any(result.success for result in transcript.artifacts.capability_results)
    assert transcript.termination_category == "budget_exhausted"
    assert all(not turn["terminator"] for turn in transcript.extra["mode_history"])


def test_forbidden_terminator_cannot_advance_active_phase(tmp_path):
    ctx = _ctx(
        tmp_path,
        max_iters=4,
        respond=lambda phase, step: "DECISION: ACT" if phase == "orchestrator" else "REPORT",
    )

    transcript = OrchestratorModeHarness({}).run_workflow(_workflow(), ctx)

    assert _phases(ctx) == ["explore", "explore", "explore", "execute"]
    assert [inv.name for inv, _ in ctx.task.calls] == ["report_result"]
    assert [result.success for result in transcript.artifacts.capability_results] == [False] * 3 + [True]
    assert ctx.budget_ledger.snapshot().task_tool_calls_used == 1


def test_code_capability_terminator_advances_nonfinal_phase(tmp_path):
    def respond(phase, step):
        if phase == "orchestrator":
            return "DECISION: ACT"
        return "```python\nprint(1)\n```" if step == "explore" else "REPORT"

    ctx = _ctx(tmp_path, respond=respond)
    explore, _, execute = _workflow().steps
    workflow = Workflow(
        steps=(replace(explore, next_steps=("execute",), terminator_capabilities=("run_python",)), execute)
    )

    transcript = OrchestratorModeHarness({}).run_workflow(workflow, ctx)

    assert _phases(ctx) == ["explore", "execute"]
    assert [turn["terminator"] for turn in transcript.extra["mode_history"]] == [True, True]
    assert transcript.termination_category == "success"


def test_final_advance_completes_transcript_without_scoring_call(tmp_path):
    ctx = _ctx(tmp_path, respond=lambda phase, step: "DECISION: **advance**")

    transcript = OrchestratorModeHarness({}).run_workflow(_workflow(), ctx)

    assert _phases(ctx) == ["explore", "plan", "execute"]
    assert len(_calls(ctx)) == 3
    assert ctx.task.calls == []
    assert transcript.termination_category == "success"
    assert ctx.budget_ledger.active_step_name() is None


def test_workflow_scratchpad_resets_between_episodes(tmp_path):
    ctx = _ctx(tmp_path, max_iters=1)
    stale = CrossEpisodeScratchpad(max_chars=5000)
    stale.append("STALE_PRIOR_EPISODE: already done")
    ctx.harness_session["scratchpad"] = stale
    harness = OrchestratorModeHarness({})

    harness.run_workflow(_workflow(), ctx)
    ctx.harness_session["scratchpad"].append("FIRST_EPISODE_OUTPUT")
    ctx.genner.inner.plist_completion.reset_mock()
    harness.run_workflow(_workflow(), ctx)

    prompts = "\n".join(
        call.args[0][-1]["content"] for call in ctx.genner.inner.plist_completion.call_args_list
    )
    assert "STALE_PRIOR_EPISODE" not in prompts
    assert "FIRST_EPISODE_OUTPUT" not in prompts


@pytest.mark.parametrize("property_name", ["on_error", "context_mode"])
def test_unsupported_native_semantics_fail_before_inference(tmp_path, property_name):
    steps = list(_workflow().steps)
    values = {"on_error": "plan"} if property_name == "on_error" else {"context_mode": "isolated"}
    steps[0] = replace(steps[0], **values)
    workflow = Workflow(steps=tuple(steps))
    workflow.validate({"run_python", "report_result"})  # Globally supported semantics.
    ctx = _ctx(tmp_path)

    with pytest.raises(HarnessError, match=f"explore.*{property_name}"):
        OrchestratorModeHarness({}).run_workflow(workflow, ctx)

    ctx.genner.inner.plist_completion.assert_not_called()
    assert ctx.task.calls == []


def test_repetition_advances_to_final_phase_and_unwinds_budget(tmp_path):
    para = "A long paragraph repeated many times to trigger the detector instead of doing useful work. " * 2
    degenerate = "\n\n".join([para] * 8)
    ctx = _ctx(
        tmp_path,
        respond=lambda phase, step: (
            degenerate if step == "explore" else "DECISION: ACT"
        ) if phase == "orchestrator" else "REPORT",
        repetition_guard={
            "min_paragraphs": 3,
            "min_paragraph_chars": 50,
            "window_size": 4,
            "first_hit_action": "warn_only",
            "second_hit_action": "end_episode",
        },
    )
    ctx.constraints = replace(
        ctx.constraints,
        step_overrides={**ctx.constraints.step_overrides, "explore": StepConstraints()},
    )

    transcript = OrchestratorModeHarness({}).run_workflow(_workflow(), ctx)

    assert "execute" in _phases(ctx)
    assert transcript.termination_category == "success"
    assert any(e.category == "repetition_collapse_force_advance" for e in ctx.recorder.events)
    assert ctx.budget_ledger.active_step_name() is None


@pytest.mark.parametrize("cancel_after_worker", [False, True])
def test_cancellation_stops_inference_and_unwinds_step_budget(tmp_path, cancel_after_worker):
    def respond(phase, step):
        if cancel_after_worker and phase != "orchestrator":
            ctx.cancel_event.set()
        return "DECISION: ACT" if phase == "orchestrator" else "working"

    ctx = _ctx(tmp_path, respond=respond)
    if not cancel_after_worker:
        ctx.cancel_event.set()

    workflow = Workflow(steps=(_workflow().steps[-1],))
    transcript = OrchestratorModeHarness({}).run_workflow(workflow, ctx)

    assert transcript.termination_category == "wall_clock"
    assert ctx.genner.inner.plist_completion.call_count == (2 if cancel_after_worker else 0)
    assert ctx.budget_ledger.active_step_name() is None


@pytest.mark.parametrize(
    "worker_response,capability_name",
    [("```python\nprint(1)\n```", "run_python"), ("REPORT", "report_result")],
    ids=["code", "parsed"],
)
def test_cancellation_at_worker_dispatch_returns_wall_clock_transcript(
    tmp_path, monkeypatch, worker_response, capability_name
):
    ctx = _ctx(
        tmp_path,
        respond=lambda phase, step: "DECISION: ACT" if phase == "orchestrator" else worker_response,
    )
    dispatch_attempts = []
    invoke = CapabilityMcpBridge.invoke

    def cancel_before_dispatch(bridge, invocation):
        dispatch_attempts.append(invocation.name)
        assert bridge.ctx.budget_ledger.active_step_name() == "execute"
        ctx.cancel_event.set()
        return invoke(bridge, invocation)

    monkeypatch.setattr(CapabilityMcpBridge, "invoke", cancel_before_dispatch)
    harness = OrchestratorModeHarness({})

    transcript = harness.run_workflow(Workflow(steps=(_workflow().steps[-1],)), ctx)

    assert dispatch_attempts == [capability_name]
    assert transcript.termination_category == "wall_clock"
    assert transcript.termination_reason == "cancel_event set"
    assert ctx.genner.inner.plist_completion.call_count == 2
    assert ctx.task.calls == []
    assert ctx.recorder.capability_pairs() == []
    assert ctx.budget_ledger.snapshot().task_tool_calls_used == 0
    assert ctx.budget_ledger.active_step_name() is None
    assert harness.telemetry() == {}


@pytest.mark.parametrize("error_phase", ["orchestrator", "execute"])
@pytest.mark.parametrize(
    "prefix,exception,category",
    [
        (CONTEXT_OVERFLOW_PREFIX, None, "context_overflow"),
        (INFERENCE_TIMEOUT_PREFIX, InferenceTimeoutError, None),
        (INFERENCE_UNAVAILABLE_PREFIX, HarnessError, None),
    ],
)
def test_inference_failures_keep_runtime_classification_and_unwind(
    tmp_path, error_phase, prefix, exception, category
):
    ctx = _ctx(
        tmp_path,
        respond=lambda phase, step: Err(f"{prefix} failed") if phase == error_phase else "DECISION: ACT",
    )
    harness = OrchestratorModeHarness({})
    workflow = Workflow(steps=(_workflow().steps[-1],))

    if exception is not None:
        with pytest.raises(exception, match="failed"):
            harness.run_workflow(workflow, ctx)
    else:
        assert harness.run_workflow(workflow, ctx).termination_category == category

    assert ctx.budget_ledger.active_step_name() is None
    assert harness.telemetry() == {}


def test_environment_error_remains_deferred_and_unwinds_budget(tmp_path):
    error = TaskEnvironmentError("container disappeared")
    ctx = _ctx(
        tmp_path,
        task=_Task(error=error),
        respond=lambda phase, step: "DECISION: ACT" if phase == "orchestrator" else "REPORT",
    )

    transcript = OrchestratorModeHarness({}).run_workflow(
        Workflow(steps=(_workflow().steps[-1],)), ctx
    )

    assert ctx.extra["task_environment_error"] is error
    assert transcript.termination_category == "harness_error"
    assert ctx.budget_ledger.active_step_name() is None
    pairs = ctx.recorder.capability_pairs()
    assert len(pairs) == 1
    assert pairs[0][1].success is False


def test_step_budget_limits_include_prior_phases_and_do_not_reset(tmp_path):
    def respond(phase, step):
        return "DECISION: ACT" if phase == "orchestrator" else "TOOL"

    ctx = _ctx(tmp_path, respond=respond, max_iters=3)
    ctx.constraints = replace(
        ctx.constraints,
        step_overrides={"execute": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=2))},
    )

    transcript = OrchestratorModeHarness({}).run_workflow(_workflow(), ctx)

    assert [call_ctx.workflow_step for _, call_ctx in ctx.task.calls] == ["explore", "explore"]
    assert transcript.artifacts.capability_results[-1].error == "budget exhausted: task_tool_calls"
    assert ctx.budget_ledger.snapshot().task_tool_calls_used == 2
    assert ctx.budget_ledger.active_step_name() is None


def test_push_logging_failure_unwinds_step_budget(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    original = ctx.recorder.log_state

    def log_state(category, payload):
        if category == "budget_ledger_step_push":
            raise RuntimeError("recorder failed")
        return original(category, payload)

    monkeypatch.setattr(ctx.recorder, "log_state", log_state)

    transcript = OrchestratorModeHarness({}).run_workflow(
        Workflow(steps=(_workflow().steps[-1],)), ctx
    )

    assert transcript.termination_category == "harness_error"
    assert "recorder failed" in transcript.termination_reason
    assert ctx.budget_ledger.active_step_name() is None
