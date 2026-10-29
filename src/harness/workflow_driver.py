"""Framework-driven workflow execution for harnesses that implement run_step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from loguru import logger

from src.harness.base import HarnessSpec
from src.harness.context import HarnessContext
from src.harness.transcript import HarnessTranscript, StepTranscript
from src.task.types import EpisodeArtifacts, Workflow, WorkflowStep


class StepFailure(RuntimeError):
    """Raised by ``HarnessSpec.run_step`` for recoverable step failures."""


class WorkflowExecutionError(RuntimeError):
    """Raised when workflow routing violates the strict v1 topology contract."""


@dataclass(frozen=True)
class StepInvocation:
    step: WorkflowStep
    ctx: HarnessContext
    fresh_conversation: bool
    is_first: bool
    is_last: bool


@dataclass(frozen=True)
class StepOutcome:
    is_ok: bool
    transcript: StepTranscript | None = None
    exception: BaseException | None = None

    @classmethod
    def ok(cls, transcript: StepTranscript) -> "StepOutcome":
        return cls(is_ok=True, transcript=transcript)

    @classmethod
    def error(cls, exception: BaseException) -> "StepOutcome":
        return cls(is_ok=False, exception=exception)


class WorkflowDriver:
    """Default execution strategy for ``HarnessSpec.run_workflow``."""

    def __init__(self, harness: HarnessSpec) -> None:
        self.harness = harness

    def run(self, workflow: Workflow, ctx: HarnessContext) -> HarnessTranscript:
        if type(self.harness).run_step is HarnessSpec.run_step:
            payload = {
                "harness": type(self.harness).__name__,
                "step_count": len(workflow.steps),
                "reason": "harness implements neither run_step nor a native run_workflow",
            }
            ctx.recorder.log_decision("workflow_collapsed", payload)
            logger.warning(f"workflow collapsed to single-step: {payload}")
            return self.harness.run_episode(ctx=ctx)

        step_transcripts: list[tuple[str, StepTranscript]] = []
        gen = self._iter_steps(workflow, ctx)
        outcome: StepOutcome | None = None
        last_executed_step_name: str | None = None
        try:
            while True:
                if outcome is not None and _global_budget_exhausted(ctx):
                    break
                invocation = gen.send(outcome) if outcome is not None else next(gen)
                if _global_budget_exhausted(ctx):
                    break
                last_executed_step_name = invocation.step.name
                ctx.recorder.set_label("last_workflow_step", invocation.step.name)
                ctx.recorder.log_decision(
                    "workflow_step_enter",
                    {
                        "step": invocation.step.name,
                        "capabilities": list(invocation.ctx.capabilities_view),
                    },
                )
                try:
                    transcript = self.harness.run_step(
                        invocation.step,
                        invocation.ctx.prompt_spec,
                        invocation.ctx,
                        fresh_conversation=invocation.fresh_conversation,
                        is_first=invocation.is_first,
                        is_last=invocation.is_last,
                    )
                except StepFailure as exc:
                    ctx.recorder.log_decision(
                        "workflow_step_exit",
                        {"step": invocation.step.name, "outcome": "error"},
                    )
                    outcome = StepOutcome.error(exc)
                    continue
                except BaseException:
                    gen.close()
                    raise
                step_transcripts.append((invocation.step.name, transcript))
                ctx.recorder.log_decision(
                    "workflow_step_exit",
                    {"step": invocation.step.name, "outcome": "ok"},
                )
                outcome = StepOutcome.ok(transcript)
        except StopIteration:
            pass
        finally:
            gen.close()
        return _assemble(step_transcripts, last_step_name=last_executed_step_name)

    def _iter_steps(
        self,
        workflow: Workflow,
        ctx: HarnessContext,
    ) -> Generator[StepInvocation, StepOutcome, None]:
        steps_by_name = {step.name: step for step in workflow.steps}
        order = workflow.topological_order()
        index_of = {step.name: i for i, step in enumerate(order)}
        total = len(order)

        current = workflow.entry_step
        seen: set[str] = set()
        while current is not None:
            if current.name in seen:
                raise WorkflowExecutionError(
                    f"step {current.name!r} would execute twice; on_error must "
                    "point forward in the topology"
                )
            seen.add(current.name)
            i = index_of[current.name]
            has_override = (
                ctx.constraints is not None
                and current.name in ctx.constraints.step_overrides
            )
            projected_ctx = ctx.with_capability_allowlist(
                _resolve_allowlist(current, ctx), workflow=workflow
            ).with_step_constraints(current.name).with_workflow_step(current.name)
            invocation = StepInvocation(
                step=current,
                ctx=projected_ctx,
                fresh_conversation=(current.context_mode == "isolated"),
                is_first=(i == 0),
                is_last=(i == total - 1),
            )
            if has_override and ctx.budget_ledger is not None:
                override = ctx.constraints.step_overrides[current.name]  # type: ignore[union-attr]
                ctx.budget_ledger.push_step(current.name, override.budgets)
            try:
                if has_override and ctx.budget_ledger is not None:
                    ctx.recorder.log_state(
                        "budget_ledger_step_push",
                        {
                            "step": current.name,
                            "effective_limit": override.budgets.max_task_tool_calls,
                            "used": ctx.budget_ledger.snapshot().task_tool_calls_used,
                        },
                    )
                outcome = yield invocation
            finally:
                if has_override and ctx.budget_ledger is not None:
                    snap = ctx.budget_ledger.pop_step(current.name)
                    ctx.recorder.log_state(
                        "budget_ledger_step_pop",
                        {
                            "step": current.name,
                            "effective_limit": snap.task_tool_calls_limit,
                            "used": snap.task_tool_calls_used,
                        },
                    )

            if outcome.is_ok:
                current = (
                    steps_by_name[current.next_steps[0]]
                    if current.next_steps
                    else None
                )
                continue
            if current.on_error is None:
                if outcome.exception is not None:
                    raise outcome.exception
                raise WorkflowExecutionError(
                    f"workflow step {current.name!r} failed without exception detail"
                )
            current = steps_by_name[current.on_error]


def _resolve_allowlist(step: WorkflowStep, ctx: HarnessContext) -> set[str]:
    if step.inherit_all_capabilities:
        return {cap.name for cap in ctx.prompt_spec.capabilities}
    return set(step.capabilities)


def _global_budget_exhausted(ctx: HarnessContext) -> bool:
    ledger = ctx.budget_ledger
    if ledger is None:
        return False
    exhaustion = ledger.exhausted()
    return exhaustion is not None and exhaustion.kind == "task_tool_calls"


def _assemble(
    step_transcripts: list[tuple[str, StepTranscript]],
    *,
    last_step_name: str | None = None,
) -> HarnessTranscript:
    if not step_transcripts:
        return HarnessTranscript(
            artifacts=EpisodeArtifacts(),
            llm_turns=0,
            termination_reason="workflow completed without executed steps",
            termination_category="success",
        )

    invocations = []
    results = []
    final_response = None
    for _, transcript in step_transcripts:
        invocations.extend(transcript.artifacts.capability_invocations)
        results.extend(transcript.artifacts.capability_results)
        if transcript.artifacts.final_response is not None:
            final_response = transcript.artifacts.final_response

    last_name, last = step_transcripts[-1]
    extra: dict = {"workflow_steps": [transcript.extra for _, transcript in step_transcripts]}
    extra["last_workflow_step"] = last_step_name or last_name
    return HarnessTranscript(
        artifacts=EpisodeArtifacts(
            capability_invocations=invocations,
            capability_results=results,
            final_response=final_response,
        ),
        llm_turns=sum(transcript.llm_turns for _, transcript in step_transcripts),
        termination_reason=last.termination_reason,
        termination_category=last.termination_category,
        extra=extra,
    )


__all__ = [
    "StepFailure",
    "StepInvocation",
    "StepOutcome",
    "WorkflowDriver",
    "WorkflowExecutionError",
]
