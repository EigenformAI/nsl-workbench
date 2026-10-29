from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from docker.models.containers import Container as DockerContainer
from loguru import logger

from src.container import ContainerManager
from src.execution.backend_runtime import BackendRuntime
from src.execution.episode_transforms import (
    categorize_termination,
    to_prompt_responses,
    to_trajectory,
)
from src.execution.telemetry import (
    framework_telemetry_from_counters,
    snapshot_framework_telemetry,
    snapshot_recorder_counters,
    snapshot_recorder_telemetry_sources,
)
from src.genner.Base import InferenceTimeoutError
from src.harness.base import HarnessError
from src.harness.budget import BudgetLedger
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.loader import (
    construct_harness,
    resolve_event_recorder_class,
    resolve_traced_genner_class,
)
from src.harness.training_row_adapter import records_to_rows
from src.helper import get_formatted_repo_info
from src.parallel import StopReason
from src.task.base import TaskEnvironmentError
from src.task.types import (
    EpisodeArtifacts,
    EpisodeConstraints,
    FinalizationContext,
    TaskCommitPreparation,
    TaskReward,
    TrustedEpisodeIdentity,
    Variation,
    Workflow,
)


@dataclass
class EpisodeRequest:
    episode_id: str
    containers: list[DockerContainer]
    container_manager: ContainerManager
    agent_container: DockerContainer
    variation: Variation
    episode_context: dict[str, Any]
    private_context: dict[str, Any] | None = None
    harness_session: dict[str, Any] | None = None
    stop_event: threading.Event | None = None
    stop_reason: StopReason | None = None
    telemetry_observer: Callable[[int | None, dict[str, str]], None] | None = None
    identity: TrustedEpisodeIdentity | None = None


@dataclass
class EpisodeOutcome:
    episode_id: str
    score: float
    success: bool
    partial: bool
    llm_turns_count: int
    train_rows: list[dict[str, Any]]
    prompt_responses: list[dict[str, Any]]
    trajectory: dict[str, Any]
    error_message: str | None
    error_category: str | None
    reward_breakdown: dict[str, Any]
    harness_error: bool
    tool_calls_count: int = 0
    task_commit_token: str | None = None
    task_commit_digest: str | None = None


def _task_episode_commit_enabled(task: Any) -> bool:
    declaration = getattr(type(task), "episode_commit_enabled", None)
    return bool(declaration(task)) if callable(declaration) else False


@dataclass
class _CancellationState:
    cancel_event: threading.Event
    wall_clock_timer: threading.Timer | None
    external_stop_watcher: threading.Thread | None


@dataclass
class _TracingState:
    recorder: Any
    traced_genner: Any
    harness_session: dict[str, Any]
    publish_telemetry: Callable[[], None]


def _build_cancellation(
    rt: BackendRuntime, episode_id: str, stop_event: threading.Event | None
) -> _CancellationState:
    cancel_event = threading.Event()
    wall_clock_timer: threading.Timer | None = None
    wall_clock_seconds = rt.config.harness.episode_wall_clock_seconds
    if wall_clock_seconds and wall_clock_seconds > 0:

        def _on_wall_clock() -> None:
            logger.warning(
                f"Episode {episode_id}: wall-clock timeout "
                f"({wall_clock_seconds}s) - signaling cancel"
            )
            cancel_event.set()

        wall_clock_timer = threading.Timer(wall_clock_seconds, _on_wall_clock)
        wall_clock_timer.daemon = True
        wall_clock_timer.start()

    external_stop_watcher: threading.Thread | None = None
    if stop_event is not None:

        def _watch_external_stop() -> None:
            stop_event.wait()
            if not cancel_event.is_set():
                cancel_event.set()

        external_stop_watcher = threading.Thread(
            target=_watch_external_stop, daemon=True
        )
        external_stop_watcher.start()

    return _CancellationState(
        cancel_event=cancel_event,
        wall_clock_timer=wall_clock_timer,
        external_stop_watcher=external_stop_watcher,
    )


def _stop_cancellation(cancellation: _CancellationState) -> None:
    if cancellation.wall_clock_timer is not None:
        cancellation.wall_clock_timer.cancel()
    if (
        cancellation.external_stop_watcher is not None
        and not cancellation.cancel_event.is_set()
    ):
        cancellation.cancel_event.set()


def _stringify_telemetry(telemetry: Any) -> dict[str, str]:
    if not isinstance(telemetry, dict):
        return {}
    return {str(key): str(value) for key, value in telemetry.items()}


def _snapshot_harness_telemetry(harness: Any) -> dict[str, str]:
    telemetry_fn = getattr(harness, "telemetry", None)
    if not callable(telemetry_fn):
        return {}
    try:
        telemetry = telemetry_fn()
    except Exception as exc:
        logger.debug(f"harness telemetry raised: {exc}")
        return {}
    return _stringify_telemetry(telemetry)


def _snapshot_recorder_backed_harness_telemetry(
    harness: Any,
    counters: dict[str, Any],
    labels: dict[str, str],
) -> dict[str, str]:
    telemetry_fn = getattr(harness, "telemetry_from_recorder_snapshot", None)
    if not callable(telemetry_fn):
        return {}
    try:
        telemetry = telemetry_fn(dict(counters), dict(labels))
    except Exception as exc:
        logger.debug(f"harness recorder telemetry raised: {exc}")
        return {}
    return _stringify_telemetry(telemetry)


def _snapshot_episode_telemetry(harness: Any, recorder: Any) -> dict[str, str]:
    if callable(getattr(harness, "telemetry_from_recorder_snapshot", None)):
        counters, labels = snapshot_recorder_telemetry_sources(recorder)
        recorder_backed = _snapshot_recorder_backed_harness_telemetry(
            harness,
            counters,
            labels,
        )
        telemetry = recorder_backed
        telemetry.update(framework_telemetry_from_counters(counters))
        return telemetry

    telemetry = _snapshot_harness_telemetry(harness)
    telemetry.update(snapshot_framework_telemetry(recorder))
    return telemetry


def _snapshot_harness_failure_extras(harness: Any) -> dict[str, Any]:
    failure_fn = getattr(harness, "failure_extras", None)
    if not callable(failure_fn):
        return {}
    try:
        extra = failure_fn()
    except Exception as exc:
        logger.warning(f"failure_extras raised: {exc}")
        return {}
    return dict(extra) if isinstance(extra, dict) else {}


def _build_recorder_and_genner(
    rt: BackendRuntime,
    req: EpisodeRequest,
    cancel_event: threading.Event,
    harness: Any,
) -> _TracingState:
    artifact_dir = Path(rt.config.train_data_save_folder) / "events"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    recorder_cls = resolve_event_recorder_class(rt.config.harness)
    recorder = recorder_cls(
        episode_id=req.episode_id,
        output_path=artifact_dir / f"events_{req.episode_id}.jsonl",
        cancel_event=cancel_event,
    )

    harness_session = req.harness_session if req.harness_session is not None else {}
    last_prompt_tokens_holder: dict[str, int | None] = {"value": None}

    def _publish_telemetry() -> None:
        if req.telemetry_observer is None:
            return
        try:
            telemetry = _snapshot_episode_telemetry(harness, recorder)
            req.telemetry_observer(
                last_prompt_tokens_holder["value"],
                telemetry,
            )
        except Exception as exc:
            logger.debug(f"telemetry_observer raised: {exc}")

    def _on_inference(usage: Any) -> None:
        if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
            last_prompt_tokens_holder["value"] = usage.prompt_tokens
        _publish_telemetry()

    traced_cls = resolve_traced_genner_class(rt.config.harness)
    traced = traced_cls(
        inner=rt.genner,
        recorder=recorder,
        cancel_event=cancel_event,
        episode_id=req.episode_id,
        on_inference=_on_inference,
    )
    return _TracingState(
        recorder=recorder,
        traced_genner=traced,
        harness_session=harness_session,
        publish_telemetry=_publish_telemetry,
    )


def _build_harness_context(
    rt: BackendRuntime,
    req: EpisodeRequest,
    tracing: _TracingState,
    cancel_event: threading.Event,
) -> HarnessContext:
    prompt_spec = rt.task.prompt_spec(req.variation, req.episode_context)
    workflow = None
    if callable(getattr(type(rt.task), "workflow", None)):
        workflow = rt.task.workflow(req.variation, req.episode_context)
    if workflow is not None:
        if not isinstance(workflow, Workflow):
            raise TypeError(
                "task.workflow() returned "
                f"{type(workflow).__name__}, expected Workflow | None"
            )
        workflow.validate({cap.name for cap in prompt_spec.capabilities})

    constraints: EpisodeConstraints = EpisodeConstraints()
    if callable(getattr(type(rt.task), "episode_constraints", None)):
        raw = rt.task.episode_constraints(req.variation, req.episode_context)
        if isinstance(raw, EpisodeConstraints):
            constraints = raw
    if constraints.step_overrides:
        if workflow is None:
            raise ValueError(
                "EpisodeConstraints.step_overrides requires a workflow but "
                "task.workflow() returned None"
            )
        step_names = {s.name for s in workflow.steps}
        unknown = sorted(set(constraints.step_overrides.keys()) - step_names)
        if unknown:
            raise ValueError(f"unknown step_overrides keys: {unknown}")

    budget_ledger = BudgetLedger(constraints.budgets)
    set_budget_ledger = getattr(tracing.traced_genner, "set_budget_ledger", None)
    if callable(set_budget_ledger):
        set_budget_ledger(budget_ledger)
    try:
        tracing.recorder.log_state(
            "episode_constraints_initialized",
            {
                "max_task_tool_calls": constraints.budgets.max_task_tool_calls,
                "max_llm_turns": constraints.budgets.max_llm_turns,
                "min_task_tool_calls_for_success": (
                    constraints.success.min_task_tool_calls_for_success
                ),
                "terminal_capability_for_success": (
                    constraints.success.terminal_capability_for_success
                ),
            },
        )
    except Exception as exc:
        logger.debug(f"episode_constraints log_state skipped: {exc}")

    harness_settings = dict(
        getattr(rt.config.harness, rt.config.harness.name, {}) or {}
    )
    config_view = HarnessConfigView(
        harness_settings=harness_settings,
        model_name=rt.config.model_name,
        train_data_save_folder=rt.config.train_data_save_folder,
        code_host_cache_path=rt.config.code_host_cache_path,
    )
    host_cache_folder = Path(rt.config.code_host_cache_path) / "episode_v2_mode"
    if req.identity is not None and req.identity.episode_id != req.episode_id:
        raise ValueError("episode request identity does not match episode_id")
    return HarnessContext(
        episode_id=req.episode_id,
        genner=tracing.traced_genner,
        task=rt.task,
        variation=req.variation,
        prompt_spec=prompt_spec,
        episode_context=req.episode_context,
        containers=req.containers,
        agent_container=req.agent_container,
        host_cache_folder=host_cache_folder,
        config=config_view,
        metrics=getattr(rt.genner, "collector", None),
        recorder=tracing.recorder,
        cancel_event=cancel_event,
        identity=req.identity,
        constraints=constraints,
        budget_ledger=budget_ledger,
        workflow=workflow,
        harness_session=tracing.harness_session,
        docker_client=rt.docker_client,
    )


def _finalize_reward(
    rt: BackendRuntime,
    req: EpisodeRequest,
    *,
    initial_state: Any,
    artifacts: EpisodeArtifacts,
    finalization_context: FinalizationContext | None = None,
) -> TaskReward:
    try:
        return rt.task.finalize_episode(
            req.containers,
            initial_state,
            req.episode_context,
            artifacts,
            private_context=req.private_context,
            finalization_context=finalization_context,
        )
    except TaskEnvironmentError:
        raise
    except Exception as exc:
        logger.error(f"finalize_episode failed: {exc}")
        return TaskReward(
            value=0.0,
            success=False,
            breakdown={"error": str(exc)},
        )


def run_episode(rt: BackendRuntime, req: EpisodeRequest) -> EpisodeOutcome:
    cancellation = _build_cancellation(rt, req.episode_id, req.stop_event)
    try:
        harness = construct_harness(rt.config.harness)
        tracing = _build_recorder_and_genner(
            rt, req, cancellation.cancel_event, harness
        )
        harness_ctx = _build_harness_context(
            rt, req, tracing, cancellation.cancel_event
        )

        initial_state = rt.task.measure_initial_state(
            req.containers,
            req.episode_context,
            private_context=req.private_context,
        )

        partial = False
        error_message: str | None = None
        error_category: str | None = None
        transcript = None
        harness_raised = False
        inference_timed_out = False

        try:
            if harness_ctx.workflow is not None:
                transcript = harness.run_workflow(harness_ctx.workflow, harness_ctx)
            else:
                transcript = harness.run_episode(ctx=harness_ctx)
        except TaskEnvironmentError:
            raise
        except InferenceTimeoutError as exc:
            inference_timed_out = True
            logger.warning(f"Inference timeout: {exc}")
            partial = True
            error_category = "inference_timeout"
            error_message = str(exc)
        except HarnessError as exc:
            harness_raised = True
            logger.error(f"Harness error: {exc}")
            partial = True
            error_category = "harness_error"
            error_message = str(exc)

        deferred_environment_error = harness_ctx.extra.get("task_environment_error")
        if isinstance(deferred_environment_error, TaskEnvironmentError):
            raise deferred_environment_error

        external_cancelled = req.stop_event is not None and req.stop_event.is_set()
        if external_cancelled:
            partial = True
            reason = req.stop_reason.get() if req.stop_reason is not None else None
            error_message = f"cancelled: {reason or 'stop signal received'}"

        tracing.publish_telemetry()

        tool_calls_count = int(
            framework_telemetry_from_counters(
                snapshot_recorder_counters(tracing.recorder)
            )["tool_calls"]
        )
        recorder_labels = tracing.recorder.snapshot_labels()

        if transcript is None:
            artifacts = EpisodeArtifacts()
            llm_turns = 0
            termination_reason = error_message or "harness returned no transcript"
            termination_category = error_category or "harness_error"
            extra = _snapshot_harness_failure_extras(harness)
        else:
            artifacts = transcript.artifacts
            llm_turns = transcript.llm_turns
            termination_reason = transcript.termination_reason
            termination_category = transcript.termination_category
            transcript_partial, transcript_category, transcript_message = (
                categorize_termination(transcript)
            )
            if transcript_partial:
                partial = True
                error_category = transcript_category
                if not external_cancelled:
                    error_message = transcript_message
            extra = dict(transcript.extra)

        # Ledger is the authoritative source for llm_turns (transcript.llm_turns
        # may read from disk for container harnesses and return stale/0 values).
        if harness_ctx.budget_ledger is not None:
            llm_turns = harness_ctx.budget_ledger.snapshot().llm_turns_used

        last_step = extra.get("last_workflow_step")
        if not isinstance(last_step, str):
            last_step = recorder_labels.get("last_workflow_step")
            if last_step:
                extra["last_workflow_step"] = last_step

        budget_exhaustion = (
            harness_ctx.budget_ledger.exhausted()
            if harness_ctx.budget_ledger is not None
            else None
        )
        preserve_agent_failure = (
            termination_category == "agent_failure"
            and extra.get("agent_failure_kind") == "output_length_retries_exhausted"
        )
        if budget_exhaustion is not None and not preserve_agent_failure:
            termination_category = "budget_exhausted"
            termination_reason = f"budget exhausted: {budget_exhaustion.kind}"
            extra["budget_exhausted_kind"] = budget_exhaustion.kind
            if budget_exhaustion.step is not None:
                extra["budget_exhausted_step"] = budget_exhaustion.step

        if inference_timed_out:
            reward = TaskReward(
                value=0.0,
                success=False,
                breakdown={"error": error_message or "inference timed out"},
            )
        else:
            reward = _finalize_reward(
                rt,
                req,
                initial_state=initial_state,
                artifacts=artifacts,
                finalization_context=FinalizationContext(
                    budget_exhaustion=budget_exhaustion,
                    tool_calls_count=tool_calls_count,
                    last_workflow_step=last_step
                    if isinstance(last_step, str)
                    else None,
                    identity=req.identity,
                ),
            )

        score = reward.value
        success = reward.success
        reward_breakdown = reward.breakdown
        if (
            last_step
            and harness_ctx.constraints
            and last_step in harness_ctx.constraints.step_overrides
        ):
            success_constraints = harness_ctx.constraints.step_overrides[
                last_step
            ].success
        else:
            success_constraints = (
                harness_ctx.constraints.success if harness_ctx.constraints else None
            )
        if success and success_constraints is not None:
            min_calls = success_constraints.min_task_tool_calls_for_success
            if tool_calls_count < min_calls:
                logger.warning(
                    f"Episode {req.episode_id}: score={score} but "
                    f"tool_calls_count={tool_calls_count} < "
                    f"min_task_tool_calls_for_success={min_calls} "
                    "- forcing success=False"
                )
                success = False
            elif (
                success
                and success_constraints.terminal_capability_for_success is not None
                and not any(
                    inv.name == success_constraints.terminal_capability_for_success
                    for inv in artifacts.capability_invocations
                )
            ):
                logger.warning(
                    f"Episode {req.episode_id}: score={score} but terminal "
                    "capability "
                    f"{success_constraints.terminal_capability_for_success!r} "
                    "was never invoked - forcing success=False"
                )
                success = False

        task_commit: TaskCommitPreparation | None = None
        if _task_episode_commit_enabled(rt.task):
            if req.identity is None:
                raise RuntimeError(
                    "commit-enabled task requires a trusted episode identity"
                )
            constrained_reward = TaskReward(
                value=score,
                success=success,
                breakdown=reward_breakdown,
            )
            task_commit = rt.task.prepare_episode_commit(
                identity=req.identity,
                reward=constrained_reward,
            )
            if not isinstance(task_commit, TaskCommitPreparation):
                raise TypeError(
                    "prepare_episode_commit() must return TaskCommitPreparation"
                )
            if not isinstance(task_commit.reward, TaskReward):
                raise TypeError("task commit reward must be a TaskReward")
            if not task_commit.token or not task_commit.intent_digest:
                raise ValueError("task commit token and intent digest must be non-empty")
            if task_commit.reward.success and not success:
                raise RuntimeError(
                    "task commit preparation cannot override framework success constraints"
                )
            score = task_commit.reward.value
            success = task_commit.reward.success
            reward_breakdown = task_commit.reward.breakdown
        logger.info(f"Score: {score:.2f} {rt.task.metric_unit}")
        logger.info(f"Episode completed - Success: {success}")

        train_rows = records_to_rows(
            tracing.recorder.inference_records,
            run_id=rt.run_id,
            version=get_formatted_repo_info(),
        )
        prompt_responses = to_prompt_responses(train_rows)
        trajectory = to_trajectory(
            episode_id=req.episode_id,
            llm_turns=llm_turns,
            termination_reason=termination_reason,
            termination_category=termination_category,
            extra=extra,
        )
        return EpisodeOutcome(
            episode_id=req.episode_id,
            score=score,
            success=success,
            partial=partial,
            llm_turns_count=llm_turns,
            train_rows=train_rows,
            prompt_responses=prompt_responses,
            trajectory=trajectory,
            error_message=error_message,
            error_category=error_category,
            reward_breakdown=reward_breakdown,
            harness_error=harness_raised,
            tool_calls_count=tool_calls_count,
            task_commit_token=task_commit.token if task_commit is not None else None,
            task_commit_digest=(
                task_commit.intent_digest if task_commit is not None else None
            ),
        )
    finally:
        _stop_cancellation(cancellation)


__all__ = ["EpisodeOutcome", "EpisodeRequest", "run_episode"]
