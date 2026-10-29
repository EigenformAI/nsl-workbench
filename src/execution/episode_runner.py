from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
import threading
from typing import Any, Optional

from src.execution.backend_runtime import BackendRuntime, _coerce_runtime
from src.execution.episode import EpisodeRequest, run_episode
from src.harness.training_row_adapter import enrich_training_rows_for_episode
from src.observability.types import UtilizationSummary
from src.task.base import TaskEnvironmentError
from src.task.types import TrustedEpisodeIdentity
from src.typing.config import AppConfig
from src.typing.trajectory import EpisodeTrajectory


@dataclass
class EpisodeInferenceMetrics:
    total_inference_ms: float = 0.0
    inference_call_count: int = 0
    average_output_tokens_per_second: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def _get_inference_metrics_for_episode(
    metrics_collector: Any,
    episode_id: str,
) -> list[Any]:
    if metrics_collector is None:
        return []

    get_method = getattr(metrics_collector, "get_metrics_for_episode", None)
    if get_method is not None and not isinstance(
        getattr(get_method, "_mock_name", None), str
    ):
        return get_method(episode_id)

    lock = getattr(metrics_collector, "_lock", None)
    if lock is None:
        inference_metrics = list(getattr(metrics_collector, "inference_metrics", []))
    else:
        with lock:
            inference_metrics = list(
                getattr(metrics_collector, "inference_metrics", [])
            )

    return [metric for metric in inference_metrics if metric.episode_id == episode_id]


def _compute_episode_inference_metrics(
    metrics_collector: Any,
    episode_id: str,
) -> EpisodeInferenceMetrics:
    episode_metrics = _get_inference_metrics_for_episode(metrics_collector, episode_id)
    total_inference_ms = sum(metric.latency_ms for metric in episode_metrics)
    inference_call_count = len(episode_metrics)
    total_input_tokens = sum(
        metric.usage.prompt_tokens or 0
        for metric in episode_metrics
        if metric.usage is not None
    )
    total_output_tokens = sum(
        metric.usage.completion_tokens or 0
        for metric in episode_metrics
        if metric.usage is not None
    )
    if total_inference_ms <= 0:
        return EpisodeInferenceMetrics(
            total_inference_ms=total_inference_ms,
            inference_call_count=inference_call_count,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

    return EpisodeInferenceMetrics(
        total_inference_ms=total_inference_ms,
        inference_call_count=inference_call_count,
        average_output_tokens_per_second=(
            total_output_tokens / (total_inference_ms / 1000)
        ),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


def _compute_context_length_stats(
    metrics_collector: Any,
    episode_id: str,
) -> tuple[Optional[int], Optional[float], Optional[float]]:
    prompt_tokens = [
        metric.usage.prompt_tokens
        for metric in _get_inference_metrics_for_episode(metrics_collector, episode_id)
        if metric.usage is not None and metric.usage.prompt_tokens is not None
    ]
    if not prompt_tokens:
        return None, None, None
    prompt_tokens = list(prompt_tokens)
    prompt_tokens.sort()
    midpoint = len(prompt_tokens) // 2
    median = (
        float(prompt_tokens[midpoint])
        if len(prompt_tokens) % 2 == 1
        else float(prompt_tokens[midpoint - 1] + prompt_tokens[midpoint]) / 2
    )
    return (
        max(prompt_tokens),
        sum(prompt_tokens) / len(prompt_tokens),
        median,
    )


def _resolve_metrics_episode_id(
    metrics_collector: Any,
    preferred_episode_id: str,
    fallback_episode_id: Optional[str] = None,
) -> str:
    if fallback_episode_id is None or fallback_episode_id == preferred_episode_id:
        return preferred_episode_id

    if _get_inference_metrics_for_episode(metrics_collector, preferred_episode_id):
        return preferred_episode_id
    if _get_inference_metrics_for_episode(metrics_collector, fallback_episode_id):
        return fallback_episode_id
    return preferred_episode_id


def _stop_utilization_sampling(
    metrics_collector: Any,
    utilization_sampling_started: bool,
) -> UtilizationSummary:
    if utilization_sampling_started and metrics_collector is not None:
        return metrics_collector.stop_utilization_sampling()
    return UtilizationSummary()


def _run_single_episode(
    rt: BackendRuntime,
    *,
    container_manager: Any,
    generation_id: int,
    episode_index: int,
    variation_index: int,
    population_outcome: Any = None,
    verified: bool = True,
    parallel_episodes: int | None = None,
    stop_event: threading.Event | None = None,
    stop_reason: Any = None,
    variation: Any = None,
    telemetry_observer: Any = None,
    harness_session: dict[str, Any] | None = None,
) -> EpisodeTrajectory:
    selected_variation = variation
    if selected_variation is None:
        variations = rt.task.list_variations()
        selected_variation = variations[variation_index % len(variations)]

    episode_context = dict(
        population_outcome.episode_context if population_outcome else {}
    )
    private_context = (
        dict(population_outcome.private_context)
        if population_outcome and population_outcome.private_context is not None
        else None
    )

    started_at = datetime.now().isoformat()
    generation_config = rt.config.generation or AppConfig.GenerationConfig()
    active_parallel_episodes = (
        parallel_episodes
        if parallel_episodes is not None
        else generation_config.parallel_episodes
    )
    episode_id = f"ep_gen{generation_id}_{episode_index:04d}_{int(time.time())}"
    identity = TrustedEpisodeIdentity(
        run_id=rt.run_id,
        generation_id=generation_id,
        episode_index=episode_index,
        episode_id=episode_id,
    )

    metrics_collector = rt.metrics
    utilization_sampling_started = False
    if metrics_collector is not None and active_parallel_episodes <= 1:
        metrics_collector.start_utilization_sampling(
            inference_metrics_url=metrics_collector.inference_metrics_url,
            inference_metrics_backend=metrics_collector.inference_metrics_backend,
            inference_metrics_api_key=metrics_collector.inference_metrics_api_key,
        )
        utilization_sampling_started = True

    container_started_at = time.perf_counter()
    if population_outcome is None:
        try:
            population_outcome, verified = container_manager.populate_with_task(
                container_manager.get_containers(),
                selected_variation,
            )
            episode_context = dict(population_outcome.episode_context)
            private_context = (
                dict(population_outcome.private_context)
                if population_outcome.private_context is not None
                else None
            )
        except Exception:
            _stop_utilization_sampling(metrics_collector, utilization_sampling_started)
            raise

    if not population_outcome.results:
        _stop_utilization_sampling(metrics_collector, utilization_sampling_started)
        raise RuntimeError("populate returned no results")

    primary_population = population_outcome.results[0]
    variation_name = primary_population.variation_name
    if not verified:
        container_overhead_seconds = time.perf_counter() - container_started_at
        utilization_summary = _stop_utilization_sampling(
            metrics_collector, utilization_sampling_started
        )
        episode_inference_metrics = _compute_episode_inference_metrics(
            metrics_collector, episode_id
        )
        peak_context_tokens, avg_context_tokens, median_context_tokens = (
            _compute_context_length_stats(metrics_collector, episode_id)
        )
        completed_at = datetime.now().isoformat()
        return EpisodeTrajectory(
            episode_id=episode_id,
            generation_id=generation_id,
            episode_index=episode_index,
            prompt_responses=[],
            trajectory={},
            score=0.0,
            episode_runtime_success=False,
            success=False,
            llm_turns_count=0,
            container_variation=variation_name,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=container_overhead_seconds,
            error_message="container population verification failed",
            container_overhead_seconds=container_overhead_seconds,
            episode_execution_seconds=0.0,
            total_inference_ms=episode_inference_metrics.total_inference_ms,
            inference_call_count=episode_inference_metrics.inference_call_count,
            average_output_tokens_per_second=(
                episode_inference_metrics.average_output_tokens_per_second
            ),
            inference_duty_cycle=0.0,
            peak_gpu_utilization_pct=utilization_summary.peak_gpu_utilization_pct,
            peak_cpu_utilization_pct=utilization_summary.peak_cpu_utilization_pct,
            avg_gpu_utilization_pct=utilization_summary.avg_gpu_utilization_pct,
            avg_cpu_utilization_pct=utilization_summary.avg_cpu_utilization_pct,
            peak_kv_cache_usage_pct=utilization_summary.peak_kv_cache_usage_pct,
            avg_kv_cache_usage_pct=utilization_summary.avg_kv_cache_usage_pct,
            peak_num_requests_running=utilization_summary.peak_num_requests_running,
            peak_num_requests_waiting=utilization_summary.peak_num_requests_waiting,
            total_input_tokens=episode_inference_metrics.total_input_tokens,
            total_output_tokens=episode_inference_metrics.total_output_tokens,
            peak_context_tokens=peak_context_tokens,
            avg_context_tokens=avg_context_tokens,
            median_context_tokens=median_context_tokens,
        )

    container_overhead_seconds = time.perf_counter() - container_started_at
    execution_started_at = time.perf_counter()
    try:
        outcome = run_episode(
            rt,
            EpisodeRequest(
                episode_id=episode_id,
                containers=container_manager.get_containers(),
                container_manager=container_manager,
                agent_container=container_manager.get_service(
                    rt.task.agent_service_name
                ),
                variation=selected_variation,
                episode_context=episode_context,
                private_context=private_context,
                harness_session=harness_session,
                stop_event=stop_event,
                stop_reason=stop_reason,
                telemetry_observer=telemetry_observer,
                identity=identity,
            ),
        )
    except TaskEnvironmentError:
        _stop_utilization_sampling(metrics_collector, utilization_sampling_started)
        raise
    except Exception as exc:
        episode_execution_seconds = time.perf_counter() - execution_started_at
        utilization_summary = _stop_utilization_sampling(
            metrics_collector, utilization_sampling_started
        )
        episode_inference_metrics = _compute_episode_inference_metrics(
            metrics_collector, episode_id
        )
        peak_context_tokens, avg_context_tokens, median_context_tokens = (
            _compute_context_length_stats(metrics_collector, episode_id)
        )
        completed_at = datetime.now().isoformat()
        duration_seconds = container_overhead_seconds + episode_execution_seconds
        return EpisodeTrajectory(
            episode_id=episode_id,
            generation_id=generation_id,
            episode_index=episode_index,
            prompt_responses=[],
            trajectory={},
            score=0.0,
            episode_runtime_success=False,
            success=False,
            llm_turns_count=0,
            container_variation=variation_name,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            error_message=str(exc),
            container_overhead_seconds=container_overhead_seconds,
            episode_execution_seconds=episode_execution_seconds,
            total_inference_ms=episode_inference_metrics.total_inference_ms,
            inference_call_count=episode_inference_metrics.inference_call_count,
            average_output_tokens_per_second=(
                episode_inference_metrics.average_output_tokens_per_second
            ),
            inference_duty_cycle=(
                (episode_inference_metrics.total_inference_ms / 1000) / duration_seconds
                if duration_seconds > 0
                and episode_inference_metrics.total_inference_ms > 0
                else 0.0
            ),
            peak_gpu_utilization_pct=utilization_summary.peak_gpu_utilization_pct,
            peak_cpu_utilization_pct=utilization_summary.peak_cpu_utilization_pct,
            avg_gpu_utilization_pct=utilization_summary.avg_gpu_utilization_pct,
            avg_cpu_utilization_pct=utilization_summary.avg_cpu_utilization_pct,
            peak_kv_cache_usage_pct=utilization_summary.peak_kv_cache_usage_pct,
            avg_kv_cache_usage_pct=utilization_summary.avg_kv_cache_usage_pct,
            peak_num_requests_running=utilization_summary.peak_num_requests_running,
            peak_num_requests_waiting=utilization_summary.peak_num_requests_waiting,
            total_input_tokens=episode_inference_metrics.total_input_tokens,
            total_output_tokens=episode_inference_metrics.total_output_tokens,
            peak_context_tokens=peak_context_tokens,
            avg_context_tokens=avg_context_tokens,
            median_context_tokens=median_context_tokens,
        )

    episode_execution_seconds = time.perf_counter() - execution_started_at
    utilization_summary = _stop_utilization_sampling(
        metrics_collector, utilization_sampling_started
    )
    completed_at = datetime.now().isoformat()
    metrics_episode_id = _resolve_metrics_episode_id(
        metrics_collector,
        outcome.episode_id,
        fallback_episode_id=episode_id,
    )
    episode_inference_metrics = _compute_episode_inference_metrics(
        metrics_collector, metrics_episode_id
    )
    peak_context_tokens, avg_context_tokens, median_context_tokens = (
        _compute_context_length_stats(metrics_collector, metrics_episode_id)
    )
    duration_seconds = container_overhead_seconds + episode_execution_seconds
    return EpisodeTrajectory(
        episode_id=outcome.episode_id,
        generation_id=generation_id,
        episode_index=episode_index,
        prompt_responses=outcome.prompt_responses,
        trajectory=outcome.trajectory.copy(),
        score=outcome.score,
        episode_runtime_success=outcome.success,
        success=outcome.success,
        llm_turns_count=outcome.llm_turns_count,
        tool_calls_count=outcome.tool_calls_count,
        container_variation=variation_name,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        metric_name=rt.task.metric_name,
        metric_unit=rt.task.metric_unit,
        higher_is_better=rt.task.higher_is_better,
        partial=outcome.partial,
        error_message=outcome.error_message,
        error_category=outcome.error_category,
        task_breakdown=outcome.reward_breakdown,
        container_overhead_seconds=container_overhead_seconds,
        episode_execution_seconds=episode_execution_seconds,
        total_inference_ms=episode_inference_metrics.total_inference_ms,
        inference_call_count=episode_inference_metrics.inference_call_count,
        average_output_tokens_per_second=(
            episode_inference_metrics.average_output_tokens_per_second
        ),
        inference_duty_cycle=(
            (episode_inference_metrics.total_inference_ms / 1000) / duration_seconds
            if duration_seconds > 0 and episode_inference_metrics.total_inference_ms > 0
            else 0.0
        ),
        peak_gpu_utilization_pct=utilization_summary.peak_gpu_utilization_pct,
        peak_cpu_utilization_pct=utilization_summary.peak_cpu_utilization_pct,
        avg_gpu_utilization_pct=utilization_summary.avg_gpu_utilization_pct,
        avg_cpu_utilization_pct=utilization_summary.avg_cpu_utilization_pct,
        peak_kv_cache_usage_pct=utilization_summary.peak_kv_cache_usage_pct,
        avg_kv_cache_usage_pct=utilization_summary.avg_kv_cache_usage_pct,
        peak_num_requests_running=utilization_summary.peak_num_requests_running,
        peak_num_requests_waiting=utilization_summary.peak_num_requests_waiting,
        total_input_tokens=episode_inference_metrics.total_input_tokens,
        total_output_tokens=episode_inference_metrics.total_output_tokens,
        peak_context_tokens=peak_context_tokens,
        avg_context_tokens=avg_context_tokens,
        median_context_tokens=median_context_tokens,
        raw_training_rows=enrich_training_rows_for_episode(
            outcome.train_rows,
            episode_id=outcome.episode_id,
            episode_index=episode_index,
            generation_id=generation_id,
            episode_score=outcome.score,
        ),
        task_commit_token=outcome.task_commit_token,
        task_commit_digest=outcome.task_commit_digest,
    )


def run_single_episode(*args: Any, **kwargs: Any) -> EpisodeTrajectory:
    if args and isinstance(args[0], BackendRuntime):
        rt = args[0]
        return _run_single_episode(rt, **kwargs)

    genner = kwargs.pop("genner", args[0] if len(args) > 0 else None)
    docker_client = kwargs.pop("docker_client", args[1] if len(args) > 1 else None)
    container_manager = kwargs.pop(
        "container_manager", args[2] if len(args) > 2 else None
    )
    config = kwargs.pop("config", args[3] if len(args) > 3 else None)
    generation_id = kwargs.pop("generation_id", args[4] if len(args) > 4 else None)
    episode_index = kwargs.pop("episode_index", args[5] if len(args) > 5 else None)
    variation_index = kwargs.pop("variation_index", args[6] if len(args) > 6 else None)
    run_id = kwargs.pop("run_id", args[7] if len(args) > 7 else None)
    metrics_collector = kwargs.pop("metrics_collector", None)
    task = kwargs.pop("task", None)

    if any(
        value is None
        for value in (
            genner,
            docker_client,
            container_manager,
            config,
            generation_id,
            episode_index,
            variation_index,
            run_id,
            task,
        )
    ):
        raise TypeError("run_single_episode missing required legacy arguments")

    rt = _coerce_runtime(
        config=config,
        run_id=run_id,
        task=task,
        genner=genner,
        docker_client=docker_client,
        metrics=metrics_collector,
    )
    return _run_single_episode(
        rt,
        container_manager=container_manager,
        generation_id=int(generation_id),
        episode_index=int(episode_index),
        variation_index=int(variation_index),
        **kwargs,
    )


__all__ = [
    "EpisodeInferenceMetrics",
    "_compute_context_length_stats",
    "run_single_episode",
]
