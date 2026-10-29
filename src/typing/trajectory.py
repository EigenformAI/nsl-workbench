from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Dict, List, Optional

from src.harness.training_row_adapter import enrich_training_rows_for_episode
from src.training_data.transforms import EpisodeTrainingRows


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


@dataclass
class EpisodeTrajectory:
    episode_id: str
    generation_id: int
    episode_index: int
    prompt_responses: List[Dict[str, Any]]
    trajectory: Dict[str, Any]
    score: float
    episode_runtime_success: bool
    success: bool
    llm_turns_count: int
    container_variation: str
    started_at: str
    completed_at: str
    duration_seconds: float
    metric_name: str = ""
    metric_unit: str = ""
    higher_is_better: bool = True
    partial: bool = False
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    task_breakdown: Dict[str, Any] = field(default_factory=dict)
    container_overhead_seconds: Optional[float] = None
    episode_execution_seconds: Optional[float] = None
    total_inference_ms: Optional[float] = None
    inference_call_count: Optional[int] = None
    average_output_tokens_per_second: Optional[float] = None
    inference_duty_cycle: Optional[float] = None
    peak_gpu_utilization_pct: Optional[float] = None
    peak_cpu_utilization_pct: Optional[float] = None
    avg_gpu_utilization_pct: Optional[float] = None
    avg_cpu_utilization_pct: Optional[float] = None
    peak_kv_cache_usage_pct: Optional[float] = None
    avg_kv_cache_usage_pct: Optional[float] = None
    peak_num_requests_running: Optional[int] = None
    peak_num_requests_waiting: Optional[int] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    peak_context_tokens: Optional[int] = None
    avg_context_tokens: Optional[float] = None
    median_context_tokens: Optional[float] = None
    tool_calls_count: int = 0
    raw_training_rows: List[Dict[str, Any]] = field(default_factory=list)
    task_commit_token: Optional[str] = None
    task_commit_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EpisodeTrajectory":
        return cls(
            episode_id=str(payload["episode_id"]),
            generation_id=int(payload["generation_id"]),
            episode_index=int(payload["episode_index"]),
            prompt_responses=list(payload.get("prompt_responses", [])),
            trajectory=dict(payload.get("trajectory", {})),
            score=float(payload.get("score", payload.get("space_freed_kb", 0.0))),
            metric_name=str(payload.get("metric_name", "")),
            metric_unit=str(payload.get("metric_unit", "")),
            higher_is_better=bool(payload.get("higher_is_better", True)),
            episode_runtime_success=bool(payload.get("episode_runtime_success", False)),
            success=bool(payload.get("success", False)),
            llm_turns_count=int(payload.get("llm_turns_count", 0)),
            container_variation=str(payload.get("container_variation", "")),
            started_at=str(payload.get("started_at", "")),
            completed_at=str(payload.get("completed_at", "")),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            partial=bool(payload.get("partial", False)),
            error_message=payload.get("error_message"),
            error_category=payload.get("error_category"),
            task_breakdown=payload.get(
                "task_breakdown",
                {
                    k: payload[k]
                    for k in (
                        "space_measurements",
                        "measurement_errors",
                    )
                    if k in payload
                },
            ),
            container_overhead_seconds=_optional_float(
                payload.get("container_overhead_seconds")
            ),
            episode_execution_seconds=_optional_float(
                payload.get("episode_execution_seconds")
            ),
            total_inference_ms=_optional_float(payload.get("total_inference_ms")),
            inference_call_count=_optional_int(payload.get("inference_call_count")),
            average_output_tokens_per_second=_optional_float(
                payload.get("average_output_tokens_per_second")
            ),
            inference_duty_cycle=_optional_float(payload.get("inference_duty_cycle")),
            peak_gpu_utilization_pct=_optional_float(
                payload.get("peak_gpu_utilization_pct")
            ),
            peak_cpu_utilization_pct=_optional_float(
                payload.get("peak_cpu_utilization_pct")
            ),
            avg_gpu_utilization_pct=_optional_float(
                payload.get("avg_gpu_utilization_pct")
            ),
            avg_cpu_utilization_pct=_optional_float(
                payload.get("avg_cpu_utilization_pct")
            ),
            peak_kv_cache_usage_pct=_optional_float(
                payload.get("peak_kv_cache_usage_pct")
            ),
            avg_kv_cache_usage_pct=_optional_float(
                payload.get("avg_kv_cache_usage_pct")
            ),
            peak_num_requests_running=_optional_int(
                payload.get("peak_num_requests_running")
            ),
            peak_num_requests_waiting=_optional_int(
                payload.get("peak_num_requests_waiting")
            ),
            total_input_tokens=_optional_int(payload.get("total_input_tokens")),
            total_output_tokens=_optional_int(payload.get("total_output_tokens")),
            peak_context_tokens=_optional_int(payload.get("peak_context_tokens")),
            avg_context_tokens=_optional_float(payload.get("avg_context_tokens")),
            median_context_tokens=_optional_float(payload.get("median_context_tokens")),
            tool_calls_count=int(payload.get("tool_calls_count", 0)),
            raw_training_rows=list(payload.get("raw_training_rows", [])),
            task_commit_token=payload.get("task_commit_token"),
            task_commit_digest=payload.get("task_commit_digest"),
        )


@dataclass
class GenerationData:
    generation_id: int
    all_episodes: List[EpisodeTrajectory] = field(default_factory=list)
    successful_episodes: List[EpisodeTrajectory] = field(default_factory=list)
    failed_episodes: List[EpisodeTrajectory] = field(default_factory=list)
    total_episodes_run: int = 0
    total_successful: int = 0
    raw_successful_row_count: int = 0
    training_row_count: int = 0
    training_row_count_is_exact: bool = True
    training_row_count_last_refreshed_episode: int = 0
    total_score: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    termination_reason: Optional[str] = None

    def add_episode(self, episode: EpisodeTrajectory) -> None:
        self.all_episodes.append(episode)
        self.total_episodes_run += 1
        if episode.success:
            self.successful_episodes.append(episode)
            self.total_successful += 1
            raw_rows = episode.raw_training_rows or episode.prompt_responses
            self.raw_successful_row_count += len(raw_rows)
            self.training_row_count += len(raw_rows)
            self.total_score += episode.score
        else:
            self.failed_episodes.append(episode)

    def set_training_row_count(
        self,
        count: int,
        *,
        is_exact: bool = True,
        last_refreshed_episode: int | None = None,
    ) -> None:
        self.training_row_count = int(count)
        self.training_row_count_is_exact = bool(is_exact)
        if last_refreshed_episode is None:
            if is_exact:
                self.training_row_count_last_refreshed_episode = self.total_episodes_run
        else:
            self.training_row_count_last_refreshed_episode = int(last_refreshed_episode)

    def mark_training_row_count_stale(self) -> None:
        self.training_row_count_is_exact = False

    def get_successful_training_row_groups(self) -> list[EpisodeTrainingRows]:
        groups: list[EpisodeTrainingRows] = []
        for episode in sorted(
            self.successful_episodes,
            key=lambda item: (
                item.episode_index if item.episode_index is not None else float("inf")
            ),
        ):
            source_rows = episode.raw_training_rows or episode.prompt_responses
            rows = enrich_training_rows_for_episode(
                list(source_rows),
                episode_id=episode.episode_id,
                episode_index=episode.episode_index,
                generation_id=episode.generation_id,
                episode_score=episode.score,
            )
            groups.append(
                EpisodeTrainingRows(
                    episode_id=episode.episode_id,
                    episode_index=episode.episode_index,
                    generation_id=episode.generation_id,
                    episode_score=episode.score,
                    rows=rows,
                )
            )
        return groups

    @property
    def success_rate(self) -> float:
        if self.total_episodes_run == 0:
            return 0.0
        return self.total_successful / self.total_episodes_run

    def to_metadata_dict(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "generation_id": self.generation_id,
            "total_episodes_run": self.total_episodes_run,
            "total_successful": self.total_successful,
            "raw_successful_row_count": self.raw_successful_row_count,
            "training_row_count": self.training_row_count,
            "training_row_count_is_exact": self.training_row_count_is_exact,
            "training_row_count_last_refreshed_episode": (
                self.training_row_count_last_refreshed_episode
            ),
            "target_count_basis": "training_rows",
            "total_score": self.total_score,
            "success_rate": self.success_rate,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_agent_seconds": sum(
                episode.duration_seconds for episode in self.all_episodes
            ),
        }
        if run_id is not None:
            payload["run_id"] = run_id
        if self.termination_reason is not None:
            payload["termination_reason"] = self.termination_reason

        if self.started_at and self.completed_at:
            started_at = datetime.fromisoformat(self.started_at)
            completed_at = datetime.fromisoformat(self.completed_at)
            elapsed_seconds = (completed_at - started_at).total_seconds()
            payload["generation_wall_clock_seconds"] = elapsed_seconds
            if elapsed_seconds > 0:
                elapsed_hours = elapsed_seconds / 3600
                elapsed_minutes = elapsed_seconds / 60
                payload["episodes_per_hour"] = self.total_episodes_run / elapsed_hours
                payload["training_rows_per_hour"] = (
                    self.training_row_count / elapsed_hours
                )
                payload["episodes_per_minute"] = (
                    self.total_episodes_run / elapsed_minutes
                )
                payload["training_rows_per_minute"] = (
                    self.training_row_count / elapsed_minutes
                )

        episodes_with_output_rate = [
            episode.average_output_tokens_per_second
            for episode in self.all_episodes
            if episode.average_output_tokens_per_second is not None
        ]
        if episodes_with_output_rate:
            payload["average_output_tokens_per_second"] = sum(
                episodes_with_output_rate
            ) / len(episodes_with_output_rate)

        episodes_with_duty_cycle = [
            episode.inference_duty_cycle
            for episode in self.all_episodes
            if episode.inference_duty_cycle is not None
        ]
        if episodes_with_duty_cycle:
            payload["average_inference_duty_cycle"] = sum(
                episodes_with_duty_cycle
            ) / len(episodes_with_duty_cycle)

        episode_peak_gpu = [
            episode.peak_gpu_utilization_pct
            for episode in self.all_episodes
            if episode.peak_gpu_utilization_pct is not None
        ]
        if episode_peak_gpu:
            payload["peak_gpu_utilization_pct"] = max(episode_peak_gpu)

        episode_peak_cpu = [
            episode.peak_cpu_utilization_pct
            for episode in self.all_episodes
            if episode.peak_cpu_utilization_pct is not None
        ]
        if episode_peak_cpu:
            payload["peak_cpu_utilization_pct"] = max(episode_peak_cpu)

        episode_avg_gpu = [
            episode.avg_gpu_utilization_pct
            for episode in self.all_episodes
            if episode.avg_gpu_utilization_pct is not None
        ]
        if episode_avg_gpu:
            payload["average_gpu_utilization_pct"] = sum(episode_avg_gpu) / len(
                episode_avg_gpu
            )

        episode_avg_cpu = [
            episode.avg_cpu_utilization_pct
            for episode in self.all_episodes
            if episode.avg_cpu_utilization_pct is not None
        ]
        if episode_avg_cpu:
            payload["average_cpu_utilization_pct"] = sum(episode_avg_cpu) / len(
                episode_avg_cpu
            )

        episode_peak_kv = [
            episode.peak_kv_cache_usage_pct
            for episode in self.all_episodes
            if episode.peak_kv_cache_usage_pct is not None
        ]
        if episode_peak_kv:
            payload["peak_kv_cache_usage_pct"] = max(episode_peak_kv)

        episode_avg_kv = [
            episode.avg_kv_cache_usage_pct
            for episode in self.all_episodes
            if episode.avg_kv_cache_usage_pct is not None
        ]
        if episode_avg_kv:
            payload["avg_kv_cache_usage_pct"] = sum(episode_avg_kv) / len(
                episode_avg_kv
            )

        episode_peak_requests_running = [
            episode.peak_num_requests_running
            for episode in self.all_episodes
            if episode.peak_num_requests_running is not None
        ]
        if episode_peak_requests_running:
            payload["peak_num_requests_running"] = max(episode_peak_requests_running)

        episode_peak_requests_waiting = [
            episode.peak_num_requests_waiting
            for episode in self.all_episodes
            if episode.peak_num_requests_waiting is not None
        ]
        if episode_peak_requests_waiting:
            payload["peak_num_requests_waiting"] = max(episode_peak_requests_waiting)

        container_overheads = [
            episode.container_overhead_seconds
            for episode in self.all_episodes
            if episode.container_overhead_seconds is not None
        ]
        if container_overheads:
            payload["total_container_overhead_seconds"] = sum(container_overheads)

        inference_seconds = [
            episode.total_inference_ms / 1000
            for episode in self.all_episodes
            if episode.total_inference_ms is not None
        ]
        if inference_seconds:
            payload["total_inference_seconds"] = sum(inference_seconds)

        episode_execution_seconds = [
            episode.episode_execution_seconds
            for episode in self.all_episodes
            if episode.episode_execution_seconds is not None
        ]
        if episode_execution_seconds:
            payload["total_episode_execution_seconds"] = sum(episode_execution_seconds)

        total_input_tokens = [
            episode.total_input_tokens
            for episode in self.all_episodes
            if episode.total_input_tokens is not None
        ]
        total_output_tokens = [
            episode.total_output_tokens
            for episode in self.all_episodes
            if episode.total_output_tokens is not None
        ]
        if total_input_tokens or total_output_tokens:
            payload["total_input_tokens"] = sum(total_input_tokens)
            payload["total_output_tokens"] = sum(total_output_tokens)
            payload["total_tokens"] = (
                payload["total_input_tokens"] + payload["total_output_tokens"]
            )
            if self.total_successful > 0:
                payload["tokens_per_successful_episode"] = (
                    payload["total_tokens"] / self.total_successful
                )

        peak_context_tokens = [
            episode.peak_context_tokens
            for episode in self.all_episodes
            if episode.peak_context_tokens is not None
        ]
        if peak_context_tokens:
            payload["peak_context_tokens"] = max(peak_context_tokens)

        avg_context_tokens = [
            episode.avg_context_tokens
            for episode in self.all_episodes
            if episode.avg_context_tokens is not None
        ]
        if avg_context_tokens:
            payload["avg_context_tokens"] = sum(avg_context_tokens) / len(
                avg_context_tokens
            )

        median_context_tokens = [
            episode.median_context_tokens
            for episode in self.all_episodes
            if episode.median_context_tokens is not None
        ]
        if median_context_tokens:
            payload["median_context_tokens"] = median(median_context_tokens)

        return payload
