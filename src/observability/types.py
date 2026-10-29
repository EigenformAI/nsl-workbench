from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class UsageInfo:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class InferenceResult:
    content: str
    usage: Optional[UsageInfo] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    reasoning_content: Optional[str] = None


@dataclass
class InferenceMetric:
    inference_id: str
    run_id: str
    backend: str
    phase: str
    success: bool
    episode_id: Optional[str] = None
    content: Optional[str] = None
    error_message: Optional[str] = None
    usage: Optional[UsageInfo] = None
    model: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens_per_second: Optional[float] = None
    output_tokens_per_second: Optional[float] = None
    total_tokens_per_second: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    host_memory_mb: Optional[float] = None
    # Phase 2: vLLM Prometheus metrics (populated when scraping is enabled)
    kv_cache_usage_pct: Optional[float] = None
    num_requests_running: Optional[int] = None
    num_requests_waiting: Optional[int] = None


@dataclass
class PhaseMetric:
    phase_name: str
    run_id: str
    duration_ms: float
    success: bool
    episode_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class ResourceSnapshot:
    gpu_memory_mb: Optional[float] = None
    host_memory_mb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    cpu_utilization_pct: Optional[float] = None


@dataclass
class UtilizationSummary:
    peak_gpu_utilization_pct: Optional[float] = None
    peak_cpu_utilization_pct: Optional[float] = None
    avg_gpu_utilization_pct: Optional[float] = None
    avg_cpu_utilization_pct: Optional[float] = None
    sample_count: int = 0
    # vLLM Prometheus metrics (populated when scraping is enabled)
    peak_kv_cache_usage_pct: Optional[float] = None
    avg_kv_cache_usage_pct: Optional[float] = None
    peak_num_requests_running: Optional[int] = None
    peak_num_requests_waiting: Optional[int] = None
    # Windowed counter deltas (preemptions, prefix-cache hit ratio, token
    # throughput) over the sampling window — ``asdict(InferenceMetricsDelta)``.
    # ``None`` when fewer than two inference scrapes were collected.
    inference_metrics_delta: Optional[Dict[str, Any]] = None


@dataclass
class LiveUtilizationSnapshot:
    """Read-only peek at current utilization state (no reset)."""

    avg_gpu_utilization_pct: Optional[float] = None
    avg_cpu_utilization_pct: Optional[float] = None
    peak_gpu_utilization_pct: Optional[float] = None
    peak_cpu_utilization_pct: Optional[float] = None
    avg_kv_cache_usage_pct: Optional[float] = None
    avg_output_tokens_per_second: Optional[float] = None
    sample_count: int = 0
