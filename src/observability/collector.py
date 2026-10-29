import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from loguru import logger

from src.observability.types import (
    InferenceMetric,
    LiveUtilizationSnapshot,
    PhaseMetric,
    ResourceSnapshot,
    UtilizationSummary,
)

from src.observability.gpu import read_gpu_memory_info, read_gpu_utilization_pct
from src.observability.vllm_metrics import (
    InferenceMetricsSnapshot,
    inference_metrics_delta,
    snapshot_inference_metrics,
    snapshot_vllm_metrics,
)

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

if TYPE_CHECKING:
    from src.typing.config import AppConfig


class MetricsCollector:
    def __init__(
        self,
        run_id: str,
        output_dir: str | Path,
        enabled: bool = True,
        record_inference: bool = True,
        record_phases: bool = True,
        record_resources: bool = True,
    ) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.record_inference_enabled = record_inference
        self.record_phases_enabled = record_phases
        self.record_resources_enabled = record_resources
        self.inference_metrics: list[InferenceMetric] = []
        self.phase_metrics: list[PhaseMetric] = []
        self._lock = threading.Lock()
        # Background utilization sampling state
        self._sampling_thread: threading.Thread | None = None
        self._sampling_stop_event: threading.Event | None = None
        self._utilization_peak_gpu: float | None = None
        self._utilization_peak_cpu: float | None = None
        self._utilization_sum_gpu: float = 0.0
        self._utilization_sum_cpu: float = 0.0
        self._utilization_gpu_count: int = 0
        self._utilization_cpu_count: int = 0
        self._utilization_sample_count: int = 0
        # Inference-server Prometheus scraping state. vLLM names are retained
        # internally for output compatibility with existing summaries.
        self.inference_metrics_url: str | None = None
        self.inference_metrics_backend: Literal["vllm", "sglang"] = "vllm"
        self.inference_metrics_api_key: str | None = None
        self.vllm_metrics_url: str | None = None
        self._inference_metrics_url: str | None = None
        self._inference_metrics_backend: Literal["vllm", "sglang"] = "vllm"
        self._inference_metrics_api_key: str | None = None
        self._vllm_peak_kv_cache: float | None = None
        self._vllm_sum_kv_cache: float = 0.0
        self._vllm_kv_cache_count: int = 0
        self._vllm_peak_requests_running: int | None = None
        self._vllm_peak_requests_waiting: int | None = None
        # First and latest counter snapshots within the active sampling window,
        # used to compute windowed deltas (preemptions, prefix-cache hit ratio).
        self._vllm_first_snapshot: InferenceMetricsSnapshot | None = None
        self._vllm_last_snapshot: InferenceMetricsSnapshot | None = None
        self._summary = {
            "inference_calls": 0,
            "phase_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_latency_ms": 0.0,
            "generation_count": 0,
            "_prompt_tokens_per_second_sum": 0.0,
            "_output_tokens_per_second_sum": 0.0,
            "_total_tokens_per_second_sum": 0.0,
        }

    @classmethod
    def from_config(cls, config: "AppConfig", run_id: str) -> "MetricsCollector":
        observability = config.observability
        output_dir = observability.metrics_output_path or config.train_data_save_folder
        return cls(
            run_id=run_id,
            output_dir=output_dir,
            enabled=observability.enabled,
            record_inference=observability.record_inference,
            record_phases=observability.record_phases,
            record_resources=observability.record_resources,
        )

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"metrics_{self.run_id}.jsonl"

    def record_inference(self, metric: InferenceMetric) -> None:
        if not self.enabled or not self.record_inference_enabled:
            return

        with self._lock:
            self.inference_metrics.append(metric)
            self._summary["inference_calls"] += 1
            self._summary["total_latency_ms"] += metric.latency_ms
            if metric.usage is not None:
                self._summary["total_input_tokens"] += metric.usage.prompt_tokens or 0
                self._summary["total_output_tokens"] += (
                    metric.usage.completion_tokens or 0
                )
            if metric.total_tokens_per_second is not None:
                self._summary["generation_count"] += 1
                self._summary["_prompt_tokens_per_second_sum"] += (
                    metric.prompt_tokens_per_second or 0.0
                )
                self._summary["_output_tokens_per_second_sum"] += (
                    metric.output_tokens_per_second or 0.0
                )
                self._summary["_total_tokens_per_second_sum"] += (
                    metric.total_tokens_per_second or 0.0
                )

    def record_inference_safe(self, metric: InferenceMetric) -> None:
        try:
            self.record_inference(metric)
        except Exception as exc:
            logger.error(f"Failed to record inference metric: {exc}")

    def record_phase(self, metric: PhaseMetric) -> None:
        if not self.enabled or not self.record_phases_enabled:
            return

        with self._lock:
            self.phase_metrics.append(metric)
            self._summary["phase_count"] += 1

    def record_phase_safe(self, metric: PhaseMetric) -> None:
        try:
            self.record_phase(metric)
        except Exception as exc:
            logger.error(f"Failed to record phase metric: {exc}")

    def snapshot_resources(self) -> ResourceSnapshot:
        if not self.enabled or not self.record_resources_enabled:
            return ResourceSnapshot()

        return ResourceSnapshot(
            gpu_memory_mb=self._read_gpu_memory_mb(),
            host_memory_mb=self._read_host_memory_mb(),
            gpu_utilization_pct=self._read_gpu_utilization_pct(),
            cpu_utilization_pct=self._read_cpu_utilization_pct(),
        )

    def start_utilization_sampling(
        self,
        interval_seconds: float = 1.0,
        inference_metrics_url: str | None = None,
        inference_metrics_backend: Literal["vllm", "sglang"] = "vllm",
        inference_metrics_api_key: str | None = None,
        vllm_metrics_url: str | None = None,
    ) -> None:
        """Start a background thread that polls GPU/CPU utilization at the given interval.

        If an inference metrics URL is provided, the thread also periodically
        scrapes the Prometheus ``/metrics`` endpoint for KV cache usage and
        queue depth gauges. ``vllm_metrics_url`` is a deprecated alias.
        """
        if not self.enabled or not self.record_resources_enabled:
            return
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        if inference_metrics_url is None:
            inference_metrics_url = vllm_metrics_url
            inference_metrics_backend = "vllm"
        self._inference_metrics_url = inference_metrics_url
        self._inference_metrics_backend = inference_metrics_backend
        self._inference_metrics_api_key = inference_metrics_api_key

        # Stop any in-progress sampling before resetting
        if self._sampling_stop_event is not None:
            self._sampling_stop_event.set()
            if self._sampling_thread is not None:
                self._sampling_thread.join(timeout=2.0)
        with self._lock:
            self._utilization_peak_gpu = None
            self._utilization_peak_cpu = None
            self._utilization_sum_gpu = 0.0
            self._utilization_sum_cpu = 0.0
            self._utilization_gpu_count = 0
            self._utilization_cpu_count = 0
            self._utilization_sample_count = 0
            self._vllm_peak_kv_cache = None
            self._vllm_sum_kv_cache = 0.0
            self._vllm_kv_cache_count = 0
            self._vllm_peak_requests_running = None
            self._vllm_peak_requests_waiting = None
            self._vllm_first_snapshot = None
            self._vllm_last_snapshot = None
        self._record_utilization_sample(
            self._read_gpu_utilization_pct(),
            self._read_cpu_utilization_pct(),
        )
        self._sampling_stop_event = threading.Event()
        self._sampling_thread = threading.Thread(
            target=self._sampling_loop,
            args=(interval_seconds,),
            daemon=True,
            name="utilization-sampler",
        )
        self._sampling_thread.start()

    def stop_utilization_sampling(self) -> UtilizationSummary:
        """Stop background sampling and return peak/average utilization for the episode."""
        if self._sampling_stop_event is not None:
            self._sampling_stop_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=5.0)
            self._sampling_thread = None
        self._sampling_stop_event = None

        with self._lock:
            peak_gpu = self._utilization_peak_gpu
            peak_cpu = self._utilization_peak_cpu
            gpu_count = self._utilization_gpu_count
            cpu_count = self._utilization_cpu_count
            avg_gpu = self._utilization_sum_gpu / gpu_count if gpu_count > 0 else None
            avg_cpu = self._utilization_sum_cpu / cpu_count if cpu_count > 0 else None
            total = self._utilization_sample_count
            # vLLM fields
            peak_kv = self._vllm_peak_kv_cache
            kv_count = self._vllm_kv_cache_count
            avg_kv = self._vllm_sum_kv_cache / kv_count if kv_count > 0 else None
            peak_running = self._vllm_peak_requests_running
            peak_waiting = self._vllm_peak_requests_waiting
            first_snapshot = self._vllm_first_snapshot
            last_snapshot = self._vllm_last_snapshot
            # Reset all state
            self._utilization_peak_gpu = None
            self._utilization_peak_cpu = None
            self._utilization_sum_gpu = 0.0
            self._utilization_sum_cpu = 0.0
            self._utilization_gpu_count = 0
            self._utilization_cpu_count = 0
            self._utilization_sample_count = 0
            self._vllm_peak_kv_cache = None
            self._vllm_sum_kv_cache = 0.0
            self._vllm_kv_cache_count = 0
            self._vllm_peak_requests_running = None
            self._vllm_peak_requests_waiting = None
            self._vllm_first_snapshot = None
            self._vllm_last_snapshot = None

        # Windowed delta needs two distinct scrapes within the window.
        inference_delta: Optional[dict] = None
        if (
            first_snapshot is not None
            and last_snapshot is not None
            and last_snapshot is not first_snapshot
        ):
            inference_delta = asdict(
                inference_metrics_delta(first_snapshot, last_snapshot)
            )

        return UtilizationSummary(
            peak_gpu_utilization_pct=peak_gpu,
            peak_cpu_utilization_pct=peak_cpu,
            avg_gpu_utilization_pct=avg_gpu,
            avg_cpu_utilization_pct=avg_cpu,
            sample_count=total,
            peak_kv_cache_usage_pct=peak_kv,
            avg_kv_cache_usage_pct=avg_kv,
            peak_num_requests_running=peak_running,
            peak_num_requests_waiting=peak_waiting,
            inference_metrics_delta=inference_delta,
        )

    def live_utilization_snapshot(self) -> LiveUtilizationSnapshot:
        """Read current peak/avg utilization without resetting. Thread-safe."""
        with self._lock:
            gpu_count = self._utilization_gpu_count
            cpu_count = self._utilization_cpu_count
            kv_count = self._vllm_kv_cache_count
            gen_count = self._summary.get("generation_count", 0)
            tok_sum = self._summary.get("_output_tokens_per_second_sum", 0.0)
            return LiveUtilizationSnapshot(
                avg_gpu_utilization_pct=(
                    self._utilization_sum_gpu / gpu_count if gpu_count > 0 else None
                ),
                avg_cpu_utilization_pct=(
                    self._utilization_sum_cpu / cpu_count if cpu_count > 0 else None
                ),
                peak_gpu_utilization_pct=self._utilization_peak_gpu,
                peak_cpu_utilization_pct=self._utilization_peak_cpu,
                avg_kv_cache_usage_pct=(
                    self._vllm_sum_kv_cache / kv_count if kv_count > 0 else None
                ),
                avg_output_tokens_per_second=(
                    float(tok_sum) / gen_count if gen_count > 0 else None
                ),
                sample_count=self._utilization_sample_count,
            )

    def _sampling_loop(self, interval_seconds: float) -> None:
        """Thread body: read GPU/CPU utilization, update peak and running sums."""
        while True:
            if self._sampling_stop_event is not None and self._sampling_stop_event.wait(
                interval_seconds
            ):
                break
            self._record_utilization_sample(
                self._read_gpu_utilization_pct(),
                self._read_cpu_utilization_pct(),
            )
            if self._inference_metrics_url:
                self._record_vllm_sample()

    def _record_utilization_sample(
        self,
        gpu_pct: Optional[float],
        cpu_pct: Optional[float],
    ) -> None:
        with self._lock:
            if gpu_pct is not None:
                if (
                    self._utilization_peak_gpu is None
                    or gpu_pct > self._utilization_peak_gpu
                ):
                    self._utilization_peak_gpu = gpu_pct
                self._utilization_sum_gpu += gpu_pct
                self._utilization_gpu_count += 1
            if cpu_pct is not None:
                if (
                    self._utilization_peak_cpu is None
                    or cpu_pct > self._utilization_peak_cpu
                ):
                    self._utilization_peak_cpu = cpu_pct
                self._utilization_sum_cpu += cpu_pct
                self._utilization_cpu_count += 1
            if gpu_pct is not None or cpu_pct is not None:
                self._utilization_sample_count += 1

    def _record_vllm_sample(self) -> None:
        if self._inference_metrics_url is None:
            return
        if self._inference_metrics_backend == "vllm":
            if self._inference_metrics_api_key:
                snapshot = snapshot_vllm_metrics(
                    self._inference_metrics_url,
                    api_key=self._inference_metrics_api_key,
                )
            else:
                snapshot = snapshot_vllm_metrics(self._inference_metrics_url)
        else:
            kwargs = {"backend": self._inference_metrics_backend}
            if self._inference_metrics_api_key:
                kwargs["api_key"] = self._inference_metrics_api_key
            snapshot = snapshot_inference_metrics(self._inference_metrics_url, **kwargs)
        if snapshot is None:
            return
        with self._lock:
            if self._vllm_first_snapshot is None:
                self._vllm_first_snapshot = snapshot
            self._vllm_last_snapshot = snapshot
            if snapshot.kv_cache_usage_pct is not None:
                if (
                    self._vllm_peak_kv_cache is None
                    or snapshot.kv_cache_usage_pct > self._vllm_peak_kv_cache
                ):
                    self._vllm_peak_kv_cache = snapshot.kv_cache_usage_pct
                self._vllm_sum_kv_cache += snapshot.kv_cache_usage_pct
                self._vllm_kv_cache_count += 1
            if snapshot.num_requests_running is not None:
                if (
                    self._vllm_peak_requests_running is None
                    or snapshot.num_requests_running > self._vllm_peak_requests_running
                ):
                    self._vllm_peak_requests_running = snapshot.num_requests_running
            if snapshot.num_requests_waiting is not None:
                if (
                    self._vllm_peak_requests_waiting is None
                    or snapshot.num_requests_waiting > self._vllm_peak_requests_waiting
                ):
                    self._vllm_peak_requests_waiting = snapshot.num_requests_waiting

    def get_metrics_for_episode(self, episode_id: str) -> list[InferenceMetric]:
        """Return inference metrics for a specific episode, properly locked."""
        with self._lock:
            return [m for m in self.inference_metrics if m.episode_id == episode_id]

    def flush(self) -> Optional[str]:
        if not self.enabled:
            with self._lock:
                self.inference_metrics.clear()
                self.phase_metrics.clear()
            return None

        with self._lock:
            inference_metrics = list(self.inference_metrics)
            phase_metrics = list(self.phase_metrics)
            self.inference_metrics.clear()
            self.phase_metrics.clear()

        if not inference_metrics and not phase_metrics:
            return str(self.output_path) if self.output_path.exists() else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("a", encoding="utf-8") as handle:
            for payload in self._serialize_metrics(inference_metrics, phase_metrics):
                line = json.dumps(payload, default=str)
                if not line:
                    raise ValueError("Refusing to write an empty metrics line")
                json.loads(line)
                handle.write(line + "\n")

        return str(self.output_path)

    def summary(self) -> dict[str, float | int]:
        with self._lock:
            summary = dict(self._summary)

        generation_count = int(summary["generation_count"])
        if generation_count > 0:
            summary["average_prompt_tokens_per_second"] = (
                float(summary["_prompt_tokens_per_second_sum"]) / generation_count
            )
            summary["average_output_tokens_per_second"] = (
                float(summary["_output_tokens_per_second_sum"]) / generation_count
            )
            summary["average_total_tokens_per_second"] = (
                float(summary["_total_tokens_per_second_sum"]) / generation_count
            )
        else:
            summary["average_prompt_tokens_per_second"] = 0.0
            summary["average_output_tokens_per_second"] = 0.0
            summary["average_total_tokens_per_second"] = 0.0

        del summary["_prompt_tokens_per_second_sum"]
        del summary["_output_tokens_per_second_sum"]
        del summary["_total_tokens_per_second_sum"]

        return summary

    def _serialize_metrics(
        self,
        inference_metrics: list[InferenceMetric],
        phase_metrics: list[PhaseMetric],
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for metric in inference_metrics:
            payload = asdict(metric)
            payload["metric_type"] = "inference"
            payloads.append(payload)
        for metric in phase_metrics:
            payload = asdict(metric)
            payload["metric_type"] = "phase"
            payloads.append(payload)
        return payloads

    def _read_gpu_memory_mb(self) -> Optional[float]:

        result = read_gpu_memory_info()
        if result is None:
            return None
        used_mb, _free_mb, _total_mb = result
        return used_mb

    def _read_host_memory_mb(self) -> Optional[float]:
        if _psutil is not None:
            return float(_psutil.Process().memory_info().rss / (1024 * 1024))

        status_path = Path("/proc/self/status")
        if not status_path.exists():
            return None

        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024

        return None

    def _read_gpu_utilization_pct(self) -> Optional[float]:
        return read_gpu_utilization_pct()

    def _read_cpu_utilization_pct(self) -> Optional[float]:
        if _psutil is None:
            return None

        try:
            return float(_psutil.cpu_percent(interval=None))
        except Exception:
            return None
