import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.observability.collector import MetricsCollector
from src.observability.types import LiveUtilizationSnapshot, UtilizationSummary


class MetricsCollectorTests(unittest.TestCase):
    def test_snapshot_resources_includes_utilization_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            with (
                patch.object(collector, "_read_gpu_memory_mb", return_value=1024.0),
                patch.object(collector, "_read_host_memory_mb", return_value=2048.0),
                patch.object(collector, "_read_gpu_utilization_pct", return_value=88.0),
                patch.object(collector, "_read_cpu_utilization_pct", return_value=35.0),
            ):
                snapshot = collector.snapshot_resources()

        self.assertEqual(snapshot.gpu_memory_mb, 1024.0)
        self.assertEqual(snapshot.host_memory_mb, 2048.0)
        self.assertEqual(snapshot.gpu_utilization_pct, 88.0)
        self.assertEqual(snapshot.cpu_utilization_pct, 35.0)

    def test_background_sampling_tracks_peak_and_average(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            # Sequence of readings: GPU 20, 80, 50; CPU 10, 30, 20
            gpu_readings = [20.0, 80.0, 50.0]
            cpu_readings = [10.0, 30.0, 20.0]
            call_count = [0]

            def gpu_read():
                idx = min(call_count[0], len(gpu_readings) - 1)
                return gpu_readings[idx]

            def cpu_read():
                idx = min(call_count[0], len(cpu_readings) - 1)
                call_count[0] += 1
                return cpu_readings[idx]

            with (
                patch.object(
                    collector, "_read_gpu_utilization_pct", side_effect=gpu_read
                ),
                patch.object(
                    collector, "_read_cpu_utilization_pct", side_effect=cpu_read
                ),
            ):
                collector.start_utilization_sampling(interval_seconds=0.05)
                time.sleep(0.25)
                summary = collector.stop_utilization_sampling()

        self.assertIsInstance(summary, UtilizationSummary)
        self.assertIsNotNone(summary.peak_gpu_utilization_pct)
        self.assertIsNotNone(summary.peak_cpu_utilization_pct)
        self.assertIsNotNone(summary.avg_gpu_utilization_pct)
        self.assertIsNotNone(summary.avg_cpu_utilization_pct)
        self.assertGreater(summary.sample_count, 0)
        self.assertEqual(summary.peak_gpu_utilization_pct, 80.0)
        self.assertEqual(summary.peak_cpu_utilization_pct, 30.0)

    def test_start_sampling_captures_immediate_sample_for_short_episode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            with (
                patch.object(collector, "_read_gpu_utilization_pct", return_value=34.0),
                patch.object(collector, "_read_cpu_utilization_pct", return_value=15.0),
            ):
                collector.start_utilization_sampling(interval_seconds=1.0)
                summary = collector.stop_utilization_sampling()

        self.assertEqual(summary.sample_count, 1)
        self.assertEqual(summary.peak_gpu_utilization_pct, 34.0)
        self.assertEqual(summary.peak_cpu_utilization_pct, 15.0)
        self.assertEqual(summary.avg_gpu_utilization_pct, 34.0)
        self.assertEqual(summary.avg_cpu_utilization_pct, 15.0)

    def test_stop_utilization_sampling_without_start_returns_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))
            summary = collector.stop_utilization_sampling()

        self.assertIsInstance(summary, UtilizationSummary)
        self.assertIsNone(summary.peak_gpu_utilization_pct)
        self.assertIsNone(summary.peak_cpu_utilization_pct)
        self.assertIsNone(summary.avg_gpu_utilization_pct)
        self.assertIsNone(summary.avg_cpu_utilization_pct)
        self.assertEqual(summary.sample_count, 0)

    def test_disabled_collector_sampling_returns_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(
                run_id="run-123", output_dir=Path(temp_dir), enabled=False
            )
            collector.start_utilization_sampling(interval_seconds=0.05)
            time.sleep(0.1)
            summary = collector.stop_utilization_sampling()

        self.assertIsNone(summary.peak_gpu_utilization_pct)
        self.assertEqual(summary.sample_count, 0)


    def test_vllm_sampling_tracks_peak_kv_cache(self) -> None:
        """When vllm_metrics_url is provided, sampling should track peak KV cache."""
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            # Mock vLLM snapshots with increasing then decreasing KV cache
            from src.observability.vllm_metrics import VllmMetricsSnapshot

            snapshots = [
                VllmMetricsSnapshot(scraped_at=1.0, kv_cache_usage_pct=20.0, num_requests_running=1, num_requests_waiting=0),
                VllmMetricsSnapshot(scraped_at=2.0, kv_cache_usage_pct=80.0, num_requests_running=2, num_requests_waiting=3),
                VllmMetricsSnapshot(scraped_at=3.0, kv_cache_usage_pct=50.0, num_requests_running=1, num_requests_waiting=0),
            ]
            call_count = [0]

            def mock_snapshot(url):
                idx = min(call_count[0], len(snapshots) - 1)
                call_count[0] += 1
                return snapshots[idx]

            with (
                patch.object(collector, "_read_gpu_utilization_pct", return_value=50.0),
                patch.object(collector, "_read_cpu_utilization_pct", return_value=20.0),
                patch("src.observability.collector.snapshot_vllm_metrics", side_effect=mock_snapshot),
            ):
                collector.start_utilization_sampling(
                    interval_seconds=0.05,
                    vllm_metrics_url="http://localhost:8000/metrics",
                )
                time.sleep(0.25)
                summary = collector.stop_utilization_sampling()

        self.assertIsNotNone(summary.peak_kv_cache_usage_pct)
        self.assertAlmostEqual(summary.peak_kv_cache_usage_pct, 80.0)
        self.assertIsNotNone(summary.avg_kv_cache_usage_pct)
        self.assertEqual(summary.peak_num_requests_waiting, 3)

    def test_vllm_sampling_graceful_when_scrape_fails(self) -> None:
        """When /metrics is unreachable, GPU/CPU sampling should continue."""
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            with (
                patch.object(collector, "_read_gpu_utilization_pct", return_value=60.0),
                patch.object(collector, "_read_cpu_utilization_pct", return_value=25.0),
                patch("src.observability.collector.snapshot_vllm_metrics", return_value=None),
            ):
                collector.start_utilization_sampling(
                    interval_seconds=0.05,
                    vllm_metrics_url="http://localhost:8000/metrics",
                )
                time.sleep(0.15)
                summary = collector.stop_utilization_sampling()

        # GPU/CPU sampling should still work
        self.assertIsNotNone(summary.peak_gpu_utilization_pct)
        self.assertAlmostEqual(summary.peak_gpu_utilization_pct, 60.0)
        # vLLM fields should be None
        self.assertIsNone(summary.peak_kv_cache_usage_pct)

    def test_no_vllm_url_means_no_vllm_fields(self) -> None:
        """Without vllm_metrics_url, vLLM fields in summary should be None."""
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            with (
                patch.object(collector, "_read_gpu_utilization_pct", return_value=50.0),
                patch.object(collector, "_read_cpu_utilization_pct", return_value=20.0),
            ):
                collector.start_utilization_sampling(interval_seconds=0.05)
                time.sleep(0.1)
                summary = collector.stop_utilization_sampling()

        self.assertIsNone(summary.peak_kv_cache_usage_pct)
        self.assertIsNone(summary.avg_kv_cache_usage_pct)
        self.assertIsNone(summary.peak_num_requests_waiting)


    def test_live_utilization_snapshot_returns_values_without_reset(self) -> None:
        """live_utilization_snapshot() should return current values without resetting them."""
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            with (
                patch.object(collector, "_read_gpu_utilization_pct", return_value=75.0),
                patch.object(collector, "_read_cpu_utilization_pct", return_value=40.0),
            ):
                collector.start_utilization_sampling(interval_seconds=0.05)
                time.sleep(0.15)

                # First call: should have values
                snap1 = collector.live_utilization_snapshot()
                self.assertIsInstance(snap1, LiveUtilizationSnapshot)
                self.assertIsNotNone(snap1.peak_gpu_utilization_pct)
                self.assertAlmostEqual(snap1.peak_gpu_utilization_pct, 75.0)
                self.assertIsNotNone(snap1.avg_gpu_utilization_pct)
                self.assertGreater(snap1.sample_count, 0)

                # Second call: values should still be there (not reset)
                snap2 = collector.live_utilization_snapshot()
                self.assertIsNotNone(snap2.peak_gpu_utilization_pct)
                self.assertGreaterEqual(snap2.sample_count, snap1.sample_count)

                # stop_utilization_sampling should still return valid summary
                summary = collector.stop_utilization_sampling()

            self.assertIsNotNone(summary.peak_gpu_utilization_pct)
            self.assertAlmostEqual(summary.peak_gpu_utilization_pct, 75.0)

    def test_live_utilization_snapshot_before_sampling(self) -> None:
        """live_utilization_snapshot() before start_utilization_sampling returns empty."""
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))
            snap = collector.live_utilization_snapshot()

        self.assertIsInstance(snap, LiveUtilizationSnapshot)
        self.assertIsNone(snap.peak_gpu_utilization_pct)
        self.assertIsNone(snap.avg_gpu_utilization_pct)
        self.assertIsNone(snap.peak_cpu_utilization_pct)
        self.assertIsNone(snap.avg_cpu_utilization_pct)
        self.assertIsNone(snap.avg_kv_cache_usage_pct)
        self.assertIsNone(snap.avg_output_tokens_per_second)
        self.assertEqual(snap.sample_count, 0)

    def test_live_utilization_snapshot_thread_safety(self) -> None:
        """Concurrent live_utilization_snapshot() calls should not raise."""
        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            with (
                patch.object(collector, "_read_gpu_utilization_pct", return_value=50.0),
                patch.object(collector, "_read_cpu_utilization_pct", return_value=25.0),
            ):
                collector.start_utilization_sampling(interval_seconds=0.05)
                time.sleep(0.1)

                errors: list[Exception] = []

                def reader():
                    try:
                        for _ in range(20):
                            collector.live_utilization_snapshot()
                    except Exception as e:
                        errors.append(e)

                threads = [threading.Thread(target=reader) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                collector.stop_utilization_sampling()

        self.assertEqual(errors, [])

    def test_live_utilization_snapshot_includes_tok_s(self) -> None:
        """live_utilization_snapshot() should include avg output tokens/sec from inference metrics."""
        from src.observability.types import InferenceMetric

        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-123", output_dir=Path(temp_dir))

            # Record some inference metrics with token rates
            for rate in [100.0, 200.0, 300.0]:
                collector.record_inference(InferenceMetric(
                    inference_id=f"inf-{rate}",
                    run_id="run-123",
                    backend="vllm",
                    phase="action",
                    success=True,
                    latency_ms=100.0,
                    output_tokens_per_second=rate,
                    total_tokens_per_second=rate,
                ))

            snap = collector.live_utilization_snapshot()
            self.assertIsNotNone(snap.avg_output_tokens_per_second)
            self.assertAlmostEqual(snap.avg_output_tokens_per_second, 200.0)


if __name__ == "__main__":
    unittest.main()
