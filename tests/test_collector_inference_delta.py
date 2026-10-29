from __future__ import annotations

import unittest
from unittest.mock import patch

from src.observability.collector import MetricsCollector
from src.observability.vllm_metrics import InferenceMetricsSnapshot


def _collector(tmp: str) -> MetricsCollector:
    c = MetricsCollector(run_id="t", output_dir=tmp, enabled=True)
    c._inference_metrics_url = "http://localhost:8000/metrics"
    c._inference_metrics_backend = "vllm"
    return c


class TestCollectorInferenceDelta(unittest.TestCase):
    def test_stop_exposes_windowed_inference_delta(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            collector = _collector(tmp)
            first = InferenceMetricsSnapshot(
                scraped_at=100.0,
                kv_cache_usage_pct=30.0,
                num_requests_running=4,
                total_preemptions=10,
                total_prompt_tokens=1000,
                prefix_cache_hits=100,
                prefix_cache_queries=1000,
            )
            last = InferenceMetricsSnapshot(
                scraped_at=160.0,
                kv_cache_usage_pct=55.0,
                num_requests_running=8,
                total_preemptions=15,
                total_prompt_tokens=7000,
                prefix_cache_hits=900,
                prefix_cache_queries=2000,
            )
            with patch(
                "src.observability.collector.snapshot_vllm_metrics",
                side_effect=[first, last],
            ):
                collector._record_vllm_sample()
                collector._record_vllm_sample()

            summary = collector.stop_utilization_sampling()

        self.assertIsNotNone(summary.inference_metrics_delta)
        delta = summary.inference_metrics_delta
        self.assertEqual(delta["window_seconds"], 60.0)
        self.assertEqual(delta["preemptions"], 5)
        self.assertEqual(delta["prompt_tokens"], 6000)
        # window prefix-cache hit rate: (900-100)/(2000-1000) = 0.8
        self.assertAlmostEqual(delta["prefix_cache_hit_rate"], 0.8)

    def test_delta_is_none_without_samples(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            collector = _collector(tmp)
            summary = collector.stop_utilization_sampling()
        self.assertIsNone(summary.inference_metrics_delta)

    def test_delta_state_resets_between_windows(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            collector = _collector(tmp)
            with patch(
                "src.observability.collector.snapshot_vllm_metrics",
                side_effect=[
                    InferenceMetricsSnapshot(scraped_at=1.0, total_preemptions=1),
                    InferenceMetricsSnapshot(scraped_at=2.0, total_preemptions=9),
                ],
            ):
                collector._record_vllm_sample()
                collector._record_vllm_sample()
            collector.stop_utilization_sampling()
            # Second window with no samples must not reuse the prior window's state.
            summary2 = collector.stop_utilization_sampling()
        self.assertIsNone(summary2.inference_metrics_delta)


if __name__ == "__main__":
    unittest.main()
