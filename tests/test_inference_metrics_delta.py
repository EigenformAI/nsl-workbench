from __future__ import annotations

import unittest

from src.observability.vllm_metrics import (
    InferenceMetricsSnapshot,
    inference_metrics_delta,
)


def _snap(scraped_at: float, **kw) -> InferenceMetricsSnapshot:
    return InferenceMetricsSnapshot(scraped_at=scraped_at, **kw)


class TestInferenceMetricsDelta(unittest.TestCase):
    def test_counter_deltas_and_token_throughput(self) -> None:
        prev = _snap(
            100.0,
            total_preemptions=21,
            total_prompt_tokens=1000,
            total_generation_tokens=200,
        )
        curr = _snap(
            160.0,
            total_preemptions=26,
            total_prompt_tokens=4000,
            total_generation_tokens=800,
        )
        delta = inference_metrics_delta(prev, curr)
        self.assertEqual(delta.window_seconds, 60.0)
        self.assertEqual(delta.preemptions, 5)
        self.assertEqual(delta.prompt_tokens, 3000)
        self.assertEqual(delta.generation_tokens, 600)
        self.assertAlmostEqual(delta.prompt_tokens_per_second, 50.0)
        self.assertAlmostEqual(delta.generation_tokens_per_second, 10.0)

    def test_windowed_cache_ratios(self) -> None:
        prev = _snap(
            0.0,
            prefix_cache_hits=100,
            prefix_cache_queries=1000,
            external_prefix_cache_hits=10,
            external_prefix_cache_queries=20,
            total_prompt_tokens=1000,
            prompt_tokens_cached=100,
        )
        curr = _snap(
            10.0,
            prefix_cache_hits=900,
            prefix_cache_queries=2000,
            external_prefix_cache_hits=40,
            external_prefix_cache_queries=80,
            total_prompt_tokens=2000,
            prompt_tokens_cached=850,
        )
        delta = inference_metrics_delta(prev, curr)
        self.assertEqual(delta.prefix_cache_hits, 800)
        self.assertEqual(delta.prefix_cache_queries, 1000)
        self.assertAlmostEqual(delta.prefix_cache_hit_rate, 0.8)
        self.assertEqual(delta.external_prefix_cache_hits, 30)
        self.assertEqual(delta.external_prefix_cache_queries, 60)
        self.assertAlmostEqual(delta.external_prefix_cache_hit_rate, 0.5)
        self.assertEqual(delta.prompt_tokens_cached, 750)
        self.assertAlmostEqual(delta.prompt_tokens_cached_rate, 0.75)

    def test_invalid_rate_denominators_yield_none(self) -> None:
        cases = (
            (
                "zero-window",
                _snap(50.0, total_prompt_tokens=0),
                _snap(50.0, total_prompt_tokens=1000),
                "prompt_tokens_per_second",
            ),
            (
                "zero-queries",
                _snap(0.0, prefix_cache_hits=10, prefix_cache_queries=100),
                _snap(10.0, prefix_cache_hits=10, prefix_cache_queries=100),
                "prefix_cache_hit_rate",
            ),
        )
        for name, prev, curr, field in cases:
            with self.subTest(name=name):
                delta = inference_metrics_delta(prev, curr)
                self.assertIsNone(getattr(delta, field))
                if name == "zero-window":
                    self.assertEqual(delta.window_seconds, 0.0)
                else:
                    self.assertEqual(delta.prefix_cache_queries, 0)

    def test_reset_or_missing_counters_yield_none(self) -> None:
        cases = (
            (
                "reset",
                _snap(0.0, total_preemptions=100, total_prompt_tokens=9999),
                _snap(10.0, total_preemptions=3, total_prompt_tokens=5),
                ("preemptions", "prompt_tokens"),
            ),
            (
                "missing",
                _snap(0.0),
                _snap(10.0, total_preemptions=5),
                ("preemptions", "prefix_cache_hit_rate"),
            ),
        )
        for name, prev, curr, fields in cases:
            with self.subTest(name=name):
                delta = inference_metrics_delta(prev, curr)
                for field in fields:
                    self.assertIsNone(getattr(delta, field))

    def test_gauges_taken_from_current_snapshot(self) -> None:
        prev = _snap(0.0, kv_cache_usage_pct=10.0, num_requests_running=1)
        curr = _snap(
            10.0,
            kv_cache_usage_pct=88.0,
            num_requests_running=8,
            num_requests_waiting=3,
        )
        delta = inference_metrics_delta(prev, curr)
        self.assertEqual(delta.kv_cache_usage_pct, 88.0)
        self.assertEqual(delta.num_requests_running, 8)
        self.assertEqual(delta.num_requests_waiting, 3)


if __name__ == "__main__":
    unittest.main()
