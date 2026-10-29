from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.observability.vllm_metrics import snapshot_vllm_metrics

SAMPLE_METRICS_RESPONSE = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="test"} 0.42
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="test"} 2
# HELP vllm:num_requests_waiting Number of requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="test"} 1
# HELP vllm:prompt_tokens_total Prefill tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="test"} 12345
# HELP vllm:generation_tokens_total Generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="test"} 6789
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="test"} 21
# The _created sibling is a unix-timestamp gauge that must NOT be mistaken for the counter value.
# TYPE vllm:num_preemptions_created gauge
vllm:num_preemptions_created{model_name="test"} 1780140784.98
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{model_name="test"} 800
# TYPE vllm:prefix_cache_hits_created gauge
vllm:prefix_cache_hits_created{model_name="test"} 1780140784.98
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="test"} 1000
# TYPE vllm:external_prefix_cache_hits_total counter
vllm:external_prefix_cache_hits_total{model_name="test"} 10
# TYPE vllm:external_prefix_cache_queries_total counter
vllm:external_prefix_cache_queries_total{model_name="test"} 40
# TYPE vllm:prompt_tokens_cached_total counter
vllm:prompt_tokens_cached_total{model_name="test"} 5000
"""

# A build/version that exposes counters WITHOUT the ``_total`` suffix (bare form,
# per current vLLM docs). The scraper must fall back to the bare name.
BARE_COUNTER_RESPONSE = """\
# TYPE vllm:num_preemptions counter
vllm:num_preemptions{model_name="test"} 7
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{model_name="test"} 3
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{model_name="test"} 9
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens{model_name="test"} 100
# TYPE vllm:generation_tokens counter
vllm:generation_tokens{model_name="test"} 50
"""


def _scrape(text: str):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = text
    mock_response.raise_for_status = MagicMock()
    with patch(
        "src.observability.vllm_metrics.requests.get", return_value=mock_response
    ):
        return snapshot_vllm_metrics("http://localhost:8000/metrics")


class TestSnapshotVllmMetrics(unittest.TestCase):
    def test_extracts_complete_vllm_snapshot(self) -> None:
        snapshot = _scrape(SAMPLE_METRICS_RESPONSE)
        self.assertIsNotNone(snapshot)
        expected = {
            "kv_cache_usage_pct": 42.0,
            "num_requests_running": 2,
            "num_requests_waiting": 1,
            "total_prompt_tokens": 12345,
            "total_generation_tokens": 6789,
            "total_preemptions": 21,
            "prefix_cache_hits": 800,
            "prefix_cache_queries": 1000,
            "external_prefix_cache_hits": 10,
            "external_prefix_cache_queries": 40,
            "prompt_tokens_cached": 5000,
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.assertEqual(getattr(snapshot, field), value)
        self.assertLess(snapshot.total_preemptions, 1000)

    def test_returns_none_for_request_failures(self) -> None:
        import requests

        for name, error in (
            ("connection-error", ConnectionError),
            ("timeout", requests.Timeout),
        ):
            with self.subTest(name=name):
                with patch(
                    "src.observability.vllm_metrics.requests.get", side_effect=error
                ):
                    snapshot = snapshot_vllm_metrics(
                        "http://localhost:8000/metrics"
                    )
                self.assertIsNone(snapshot)

    def test_falls_back_to_bare_counter_names(self) -> None:
        # A build that omits the ``_total`` suffix must still be scraped.
        snapshot = _scrape(BARE_COUNTER_RESPONSE)
        self.assertEqual(snapshot.total_preemptions, 7)
        self.assertEqual(snapshot.prefix_cache_hits, 3)
        self.assertEqual(snapshot.prefix_cache_queries, 9)
        self.assertEqual(snapshot.total_prompt_tokens, 100)
        self.assertEqual(snapshot.total_generation_tokens, 50)

    def test_absent_metrics_default_to_none(self) -> None:
        snapshot = _scrape("# Only comments\n")
        self.assertIsNotNone(snapshot)
        for field in (
            "kv_cache_usage_pct",
            "num_requests_running",
            "num_requests_waiting",
            "total_prompt_tokens",
            "total_generation_tokens",
            "total_preemptions",
            "prefix_cache_hits",
            "prefix_cache_queries",
            "external_prefix_cache_hits",
            "external_prefix_cache_queries",
            "prompt_tokens_cached",
        ):
            with self.subTest(field=field):
                self.assertIsNone(getattr(snapshot, field))


if __name__ == "__main__":
    unittest.main()
