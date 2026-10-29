from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.observability.vllm_metrics import snapshot_inference_metrics

SAMPLE_SGLANG_METRICS_RESPONSE = """\
# HELP sglang:token_usage KV token usage fraction.
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="test"} 0.37
# HELP sglang:num_running_reqs Number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="test"} 3
# HELP sglang:num_queue_reqs Number of queued requests.
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="test"} 2
# HELP sglang:prompt_tokens_total Prompt tokens.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="test"} 111
# HELP sglang:gen_throughput_total Generation token counter analogue.
# TYPE sglang:gen_throughput_total counter
sglang:gen_throughput_total{model_name="test"} 222
"""


class TestSnapshotSglangMetrics(unittest.TestCase):
    def test_extracts_sglang_metric_names(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_SGLANG_METRICS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("src.observability.vllm_metrics.requests.get", return_value=mock_response):
            snapshot = snapshot_inference_metrics(
                "http://localhost:30000/metrics",
                backend="sglang",
            )

        self.assertIsNotNone(snapshot)
        self.assertAlmostEqual(snapshot.kv_cache_usage_pct, 37.0)
        self.assertEqual(snapshot.num_requests_running, 3)
        self.assertEqual(snapshot.num_requests_waiting, 2)
        self.assertEqual(snapshot.total_prompt_tokens, 111)
        self.assertEqual(snapshot.total_generation_tokens, 222)


if __name__ == "__main__":
    unittest.main()
