from __future__ import annotations

import unittest

from src.observability.prometheus import parse_prometheus_text


SAMPLE_VLLM_METRICS = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="nsl-test-loop",engine="default"} 0.42
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="nsl-test-loop",engine="default"} 2
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="nsl-test-loop",engine="default"} 0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="nsl-test-loop",engine="default"} 12345
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="nsl-test-loop",engine="default"} 6789
# HELP vllm:e2e_request_latency_seconds Histogram of end to end request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{model_name="nsl-test-loop",le="0.5"} 3
vllm:e2e_request_latency_seconds_bucket{model_name="nsl-test-loop",le="1.0"} 7
vllm:e2e_request_latency_seconds_bucket{model_name="nsl-test-loop",le="+Inf"} 10
vllm:e2e_request_latency_seconds_count{model_name="nsl-test-loop"} 10
vllm:e2e_request_latency_seconds_sum{model_name="nsl-test-loop"} 8.5
"""


class TestParsePrometheusText(unittest.TestCase):
    def test_parses_gauge_metrics(self) -> None:
        result = parse_prometheus_text(SAMPLE_VLLM_METRICS)

        self.assertIn("vllm:kv_cache_usage_perc", result)
        entries = result["vllm:kv_cache_usage_perc"]
        self.assertEqual(len(entries), 1)
        labels, value = entries[0]
        self.assertAlmostEqual(value, 0.42)
        self.assertEqual(labels["model_name"], "nsl-test-loop")

    def test_parses_counter_metrics(self) -> None:
        result = parse_prometheus_text(SAMPLE_VLLM_METRICS)

        self.assertIn("vllm:prompt_tokens_total", result)
        entries = result["vllm:prompt_tokens_total"]
        self.assertEqual(len(entries), 1)
        _, value = entries[0]
        self.assertAlmostEqual(value, 12345.0)

    def test_parses_integer_gauge(self) -> None:
        result = parse_prometheus_text(SAMPLE_VLLM_METRICS)

        entries = result["vllm:num_requests_running"]
        self.assertEqual(len(entries), 1)
        _, value = entries[0]
        self.assertAlmostEqual(value, 2.0)

    def test_skips_comment_and_type_lines(self) -> None:
        result = parse_prometheus_text(SAMPLE_VLLM_METRICS)

        # No key should start with #
        for key in result:
            self.assertFalse(key.startswith("#"))

    def test_parses_histogram_lines(self) -> None:
        """Histogram bucket/count/sum lines should be parsed (not skipped)."""
        result = parse_prometheus_text(SAMPLE_VLLM_METRICS)

        self.assertIn("vllm:e2e_request_latency_seconds_bucket", result)
        self.assertIn("vllm:e2e_request_latency_seconds_count", result)
        self.assertIn("vllm:e2e_request_latency_seconds_sum", result)

    def test_only_comments(self) -> None:
        result = parse_prometheus_text("# HELP foo\n# TYPE foo gauge\n")
        self.assertEqual(result, {})

    def test_metric_without_labels(self) -> None:
        text = "process_cpu_seconds_total 42.5\n"
        result = parse_prometheus_text(text)

        self.assertIn("process_cpu_seconds_total", result)
        labels, value = result["process_cpu_seconds_total"][0]
        self.assertEqual(labels, {})
        self.assertAlmostEqual(value, 42.5)

    def test_multiple_label_values(self) -> None:
        text = (
            'http_requests_total{method="GET",status="200"} 100\n'
            'http_requests_total{method="POST",status="201"} 50\n'
        )
        result = parse_prometheus_text(text)

        entries = result["http_requests_total"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0][0]["method"], "GET")
        self.assertEqual(entries[1][0]["method"], "POST")


if __name__ == "__main__":
    unittest.main()
