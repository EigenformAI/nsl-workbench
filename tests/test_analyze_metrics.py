from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.analyze_metrics import (
    analyze_paths,
    build_generation_benchmarks_by_harness_profile_task_table,
    build_generation_comparison_table,
    build_kv_cache_usage_table,
    build_latency_by_model_table,
    build_queue_depth_table,
    discover_run_files,
    load_run_doc,
    render_run_table,
    render_text_report,
    summarize_run,
    summarize_run_doc,
)


# Deliberately synthetic reporting fixtures, not recorded experiment results.
_GENERATION_ROLLUP = [
    {
        "generation_id": 0,
        "generation_dir": "data/gen_0",
        "served_adapter_dir": None,
        "served_artifact_path": None,
        "training_export_format": "lora",
        "trained_adapter_dir": "/models/after_gen_0",
        "trained_artifact_path": "/models/after_gen_0",
        "training_data_paths": ["data/gen_0/sft.jsonl"],
        "metrics": {
            "generation_id": 0,
            "total_episodes_run": 4,
            "total_successful": 1,
            "training_row_count": 10,
            "total_score": 12.0,
            "success_rate": 0.25,
            "started_at": "2024-01-01T00:00:00Z",
            "completed_at": "2024-01-01T00:05:00Z",
            "total_agent_seconds": 200.0,
            "run_id": "synthetic-reporting-run",
            "episodes_per_hour": 36.0,
            "episodes_per_minute": 0.6,
            "training_rows_per_minute": 1.5,
            "average_output_tokens_per_second": 40.0,
            "average_inference_duty_cycle": 0.8,
            "peak_gpu_utilization_pct": 100.0,
            "average_gpu_utilization_pct": 80.0,
            "peak_cpu_utilization_pct": 20.0,
            "average_cpu_utilization_pct": 10.0,
            "total_input_tokens": 5000,
            "total_output_tokens": 2000,
            "total_tokens": 7000,
            "tokens_per_successful_episode": 7000.0,
            "peak_context_tokens": 500,
            "avg_context_tokens": 300.0,
            "median_context_tokens": 250.0,
        },
    },
    {
        "generation_id": 1,
        "generation_dir": "data/gen_1",
        "served_adapter_dir": "/models/after_gen_0",
        "served_artifact_path": "/models/after_gen_0",
        "training_export_format": "lora",
        "trained_adapter_dir": None,
        "trained_artifact_path": None,
        "training_data_paths": [],
        "metrics": {
            "generation_id": 1,
            "total_episodes_run": 1,
            "total_successful": 1,
            "training_row_count": 10,
            "total_score": 5000.0,
            "success_rate": 1.0,
            "started_at": "2024-01-01T00:05:00Z",
            "completed_at": "2024-01-01T00:07:00Z",
            "total_agent_seconds": 100.0,
            "run_id": "synthetic-reporting-run",
            "episodes_per_hour": 36.0,
            "episodes_per_minute": 0.6,
            "training_rows_per_minute": 6.0,
            "average_output_tokens_per_second": 60.0,
            "average_inference_duty_cycle": 0.6,
            "peak_gpu_utilization_pct": 100.0,
            "average_gpu_utilization_pct": 60.0,
            "peak_cpu_utilization_pct": 30.0,
            "average_cpu_utilization_pct": 20.0,
            "total_input_tokens": 2000,
            "total_output_tokens": 1000,
            "total_tokens": 3000,
            "tokens_per_successful_episode": 3000.0,
            "peak_context_tokens": 400,
            "avg_context_tokens": 200.0,
            "median_context_tokens": 150.0,
        },
    },
]


def _base_run_doc(
    run_id: str = "synthetic-reporting-run",
    *,
    harness_profile: str | None = "stub_profile",
    task_class: str = "tests.fixtures.stub_task.ValidStubTask",
    task_name: str = "valid-stub",
    started_at: str = "2024-01-01T00:00:00Z",
    hardware_tags: list[str] | None = None,
    load_tags: list[str] | None = None,
    generations: list[dict] | None = None,
) -> dict:
    generation_rollup = json.loads(json.dumps(generations or _GENERATION_ROLLUP))
    for generation in generation_rollup:
        metrics = generation.get("metrics", {})
        if isinstance(metrics, dict):
            metrics["run_id"] = run_id

    return {
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "ended_at": "2024-01-01T00:08:00Z",
        "harness_type": "container" if harness_profile is not None else "orchestrator_modes",
        "harness_profile": harness_profile,
        "harness_class": "src.harness.container.ContainerHarness"
        if harness_profile is not None
        else "src.harness.orchestrator_modes.OrchestratorModeHarness",
        "task_class": task_class,
        "task_name": task_name,
        "config_path": f"config/{task_name}-{harness_profile or 'default'}.toml",
        "model_name": "vllm:nsl-test",
        "commit_id": "abc1234",
        "git_dirty": False,
        "hardware_tags": hardware_tags if hardware_tags is not None else ["rtx-4090", "24gb"],
        "load_tags": load_tags if load_tags is not None else ["nightly-bench"],
        "num_generations": len(generation_rollup),
        "training_window_size": 2,
        "training_export_format": "lora",
        "total_wall_clock_seconds": 500.0,
        "total_agent_seconds": 300.0,
        "total_tokens": 10000,
        "generations": generation_rollup,
    }


def _write_run_file(directory: Path, run_doc: dict) -> Path:
    file_path = directory / "run.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(run_doc), encoding="utf-8")
    return file_path


def _write_metrics_jsonl(directory: Path, run_id: str, records: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"metrics_{run_id}.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return file_path


def _make_inference_record(run_id: str, episode_id: str = "ep_0") -> dict:
    return {
        "metric_type": "inference",
        "inference_id": "test-id",
        "run_id": run_id,
        "backend": "vllm",
        "model": "nsl-test",
        "phase": "orchestrator",
        "success": True,
        "episode_id": episode_id,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "latency_ms": 500.0,
        "prompt_tokens_per_second": 200.0,
        "output_tokens_per_second": 100.0,
        "total_tokens_per_second": 300.0,
        "gpu_memory_mb": 8.0,
        "host_memory_mb": 1000.0,
    }


class TestDiscoverRunFiles(unittest.TestCase):
    def test_finds_run_json_in_nested_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(root / "run_a", _base_run_doc("run-a"))
            _write_run_file(root / "run_b", _base_run_doc("run-b"))

            found = discover_run_files([root])

            self.assertEqual(len(found), 2)
            self.assertTrue(all(path.name == "run.json" for path in found))

    def test_accepts_direct_run_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_path = _write_run_file(root, _base_run_doc())

            found = discover_run_files([file_path])

            self.assertEqual(found, [file_path.resolve()])

    def test_legacy_filenames_no_longer_discovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "LAST_RUN_LOG.json").write_text("{}", encoding="utf-8")
            (root / "abc_full_run_123.json").write_text("{}", encoding="utf-8")
            (root / "orchestration_summary.json").write_text("{}", encoding="utf-8")

            found = discover_run_files([root])

            self.assertEqual(found, [])


class TestLoadRunDoc(unittest.TestCase):
    def test_parses_valid_run_doc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = _write_run_file(Path(tmpdir), _base_run_doc())

            result = load_run_doc(file_path)

            self.assertEqual(result["run_id"], "synthetic-reporting-run")
            self.assertEqual(result["task_name"], "valid-stub")

    def test_malformed_run_json_is_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "run.json"
            file_path.write_text("{not json", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                load_run_doc(file_path)


class TestSummarizeRunDoc(unittest.TestCase):
    def test_extracts_identity_and_rollup(self):
        result = summarize_run_doc(_base_run_doc())

        self.assertEqual(result["run_id"], "synthetic-reporting-run")
        self.assertEqual(result["harness_profile"], "stub_profile")
        self.assertEqual(result["task_name"], "valid-stub")
        self.assertEqual(result["num_generations"], 2)
        self.assertAlmostEqual(result["total_wall_clock_seconds"], 500.0)
        self.assertEqual(result["total_tokens"], 10000)
        self.assertEqual(len(result["generations"]), 2)
        self.assertAlmostEqual(result["generations"][0]["success_rate"], 0.25)


class TestGenerationTables(unittest.TestCase):
    def test_generation_comparison_rows_have_expected_fields(self):
        summaries = [summarize_run_doc(_base_run_doc())]

        table = build_generation_comparison_table(summaries)

        self.assertEqual(len(table), 2)
        expected_keys = {
            "run_id",
            "generation_id",
            "success_rate",
            "episodes_per_minute",
            "tokens_per_successful_episode",
            "average_gpu_utilization_pct",
            "average_inference_duty_cycle",
            "total_episodes_run",
            "total_successful",
            "total_tokens",
            "peak_context_tokens",
            "avg_context_tokens",
            "median_context_tokens",
        }
        for row in table:
            self.assertTrue(expected_keys.issubset(row.keys()))

    def test_generation_benchmarks_by_harness_profile_task_groups_by_pair(self):
        run_docs = {
            "run-a": _base_run_doc("run-a", harness_profile="stub_profile", task_name="valid-stub"),
            "run-b": _base_run_doc("run-b", harness_profile="ms_agent", task_name="valid-stub"),
        }
        records = [_make_inference_record("run-a"), _make_inference_record("run-b")]

        table = build_generation_benchmarks_by_harness_profile_task_table(records, run_docs)

        pairs = {(row["harness_profile"], row["task_name"]) for row in table}
        self.assertEqual(pairs, {("stub_profile", "valid-stub"), ("ms_agent", "valid-stub")})


class TestAnalyzePathsRunJson(unittest.TestCase):
    def test_run_metadata_loaded_from_run_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(root / "run-a", _base_run_doc("run-a"))
            _write_metrics_jsonl(root / "metrics", "run-a", [_make_inference_record("run-a")])

            report = analyze_paths([root])

            self.assertEqual(report["run_count"], 1)
            self.assertEqual(report["run_file_count"], 1)
            self.assertEqual(report["metric_run_without_run_json_count"], 0)
            summary = report["runs"][0]
            self.assertEqual(summary["harness_profile"], "stub_profile")
            self.assertEqual(summary["task_name"], "valid-stub")
            self.assertEqual(summary["config_path"], "config/valid-stub-stub_profile.toml")
            self.assertEqual(summary["hardware_tags"], ["rtx-4090", "24gb"])
            self.assertEqual(summary["load_tags"], ["nightly-bench"])

    def test_metric_only_runs_are_ignored_and_counted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_metrics_jsonl(root / "metrics", "run-a", [_make_inference_record("run-a")])

            report = analyze_paths([root])

            self.assertEqual(report["run_count"], 0)
            self.assertEqual(report["metric_run_without_run_json_count"], 1)
            self.assertEqual(report["metric_file_count"], 0)

    def test_filter_by_harness_profile_keeps_only_matching_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(root / "run-a", _base_run_doc("run-a", harness_profile="stub_profile"))
            _write_run_file(root / "run-b", _base_run_doc("run-b", harness_profile="ms_agent"))
            _write_metrics_jsonl(root / "metrics", "run-a", [_make_inference_record("run-a")])
            _write_metrics_jsonl(root / "metrics", "run-b", [_make_inference_record("run-b")])

            report = analyze_paths([root], harness_profile=["stub_profile"])

            self.assertEqual([run["run_id"] for run in report["runs"]], ["run-a"])
            self.assertEqual([doc["run_id"] for doc in report["run_docs"]], ["run-a"])

    def test_filter_by_task_matches_name_or_dotted_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(
                root / "run-a",
                _base_run_doc(
                    "run-a",
                    task_class="tests.fixtures.stub_task.ValidStubTask",
                    task_name="valid-stub",
                ),
            )
            _write_run_file(
                root / "run-b",
                _base_run_doc(
                    "run-b",
                    task_class="tests.fixtures.stub_task.ZeroVariationTask",
                    task_name="other-stub",
                ),
            )

            by_name = analyze_paths([root], task=["valid-stub"])
            by_class = analyze_paths([root], task=["tests.fixtures.stub_task.ZeroVariationTask"])

            self.assertEqual([run["run_id"] for run in by_name["runs"]], ["run-a"])
            self.assertEqual([run["run_id"] for run in by_class["runs"]], ["run-b"])

    def test_filter_by_since_until_uses_started_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(root / "run-a", _base_run_doc("run-a", started_at="2024-01-01T00:00:00Z"))
            _write_run_file(root / "run-b", _base_run_doc("run-b", started_at="2026-04-09T00:00:00Z"))
            _write_run_file(root / "run-c", _base_run_doc("run-c", started_at="2026-04-10T00:00:00Z"))

            report = analyze_paths([root], since=date(2026, 4, 9), until=date(2026, 4, 10))

            self.assertEqual([run["run_id"] for run in report["runs"]], ["run-b"])

    def test_filter_combination_is_and_across_flags_or_within_repeats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(root / "run-a", _base_run_doc("run-a", harness_profile="stub_profile", task_name="valid-stub"))
            _write_run_file(root / "run-b", _base_run_doc("run-b", harness_profile="ms_agent", task_name="valid-stub"))
            _write_run_file(root / "run-c", _base_run_doc("run-c", harness_profile="stub_profile", task_name="other-stub"))

            report = analyze_paths(
                [root],
                harness_profile=["stub_profile", "ms_agent"],
                task=["valid-stub"],
            )

            self.assertEqual([run["run_id"] for run in report["runs"]], ["run-a", "run-b"])

    def test_selected_run_ids_filter_all_tables_and_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(root / "run-a", _base_run_doc("run-a", harness_profile="stub_profile", hardware_tags=["rtx-4090"], load_tags=["nightly"]))
            _write_run_file(root / "run-b", _base_run_doc("run-b", harness_profile="ms_agent", hardware_tags=["a100"], load_tags=["manual"]))
            _write_metrics_jsonl(root / "metrics-a", "run-a", [_make_inference_record("run-a")])
            _write_metrics_jsonl(root / "metrics-b", "run-b", [_make_inference_record("run-b")])

            report = analyze_paths([root], run_id="run-a")

            tables = report["comparison_tables"]
            self.assertEqual(report["metric_file_count"], 1)
            self.assertEqual([row["hardware_tag"] for row in tables["generation_benchmarks_by_hardware_tag"]], ["rtx-4090"])
            self.assertEqual([row["load_tag"] for row in tables["generation_benchmarks_by_load_tag"]], ["nightly"])
            self.assertEqual(
                [(row["harness_profile"], row["task_name"]) for row in tables["generation_benchmarks_by_harness_profile_task"]],
                [("stub_profile", "valid-stub")],
            )
            self.assertEqual([doc["run_id"] for doc in report["run_docs"]], ["run-a"])

    def test_hardware_tag_table_actually_populated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(root / "run-a", _base_run_doc("run-a", hardware_tags=["rtx-4090", "24gb"]))
            _write_metrics_jsonl(root / "metrics", "run-a", [_make_inference_record("run-a")])

            report = analyze_paths([root])

            tags = {row["hardware_tag"] for row in report["comparison_tables"]["generation_benchmarks_by_hardware_tag"]}
            self.assertEqual(tags, {"rtx-4090", "24gb"})


class TestRenderRunTable(unittest.TestCase):
    def test_renders_markdown_table(self):
        table = render_run_table(summarize_run_doc(_base_run_doc()))

        self.assertIn("Run synthetic-reporting-run Summary", table)
        self.assertIn("| Metric", table)
        self.assertIn("| Gen 0", table)
        self.assertIn("| Total", table)
        self.assertIn("Success rate", table)

    def test_text_report_includes_run_identity_tags_and_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run_file(root / "run-a", _base_run_doc("run-a"))
            _write_metrics_jsonl(root / "metrics", "run-a", [_make_inference_record("run-a")])

            text = render_text_report(analyze_paths([root]))

            self.assertIn("Runs: 1 (1 with run.json, 0 metric-only)", text)
            self.assertIn("harness_profile=stub_profile", text)
            self.assertIn("task=valid-stub", text)
            self.assertIn("hw=[rtx-4090,24gb]", text)
            self.assertIn("load=[nightly-bench]", text)
            self.assertIn("Generation benchmarks by harness profile/task", text)


class TestThroughputFields(unittest.TestCase):
    def _make_records(self, run_id: str, count: int = 5) -> list[dict]:
        records = []
        for i in range(count):
            rec = _make_inference_record(run_id, episode_id=f"ep_{i}")
            rec["latency_ms"] = 1000.0
            rec["usage"]["completion_tokens"] = 100
            rec["output_tokens_per_second"] = 100.0
            records.append(rec)
        return records

    def test_summarize_run_has_renamed_throughput_fields(self):
        summary = summarize_run("run-A", self._make_records("run-A"), _base_run_doc("run-A"))

        self.assertIn("throughput_output_tokens_per_second", summary)
        self.assertIn("avg_per_call_output_tokens_per_second", summary)
        self.assertIn("median_per_call_output_tokens_per_second", summary)
        self.assertNotIn("output_tokens_per_second", summary)

    def test_throughput_uses_ratio_of_totals(self):
        summary = summarize_run("run-A", self._make_records("run-A", count=2), _base_run_doc("run-A"))

        self.assertAlmostEqual(summary["throughput_output_tokens_per_second"], 100.0)

    def test_per_call_uses_individual_rates(self):
        summary = summarize_run("run-A", self._make_records("run-A", count=2), _base_run_doc("run-A"))

        self.assertAlmostEqual(summary["avg_per_call_output_tokens_per_second"], 100.0)
        self.assertAlmostEqual(summary["median_per_call_output_tokens_per_second"], 100.0)


class TestBuildLatencyByModelTable(unittest.TestCase):
    def _make_summary(
        self,
        model: str,
        run_id: str,
        total_latency_ms: float,
        total_output_tokens: int,
        inference_calls: int,
    ) -> dict:
        return {
            "model": model,
            "run_id": run_id,
            "total_inference_latency_ms": total_latency_ms,
            "total_output_tokens": total_output_tokens,
            "inference_calls": inference_calls,
        }

    def _make_inference_records(
        self, model: str, run_id: str, latencies: list[float], output_tokens: list[int]
    ) -> list[dict]:
        records = []
        for lat, tok in zip(latencies, output_tokens):
            rec = _make_inference_record(run_id)
            rec["model"] = model
            rec["latency_ms"] = lat
            rec["usage"]["completion_tokens"] = tok
            rec["output_tokens_per_second"] = tok / (lat / 1000) if lat > 0 else 0
            records.append(rec)
        return records

    def test_table_has_throughput_and_per_call_columns(self):
        summaries = [self._make_summary("model-a", "run-1", 5000.0, 500, 5)]
        inference_records = self._make_inference_records("model-a", "run-1", [1000.0] * 5, [100] * 5)

        rows = build_latency_by_model_table(summaries, inference_records)

        self.assertIn("throughput_output_tokens_per_second", rows[0])
        self.assertIn("avg_per_call_output_tokens_per_second", rows[0])
        self.assertIn("median_per_call_output_tokens_per_second", rows[0])

    def test_throughput_column_uses_ratio_of_totals(self):
        summaries = [self._make_summary("model-a", "run-1", 10000.0, 1000, 10)]
        inference_records = self._make_inference_records("model-a", "run-1", [1000.0] * 10, [100] * 10)

        rows = build_latency_by_model_table(summaries, inference_records)

        self.assertAlmostEqual(rows[0]["throughput_output_tokens_per_second"], 100.0)

    def test_per_call_columns_use_individual_records(self):
        summaries = [self._make_summary("model-a", "run-1", 11000.0, 600, 2)]
        inference_records = self._make_inference_records("model-a", "run-1", [1000.0, 10000.0], [500, 100])

        row = build_latency_by_model_table(summaries, inference_records)[0]

        self.assertAlmostEqual(row["avg_per_call_output_tokens_per_second"], 255.0)
        self.assertAlmostEqual(row["median_per_call_output_tokens_per_second"], 255.0)
        self.assertAlmostEqual(row["throughput_output_tokens_per_second"], 600 / 11, places=1)


def _make_inference_record_with_vllm(
    run_id: str,
    backend: str = "vllm",
    model: str = "nsl-test",
    kv_cache_usage_pct: float | None = None,
    num_requests_running: int | None = None,
    num_requests_waiting: int | None = None,
) -> dict:
    rec = _make_inference_record(run_id)
    rec["backend"] = backend
    rec["model"] = model
    if kv_cache_usage_pct is not None:
        rec["kv_cache_usage_pct"] = kv_cache_usage_pct
    if num_requests_running is not None:
        rec["num_requests_running"] = num_requests_running
    if num_requests_waiting is not None:
        rec["num_requests_waiting"] = num_requests_waiting
    return rec


class TestBuildKvCacheUsageTable(unittest.TestCase):
    def test_computes_avg_and_peak_by_model(self):
        records = [
            _make_inference_record_with_vllm("run-1", kv_cache_usage_pct=30.0),
            _make_inference_record_with_vllm("run-1", kv_cache_usage_pct=80.0),
            _make_inference_record_with_vllm("run-1", kv_cache_usage_pct=50.0),
        ]

        row = build_kv_cache_usage_table(records)[0]

        self.assertAlmostEqual(row["peak_kv_cache_pct"], 80.0)
        self.assertAlmostEqual(row["avg_kv_cache_pct"], (30.0 + 80.0 + 50.0) / 3)

    def test_empty_when_no_kv_cache_data(self):
        rows = build_kv_cache_usage_table([_make_inference_record_with_vllm("run-1")])
        self.assertEqual(rows, [])


class TestBuildQueueDepthTable(unittest.TestCase):
    def test_computes_avg_and_peak_by_model(self):
        records = [
            _make_inference_record_with_vllm("run-1", num_requests_running=1, num_requests_waiting=0),
            _make_inference_record_with_vllm("run-1", num_requests_running=3, num_requests_waiting=5),
            _make_inference_record_with_vllm("run-1", num_requests_running=2, num_requests_waiting=2),
        ]

        row = build_queue_depth_table(records)[0]

        self.assertEqual(row["peak_requests_waiting"], 5)
        self.assertAlmostEqual(row["avg_requests_waiting"], (0 + 5 + 2) / 3)
        self.assertEqual(row["peak_requests_running"], 3)

    def test_empty_when_no_queue_data(self):
        rows = build_queue_depth_table([_make_inference_record_with_vllm("run-1")])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
