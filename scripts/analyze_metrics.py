from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Sequence


def discover_metrics_files(paths: Sequence[str | Path]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.name.startswith("metrics_") and path.suffix == ".jsonl":
                discovered.add(path.resolve())
            continue
        if path.is_dir():
            for match in path.rglob("metrics_*.jsonl"):
                if match.is_file():
                    discovered.add(match.resolve())
    return sorted(discovered)


RUN_FILENAME = "run.json"


def discover_run_files(paths: Sequence[str | Path]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.name == RUN_FILENAME:
                discovered.add(path.resolve())
            continue
        if path.is_dir():
            for match in path.rglob(RUN_FILENAME):
                if match.is_file():
                    discovered.add(match.resolve())
    return sorted(discovered)


def load_run_doc(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"run.json must contain an object: {file_path}")
    return payload


_GENERATION_METRIC_KEYS = (
    "generation_id",
    "total_episodes_run",
    "total_successful",
    "success_rate",
    "episodes_per_minute",
    "training_rows_per_minute",
    "tokens_per_successful_episode",
    "average_inference_duty_cycle",
    "average_gpu_utilization_pct",
    "average_cpu_utilization_pct",
    "peak_gpu_utilization_pct",
    "peak_cpu_utilization_pct",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "peak_context_tokens",
    "avg_context_tokens",
    "median_context_tokens",
    "total_agent_seconds",
    "average_output_tokens_per_second",
    "total_inference_seconds",
    "generation_wall_clock_seconds",
)


def summarize_run_doc(run_doc: dict[str, Any]) -> dict[str, Any]:
    generations = []
    for gen in run_doc.get("generations", []):
        metrics = gen.get("metrics", {})
        gen_summary: dict[str, Any] = {}
        for key in _GENERATION_METRIC_KEYS:
            if key in metrics:
                gen_summary[key] = metrics[key]
        generations.append(gen_summary)

    return {
        "run_id": run_doc.get("run_id", "unknown"),
        "status": run_doc.get("status"),
        "started_at": run_doc.get("started_at"),
        "ended_at": run_doc.get("ended_at"),
        "harness_type": run_doc.get("harness_type"),
        "harness_profile": run_doc.get("harness_profile"),
        "harness_class": run_doc.get("harness_class"),
        "task_class": run_doc.get("task_class"),
        "task_name": run_doc.get("task_name"),
        "config_path": run_doc.get("config_path"),
        "model_name": run_doc.get("model_name"),
        "commit_id": run_doc.get("commit_id"),
        "git_dirty": run_doc.get("git_dirty"),
        "hardware_tags": _normalize_tags(run_doc.get("hardware_tags")),
        "load_tags": _normalize_tags(run_doc.get("load_tags")),
        "num_generations": run_doc.get("num_generations", 0),
        "training_export_format": run_doc.get("training_export_format", "unknown"),
        "total_wall_clock_seconds": run_doc.get("total_wall_clock_seconds", 0.0),
        "total_agent_seconds": run_doc.get("total_agent_seconds", 0.0),
        "total_tokens": run_doc.get("total_tokens", 0),
        "generations": generations,
    }


def build_generation_comparison_table(
    run_doc_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in run_doc_summaries:
        run_id = summary.get("run_id", "unknown")
        for gen in summary.get("generations", []):
            row: dict[str, Any] = {"run_id": run_id}
            for key in _GENERATION_METRIC_KEYS:
                row[key] = gen.get(key)
            rows.append(row)
    return rows


def _select_run_ids(
    run_docs: dict[str, dict[str, Any]],
    *,
    run_id: str | None,
    harness_profile: list[str] | None,
    task: list[str] | None,
    since: date | None,
    until: date | None,
) -> set[str]:
    selected = set(run_docs)
    if run_id:
        selected = {run_id} if run_id in run_docs else set()

    if harness_profile:
        profiles = {item.strip() for item in harness_profile if item.strip()}
        selected = {
            rid
            for rid in selected
            if str(run_docs[rid].get("harness_profile") or "") in profiles
        }

    if task:
        task_values = {item.strip() for item in task if item.strip()}
        selected = {
            rid
            for rid in selected
            if str(run_docs[rid].get("task_class") or "") in task_values
            or str(run_docs[rid].get("task_name") or "") in task_values
        }

    if since is not None or until is not None:
        date_filtered: set[str] = set()
        for rid in selected:
            started_date = _run_started_utc_date(run_docs[rid].get("started_at"))
            if since is not None and started_date < since:
                continue
            if until is not None and started_date >= until:
                continue
            date_filtered.add(rid)
        selected = date_filtered

    return selected


def analyze_paths(
    paths: Sequence[str | Path],
    run_id: str | None = None,
    harness_profile: list[str] | None = None,
    task: list[str] | None = None,
    since: date | None = None,
    until: date | None = None,
) -> dict[str, Any]:
    metric_files = discover_metrics_files(paths)
    run_files = discover_run_files(paths)
    run_docs: dict[str, dict[str, Any]] = {}
    for run_file in run_files:
        run_doc = load_run_doc(run_file)
        doc_run_id = run_doc.get("run_id")
        if isinstance(doc_run_id, str) and doc_run_id:
            run_docs[doc_run_id] = run_doc

    selected_run_ids = _select_run_ids(
        run_docs,
        run_id=run_id,
        harness_profile=harness_profile,
        task=task,
        since=since,
        until=until,
    )
    run_doc_summaries = [
        summarize_run_doc(run_docs[rid]) for rid in sorted(selected_run_ids)
    ]

    records_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_inference_records: list[dict[str, Any]] = []
    all_phase_records: list[dict[str, Any]] = []
    contributing_files: set[Path] = set()
    metric_run_ids: set[str] = set()

    for metric_file in metric_files:
        for record in load_metrics_file(metric_file):
            record_run_id = record.get("run_id")
            if not isinstance(record_run_id, str) or not record_run_id:
                continue
            metric_run_ids.add(record_run_id)
            if record_run_id not in selected_run_ids:
                continue
            contributing_files.add(metric_file)
            records_by_run[record_run_id].append(record)
            metric_type = record.get("metric_type")
            if metric_type == "inference":
                all_inference_records.append(record)
            elif metric_type == "phase":
                all_phase_records.append(record)

    run_summaries = [
        summarize_run(rid, records_by_run.get(rid, []), run_docs[rid])
        for rid in sorted(selected_run_ids)
    ]

    return {
        "run_count": len(run_summaries),
        "run_file_count": len(selected_run_ids),
        "run_file_count_total": len(run_files),
        "metric_run_without_run_json_count": len(metric_run_ids - set(run_docs)),
        "metric_file_count": len(contributing_files),
        "metric_file_count_total": len(metric_files),
        "runs": run_summaries,
        "run_docs": run_doc_summaries,
        "comparison_tables": {
            "latency_by_model": build_latency_by_model_table(
                run_summaries, all_inference_records
            ),
            "retry_rates_by_phase": build_retry_rates_by_phase_table(all_phase_records),
            "memory_profiles_by_model": build_memory_profiles_by_model_table(
                all_inference_records
            ),
            "generation_benchmarks_by_backend": build_generation_benchmarks_by_backend_table(
                all_inference_records
            ),
            "generation_benchmarks_by_hardware_tag": build_generation_benchmarks_by_tag_table(
                all_inference_records,
                run_docs,
                tag_metadata_key="hardware_tags",
                tag_output_key="hardware_tag",
            ),
            "generation_benchmarks_by_load_tag": build_generation_benchmarks_by_tag_table(
                all_inference_records,
                run_docs,
                tag_metadata_key="load_tags",
                tag_output_key="load_tag",
            ),
            "generation_benchmarks_by_harness_profile_task": build_generation_benchmarks_by_harness_profile_task_table(
                all_inference_records,
                run_docs,
            ),
            "generation_comparison": build_generation_comparison_table(
                run_doc_summaries
            ),
            "kv_cache_usage": build_kv_cache_usage_table(all_inference_records),
            "queue_depth": build_queue_depth_table(all_inference_records),
        },
    }


def load_metrics_file(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def summarize_run(
    run_id: str,
    records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    inference_records = [
        record for record in records if record.get("metric_type") == "inference"
    ]
    phase_records = [
        record for record in records if record.get("metric_type") == "phase"
    ]

    inference_calls = len(inference_records)
    successful_inference_calls = sum(
        1 for record in inference_records if record.get("success")
    )
    failed_inference_calls = inference_calls - successful_inference_calls
    phase_count = len(phase_records)
    total_retry_count = sum(
        _as_int(record.get("retry_count")) for record in phase_records
    )

    total_inference_latency_ms = sum(
        _as_float(record.get("latency_ms")) for record in inference_records
    )
    average_inference_latency_ms = (
        total_inference_latency_ms / inference_calls if inference_calls else 0.0
    )
    total_phase_duration_ms = sum(
        _as_float(record.get("duration_ms")) for record in phase_records
    )

    usage_dicts = [_usage(record) for record in inference_records]
    total_input_tokens = sum(
        _as_int(usage.get("prompt_tokens")) for usage in usage_dicts
    )
    total_output_tokens = sum(
        _as_int(usage.get("completion_tokens")) for usage in usage_dicts
    )
    total_tokens = sum(_usage_total_tokens(usage) for usage in usage_dicts)

    throughput_output_tokens_per_second = 0.0
    if total_inference_latency_ms > 0:
        throughput_output_tokens_per_second = total_output_tokens / (
            total_inference_latency_ms / 1000
        )

    prompt_rates = _collect_generation_rates(
        inference_records, "prompt_tokens_per_second"
    )
    output_rates = _collect_generation_rates(
        inference_records, "output_tokens_per_second"
    )
    total_rates = _collect_generation_rates(
        inference_records, "total_tokens_per_second"
    )

    host_memory_samples = [
        _as_float(record.get("host_memory_mb"))
        for record in inference_records
        if _has_number(record.get("host_memory_mb"))
    ]
    gpu_memory_samples = [
        _as_float(record.get("gpu_memory_mb"))
        for record in inference_records
        if _has_number(record.get("gpu_memory_mb"))
    ]

    hardware_tags = _normalize_tags(metadata.get("hardware_tags"))
    load_tags = _normalize_tags(metadata.get("load_tags"))
    backend = infer_backend(inference_records)
    model = infer_model(inference_records)
    if model == "unknown" and metadata.get("model_name"):
        model = str(metadata.get("model_name"))

    return {
        "run_id": run_id,
        "status": metadata.get("status"),
        "started_at": metadata.get("started_at"),
        "ended_at": metadata.get("ended_at"),
        "harness_type": metadata.get("harness_type"),
        "harness_profile": metadata.get("harness_profile"),
        "harness_class": metadata.get("harness_class"),
        "task_class": metadata.get("task_class"),
        "task_name": metadata.get("task_name"),
        "config_path": metadata.get("config_path"),
        "model_name": metadata.get("model_name"),
        "commit_id": metadata.get("commit_id"),
        "git_dirty": metadata.get("git_dirty"),
        "hardware_tags": hardware_tags,
        "load_tags": load_tags,
        "backend": backend,
        "model": model,
        "inference_calls": inference_calls,
        "successful_inference_calls": successful_inference_calls,
        "failed_inference_calls": failed_inference_calls,
        "success_rate": (
            successful_inference_calls / inference_calls if inference_calls else 0.0
        ),
        "phase_count": phase_count,
        "total_retry_count": total_retry_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_inference_latency_ms": total_inference_latency_ms,
        "average_inference_latency_ms": average_inference_latency_ms,
        "total_phase_duration_ms": total_phase_duration_ms,
        "throughput_output_tokens_per_second": throughput_output_tokens_per_second,
        "avg_per_call_output_tokens_per_second": _average_or_zero(output_rates),
        "median_per_call_output_tokens_per_second": _median_or_none(output_rates),
        "generation_count": len(total_rates),
        "average_prompt_tokens_per_second": _average_or_zero(prompt_rates),
        "average_total_tokens_per_second": _average_or_zero(total_rates),
        "median_total_tokens_per_second": _median_or_none(total_rates),
        "peak_total_tokens_per_second": max(total_rates) if total_rates else None,
        "average_host_memory_mb": _average_or_none(host_memory_samples),
        "peak_host_memory_mb": max(host_memory_samples)
        if host_memory_samples
        else None,
        "average_gpu_memory_mb": _average_or_none(gpu_memory_samples),
        "peak_gpu_memory_mb": max(gpu_memory_samples) if gpu_memory_samples else None,
    }


def build_latency_by_model_table(
    run_summaries: Sequence[dict[str, Any]],
    inference_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in run_summaries:
        grouped[str(summary.get("model", "unknown"))].append(summary)

    # Group raw inference records by model for per-call stats
    run_model_map = infer_models_by_run(inference_records)
    records_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id in run_model_map:
            model = run_model_map[run_id]
        else:
            model = infer_model([record])
        records_by_model[model].append(record)

    rows: list[dict[str, Any]] = []
    for model, summaries in sorted(grouped.items()):
        total_inference_calls = sum(
            _as_int(item.get("inference_calls")) for item in summaries
        )
        total_latency_ms = sum(
            _as_float(item.get("total_inference_latency_ms")) for item in summaries
        )
        total_output_tokens = sum(
            _as_int(item.get("total_output_tokens")) for item in summaries
        )
        average_latency_ms = (
            total_latency_ms / total_inference_calls if total_inference_calls else 0.0
        )
        throughput_output_tokens_per_second = 0.0
        if total_latency_ms > 0:
            throughput_output_tokens_per_second = total_output_tokens / (
                total_latency_ms / 1000
            )

        # Per-call stats from raw inference records
        model_records = records_by_model.get(model, [])
        per_call_output_rates = _collect_generation_rates(
            model_records, "output_tokens_per_second"
        )

        rows.append(
            {
                "model": model,
                "run_count": len(summaries),
                "total_inference_calls": total_inference_calls,
                "total_latency_ms": total_latency_ms,
                "average_latency_ms": average_latency_ms,
                "throughput_output_tokens_per_second": throughput_output_tokens_per_second,
                "avg_per_call_output_tokens_per_second": _average_or_zero(
                    per_call_output_rates
                ),
                "median_per_call_output_tokens_per_second": _median_or_none(
                    per_call_output_rates
                ),
            }
        )
    return rows


def build_retry_rates_by_phase_table(
    phase_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in phase_records:
        phase_name = record.get("phase_name")
        if isinstance(phase_name, str) and phase_name:
            grouped[phase_name].append(record)

    rows: list[dict[str, Any]] = []
    for phase_name, records in sorted(grouped.items()):
        total_retry_count = sum(
            _as_int(record.get("retry_count")) for record in records
        )
        runs_with_retries = {
            str(record.get("run_id"))
            for record in records
            if _as_int(record.get("retry_count")) > 0 and record.get("run_id")
        }
        failure_count = sum(1 for record in records if not record.get("success"))
        rows.append(
            {
                "phase_name": phase_name,
                "phase_occurrences": len(records),
                "run_count": len(
                    {
                        str(record.get("run_id"))
                        for record in records
                        if record.get("run_id")
                    }
                ),
                "total_retry_count": total_retry_count,
                "average_retry_count": total_retry_count / len(records)
                if records
                else 0.0,
                "runs_with_retries": len(runs_with_retries),
                "failure_count": failure_count,
                "failure_rate": failure_count / len(records) if records else 0.0,
            }
        )
    return rows


def build_memory_profiles_by_model_table(
    inference_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_model_map = infer_models_by_run(inference_records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id in run_model_map:
            model = run_model_map[run_id]
        else:
            model = infer_model([record])
        grouped[model].append(record)

    rows: list[dict[str, Any]] = []
    for model, records in sorted(grouped.items()):
        host_memory_samples = [
            _as_float(record.get("host_memory_mb"))
            for record in records
            if _has_number(record.get("host_memory_mb"))
        ]
        gpu_memory_samples = [
            _as_float(record.get("gpu_memory_mb"))
            for record in records
            if _has_number(record.get("gpu_memory_mb"))
        ]
        rows.append(
            {
                "model": model,
                "run_count": len(
                    {
                        str(record.get("run_id"))
                        for record in records
                        if record.get("run_id")
                    }
                ),
                "sample_count": len(records),
                "average_host_memory_mb": _average_or_none(host_memory_samples),
                "peak_host_memory_mb": max(host_memory_samples)
                if host_memory_samples
                else None,
                "average_gpu_memory_mb": _average_or_none(gpu_memory_samples),
                "peak_gpu_memory_mb": max(gpu_memory_samples)
                if gpu_memory_samples
                else None,
            }
        )
    return rows


def build_kv_cache_usage_table(
    inference_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Average and peak KV cache usage by model/backend."""
    run_model_map = infer_models_by_run(inference_records)
    grouped: dict[str, list[float]] = defaultdict(list)
    run_sets: dict[str, set[str]] = defaultdict(set)
    for record in inference_records:
        kv = record.get("kv_cache_usage_pct")
        if not _has_number(kv):
            continue
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id in run_model_map:
            model = run_model_map[run_id]
        else:
            model = infer_model([record])
        grouped[model].append(float(kv))
        if isinstance(run_id, str):
            run_sets[model].add(run_id)

    rows: list[dict[str, Any]] = []
    for model, samples in sorted(grouped.items()):
        rows.append(
            {
                "model": model,
                "run_count": len(run_sets.get(model, set())),
                "sample_count": len(samples),
                "avg_kv_cache_pct": _average_or_none(samples),
                "peak_kv_cache_pct": max(samples) if samples else None,
            }
        )
    return rows


def build_queue_depth_table(
    inference_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Average and peak queue depth by model/backend."""
    run_model_map = infer_models_by_run(inference_records)
    grouped_running: dict[str, list[int]] = defaultdict(list)
    grouped_waiting: dict[str, list[int]] = defaultdict(list)
    run_sets: dict[str, set[str]] = defaultdict(set)
    has_data: set[str] = set()
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id in run_model_map:
            model = run_model_map[run_id]
        else:
            model = infer_model([record])
        running = record.get("num_requests_running")
        waiting = record.get("num_requests_waiting")
        if _has_number(running):
            grouped_running[model].append(int(running))
            has_data.add(model)
        if _has_number(waiting):
            grouped_waiting[model].append(int(waiting))
            has_data.add(model)
        if isinstance(run_id, str) and model in has_data:
            run_sets[model].add(run_id)

    rows: list[dict[str, Any]] = []
    for model in sorted(has_data):
        running_samples = grouped_running.get(model, [])
        waiting_samples = grouped_waiting.get(model, [])
        rows.append(
            {
                "model": model,
                "run_count": len(run_sets.get(model, set())),
                "avg_requests_running": _average_or_none(
                    [float(x) for x in running_samples]
                ),
                "peak_requests_running": max(running_samples)
                if running_samples
                else None,
                "avg_requests_waiting": _average_or_none(
                    [float(x) for x in waiting_samples]
                ),
                "peak_requests_waiting": max(waiting_samples)
                if waiting_samples
                else None,
            }
        )
    return rows


def build_generation_benchmarks_by_backend_table(
    inference_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_backend_map = infer_backends_by_run(inference_records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id in run_backend_map:
            backend = run_backend_map[run_id]
        else:
            backend = infer_backend([record])
        grouped[backend].append(record)

    rows: list[dict[str, Any]] = []
    for backend, records in sorted(grouped.items()):
        row = _build_generation_benchmark_row(records)
        row["backend"] = backend
        rows.append(row)
    return rows


def build_generation_benchmarks_by_tag_table(
    inference_records: Sequence[dict[str, Any]],
    run_docs: dict[str, dict[str, Any]],
    tag_metadata_key: str,
    tag_output_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        tags = _normalize_tags(run_docs.get(run_id, {}).get(tag_metadata_key))
        for tag in tags:
            grouped[tag].append(record)

    rows: list[dict[str, Any]] = []
    for tag, records in sorted(grouped.items()):
        row = _build_generation_benchmark_row(records)
        row[tag_output_key] = tag
        rows.append(row)
    return rows


def build_generation_benchmarks_by_harness_profile_task_table(
    inference_records: Sequence[dict[str, Any]],
    run_docs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        run_doc = run_docs.get(run_id, {})
        harness_profile = str(run_doc.get("harness_profile") or "unknown")
        task_name = str(run_doc.get("task_name") or run_doc.get("task_class") or "unknown")
        grouped[(harness_profile, task_name)].append(record)

    rows: list[dict[str, Any]] = []
    for (harness_profile, task_name), records in sorted(grouped.items()):
        row = _build_generation_benchmark_row(records)
        row["harness_profile"] = harness_profile
        row["task_name"] = task_name
        rows.append(row)
    return rows


def _build_generation_benchmark_row(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    prompt_rates = _collect_generation_rates(records, "prompt_tokens_per_second")
    output_rates = _collect_generation_rates(records, "output_tokens_per_second")
    total_rates = _collect_generation_rates(records, "total_tokens_per_second")
    latency_samples = [
        _as_float(record.get("latency_ms"))
        for record in records
        if _has_number(record.get("latency_ms"))
    ]
    return {
        "run_count": len(
            {str(record.get("run_id")) for record in records if record.get("run_id")}
        ),
        "inference_call_count": len(records),
        "generation_count": len(total_rates),
        "success_rate": (
            sum(1 for record in records if record.get("success")) / len(records)
            if records
            else 0.0
        ),
        "average_latency_ms": _average_or_zero(latency_samples),
        "median_latency_ms": _median_or_none(latency_samples),
        "p95_latency_ms": _percentile_or_none(latency_samples, 95.0),
        "average_prompt_tokens_per_second": _average_or_zero(prompt_rates),
        "median_prompt_tokens_per_second": _median_or_none(prompt_rates),
        "peak_prompt_tokens_per_second": max(prompt_rates) if prompt_rates else None,
        "average_output_tokens_per_second": _average_or_zero(output_rates),
        "median_output_tokens_per_second": _median_or_none(output_rates),
        "peak_output_tokens_per_second": max(output_rates) if output_rates else None,
        "average_total_tokens_per_second": _average_or_zero(total_rates),
        "median_total_tokens_per_second": _median_or_none(total_rates),
        "peak_total_tokens_per_second": max(total_rates) if total_rates else None,
    }


def infer_backend(inference_records: Sequence[dict[str, Any]]) -> str:
    candidates = [
        str(record.get("backend"))
        for record in inference_records
        if record.get("backend") not in (None, "")
    ]
    if not candidates:
        return "unknown"
    return Counter(candidates).most_common(1)[0][0]


def infer_model(inference_records: Sequence[dict[str, Any]]) -> str:
    candidates = [
        str(record.get("model") or _usage(record).get("model"))
        for record in inference_records
        if (record.get("model") or _usage(record).get("model")) not in (None, "")
    ]
    if not candidates:
        return "unknown"
    return Counter(candidates).most_common(1)[0][0]


def infer_backends_by_run(
    inference_records: Sequence[dict[str, Any]],
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id:
            grouped[run_id].append(record)

    return {run_id: infer_backend(records) for run_id, records in grouped.items()}


def infer_models_by_run(
    inference_records: Sequence[dict[str, Any]],
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id:
            grouped[run_id].append(record)

    return {run_id: infer_model(records) for run_id, records in grouped.items()}


_ORCHESTRATION_TABLE_ROWS: list[tuple[str, str, str | None]] = [
    # (label, metric_key, total_mode)
    # total_mode: "sum", "avg", "max", None (show dash), "wall_clock" (special)
    ("Episodes", "total_episodes_run", "sum"),
    ("Successful", "total_successful", "sum"),
    ("Success rate", "success_rate", "avg"),
    ("Tokens (in+out)", "total_tokens", "sum"),
    ("Agent seconds", "total_agent_seconds", "sum"),
    ("Gen inference time", "total_inference_seconds", "sum"),
    ("Wall clock", None, "wall_clock"),
    ("Peak GPU%", "peak_gpu_utilization_pct", "max"),
    ("Avg GPU%", "average_gpu_utilization_pct", None),
    ("Tokens/successful ep", "tokens_per_successful_episode", None),
    ("Peak context tokens", "peak_context_tokens", "max"),
    ("Avg context tokens", "avg_context_tokens", None),
    ("Inference duty cycle", "average_inference_duty_cycle", None),
]


def _fmt_cell(value: Any, key: str) -> str:
    if value is None:
        return "—"
    if key == "success_rate":
        return f"{value * 100:.0f}%"
    if key in ("peak_gpu_utilization_pct", "average_gpu_utilization_pct"):
        return f"{value:.1f}%"
    if key in ("total_agent_seconds", "total_inference_seconds"):
        return f"{value:.1f}s"
    if key == "average_inference_duty_cycle":
        return f"{value:.2f}"
    if key in ("avg_context_tokens",):
        return f"{value:.1f}"
    if isinstance(value, float):
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_run_table(summary: dict[str, Any]) -> str:
    run_id = summary.get("run_id", "unknown")
    generations = summary.get("generations", [])
    if not generations:
        return ""

    gen_headers = [f"Gen {g.get('generation_id', '?')}" for g in generations]
    all_headers = ["Metric"] + gen_headers + ["Total"]

    rows: list[list[str]] = []
    for label, key, total_mode in _ORCHESTRATION_TABLE_ROWS:
        cells = [label]
        values = []
        for gen in generations:
            val = gen.get(key) if key else None
            values.append(val)
            cells.append(_fmt_cell(val, key or ""))

        # Compute total column
        if total_mode == "wall_clock":
            cells = [label] + ["—"] * len(generations)
            wc = summary.get("total_wall_clock_seconds")
            cells.append(f"{wc:.1f}s" if wc is not None else "—")
        elif total_mode == "sum":
            numeric = [v for v in values if v is not None]
            total = sum(numeric) if numeric else None
            total_cell = _fmt_cell(total, key or "")
            # Annotate with coverage when some generations lack data
            if numeric and len(numeric) < len(values):
                total_cell += f" ({len(numeric)}/{len(values)} gens)"
            cells.append(total_cell)
        elif total_mode == "avg":
            numeric = [v for v in values if v is not None]
            total = sum(numeric) / len(numeric) if numeric else None
            if key == "success_rate" and total is not None:
                cells.append(f"~{total * 100:.0f}%")
            else:
                cells.append(_fmt_cell(total, key or ""))
        elif total_mode == "max":
            numeric = [v for v in values if v is not None]
            total = max(numeric) if numeric else None
            cells.append(_fmt_cell(total, key or ""))
        else:
            cells.append("—")

        rows.append(cells)

    # Compute column widths
    col_widths = [len(h) for h in all_headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        padded = [cells[i].ljust(col_widths[i]) for i in range(len(cells))]
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    lines = [
        f"### Run {run_id} Summary",
        "",
        _fmt_row(all_headers),
        separator,
    ]
    for row in rows:
        lines.append(_fmt_row(row))

    return "\n".join(lines)


def render_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Metrics analysis")
    run_file_count = report.get("run_file_count", report["run_count"])
    metric_only_count = report.get("metric_run_without_run_json_count", 0)
    lines.append(
        f"Runs: {report['run_count']} ({run_file_count} with run.json, {metric_only_count} metric-only)"
    )
    metric_file_count = report["metric_file_count"]
    metric_file_count_total = report.get("metric_file_count_total", metric_file_count)
    if metric_file_count != metric_file_count_total:
        lines.append(
            f"Metrics files: {metric_file_count} (of {metric_file_count_total} discovered)"
        )
    else:
        lines.append(f"Metrics files: {metric_file_count}")
    lines.append("")
    lines.append("Runs")
    for run in report["runs"]:
        lines.append(
            " - "
            f"{run['run_id']} | harness_profile={run.get('harness_profile') or 'n/a'} | "
            f"task={run.get('task_name') or run.get('task_class') or 'n/a'} | "
            f"hw=[{_format_tags(run['hardware_tags'])}] | load=[{_format_tags(run['load_tags'])}] | "
            f"backend={run['backend']} | model={run['model']} | commit={run.get('commit_id') or 'unknown'} | "
            f"inference_calls={run['inference_calls']} | avg_latency_ms={run['average_inference_latency_ms']:.2f} | "
            f"avg_total_tok_s={run['average_total_tokens_per_second']:.2f}"
        )
    lines.append("")
    lines.append("Latency by model")
    for row in report["comparison_tables"]["latency_by_model"]:
        lines.append(
            " - "
            f"{row['model']} | runs={row['run_count']} | total_inference_calls={row['total_inference_calls']} | "
            f"avg_latency_ms={row['average_latency_ms']:.2f} | "
            f"throughput_tok_s={row['throughput_output_tokens_per_second']:.2f} | "
            f"avg_per_call_tok_s={row['avg_per_call_output_tokens_per_second']:.2f} | "
            f"median_per_call_tok_s={_format_optional_float(row['median_per_call_output_tokens_per_second'])}"
        )
    lines.append("")
    lines.append("Generation benchmarks by backend")
    for row in report["comparison_tables"]["generation_benchmarks_by_backend"]:
        lines.append(
            " - "
            f"{row['backend']} | runs={row['run_count']} | generations={row['generation_count']} | "
            f"avg_output_tok_s={row['average_output_tokens_per_second']:.2f} | "
            f"avg_total_tok_s={row['average_total_tokens_per_second']:.2f} | "
            f"median_total_tok_s={_format_optional_float(row['median_total_tokens_per_second'])} | "
            f"p95_latency_ms={_format_optional_float(row['p95_latency_ms'])}"
        )
    lines.append("")
    lines.append("Generation benchmarks by harness profile/task")
    for row in report["comparison_tables"]["generation_benchmarks_by_harness_profile_task"]:
        lines.append(
            " - "
            f"{row['harness_profile']} | task={row['task_name']} | runs={row['run_count']} | generations={row['generation_count']} | "
            f"avg_total_tok_s={row['average_total_tokens_per_second']:.2f} | "
            f"p95_latency_ms={_format_optional_float(row['p95_latency_ms'])}"
        )
    lines.append("")
    lines.append("Generation benchmarks by hardware tag")
    for row in report["comparison_tables"]["generation_benchmarks_by_hardware_tag"]:
        lines.append(
            " - "
            f"{row['hardware_tag']} | runs={row['run_count']} | generations={row['generation_count']} | "
            f"avg_total_tok_s={row['average_total_tokens_per_second']:.2f} | "
            f"p95_latency_ms={_format_optional_float(row['p95_latency_ms'])}"
        )
    lines.append("")
    lines.append("Generation benchmarks by load tag")
    for row in report["comparison_tables"]["generation_benchmarks_by_load_tag"]:
        lines.append(
            " - "
            f"{row['load_tag']} | runs={row['run_count']} | generations={row['generation_count']} | "
            f"avg_total_tok_s={row['average_total_tokens_per_second']:.2f} | "
            f"p95_latency_ms={_format_optional_float(row['p95_latency_ms'])}"
        )
    lines.append("")
    lines.append("Retry rates by phase")
    for row in report["comparison_tables"]["retry_rates_by_phase"]:
        lines.append(
            " - "
            f"{row['phase_name']} | occurrences={row['phase_occurrences']} | total_retry_count={row['total_retry_count']} | "
            f"avg_retry_count={row['average_retry_count']:.2f} | failure_rate={row['failure_rate']:.2%}"
        )
    lines.append("")
    lines.append("Memory profiles by model")
    for row in report["comparison_tables"]["memory_profiles_by_model"]:
        lines.append(
            " - "
            f"{row['model']} | avg_host_memory_mb={_format_optional_float(row['average_host_memory_mb'])} | "
            f"peak_host_memory_mb={_format_optional_float(row['peak_host_memory_mb'])} | "
            f"avg_gpu_memory_mb={_format_optional_float(row['average_gpu_memory_mb'])} | "
            f"peak_gpu_memory_mb={_format_optional_float(row['peak_gpu_memory_mb'])}"
        )
    kv_cache_rows = report["comparison_tables"].get("kv_cache_usage", [])
    queue_depth_rows = report["comparison_tables"].get("queue_depth", [])
    if kv_cache_rows or queue_depth_rows:
        lines.append("")
        lines.append("vLLM server metrics")
        for row in kv_cache_rows:
            lines.append(
                " - "
                f"{row['model']} | "
                f"avg_kv_cache_pct={_format_optional_float(row['avg_kv_cache_pct'])} | "
                f"peak_kv_cache_pct={_format_optional_float(row['peak_kv_cache_pct'])} | "
                f"samples={row['sample_count']}"
            )
        for row in queue_depth_rows:
            lines.append(
                " - "
                f"{row['model']} | "
                f"avg_queue_depth={_format_optional_float(row['avg_requests_waiting'])} | "
                f"peak_queue_depth={row['peak_requests_waiting']}"
            )
    for run_doc in report.get("run_docs", []):
        table = render_run_table(run_doc)
        if table:
            lines.append("")
            lines.append(table)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["data"])
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Filter results to a specific run ID",
    )
    parser.add_argument("--harness-profile", action="append", default=None)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--since", type=_parse_iso_date, default=None)
    parser.add_argument("--until", type=_parse_iso_date, default=None)
    args = parser.parse_args(argv)

    report = analyze_paths(
        args.paths,
        run_id=args.run_id,
        harness_profile=args.harness_profile,
        task=args.task,
        since=args.since,
        until=args.until,
    )
    if args.format == "json":
        rendered = json.dumps(report, indent=args.indent, sort_keys=True)
    else:
        rendered = render_text_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8"
        )
    else:
        print(rendered)

    return 0


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD date, got {value!r}"
        ) from exc


def _run_started_utc_date(value: Any) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError("run.json started_at must be an offset-aware UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid run.json started_at timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("run.json started_at must be an offset-aware UTC timestamp")
    return parsed.date()


def _usage(record: dict[str, Any]) -> dict[str, Any]:
    usage = record.get("usage")
    if isinstance(usage, dict):
        return usage
    return {}


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = re.sub(r"[^a-z0-9]+", "-", item.strip().lower()).strip("-")
        if not tag or tag in seen:
            continue
        normalized.append(tag)
        seen.add(tag)
    return normalized


def _collect_generation_rates(
    records: Sequence[dict[str, Any]],
    rate_key: str,
) -> list[float]:
    rates: list[float] = []
    for record in records:
        rate = _generation_rate(record, rate_key)
        if rate is not None:
            rates.append(rate)
    return rates


def _generation_rate(record: dict[str, Any], rate_key: str) -> float | None:
    precomputed = record.get(rate_key)
    if _has_number(precomputed):
        return float(precomputed)

    usage = _usage(record)
    latency_ms = _as_float(record.get("latency_ms"))
    if latency_ms <= 0:
        return None

    latency_seconds = latency_ms / 1000
    prompt_tokens = _optional_int(usage.get("prompt_tokens"))
    completion_tokens = _optional_int(usage.get("completion_tokens"))
    total_tokens = _optional_int(usage.get("total_tokens"))
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    if rate_key == "prompt_tokens_per_second" and prompt_tokens is not None:
        return float(prompt_tokens) / latency_seconds
    if rate_key == "output_tokens_per_second" and completion_tokens is not None:
        return float(completion_tokens) / latency_seconds
    if rate_key == "total_tokens_per_second" and total_tokens is not None:
        return float(total_tokens) / latency_seconds
    return None


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    total_tokens = usage.get("total_tokens")
    if _has_number(total_tokens):
        return _as_int(total_tokens)
    return _as_int(usage.get("prompt_tokens")) + _as_int(usage.get("completion_tokens"))


def _average_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _average_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _percentile_or_none(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    percentile = min(max(percentile, 0.0), 100.0)
    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = rank - lower_index
    return float(lower_value + (upper_value - lower_value) * fraction)


def _has_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int(value: Any) -> int:
    if _has_number(value):
        return int(value)
    return 0


def _optional_int(value: Any) -> int | None:
    if _has_number(value):
        return int(value)
    return None


def _as_float(value: Any) -> float:
    if _has_number(value):
        return float(value)
    return 0.0


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _format_tags(tags: Any) -> str:
    normalized = _normalize_tags(tags)
    if not normalized:
        return "n/a"
    return ",".join(normalized)


if __name__ == "__main__":
    raise SystemExit(main())
