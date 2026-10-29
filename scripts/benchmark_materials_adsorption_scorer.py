#!/usr/bin/env python3
"""Benchmark materials scorer latency, cache reuse, concurrency, and RSS."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_LINEAR_ALGEBRA_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
for _name in _LINEAR_ALGEBRA_THREAD_VARIABLES:
    os.environ.setdefault(_name, "1")

import psutil

from scripts.calibrate_materials_adsorption import _outcomes
from src.typing.training import save_durable_json
from tasks.materials_adsorption.compiler import RuleCompiler
from tasks.materials_adsorption.dataset import load_dataset
from tasks.materials_adsorption.scoring import PairwiseScorer, ScorerConfig


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(probability * len(ordered)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--submissions", type=int, default=8)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--gate-pool-size", type=int, default=2)
    parser.add_argument("--peak-submission-rate", type=float, default=80 / 600)
    parser.add_argument("--required-headroom", type=float, default=0.25)
    parser.add_argument("--skip-revision-latency", action="store_true")
    args = parser.parse_args()
    if args.submissions < max(args.pool_sizes):
        parser.error("--submissions must be at least the largest pool size")
    if args.gate_pool_size not in args.pool_sizes:
        parser.error("--gate-pool-size must be included in --pool-sizes")
    if args.peak_submission_rate <= 0 or args.required_headroom < 0:
        parser.error("capacity gate rates must be non-negative and peak rate must be positive")
    if any(os.environ[name] != "1" for name in _LINEAR_ALGEBRA_THREAD_VARIABLES):
        parser.error("capacity benchmarks require linear algebra thread variables set to 1")

    process = psutil.Process()
    load_started = time.perf_counter()
    loaded = load_dataset(args.dataset)
    outcomes = _outcomes(loaded)
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    compiler = RuleCompiler(loaded.registry)
    artifacts = {
        str(control["name"]): compiler.compile(str(control["expression"])).evaluate(
            loaded.universe.candidates,
            dataset_hash=loaded.universe.manifest_hash,
            universe_hash=loaded.universe.universe_hash,
        )
        for control in controls
        if not control.get("expect_compile_error")
    }
    load_seconds = time.perf_counter() - load_started
    scorer = PairwiseScorer(
        ScorerConfig(
            folds=args.folds,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    )
    candidate = artifacts["coordination_distance_interaction"]
    archive_candidates = [
        artifacts["minimum_contact"],
        artifacts["site_height"],
        artifacts["contact_coordination"],
    ]
    rss_after_load = process.memory_info().rss
    peak_rss = rss_after_load
    stop_sampling = threading.Event()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stop_sampling.wait(0.05):
            peak_rss = max(peak_rss, process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()

    revision_results = []
    for archive_width in (() if args.skip_revision_latency else (0, 1, 3)):
        archive = tuple(archive_candidates[:archive_width])
        started = time.perf_counter()
        cold_receipt = scorer.score(
            candidate,
            outcomes,
            archive=archive,
            archive_revision=archive_width,
        )
        cold_seconds = time.perf_counter() - started
        started = time.perf_counter()
        cached_receipt = scorer.score(
            candidate,
            outcomes,
            archive=archive,
            archive_revision=archive_width,
        )
        cached_seconds = time.perf_counter() - started
        if cold_receipt != cached_receipt:
            raise RuntimeError("cold and cached scorer receipts differ")
        revision_results.append(
            {
                "archive_width": archive_width,
                "cold_seconds": cold_seconds,
                "cached_seconds": cached_seconds,
                "receipt_hash": cold_receipt.receipt_hash,
            }
        )

    pool_results = []
    archive = tuple(archive_candidates)
    try:
        for pool_size in args.pool_sizes:
            submitted_at: list[float] = []

            def score_one(submission_index: int) -> dict[str, Any]:
                started = time.perf_counter()
                receipt = scorer.score(
                    candidate,
                    outcomes,
                    archive=archive,
                    archive_revision=3,
                )
                finished = time.perf_counter()
                return {
                    "submission_index": submission_index,
                    "queue_wait_seconds": started - submitted_at[submission_index],
                    "service_seconds": finished - started,
                    "receipt_hash": receipt.receipt_hash,
                }

            wall_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=pool_size) as pool:
                futures = []
                for submission_index in range(args.submissions):
                    submitted_at.append(time.perf_counter())
                    futures.append(pool.submit(score_one, submission_index))
                jobs = [future.result() for future in futures]
            wall_seconds = time.perf_counter() - wall_started
            service = [float(job["service_seconds"]) for job in jobs]
            queue_wait = [float(job["queue_wait_seconds"]) for job in jobs]
            pool_results.append(
                {
                    "pool_size": pool_size,
                    "submissions": args.submissions,
                    "wall_seconds": wall_seconds,
                    "throughput_per_second": args.submissions / wall_seconds,
                    "service_p50_seconds": statistics.median(service),
                    "service_p95_seconds": _percentile(service, 0.95),
                    "queue_wait_p50_seconds": statistics.median(queue_wait),
                    "queue_wait_p95_seconds": _percentile(queue_wait, 0.95),
                    "receipts_identical": len(
                        {str(job["receipt_hash"]) for job in jobs}
                    )
                    == 1,
                }
            )
    finally:
        stop_sampling.set()
        sampler.join()

    required_throughput = args.peak_submission_rate * (1.0 + args.required_headroom)
    gate_result = next(
        item for item in pool_results if item["pool_size"] == args.gate_pool_size
    )
    measured_throughput = float(gate_result["throughput_per_second"])
    capacity_passed = (
        measured_throughput >= required_throughput
        and bool(gate_result["receipts_identical"])
    )
    payload = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "systems": len({item.system_alias for item in outcomes}),
        "placements": len(outcomes),
        "load_and_compile_seconds": load_seconds,
        "rss_after_load_bytes": rss_after_load,
        "peak_rss_bytes": peak_rss,
        "scorer_config": {
            "folds": args.folds,
            "bootstrap_replicates": args.bootstrap_replicates,
            "max_pairs_per_system": scorer.config.max_pairs_per_system,
            "scorer_hash": scorer.config.scorer_hash,
        },
        "linear_algebra_threads": {
            name: os.environ.get(name)
            for name in _LINEAR_ALGEBRA_THREAD_VARIABLES
        },
        "revision_results": revision_results,
        "pool_results": pool_results,
        "capacity_gate": {
            "pool_size": args.gate_pool_size,
            "peak_submission_rate_per_second": args.peak_submission_rate,
            "required_headroom": args.required_headroom,
            "required_throughput_per_second": required_throughput,
            "measured_throughput_per_second": measured_throughput,
            "receipts_identical": gate_result["receipts_identical"],
            "passed": capacity_passed,
        },
        "baseline_cache": scorer.baseline_cache_info(),
        "pair_topology_cache": scorer.pair_topology_cache_info(),
    }
    save_durable_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["capacity_gate"]["passed"]:
        raise SystemExit("materials adsorption scorer failed the capacity headroom gate")


if __name__ == "__main__":
    main()
