"""Freeze the pooled shard-null percentile calibration for feature probes."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.calibrate_materials_adsorption import _outcomes
from src.typing.training import save_durable_json
from tasks.materials_adsorption.compiler import ConsequenceArtifact, RuleCompiler
from tasks.materials_adsorption.dataset import load_dataset
from tasks.materials_adsorption.probe_calibration import (
    SHARD_NULL_PERMUTATIONS,
    SHARD_NULL_PERMUTATION_SEED,
    SHARD_PARTITION_VERSION,
    ShardNullCalibration,
)
from tasks.materials_adsorption.scoring import (
    PairwiseScorer,
    PermutationNullCalibration,
    PlacementOutcome,
    ScorerConfig,
    _artifact_feature_column,
    _build_pair_topology,
    _stable_digest,
)
from tasks.materials_adsorption.task import MaterialsAdsorptionTask


_WORKER_SCORER: PairwiseScorer | None = None
_WORKER_REPRESENTATIVES: tuple[tuple[ConsequenceArtifact, ...], ...] = ()
_WORKER_INVERSES: tuple[np.ndarray, ...] = ()
_WORKER_COMPLEXITIES: np.ndarray | None = None
_WORKER_SHARD_SYSTEMS: tuple[tuple[tuple[PlacementOutcome, ...], ...], ...] = ()


def _load_inputs(
    dataset_path: Path,
    permutation_null_path: Path,
    attainability_path: Path,
    *,
    folds: int,
    l2: float,
    complexity_lambda: float,
) -> tuple[
    Any,
    PairwiseScorer,
    tuple[ConsequenceArtifact, ...],
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[PlacementOutcome, ...],
    dict[str, Any],
]:
    loaded = load_dataset(dataset_path)
    compiler = RuleCompiler(loaded.registry)
    null = PermutationNullCalibration.from_dict(
        json.loads(permutation_null_path.read_text(encoding="utf-8"))
    )
    attainability = json.loads(attainability_path.read_text(encoding="utf-8"))
    base_scorer = PairwiseScorer(
        ScorerConfig(folds=folds, l2=l2, complexity_lambda=complexity_lambda)
    )
    scorer = base_scorer.with_calibration(null)

    expected = (
        (null.compiler_hash, compiler.compiler_hash, "permutation compiler hash"),
        (null.universe_hash, loaded.universe.universe_hash, "permutation universe hash"),
        (attainability.get("dataset_hash"), loaded.universe.manifest_hash, "dataset hash"),
        (attainability.get("universe_hash"), loaded.universe.universe_hash, "universe hash"),
        (
            attainability.get("descriptor_registry_hash"),
            loaded.registry.registry_hash,
            "descriptor registry hash",
        ),
        (attainability.get("compiler_hash"), compiler.compiler_hash, "compiler hash"),
        (attainability.get("scorer_hash"), scorer.config.scorer_hash, "scorer hash"),
    )
    for actual, wanted, label in expected:
        if actual != wanted:
            raise ValueError(f"attainability {label} mismatch")

    rows = attainability.get("candidates")
    if not isinstance(rows, list) or len(rows) != 696:
        raise ValueError("attainability reference must contain the 696 W1 candidates")
    artifacts = []
    for row in rows:
        artifact = compiler.compile(str(row["expression"])).evaluate(
            loaded.universe.candidates,
            dataset_hash=loaded.universe.manifest_hash,
            universe_hash=loaded.universe.universe_hash,
        )
        if artifact.rule_hash != row.get("rule_hash"):
            raise ValueError("attainability candidate rule hash mismatch")
        artifacts.append(artifact)

    shards = tuple(sorted(MaterialsAdsorptionTask._evidence_source_aliases(loaded).items()))
    if len(shards) != 11:
        raise ValueError("shard-null calibration requires exactly 11 evidence shards")
    return (
        loaded,
        scorer,
        tuple(artifacts),
        shards,
        _outcomes(loaded),
        attainability,
    )


def _prepare_worker_state(
    dataset_path: str,
    permutation_null_path: str,
    attainability_path: str,
    folds: int,
    l2: float,
    complexity_lambda: float,
) -> None:
    global _WORKER_SCORER
    global _WORKER_REPRESENTATIVES
    global _WORKER_INVERSES
    global _WORKER_COMPLEXITIES
    global _WORKER_SHARD_SYSTEMS

    loaded, scorer, artifacts, shards, outcomes, _ = _load_inputs(
        Path(dataset_path),
        Path(permutation_null_path),
        Path(attainability_path),
        folds=folds,
        l2=l2,
        complexity_lambda=complexity_lambda,
    )
    source_group_by_alias = {
        loaded.source_aliases[(record.source_system_id, record.source_placement_id)][
            0
        ]: record.source_group
        for record in loaded.records
        if (record.source_system_id, record.source_placement_id) in loaded.source_aliases
    }
    representatives_by_shard = []
    inverses = []
    shard_systems = []
    for _alias, source_groups in shards:
        shard_outcomes = tuple(
            outcome
            for outcome in outcomes
            if source_group_by_alias[outcome.candidate_alias] in source_groups
        )
        by_system: dict[str, list[PlacementOutcome]] = {}
        for outcome in shard_outcomes:
            by_system.setdefault(outcome.system_alias, []).append(outcome)
        systems = tuple(
            tuple(sorted(group, key=lambda item: item.candidate_alias))
            for _system, group in sorted(by_system.items())
        )
        if len(systems) < 2:
            raise ValueError("each shard must contain at least two systems")
        shard_systems.append(systems)

        topology = _build_pair_topology(
            shard_outcomes,
            max_pairs_per_system=scorer.config.max_pairs_per_system,
            pair_seed=scorer.config.pair_seed,
        )
        representatives = []
        inverse = []
        positions: dict[bytes, int] = {}
        for artifact in artifacts:
            column = _artifact_feature_column(topology, artifact).tobytes()
            position = positions.get(column)
            if position is None:
                position = len(representatives)
                positions[column] = position
                representatives.append(artifact)
            inverse.append(position)
        representatives_by_shard.append(tuple(representatives))
        inverses.append(np.asarray(inverse, dtype=np.intp))

    _WORKER_SCORER = scorer
    _WORKER_REPRESENTATIVES = tuple(representatives_by_shard)
    _WORKER_INVERSES = tuple(inverses)
    _WORKER_COMPLEXITIES = np.asarray(
        [artifact.complexity for artifact in artifacts], dtype=np.float64
    )
    _WORKER_SHARD_SYSTEMS = tuple(shard_systems)


def _score_job(job: tuple[int, int | None, int]) -> tuple[int, int | None, list[float]]:
    shard_index, permutation_index, seed = job
    scorer = _WORKER_SCORER
    complexities = _WORKER_COMPLEXITIES
    if scorer is None or complexities is None:
        raise RuntimeError("shard-null worker is uninitialized")

    outcomes = []
    for system in _WORKER_SHARD_SYSTEMS[shard_index]:
        if permutation_index is None:
            outcomes.extend(system)
            continue
        energies = [item.energy for item in system]
        rng = random.Random(
            int.from_bytes(
                _stable_digest(seed, permutation_index, system[0].system_alias)[:8],
                "big",
            )
        )
        rng.shuffle(energies)
        outcomes.extend(
            replace(item, energy=energy)
            for item, energy in zip(system, energies, strict=True)
        )

    scores = scorer.score_stage1(_WORKER_REPRESENTATIVES[shard_index], outcomes)
    raw_gains = np.fromiter(
        (score.stage1_raw_gain for score in scores),
        dtype=np.float64,
        count=len(scores),
    )
    expanded = (
        raw_gains[_WORKER_INVERSES[shard_index]]
        - scorer.config.complexity_lambda * complexities
    )
    return shard_index, permutation_index, expanded.tolist()


def build_calibration(
    dataset_path: Path,
    permutation_null_path: Path,
    attainability_path: Path,
    *,
    workers: int,
    folds: int,
    l2: float,
    complexity_lambda: float,
    permutations: int,
    seed: int,
) -> ShardNullCalibration:
    loaded, scorer, artifacts, shards, _outcomes_all, _attainability = _load_inputs(
        dataset_path,
        permutation_null_path,
        attainability_path,
        folds=folds,
        l2=l2,
        complexity_lambda=complexity_lambda,
    )
    jobs = [
        (shard_index, permutation_index, seed)
        for shard_index in range(len(shards))
        for permutation_index in (None, *range(permutations))
    ]
    null_statistics = {alias: [] for alias, _groups in shards}
    universe_gains: dict[str, Sequence[float]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_prepare_worker_state,
        initargs=(
            str(dataset_path),
            str(permutation_null_path),
            str(attainability_path),
            folds,
            l2,
            complexity_lambda,
        ),
    ) as pool:
        for shard_index, permutation_index, gains in pool.map(
            _score_job, jobs, chunksize=1
        ):
            alias = shards[shard_index][0]
            if permutation_index is None:
                universe_gains[alias] = gains
            else:
                null_statistics[alias].extend(gains)

    calibration = ShardNullCalibration.create(
        shard_partition_version=SHARD_PARTITION_VERSION,
        permutations=permutations,
        seed=seed,
        shard_statistics=null_statistics,
        shard_universe_gains=universe_gains,
        candidate_count=len(artifacts),
        dataset_hash=loaded.universe.manifest_hash,
        universe_hash=loaded.universe.universe_hash,
        descriptor_registry_hash=loaded.registry.registry_hash,
        compiler_hash=RuleCompiler(loaded.registry).compiler_hash,
        scorer_hash=scorer.config.scorer_hash,
        fold_seed=scorer.config.fold_seed,
        stage1_threshold=scorer.config.stage1_threshold,
        complexity_lambda=scorer.config.complexity_lambda,
    )
    expected_count = len(shards) * permutations * len(artifacts)
    if calibration.null_sample_count != expected_count:
        raise RuntimeError("shard-null pooled sample count mismatch")
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("permutation_null", type=Path)
    parser.add_argument("attainability", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--complexity-lambda", type=float, default=0.001)
    parser.add_argument("--permutations", type=int, default=SHARD_NULL_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SHARD_NULL_PERMUTATION_SEED)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    calibration = build_calibration(
        args.dataset,
        args.permutation_null,
        args.attainability,
        workers=args.workers,
        folds=args.folds,
        l2=args.l2,
        complexity_lambda=args.complexity_lambda,
        permutations=args.permutations,
        seed=args.seed,
    )
    save_durable_json(calibration.to_dict(), args.output)


if __name__ == "__main__":
    main()
