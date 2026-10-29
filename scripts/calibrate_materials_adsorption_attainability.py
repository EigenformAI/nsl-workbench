"""Measure feature attainability against the frozen materials stage-1 bar."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

from scripts.calibrate_materials_adsorption import _outcomes
from src.typing.training import save_durable_json
from tasks.materials_adsorption.compiler import (
    ConsequenceArtifact,
    RuleCompiler,
    RuleEvaluationError,
    RuleValidationError,
)
from tasks.materials_adsorption.dataset import CandidateState, load_dataset
from tasks.materials_adsorption.scoring import (
    PairwiseScorer,
    PermutationNullCalibration,
    ScorerConfig,
)


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    family: str
    expression: str
    artifact: ConsequenceArtifact


def _compile_candidate(
    compiler: RuleCompiler,
    candidates: Sequence[CandidateState],
    expression: str,
    family: str,
    *,
    dataset_hash: str,
    universe_hash: str,
) -> CalibrationCandidate:
    artifact = compiler.compile(expression).evaluate(
        tuple(candidates), dataset_hash=dataset_hash, universe_hash=universe_hash
    )
    return CalibrationCandidate(family, expression, artifact)


def _random_expression(rng: random.Random, names: Sequence[str]) -> str:
    a, b, c = (rng.choice(names) for _ in range(3))
    templates = (
        f"-({a})",
        f"abs({a})",
        f"({a}) ** 2",
        f"sqrt(abs({a}))",
        f"clip({a}, -10.0, 10.0)",
        f"({a}) * ({b})",
        f"abs({a}) * ({b})",
        f"({a}) / (abs({b}) + 1.0)",
        f"(({a}) * ({b})) * ({c})",
        f"(({a}) / (abs({b}) + 1.0)) * ({c})",
        f"({a}) / (abs(({b}) * ({c})) + 1.0)",
    )
    return rng.choice(templates)


def generate_candidates(
    compiler: RuleCompiler,
    candidates: Sequence[CandidateState],
    *,
    dataset_hash: str,
    universe_hash: str,
    random_count: int = 500,
    seed: int = 41,
) -> tuple[CalibrationCandidate, ...]:
    if random_count < 0:
        raise ValueError("random_count must be non-negative")
    names = tuple(
        descriptor.name
        for descriptor in compiler.registry.descriptors
        if descriptor.value_type == "number"
    )
    if not names:
        raise ValueError("attainability calibration requires numeric descriptors")
    generated: list[CalibrationCandidate] = []
    seen: set[str] = set()

    def add(expression: str, family: str) -> None:
        if expression in seen:
            return
        item = _compile_candidate(
            compiler,
            candidates,
            expression,
            family,
            dataset_hash=dataset_hash,
            universe_hash=universe_hash,
        )
        seen.add(expression)
        generated.append(item)

    for name in names:
        add(name, "single_descriptor")
    for left_index, left in enumerate(names[:-1]):
        for right in names[left_index + 1 :]:
            add(f"({left}) * ({right})", "pair_product")
            add(
                f"(({left}) / ({right})) if ({right}) != 0 else 0.0",
                "pair_ratio",
            )

    rng = random.Random(seed)
    attempts = 0
    maximum_attempts = max(1_000, random_count * 100)
    accepted = 0
    while accepted < random_count and attempts < maximum_attempts:
        attempts += 1
        expression = _random_expression(rng, names)
        if expression in seen:
            continue
        try:
            add(expression, "random")
        except (RuleEvaluationError, RuleValidationError, ArithmeticError, ValueError):
            continue
        accepted += 1
    if accepted != random_count:
        raise RuntimeError(
            f"unable to generate {random_count} unique valid random expressions "
            f"after {maximum_attempts} attempts"
        )
    return tuple(generated)


def _percentile(values: Sequence[float], fraction: float) -> float:
    index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
    return values[index]


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("distribution requires at least one value")
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p10": _percentile(ordered, 0.10),
        "p25": _percentile(ordered, 0.25),
        "median": _percentile(ordered, 0.50),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "maximum": ordered[-1],
        "mean": fmean(ordered),
        "values": ordered,
    }


def build_report(
    candidates: Sequence[Any],
    scorer: Any,
    outcomes: Any,
    *,
    dataset_hash: str,
    universe_hash: str,
    descriptor_registry_hash: str,
    compiler_hash: str,
    calibration_hash: str,
    seed: int,
    requested_random_count: int,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("attainability report requires candidates")
    for candidate in candidates:
        artifact = candidate.artifact
        expected = {
            "dataset_hash": dataset_hash,
            "universe_hash": universe_hash,
            "descriptor_registry_hash": descriptor_registry_hash,
            "compiler_hash": compiler_hash,
        }
        for field_name, expected_value in expected.items():
            if str(getattr(artifact, field_name)) != expected_value:
                raise ValueError(
                    f"candidate artifact has incompatible {field_name}"
                )
    scores = scorer.score_stage1(
        tuple(item.artifact for item in candidates), outcomes
    )
    if len(scores) != len(candidates):
        raise ValueError("stage-1 scorer did not return one result per candidate")
    rows = []
    for candidate, score in zip(candidates, scores, strict=True):
        rows.append(
            {
                "family": candidate.family,
                "expression": candidate.expression,
                "rule_hash": candidate.artifact.rule_hash,
                "artifact_hash": candidate.artifact.artifact_hash,
                "behavior_hash": candidate.artifact.behavior_hash,
                "complexity": score.complexity,
                "complexity_charge": score.complexity_charge,
                "stage1_raw_gain": score.stage1_raw_gain,
                "stage1_gain": score.stage1_gain,
                "stage1_threshold": score.stage1_threshold,
                "converged": score.converged,
                "signal_pass": score.signal_pass,
            }
        )
    maximum_raw = min(rows, key=lambda row: (-row["stage1_raw_gain"], row["expression"]))
    maximum_gain = min(rows, key=lambda row: (-row["stage1_gain"], row["expression"]))
    pass_set = [row["rule_hash"] for row in rows if row["signal_pass"]]
    return {
        "schema_version": 1,
        "dataset_hash": dataset_hash,
        "universe_hash": universe_hash,
        "descriptor_registry_hash": descriptor_registry_hash,
        "compiler_hash": compiler_hash,
        "scorer_hash": scorer.config.scorer_hash,
        "calibration_hash": calibration_hash,
        "stage1_threshold": scorer.config.stage1_threshold,
        "bar_interpretation": (
            "frozen permutation-null threshold bound to this compiler and universe"
        ),
        "complexity_lambda": scorer.config.complexity_lambda,
        "random_seed": seed,
        "requested_random_count": requested_random_count,
        "pair_policy": "each unordered numeric-descriptor pair: product and canonical zero-safe ratio",
        "candidate_family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "stage1_raw_gain_distribution": _distribution(
            row["stage1_raw_gain"] for row in rows
        ),
        "pass_count": len(pass_set),
        "pass_rate": len(pass_set) / len(rows),
        "pass_set": pass_set,
        "maximum_attainability": dict(maximum_raw),
        "maximum_penalized_gain": dict(maximum_gain),
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--random-expressions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--complexity-lambda", type=float, default=0.001)
    args = parser.parse_args()

    loaded = load_dataset(args.dataset)
    compiler = RuleCompiler(loaded.registry)
    calibration = PermutationNullCalibration.from_dict(
        json.loads(args.calibration.read_text(encoding="utf-8"))
    )
    if calibration.compiler_hash != compiler.compiler_hash:
        raise ValueError("calibration compiler hash mismatch")
    if calibration.universe_hash != loaded.universe.universe_hash:
        raise ValueError("calibration universe hash mismatch")
    base_scorer = PairwiseScorer(
        ScorerConfig(
            folds=args.folds,
            l2=args.l2,
            complexity_lambda=args.complexity_lambda,
        )
    )
    scorer = base_scorer.with_calibration(calibration)
    candidates = generate_candidates(
        compiler,
        loaded.universe.candidates,
        dataset_hash=loaded.universe.manifest_hash,
        universe_hash=loaded.universe.universe_hash,
        random_count=args.random_expressions,
        seed=args.seed,
    )
    report = build_report(
        candidates,
        scorer,
        _outcomes(loaded),
        dataset_hash=loaded.universe.manifest_hash,
        universe_hash=loaded.universe.universe_hash,
        descriptor_registry_hash=loaded.registry.registry_hash,
        compiler_hash=compiler.compiler_hash,
        calibration_hash=calibration.calibration_hash,
        seed=args.seed,
        requested_random_count=args.random_expressions,
    )
    save_durable_json(report, args.output)


if __name__ == "__main__":
    main()
