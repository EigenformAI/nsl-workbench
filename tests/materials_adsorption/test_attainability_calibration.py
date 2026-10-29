from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.calibrate_materials_adsorption_attainability import (
    build_report,
    generate_candidates,
)
from tasks.materials_adsorption.compiler import RuleCompiler
from tasks.materials_adsorption.dataset import CandidateState
from tasks.materials_adsorption.descriptors import DescriptorRegistry, DescriptorSpec


@pytest.fixture
def registry() -> DescriptorRegistry:
    return DescriptorRegistry(
        version="attainability-test-v1",
        descriptors=(
            DescriptorSpec("alpha", "number", "count", "one"),
            DescriptorSpec("beta", "number", "angstrom", "two"),
            DescriptorSpec("gamma", "number", "dimensionless", "three"),
            DescriptorSpec("enabled", "boolean", "boolean", "flag"),
        ),
    )


@pytest.fixture
def candidates() -> tuple[CandidateState, ...]:
    return (
        CandidateState(
            "cfg-a",
            "sys-a",
            {"alpha": 0.0, "beta": 2.0, "gamma": -1.0, "enabled": True},
        ),
        CandidateState(
            "cfg-b",
            "sys-a",
            {"alpha": 2.0, "beta": 0.0, "gamma": 3.0, "enabled": False},
        ),
    )


def test_candidate_generation_is_complete_seeded_and_valid(
    registry: DescriptorRegistry, candidates: tuple[CandidateState, ...]
) -> None:
    compiler = RuleCompiler(registry)

    generated = generate_candidates(
        compiler,
        candidates,
        dataset_hash="dataset",
        universe_hash="universe",
        random_count=25,
        seed=41,
    )
    replay = generate_candidates(
        compiler,
        candidates,
        dataset_hash="dataset",
        universe_hash="universe",
        random_count=25,
        seed=41,
    )

    assert [item.expression for item in generated] == [item.expression for item in replay]
    assert len({item.expression for item in generated}) == len(generated)
    assert sum(item.family == "single_descriptor" for item in generated) == 3
    assert sum(item.family == "pair_product" for item in generated) == 3
    assert sum(item.family == "pair_ratio" for item in generated) == 3
    assert sum(item.family == "random" for item in generated) == 25
    assert all(item.artifact.support_count == 2 for item in generated)
    assert all(item.artifact.complexity == 1 for item in generated[:3])


@dataclass(frozen=True)
class _Stage1:
    complexity: int
    complexity_charge: float
    stage1_raw_gain: float
    stage1_gain: float
    stage1_threshold: float
    converged: bool
    signal_pass: bool


class _Scorer:
    class _Config:
        scorer_hash = "scorer"
        complexity_lambda = 0.001
        stage1_threshold = -0.001

    config = _Config()

    def score_stage1(self, artifacts: tuple[object, ...], outcomes: object) -> tuple[_Stage1, ...]:
        assert outcomes == "outcomes"
        return tuple(
            _Stage1(
                complexity=index + 1,
                complexity_charge=0.001 * (index + 1),
                stage1_raw_gain=value,
                stage1_gain=value - 0.001 * (index + 1),
                stage1_threshold=-0.001,
                converged=True,
                signal_pass=value - 0.001 * (index + 1) > -0.001,
            )
            for index, value in enumerate((0.003, -0.004))
        )


@dataclass(frozen=True)
class _Artifact:
    rule_hash: str
    artifact_hash: str
    behavior_hash: str
    dataset_hash: str = "dataset"
    universe_hash: str = "universe"
    descriptor_registry_hash: str = "registry"
    compiler_hash: str = "compiler"


@dataclass(frozen=True)
class _Candidate:
    family: str
    expression: str
    artifact: _Artifact


def test_report_records_distribution_pass_and_maximum() -> None:
    report = build_report(
        (_Candidate("single_descriptor", "alpha", _Artifact("r1", "a1", "b1")),
         _Candidate("random", "-beta", _Artifact("r2", "a2", "b2"))),
        _Scorer(),
        "outcomes",
        dataset_hash="dataset",
        universe_hash="universe",
        descriptor_registry_hash="registry",
        compiler_hash="compiler",
        calibration_hash="calibration",
        seed=41,
        requested_random_count=1,
    )

    assert report["stage1_raw_gain_distribution"]["values"] == [-0.004, 0.003]
    assert report["pass_set"] == ["r1"]
    assert "bound to this compiler and universe" in report["bar_interpretation"]
    assert report["maximum_attainability"]["expression"] == "alpha"
    assert report["candidates"][0]["complexity"] == 1
    assert report["candidates"][0]["complexity_charge"] == pytest.approx(0.001)
    assert [item["signal_pass"] for item in report["candidates"]] == [True, False]
    assert report["candidate_family_counts"] == {"random": 1, "single_descriptor": 1}
