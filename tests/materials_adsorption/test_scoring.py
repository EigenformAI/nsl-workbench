from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tasks.materials_adsorption import scoring
from tasks.materials_adsorption.compiler import (
    ConsequenceArtifact,
    ConsequenceEntry,
    RuleCompiler,
)
from tasks.materials_adsorption.dataset import load_dataset
from tasks.materials_adsorption.scoring import (
    PairwiseScorer,
    PermutationNullCalibration,
    PlacementOutcome,
    ScorerConfig,
    build_pairwise_examples,
)


def _artifact(name: str, values: dict[str, float], *, complexity: int = 1) -> ConsequenceArtifact:
    return ConsequenceArtifact.create(
        rule_hash=name,
        compiler_hash="compiler",
        descriptor_hash="descriptors",
        dataset_hash="dataset",
        universe_hash="universe",
        complexity=complexity,
        entries=[
            ConsequenceEntry(alias, True, value, 0.0, None)
            for alias, value in values.items()
        ],
    )


def _fixture() -> tuple[list[PlacementOutcome], ConsequenceArtifact, ConsequenceArtifact]:
    outcomes: list[PlacementOutcome] = []
    useful: dict[str, float] = {}
    harmful: dict[str, float] = {}
    for system_index in range(8):
        system = f"sys-{system_index}"
        axis = ("id", "ood_adsorbate", "ood_catalyst", "ood_both")[system_index % 4]
        for placement_index, energy in enumerate((0.0, 1.0, 2.0, 3.0)):
            alias = f"cfg-{system_index}-{placement_index}"
            outcomes.append(
                PlacementOutcome(alias, system, energy, axis, f"block-{system_index % 2}")
            )
            useful[alias] = -energy + (system_index * 0.001)
            harmful[alias] = energy
    return outcomes, _artifact("useful", useful), _artifact("harmful", harmful)


def test_committed_v4_calibration_matches_compiler_universe_and_scorer() -> None:
    path = (
        Path(__file__).parents[2]
        / "assets/materials-adsorption/oc20-dense-mini-v1/permutation-null-v4.json"
    )
    calibration = PermutationNullCalibration.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )

    scorer = PairwiseScorer(ScorerConfig())
    loaded = load_dataset(
        Path(__file__).parents[2]
        / "assets/materials-adsorption/oc20-dense-mini-v1/dataset.json"
    )
    assert calibration.scorer_hash == scorer.config.scorer_hash
    assert calibration.compiler_hash == RuleCompiler(loaded.registry).compiler_hash
    assert calibration.universe_hash == loaded.universe.universe_hash

    calibrated = scorer.with_calibration(calibration)

    assert calibrated.config.stage1_threshold == calibration.threshold


def test_pair_weights_give_each_system_equal_total_weight() -> None:
    outcomes, useful, _ = _fixture()
    examples = build_pairwise_examples(outcomes, (useful,))
    totals: dict[str, float] = {}
    for example in examples:
        totals[example.system_alias] = totals.get(example.system_alias, 0.0) + example.weight

    assert len(set(round(value, 12) for value in totals.values())) == 1
    assert abs(sum(totals.values()) - 1.0) < 1e-12


def test_pair_sampling_is_stable_under_row_reordering() -> None:
    outcomes, useful, _ = _fixture()

    forward = build_pairwise_examples(outcomes, (useful,))
    reverse = build_pairwise_examples(list(reversed(outcomes)), (useful,))

    assert forward == reverse


def test_empty_archive_learns_either_feature_polarity() -> None:
    outcomes, useful, inverse = _fixture()
    scorer = PairwiseScorer(
        ScorerConfig(
            folds=4,
            l2=0.1,
            complexity_lambda=0.0,
            stage1_threshold=0.001,
            bootstrap_replicates=300,
            bootstrap_seed=7,
        )
    )

    useful_receipt = scorer.score(useful, outcomes, archive=())
    inverse_receipt = scorer.score(inverse, outcomes, archive=())

    assert useful_receipt.signal_pass
    assert useful_receipt.marginal_pass
    assert useful_receipt.stage2_lower_bound > 0
    assert inverse_receipt.signal_pass
    assert inverse_receipt.marginal_pass
    useful_coefficient = useful_receipt.candidate_model.folds[0].coefficients[1]
    inverse_coefficient = inverse_receipt.candidate_model.folds[0].coefficients[1]
    assert useful_coefficient > 0
    assert inverse_coefficient < 0


def test_batch_stage1_matches_full_receipt_and_charges_single_node() -> None:
    outcomes, useful, _ = _fixture()
    scorer = PairwiseScorer(
        ScorerConfig(
            folds=4,
            l2=0.1,
            complexity_lambda=0.001,
            stage1_threshold=-0.001,
            bootstrap_replicates=20,
        )
    )

    batch = scorer.score_stage1((useful,), outcomes)[0]
    receipt = scorer.score(useful, outcomes)

    assert batch.complexity == 1
    assert batch.complexity_charge == pytest.approx(0.001)
    assert batch.stage1_raw_gain == pytest.approx(receipt.stage1_raw_gain)
    assert batch.stage1_gain == pytest.approx(receipt.stage1_gain)
    assert batch.converged
    assert batch.signal_pass is receipt.signal_pass


def test_duplicate_keeps_stage1_success_but_has_no_stage2_marginal_information() -> None:
    outcomes, useful, _ = _fixture()
    duplicate = _artifact(
        "duplicate",
        {entry.candidate_alias: entry.value for entry in useful.entries},
    )
    scorer = PairwiseScorer(
        ScorerConfig(
            folds=4,
            l2=0.1,
            complexity_lambda=0.0,
            stage1_threshold=0.001,
            bootstrap_replicates=300,
            bootstrap_seed=7,
        )
    )

    receipt = scorer.score(duplicate, outcomes, archive=(useful,))

    assert receipt.signal_pass
    assert receipt.stage1_gain > 0
    assert not receipt.marginal_pass
    assert receipt.stage2_lower_bound <= 1e-8


def test_affine_duplicate_is_rejected_by_marginal_scorer() -> None:
    outcomes, useful, _ = _fixture()
    system_by_candidate = {item.candidate_alias: item.system_alias for item in outcomes}
    affine_duplicate = _artifact(
        "affine-duplicate",
        {
            entry.candidate_alias: 2.0 * float(entry.value)
            + 0.1 * int(system_by_candidate[entry.candidate_alias].split("-")[-1])
            for entry in useful.entries
        },
    )
    scorer = PairwiseScorer(
        ScorerConfig(
            folds=4,
            l2=0.1,
            complexity_lambda=0.0,
            stage1_threshold=0.001,
            bootstrap_replicates=300,
            bootstrap_seed=7,
        )
    )

    receipt = scorer.score(affine_duplicate, outcomes, archive=(useful,))

    assert receipt.signal_pass
    assert not receipt.marginal_pass
    assert receipt.stage2_lower_bound <= 1e-8


def test_revisioned_baseline_cache_reuses_identical_fits() -> None:
    outcomes, useful, inverse = _fixture()
    scorer = PairwiseScorer(ScorerConfig(folds=4, bootstrap_replicates=20))

    first = scorer.score(useful, outcomes, archive_revision=3)
    cached_replay = scorer.score(useful, outcomes, archive_revision=3)
    second = scorer.score(inverse, outcomes, archive_revision=3)

    assert scorer.baseline_cache_info() == {"entries": 1, "hits": 2, "misses": 1}
    assert cached_replay == first
    assert first.intercept_model == second.intercept_model
    assert first.baseline_model == second.baseline_model


def test_baseline_cache_retains_only_current_archive_feature_matrix() -> None:
    outcomes, useful, inverse = _fixture()
    scorer_instance = PairwiseScorer(ScorerConfig(folds=4, bootstrap_replicates=20))

    scorer_instance.score(useful, outcomes, archive_revision=0)
    scorer_instance.score(inverse, outcomes, archive=(useful,), archive_revision=1)

    retained = [
        entry.archive_features
        for entry in scorer_instance._baseline_cache.values()
        if entry.archive_features is not None
    ]
    assert len(retained) == 1


def test_permutation_null_calibration_is_deterministic_and_versioned() -> None:
    outcomes, useful, harmful = _fixture()
    scorer = PairwiseScorer(
        ScorerConfig(
            folds=4,
            l2=0.1,
            complexity_lambda=0.0,
            bootstrap_replicates=20,
        )
    )

    first = scorer.calibrate_stage1_null(
        (useful, harmful), outcomes, permutations=12, percentile=0.9, seed=41
    )
    replay = scorer.calibrate_stage1_null(
        (useful, harmful), outcomes, permutations=12, percentile=0.9, seed=41
    )

    assert first == replay
    assert first.calibration_hash
    assert first.compiler_hash == useful.compiler_hash
    assert first.universe_hash == useful.universe_hash
    assert len(first.statistics) == 24
    assert type(first).from_dict(first.to_dict()) == first
    assert scorer.pair_topology_cache_info() == {"entries": 0, "hits": 0, "misses": 0}
    calibrated = scorer.with_calibration(first)
    assert calibrated.config.stage1_threshold == first.threshold


def test_group_degradation_is_not_charged_candidate_complexity() -> None:
    outcomes, useful, _ = _fixture()
    scorer = PairwiseScorer(
        ScorerConfig(
            folds=4,
            complexity_lambda=0.01,
            bootstrap_replicates=30,
        )
    )

    simple = scorer.score(useful, outcomes)
    complex_rule = scorer.score(
        _artifact(
            "complex-useful",
            {entry.candidate_alias: float(entry.value) for entry in useful.entries},
            complexity=1_000,
        ),
        outcomes,
    )

    assert [item.stage2_gain for item in simple.axis_diagnostics] == [
        item.stage2_gain for item in complex_rule.axis_diagnostics
    ]
    axis_payload = simple.to_dict()["axis_diagnostics"][0]
    assert set(axis_payload) == {
        "axis",
        "system_count",
        "stage1_gain",
        "stage2_gain",
        "stage2_lower_bound",
    }


def test_nonconverged_folds_fail_closed() -> None:
    outcomes, useful, _ = _fixture()
    receipt = PairwiseScorer(
        ScorerConfig(
            folds=4,
            complexity_lambda=0.0,
            stage1_threshold=-1.0,
            bootstrap_replicates=20,
            max_iterations=1,
            tolerance=1e-30,
        )
    ).score(useful, outcomes)

    assert not receipt.signal_pass
    assert not receipt.marginal_pass
    assert any(
        not fold.converged
        for model in (
            receipt.intercept_model,
            receipt.candidate_model,
            receipt.baseline_model,
            receipt.augmented_model,
        )
        for fold in model.folds
    )


def test_chemistry_block_collapse_blocks_admission() -> None:
    outcomes: list[PlacementOutcome] = []
    values: dict[str, float] = {}
    for system_index in range(12):
        block = "majority" if system_index < 8 else "minority"
        for placement_index, energy in enumerate((0.0, 1.0, 2.0, 3.0)):
            alias = f"block-cfg-{system_index}-{placement_index}"
            outcomes.append(PlacementOutcome(alias, f"sys-{system_index}", energy, "id", block))
            values[alias] = -energy if block == "majority" else energy

    receipt = PairwiseScorer(
        ScorerConfig(
            folds=4,
            complexity_lambda=0.0,
            stage1_threshold=-1.0,
            bootstrap_replicates=50,
            max_block_degradation=0.0,
            min_block_systems=2,
        )
    ).score(_artifact("block-collapse", values), outcomes)

    by_block = {item.block: item for item in receipt.block_diagnostics}
    assert by_block["minority"].catastrophic_degradation
    assert not receipt.marginal_pass


def _scalar_coordinate_fit(
    rows: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: ScorerConfig,
) -> tuple[np.ndarray, bool]:
    normalized_weights = weights / weights.sum()
    coefficients = np.zeros(rows.shape[1], dtype=np.float64)
    scores = np.zeros(rows.shape[0], dtype=np.float64)
    for _ in range(config.max_iterations):
        largest_change = 0.0
        for coordinate in range(rows.shape[1]):
            gradient = 0.0
            curvature = 0.0
            for row, label, weight, score in zip(
                rows, labels, normalized_weights, scores, strict=True
            ):
                probability = 1.0 / (1.0 + math.exp(-float(score)))
                value = float(row[coordinate])
                gradient += float(weight) * (probability - float(label)) * value
                curvature += (
                    float(weight) * probability * (1.0 - probability) * value * value
                )
            if coordinate:
                gradient += config.l2 * coefficients[coordinate]
                curvature += config.l2
            if curvature <= 1e-18:
                continue
            change = -gradient / curvature
            coefficients[coordinate] += change
            scores += change * rows[:, coordinate]
            largest_change = max(largest_change, abs(change))
        if largest_change <= config.tolerance:
            return coefficients, True
    return coefficients, False


def test_vectorized_fit_matches_scalar_reference() -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(300, 3))
    rows = np.column_stack((np.ones(features.shape[0]), features))
    labels = (rng.random(features.shape[0]) < 0.5).astype(np.float64)
    weights = rng.uniform(0.1, 2.0, size=features.shape[0])
    config = ScorerConfig(l2=0.2, max_iterations=500, tolerance=1e-10)

    expected, expected_converged = _scalar_coordinate_fit(rows, labels, weights, config)
    actual, _, actual_converged = scoring._fit_logistic_array(
        rows, labels, weights, config
    )

    assert actual_converged == expected_converged
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-8)


def test_warm_start_reaches_same_optimum_in_fewer_iterations() -> None:
    rng = np.random.default_rng(23)
    first = rng.normal(size=500)
    second = first + rng.normal(scale=0.05, size=500)
    rows = np.column_stack((np.ones(first.size), first, second))
    labels = (first + rng.normal(scale=0.5, size=first.size) > 0.0).astype(np.float64)
    weights = np.ones(first.size, dtype=np.float64)
    config = ScorerConfig(l2=0.1, max_iterations=100, tolerance=1e-8)

    cold, cold_iterations, cold_converged = scoring._fit_logistic_array(
        rows, labels, weights, config
    )
    warm, warm_iterations, warm_converged = scoring._fit_logistic_array(
        rows, labels, weights, config, initial_coefficients=cold
    )

    assert cold_converged and warm_converged
    assert np.allclose(warm, cold, rtol=0.0, atol=1e-8)
    assert warm_iterations < cold_iterations


def test_pair_topology_is_reused_across_candidates() -> None:
    outcomes, useful, harmful = _fixture()
    scorer_instance = PairwiseScorer(ScorerConfig(folds=4, bootstrap_replicates=20))

    scorer_instance.score(useful, outcomes)
    first = scorer_instance.pair_topology_cache_info()
    scorer_instance.score(harmful, outcomes)

    assert first == {"entries": 1, "hits": 0, "misses": 1}
    assert scorer_instance.pair_topology_cache_info() == {
        "entries": 1,
        "hits": 1,
        "misses": 1,
    }


def test_pair_topology_cache_invalidates_when_targets_change() -> None:
    outcomes, useful, _ = _fixture()
    scorer_instance = PairwiseScorer(ScorerConfig(folds=4, bootstrap_replicates=20))

    scorer_instance.score(useful, outcomes)
    changed = list(outcomes)
    changed[0] = replace(changed[0], energy=changed[1].energy)
    scorer_instance.score(useful, changed)

    assert scorer_instance.pair_topology_cache_info() == {
        "entries": 2,
        "hits": 0,
        "misses": 2,
    }


def test_vectorized_solver_fails_closed_on_singular_hessian() -> None:
    rows = np.ones((20, 2), dtype=np.float64)
    labels = np.tile(np.array([0.0, 1.0]), 10)
    weights = np.ones(20, dtype=np.float64)

    coefficients, _, converged = scoring._fit_logistic_array(
        rows,
        labels,
        weights,
        ScorerConfig(l2=0.0, max_iterations=20),
    )

    assert not converged
    assert np.all(np.isfinite(coefficients))


def test_vectorized_bootstrap_preserves_seeded_sampling() -> None:
    deltas = {f"system-{index}": index / 7.0 - 2.0 for index in range(9)}
    config = ScorerConfig(
        bootstrap_replicates=37,
        bootstrap_seed=31,
        lower_tail_probability=0.1,
    )
    systems = sorted(deltas)
    rng = random.Random(
        int.from_bytes(scoring._stable_digest(config.bootstrap_seed, "test")[:8], "big")
    )
    replicates = sorted(
        sum(deltas[systems[rng.randrange(len(systems))]] for _ in systems) / len(systems)
        for _ in range(config.bootstrap_replicates)
    )
    expected = replicates[math.floor(config.lower_tail_probability * len(replicates))]

    assert scoring._bootstrap_lower_bound(deltas, config, salt="test") == expected


def test_vectorized_receipts_are_deterministic_across_instances() -> None:
    outcomes, useful, harmful = _fixture()
    config = ScorerConfig(folds=4, bootstrap_replicates=20)

    first = PairwiseScorer(config).score(useful, outcomes, archive=(harmful,))
    second = PairwiseScorer(config).score(useful, outcomes, archive=(harmful,))

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_scorer_version_rejects_prevectorization_calibration() -> None:
    old_hash = "a0a0977f027e42a3854df0224b8bb668120bf6d3e5e07cad30db229831e7b80c"
    scorer_instance = PairwiseScorer()

    assert scorer_instance.config.scorer_hash != old_hash
    stale = PermutationNullCalibration.create(
        percentile=0.95,
        permutations=1,
        seed=0,
        statistics=(0.1,),
        scorer_hash=old_hash,
        compiler_hash="compiler",
        universe_hash="universe",
    )
    with pytest.raises(ValueError, match="scorer hash mismatch"):
        scorer_instance.with_calibration(stale)
