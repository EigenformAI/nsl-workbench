from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from src.task.base import TaskEnvironmentError
from src.task.types import (
    CapabilityExecutionContext,
    CapabilityInvocation,
    TrustedEpisodeIdentity,
)
from src.training_data.transforms import EpisodeTrainingRows, TrainingDataExportContext
from tasks.materials_adsorption.compiler import RuleCompiler
from tasks.materials_adsorption.dataset import LoadedDataset, load_dataset
from tasks.materials_adsorption.probe_calibration import (
    PERCENTILE_BIN_LABELS,
    SHARD_NULL_PERMUTATIONS,
    SHARD_NULL_PERMUTATION_SEED,
    SHARD_PARTITION_VERSION,
    ShardNullCalibration,
)
from tasks.materials_adsorption.scoring import PairwiseScorer, ScorerConfig
from tasks.materials_adsorption.task import (
    PROBE_CALL_LIMIT,
    PROBE_LEAK_BUDGET_BITS,
    MaterialsAdsorptionTask,
)
from tasks.materials_adsorption.training import MaterialsRewriteTransform


_SCORER_VALUES = {
    "folds": 3,
    "complexity_lambda": 0.0,
    "stage1_threshold": 74.0,
    "bootstrap_replicates": 20,
}
_PRE_TELEMETRY_RECEIPT_HASH = (
    "d435a8c08c2d85fe1ebe89d9b873be7c05ff427ff19a32dfbb64b49dc0602e88"
)


def _write_dataset(path: Path) -> None:
    records = []
    for system_index in range(6):
        for placement_index in range(4):
            records.append(
                {
                    "source_system_id": f"private-system-{system_index}",
                    "source_placement_id": f"placement-{placement_index}",
                    "source_filename": f"private/{system_index}-{placement_index}.traj",
                    "adsorbate_group": f"ads-{system_index % 2}",
                    "catalyst_group": f"cat-{system_index % 3}",
                    "structure_group": "surface",
                    "source_group": f"source-{system_index}",
                    "descriptors": {
                        "favorability": float(-placement_index),
                        "coordination": float(placement_index + 1),
                    },
                    "energy": float(placement_index),
                    "axis": "id",
                    "status": "valid",
                }
            )
    path.write_text(
        json.dumps(
            {
                "version": "probe-calibration-test-v1",
                "alias_salt": "private-probe-calibration-salt",
                "descriptor_registry": {
                    "version": "probe-calibration-test-v1",
                    "descriptors": [
                        {
                            "name": "favorability",
                            "value_type": "number",
                            "unit": "dimensionless",
                            "family": "chemistry",
                        },
                        {
                            "name": "coordination",
                            "value_type": "number",
                            "unit": "count",
                            "family": "site",
                        },
                    ],
                },
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def _write_runtime_artifacts(
    directory: Path,
    dataset_path: Path,
    *,
    calibration_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], LoadedDataset]:
    loaded = load_dataset(dataset_path)
    compiler = RuleCompiler(loaded.registry)
    scorer = PairwiseScorer(ScorerConfig(**_SCORER_VALUES))
    shard_aliases = MaterialsAdsorptionTask._evidence_source_aliases(loaded)
    values = tuple(float(index) for index in range(SHARD_NULL_PERMUTATIONS))
    calibration_values: dict[str, Any] = {
        "shard_partition_version": SHARD_PARTITION_VERSION,
        "permutations": SHARD_NULL_PERMUTATIONS,
        "seed": SHARD_NULL_PERMUTATION_SEED,
        "shard_statistics": {alias: values for alias in shard_aliases},
        "dataset_hash": loaded.universe.manifest_hash,
        "universe_hash": loaded.universe.universe_hash,
        "descriptor_registry_hash": loaded.registry.registry_hash,
        "compiler_hash": compiler.compiler_hash,
        "scorer_hash": scorer.config.scorer_hash,
        "fold_seed": scorer.config.fold_seed,
        "stage1_threshold": scorer.config.stage1_threshold,
        "complexity_lambda": scorer.config.complexity_lambda,
    }
    calibration_values.update(calibration_overrides or {})
    if "permutations" in (calibration_overrides or {}):
        count = int(calibration_values["permutations"])
        calibration_values["shard_statistics"] = {
            alias: tuple(float(index) for index in range(count))
            for alias in shard_aliases
        }
    shard_null = ShardNullCalibration.create(**calibration_values)
    shard_null_path = directory / "shard-null.json"
    shard_null_path.write_text(
        json.dumps(shard_null.to_dict(), sort_keys=True), encoding="utf-8"
    )

    attainability_values = [index / 1_000 - 0.01 for index in range(20)]
    attainability_path = directory / "attainability.json"
    attainability_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_hash": loaded.universe.manifest_hash,
                "universe_hash": loaded.universe.universe_hash,
                "descriptor_registry_hash": loaded.registry.registry_hash,
                "compiler_hash": compiler.compiler_hash,
                "scorer_hash": scorer.config.scorer_hash,
                "stage1_threshold": scorer.config.stage1_threshold,
                "bar_interpretation": (
                    "frozen permutation-null threshold bound to this compiler "
                    "and universe"
                ),
                "complexity_lambda": scorer.config.complexity_lambda,
                "stage1_raw_gain_distribution": {
                    "count": len(attainability_values),
                    "values": attainability_values,
                },
                "candidates": [
                    {
                        "rule_hash": f"attainability-rule-{index}",
                        "stage1_raw_gain": value,
                        "stage1_gain": value,
                    }
                    for index, value in enumerate(attainability_values)
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return (
        {
            "dataset_path": str(dataset_path),
            "archive_path": str(directory / "archive.sqlite"),
            "blob_dir": str(directory / "blobs"),
            "shard_null_path": str(shard_null_path),
            "attainability_path": str(attainability_path),
            "allow_uncalibrated_threshold": True,
            "scorer": dict(_SCORER_VALUES),
        },
        loaded,
    )


def _task_with_population(
    directory: Path, dataset_path: Path
) -> tuple[MaterialsAdsorptionTask, dict[str, Any]]:
    config, _ = _write_runtime_artifacts(directory, dataset_path)
    task = MaterialsAdsorptionTask(config)
    population = task.populate(
        [type("Container", (), {"id": "agent"})()], task.list_variations()[0]
    )
    context = population.episode_context
    compiled = task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "favorability"}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "code", context),
    )
    assert compiled.success
    return task, context


def _probe(
    task: MaterialsAdsorptionTask,
    context: dict[str, Any],
    *,
    recorder: Any | None = None,
):
    return task.execute_capability(
        CapabilityInvocation("probe_feature", {}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "code", context, recorder=recorder),
    )


def _admit(task: MaterialsAdsorptionTask, context: dict[str, Any]):
    context["_materials_evidence_plan"] = {
        "summary": "synthetic evidence",
        "evidence_aliases": [],
    }
    context["_materials_hypothesis"] = {
        "statement": "Initial favorability ranks adsorption placements.",
        "rationale": "Synthetic monotone control.",
        "applicability": "all placements",
        "discriminating_test": "Compare within each held-out system.",
        "parent_hypothesis_ids": [],
    }
    recorder = MagicMock()
    result = task.execute_capability(
        CapabilityInvocation("submit_hypothesis", {}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext(
            "ep-0",
            "verify",
            context,
            identity=TrustedEpisodeIdentity("run", 0, 0, "ep-0"),
            recorder=recorder,
        ),
    )
    return result, recorder


def _key_values(value: Any, key: str) -> list[Any]:
    if isinstance(value, dict):
        found = [value[key]] if key in value else []
        for item in value.values():
            found.extend(_key_values(item, key))
        return found
    if isinstance(value, list):
        found = []
        for item in value:
            found.extend(_key_values(item, key))
        return found
    return []


def test_probe_returns_exact_twenty_bin_percentile_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    task, context = _task_with_population(tmp_path, dataset_path)
    task._scorer.score_stage1 = MagicMock(  # type: ignore[method-assign]
        return_value=(MagicMock(stage1_gain=42.5),)
    )

    result = _probe(task, context)

    assert PERCENTILE_BIN_LABELS == (
        "p0",
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p75",
        "p80",
        "p84",
        "p88",
        "p90",
        "p92",
        "p94",
        "p95",
        "p96",
        "p97",
        "p98",
        "p99",
    )
    assert task._shard_null_calibration.quantile_edges == (
        9.0,
        19.0,
        29.0,
        39.0,
        49.0,
        59.0,
        69.0,
        74.0,
        79.0,
        83.0,
        87.0,
        89.0,
        91.0,
        93.0,
        94.0,
        95.0,
        96.0,
        97.0,
        98.0,
    )
    assert result.success
    assert result.output == {"percentile_bin": "p40"}
    assert not any(isinstance(value, (int, float)) for value in result.output.values())


def test_same_artifact_probe_is_deterministic_leak_bounded_and_audited(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    task, context = _task_with_population(tmp_path, dataset_path)
    task._scorer.score_stage1 = MagicMock(  # type: ignore[method-assign]
        return_value=(MagicMock(stage1_gain=42.5),)
    )
    recorder = MagicMock()

    first = _probe(task, context, recorder=recorder)
    second = _probe(task, context, recorder=recorder)

    latest_hash = context["_materials_latest_compiled_rule_hash"]
    artifact_hash = context["_materials_compiled"][latest_hash][
        "artifact"
    ].artifact_hash
    expected_payload = {
        "artifact_hash": artifact_hash,
        "percentile_bin": "p40",
    }
    assert first.output == second.output == {"percentile_bin": "p40"}
    assert recorder.log_observation.call_args_list == [
        call(category="materials_probe_feature", payload=expected_payload),
        call(
            category="materials_universe_rank",
            payload={
                "artifact_hash": artifact_hash,
                "scope": "probe",
                "universe_percentile": 43.0,
            },
        ),
        call(category="materials_probe_feature", payload=expected_payload),
        call(
            category="materials_universe_rank",
            payload={
                "artifact_hash": artifact_hash,
                "scope": "probe",
                "universe_percentile": 43.0,
            },
        ),
    ]
    assert context["_materials_probed_artifact_hashes"] == {artifact_hash}
    assert math.log2(len(PERCENTILE_BIN_LABELS)) <= PROBE_LEAK_BUDGET_BITS
    assert (
        PROBE_CALL_LIMIT * math.log2(len(PERCENTILE_BIN_LABELS))
        <= PROBE_LEAK_BUDGET_BITS
    )
    assert "universe_percentile" not in json.dumps(first.output, sort_keys=True)


def test_server_only_probe_telemetry_cannot_abort_the_probe(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    task, context = _task_with_population(tmp_path, dataset_path)
    calibration = MagicMock(wraps=task._shard_null_calibration)
    calibration.universe_percentile.side_effect = ValueError("telemetry broke")
    task._shard_null_calibration = calibration
    recorder = MagicMock()

    result = _probe(task, context, recorder=recorder)

    assert result.success
    assert set(result.output) == {"percentile_bin"}
    recorder.log_warning.assert_called_once()


@pytest.mark.parametrize(
    ("gain", "expected"),
    [
        (8.999, "p0"),
        (9.0, "p10"),
        (79.0, "p80"),
        (98.0, "p99"),
        (999.0, "p99"),
    ],
)
def test_non_uniform_percentile_bins_use_lower_edge_labels(
    tmp_path: Path, gain: float, expected: str
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    config, _ = _write_runtime_artifacts(tmp_path, dataset_path)
    calibration = ShardNullCalibration.from_dict(
        json.loads(Path(config["shard_null_path"]).read_text(encoding="utf-8"))
    )

    assert calibration.percentile_bin(gain) == expected


def test_probe_is_invariant_to_energies_outside_its_shard(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_dataset(first_path)
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    first_loaded = load_dataset(first_path)
    shard_map = MaterialsAdsorptionTask._evidence_source_aliases(first_loaded)
    selected_groups = next(iter(shard_map.values()))
    outside = [
        record
        for record in payload["records"]
        if record["source_group"] not in selected_groups
    ]
    outside_energies = [record["energy"] for record in outside]
    for record, energy in zip(outside, reversed(outside_energies), strict=True):
        record["energy"] = energy + 100.0
    second_path.write_text(json.dumps(payload), encoding="utf-8")

    outputs = []
    evidence_aliases = []
    for index, dataset_path in enumerate((first_path, second_path)):
        directory = tmp_path / f"run-{index}"
        directory.mkdir()
        task, context = _task_with_population(directory, dataset_path)
        outputs.append(_probe(task, context).output)
        evidence_aliases.append(context["evidence_alias"])

    assert outside
    assert evidence_aliases[0] == evidence_aliases[1]
    assert outputs[0] == outputs[1]


def test_disclosure_states_bar_percentile_and_stage2_without_private_details(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    config, _ = _write_runtime_artifacts(tmp_path, dataset_path)

    contract = MaterialsAdsorptionTask(config)._rule_contract().lower()
    normalized = contract.replace("-", " ")

    assert "75th percentile" in normalized
    assert "stage 2" in normalized
    assert "full candidate universe" in normalized
    assert "current archive" in normalized
    for forbidden in (
        "74",
        "fold count",
        "fold assignment",
        "100 permutations",
        "seed 41",
        "materials_universe_rank",
        "calibration_hash",
        "scorer_hash",
    ):
        assert forbidden not in contract


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"shard_partition_version": "wrong-partition"}, "partition version"),
        ({"permutations": SHARD_NULL_PERMUTATIONS + 1}, "permutation count"),
        ({"seed": SHARD_NULL_PERMUTATION_SEED + 1}, "permutation seed"),
        ({"scorer_hash": "wrong-scorer"}, "scorer hash mismatch"),
        ({"fold_seed": "wrong-folds"}, "fold seed mismatch"),
        ({"compiler_hash": "wrong-compiler"}, "compiler hash mismatch"),
        ({"universe_hash": "wrong-universe"}, "universe hash mismatch"),
        ({"dataset_hash": "wrong-dataset"}, "dataset hash mismatch"),
        (
            {"descriptor_registry_hash": "wrong-registry"},
            "descriptor registry hash mismatch",
        ),
        (
            {
                "shard_statistics": {
                    "evidence_unknown": (0.0,) * SHARD_NULL_PERMUTATIONS
                }
            },
            "shard aliases mismatch",
        ),
    ],
)
def test_task_rejects_incompatible_shard_null_artifact(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)

    with pytest.raises((ValueError, TaskEnvironmentError), match=message):
        config, _ = _write_runtime_artifacts(
            tmp_path, dataset_path, calibration_overrides=overrides
        )
        task = MaterialsAdsorptionTask(config)
        task.populate(
            [type("Container", (), {"id": "agent"})()],
            task.list_variations()[0],
        )


def test_private_rank_receipt_bytes_and_hash_are_stable_but_not_exposed(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    runs = []
    for index in range(2):
        directory = tmp_path / f"admission-{index}"
        directory.mkdir()
        task, context = _task_with_population(directory, dataset_path)
        result, recorder = _admit(task, context)
        receipt_hash = context["_materials_verification"]["receipt_hash"]
        receipt_bytes = task._archive_store().blobs.get(receipt_hash)
        runs.append((result, receipt_hash, receipt_bytes, recorder))

    first_result, first_hash, first_bytes, first_recorder = runs[0]
    second_result, second_hash, second_bytes, second_recorder = runs[1]
    first_payload = json.loads(first_bytes)
    rank_values = _key_values(first_payload, "materials_universe_rank")

    assert first_result.success and second_result.success
    assert "materials_universe_rank" not in json.dumps(
        first_result.output, sort_keys=True
    )
    assert "materials_universe_rank" not in json.dumps(
        second_result.output, sort_keys=True
    )
    assert rank_values == []
    latest_hash = context["_materials_latest_compiled_rule_hash"]
    artifact_hash = context["_materials_compiled"][latest_hash][
        "artifact"
    ].artifact_hash
    for recorder in (first_recorder, second_recorder):
        assert recorder.log_observation.call_args_list == [
            call(
                category="materials_universe_rank",
                payload={
                    "artifact_hash": artifact_hash,
                    "scope": "admission",
                    "universe_percentile": 100.0,
                },
            )
        ]
    assert first_bytes == second_bytes
    assert first_hash == second_hash == _PRE_TELEMETRY_RECEIPT_HASH
    assert first_hash == hashlib.sha256(first_bytes).hexdigest()
    assert first_bytes == json.dumps(
        first_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def test_frozen_v2_shard_null_artifact_has_expected_grid_and_population() -> None:
    path = (
        Path(__file__).parents[2]
        / "assets/materials-adsorption/oc20-dense-mini-v1/shard-null-v2.json"
    )
    calibration = ShardNullCalibration.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert calibration.percentile_bin_labels == PERCENTILE_BIN_LABELS
    assert len(calibration.quantile_edges) == 19
    assert all(
        left < right
        for left, right in zip(
            calibration.quantile_edges, calibration.quantile_edges[1:]
        )
    )
    assert calibration.candidate_count == 696
    assert calibration.null_sample_count == 765_600
    assert len(calibration.shard_universe_gains) == 11
    assert all(len(gains) == 696 for gains in calibration.shard_universe_gains.values())


def test_server_only_admission_telemetry_cannot_abort_submission(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    task, context = _task_with_population(tmp_path, dataset_path)
    reference = MagicMock(wraps=task._attainability_reference)
    reference.percentile.side_effect = ValueError("telemetry broke")
    task._attainability_reference = reference

    result, recorder = _admit(task, context)

    assert result.success
    assert context["_materials_verification"]["receipt_hash"]
    recorder.log_warning.assert_called_once()


def test_training_row_drops_materials_universe_rank_telemetry(tmp_path: Path) -> None:
    context = TrainingDataExportContext(
        generation_id=0,
        run_id="run",
        task_name="materials-adsorption",
        source_generation_dir=tmp_path,
        source_all_episodes_path=tmp_path / "all.jsonl",
        export_recipe_hash="recipe",
    )
    row = {
        "row_id": "rewrite",
        "parent_row_id": None,
        "prompt": "public prompt",
        "raw_response": "public target",
        "interaction_type": "inference",
        "source_interaction_type": "inference",
        "timestamp": "2026-07-28T00:00:00",
        "success": True,
        "episode_id": "ep-0",
        "episode_index": 0,
        "generation_id": 0,
        "episode_score": 1.0,
        "source_episode_id": "ep-0",
        "source_row_index": 0,
        "workflow_step": "rewrite",
        "actor_role": "assistant",
        "materials_universe_rank": 7,
        "record_meta": {
            "materials_universe_rank": 7,
            "private_telemetry": {"materials_universe_rank": 7},
            "public_provenance": {
                "evidence_alias": "evidence-public",
                "materials_universe_rank": 7,
            },
        },
    }

    transformed = MaterialsRewriteTransform().transform_export_rows(
        context,
        [
            EpisodeTrainingRows(
                episode_id="ep-0",
                episode_index=0,
                generation_id=0,
                episode_score=1.0,
                rows=[row],
            )
        ],
    )

    assert len(transformed) == 1
    assert "materials_universe_rank" not in json.dumps(
        transformed[0].rows[0], sort_keys=True
    )
    assert transformed[0].rows[0]["record_meta"] == {
        "public_provenance": {"evidence_alias": "evidence-public"}
    }
