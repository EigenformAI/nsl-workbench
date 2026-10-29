from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.framework.capability_bridge import CapabilityMcpBridge
from src.harness.budget import BudgetLedger
from src.task.types import (
    CapabilityExecutionContext,
    CapabilityInvocation,
    EpisodeArtifacts,
    TrustedEpisodeIdentity,
)
from tasks.materials_adsorption.task import (
    PROBE_CALL_LIMIT,
    MaterialsAdsorptionTask,
)
from tasks.materials_adsorption.compiler import RuleCompiler
from tasks.materials_adsorption.dataset import load_dataset
from tasks.materials_adsorption.scoring import (
    PairwiseScorer,
    PermutationNullCalibration,
    ScorerConfig,
)
from src.task.base import TaskEnvironmentError


def test_task_requires_frozen_permutation_null_by_default() -> None:
    with pytest.raises(ValueError, match="permutation-null"):
        MaterialsAdsorptionTask({})


def test_task_declares_state_bearing_exploration_workflow(tmp_path) -> None:
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(tmp_path / "dataset.json"),
            "allow_uncalibrated_threshold": True,
        }
    )
    variation = task.list_variations()[0]
    episode_context = {
        "candidate_universe_hash": "universe-hash",
        "descriptor_registry": {
            "version": "test-v1",
            "registry_hash": "registry-hash",
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
        "evidence_alias": "evidence_public",
        "evidence": [
            {
                "alias": "candidate_a",
                "system_alias": "system_a",
                "descriptors": {"favorability": 1.0, "coordination": 2.0},
            }
        ],
        "rule_contract": "Use only public descriptors.",
    }
    workflow = task.workflow(variation, episode_context)

    assert workflow is not None
    assert [step.name for step in workflow.topological_order()] == [
        "explore",
        "hypothesise",
        "code",
        "verify",
        "rewrite",
    ]
    steps = {step.name: step for step in workflow.steps}
    for descriptor in ("favorability", "coordination"):
        assert descriptor in steps["explore"].prompt
        assert descriptor in steps["code"].prompt
    assert "clip" in steps["code"].prompt
    assert "comparison" in steps["code"].prompt
    assert steps["explore"].capabilities == (
        "inspect_evidence",
        "run_python",
        "record_evidence_plan",
    )
    assert steps["hypothesise"].capabilities == (
        "inspect_evidence",
        "run_python",
        "record_hypothesis",
    )
    assert steps["code"].capabilities == (
        "inspect_evidence",
        "run_python",
        "compile_feature",
        "probe_feature",
        "finalize_feature",
        "abandon_feature",
    )
    assert steps["code"].terminator_capabilities == (
        "finalize_feature",
        "abandon_feature",
    )
    assert (
        "using all probe_feature calls does not block finalize_feature"
        in steps["code"].prompt
    )
    assert "submit_hypothesis" in steps["code"].prompt
    assert "submit_rewrite" in steps["code"].prompt
    assert "compile_feature returns a rule_hash" in steps["code"].prompt
    assert "finalize the strongest by its rule_hash" in steps["code"].prompt
    assert steps["verify"].context_mode == "isolated"
    assert steps["verify"].terminator_capabilities == ("submit_hypothesis",)
    assert steps["rewrite"].context_mode == "isolated"
    assert steps["rewrite"].capabilities == (
        "get_accepted_hypothesis",
        "submit_rewrite",
    )
    assert "get_accepted_hypothesis" in steps["rewrite"].prompt

    constraints = task.episode_constraints(variation, episode_context)
    assert constraints.budgets.max_task_tool_calls == 44
    assert constraints.budgets.max_llm_turns == 52
    assert constraints.budgets.max_task_tool_calls_by_name["compile_feature"] == 8
    assert constraints.budgets.max_task_tool_calls_by_name["probe_feature"] == 3
    assert constraints.budgets.max_task_tool_calls_by_name["run_python"] == 10
    assert "inspect_evidence" not in constraints.budgets.max_task_tool_calls_by_name
    assert task.episode_commit_enabled()
    assert task.parallel_checkpoint_safe()

    capabilities = {
        capability.name: capability
        for capability in task.prompt_spec(variation, episode_context).capabilities
    }
    for name in ("probe_feature", "finalize_feature"):
        assert capabilities[name].schema == {
            "type": "object",
            "additionalProperties": False,
            "properties": {"rule_hash": {"type": "string"}},
        }


def test_submit_hypothesis_is_one_shot_and_verify_step_only(tmp_path) -> None:
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(tmp_path / "dataset.json"),
            "allow_uncalibrated_threshold": True,
        }
    )

    context = {}
    assert (
        task.validate_submission_context("ep-1", "code", context)
        == "wrong_workflow_step"
    )
    assert task.validate_submission_context("ep-1", "verify", context) is None
    task.mark_submission("ep-1", context)
    assert (
        task.validate_submission_context("ep-1", "verify", context)
        == "already_submitted"
    )


def _write_synthetic_dataset(path) -> None:
    records = []
    for system_index in range(8):
        for placement_index, energy in enumerate((0.0, 1.0, 2.0, 3.0)):
            records.append(
                {
                    "source_system_id": f"source-system-{system_index}",
                    "source_placement_id": f"placement-{placement_index}",
                    "source_filename": f"private/{system_index}-{placement_index}.traj",
                    "adsorbate_group": f"ads-{system_index % 2}",
                    "catalyst_group": f"cat-{system_index % 3}",
                    "structure_group": "surface",
                    "source_group": "synthetic",
                    "descriptors": {
                        "favorability": -energy + system_index * 0.001,
                        "coordination": float(placement_index + 1),
                    },
                    "energy": energy,
                    "axis": ("id", "ood_adsorbate", "ood_catalyst", "ood_both")[
                        system_index % 4
                    ],
                    "status": "valid",
                }
            )
    path.write_text(
        json.dumps(
            {
                "version": "synthetic-v1",
                "alias_salt": "private-test-salt",
                "descriptor_registry": {
                    "version": "synthetic-v1",
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


def _write_32_group_dataset(path) -> None:
    records = []
    for system_index, group_size in enumerate((2, 6, *([8] * 30))):
        for placement_index in range(group_size):
            records.append(
                {
                    "source_system_id": f"source-system-{system_index:02d}",
                    "source_placement_id": f"placement-{placement_index}",
                    "source_filename": (
                        f"private/{system_index:02d}-{placement_index}.traj"
                    ),
                    "adsorbate_group": f"ads-{system_index % 2}",
                    "catalyst_group": f"cat-{system_index % 3}",
                    "structure_group": "surface",
                    "source_group": f"group-{system_index:02d}",
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
                "version": "partition-v1",
                "alias_salt": "private-partition-salt",
                "descriptor_registry": {
                    "version": "partition-v1",
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


def test_evidence_groups_hash_partition_into_fixed_safe_triples(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_32_group_dataset(dataset_path)
    loaded = load_dataset(dataset_path)

    first = MaterialsAdsorptionTask._evidence_source_aliases(loaded)
    second = MaterialsAdsorptionTask._evidence_source_aliases(loaded)
    counts = {
        group: sum(record.source_group == group for record in loaded.records)
        for group in {record.source_group for record in loaded.records}
    }
    row_counts = [sum(counts[group] for group in groups) for groups in first.values()]
    odd_shards = {
        next(alias for alias, groups in first.items() if odd_group in groups)
        for odd_group in ("group-00", "group-01")
    }

    assert first == second
    assert len(first) == 11
    assert all(len(groups) == 3 for groups in first.values())
    assert min(row_counts) == 18
    assert max(row_counts) == 24
    assert len(odd_shards) == 2


def test_compile_echo_discloses_leak_free_guard_economics(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_synthetic_dataset(dataset_path)
    task = MaterialsAdsorptionTask(
        {
            "dataset_path": str(dataset_path),
            "allow_uncalibrated_threshold": True,
            "scorer": {"complexity_lambda": 0.007},
        }
    )
    context = {}

    result = task.execute_capability(
        CapabilityInvocation(
            "compile_feature",
            {"expression": "favorability", "applicable": "coordination >= 2"},
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "code", context),
    )

    assert result.success
    assert result.output["complexity"] == 1
    assert result.output["complexity_charge"] == pytest.approx(0.007)
    latest_hash = context["_materials_latest_compiled_rule_hash"]
    artifact = context["_materials_compiled"][latest_hash]["artifact"]
    values = [
        entry.value
        for entry in artifact.entries
        if entry.applicable and entry.value is not None
    ]
    assert result.output["support_count_over_universe"] == len(values)
    assert result.output["variance_ok"] is (
        bool(values) and max(values) - min(values) > 1e-12
    )
    assert "energy" not in json.dumps(result.output, sort_keys=True)


def test_compile_echo_is_invariant_to_private_energy_permutation(tmp_path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_synthetic_dataset(first_path)
    payload = json.loads(first_path.read_text())
    energies = [record["energy"] for record in payload["records"]]
    for record, energy in zip(payload["records"], reversed(energies), strict=True):
        record["energy"] = energy
    second_path.write_text(json.dumps(payload), encoding="utf-8")

    outputs = []
    for dataset_path in (first_path, second_path):
        task = MaterialsAdsorptionTask(
            {
                "dataset_path": str(dataset_path),
                "allow_uncalibrated_threshold": True,
            }
        )
        result = task.execute_capability(
            CapabilityInvocation("compile_feature", {"expression": "favorability"}),
            [],
            task.list_variations()[0],
            CapabilityExecutionContext("ep-0", "code", {}),
        )
        outputs.append(
            {
                key: result.output[key]
                for key in (
                    "complexity",
                    "complexity_charge",
                    "support_count_over_universe",
                    "variance_ok",
                )
            }
        )

    assert outputs[0] == outputs[1]


def test_probe_and_finalize_select_retained_feature_by_rule_hash(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_synthetic_dataset(dataset_path)
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(dataset_path),
            "allow_uncalibrated_threshold": True,
            "scorer": {"folds": 4, "bootstrap_replicates": 20},
        }
    )
    variation = task.list_variations()[0]
    context = {
        "evidence_alias": "evidence_test",
        "evidence": [
            {"alias": "candidate", "system_alias": "system", "descriptors": {}}
        ],
        "_materials_evidence_plan": {
            "summary": "synthetic evidence",
            "evidence_aliases": [],
        },
        "_materials_hypothesis": {
            "statement": "Favorability ranks adsorption placements.",
            "rationale": "Synthetic monotone control.",
            "applicability": "all placements",
            "discriminating_test": "Compare placements within each system.",
            "parent_hypothesis_ids": [],
        },
    }

    first = task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "favorability"}),
        [],
        variation,
        CapabilityExecutionContext("ep-0", "code", context),
    )
    second = task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "coordination"}),
        [],
        variation,
        CapabilityExecutionContext("ep-0", "code", context),
    )
    failed = task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "private_energy"}),
        [],
        variation,
        CapabilityExecutionContext("ep-0", "code", context),
    )
    first_hash = first.output["rule_hash"]
    second_hash = second.output["rule_hash"]

    assert first.success and second.success and not failed.success
    assert set(context["_materials_compiled"]) == {first_hash, second_hash}
    assert context["_materials_latest_compiled_rule_hash"] == second_hash

    original_score_stage1 = task._scorer.score_stage1
    task._scorer.score_stage1 = MagicMock(  # type: ignore[method-assign]
        return_value=(MagicMock(stage1_gain=0.0),)
    )
    task._shard_null_calibration = MagicMock()
    task._shard_null_calibration.percentile_bin.return_value = "p97"

    latest_probe = task.execute_capability(
        CapabilityInvocation("probe_feature", {}),
        [],
        variation,
        CapabilityExecutionContext("ep-0", "code", context),
    )
    selected_probe = task.execute_capability(
        CapabilityInvocation("probe_feature", {"rule_hash": first_hash}),
        [],
        variation,
        CapabilityExecutionContext("ep-0", "code", context),
    )
    probed_hashes = [
        call.args[0][0].rule_hash for call in task._scorer.score_stage1.call_args_list
    ]
    task._scorer.score_stage1 = original_score_stage1  # type: ignore[method-assign]

    finalized = task.execute_capability(
        CapabilityInvocation("finalize_feature", {"rule_hash": first_hash}),
        [],
        variation,
        CapabilityExecutionContext("ep-0", "code", context),
    )
    submitted = task.execute_capability(
        CapabilityInvocation("submit_hypothesis", {}),
        [],
        variation,
        CapabilityExecutionContext(
            "ep-0",
            "verify",
            context,
            identity=TrustedEpisodeIdentity("run", 0, 0, "ep-0"),
        ),
    )

    assert latest_probe.output == selected_probe.output == {"percentile_bin": "p97"}
    assert probed_hashes == [second_hash, first_hash]
    assert finalized.output == {"accepted": True, "rule_hash": first_hash}
    assert context["_materials_finalized_rule_hash"] == first_hash
    assert submitted.success
    assert context["_materials_verification"]["canonical_rule_hash"] == first_hash
    assert (
        context["_materials_verification"]["hypothesis_record"]["expression"]
        == "favorability"
    )


@pytest.mark.parametrize("selector", ["unknown-rule-hash", 7, None])
def test_probe_and_finalize_reject_invalid_rule_hash(tmp_path, selector) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_synthetic_dataset(dataset_path)
    task = MaterialsAdsorptionTask(
        {
            "dataset_path": str(dataset_path),
            "allow_uncalibrated_threshold": True,
        }
    )
    variation = task.list_variations()[0]
    context = {}
    compiled = task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "favorability"}),
        [],
        variation,
        CapabilityExecutionContext("ep-0", "code", context),
    )
    assert compiled.success

    for capability in ("probe_feature", "finalize_feature"):
        result = task.execute_capability(
            CapabilityInvocation(capability, {"rule_hash": selector}),
            [],
            variation,
            CapabilityExecutionContext("ep-0", "code", context),
        )
        assert not result.success
        assert result.error == (
            "unknown_rule_hash"
            if isinstance(selector, str)
            else "rule_hash must be a string"
        )


@pytest.mark.parametrize(
    ("compiler_hash", "universe_hash", "message"),
    [
        ("wrong-compiler", None, "compiler hash mismatch"),
        (None, "wrong-universe", "universe hash mismatch"),
    ],
)
def test_task_rejects_incompatible_calibration(
    tmp_path, compiler_hash, universe_hash, message
) -> None:
    dataset_path = tmp_path / "dataset.json"
    calibration_path = tmp_path / "null-v4.json"
    _write_synthetic_dataset(dataset_path)
    loaded = load_dataset(dataset_path)
    calibration = PermutationNullCalibration.create(
        percentile=0.95,
        permutations=1,
        seed=41,
        statistics=(0.0,),
        scorer_hash=PairwiseScorer(ScorerConfig()).config.scorer_hash,
        compiler_hash=compiler_hash or RuleCompiler(loaded.registry).compiler_hash,
        universe_hash=universe_hash or loaded.universe.universe_hash,
    )
    calibration_path.write_text(json.dumps(calibration.to_dict()), encoding="utf-8")
    task = MaterialsAdsorptionTask(
        {
            "dataset_path": str(dataset_path),
            "permutation_null_path": str(calibration_path),
            "allow_uncalibrated_threshold": True,
        }
    )

    with pytest.raises(TaskEnvironmentError, match=message):
        task.execute_capability(
            CapabilityInvocation("compile_feature", {"expression": "favorability"}),
            [],
            task.list_variations()[0],
            CapabilityExecutionContext("ep-0", "code", {}),
        )


def test_prompts_disclose_pricing_and_ground_exploration(tmp_path) -> None:
    task = MaterialsAdsorptionTask(
        {
            "dataset_path": str(tmp_path / "dataset.json"),
            "allow_uncalibrated_threshold": True,
            "scorer": {"complexity_lambda": 0.007, "stage1_threshold": -0.0012},
        }
    )
    workflow = task.workflow(task.list_variations()[0], {})
    steps = {step.name: step for step in workflow.steps}

    assert "actual evidence rows" in steps["explore"].prompt
    assert "registry is already" in steps["explore"].prompt
    assert "declared descriptors" in steps["hypothesise"].prompt
    assert "evidence aliases" in steps["hypothesise"].prompt
    code_prompt = steps["code"].prompt
    assert "0.007" in code_prompt
    assert "feature-expression AST nodes" in code_prompt
    assert "applicability and uncertainty expressions are not charged" in code_prompt
    assert "Names, constants, and operators count as 1" in code_prompt
    assert "function calls and conditional expressions count as 2" in code_prompt
    assert "frozen chance reference" in code_prompt
    assert "simplest expression" in code_prompt


def test_probe_feature_is_code_only_and_requires_a_compiled_feature(tmp_path) -> None:
    task = MaterialsAdsorptionTask(
        {
            "dataset_path": str(tmp_path / "missing.json"),
            "allow_uncalibrated_threshold": True,
        }
    )
    invocation = CapabilityInvocation("probe_feature", {})

    wrong_step = task.execute_capability(
        invocation,
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "explore", {}),
    )
    no_compile = task.execute_capability(
        invocation,
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "code", {}),
    )

    assert not wrong_step.success
    assert wrong_step.error == "wrong_workflow_step"
    assert not no_compile.success
    assert no_compile.error == "no_compiled_feature"


def test_probe_scores_three_single_system_source_groups_as_one_shard(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_synthetic_dataset(dataset_path)
    payload = json.loads(dataset_path.read_text())
    for record in payload["records"]:
        record["source_group"] = record["source_system_id"]
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(dataset_path),
            "allow_uncalibrated_threshold": True,
            "scorer": {"folds": 3, "complexity_lambda": 0.0},
        }
    )
    population = task.populate(
        [type("Container", (), {"id": "agent"})()], task.list_variations()[0]
    )
    context = population.episode_context
    task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "favorability"}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "code", context),
    )
    task._shard_null_calibration = MagicMock()
    task._shard_null_calibration.percentile_bin.return_value = "p50"

    result = task.execute_capability(
        CapabilityInvocation("probe_feature", {}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "code", context),
    )

    assert len({row["system_alias"] for row in context["evidence"]}) == 3
    assert len(context["evidence"]) == 12
    assert result.success
    assert result.output == {"percentile_bin": "p50"}


def test_probe_budget_rejects_the_fourth_call(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_synthetic_dataset(dataset_path)
    task = MaterialsAdsorptionTask(
        {
            "dataset_path": str(dataset_path),
            "allow_uncalibrated_threshold": True,
        }
    )
    variation = task.list_variations()[0]
    episode_context = {
        "evidence_alias": "evidence_test",
        "evidence": [
            {"alias": "candidate", "system_alias": "system", "descriptors": {}}
        ],
    }
    task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "favorability"}),
        [],
        variation,
        CapabilityExecutionContext("ep-0", "code", episode_context),
    )
    task._scorer.score_stage1 = MagicMock(  # type: ignore[method-assign]
        return_value=(MagicMock(stage1_gain=0.0),)
    )
    task._shard_null_calibration = MagicMock()
    task._shard_null_calibration.percentile_bin.return_value = "p50"
    constraints = task.episode_constraints(variation, episode_context)
    harness_context = MagicMock()
    harness_context.constraints = constraints
    harness_context.budget_ledger = BudgetLedger(constraints.budgets)
    harness_context.cancel_event.is_set.return_value = False
    harness_context.config.get.return_value = 0
    harness_context.prompt_spec = task.prompt_spec(variation, episode_context)
    harness_context.execute_capability.side_effect = lambda invocation: (
        task.execute_capability(
            invocation,
            [],
            variation,
            CapabilityExecutionContext(
                "ep-0", "code", episode_context, recorder=harness_context.recorder
            ),
        )
    )
    bridge = CapabilityMcpBridge(harness_context, token="")

    assert all(
        bridge.invoke(CapabilityInvocation("probe_feature", {})).success
        for _ in range(PROBE_CALL_LIMIT)
    )
    rejected = bridge.invoke(CapabilityInvocation("probe_feature", {}))

    assert not rejected.success
    assert rejected.error is not None
    assert "budget exhausted" in rejected.error


def test_rule_contract_discloses_probe_and_stage2_without_fold_details(
    tmp_path,
) -> None:
    task = MaterialsAdsorptionTask(
        {
            "dataset_path": str(tmp_path / "missing.json"),
            "allow_uncalibrated_threshold": True,
        }
    )

    contract = task._rule_contract()

    assert "probe up to 3 compiled features" in contract
    assert "evidence shard" in contract
    assert "percentile" in contract
    assert "frozen chance reference" in contract
    assert "held-out data" in contract
    assert "generalize to the universe" in contract
    assert "cross-fold lower bound" in contract
    assert "stability, not just mean effect" in contract
    assert "fold count" not in contract
    assert "fold assignment" not in contract


def test_unverified_episode_finalizes_as_incomplete(tmp_path) -> None:
    task = MaterialsAdsorptionTask(
        {
            "dataset_path": str(tmp_path / "dataset.json"),
            "allow_uncalibrated_threshold": True,
        }
    )

    reward = task.finalize_episode(
        [],
        task.measure_initial_state([], {}),
        {},
        EpisodeArtifacts(),
    )

    assert reward.breakdown["rejection_stage"] == "incomplete"


def _run_candidate(task, population, identity):
    context = population.episode_context
    evidence_result = task.execute_capability(
        CapabilityInvocation(
            "record_evidence_plan",
            {"summary": "Use the planted public descriptor.", "evidence_aliases": []},
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext(
            identity.episode_id, "explore", context, identity=identity
        ),
    )
    hypothesis_result = task.execute_capability(
        CapabilityInvocation(
            "record_hypothesis",
            {
                "statement": "More favorable initial chemistry ranks lower energies.",
                "rationale": "The planted control checks the executable path.",
                "applicability": "all supplied placements",
                "discriminating_test": "Compare rankings within each system.",
                "parent_hypothesis_ids": [],
            },
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext(
            identity.episode_id, "hypothesise", context, identity=identity
        ),
    )
    compile_result = task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "favorability"}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext(
            identity.episode_id, "code", context, identity=identity
        ),
    )
    finalize_result = task.execute_capability(
        CapabilityInvocation("finalize_feature", {}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext(
            identity.episode_id, "code", context, identity=identity
        ),
    )
    verify_result = task.execute_capability(
        CapabilityInvocation(
            "submit_hypothesis",
            {},
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext(
            identity.episode_id, "verify", context, identity=identity
        ),
    )
    rewrite_result = task.execute_capability(
        CapabilityInvocation("submit_rewrite", {"target": "public rewrite"}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext(
            identity.episode_id, "rewrite", context, identity=identity
        ),
    )
    artifacts = EpisodeArtifacts(
        capability_invocations=[
            CapabilityInvocation("record_evidence_plan", {}),
            CapabilityInvocation("record_hypothesis", {}),
            CapabilityInvocation("compile_feature", {}),
            CapabilityInvocation("finalize_feature", {}),
            CapabilityInvocation("submit_hypothesis", {}),
            CapabilityInvocation("submit_rewrite", {}),
        ],
        capability_results=[
            evidence_result,
            hypothesis_result,
            compile_result,
            finalize_result,
            verify_result,
            rewrite_result,
        ],
    )
    initial = task.measure_initial_state([], context)
    reward = task.finalize_episode([], initial, context, artifacts)
    return compile_result, verify_result, rewrite_result, reward


def test_task_succeeds_without_inspection_and_commits_duplicate_semantics(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_synthetic_dataset(dataset_path)
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(dataset_path),
            "allow_uncalibrated_threshold": True,
            "scorer": {
                "folds": 4,
                "complexity_lambda": 0.0,
                "stage1_threshold": 0.001,
                "bootstrap_replicates": 100,
            },
        }
    )
    container = type("Container", (), {"id": "agent"})()

    first_population = task.populate([container], task.list_variations()[0])
    first_identity = TrustedEpisodeIdentity("run", 0, 0, "ep-0")
    compiled, verified, rewritten, first_reward = _run_candidate(
        task, first_population, first_identity
    )

    assert compiled.success
    assert verified.output == {
        "accepted": True,
        "training_success": True,
        "archive_admission": True,
        "signal_band": "positive",
        "marginal_band": "positive",
    }
    assert rewritten.success
    assert first_reward.success
    assert first_reward.breakdown["canonical_rule_hash"]
    prepared = task.prepare_episode_commit(first_identity, first_reward)
    task.apply_episode_commit(prepared.token, prepared.intent_digest)

    second_population = task.populate([container], task.list_variations()[0])
    second_identity = TrustedEpisodeIdentity("run", 0, 1, "ep-1")
    _, duplicate_verified, duplicate_rewritten, duplicate_reward = _run_candidate(
        task, second_population, second_identity
    )

    assert duplicate_verified.output["training_success"] is True
    assert duplicate_verified.output["archive_admission"] is False
    assert duplicate_rewritten.success
    assert duplicate_reward.success
    assert duplicate_reward.breakdown["archive_admission"] is False


def test_failed_signal_cannot_submit_rewrite(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_synthetic_dataset(dataset_path)
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(dataset_path),
            "allow_uncalibrated_threshold": True,
            "scorer": {"folds": 4, "bootstrap_replicates": 20},
        }
    )
    container = type("Container", (), {"id": "agent"})()
    population = task.populate([container], task.list_variations()[0])
    identity = TrustedEpisodeIdentity("run", 0, 0, "ep-0")
    context = population.episode_context
    task.execute_capability(
        CapabilityInvocation(
            "record_evidence_plan",
            {"summary": "negative control", "evidence_aliases": []},
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "explore", context, identity=identity),
    )
    task.execute_capability(
        CapabilityInvocation(
            "record_hypothesis",
            {
                "statement": "constant",
                "rationale": "negative control",
                "applicability": "all",
                "discriminating_test": "none",
                "parent_hypothesis_ids": [],
            },
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "hypothesise", context, identity=identity),
    )
    task.execute_capability(
        CapabilityInvocation("compile_feature", {"expression": "0.0"}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "code", context, identity=identity),
    )
    task.execute_capability(
        CapabilityInvocation(
            "submit_hypothesis",
            {},
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "verify", context, identity=identity),
    )

    rewrite = task.execute_capability(
        CapabilityInvocation("submit_rewrite", {"target": "should fail"}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "rewrite", context, identity=identity),
    )

    assert not rewrite.success
    assert rewrite.error == "not_training_eligible"


def test_missing_workflow_step_is_denied_by_default(tmp_path) -> None:
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(tmp_path / "dataset.json"),
            "allow_uncalibrated_threshold": True,
        }
    )

    result = task.execute_capability(
        CapabilityInvocation(
            "record_evidence_plan", {"summary": "plan", "evidence_aliases": []}
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", None, {}),
    )

    assert not result.success
    assert result.error == "wrong_workflow_step"


def test_inspect_evidence_is_bounded_deterministic_and_public(tmp_path) -> None:
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(tmp_path / "missing-private-dataset.json"),
            "allow_uncalibrated_threshold": True,
        }
    )
    descriptors = [
        {
            "name": "favorability",
            "value_type": "number",
            "unit": "dimensionless",
            "family": "chemistry",
        },
        {
            "name": "is_bridge",
            "value_type": "boolean",
            "unit": "dimensionless",
            "family": "site",
        },
    ]
    context = {
        "evidence_alias": "evidence_public",
        "descriptor_registry": {"descriptors": descriptors},
        "evidence": [
            {
                "alias": f"candidate_{index:02d}",
                "system_alias": f"system_{index % 2}",
                "descriptors": {
                    "favorability": float(index),
                    "is_bridge": index % 2 == 0,
                },
            }
            for index in range(20)
        ],
        "dataset_path": "/private/path-must-not-appear",
        "_materials_verification": {"signal_band": "private"},
    }
    invocation = CapabilityInvocation(
        "inspect_evidence",
        {
            "view": "evidence",
            "filters": [
                {"descriptor": "favorability", "op": "gte", "value": 5.0},
                {"descriptor": "is_bridge", "op": "eq", "value": True},
            ],
            "sort_by": "favorability",
            "sort_direction": "desc",
            "top_k": 3,
        },
    )

    result = task.execute_capability(
        invocation,
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "explore", context),
    )

    assert result.success
    assert [row["alias"] for row in result.output["items"]] == [
        "candidate_18",
        "candidate_16",
        "candidate_14",
    ]
    assert result.output["returned"] == 3
    assert result.output["matched"] == 7
    assert result.output["truncated"] is True
    serialized = json.dumps(result.output, sort_keys=True)
    assert "/private" not in serialized
    assert "signal_band" not in serialized
    assert "energy" not in serialized

    oversized = task.execute_capability(
        CapabilityInvocation("inspect_evidence", {"view": "evidence", "top_k": 11}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "code", context),
    )
    assert not oversized.success
    assert oversized.error == "top_k must be an integer between 1 and 10"

    denied = task.execute_capability(
        invocation,
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "verify", context),
    )
    assert not denied.success
    assert denied.error == "wrong_workflow_step"

    incomplete_context = {
        **context,
        "evidence": [
            context["evidence"][0],
            {
                "alias": "candidate_missing",
                "system_alias": "system_missing",
                "descriptors": {"is_bridge": False},
            },
        ],
    }
    missing_sort_value = task.execute_capability(
        CapabilityInvocation(
            "inspect_evidence",
            {
                "view": "evidence",
                "sort_by": "favorability",
                "top_k": 3,
            },
        ),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext("ep-0", "explore", incomplete_context),
    )
    assert not missing_sort_value.success
    assert missing_sort_value.error == (
        "sort descriptor is unavailable in one or more rows: favorability"
    )


def test_rewrite_reads_only_the_public_accepted_hypothesis(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_synthetic_dataset(dataset_path)
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(dataset_path),
            "allow_uncalibrated_threshold": True,
            "scorer": {
                "folds": 4,
                "complexity_lambda": 0.0,
                "stage1_threshold": 0.001,
                "bootstrap_replicates": 20,
            },
        }
    )
    container = type("Container", (), {"id": "agent"})()
    population = task.populate([container], task.list_variations()[0])
    identity = TrustedEpisodeIdentity("run", 0, 0, "ep-0")
    _run_candidate(task, population, identity)

    result = task.execute_capability(
        CapabilityInvocation("get_accepted_hypothesis", {}),
        [],
        task.list_variations()[0],
        CapabilityExecutionContext(
            identity.episode_id,
            "rewrite",
            population.episode_context,
            identity=identity,
        ),
    )

    assert result.success
    assert result.output == {
        "hypothesis": "More favorable initial chemistry ranks lower energies.",
        "rationale": "The planted control checks the executable path.",
        "applicability": "all supplied placements",
        "evidence_aliases": [],
        "expression": "favorability",
        "uncertainty": "0.0",
        "discriminating_test": "Compare rankings within each system.",
        "parent_hypothesis_ids": [],
    }
    forbidden = {
        "signal_band",
        "marginal_band",
        "training_success",
        "archive_admission",
        "canonical_rule_hash",
        "behavior_hash",
        "consequence_hash",
        "receipt_hash",
        "score_start_revision",
    }
    assert forbidden.isdisjoint(result.output)


def test_compile_feature_has_eight_attempt_budget_before_bridge_rejection(
    tmp_path,
) -> None:
    task = MaterialsAdsorptionTask(
        {
            "archive_path": str(tmp_path / "archive.sqlite"),
            "blob_dir": str(tmp_path / "blobs"),
            "dataset_path": str(tmp_path / "missing-dataset.json"),
            "allow_uncalibrated_threshold": True,
        }
    )
    variation = task.list_variations()[0]
    episode_context = {}
    constraints = task.episode_constraints(variation, episode_context)
    harness_context = MagicMock()
    harness_context.constraints = constraints
    harness_context.budget_ledger = BudgetLedger(constraints.budgets)
    harness_context.cancel_event.is_set.return_value = False
    harness_context.config.get.return_value = 0
    harness_context.prompt_spec = task.prompt_spec(variation, episode_context)
    harness_context.execute_capability.side_effect = lambda invocation: (
        task.execute_capability(
            invocation,
            [],
            variation,
            CapabilityExecutionContext("ep-0", "code", episode_context),
        )
    )
    bridge = CapabilityMcpBridge(harness_context, token="")

    for _ in range(8):
        result = bridge.invoke(
            CapabilityInvocation("compile_feature", {"expression": 7})
        )
        assert not result.success
        assert result.error == "rule expressions must be strings"

    abandoned = bridge.invoke(CapabilityInvocation("abandon_feature", {}))
    assert abandoned.success
    assert abandoned.output == {"accepted": True, "compiled": False}

    rejected = bridge.invoke(CapabilityInvocation("compile_feature", {"expression": 7}))
    assert not rejected.success
    assert rejected.error is not None
    assert "budget exhausted" in rejected.error
    assert rejected.error != "wrong_workflow_step"
