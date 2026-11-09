from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

import src.viz as viz_bundle
import tasks.materials_adsorption.viz as viz_extractor
from scripts.export_viz import main as export_main
from src.task.types import TrustedEpisodeIdentity
from src.viz import write_viz_bundle
from tasks.materials_adsorption.archive import ArchiveStore
from tasks.materials_adsorption.compiler import ConsequenceArtifact, ConsequenceEntry
from tasks.materials_adsorption.scoring import (
    PairwiseScorer,
    PlacementOutcome,
    ScorerConfig,
    ScoringReceipt,
)
from tasks.materials_adsorption.viz import extract_run_viz


_RUN = "run-new"
_OLD_RUN = "run-old"
_SCRIPT_BREAKOUT = "</script><img src=x onerror=alert(1)>"
_TEMPLATE = (
    '<!doctype html><title>t</title><script id="viz-data" '
    'type="application/json">__VIZ_DATA__</script><p>after</p>'
)
_FAMILIES = {
    "alpha": "site",
    "beta": "site",
    "zeta": "site",
    "gamma": "chemistry",
    "delta": "chemistry",
    "epsilon": "bulk",
}
_PERMUTATION = (5, 11, 2, 8, 0, 9, 3, 7, 1, 10, 6, 4)
_PROCESS = {
    "calls": {"compile_feature": 2, "inspect_evidence": 3, "submit_hypothesis": 1},
    "compile_attempts": 2,
    "compile_successes": 1,
    "first_compile_success_attempt": 2,
    "probes": 2,
    "probe_percentiles": [41.5, 88.0],
    "admission_percentile": 90.25,
}


def _write_dataset(
    path: Path,
    *,
    families: dict[str, str] | None = None,
    reverse: bool = False,
) -> None:
    families = families or _FAMILIES
    records = []
    for index, shuffled in enumerate(_PERMUTATION):
        records.append(
            {
                "source_system_id": f"source-system-{index:02d}",
                "source_placement_id": "placement-0",
                "source_filename": f"private/{index}.traj",
                "adsorbate_group": f"ads-{index % 2}",
                "catalyst_group": f"cat-{index % 3}",
                "structure_group": "surface",
                "source_group": "synthetic",
                "descriptors": {
                    "alpha": float(index),
                    "beta": float(index**2),
                    "zeta": float(index // 3),
                    "gamma": float(shuffled),
                    "delta": -3.0 * shuffled,
                    "epsilon": 1.0,
                },
                "energy": float(index),
                "axis": ("id", "ood_adsorbate", "ood_catalyst", "ood_both")[index % 4],
                "status": "valid",
            }
        )
    descriptors = [
        {"name": name, "value_type": "number", "unit": "unitless", "family": family}
        for name, family in families.items()
    ]
    if reverse:
        records.reverse()
        descriptors.reverse()
    path.write_text(
        json.dumps(
            {
                "version": "viz-synthetic-v1",
                "alias_salt": "viz-test-salt",
                "descriptor_registry": {
                    "version": "viz-synthetic-v1",
                    "descriptors": descriptors,
                },
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def _placements() -> list[PlacementOutcome]:
    outcomes = []
    for system_index in range(8):
        axis = ("id", "ood_adsorbate", "ood_catalyst", "ood_both")[system_index % 4]
        for placement_index, energy in enumerate((0.0, 1.0, 2.0, 3.0)):
            outcomes.append(
                PlacementOutcome(
                    f"cfg-{system_index}-{placement_index}",
                    f"sys-{system_index}",
                    energy,
                    axis,
                    f"block-{system_index % 2}",
                )
            )
    return outcomes


def _artifact(rule_hash: str, offset: float) -> ConsequenceArtifact:
    return ConsequenceArtifact.create(
        rule_hash=rule_hash,
        compiler_hash="compiler",
        descriptor_hash="descriptors",
        dataset_hash="dataset",
        universe_hash="universe",
        complexity=3,
        entries=[
            ConsequenceEntry(
                outcome.candidate_alias,
                True,
                -outcome.energy + offset * position,
                0.0,
                None,
            )
            for position, outcome in enumerate(_placements())
        ],
    )


def _record(expression: str, **fields: Any) -> dict[str, Any]:
    return {
        "hypothesis": f"hypothesis for {expression}",
        "rationale": "rationale",
        "applicability": "everywhere",
        "evidence_aliases": ["cfg_LEAK_alias"],
        "expression": expression,
        "uncertainty": "0.0",
        "discriminating_test": "test",
        "parent_hypothesis_ids": [],
        **fields,
    }


def _outcome(stage: str | None, **fields: Any) -> dict[str, Any]:
    return {
        "rejection_stage": stage,
        "materialized": True,
        "parent_rule_hashes": [],
        "referenced_descriptors": ["alpha"],
        "process": _PROCESS,
        "stop_cause": None,
        "rewrite": None,
        **fields,
    }


@dataclass
class _Run:
    archive: Path
    blobs: Path
    generations: Path
    dataset: Path
    receipts: dict[str, ScoringReceipt]
    behavior_hashes: list[str]

    def extract(self, **overrides: Any) -> dict[str, Any]:
        arguments = {
            "archive_path": self.archive,
            "blob_dir": self.blobs,
            "generation_root": self.generations,
            "dataset_path": self.dataset,
            **overrides,
        }
        return extract_run_viz(**arguments)


class _Builder:
    def __init__(self, tmp_path: Path) -> None:
        self.store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
        self.scorer = PairwiseScorer(
            ScorerConfig(
                folds=4,
                complexity_lambda=0.0,
                stage1_threshold=0.001,
                bootstrap_replicates=20,
            )
        )
        self.admitted: list[ConsequenceArtifact] = []
        self.receipts: dict[str, ScoringReceipt] = {}
        self.receipt_hashes: dict[str, str] = {}
        self.behavior_hashes: list[str] = []

    def score(
        self,
        artifact: ConsequenceArtifact,
        scorer: PairwiseScorer | None = None,
        *,
        fresh: bool = False,
    ) -> ScoringReceipt:
        archive = () if fresh else tuple(self.admitted)
        receipt = (scorer or self.scorer).score(
            artifact, _placements(), archive=archive, archive_revision=len(archive)
        )
        self.receipts[artifact.rule_hash] = receipt
        return receipt

    def commit(
        self,
        run_id: str,
        generation: int,
        index: int,
        *,
        outcome: dict[str, Any],
        artifact: ConsequenceArtifact | None = None,
        receipt: ScoringReceipt | None = None,
        record: dict[str, Any] | None = None,
        admit: bool = False,
    ) -> None:
        identity = TrustedEpisodeIdentity(
            run_id, generation, index, f"ep-{run_id}-{generation}-{index}"
        )
        rule_hash = (
            artifact.rule_hash
            if artifact is not None
            else f"unmaterialized-{run_id}-{generation}-{index}"
        )
        behavior_hash = None if artifact is None else artifact.behavior_hash
        if behavior_hash is not None:
            self.behavior_hashes.append(behavior_hash)
        consequence_hash = None
        receipt_hash = None
        if artifact is not None and receipt is not None:
            consequence_hash = self.store.blobs.put(_blob(artifact.to_dict()))
            receipt_hash = self.store.blobs.put(_blob(receipt.to_dict()))
            self.receipt_hashes[artifact.rule_hash] = receipt_hash
        prepared = self.store.prepare_episode_commit(
            identity=identity,
            training_success=admit,
            requested_admission=admit,
            canonical_rule_hash=rule_hash,
            behavior_hash=behavior_hash,
            consequence_hash=consequence_hash,
            receipt_hash=receipt_hash,
            outcome=outcome,
            archive_record={
                "canonical_rule_hash": rule_hash,
                "behavior_hash": behavior_hash,
                "episode_id": identity.episode_id,
                **(record or {}),
            },
        )
        result = self.store.apply_episode_commit(
            prepared.token, prepared.intent_digest
        )
        if result.archive_admission and artifact is not None:
            self.admitted.append(artifact)


def _blob(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _journal_record(
    generation: int, index: int, episode_id: str, **fields: Any
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "generation_id": generation,
        "episode_index": index,
        "started_at": "2026-11-01T10:00:00",
        "completed_at": "2026-11-01T10:05:00",
        "duration_seconds": 300.0,
        "error_message": "LEAK error message",
        "error_category": None,
        "prompt_responses": [{"content": "LEAK prompt"}],
        "raw_training_rows": [{"content": "LEAK row"}],
        "task_commit_token": "LEAK token",
        "task_breakdown": {"hypothesis_record": {"hypothesis": "LEAK breakdown"}},
        "trajectory": {
            "termination_category": "completed",
            "termination_reason": "LEAK reason",
            "extra": {
                "last_workflow_step": "verify",
                "budget_exhausted_kind": None,
                "stderr_tail": "LEAK stderr",
                "scratch_dir": "/tmp/LEAK-scratch",
            },
        },
        **fields,
    }


def _write_journal(path: Path, records: list[dict[str, Any]], tail: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(lines + tail, encoding="utf-8")


def _write_run(tmp_path: Path) -> _Run:
    dataset = tmp_path / "dataset.json"
    _write_dataset(dataset)
    builder = _Builder(tmp_path)

    old = _artifact("rule-old", 0.0005)
    builder.commit(
        _OLD_RUN,
        0,
        0,
        outcome=_outcome(None),
        artifact=old,
        receipt=builder.score(old),
        record=_record("alpha"),
        admit=True,
    )

    rule_a = _artifact("rule-a", 0.001)
    builder.commit(
        _RUN,
        0,
        0,
        outcome=_outcome(None, rewrite="Lower alpha ranks first."),
        artifact=rule_a,
        receipt=builder.score(rule_a),
        record=_record("alpha", hypothesis=_SCRIPT_BREAKOUT),
        admit=True,
    )
    rule_b = _artifact("rule-b", -0.002)
    builder.commit(
        _RUN,
        0,
        1,
        outcome=_outcome(None, referenced_descriptors=["gamma", "beta"]),
        artifact=rule_b,
        receipt=builder.score(rule_b),
        record=_record("beta + gamma"),
        admit=True,
    )
    rule_c = _artifact("rule-c", 0.003)
    builder.commit(
        _RUN,
        0,
        3,
        outcome={"rejection_stage": None, "materialized": True},
        artifact=rule_c,
        receipt=builder.score(rule_c, fresh=True),
        record=_record("alpha + beta", uncertainty="abs(gamma)"),
    )
    builder.commit(
        _RUN,
        0,
        4,
        outcome={"rejection_stage": "incomplete", "materialized": False},
    )
    builder.commit(
        _RUN,
        0,
        5,
        outcome=_outcome(
            "explore",
            materialized=False,
            referenced_descriptors=None,
            stop_cause="tool_budget",
            process={**_PROCESS, "calls": {"inspect_evidence": 2}},
        ),
    )

    rule_d = _artifact("rule-d", -0.004)
    builder.commit(
        _RUN,
        1,
        0,
        outcome=_outcome(None, parent_rule_hashes=["rule-a", "rule-b"]),
        artifact=rule_d,
        receipt=builder.score(rule_d),
        record=_record(
            "alpha", parent_hypothesis_ids=["rule-a", "rule-old", "made-up", "rule-a"]
        ),
        admit=True,
    )
    builder.commit(
        _RUN,
        1,
        1,
        outcome=_outcome(
            "guard",
            parent_rule_hashes=["rule-b", "rule-a"],
            referenced_descriptors=["delta"],
        ),
        artifact=_artifact("rule-e", 0.005),
        record=_record("delta", parent_hypothesis_ids=["rule-b"]),
    )
    strict = PairwiseScorer(
        ScorerConfig(
            folds=4,
            complexity_lambda=0.0,
            stage1_threshold=50.0,
            bootstrap_replicates=20,
        )
    )
    rule_f = _artifact("rule-f", 0.006)
    builder.commit(
        _RUN,
        1,
        2,
        outcome={"rejection_stage": "signal", "materialized": True},
        artifact=rule_f,
        receipt=builder.score(rule_f, strict),
        record=_record("zeta"),
    )
    rule_g = _artifact("rule-g", 0.007)
    builder.commit(
        _RUN,
        1,
        3,
        outcome={"rejection_stage": "marginal", "materialized": True},
        artifact=rule_g,
        receipt=builder.score(rule_g),
        record=_record("alpha"),
    )
    rule_h = _artifact("rule-h", 0.008)
    builder.commit(
        _RUN,
        1,
        4,
        outcome={"rejection_stage": None, "materialized": True},
        artifact=rule_h,
        receipt=builder.score(rule_h),
        record=_record("alpha"),
    )
    builder.commit(
        _RUN,
        1,
        5,
        outcome=_outcome("mystery"),
        artifact=_artifact("rule-i", 0.009),
        record=_record("alpha"),
    )
    builder.commit(
        _RUN,
        1,
        6,
        outcome=_outcome(None),
        artifact=rule_a,
        receipt=builder.receipts["rule-a"],
        record=_record("alpha"),
        admit=True,
    )
    builder.commit(
        _RUN,
        1,
        7,
        outcome={"rejection_stage": None, "materialized": False},
    )

    blobs = tmp_path / "blobs"
    missing = builder.receipt_hashes["rule-g"]
    (blobs / missing[:2] / missing).unlink()
    corrupt = builder.receipt_hashes["rule-h"]
    (blobs / corrupt[:2] / corrupt).write_bytes(b'{"marginal_pass": true}')

    generations = tmp_path / "generations"
    _write_journal(
        generations / _RUN / "generation_0" / "all_episodes.jsonl",
        [
            _journal_record(0, 0, f"ep-{_RUN}-0-0"),
            _journal_record(0, 1, f"ep-{_RUN}-0-1"),
            _journal_record(
                0,
                2,
                "ep-crash-first",
                error_category="docker",
                task_commit_token=None,
            ),
            _journal_record(
                0,
                2,
                "ep-crash-retry",
                error_category="harness",
                task_commit_token=None,
                trajectory={
                    "termination_category": "wall_clock",
                    "extra": {
                        "last_workflow_step": "code",
                        "budget_exhausted_kind": "task_tool_calls",
                        "stderr_tail": "LEAK stderr",
                    },
                },
            ),
            _journal_record(0, 3, f"ep-{_RUN}-0-3"),
            _journal_record(0, 4, "ep-from-an-older-attempt"),
        ],
        tail='{"episode_id": "ep-truncated", "generation_id": 0, "episode_index": 9',
    )
    _write_journal(
        generations / _RUN / "generation_1" / "all_episodes.jsonl",
        [
            _journal_record(1, 0, f"ep-{_RUN}-1-0"),
            _journal_record(1, 8, "ep-commit-prepared-not-applied"),
        ],
    )
    (generations / _RUN / "generation_2").mkdir()
    return _Run(
        archive=tmp_path / "archive.sqlite",
        blobs=blobs,
        generations=generations,
        dataset=dataset,
        receipts=builder.receipts,
        behavior_hashes=builder.behavior_hashes,
    )


@pytest.fixture(scope="module")
def built_run(tmp_path_factory) -> _Run:
    return _write_run(tmp_path_factory.mktemp("viz-run"))


def _copy_run(built_run: _Run, tmp_path: Path) -> _Run:
    shutil.copytree(built_run.archive.parent, tmp_path, dirs_exist_ok=True)
    return replace(
        built_run,
        archive=tmp_path / built_run.archive.name,
        blobs=tmp_path / built_run.blobs.name,
        generations=tmp_path / built_run.generations.name,
        dataset=tmp_path / built_run.dataset.name,
    )


def _episodes(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {episode["key"]: episode for episode in document["episodes"]}


def test_every_episode_stops_at_a_listed_stage_including_crash_and_old_rows(
    built_run, tmp_path
) -> None:
    document = _copy_run(built_run, tmp_path).extract()

    assert {key: episode["stage"] for key, episode in _episodes(document).items()} == {
        "g0-e0": "admitted",
        "g0-e1": "admitted",
        "g0-e2": "crash",
        "g0-e3": "rewrite",
        "g0-e4": "incomplete",
        "g0-e5": "explore",
        "g1-e0": "admitted",
        "g1-e1": "guard",
        "g1-e2": "signal",
        "g1-e3": "marginal",
        "g1-e4": "rewrite",
        "g1-e5": "mystery",
        "g1-e6": "archive_gate",
        "g1-e7": "error",
        "g1-e8": "pending",
    }
    assert [(stage["id"], stage["sieve"]) for stage in document["stages"]] == [
        ("explore", True),
        ("hypothesise", True),
        ("compile", True),
        ("submit", True),
        ("guard", True),
        ("signal", True),
        ("marginal", True),
        ("rewrite", True),
        ("archive_gate", True),
        ("admitted", True),
        ("incomplete", False),
        ("error", False),
        ("crash", False),
        ("mystery", False),
        ("pending", False),
    ]
    assert document["stages"][-1]["label"] == "Commit pending"
    assert [episode["key"] for episode in document["episodes"]][:6] == [
        "g0-e0",
        "g0-e1",
        "g0-e3",
        "g0-e4",
        "g0-e5",
        "g0-e2",
    ]
    assert document["generations"] == [0, 1, 2]
    assert document["archive_revision"] == 4
    assert document["sources"]["outcomes"] == 13
    assert document["sources"]["journal_only"] == 2


def test_crash_rows_use_the_last_journal_record_and_copy_only_allowlisted_fields(
    built_run, tmp_path
) -> None:
    run = _copy_run(built_run, tmp_path)
    document = run.extract()
    episodes = _episodes(document)
    serialized = json.dumps(document)

    assert episodes["g0-e2"]["episode_id"] == "ep-crash-retry"
    assert episodes["g0-e2"]["committed_at"] is None
    assert episodes["g0-e2"]["journal"] == {
        "episode_id": "ep-crash-retry",
        "started_at": "2026-11-01T10:00:00",
        "completed_at": "2026-11-01T10:05:00",
        "duration_seconds": 300.0,
        "error_category": "harness",
        "termination_category": "wall_clock",
        "last_workflow_step": "code",
        "budget_exhausted_kind": "task_tool_calls",
    }
    assert episodes["g0-e0"]["journal"]["last_workflow_step"] == "verify"
    assert episodes["g0-e4"]["journal"] is None
    assert episodes["g1-e8"]["stage"] == "pending"
    assert set(episodes["g1-e8"]["journal"]) == set(episodes["g0-e2"]["journal"])
    assert "g0-e9" not in episodes
    assert "LEAK" not in serialized
    assert not any(behavior in serialized for behavior in run.behavior_hashes)


def test_a_commit_landing_between_the_journal_and_archive_reads_is_not_a_crash(
    built_run, tmp_path, monkeypatch
) -> None:
    run = _copy_run(built_run, tmp_path)
    journal = run.generations / _RUN / "generation_1" / "all_episodes.jsonl"
    read_journals = viz_extractor._read_journals

    def commit_then_read_journals(run_dir: Path):
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_journal_record(1, 9, f"ep-{_RUN}-1-9")) + "\n")
        _Builder(tmp_path).commit(
            _RUN, 1, 9, outcome=_outcome("guard", materialized=False)
        )
        return read_journals(run_dir)

    monkeypatch.setattr(viz_extractor, "_read_journals", commit_then_read_journals)
    document = run.extract()

    assert _episodes(document)["g1-e9"]["stage"] == "guard"
    assert _episodes(document)["g1-e9"]["journal"]["episode_id"] == f"ep-{_RUN}-1-9"
    assert document["sources"]["journal_only"] == 2


def test_trusted_lineage_comes_from_outcomes_and_claims_only_link_archived_rules(
    built_run, tmp_path
) -> None:
    run = _copy_run(built_run, tmp_path)
    document = run.extract()
    episodes = _episodes(document)

    assert episodes["g1-e0"]["parents"] == ["rule-a", "rule-b"]
    assert episodes["g1-e1"]["parents"] == ["rule-b", "rule-a"]
    assert episodes["g0-e0"]["parents"] == []
    assert episodes["g1-e2"]["parents"] is None
    assert episodes["g1-e0"]["claimed_parents"] == ["rule-a", "rule-old", "made-up"]
    assert document["edges"] == [
        {"parent": "rule-a", "child": "g1-e0", "kind": "trusted"},
        {"parent": "rule-b", "child": "g1-e0", "kind": "trusted"},
        {"parent": "rule-a", "child": "g1-e0", "kind": "claimed"},
        {"parent": "rule-old", "child": "g1-e0", "kind": "claimed"},
        {"parent": "rule-b", "child": "g1-e1", "kind": "trusted"},
        {"parent": "rule-a", "child": "g1-e1", "kind": "trusted"},
        {"parent": "rule-b", "child": "g1-e1", "kind": "claimed"},
    ]
    assert document["rules"] == [
        {
            "rule_id": rule_id,
            "revision": revision,
            "generation": generation,
            "episode_key": key,
            "inputs": inputs,
            "parents": parents,
            "stage2_gain": run.receipts[rule_id].stage2_gain,
        }
        for rule_id, revision, generation, key, inputs, parents in (
            ("rule-a", 2, 0, "g0-e0", ["alpha"], []),
            ("rule-b", 3, 0, "g0-e1", ["beta", "gamma"], []),
            ("rule-d", 4, 1, "g1-e0", ["alpha"], ["rule-a", "rule-b"]),
        )
    ]


def test_old_archives_recompile_inputs_from_the_record_and_count_the_gaps(
    built_run, tmp_path
) -> None:
    document = _copy_run(built_run, tmp_path).extract()
    episodes = _episodes(document)

    assert episodes["g0-e3"]["inputs"] == ["alpha", "beta", "gamma"]
    assert episodes["g1-e2"]["inputs"] == ["zeta"]
    assert episodes["g0-e4"]["inputs"] is None
    assert episodes["g0-e5"]["inputs"] is None
    assert episodes["g0-e4"]["rule_id"] is None
    assert episodes["g0-e4"]["record"] is None
    assert document["sources"]["fields_not_recorded"] == {
        "parents": 6,
        "inputs": 6,
        "process": 6,
        "rewrite": 6,
    }


def test_new_outcome_fields_and_public_record_reach_the_episode(
    built_run, tmp_path
) -> None:
    document = _copy_run(built_run, tmp_path).extract()
    episodes = _episodes(document)

    assert episodes["g0-e0"]["process"] == _PROCESS
    assert episodes["g0-e0"]["rewrite"] == "Lower alpha ranks first."
    assert episodes["g0-e5"]["stop_cause"] == "tool_budget"
    assert episodes["g0-e5"]["process"]["calls"] == {"inspect_evidence": 2}
    assert episodes["g0-e3"]["process"] is None
    assert episodes["g0-e0"]["record"] == {
        "hypothesis": _SCRIPT_BREAKOUT,
        "rationale": "rationale",
        "applicability": "everywhere",
        "expression": "alpha",
        "uncertainty": "0.0",
        "discriminating_test": "test",
    }
    assert episodes["g0-e0"]["rule_id"] == "rule-a"
    assert episodes["g0-e0"]["training_success"] is True
    assert isinstance(episodes["g0-e0"]["committed_at"], float)


def test_scores_come_from_verified_receipt_blobs_and_bad_blobs_are_counted(
    built_run, tmp_path
) -> None:
    run = _copy_run(built_run, tmp_path)
    document = run.extract()
    episodes = _episodes(document)
    receipt = run.receipts["rule-d"]

    assert episodes["g1-e0"]["score"] == {
        "stage1_gain": receipt.stage1_gain,
        "stage1_threshold": receipt.stage1_threshold,
        "stage2_gain": receipt.stage2_gain,
        "stage2_lower_bound": receipt.stage2_lower_bound,
        "complexity": 3,
        "complexity_charge": receipt.complexity_charge,
        "signal_pass": receipt.signal_pass,
        "marginal_pass": receipt.marginal_pass,
        "archive_revision": 3,
        "axes": {
            item.axis: {
                "stage2_gain": item.stage2_gain,
                "stage2_lower_bound": item.stage2_lower_bound,
                "system_count": item.system_count,
            }
            for item in receipt.axis_diagnostics
        },
    }
    assert episodes["g1-e2"]["score"]["signal_pass"] is False
    assert episodes["g1-e3"]["score"] is None
    assert episodes["g1-e4"]["score"] is None
    assert episodes["g1-e1"]["score"] is None
    assert document["sources"]["missing_blobs"] == 2


def test_old_rows_that_passed_both_gates_stay_rewrite_without_their_receipt(
    built_run, tmp_path
) -> None:
    run = _copy_run(built_run, tmp_path)
    with closing(sqlite3.connect(run.archive)) as connection:
        (receipt_hash,) = connection.execute(
            """
            SELECT receipt_hash FROM outcomes
            WHERE run_id = ? AND generation_id = 0 AND episode_index = 3
            """,
            (_RUN,),
        ).fetchone()
    (run.blobs / receipt_hash[:2] / receipt_hash).unlink()

    document = run.extract()
    episode = _episodes(document)["g0-e3"]

    assert (episode["stage"], episode["score"]) == ("rewrite", None)
    assert document["sources"]["missing_blobs"] == 3


def test_latest_run_is_the_default_and_older_runs_stay_selectable(
    built_run, tmp_path
) -> None:
    run = _copy_run(built_run, tmp_path)

    latest = run.extract()
    older = run.extract(run_id=_OLD_RUN)

    assert latest["run_id"] == _RUN
    assert latest["runs_available"] == [_OLD_RUN, _RUN]
    assert older["run_id"] == _OLD_RUN
    assert [episode["key"] for episode in older["episodes"]] == ["g0-e0"]
    assert [rule["rule_id"] for rule in older["rules"]] == ["rule-old"]
    with pytest.raises(ValueError, match="no outcomes and no run directory"):
        run.extract(run_id="run-unknown")
    with pytest.raises(ValueError, match="plain directory name"):
        run.extract(run_id="../escape")


def test_extraction_never_writes_the_archive_even_while_a_writer_holds_the_lock(
    built_run, tmp_path
) -> None:
    run = _copy_run(built_run, tmp_path)
    blob_files = sorted(path for path in run.blobs.rglob("*"))
    writer = sqlite3.connect(run.archive, isolation_level=None)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "UPDATE metadata SET value = '99' WHERE key = 'archive_revision'"
        )
        before = (run.archive.read_bytes(), run.archive.stat().st_mtime_ns)

        document = run.extract()

        after = (run.archive.read_bytes(), run.archive.stat().st_mtime_ns)
    finally:
        writer.rollback()
        writer.close()

    assert after == before
    assert document["archive_revision"] == 4
    assert sorted(path for path in run.blobs.rglob("*")) == blob_files


def test_family_grouping_orders_families_and_inputs_by_name(
    built_run, tmp_path
) -> None:
    document = _copy_run(built_run, tmp_path).extract()
    inputs = document["inputs"]

    assert inputs["grouping"] == "family"
    assert [(group["id"], group["inputs"]) for group in inputs["groups"]] == [
        ("bulk", ["epsilon"]),
        ("chemistry", ["delta", "gamma"]),
        ("site", ["alpha", "beta", "zeta"]),
    ]
    assert inputs["descriptors"][0] == {
        "name": "alpha",
        "family": "site",
        "unit": "unitless",
        "value_type": "number",
    }
    frozen_path = tmp_path / "generations" / _RUN / "viz-input-groups.json"
    assert json.loads(frozen_path.read_text(encoding="utf-8")) == {
        "grouping": "family",
        "registry_hash": inputs["registry_hash"],
        "universe_hash": inputs["universe_hash"],
        "groups": inputs["groups"],
    }


def test_data_grouping_clusters_correlated_inputs_independent_of_file_order(
    built_run, tmp_path
) -> None:
    run = _copy_run(built_run, tmp_path)
    single_family = dict.fromkeys(_FAMILIES, "all")
    forward = tmp_path / "forward.json"
    backward = tmp_path / "backward.json"
    _write_dataset(forward, families=single_family)
    _write_dataset(backward, families=single_family, reverse=True)

    first = run.extract(dataset_path=forward, frozen_groups_path=tmp_path / "a.json")
    second = run.extract(dataset_path=backward, frozen_groups_path=tmp_path / "b.json")

    assert first["inputs"]["grouping"] == "data"
    assert [group["inputs"] for group in first["inputs"]["groups"]] == [
        ["alpha", "beta", "zeta"],
        ["delta", "gamma"],
        ["epsilon"],
    ]
    assert second["inputs"] == first["inputs"]


def test_frozen_groups_are_reused_and_a_different_dataset_is_refused(
    built_run, tmp_path
) -> None:
    run = _copy_run(built_run, tmp_path)
    frozen_path = tmp_path / "generations" / _RUN / "viz-input-groups.json"
    run.extract()
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["groups"][0]["label"] = "Edited by a human"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    reused = run.extract(input_grouping="auto")
    other_dataset = tmp_path / "other.json"
    _write_dataset(other_dataset, families=dict.fromkeys(_FAMILIES, "all"))

    assert reused["inputs"]["groups"][0]["label"] == "Edited by a human"
    with pytest.raises(ValueError, match="different dataset"):
        run.extract(dataset_path=other_dataset)
    with pytest.raises(ValueError, match="frozen as 'family', not 'data'"):
        run.extract(input_grouping="data")


def test_exported_page_carries_hostile_text_as_inert_data(built_run, tmp_path) -> None:
    document = _copy_run(built_run, tmp_path).extract()
    template = tmp_path / "template.html"
    template.write_text(_TEMPLATE, encoding="utf-8")

    index = write_viz_bundle(document, tmp_path / "out", template_path=template)
    page = index.read_text(encoding="utf-8")
    parser = _ScriptData()
    parser.feed(page)

    assert _SCRIPT_BREAKOUT not in page
    assert page.count("</script>") == 1
    assert json.loads(parser.data) == document
    assert json.loads((tmp_path / "out" / "viz.json").read_text()) == document


_CONFIG_PATHS = {
    "code_host_cache_path": "./code/cache",
    "docker_compose_dir": "./docker/top-level",
    "task_docker_compose_dir": "./docker/compose",
    "generation_output_dir": "./generations",
}


def _write_config(tmp_path: Path, **overrides: str) -> None:
    paths = {**_CONFIG_PATHS, **overrides}
    (tmp_path / "config.toml").write_text(
        "\n".join(
            [
                f'code_host_cache_path = "{paths["code_host_cache_path"]}"',
                f'docker_compose_dir = "{paths["docker_compose_dir"]}"',
                "[task.config]",
                'dataset_path = "./dataset.json"',
                'archive_path = "./archive.sqlite"',
                'blob_dir = "./blobs"',
                f'docker_compose_dir = "{paths["task_docker_compose_dir"]}"',
                "[generation]",
                f'generation_output_dir = "{paths["generation_output_dir"]}"',
            ]
        ),
        encoding="utf-8",
    )


def test_cli_reads_paths_from_config_and_writes_under_the_run_directory(
    built_run, tmp_path, monkeypatch, capsys
) -> None:
    _copy_run(built_run, tmp_path)
    _write_config(tmp_path)
    template = tmp_path / "template.html"
    template.write_text(_TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(viz_bundle, "DEFAULT_TEMPLATE_PATH", template)
    monkeypatch.chdir(tmp_path)

    export_main(["--config", "config.toml"])

    index = Path(capsys.readouterr().out.strip())
    assert index == Path("generations") / _RUN / "viz" / "index.html"
    payload = json.loads((tmp_path / index.parent / "viz.json").read_text())
    assert payload["run_id"] == _RUN
    assert payload["inputs"]["grouping"] == "family"


def test_cli_names_the_resolved_archive_path_when_none_exists(
    tmp_path, monkeypatch
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as raised:
        export_main(["--config", "config.toml"])

    message = str(raised.value)
    assert f"no archive at {tmp_path / 'archive.sqlite'}." in message
    assert f"current directory ({tmp_path})" in message
    assert "--archive" in message
    assert list(tmp_path.iterdir()) == [tmp_path / "config.toml"]


@pytest.mark.parametrize(
    "out",
    [
        "docker/compose/viz",
        "docker/top-level",
        "code/cache/slot_0/viz",
        "exports/public-evidence/viz",
    ],
)
def test_cli_refuses_output_an_agent_could_reach(
    built_run, tmp_path, monkeypatch, out: str
) -> None:
    _copy_run(built_run, tmp_path)
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="refusing"):
        export_main(["--config", "config.toml", "--out", out])

    assert not (tmp_path / out).exists()


@pytest.mark.parametrize(
    "override",
    [
        {"code_host_cache_path": "./generations"},
        {"docker_compose_dir": "./generations"},
        {"task_docker_compose_dir": f"./generations/{_RUN}"},
        {"generation_output_dir": "./exports/public-evidence/generations"},
    ],
)
def test_cli_refuses_a_default_output_an_agent_could_reach_before_writing(
    built_run, tmp_path, monkeypatch, override: dict[str, str]
) -> None:
    _copy_run(built_run, tmp_path)
    _write_config(tmp_path, **override)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / override.get("generation_output_dir", "generations") / _RUN

    with pytest.raises(SystemExit, match="refusing"):
        export_main(["--config", "config.toml"])

    assert not (run_dir / "viz").exists()
    assert not (run_dir / "viz-input-groups.json").exists()


def test_cli_refuses_frozen_groups_an_agent_could_reach_before_writing(
    built_run, tmp_path, monkeypatch
) -> None:
    _copy_run(built_run, tmp_path)
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    frozen = Path("exports/public-evidence/groups.json")

    with pytest.raises(SystemExit, match="refusing"):
        export_main(
            ["--config", "config.toml", "--out", "viz", "--frozen-groups", str(frozen)]
        )

    assert not (tmp_path / frozen).exists()
    assert not (tmp_path / "viz").exists()


def test_cli_expands_a_home_relative_out_before_checking_and_writing(
    built_run, tmp_path, monkeypatch, capsys
) -> None:
    run = _copy_run(built_run, tmp_path)
    home = tmp_path / "home"
    cwd = tmp_path / "public-evidence"
    cwd.mkdir()
    template = tmp_path / "template.html"
    template.write_text(_TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(viz_bundle, "DEFAULT_TEMPLATE_PATH", template)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    arguments = [
        "--archive",
        str(run.archive),
        "--blobs",
        str(run.blobs),
        "--generations",
        str(run.generations),
        "--dataset",
        str(run.dataset),
        "--out=~/viz",
    ]

    export_main(arguments)

    assert Path(capsys.readouterr().out.strip()) == home / "viz" / "index.html"
    assert (home / "viz" / "viz.json").is_file()
    assert list(cwd.iterdir()) == []
    monkeypatch.setenv("HOME", str(cwd / "home"))
    with pytest.raises(SystemExit, match="refusing.*public-evidence"):
        export_main(arguments)
    assert list(cwd.iterdir()) == []


def test_cli_explicit_paths_select_a_run_and_report_errors_cleanly(
    built_run, tmp_path, monkeypatch, capsys
) -> None:
    run = _copy_run(built_run, tmp_path)
    template = tmp_path / "template.html"
    template.write_text(_TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(viz_bundle, "DEFAULT_TEMPLATE_PATH", template)
    arguments = [
        "--archive",
        str(run.archive),
        "--blobs",
        str(run.blobs),
        "--generations",
        str(run.generations),
        "--dataset",
        str(run.dataset),
    ]

    export_main(
        [
            *arguments,
            "--run-id",
            _OLD_RUN,
            "--out",
            str(tmp_path / "viz"),
            "--input-grouping",
            "data",
            "--frozen-groups",
            str(tmp_path / "groups.json"),
        ]
    )

    assert Path(capsys.readouterr().out.strip()) == tmp_path / "viz" / "index.html"
    payload = json.loads((tmp_path / "viz" / "viz.json").read_text())
    assert payload["run_id"] == _OLD_RUN
    assert json.loads((tmp_path / "groups.json").read_text())["grouping"] == "data"
    with pytest.raises(SystemExit, match="export_viz: .*run-missing"):
        export_main([*arguments, "--run-id", "run-missing"])


class _ScriptData(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside = False
        self.data = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._inside = tag == "script" and ("id", "viz-data") in attrs

    def handle_endtag(self, tag: str) -> None:
        self._inside = False

    def handle_data(self, data: str) -> None:
        if self._inside:
            self.data += data
