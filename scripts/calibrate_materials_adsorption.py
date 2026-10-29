"""Calibrate the frozen materials stage-1 permutation-null threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.typing.training import save_durable_json
from tasks.materials_adsorption.compiler import RuleCompiler
from tasks.materials_adsorption.compiler import RuleValidationError
from tasks.materials_adsorption.dataset import EndpointStatus, load_dataset
from tasks.materials_adsorption.scoring import (
    PairwiseScorer,
    PlacementOutcome,
    ScorerConfig,
)


def _outcomes(loaded: Any) -> tuple[PlacementOutcome, ...]:
    result = []
    for record in loaded.records:
        if record.status is not EndpointStatus.VALID or record.energy is None:
            continue
        candidate_alias, system_alias = loaded.source_aliases[
            (record.source_system_id, record.source_placement_id)
        ]
        result.append(
            PlacementOutcome(
                candidate_alias=candidate_alias,
                system_alias=system_alias,
                energy=float(record.energy),
                axis=record.axis,
                chemistry_block=f"{record.adsorbate_group}:{record.catalyst_group}",
            )
        )
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "controls",
        type=Path,
        help="JSON list of {expression, applicable?, uncertainty?} controls",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--percentile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--complexity-lambda", type=float, default=0.001)
    args = parser.parse_args()

    loaded = load_dataset(args.dataset)
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    if not isinstance(controls, list) or not controls:
        raise ValueError("controls must be a non-empty JSON list")
    compiler = RuleCompiler(loaded.registry)
    artifacts = []
    artifacts_by_name = {}
    rejected_controls = []
    seen_names: set[str] = set()
    for control_index, control in enumerate(controls):
        if not isinstance(control, dict):
            raise ValueError("each control must be a JSON object")
        name = str(control.get("name") or f"control-{control_index}")
        if name in seen_names:
            raise ValueError(f"duplicate control name: {name}")
        seen_names.add(name)
        try:
            rule = compiler.compile(
                str(control["expression"]),
                applicable=str(control.get("applicable", "True")),
                uncertainty=str(control.get("uncertainty", "0.0")),
            )
        except RuleValidationError as exc:
            if control.get("expect_compile_error") is not True:
                raise
            rejected_controls.append({"name": name, "error": str(exc)})
            continue
        if control.get("expect_compile_error") is True:
            raise ValueError(f"control {name!r} unexpectedly compiled")
        artifact = rule.evaluate(
            loaded.universe.candidates,
            dataset_hash=loaded.universe.manifest_hash,
            universe_hash=loaded.universe.universe_hash,
        )
        artifacts.append(artifact)
        artifacts_by_name[name] = artifact

    scorer = PairwiseScorer(
        ScorerConfig(
            folds=args.folds,
            l2=args.l2,
            complexity_lambda=args.complexity_lambda,
        )
    )
    calibration = scorer.calibrate_stage1_null(
        artifacts,
        _outcomes(loaded),
        permutations=args.permutations,
        percentile=args.percentile,
        seed=args.seed,
    )
    calibrated = scorer.with_calibration(calibration)
    outcomes = _outcomes(loaded)
    control_report = []
    gains: dict[str, float] = {}
    for control_index, control in enumerate(controls):
        name = str(control.get("name") or f"control-{control_index}")
        artifact = artifacts_by_name.get(name)
        if artifact is None:
            continue
        duplicate_of = control.get("duplicate_of")
        archive = ()
        if duplicate_of is not None:
            try:
                archive = (artifacts_by_name[str(duplicate_of)],)
            except KeyError as exc:
                raise ValueError(
                    f"control {name!r} references unknown duplicate_of control"
                ) from exc
        receipt = calibrated.score(artifact, outcomes, archive=archive)
        gains[name] = receipt.stage1_gain
        result = {
            "name": name,
            "duplicate_of": duplicate_of,
            "stage1_gain": receipt.stage1_gain,
            "stage1_threshold": receipt.stage1_threshold,
            "signal_pass": receipt.signal_pass,
            "stage2_gain": receipt.stage2_gain,
            "stage2_lower_bound": receipt.stage2_lower_bound,
            "marginal_pass": receipt.marginal_pass,
        }
        if "expect_signal" in control and receipt.signal_pass is not bool(
            control["expect_signal"]
        ):
            raise ValueError(f"control {name!r} violated expect_signal")
        if "expect_marginal" in control and receipt.marginal_pass is not bool(
            control["expect_marginal"]
        ):
            raise ValueError(f"control {name!r} violated expect_marginal")
        control_report.append(result)

    for control_index, control in enumerate(controls):
        better_than = control.get("stage1_better_than")
        if better_than is None:
            continue
        name = str(control.get("name") or f"control-{control_index}")
        if name not in gains or str(better_than) not in gains:
            raise ValueError(f"control ordering references an unavailable control: {name!r}")
        if gains[name] <= gains[str(better_than)]:
            raise ValueError(
                f"control ordering failed: {name!r} did not beat {better_than!r}"
            )

    payload = calibration.to_dict()
    payload["control_report"] = control_report
    payload["compile_rejections"] = rejected_controls
    save_durable_json(payload, args.output)


if __name__ == "__main__":
    main()
