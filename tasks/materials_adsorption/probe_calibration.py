"""Frozen calibration and server-only ranking references for feature probes."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PERCENTILE_BIN_LABELS = (
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
_PERCENTILE_BIN_PERCENTILES = tuple(
    int(label.removeprefix("p")) for label in PERCENTILE_BIN_LABELS
)
SHARD_PARTITION_VERSION = "materials-evidence-shard-v2"
SHARD_NULL_PERMUTATIONS = 100
SHARD_NULL_PERMUTATION_SEED = 41
COMPLEXITY_SCHEME = "feature-expression-nodes-v1"


def _content_hash(domain: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + encoded).hexdigest()


def _ordered_finite(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise ValueError(f"{label} must contain finite values")
    return ordered


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
    return float(values[index])


def empirical_percentile(value: float, reference: Sequence[float]) -> float:
    """Return the inclusive empirical-CDF percentile in the range [0, 100]."""

    if not math.isfinite(value) or not reference:
        raise ValueError("percentile ranking requires finite non-empty inputs")
    return 100.0 * bisect.bisect_right(reference, value) / len(reference)


@dataclass(frozen=True)
class ShardNullCalibration:
    schema_version: int
    shard_partition_version: str
    permutations: int
    seed: int
    candidate_count: int
    null_sample_count: int
    quantile_method: str
    percentile_bin_labels: tuple[str, ...]
    quantile_edges: tuple[float, ...]
    stage1_threshold: float
    bar_percentile: float
    shard_universe_gains: Mapping[str, tuple[float, ...]]
    dataset_hash: str
    universe_hash: str
    descriptor_registry_hash: str
    compiler_hash: str
    scorer_hash: str
    fold_seed: str
    complexity_lambda: float
    complexity_scheme: str
    calibration_hash: str

    @classmethod
    def create(
        cls,
        *,
        shard_partition_version: str,
        permutations: int,
        seed: int,
        shard_statistics: Mapping[str, Sequence[float]],
        dataset_hash: str,
        universe_hash: str,
        descriptor_registry_hash: str,
        compiler_hash: str,
        scorer_hash: str,
        fold_seed: str,
        stage1_threshold: float,
        complexity_lambda: float,
        shard_universe_gains: Mapping[str, Sequence[float]] | None = None,
        candidate_count: int | None = None,
        complexity_scheme: str = COMPLEXITY_SCHEME,
    ) -> ShardNullCalibration:
        if permutations < 1:
            raise ValueError("shard-null permutations must be positive")
        if not shard_statistics:
            raise ValueError("shard-null calibration requires shard statistics")
        ordered_statistics = {
            str(alias): _ordered_finite(values, label=f"statistics for {alias}")
            for alias, values in sorted(shard_statistics.items())
        }
        lengths = {len(values) for values in ordered_statistics.values()}
        if len(lengths) != 1:
            raise ValueError("shard-null statistics must have equal lengths")
        per_shard_count = lengths.pop()
        inferred_candidate_count, remainder = divmod(per_shard_count, permutations)
        if remainder or inferred_candidate_count < 1:
            raise ValueError("shard-null statistics do not match permutation count")
        if candidate_count is None:
            candidate_count = inferred_candidate_count
        if candidate_count != inferred_candidate_count:
            raise ValueError("shard-null candidate count mismatch")

        pooled = tuple(
            sorted(value for values in ordered_statistics.values() for value in values)
        )
        edges = tuple(
            _nearest_rank(pooled, percentile / 100.0)
            for percentile in _PERCENTILE_BIN_PERCENTILES[1:]
        )
        universe_source = shard_universe_gains or ordered_statistics
        universe_gains = {
            str(alias): _ordered_finite(values, label=f"universe gains for {alias}")
            for alias, values in sorted(universe_source.items())
        }
        if set(universe_gains) != set(ordered_statistics):
            raise ValueError("shard-null universe gains aliases mismatch")
        threshold = float(stage1_threshold)
        if not math.isfinite(threshold):
            raise ValueError("stage-1 threshold must be finite")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "shard_partition_version": str(shard_partition_version),
            "permutations": int(permutations),
            "seed": int(seed),
            "candidate_count": int(candidate_count),
            "null_sample_count": len(pooled),
            "quantile_method": "nearest-rank-ceil-v1",
            "percentile_bin_labels": list(PERCENTILE_BIN_LABELS),
            "quantile_edges": list(edges),
            "stage1_threshold": threshold,
            "bar_percentile": empirical_percentile(threshold, pooled),
            "shard_universe_gains": {
                alias: list(values) for alias, values in universe_gains.items()
            },
            "dataset_hash": str(dataset_hash),
            "universe_hash": str(universe_hash),
            "descriptor_registry_hash": str(descriptor_registry_hash),
            "compiler_hash": str(compiler_hash),
            "scorer_hash": str(scorer_hash),
            "fold_seed": str(fold_seed),
            "complexity_lambda": float(complexity_lambda),
            "complexity_scheme": str(complexity_scheme),
        }
        return cls._from_validated_payload(payload)

    @classmethod
    def _from_validated_payload(
        cls, payload: Mapping[str, Any]
    ) -> ShardNullCalibration:
        hash_payload = dict(payload)
        calibration_hash = _content_hash("materials-shard-null-v1", hash_payload)
        return cls(
            schema_version=int(payload["schema_version"]),
            shard_partition_version=str(payload["shard_partition_version"]),
            permutations=int(payload["permutations"]),
            seed=int(payload["seed"]),
            candidate_count=int(payload["candidate_count"]),
            null_sample_count=int(payload["null_sample_count"]),
            quantile_method=str(payload["quantile_method"]),
            percentile_bin_labels=tuple(payload["percentile_bin_labels"]),
            quantile_edges=tuple(float(value) for value in payload["quantile_edges"]),
            stage1_threshold=float(payload["stage1_threshold"]),
            bar_percentile=float(payload["bar_percentile"]),
            shard_universe_gains={
                str(alias): tuple(float(value) for value in values)
                for alias, values in payload["shard_universe_gains"].items()
            },
            dataset_hash=str(payload["dataset_hash"]),
            universe_hash=str(payload["universe_hash"]),
            descriptor_registry_hash=str(payload["descriptor_registry_hash"]),
            compiler_hash=str(payload["compiler_hash"]),
            scorer_hash=str(payload["scorer_hash"]),
            fold_seed=str(payload["fold_seed"]),
            complexity_lambda=float(payload["complexity_lambda"]),
            complexity_scheme=str(payload["complexity_scheme"]),
            calibration_hash=calibration_hash,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShardNullCalibration:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported shard-null calibration schema")
        try:
            calibration = cls._from_validated_payload(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "calibration_hash"
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid shard-null calibration payload") from exc
        if calibration.percentile_bin_labels != PERCENTILE_BIN_LABELS:
            raise ValueError("shard-null percentile labels mismatch")
        if len(calibration.quantile_edges) != len(PERCENTILE_BIN_LABELS) - 1:
            raise ValueError("shard-null quantile edge count mismatch")
        if tuple(sorted(calibration.quantile_edges)) != calibration.quantile_edges:
            raise ValueError("shard-null quantile edges are not sorted")
        if calibration.quantile_method != "nearest-rank-ceil-v1":
            raise ValueError("unsupported shard-null quantile method")
        if calibration.complexity_scheme != COMPLEXITY_SCHEME:
            raise ValueError("unsupported shard-null complexity scheme")
        if any(
            tuple(sorted(values)) != values or not values
            for values in calibration.shard_universe_gains.values()
        ):
            raise ValueError("invalid shard-null universe gain distribution")
        if calibration.calibration_hash != payload.get("calibration_hash"):
            raise ValueError("shard-null calibration hash mismatch")
        return calibration

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shard_partition_version": self.shard_partition_version,
            "permutations": self.permutations,
            "seed": self.seed,
            "candidate_count": self.candidate_count,
            "null_sample_count": self.null_sample_count,
            "quantile_method": self.quantile_method,
            "percentile_bin_labels": list(self.percentile_bin_labels),
            "quantile_edges": list(self.quantile_edges),
            "stage1_threshold": self.stage1_threshold,
            "bar_percentile": self.bar_percentile,
            "shard_universe_gains": {
                alias: list(values)
                for alias, values in sorted(self.shard_universe_gains.items())
            },
            "dataset_hash": self.dataset_hash,
            "universe_hash": self.universe_hash,
            "descriptor_registry_hash": self.descriptor_registry_hash,
            "compiler_hash": self.compiler_hash,
            "scorer_hash": self.scorer_hash,
            "fold_seed": self.fold_seed,
            "complexity_lambda": self.complexity_lambda,
            "complexity_scheme": self.complexity_scheme,
            "calibration_hash": self.calibration_hash,
        }

    def percentile_bin(self, gain: float) -> str:
        if not math.isfinite(gain):
            raise ValueError("probe gain must be finite")
        return self.percentile_bin_labels[
            bisect.bisect_right(self.quantile_edges, gain)
        ]

    def universe_percentile(self, shard_alias: str, gain: float) -> float:
        try:
            reference = self.shard_universe_gains[shard_alias]
        except KeyError as exc:
            raise ValueError("unknown evidence shard for universe ranking") from exc
        return empirical_percentile(gain, reference)


@dataclass(frozen=True)
class AttainabilityReference:
    dataset_hash: str
    universe_hash: str
    descriptor_registry_hash: str
    compiler_hash: str
    scorer_hash: str
    stage1_threshold: float
    complexity_lambda: float
    gains: tuple[float, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AttainabilityReference:
        try:
            candidates = payload["candidates"]
            gains = _ordered_finite(
                tuple(float(row["stage1_gain"]) for row in candidates),
                label="attainability gains",
            )
            return cls(
                dataset_hash=str(payload["dataset_hash"]),
                universe_hash=str(payload["universe_hash"]),
                descriptor_registry_hash=str(payload["descriptor_registry_hash"]),
                compiler_hash=str(payload["compiler_hash"]),
                scorer_hash=str(payload["scorer_hash"]),
                stage1_threshold=float(payload["stage1_threshold"]),
                complexity_lambda=float(payload["complexity_lambda"]),
                gains=gains,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid attainability reference") from exc

    def percentile(self, gain: float) -> float:
        return empirical_percentile(gain, self.gains)
