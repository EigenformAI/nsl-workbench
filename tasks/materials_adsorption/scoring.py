from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


_SCORER_VERSION = "materials-adsorption-scorer-v3-vectorized"
_PAIR_SAMPLING_VERSION = "grouped-hash-oriented-pairs-v1"
_MAX_CACHED_ARCHIVE_FEATURE_BYTES = 256 * 1024 * 1024
_MAX_CACHED_PAIR_TOPOLOGIES = 8


def _content_hash(kind: str, payload: object) -> str:
    encoded = json.dumps(
        {"kind": kind, "payload": payload},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_digest(*parts: object) -> bytes:
    data = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(data).digest()


@dataclass(frozen=True)
class PlacementOutcome:
    candidate_alias: str
    system_alias: str
    energy: float
    axis: str
    chemistry_block: str

    def __post_init__(self) -> None:
        if not self.candidate_alias or not self.system_alias:
            raise ValueError("placement and system aliases must be non-empty")
        if not math.isfinite(self.energy):
            raise ValueError("placement energy must be finite")
        if not self.axis or not self.chemistry_block:
            raise ValueError("axis and chemistry block must be non-empty")


@dataclass(frozen=True)
class PairwiseExample:
    system_alias: str
    axis: str
    chemistry_block: str
    first_alias: str
    second_alias: str
    label: float
    feature_differences: tuple[float, ...]
    weight: float

    @property
    def features(self) -> tuple[float, ...]:
        return self.feature_differences


@dataclass(frozen=True, eq=False)
class _PairTopology:
    aliases: tuple[str, ...]
    system_aliases: tuple[str, ...]
    axes: tuple[str, ...]
    chemistry_blocks: tuple[str, ...]
    first_indices: NDArray[np.intp]
    second_indices: NDArray[np.intp]
    labels: NDArray[np.float64]
    weights: NDArray[np.float64]
    system_indices: NDArray[np.intp]
    pair_target_hash: str


@dataclass(frozen=True, eq=False)
class _PairwiseArrays:
    topology: _PairTopology
    features: NDArray[np.float64]


@dataclass(frozen=True)
class ScorerConfig:
    folds: int = 5
    l2: float = 0.1
    complexity_lambda: float = 0.001
    stage1_threshold: float = 0.0
    bootstrap_replicates: int = 1_000
    bootstrap_seed: int = 0
    lower_tail_probability: float = 0.05
    max_block_degradation: float = 0.05
    min_block_systems: int = 2
    max_pairs_per_system: int | None = 256
    pair_seed: str = _PAIR_SAMPLING_VERSION
    fold_seed: str = "grouped-system-folds-v1"
    max_iterations: int = 500
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        if self.l2 < 0 or self.complexity_lambda < 0:
            raise ValueError("regularization and complexity charges must be non-negative")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 < self.lower_tail_probability < 0.5:
            raise ValueError("lower_tail_probability must be between zero and one half")
        if self.max_block_degradation < 0:
            raise ValueError("max_block_degradation must be non-negative")
        if self.min_block_systems < 1:
            raise ValueError("min_block_systems must be positive")
        if self.max_pairs_per_system is not None and self.max_pairs_per_system < 1:
            raise ValueError("max_pairs_per_system must be positive when set")
        if self.max_iterations < 1 or self.tolerance <= 0:
            raise ValueError("solver iteration and tolerance settings must be positive")

    @property
    def scorer_hash(self) -> str:
        return _content_hash(
            "scorer-config",
            {"version": _SCORER_VERSION, **asdict(self)},
        )


@dataclass(frozen=True)
class FoldFitReceipt:
    fold: int
    training_systems: tuple[str, ...]
    validation_systems: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ModelReceipt:
    feature_hashes: tuple[str, ...]
    folds: tuple[FoldFitReceipt, ...]
    l2: float
    nonnegative_feature_coefficients: bool
    model_hash: str

    @classmethod
    def create(
        cls,
        *,
        feature_hashes: Sequence[str],
        folds: Sequence[FoldFitReceipt],
        l2: float,
    ) -> ModelReceipt:
        fold_tuple = tuple(folds)
        payload = {
            "feature_hashes": tuple(feature_hashes),
            "folds": [asdict(item) for item in fold_tuple],
            "l2": l2,
            "nonnegative_feature_coefficients": False,
        }
        return cls(
            feature_hashes=tuple(feature_hashes),
            folds=fold_tuple,
            l2=l2,
            nonnegative_feature_coefficients=False,
            model_hash=_content_hash("model-receipt", payload),
        )


@dataclass(frozen=True)
class AxisDiagnostic:
    """Numeric per-axis receipt telemetry; never an admission gate."""

    axis: str
    system_count: int
    stage1_gain: float
    stage2_gain: float
    stage2_lower_bound: float


@dataclass(frozen=True)
class BlockDiagnostic:
    """Numeric per-block telemetry plus the derived admission-gate outcome."""

    block: str
    system_count: int
    stage1_gain: float
    stage2_gain: float
    stage2_lower_bound: float
    catastrophic_degradation: bool


@dataclass(frozen=True)
class ScoringReceipt:
    """Immutable score evidence with numeric group diagnostics as telemetry."""

    candidate_rule_hash: str
    candidate_artifact_hash: str
    dataset_hash: str
    descriptor_hash: str
    compiler_hash: str
    universe_hash: str
    scorer_hash: str
    archive_revision: int
    archive_rule_hashes: tuple[str, ...]
    complexity: int
    complexity_charge: float
    stage1_raw_gain: float
    stage1_gain: float
    stage1_threshold: float
    signal_pass: bool
    stage2_raw_gain: float
    stage2_gain: float
    stage2_lower_bound: float
    marginal_pass: bool
    axis_diagnostics: tuple[AxisDiagnostic, ...]
    block_diagnostics: tuple[BlockDiagnostic, ...]
    intercept_model: ModelReceipt
    candidate_model: ModelReceipt
    baseline_model: ModelReceipt
    augmented_model: ModelReceipt
    receipt_hash: str = field(compare=True)

    @classmethod
    def create(cls, **values: Any) -> ScoringReceipt:
        payload = _jsonable(values)
        return cls(**values, receipt_hash=_content_hash("scoring-receipt", payload))

    @property
    def baseline_model_hash(self) -> str:
        return self.baseline_model.model_hash

    @property
    def augmented_model_hash(self) -> str:
        return self.augmented_model.model_hash

    @property
    def axis_gains(self) -> Mapping[str, float]:
        return {item.axis: item.stage2_gain for item in self.axis_diagnostics}

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class Stage1Score:
    complexity: int
    complexity_charge: float
    stage1_raw_gain: float
    stage1_gain: float
    stage1_threshold: float
    converged: bool
    signal_pass: bool


@dataclass(frozen=True)
class PermutationNullCalibration:
    schema_version: int
    percentile: float
    threshold: float
    permutations: int
    seed: int
    statistics: tuple[float, ...]
    scorer_hash: str
    compiler_hash: str
    universe_hash: str
    calibration_hash: str

    @classmethod
    def create(
        cls,
        *,
        percentile: float,
        permutations: int,
        seed: int,
        statistics: Sequence[float],
        scorer_hash: str,
        compiler_hash: str,
        universe_hash: str,
    ) -> PermutationNullCalibration:
        ordered = tuple(sorted(float(value) for value in statistics))
        index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
        threshold = ordered[max(0, index)]
        payload = {
            "schema_version": 4,
            "percentile": percentile,
            "threshold": threshold,
            "permutations": permutations,
            "seed": seed,
            "statistics": ordered,
            "scorer_hash": scorer_hash,
            "compiler_hash": compiler_hash,
            "universe_hash": universe_hash,
        }
        return cls(
            **payload,
            calibration_hash=_content_hash("permutation-null-calibration", payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PermutationNullCalibration:
        if payload.get("schema_version") != 4:
            raise ValueError("unsupported permutation calibration schema")
        calibration = cls.create(
            percentile=float(payload["percentile"]),
            permutations=int(payload["permutations"]),
            seed=int(payload["seed"]),
            statistics=tuple(float(value) for value in payload["statistics"]),
            scorer_hash=str(payload["scorer_hash"]),
            compiler_hash=str(payload["compiler_hash"]),
            universe_hash=str(payload["universe_hash"]),
        )
        if calibration.threshold != float(payload["threshold"]):
            raise ValueError("permutation calibration threshold mismatch")
        if calibration.calibration_hash != payload.get("calibration_hash"):
            raise ValueError("permutation calibration hash mismatch")
        return calibration


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _artifact_entries(artifact: Any) -> dict[str, tuple[bool, float]]:
    indexed: dict[str, tuple[bool, float]] = {}
    for entry in artifact.entries:
        alias = entry.candidate_alias
        if alias in indexed:
            raise ValueError(f"duplicate consequence entry for {alias!r}")
        applicable = bool(entry.applicable)
        if applicable:
            value = float(entry.value)
            if not math.isfinite(value):
                raise ValueError(f"non-finite consequence value for {alias!r}")
        else:
            value = 0.0
        indexed[alias] = (applicable, value)
    return indexed


def _build_pair_topology(
    outcomes: Iterable[PlacementOutcome],
    *,
    max_pairs_per_system: int | None,
    pair_seed: str,
) -> _PairTopology:
    by_alias: dict[str, PlacementOutcome] = {}
    by_system: dict[str, list[PlacementOutcome]] = {}
    for outcome in outcomes:
        if outcome.candidate_alias in by_alias:
            raise ValueError(f"duplicate placement outcome for {outcome.candidate_alias!r}")
        by_alias[outcome.candidate_alias] = outcome
        by_system.setdefault(outcome.system_alias, []).append(outcome)

    aliases = tuple(sorted(by_alias))
    alias_indices = {alias: index for index, alias in enumerate(aliases)}
    system_pairs: dict[str, list[tuple[PlacementOutcome, PlacementOutcome, float]]] = {}
    for system_alias in sorted(by_system):
        placements = sorted(by_system[system_alias], key=lambda item: item.candidate_alias)
        if len({item.axis for item in placements}) != 1:
            raise ValueError(f"system {system_alias!r} spans multiple evaluation axes")
        if len({item.chemistry_block for item in placements}) != 1:
            raise ValueError(f"system {system_alias!r} spans multiple chemistry blocks")

        pairs: list[tuple[PlacementOutcome, PlacementOutcome, float]] = []
        for first_index, canonical_first in enumerate(placements[:-1]):
            for canonical_second in placements[first_index + 1 :]:
                if canonical_first.energy == canonical_second.energy:
                    continue
                first, second = canonical_first, canonical_second
                if _stable_digest(
                    pair_seed,
                    system_alias,
                    canonical_first.candidate_alias,
                    canonical_second.candidate_alias,
                )[0] & 1:
                    first, second = canonical_second, canonical_first
                pairs.append(
                    (
                        first,
                        second,
                        1.0 if first.energy < second.energy else 0.0,
                    )
                )

        if max_pairs_per_system is not None and len(pairs) > max_pairs_per_system:
            pairs.sort(
                key=lambda pair: _stable_digest(
                    pair_seed,
                    "sample",
                    system_alias,
                    min(pair[0].candidate_alias, pair[1].candidate_alias),
                    max(pair[0].candidate_alias, pair[1].candidate_alias),
                )
            )
            pairs = pairs[:max_pairs_per_system]
        pairs.sort(key=lambda pair: (pair[0].candidate_alias, pair[1].candidate_alias))
        if pairs:
            system_pairs[system_alias] = pairs

    if not system_pairs:
        empty_indices = np.empty(0, dtype=np.intp)
        empty_floats = np.empty(0, dtype=np.float64)
        return _PairTopology(
            aliases=aliases,
            system_aliases=(),
            axes=(),
            chemistry_blocks=(),
            first_indices=empty_indices,
            second_indices=empty_indices.copy(),
            labels=empty_floats,
            weights=empty_floats.copy(),
            system_indices=empty_indices.copy(),
            pair_target_hash=hashlib.sha256().hexdigest(),
        )

    system_weight = 1.0 / len(system_pairs)
    system_aliases = tuple(sorted(system_pairs))
    axes: list[str] = []
    chemistry_blocks: list[str] = []
    first_indices: list[int] = []
    second_indices: list[int] = []
    labels: list[float] = []
    weights: list[float] = []
    pair_system_indices: list[int] = []
    target_digest = hashlib.sha256()
    for system_index, system_alias in enumerate(system_aliases):
        pairs = system_pairs[system_alias]
        axes.append(pairs[0][0].axis)
        chemistry_blocks.append(pairs[0][0].chemistry_block)
        pair_weight = system_weight / len(pairs)
        for first, second, label in pairs:
            first_indices.append(alias_indices[first.candidate_alias])
            second_indices.append(alias_indices[second.candidate_alias])
            labels.append(label)
            weights.append(pair_weight)
            pair_system_indices.append(system_index)
            target_digest.update(
                "\x1f".join(
                    (
                        system_alias,
                        first.candidate_alias,
                        second.candidate_alias,
                        repr(label),
                        repr(pair_weight),
                    )
                ).encode("utf-8")
            )
            target_digest.update(b"\n")
    topology = _PairTopology(
        aliases=aliases,
        system_aliases=system_aliases,
        axes=tuple(axes),
        chemistry_blocks=tuple(chemistry_blocks),
        first_indices=np.asarray(first_indices, dtype=np.intp),
        second_indices=np.asarray(second_indices, dtype=np.intp),
        labels=np.asarray(labels, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        system_indices=np.asarray(pair_system_indices, dtype=np.intp),
        pair_target_hash=target_digest.hexdigest(),
    )
    for array in (
        topology.first_indices,
        topology.second_indices,
        topology.labels,
        topology.weights,
        topology.system_indices,
    ):
        array.flags.writeable = False
    return topology


def _artifact_feature_column(
    topology: _PairTopology, artifact: Any
) -> NDArray[np.float64]:
    entries = _artifact_entries(artifact)
    values = np.zeros(len(topology.aliases), dtype=np.float64)
    applicable = np.zeros(len(topology.aliases), dtype=np.bool_)
    for index, alias in enumerate(topology.aliases):
        entry = entries.get(alias)
        if entry is not None and entry[0]:
            applicable[index] = True
            values[index] = entry[1]
    pair_applicable = applicable[topology.first_indices] & applicable[topology.second_indices]
    differences = np.zeros(topology.labels.size, dtype=np.float64)
    differences[pair_applicable] = (
        values[topology.first_indices[pair_applicable]]
        - values[topology.second_indices[pair_applicable]]
    )
    differences.flags.writeable = False
    return differences


def _feature_matrix(
    topology: _PairTopology, artifacts: Sequence[Any]
) -> NDArray[np.float64]:
    if not artifacts:
        matrix = np.empty((topology.labels.size, 0), dtype=np.float64)
    else:
        matrix = np.column_stack(
            tuple(_artifact_feature_column(topology, artifact) for artifact in artifacts)
        )
    matrix.flags.writeable = False
    return matrix


def build_pairwise_examples(
    outcomes: Iterable[PlacementOutcome],
    artifacts: Sequence[Any],
    *,
    max_pairs_per_system: int | None = None,
    pair_seed: str = _PAIR_SAMPLING_VERSION,
) -> tuple[PairwiseExample, ...]:
    """Build stable within-system comparisons with equal total system weight."""
    topology = _build_pair_topology(
        outcomes,
        max_pairs_per_system=max_pairs_per_system,
        pair_seed=pair_seed,
    )
    features = _feature_matrix(topology, artifacts)
    examples: list[PairwiseExample] = []
    for pair_index in range(topology.labels.size):
        system_index = int(topology.system_indices[pair_index])
        examples.append(
            PairwiseExample(
                system_alias=topology.system_aliases[system_index],
                axis=topology.axes[system_index],
                chemistry_block=topology.chemistry_blocks[system_index],
                first_alias=topology.aliases[int(topology.first_indices[pair_index])],
                second_alias=topology.aliases[int(topology.second_indices[pair_index])],
                label=float(topology.labels[pair_index]),
                feature_differences=tuple(float(value) for value in features[pair_index]),
                weight=float(topology.weights[pair_index]),
            )
        )
    return tuple(examples)


@dataclass(frozen=True)
class _CrossValidationResult:
    system_losses: Mapping[str, float]
    receipt: ModelReceipt


def _group_folds(
    systems: Iterable[str], folds: int, fold_seed: str
) -> tuple[tuple[str, ...], ...]:
    ordered = sorted(set(systems), key=lambda item: (_stable_digest(fold_seed, item), item))
    if len(ordered) < 2:
        raise ValueError("grouped cross-validation requires at least two systems")
    fold_count = min(folds, len(ordered))
    grouped: list[list[str]] = [[] for _ in range(fold_count)]
    for index, system in enumerate(ordered):
        grouped[index % fold_count].append(system)
    return tuple(tuple(sorted(group)) for group in grouped)


def _deduplicate_columns_array(
    data: _PairwiseArrays, feature_indices: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int | None, ...]]:
    unique_indices: list[int] = []
    requested_to_unique: list[int | None] = []
    seen: dict[bytes, int] = {}
    weights = data.topology.weights
    total_weight = float(weights.sum())
    for feature_index in feature_indices:
        values = data.features[:, feature_index]
        mean = float(np.dot(weights, values) / total_weight)
        centered = values - mean
        variance = float(np.dot(weights, centered * centered) / total_weight)
        scale = math.sqrt(variance) if variance > 1e-24 else 1.0
        normalized = np.round(centered / scale, decimals=12)
        nonzero = np.flatnonzero(np.abs(normalized) > 1e-12)
        if nonzero.size and normalized[int(nonzero[0])] < 0.0:
            normalized = -normalized
        normalized[normalized == 0.0] = 0.0
        column = normalized.tobytes()
        existing = seen.get(column)
        if existing is None:
            existing = len(unique_indices)
            seen[column] = existing
            unique_indices.append(feature_index)
            requested_to_unique.append(existing)
        else:
            requested_to_unique.append(None)
    return tuple(unique_indices), tuple(requested_to_unique)


def _fit_logistic_array(
    rows: NDArray[np.float64],
    labels: NDArray[np.float64],
    weights: NDArray[np.float64],
    config: ScorerConfig,
    *,
    initial_coefficients: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], int, bool]:
    normalized_weights = weights / weights.sum()
    if initial_coefficients is None:
        coefficients = np.zeros(rows.shape[1], dtype=np.float64)
    else:
        coefficients = np.asarray(initial_coefficients, dtype=np.float64).copy()
        if coefficients.shape != (rows.shape[1],) or not np.all(np.isfinite(coefficients)):
            raise ValueError("initial coefficients are incompatible or non-finite")

    for iteration in range(1, config.max_iterations + 1):
        scores = rows @ coefficients
        probabilities = np.empty_like(scores)
        nonnegative = scores >= 0.0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
        exponentials = np.exp(scores[~nonnegative])
        probabilities[~nonnegative] = exponentials / (1.0 + exponentials)
        residual_weights = normalized_weights * (probabilities - labels)
        gradient = rows.T @ residual_weights
        if coefficients.size > 1:
            gradient[1:] += config.l2 * coefficients[1:]
        curvature_weights = normalized_weights * probabilities * (1.0 - probabilities)
        hessian = rows.T @ (rows * curvature_weights[:, None])
        if coefficients.size > 1:
            feature_diagonal = np.arange(1, coefficients.size)
            hessian[feature_diagonal, feature_diagonal] += config.l2
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
            return coefficients, iteration, False
        try:
            change = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return coefficients, iteration, False
        updated = coefficients - change
        if not np.all(np.isfinite(updated)):
            return coefficients, iteration, False
        coefficients = updated
        if float(np.max(np.abs(change))) <= config.tolerance:
            return coefficients, iteration, True
    return coefficients, config.max_iterations, False


def _cross_validate(
    data: _PairwiseArrays,
    feature_indices: Sequence[int],
    feature_hashes: Sequence[str],
    config: ScorerConfig,
    *,
    initial: _CrossValidationResult | None = None,
    initial_feature_indices: Sequence[int] = (),
) -> _CrossValidationResult:
    topology = data.topology
    grouped_folds = _group_folds(
        topology.system_aliases, config.folds, config.fold_seed
    )
    unique_indices, requested_to_unique = _deduplicate_columns_array(data, feature_indices)
    all_systems = set(topology.system_aliases)
    system_lookup = {system: index for index, system in enumerate(topology.system_aliases)}
    system_loss_totals = np.zeros(len(topology.system_aliases), dtype=np.float64)
    system_weight_totals = np.zeros(len(topology.system_aliases), dtype=np.float64)
    fold_receipts: list[FoldFitReceipt] = []

    for fold_index, validation_system_tuple in enumerate(grouped_folds):
        validation_systems = set(validation_system_tuple)
        training_systems = all_systems - validation_systems
        validation_system_indices = np.fromiter(
            (system_lookup[system] for system in validation_system_tuple),
            dtype=np.intp,
        )
        validation_mask = np.isin(topology.system_indices, validation_system_indices)
        training_mask = ~validation_mask
        training_weights = topology.weights[training_mask]
        selected = data.features[np.ix_(training_mask, unique_indices)]
        total_weight = float(training_weights.sum())
        if unique_indices:
            means_array = (training_weights @ selected) / total_weight
            centered = selected - means_array
            variances = (training_weights @ (centered * centered)) / total_weight
            scales_array = np.where(variances > 1e-24, np.sqrt(variances), 1.0)
            rows = np.column_stack((np.ones(selected.shape[0]), centered / scales_array))
        else:
            means_array = np.empty(0, dtype=np.float64)
            scales_array = np.empty(0, dtype=np.float64)
            rows = np.ones((int(training_mask.sum()), 1), dtype=np.float64)
        initial_coefficients = None
        if initial is not None:
            initial_fold = initial.receipt.folds[fold_index]
            initial_by_feature = {
                feature_index: initial_fold.coefficients[position + 1]
                for position, feature_index in enumerate(initial_feature_indices)
            }
            initial_coefficients = np.zeros(len(unique_indices) + 1, dtype=np.float64)
            initial_coefficients[0] = initial_fold.coefficients[0]
            for position, feature_index in enumerate(unique_indices, start=1):
                initial_coefficients[position] = initial_by_feature.get(feature_index, 0.0)
        coefficients, iterations, converged = _fit_logistic_array(
            rows,
            topology.labels[training_mask],
            training_weights,
            config,
            initial_coefficients=initial_coefficients,
        )

        expanded_means: list[float] = []
        expanded_scales: list[float] = []
        expanded_coefficients = [float(coefficients[0])]
        for unique_position in requested_to_unique:
            if unique_position is None:
                expanded_means.append(0.0)
                expanded_scales.append(1.0)
                expanded_coefficients.append(0.0)
            else:
                expanded_means.append(float(means_array[unique_position]))
                expanded_scales.append(float(scales_array[unique_position]))
                expanded_coefficients.append(float(coefficients[unique_position + 1]))

        validation_features = data.features[np.ix_(validation_mask, unique_indices)]
        if unique_indices:
            validation_rows = np.column_stack(
                (
                    np.ones(validation_features.shape[0]),
                    (validation_features - means_array) / scales_array,
                )
            )
        else:
            validation_rows = np.ones((int(validation_mask.sum()), 1), dtype=np.float64)
        validation_scores = validation_rows @ coefficients
        validation_labels = topology.labels[validation_mask]
        losses = np.logaddexp(0.0, validation_scores) - validation_labels * validation_scores
        validation_weights = topology.weights[validation_mask]
        validation_pair_systems = topology.system_indices[validation_mask]
        system_loss_totals += np.bincount(
            validation_pair_systems,
            weights=validation_weights * losses,
            minlength=len(topology.system_aliases),
        )
        system_weight_totals += np.bincount(
            validation_pair_systems,
            weights=validation_weights,
            minlength=len(topology.system_aliases),
        )

        fold_receipts.append(
            FoldFitReceipt(
                fold=fold_index,
                training_systems=tuple(sorted(training_systems)),
                validation_systems=tuple(sorted(validation_systems)),
                means=tuple(expanded_means),
                scales=tuple(expanded_scales),
                coefficients=tuple(expanded_coefficients),
                iterations=iterations,
                converged=converged,
            )
        )

    system_losses = {
        system: float(
            system_loss_totals[system_lookup[system]]
            / system_weight_totals[system_lookup[system]]
        )
        for system in sorted(topology.system_aliases)
    }
    return _CrossValidationResult(
        system_losses=system_losses,
        receipt=ModelReceipt.create(
            feature_hashes=feature_hashes,
            folds=fold_receipts,
            l2=config.l2,
        ),
    )


def _mean_delta(
    baseline: Mapping[str, float], augmented: Mapping[str, float]
) -> float:
    systems = sorted(baseline)
    if systems != sorted(augmented):
        raise ValueError("cross-validation results cover different systems")
    return sum(baseline[item] - augmented[item] for item in systems) / len(systems)


def _bootstrap_lower_bound(
    deltas: Mapping[str, float], config: ScorerConfig, *, salt: str
) -> float:
    systems = sorted(deltas)
    seed_bytes = _stable_digest(config.bootstrap_seed, salt)
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    replicates = []
    for _ in range(config.bootstrap_replicates):
        replicates.append(
            sum(deltas[systems[rng.randrange(len(systems))]] for _ in systems) / len(systems)
        )
    replicates.sort()
    index = max(
        0,
        min(
            len(replicates) - 1,
            math.floor(config.lower_tail_probability * len(replicates)),
        ),
    )
    return replicates[index]


def _artifact_hash(artifact: Any) -> str:
    artifact_hash = getattr(artifact, "artifact_hash", None)
    if artifact_hash:
        return str(artifact_hash)
    return _content_hash(
        "consequence-artifact-fallback",
        {
            "rule_hash": artifact.rule_hash,
            "entries": [
                {
                    "candidate_alias": entry.candidate_alias,
                    "applicable": entry.applicable,
                    "value": entry.value,
                }
                for entry in artifact.entries
            ],
        },
    )


def _outcome_target_hash(outcomes: Sequence[PlacementOutcome]) -> str:
    digest = hashlib.sha256()
    for outcome in sorted(outcomes, key=lambda item: item.candidate_alias):
        digest.update(
            "\x1f".join(
                (
                    outcome.candidate_alias,
                    outcome.system_alias,
                    repr(outcome.energy),
                    outcome.axis,
                    outcome.chemistry_block,
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _descriptor_hash(artifact: Any) -> str:
    value = getattr(artifact, "descriptor_hash", None)
    if value is None:
        value = getattr(artifact, "descriptor_registry_hash")
    return str(value)


@dataclass(frozen=True, eq=False)
class _BaselineCacheEntry:
    intercept: _CrossValidationResult
    baseline: _CrossValidationResult
    archive_features: NDArray[np.float64] | None


class PairwiseScorer:
    def __init__(self, config: ScorerConfig | None = None) -> None:
        self.config = config or ScorerConfig()
        self._baseline_cache: dict[tuple[Any, ...], _BaselineCacheEntry] = {}
        self._baseline_cache_lock = threading.Lock()
        self._baseline_cache_hits = 0
        self._baseline_cache_misses = 0
        self._pair_topology_cache: dict[tuple[str, str, int | None, str], _PairTopology] = {}
        self._pair_topology_cache_lock = threading.Lock()
        self._pair_topology_cache_hits = 0
        self._pair_topology_cache_misses = 0

    def baseline_cache_info(self) -> dict[str, int]:
        with self._baseline_cache_lock:
            return {
                "entries": len(self._baseline_cache),
                "hits": self._baseline_cache_hits,
                "misses": self._baseline_cache_misses,
            }

    def pair_topology_cache_info(self) -> dict[str, int]:
        with self._pair_topology_cache_lock:
            return {
                "entries": len(self._pair_topology_cache),
                "hits": self._pair_topology_cache_hits,
                "misses": self._pair_topology_cache_misses,
            }

    def _pair_topology(
        self,
        outcomes: Sequence[PlacementOutcome],
        *,
        dataset_hash: str,
    ) -> _PairTopology:
        cache_key = (
            dataset_hash,
            _outcome_target_hash(outcomes),
            self.config.max_pairs_per_system,
            self.config.pair_seed,
        )
        with self._pair_topology_cache_lock:
            cached = self._pair_topology_cache.get(cache_key)
            if cached is not None:
                self._pair_topology_cache_hits += 1
                return cached
        topology = _build_pair_topology(
            outcomes,
            max_pairs_per_system=self.config.max_pairs_per_system,
            pair_seed=self.config.pair_seed,
        )
        with self._pair_topology_cache_lock:
            existing = self._pair_topology_cache.get(cache_key)
            if existing is not None:
                self._pair_topology_cache_hits += 1
                return existing
            if len(self._pair_topology_cache) >= _MAX_CACHED_PAIR_TOPOLOGIES:
                self._pair_topology_cache.pop(next(iter(self._pair_topology_cache)))
            self._pair_topology_cache[cache_key] = topology
            self._pair_topology_cache_misses += 1
        return topology

    def score(
        self,
        candidate: Any,
        outcomes: Iterable[PlacementOutcome],
        *,
        archive: Sequence[Any] = (),
        archive_revision: int = 0,
    ) -> ScoringReceipt:
        if archive_revision < 0:
            raise ValueError("archive_revision must be non-negative")
        archive_tuple = tuple(sorted(archive, key=lambda item: item.rule_hash))
        self._validate_artifacts(candidate, archive_tuple)
        outcome_tuple = tuple(outcomes)
        topology = self._pair_topology(
            outcome_tuple,
            dataset_hash=str(candidate.dataset_hash),
        )
        if not topology.labels.size:
            raise ValueError("scoring requires at least one non-tied pair")

        archive_count = len(archive_tuple)
        archive_hashes = tuple(str(item.rule_hash) for item in archive_tuple)
        candidate_hash = str(candidate.rule_hash)
        cache_key = (
            archive_revision,
            self.config.scorer_hash,
            str(candidate.dataset_hash),
            str(candidate.universe_hash),
            topology.pair_target_hash,
            tuple((_artifact_hash(item), str(item.rule_hash)) for item in archive_tuple),
        )
        with self._baseline_cache_lock:
            cached = self._baseline_cache.get(cache_key)
            if cached is None:
                archive_features = _feature_matrix(topology, archive_tuple)
                archive_data = _PairwiseArrays(topology, archive_features)
                intercept = _cross_validate(archive_data, (), (), self.config)
                baseline = (
                    intercept
                    if not archive_count
                    else _cross_validate(
                        archive_data,
                        tuple(range(archive_count)),
                        archive_hashes,
                        self.config,
                        initial=intercept,
                    )
                )
                # Keep lightweight fits per revision, but dense columns only for the latest one.
                for existing_key, existing_entry in tuple(self._baseline_cache.items()):
                    if existing_entry.archive_features is not None:
                        self._baseline_cache[existing_key] = replace(
                            existing_entry, archive_features=None
                        )
                retained_features = (
                    archive_features
                    if archive_features.nbytes <= _MAX_CACHED_ARCHIVE_FEATURE_BYTES
                    else None
                )
                cached = _BaselineCacheEntry(intercept, baseline, retained_features)
                if len(self._baseline_cache) >= 64:
                    self._baseline_cache.pop(next(iter(self._baseline_cache)))
                self._baseline_cache[cache_key] = cached
                self._baseline_cache_misses += 1
            else:
                self._baseline_cache_hits += 1
        intercept = cached.intercept
        baseline = cached.baseline
        archive_features = cached.archive_features
        if archive_features is None:
            archive_features = _feature_matrix(topology, archive_tuple)
        candidate_feature = _artifact_feature_column(topology, candidate)
        candidate_features = candidate_feature[:, None]
        candidate_features.flags.writeable = False
        candidate_data = _PairwiseArrays(topology, candidate_features)
        candidate_only = _cross_validate(
            candidate_data,
            (0,),
            (candidate_hash,),
            self.config,
            initial=intercept,
        )
        if not archive_count:
            augmented = candidate_only
        else:
            augmented_features = np.column_stack(
                (archive_features, candidate_feature)
            )
            augmented_features.flags.writeable = False
            augmented = _cross_validate(
                _PairwiseArrays(topology, augmented_features),
                tuple(range(archive_count + 1)),
                archive_hashes + (candidate_hash,),
                self.config,
                initial=baseline,
                initial_feature_indices=tuple(range(archive_count)),
            )

        complexity = int(getattr(candidate, "complexity", 0))
        complexity_charge = self.config.complexity_lambda * complexity
        stage1_raw = _mean_delta(intercept.system_losses, candidate_only.system_losses)
        stage1_gain = stage1_raw - complexity_charge
        stage2_deltas = {
            system: baseline.system_losses[system] - augmented.system_losses[system]
            for system in baseline.system_losses
        }
        stage2_raw = sum(stage2_deltas.values()) / len(stage2_deltas)
        stage2_gain = stage2_raw - complexity_charge
        stage2_lower_bound = (
            _bootstrap_lower_bound(stage2_deltas, self.config, salt=candidate_hash)
            - complexity_charge
        )

        system_metadata = {
            system: (topology.axes[index], topology.chemistry_blocks[index])
            for index, system in enumerate(topology.system_aliases)
        }
        axes: dict[str, list[str]] = {}
        for system in sorted(stage2_deltas):
            axes.setdefault(system_metadata[system][0], []).append(system)
        axis_diagnostics: list[AxisDiagnostic] = []
        stage1_deltas = {
            system: intercept.system_losses[system] - candidate_only.system_losses[system]
            for system in intercept.system_losses
        }
        for axis in sorted(axes):
            systems = axes[axis]
            axis_stage1 = sum(stage1_deltas[item] for item in systems) / len(systems)
            axis_stage2_raw = sum(stage2_deltas[item] for item in systems) / len(systems)
            axis_lcb = _bootstrap_lower_bound(
                {item: stage2_deltas[item] for item in systems},
                self.config,
                salt=f"{candidate_hash}:axis:{axis}",
            )
            axis_diagnostics.append(
                AxisDiagnostic(
                    axis=axis,
                    system_count=len(systems),
                    stage1_gain=axis_stage1,
                    stage2_gain=axis_stage2_raw,
                    stage2_lower_bound=axis_lcb,
                )
            )

        blocks: dict[str, list[str]] = {}
        for system in sorted(stage2_deltas):
            blocks.setdefault(system_metadata[system][1], []).append(system)
        block_diagnostics: list[BlockDiagnostic] = []
        catastrophic_block = False
        for block in sorted(blocks):
            systems = blocks[block]
            block_stage1 = sum(stage1_deltas[item] for item in systems) / len(systems)
            block_stage2 = sum(stage2_deltas[item] for item in systems) / len(systems)
            block_lcb = _bootstrap_lower_bound(
                {item: stage2_deltas[item] for item in systems},
                self.config,
                salt=f"{candidate_hash}:block:{block}",
            )
            degraded = (
                len(systems) >= self.config.min_block_systems
                and block_stage2 < -self.config.max_block_degradation
            )
            catastrophic_block = catastrophic_block or degraded
            block_diagnostics.append(
                BlockDiagnostic(
                    block=block,
                    system_count=len(systems),
                    stage1_gain=block_stage1,
                    stage2_gain=block_stage2,
                    stage2_lower_bound=block_lcb,
                    catastrophic_degradation=degraded,
                )
            )

        stage1_converged = all(
            fold.converged
            for model in (intercept.receipt, candidate_only.receipt)
            for fold in model.folds
        )
        stage2_converged = all(
            fold.converged
            for model in (baseline.receipt, augmented.receipt)
            for fold in model.folds
        )
        signal_pass = stage1_converged and stage1_gain > self.config.stage1_threshold
        marginal_pass = (
            signal_pass
            and stage2_converged
            and stage2_lower_bound > 0.0
            and not catastrophic_block
        )
        return ScoringReceipt.create(
            candidate_rule_hash=candidate_hash,
            candidate_artifact_hash=_artifact_hash(candidate),
            dataset_hash=str(candidate.dataset_hash),
            descriptor_hash=_descriptor_hash(candidate),
            compiler_hash=str(candidate.compiler_hash),
            universe_hash=str(candidate.universe_hash),
            scorer_hash=self.config.scorer_hash,
            archive_revision=archive_revision,
            archive_rule_hashes=archive_hashes,
            complexity=complexity,
            complexity_charge=complexity_charge,
            stage1_raw_gain=stage1_raw,
            stage1_gain=stage1_gain,
            stage1_threshold=self.config.stage1_threshold,
            signal_pass=signal_pass,
            stage2_raw_gain=stage2_raw,
            stage2_gain=stage2_gain,
            stage2_lower_bound=stage2_lower_bound,
            marginal_pass=marginal_pass,
            axis_diagnostics=tuple(axis_diagnostics),
            block_diagnostics=tuple(block_diagnostics),
            intercept_model=intercept.receipt,
            candidate_model=candidate_only.receipt,
            baseline_model=baseline.receipt,
            augmented_model=augmented.receipt,
        )

    def calibrate_stage1_null(
        self,
        candidates: Sequence[Any],
        outcomes: Iterable[PlacementOutcome],
        *,
        permutations: int = 100,
        percentile: float = 0.95,
        seed: int = 0,
    ) -> PermutationNullCalibration:
        """Freeze a stage-1 threshold from within-system label permutations.

        This is an offline Phase 0 operation. It deliberately preserves system
        sizes, axes, chemistry blocks, folds, pair sampling, and feature fields
        while breaking the relationship between placements and endpoint energy.
        """
        if not candidates:
            raise ValueError("permutation calibration requires candidate controls")
        compiler_hashes = {str(candidate.compiler_hash) for candidate in candidates}
        universe_hashes = {str(candidate.universe_hash) for candidate in candidates}
        if len(compiler_hashes) != 1 or len(universe_hashes) != 1:
            raise ValueError("permutation controls must share compiler and universe hashes")
        if permutations < 1:
            raise ValueError("permutations must be positive")
        if not 0.0 < percentile < 1.0:
            raise ValueError("percentile must be between zero and one")
        outcome_tuple = tuple(outcomes)
        by_system: dict[str, list[PlacementOutcome]] = {}
        for outcome in outcome_tuple:
            by_system.setdefault(outcome.system_alias, []).append(outcome)
        statistics: list[float] = []
        for permutation_index in range(permutations):
            permuted: list[PlacementOutcome] = []
            for system_alias in sorted(by_system):
                group = sorted(
                    by_system[system_alias], key=lambda item: item.candidate_alias
                )
                energies = [item.energy for item in group]
                rng = random.Random(
                    int.from_bytes(
                        _stable_digest(seed, permutation_index, system_alias)[:8],
                        "big",
                    )
                )
                rng.shuffle(energies)
                permuted.extend(
                    replace(item, energy=energy)
                    for item, energy in zip(group, energies, strict=True)
                )
            statistics.extend(self._stage1_statistics(candidates, permuted))
        return PermutationNullCalibration.create(
            percentile=percentile,
            permutations=permutations,
            seed=seed,
            statistics=statistics,
            scorer_hash=self.config.scorer_hash,
            compiler_hash=compiler_hashes.pop(),
            universe_hash=universe_hashes.pop(),
        )

    def with_calibration(
        self, calibration: PermutationNullCalibration
    ) -> PairwiseScorer:
        if calibration.scorer_hash != self.config.scorer_hash:
            raise ValueError("permutation calibration scorer hash mismatch")
        return PairwiseScorer(
            replace(self.config, stage1_threshold=calibration.threshold)
        )

    def _stage1_statistic(
        self,
        candidate: Any,
        outcomes: Iterable[PlacementOutcome],
    ) -> float:
        return self._stage1_statistics((candidate,), tuple(outcomes))[0]

    def score_stage1(
        self,
        candidates: Sequence[Any],
        outcomes: Iterable[PlacementOutcome],
    ) -> tuple[Stage1Score, ...]:
        """Score candidate-only stage 1 in one offline batch."""
        if not candidates:
            return ()
        candidate_tuple = tuple(candidates)
        outcome_tuple = tuple(outcomes)
        topology = _build_pair_topology(
            outcome_tuple,
            max_pairs_per_system=self.config.max_pairs_per_system,
            pair_seed=self.config.pair_seed,
        )
        if not topology.labels.size:
            raise ValueError("stage-1 scoring requires at least one non-tied pair")
        first = candidate_tuple[0]
        for candidate in candidate_tuple[1:]:
            self._validate_artifacts(candidate, (first,))
        intercept_data = _PairwiseArrays(topology, _feature_matrix(topology, ()))
        intercept = _cross_validate(intercept_data, (), (), self.config)
        results: list[Stage1Score] = []
        for candidate in candidate_tuple:
            candidate_only = _cross_validate(
                _PairwiseArrays(
                    topology, _artifact_feature_column(topology, candidate)[:, None]
                ),
                (0,),
                (str(candidate.rule_hash),),
                self.config,
                initial=intercept,
            )
            complexity = int(getattr(candidate, "complexity", 0))
            charge = self.config.complexity_lambda * complexity
            raw_gain = _mean_delta(
                intercept.system_losses, candidate_only.system_losses
            )
            gain = raw_gain - charge
            converged = all(
                fold.converged
                for model in (intercept.receipt, candidate_only.receipt)
                for fold in model.folds
            )
            results.append(
                Stage1Score(
                    complexity=complexity,
                    complexity_charge=charge,
                    stage1_raw_gain=raw_gain,
                    stage1_gain=gain,
                    stage1_threshold=self.config.stage1_threshold,
                    converged=converged,
                    signal_pass=converged and gain > self.config.stage1_threshold,
                )
            )
        return tuple(results)

    def _stage1_statistics(
        self,
        candidates: Sequence[Any],
        outcomes: Sequence[PlacementOutcome],
    ) -> tuple[float, ...]:
        return tuple(
            result.stage1_gain for result in self.score_stage1(candidates, outcomes)
        )

    @staticmethod
    def _validate_artifacts(candidate: Any, archive: Sequence[Any]) -> None:
        compatibility_fields = ("dataset_hash", "compiler_hash", "universe_hash")
        for artifact in archive:
            for field_name in compatibility_fields:
                if getattr(artifact, field_name) != getattr(candidate, field_name):
                    raise ValueError(
                        f"archive artifact {artifact.rule_hash!r} has incompatible {field_name}"
                    )
            if _descriptor_hash(artifact) != _descriptor_hash(candidate):
                raise ValueError(
                    f"archive artifact {artifact.rule_hash!r} has incompatible descriptor_hash"
                )


__all__ = [
    "AxisDiagnostic",
    "BlockDiagnostic",
    "FoldFitReceipt",
    "ModelReceipt",
    "PairwiseExample",
    "PairwiseScorer",
    "PermutationNullCalibration",
    "PlacementOutcome",
    "ScorerConfig",
    "ScoringReceipt",
    "Stage1Score",
    "build_pairwise_examples",
]
