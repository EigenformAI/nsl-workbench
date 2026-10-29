"""Materials adsorption feature-hypothesis task contract.

The task deliberately keeps the private adsorption dataset out of construction,
prompts, and capability results. Dataset and scorer integration happens only at
the execution boundary; unit-level workflow inspection therefore does not
require the large OC20-Dense artifact to be present.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docker.models.containers import Container

from src.container import container_to_service
from src.task.base import TaskEnvironmentError, TaskSpec
from src.task.types import (
    BudgetConstraints,
    Capability,
    CapabilityExecutionContext,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
    EpisodeConstraints,
    PopulationOutcome,
    PopulationResult,
    SuccessConstraints,
    TaskCommitPreparation,
    TaskPromptSpec,
    TaskReward,
    TrustedEpisodeIdentity,
    Variation,
    Workflow,
    WorkflowStep,
)
from tasks.materials_adsorption.training import MaterialsRewriteTransform
from tasks.materials_adsorption.compiler import (
    ConsequenceArtifact,
    RuleCompiler,
    RuleEvaluationError,
    RuleValidationError,
    rule_expression_contract,
)
from tasks.materials_adsorption.dataset import (
    EndpointStatus,
    LoadedDataset,
    load_dataset,
)
from tasks.materials_adsorption.compute import (
    COMPUTE_SERVICE_NAME,
    build_public_export,
    execute_public_python,
    public_export_mount,
)
from tasks.materials_adsorption.scoring import (
    PairwiseScorer,
    PermutationNullCalibration,
    PlacementOutcome,
    ScorerConfig,
)
from tasks.materials_adsorption.probe_calibration import (
    AttainabilityReference,
    PERCENTILE_BIN_LABELS,
    SHARD_NULL_PERMUTATIONS,
    SHARD_NULL_PERMUTATION_SEED,
    SHARD_PARTITION_VERSION,
    ShardNullCalibration,
)


_CAPABILITY_STEPS = {
    "inspect_evidence": frozenset({"explore", "hypothesise", "code"}),
    "run_python": frozenset({"explore", "hypothesise", "code"}),
    "record_evidence_plan": frozenset({"explore"}),
    "record_hypothesis": frozenset({"hypothesise"}),
    "compile_feature": frozenset({"code"}),
    "probe_feature": frozenset({"code"}),
    "finalize_feature": frozenset({"code"}),
    "abandon_feature": frozenset({"code"}),
    "submit_hypothesis": frozenset({"verify"}),
    "get_accepted_hypothesis": frozenset({"rewrite"}),
    "submit_rewrite": frozenset({"rewrite"}),
}

_DEFAULT_COMPOSE_DIR = "docker/materials-adsorption-compose"

_PUBLIC_CONTEXT_KEYS = (
    "evidence",
    "evidence_alias",
    "parent_hypotheses",
    "residual_summary",
    "descriptor_registry",
    "rule_contract",
    "candidate_universe_hash",
)

_INSPECT_MAX_FILTERS = 3
_INSPECT_MAX_ROWS = 10

PROBE_CALL_LIMIT = 3
PROBE_LEAK_BUDGET_BITS = 16.0
EVIDENCE_GROUPS_PER_SHARD = 3

if PROBE_CALL_LIMIT * math.log2(len(PERCENTILE_BIN_LABELS)) > PROBE_LEAK_BUDGET_BITS:
    raise RuntimeError("probe configuration exceeds the per-episode leak budget")


def _ordinal_percentile(value: float) -> str:
    rounded = round(value)
    suffix = (
        "th"
        if 10 < rounded % 100 < 14
        else {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    )
    return f"{rounded}{suffix} percentile"


@dataclass(frozen=True)
class MaterialsAdsorptionState:
    """Label-free state visible to generic reward finalization."""

    dataset_hash: str | None = None
    candidate_universe_hash: str | None = None


class MaterialsAdsorptionTask(TaskSpec[MaterialsAdsorptionState]):
    """Discover executable rules for ranking adsorption configurations."""

    name = "materials-adsorption"
    description = "Discover executable adsorption-configuration ranking rules"
    metric_name = "training_success"
    metric_unit = "boolean"
    higher_is_better = True
    agent_service_name = "agent-service"

    def __init__(self, task_config: dict[str, Any]) -> None:
        # Store handles only. In particular, do not open the private dataset or
        # archive here: task loading and workflow tests run without those assets.
        self._task_config = dict(task_config)
        self._dataset_path = Path(
            task_config.get("dataset_path", "data/materials-adsorption/dataset.json")
        )
        self._archive_path = Path(
            task_config.get("archive_path", "data/materials-adsorption/archive.sqlite")
        )
        self._blob_dir = Path(
            task_config.get("blob_dir", "data/materials-adsorption/blobs")
        )
        self._docker_compose_dir = str(
            task_config.get("docker_compose_dir", _DEFAULT_COMPOSE_DIR)
        )

        scorer_config = task_config.get("scorer", {})
        if not isinstance(scorer_config, Mapping):
            raise ValueError("scorer task config must be a mapping")
        self._scorer = PairwiseScorer(ScorerConfig(**dict(scorer_config)))
        self._permutation_calibration: PermutationNullCalibration | None = None
        null_path = task_config.get("permutation_null_path")
        if null_path:
            try:
                calibration_payload = json.loads(
                    Path(str(null_path)).read_text(encoding="utf-8")
                )
                calibration = PermutationNullCalibration.from_dict(calibration_payload)
                self._scorer = self._scorer.with_calibration(calibration)
                self._permutation_calibration = calibration
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError("invalid frozen permutation-null calibration") from exc
        elif not bool(task_config.get("allow_uncalibrated_threshold", False)):
            raise ValueError(
                "a frozen permutation-null calibration is required; "
                "set allow_uncalibrated_threshold only for plumbing smoke tests"
            )
        allow_uncalibrated = bool(
            task_config.get("allow_uncalibrated_threshold", False)
        )
        self._shard_null_calibration: ShardNullCalibration | None = None
        shard_null_path = task_config.get("shard_null_path")
        if shard_null_path:
            try:
                payload = json.loads(
                    Path(str(shard_null_path)).read_text(encoding="utf-8")
                )
                self._shard_null_calibration = ShardNullCalibration.from_dict(payload)
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError("invalid frozen shard-null calibration") from exc
        elif not allow_uncalibrated:
            raise ValueError("a frozen shard-null calibration is required")

        self._attainability_reference: AttainabilityReference | None = None
        attainability_path = task_config.get("attainability_path")
        if attainability_path:
            try:
                payload = json.loads(
                    Path(str(attainability_path)).read_text(encoding="utf-8")
                )
                self._attainability_reference = AttainabilityReference.from_dict(
                    payload
                )
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError("invalid frozen attainability reference") from exc
        elif not allow_uncalibrated:
            raise ValueError("a frozen attainability reference is required")
        max_scorers = int(task_config.get("max_concurrent_scorers", 2))
        if max_scorers < 1:
            raise ValueError("max_concurrent_scorers must be positive")
        # A semaphore is run-level resource coordination, not episode state.
        import threading

        self._scorer_slots = threading.BoundedSemaphore(max_scorers)
        self._dataset_lock = threading.Lock()
        self._loaded_dataset: LoadedDataset | None = None
        self._minimum_support = int(task_config.get("minimum_support", 2))

    @property
    def docker_compose_dir(self) -> str:
        return self._docker_compose_dir

    def list_variations(self) -> list[Variation]:
        return [
            Variation(
                name="oc20-dense-rule-discovery",
                description=(
                    "Propose a pre-DFT rule over anonymized adsorption placements"
                ),
            )
        ]

    def populate(
        self,
        containers: list[Container],
        variation: Variation,
    ) -> PopulationOutcome:
        """Attach a private dataset handle without publishing dataset contents."""

        if not self._dataset_path.is_file():
            raise TaskEnvironmentError(
                "The configured materials dataset is unavailable to the task host"
            )

        loaded = self._load_dataset()
        dataset_hash = loaded.universe.manifest_hash
        store = self._archive_store()
        source_aliases = self._evidence_source_aliases(loaded)
        store.register_evidence_sources(source_aliases.keys())
        evidence_alias = store.assign_evidence_source(source_aliases.keys())
        source_groups = source_aliases[evidence_alias]
        candidates_by_alias = {
            candidate.alias: candidate for candidate in loaded.universe.candidates
        }
        evidence_candidates = []
        for record in loaded.records:
            if (
                record.status is not EndpointStatus.VALID
                or record.source_group not in source_groups
            ):
                continue
            candidate_alias, _system_alias = loaded.source_aliases[
                (record.source_system_id, record.source_placement_id)
            ]
            evidence_candidates.append(
                candidates_by_alias[candidate_alias].public_payload()
            )

        rule_contract = self._rule_contract()
        for container in containers:
            try:
                is_compute = container_to_service(container) == COMPUTE_SERVICE_NAME
            except (AttributeError, ValueError):
                is_compute = False
            if not is_compute:
                continue
            export_path = public_export_mount(container)
            if export_path is None:
                raise TaskEnvironmentError(
                    "materials compute sidecar has no /data public-export mount"
                )
            build_public_export(
                loaded,
                evidence_candidates,
                evidence_alias,
                rule_contract,
                export_path,
            )

        admitted_records = store.admitted_records()
        for record in admitted_records:
            rule_hash = record.get("canonical_rule_hash")
            if isinstance(rule_hash, str) and rule_hash:
                store.register_parent(rule_hash, quality=1.0)
        parent_hypotheses: list[dict[str, Any]] = []
        if len(admitted_records) >= 2:
            parent_ids = set(store.assign_parent_pair())
            parent_hypotheses = [
                record
                for record in admitted_records
                if record.get("canonical_rule_hash") in parent_ids
            ]
        results = [
            PopulationResult(
                container_id=str(container.id),
                variation_name=variation.name,
                description="Prepared label-free materials hypothesis workspace",
                success=True,
            )
            for container in containers
        ]
        return PopulationOutcome(
            results=results,
            episode_context={
                "dataset_hash": dataset_hash,
                "candidate_universe_hash": loaded.universe.universe_hash,
                "descriptor_registry": loaded.registry.to_dict(),
                "evidence_alias": evidence_alias,
                "evidence": evidence_candidates,
                "parent_hypotheses": parent_hypotheses,
                "rule_contract": rule_contract,
            },
            private_context={"dataset_path": str(self._dataset_path)},
        )

    @staticmethod
    def _evidence_source_aliases(
        loaded: LoadedDataset,
    ) -> dict[str, tuple[str, ...]]:
        counts: dict[str, int] = {}
        for record in loaded.records:
            if record.status is EndpointStatus.VALID:
                counts[record.source_group] = counts.get(record.source_group, 0) + 1

        def partition_digest(domain: str, source_group: str) -> bytes:
            return hmac.new(
                loaded.alias_salt.encode("utf-8"),
                f"materials-evidence-{domain}\0{source_group}".encode("utf-8"),
                hashlib.sha256,
            ).digest()

        ordered = sorted(
            counts,
            key=lambda group: (partition_digest("partition-v2", group), group),
        )
        if not ordered:
            raise TaskEnvironmentError("materials dataset has no valid evidence groups")

        shard_count = max(
            1,
            math.ceil(len(ordered) / EVIDENCE_GROUPS_PER_SHARD),
        )
        shards: list[list[str]] = [[] for _ in range(shard_count)]
        odd_groups = [group for group in ordered if counts[group] in {2, 6}]
        if len(odd_groups) != 2 or sorted(counts[group] for group in odd_groups) != [
            2,
            6,
        ]:
            odd_groups = []
        for shard, group in zip(shards, odd_groups, strict=False):
            shard.append(group)

        remaining = [group for group in ordered if group not in odd_groups]
        for group in remaining:
            target = next(
                shard for shard in shards if len(shard) < EVIDENCE_GROUPS_PER_SHARD
            )
            target.append(group)

        duplicate_order = sorted(
            remaining or ordered,
            key=lambda group: (partition_digest("duplicate-v2", group), group),
        )
        for shard in shards:
            while len(shard) < EVIDENCE_GROUPS_PER_SHARD:
                duplicate = next(
                    (group for group in duplicate_order if group not in shard),
                    shard[0],
                )
                shard.append(duplicate)

        aliases: dict[str, tuple[str, ...]] = {}
        for shard in shards:
            groups = tuple(shard)
            digest = hmac.new(
                loaded.alias_salt.encode("utf-8"),
                ("materials-evidence-shard-v2\0" + "\0".join(groups)).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            aliases[f"evidence_{digest[:20]}"] = groups
        return aliases

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_dataset(self) -> LoadedDataset:
        with self._dataset_lock:
            if self._loaded_dataset is not None:
                return self._loaded_dataset
            try:
                loaded = load_dataset(self._dataset_path)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise TaskEnvironmentError(
                    "The configured materials dataset failed validation"
                ) from exc
            calibration = self._permutation_calibration
            if calibration is not None:
                compiler_hash = RuleCompiler(loaded.registry).compiler_hash
                if calibration.compiler_hash != compiler_hash:
                    raise TaskEnvironmentError(
                        "frozen permutation-null compiler hash mismatch"
                    )
                if calibration.universe_hash != loaded.universe.universe_hash:
                    raise TaskEnvironmentError(
                        "frozen permutation-null universe hash mismatch"
                    )
            self._validate_probe_references(loaded)
            self._loaded_dataset = loaded
            return loaded

    def _validate_probe_references(self, loaded: LoadedDataset) -> None:
        calibration = self._shard_null_calibration
        if calibration is None:
            return
        compiler_hash = RuleCompiler(loaded.registry).compiler_hash
        expected = (
            (
                calibration.shard_partition_version,
                SHARD_PARTITION_VERSION,
                "partition version",
            ),
            (calibration.permutations, SHARD_NULL_PERMUTATIONS, "permutation count"),
            (calibration.seed, SHARD_NULL_PERMUTATION_SEED, "permutation seed"),
            (calibration.dataset_hash, loaded.universe.manifest_hash, "dataset hash"),
            (calibration.universe_hash, loaded.universe.universe_hash, "universe hash"),
            (
                calibration.descriptor_registry_hash,
                loaded.registry.registry_hash,
                "descriptor registry hash",
            ),
            (calibration.compiler_hash, compiler_hash, "compiler hash"),
            (calibration.scorer_hash, self._scorer.config.scorer_hash, "scorer hash"),
            (calibration.fold_seed, self._scorer.config.fold_seed, "fold seed"),
            (
                calibration.complexity_lambda,
                self._scorer.config.complexity_lambda,
                "complexity lambda",
            ),
            (
                calibration.stage1_threshold,
                self._scorer.config.stage1_threshold,
                "stage-1 threshold",
            ),
        )
        for actual, wanted, label in expected:
            if actual != wanted:
                raise TaskEnvironmentError(f"frozen shard-null {label} mismatch")
        aliases = set(self._evidence_source_aliases(loaded))
        if set(calibration.shard_universe_gains) != aliases:
            raise TaskEnvironmentError("frozen shard-null shard aliases mismatch")

        attainability = self._attainability_reference
        if attainability is None:
            raise TaskEnvironmentError("frozen attainability reference is unavailable")
        attainability_expected = (
            (attainability.dataset_hash, loaded.universe.manifest_hash, "dataset hash"),
            (
                attainability.universe_hash,
                loaded.universe.universe_hash,
                "universe hash",
            ),
            (
                attainability.descriptor_registry_hash,
                loaded.registry.registry_hash,
                "descriptor registry hash",
            ),
            (attainability.compiler_hash, compiler_hash, "compiler hash"),
            (attainability.scorer_hash, self._scorer.config.scorer_hash, "scorer hash"),
            (
                attainability.stage1_threshold,
                self._scorer.config.stage1_threshold,
                "stage-1 threshold",
            ),
            (
                attainability.complexity_lambda,
                self._scorer.config.complexity_lambda,
                "complexity lambda",
            ),
        )
        for actual, wanted, label in attainability_expected:
            if actual != wanted:
                raise TaskEnvironmentError(f"frozen attainability {label} mismatch")

    def _rule_contract(self) -> str:
        config = self._scorer.config
        calibration = self._shard_null_calibration
        bar_percentile = (
            _ordinal_percentile(calibration.bar_percentile)
            if calibration is not None
            else "disclosed bar percentile"
        )
        return (
            "Use only declared pre-DFT descriptors in a restricted Python "
            "expression. IDs, lookup tables, files, network access, and "
            "endpoint-derived values are forbidden. The complexity charge is "
            f"lambda={config.complexity_lambda:g} times feature-expression AST "
            "nodes; applicability and uncertainty expressions are not charged. "
            "Names, constants, and operators count as 1 node; function calls and "
            "conditional expressions count as 2 nodes. At least "
            f"{self._minimum_support} "
            "applicable candidates are required. Prefer the simplest expression that "
            "captures the effect; each additional term must earn more than its charge. "
            "You may probe up to 3 compiled features against your evidence shard for "
            "the percentile of the feature's penalized shard gain against a frozen "
            "chance reference of real candidate features evaluated on shards. The "
            f"admission bar is approximately the {bar_percentile} of that reference. "
            "Shard estimates are noisy: seek features well above the bar percentile, "
            "then expect admission to be scored on held-out data across the full "
            "candidate universe. Probe on the shard, then generalize to the universe. "
            "After passing the signal bar, Stage 2 compares against the current archive "
            "and requires the cross-fold lower bound of gain to be positive, "
            "demonstrating stability, not just mean effect."
        )

    def prompt_spec(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> TaskPromptSpec:
        del variation
        public_context = self._public_context(episode_context)
        return TaskPromptSpec(
            system_instruction=(
                "You are a materials scientist constructing a compact, executable "
                "theory of adsorption placement preference. Use only the evidence "
                "and pre-DFT descriptors supplied to you. Never infer, request, or "
                "encode private energies, endpoint labels, folds, source IDs, file "
                "paths, or verifier internals."
            ),
            environment_context=(
                json.dumps(public_context, sort_keys=True, ensure_ascii=True)
                if public_context
                else ""
            ),
            capabilities=self._capabilities(),
        )

    @staticmethod
    def _public_context(episode_context: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: episode_context[key]
            for key in _PUBLIC_CONTEXT_KEYS
            if key in episode_context
        }

    @classmethod
    def _public_context_text(cls, episode_context: Mapping[str, Any]) -> str:
        return json.dumps(
            cls._public_context(episode_context),
            sort_keys=True,
            ensure_ascii=True,
        )

    @staticmethod
    def _capabilities() -> list[Capability]:
        object_schema = {
            "type": "object",
            "additionalProperties": False,
        }
        return [
            Capability(
                name="inspect_evidence",
                description=(
                    "Query the bounded public descriptor registry or evidence rows "
                    "using conjunctive descriptor filters and deterministic sorting."
                ),
                schema={
                    **object_schema,
                    "properties": {
                        "view": {
                            "type": "string",
                            "enum": ["evidence", "descriptor_registry"],
                        },
                        "filters": {
                            "type": "array",
                            "maxItems": _INSPECT_MAX_FILTERS,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "descriptor": {"type": "string"},
                                    "op": {
                                        "type": "string",
                                        "enum": ["eq", "ne", "lt", "lte", "gt", "gte"],
                                    },
                                    "value": {"type": ["number", "boolean"]},
                                },
                                "required": ["descriptor", "op", "value"],
                            },
                        },
                        "sort_by": {"type": "string"},
                        "sort_direction": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _INSPECT_MAX_ROWS,
                            "default": _INSPECT_MAX_ROWS,
                        },
                    },
                    "required": ["view"],
                },
            ),
            Capability(
                name="run_python",
                description=(
                    "Run Python with numpy/pandas in a network-isolated sidecar over "
                    "the read-only public files in /data."
                ),
                runs_code=True,
                schema={
                    **object_schema,
                    "properties": {"snippet": {"type": "string", "maxLength": 8192}},
                    "required": ["snippet"],
                },
            ),
            Capability(
                name="record_evidence_plan",
                description="Record a bounded, label-free evidence exploration plan.",
                schema={
                    **object_schema,
                    "properties": {
                        "summary": {"type": "string"},
                        "evidence_aliases": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["summary", "evidence_aliases"],
                },
            ),
            Capability(
                name="record_hypothesis",
                description="Record the mechanism, applicability, and falsifying test.",
                schema={
                    **object_schema,
                    "properties": {
                        "statement": {"type": "string"},
                        "rationale": {"type": "string"},
                        "applicability": {"type": "string"},
                        "discriminating_test": {"type": "string"},
                        "parent_hypothesis_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "statement",
                        "rationale",
                        "applicability",
                        "discriminating_test",
                    ],
                },
            ),
            Capability(
                name="compile_feature",
                description=(
                    "Compile a restricted descriptor expression without evaluating "
                    "private labels."
                ),
                runs_code=True,
                schema={
                    **object_schema,
                    "properties": {
                        "expression": {"type": "string"},
                        "applicable": {"type": "string"},
                        "uncertainty": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            ),
            Capability(
                name="probe_feature",
                description=(
                    "Probe a compiled feature selected by rule_hash against the "
                    "episode evidence shard and return its frozen-null percentile "
                    "bin. Omit rule_hash to use the most recent successful compile."
                ),
                schema={
                    **object_schema,
                    "properties": {"rule_hash": {"type": "string"}},
                },
            ),
            Capability(
                name="finalize_feature",
                description=(
                    "Finish the code step with a compiled feature selected by "
                    "rule_hash. Omit rule_hash to use the most recent successful "
                    "compile. This does not evaluate private labels."
                ),
                schema={
                    **object_schema,
                    "properties": {"rule_hash": {"type": "string"}},
                },
            ),
            Capability(
                name="abandon_feature",
                description=(
                    "End the code step without a feature after all compile attempts "
                    "have failed."
                ),
                schema={**object_schema, "properties": {}},
            ),
            Capability(
                name="submit_hypothesis",
                description=(
                    "Submit the stored evidence plan, hypothesis, and compiled feature "
                    "once for trusted private verification."
                ),
                publishes_metric=True,
                schema={**object_schema, "properties": {}},
            ),
            Capability(
                name="get_accepted_hypothesis",
                description=(
                    "Read the accepted public hypothesis record for rewriting. "
                    "Verifier scores, thresholds, hashes, and receipts are omitted."
                ),
                schema={**object_schema, "properties": {}},
            ),
            Capability(
                name="submit_rewrite",
                description=(
                    "Submit a concise public training rewrite containing the "
                    "mechanism, rule, scope, and discriminating test."
                ),
                schema={
                    **object_schema,
                    "properties": {
                        "target": {"type": "string"},
                    },
                    "required": ["target"],
                },
            ),
        ]

    def workflow(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> Workflow:
        del variation
        public_context = self._public_context_text(episode_context)
        return Workflow(
            steps=(
                WorkflowStep(
                    name="explore",
                    prompt=(
                        "Inspect actual evidence rows before hypothesising; the "
                        "descriptor registry is already embedded in this prompt. You "
                        "may call inspect_evidence or use run_python over the read-only "
                        "public files in /data; both are optional. End "
                        "the step by calling record_evidence_plan with the evidence "
                        "aliases that ground your plan.\n\nPublic task context:\n"
                        f"{public_context}"
                    ),
                    capabilities=(
                        "inspect_evidence",
                        "run_python",
                        "record_evidence_plan",
                    ),
                    inherit_all_capabilities=False,
                    next_steps=("hypothesise",),
                    is_entry=True,
                    context_mode="inherit",
                    terminator_capabilities=("record_evidence_plan",),
                ),
                WorkflowStep(
                    name="hypothesise",
                    prompt=(
                        "Develop one evidence-grounded mechanism in terms of the "
                        "declared descriptors, cite the evidence aliases supporting it, "
                        "and state its scientific scope and discriminating test. You "
                        "may call inspect_evidence or run_python for additional public "
                        "analysis. End by calling "
                        "record_hypothesis."
                    ),
                    capabilities=(
                        "inspect_evidence",
                        "run_python",
                        "record_hypothesis",
                    ),
                    inherit_all_capabilities=False,
                    next_steps=("code",),
                    context_mode="inherit",
                    terminator_capabilities=("record_hypothesis",),
                ),
                WorkflowStep(
                    name="code",
                    prompt=(
                        "Author and repair one restricted pre-DFT feature expression. "
                        "You have up to 8 compile attempts. Use compile_feature until "
                        "it returns accepted=true; compile_feature returns a rule_hash. "
                        "Probe up to 3 candidate features by passing their rule_hash "
                        "to probe_feature, then finalize the strongest by its rule_hash. "
                        "Omitting rule_hash uses the most recent successful compile. "
                        "Remember: using all probe_feature calls does not "
                        "block finalize_feature, submit_hypothesis, or submit_rewrite; "
                        "continue through those steps after the probe budget is "
                        "exhausted. If all attempts fail, call abandon_feature instead "
                        "of exceeding the compile budget. You may call inspect_evidence "
                        "for public evidence or run_python to compute support counts, "
                        "variances, and descriptor relationships from "
                        "/data/universe.csv; both are optional. Prefer the simplest "
                        "expression that captures the effect.\n\n"
                        f"{rule_expression_contract()}\n\n{self._rule_contract()}\n\n"
                        "Public task context:\n"
                        f"{public_context}"
                    ),
                    capabilities=(
                        "inspect_evidence",
                        "run_python",
                        "compile_feature",
                        "probe_feature",
                        "finalize_feature",
                        "abandon_feature",
                    ),
                    inherit_all_capabilities=False,
                    next_steps=("verify",),
                    context_mode="isolated",
                    terminator_capabilities=("finalize_feature", "abandon_feature"),
                ),
                WorkflowStep(
                    name="verify",
                    prompt=(
                        "Call submit_hypothesis with an empty object to submit the "
                        "server-stored evidence plan, hypothesis, and compiled feature "
                        "exactly once for private scoring."
                    ),
                    capabilities=("submit_hypothesis",),
                    inherit_all_capabilities=False,
                    next_steps=("rewrite",),
                    context_mode="isolated",
                    terminator_capabilities=("submit_hypothesis",),
                ),
                WorkflowStep(
                    name="rewrite",
                    prompt=(
                        "Call get_accepted_hypothesis to read only the accepted public "
                        "scientific record. Then call submit_rewrite with a concise "
                        "plain-language rewrite covering its mechanism, expression, "
                        "scope, uncertainty, and discriminating test. Do not infer or "
                        "request verifier information."
                    ),
                    capabilities=("get_accepted_hypothesis", "submit_rewrite"),
                    inherit_all_capabilities=False,
                    context_mode="isolated",
                    terminator_capabilities=("submit_rewrite",),
                ),
            )
        )

    def episode_constraints(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> EpisodeConstraints:
        del variation, episode_context
        return EpisodeConstraints(
            budgets=BudgetConstraints(
                max_task_tool_calls=44,
                max_task_tool_calls_by_name={
                    "record_evidence_plan": 1,
                    "record_hypothesis": 1,
                    "compile_feature": 8,
                    "probe_feature": PROBE_CALL_LIMIT,
                    "run_python": 10,
                    "finalize_feature": 1,
                    "abandon_feature": 1,
                    "submit_hypothesis": 1,
                    "get_accepted_hypothesis": 1,
                    "submit_rewrite": 1,
                },
                max_llm_turns=52,
            ),
            success=SuccessConstraints(
                min_task_tool_calls_for_success=1,
                terminal_capability_for_success="submit_rewrite",
            ),
        )

    def validate_submission_context(
        self,
        episode_id: str,
        workflow_step: str | None,
        episode_context: dict[str, Any],
    ) -> str | None:
        if workflow_step != "verify":
            return "wrong_workflow_step"
        if episode_context.get("_materials_submission_episode_id") == episode_id:
            return "already_submitted"
        return None

    @staticmethod
    def mark_submission(episode_id: str, episode_context: dict[str, Any]) -> None:
        episode_context["_materials_submission_episode_id"] = episode_id

    def _claim_submission(self, ctx: CapabilityExecutionContext) -> str | None:
        error = self.validate_submission_context(
            ctx.episode_id, ctx.workflow_step, ctx.episode_context
        )
        if error is None:
            self.mark_submission(ctx.episode_id, ctx.episode_context)
        return error

    def execute_capability(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: Variation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        del variation
        expected_steps = _CAPABILITY_STEPS.get(invocation.name)
        if expected_steps is None:
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error="unknown_capability",
            )
        if ctx.workflow_step not in expected_steps:
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error="wrong_workflow_step",
            )
        if invocation.name == "inspect_evidence":
            return self._inspect_evidence(invocation, ctx)
        if invocation.name == "run_python":
            compute = []
            for container in containers:
                try:
                    if container_to_service(container) == COMPUTE_SERVICE_NAME:
                        compute.append(container)
                except (AttributeError, ValueError):
                    continue
            if len(compute) != 1:
                raise TaskEnvironmentError(
                    "expected exactly one isolated materials compute sidecar"
                )
            timeout_s = float(self._task_config.get("tool_call_timeout_s", 120))
            return execute_public_python(
                compute[0], invocation.input.get("snippet"), timeout_s=timeout_s
            )
        if invocation.name == "record_evidence_plan":
            ctx.episode_context["_materials_evidence_plan"] = {
                "summary": invocation.input.get("summary"),
                "evidence_aliases": list(
                    invocation.input.get("evidence_aliases") or []
                ),
            }
        elif invocation.name == "record_hypothesis":
            ctx.episode_context["_materials_hypothesis"] = dict(invocation.input)
        elif invocation.name == "compile_feature":
            return self._compile_feature(invocation, ctx)
        elif invocation.name == "probe_feature":
            return self._probe_feature(invocation, ctx)
        elif invocation.name == "finalize_feature":
            compiled_state, error = self._resolve_compiled_state(
                ctx.episode_context,
                invocation.input,
                missing_error="feature_not_compiled",
            )
            if error is not None:
                return CapabilityResult(
                    name=invocation.name,
                    success=False,
                    error=error,
                )
            if compiled_state is None:
                raise TaskEnvironmentError("compiled feature selection failed")
            artifact = compiled_state.get("artifact")
            if not isinstance(artifact, ConsequenceArtifact):
                raise TaskEnvironmentError(
                    "compiled consequence artifact is unavailable"
                )
            ctx.episode_context["_materials_finalized_rule_hash"] = artifact.rule_hash
            return CapabilityResult(
                name=invocation.name,
                output={"accepted": True, "rule_hash": artifact.rule_hash},
                success=True,
            )
        elif invocation.name == "abandon_feature":
            return CapabilityResult(
                name=invocation.name,
                output={"accepted": True, "compiled": False},
                success=True,
            )
        elif invocation.name == "submit_hypothesis":
            error = self._claim_submission(ctx)
            if error is not None:
                return CapabilityResult(
                    name=invocation.name,
                    success=False,
                    error=error,
                )

            return self._submit_hypothesis(invocation, ctx)
        elif invocation.name == "get_accepted_hypothesis":
            verification = ctx.episode_context.get("_materials_verification")
            record = (
                verification.get("hypothesis_record")
                if isinstance(verification, dict)
                else None
            )
            if not isinstance(record, dict):
                return CapabilityResult(
                    name=invocation.name,
                    success=False,
                    error="hypothesis_not_accepted",
                )
            return CapabilityResult(
                name=invocation.name,
                output=dict(record),
                success=True,
            )
        elif invocation.name == "submit_rewrite":
            if ctx.episode_context.get("_materials_training_success") is not True:
                return CapabilityResult(
                    name=invocation.name,
                    success=False,
                    error="not_training_eligible",
                )
            target = invocation.input.get("target")
            if not isinstance(target, str) or not target.strip():
                return CapabilityResult(
                    name=invocation.name,
                    success=False,
                    error="submit_rewrite requires a non-empty target",
                )
        return CapabilityResult(
            name=invocation.name,
            output={"accepted": True},
            success=True,
        )

    def _inspect_evidence(
        self,
        invocation: CapabilityInvocation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        def reject(message: str) -> CapabilityResult:
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error=message,
            )

        top_k = invocation.input.get("top_k", _INSPECT_MAX_ROWS)
        if type(top_k) is not int or not 1 <= top_k <= _INSPECT_MAX_ROWS:
            return reject("top_k must be an integer between 1 and 10")
        view = invocation.input.get("view")
        registry_payload = ctx.episode_context.get("descriptor_registry")
        descriptors = (
            registry_payload.get("descriptors")
            if isinstance(registry_payload, dict)
            else None
        )
        if not isinstance(descriptors, list):
            return reject("descriptor_registry is unavailable")
        descriptor_by_name = {
            item["name"]: item
            for item in descriptors
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }

        if view == "descriptor_registry":
            if invocation.input.get("filters"):
                return reject("descriptor_registry does not support filters")
            sort_by = invocation.input.get("sort_by", "name")
            if sort_by not in {"name", "value_type", "unit", "family"}:
                return reject("invalid descriptor_registry sort_by")
            direction = invocation.input.get("sort_direction", "asc")
            if direction not in {"asc", "desc"}:
                return reject("sort_direction must be asc or desc")
            rows = [
                {
                    key: item[key]
                    for key in ("name", "value_type", "unit", "family")
                    if key in item
                }
                for item in descriptor_by_name.values()
            ]
            rows.sort(key=lambda row: str(row.get("name", "")))
            rows.sort(
                key=lambda row: str(row.get(sort_by, "")),
                reverse=direction == "desc",
            )
            return CapabilityResult(
                name=invocation.name,
                output={
                    "view": view,
                    "items": rows[:top_k],
                    "returned": min(len(rows), top_k),
                    "matched": len(rows),
                    "truncated": len(rows) > top_k,
                },
                success=True,
            )
        if view != "evidence":
            return reject("view must be evidence or descriptor_registry")

        filters = invocation.input.get("filters", [])
        if not isinstance(filters, list) or len(filters) > _INSPECT_MAX_FILTERS:
            return reject("filters must contain at most 3 entries")
        normalized_filters: list[tuple[str, str, bool | float]] = []
        for item in filters:
            if not isinstance(item, dict):
                return reject("each filter must be an object")
            descriptor = item.get("descriptor")
            op = item.get("op")
            value = item.get("value")
            definition = descriptor_by_name.get(descriptor)
            if definition is None:
                return reject(f"unknown descriptor: {descriptor}")
            if op not in {"eq", "ne", "lt", "lte", "gt", "gte"}:
                return reject(f"unsupported filter operator: {op}")
            value_type = definition.get("value_type")
            if value_type == "boolean":
                if type(value) is not bool or op not in {"eq", "ne"}:
                    return reject(
                        "boolean descriptors support eq/ne boolean filters only"
                    )
            elif value_type == "number":
                if type(value) not in {int, float}:
                    return reject("numeric descriptor filters require a number")
                value = float(value)
            else:
                return reject(f"unsupported descriptor type: {value_type}")
            normalized_filters.append((descriptor, op, value))

        evidence = ctx.episode_context.get("evidence")
        if not isinstance(evidence, list):
            return reject("public evidence is unavailable")
        rows: list[dict[str, Any]] = []
        comparisons = {
            "eq": lambda left, right: left == right,
            "ne": lambda left, right: left != right,
            "lt": lambda left, right: left < right,
            "lte": lambda left, right: left <= right,
            "gt": lambda left, right: left > right,
            "gte": lambda left, right: left >= right,
        }
        for raw_row in evidence:
            if not isinstance(raw_row, dict):
                continue
            raw_descriptors = raw_row.get("descriptors")
            if not isinstance(raw_descriptors, dict):
                continue
            if not all(
                descriptor in raw_descriptors
                and comparisons[op](raw_descriptors[descriptor], value)
                for descriptor, op, value in normalized_filters
            ):
                continue
            rows.append(
                {
                    "alias": raw_row.get("alias"),
                    "system_alias": raw_row.get("system_alias"),
                    "descriptors": {
                        name: raw_descriptors[name]
                        for name in descriptor_by_name
                        if name in raw_descriptors
                    },
                }
            )

        sort_by = invocation.input.get("sort_by")
        if sort_by is not None and sort_by not in descriptor_by_name:
            return reject(f"unknown sort descriptor: {sort_by}")
        direction = invocation.input.get("sort_direction", "asc")
        if direction not in {"asc", "desc"}:
            return reject("sort_direction must be asc or desc")
        rows.sort(key=lambda row: str(row.get("alias", "")))
        if sort_by is not None:
            if any(sort_by not in row["descriptors"] for row in rows):
                return reject(
                    f"sort descriptor is unavailable in one or more rows: {sort_by}"
                )
            rows.sort(
                key=lambda row: row["descriptors"][sort_by],
                reverse=direction == "desc",
            )
        return CapabilityResult(
            name=invocation.name,
            output={
                "view": view,
                "evidence_alias": ctx.episode_context.get("evidence_alias"),
                "items": rows[:top_k],
                "returned": min(len(rows), top_k),
                "matched": len(rows),
                "truncated": len(rows) > top_k,
            },
            success=True,
        )

    def _compile_feature(
        self,
        invocation: CapabilityInvocation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        source = invocation.input.get("expression")
        applicable = invocation.input.get("applicable", "True")
        uncertainty = invocation.input.get("uncertainty", "0.0")
        if not all(
            isinstance(value, str) for value in (source, applicable, uncertainty)
        ):
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error="rule expressions must be strings",
            )
        try:
            loaded = self._load_dataset()
            compiled = RuleCompiler(loaded.registry).compile(
                source,
                applicable=applicable,
                uncertainty=uncertainty,
            )
            artifact = compiled.evaluate(
                loaded.universe.candidates,
                dataset_hash=loaded.universe.manifest_hash,
                universe_hash=loaded.universe.universe_hash,
            )
        except (RuleValidationError, RuleEvaluationError) as exc:
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error=str(exc),
            )
        compiled_states = ctx.episode_context.setdefault("_materials_compiled", {})
        if not isinstance(compiled_states, dict):
            raise TaskEnvironmentError("compiled feature register is invalid")
        compiled_states[artifact.rule_hash] = {
            "source": source,
            "applicable": applicable,
            "uncertainty": uncertainty,
            "artifact": artifact,
        }
        ctx.episode_context["_materials_latest_compiled_rule_hash"] = artifact.rule_hash
        support_count, variance_ok = self._guard_diagnostics(artifact)
        return CapabilityResult(
            name=invocation.name,
            output={
                "accepted": True,
                "rule_hash": artifact.rule_hash,
                "support_count": support_count,
                "support_count_over_universe": support_count,
                "complexity": artifact.complexity,
                "complexity_charge": (
                    self._scorer.config.complexity_lambda * artifact.complexity
                ),
                "variance_ok": variance_ok,
                "source": source,
                "applicable": applicable,
                "uncertainty": uncertainty,
            },
            success=True,
        )

    @staticmethod
    def _guard_diagnostics(artifact: ConsequenceArtifact) -> tuple[int, bool]:
        applicable_values = [
            entry.value
            for entry in artifact.entries
            if entry.applicable and entry.value is not None
        ]
        return len(applicable_values), bool(applicable_values) and (
            max(applicable_values) - min(applicable_values) > 1e-12
        )

    def _probe_feature(
        self,
        invocation: CapabilityInvocation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        compiled_state, error = self._resolve_compiled_state(
            ctx.episode_context,
            invocation.input,
            missing_error="no_compiled_feature",
        )
        if error is not None:
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error=error,
            )
        if compiled_state is None:
            raise TaskEnvironmentError("compiled feature selection failed")
        artifact = compiled_state.get("artifact")
        if not isinstance(artifact, ConsequenceArtifact):
            raise TaskEnvironmentError("compiled consequence artifact is unavailable")

        evidence = ctx.episode_context.get("evidence")
        if not isinstance(evidence, list):
            raise TaskEnvironmentError("episode evidence shard is unavailable")
        evidence_aliases = {
            row.get("alias")
            for row in evidence
            if isinstance(row, dict) and isinstance(row.get("alias"), str)
        }
        loaded = self._load_dataset()
        outcomes = tuple(
            outcome
            for outcome in self._placement_outcomes(loaded)
            if outcome.candidate_alias in evidence_aliases
        )
        try:
            with self._scorer_slots:
                score = self._scorer.score_stage1((artifact,), outcomes)[0]
        except (IndexError, ValueError) as exc:
            raise TaskEnvironmentError(
                "probe scoring failed on the evidence shard"
            ) from exc

        gain = score.stage1_gain
        calibration = self._shard_null_calibration
        if calibration is None:
            raise TaskEnvironmentError("frozen shard-null calibration is unavailable")
        try:
            percentile_bin = calibration.percentile_bin(gain)
        except ValueError as exc:
            raise TaskEnvironmentError("probe percentile calibration failed") from exc

        probed_hashes = ctx.episode_context.setdefault(
            "_materials_probed_artifact_hashes", set()
        )
        if not isinstance(probed_hashes, set):
            raise TaskEnvironmentError("probe audit state is invalid")
        probed_hashes.add(artifact.artifact_hash)
        disclosed_bits = len(probed_hashes) * math.log2(len(PERCENTILE_BIN_LABELS))
        if disclosed_bits > PROBE_LEAK_BUDGET_BITS:
            raise TaskEnvironmentError("probe leak budget exceeded")
        if ctx.recorder is not None:
            ctx.recorder.log_observation(
                category="materials_probe_feature",
                payload={
                    "artifact_hash": artifact.artifact_hash,
                    "percentile_bin": percentile_bin,
                },
            )
        self._record_universe_rank(
            ctx,
            artifact_hash=artifact.artifact_hash,
            scope="probe",
            gain=gain,
        )
        return CapabilityResult(
            name=invocation.name,
            output={"percentile_bin": percentile_bin},
            success=True,
        )

    @staticmethod
    def _resolve_compiled_state(
        episode_context: Mapping[str, Any],
        selector: Mapping[str, Any],
        *,
        missing_error: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        compiled_states = episode_context.get("_materials_compiled")
        if not isinstance(compiled_states, dict) or not compiled_states:
            return None, missing_error
        if "rule_hash" in selector:
            rule_hash = selector["rule_hash"]
            if not isinstance(rule_hash, str):
                return None, "rule_hash must be a string"
            if rule_hash not in compiled_states:
                return None, "unknown_rule_hash"
        else:
            rule_hash = episode_context.get("_materials_latest_compiled_rule_hash")
            if not isinstance(rule_hash, str):
                raise TaskEnvironmentError("latest compiled feature is unavailable")
        compiled_state = compiled_states.get(rule_hash)
        if not isinstance(compiled_state, dict):
            raise TaskEnvironmentError("compiled feature register is invalid")
        artifact = compiled_state.get("artifact")
        if not isinstance(artifact, ConsequenceArtifact):
            raise TaskEnvironmentError("compiled consequence artifact is unavailable")
        if artifact.rule_hash != rule_hash:
            raise TaskEnvironmentError("compiled feature register hash mismatch")
        return compiled_state, None

    def _submit_hypothesis(
        self,
        invocation: CapabilityInvocation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        if ctx.identity is None:
            raise TaskEnvironmentError("trusted episode identity is unavailable")
        finalized_rule_hash = ctx.episode_context.get("_materials_finalized_rule_hash")
        selector = (
            {"rule_hash": finalized_rule_hash}
            if finalized_rule_hash is not None
            else {}
        )
        compiled_state, error = self._resolve_compiled_state(
            ctx.episode_context,
            selector,
            missing_error="feature_not_compiled",
        )
        if error is not None:
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error=error,
            )
        if compiled_state is None:
            raise TaskEnvironmentError("compiled feature selection failed")
        artifact = compiled_state.get("artifact")
        if not isinstance(artifact, ConsequenceArtifact):
            raise TaskEnvironmentError("compiled consequence artifact is unavailable")
        hypothesis = ctx.episode_context.get("_materials_hypothesis")
        if not isinstance(hypothesis, dict):
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error="hypothesis_not_recorded",
            )
        evidence_plan = ctx.episode_context.get("_materials_evidence_plan")
        if not isinstance(evidence_plan, dict):
            return CapabilityResult(
                name=invocation.name,
                success=False,
                error="evidence_plan_not_recorded",
            )
        public_record = self._public_hypothesis_record(
            hypothesis=hypothesis,
            evidence_plan=evidence_plan,
            compiled_state=compiled_state,
        )
        support_count, variance_ok = self._guard_diagnostics(artifact)
        support_ok = support_count >= self._minimum_support
        if not support_ok or not variance_ok:
            ctx.episode_context["_materials_training_success"] = False
            ctx.episode_context["_materials_verification"] = {
                "training_success": False,
                "archive_admission": False,
                "rejection_stage": "guard",
                "canonical_rule_hash": artifact.rule_hash,
                "behavior_hash": artifact.behavior_hash,
                "consequence_hash": None,
                "receipt_hash": None,
                "hypothesis_record": public_record,
            }
            return CapabilityResult(
                name=invocation.name,
                output={
                    "accepted": True,
                    "training_success": False,
                    "archive_admission": False,
                    "signal_band": "nonpositive",
                    "marginal_band": "not_evaluated",
                },
                success=True,
            )

        loaded = self._load_dataset()
        outcomes = self._placement_outcomes(loaded)
        store = self._archive_store()
        with self._scorer_slots:
            score_start_revision, consequence_refs = store.scoring_snapshot()
            archive: list[ConsequenceArtifact] = []
            for _rule_hash, blob_hash in consequence_refs:
                try:
                    payload = json.loads(store.blobs.get(blob_hash))
                    archive.append(ConsequenceArtifact.from_dict(payload))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise TaskEnvironmentError(
                        "an admitted consequence artifact is corrupt"
                    ) from exc
            receipt = self._scorer.score(
                artifact,
                outcomes,
                archive=archive,
                archive_revision=score_start_revision,
            )
        self._record_universe_rank(
            ctx,
            artifact_hash=artifact.artifact_hash,
            scope="admission",
            gain=receipt.stage1_gain,
        )
        consequence_hash = store.blobs.put(
            json.dumps(
                artifact.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        receipt_hash = store.blobs.put(
            json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        ctx.episode_context["_materials_training_success"] = receipt.signal_pass
        ctx.episode_context["_materials_verification"] = {
            "training_success": receipt.signal_pass,
            "archive_admission": receipt.marginal_pass,
            "rejection_stage": None
            if receipt.marginal_pass
            else ("marginal" if receipt.signal_pass else "signal"),
            "canonical_rule_hash": artifact.rule_hash,
            "behavior_hash": artifact.behavior_hash,
            "consequence_hash": consequence_hash,
            "receipt_hash": receipt_hash,
            "score_start_revision": score_start_revision,
            "hypothesis_record": public_record,
        }
        return CapabilityResult(
            name=invocation.name,
            output={
                "accepted": True,
                "training_success": receipt.signal_pass,
                "archive_admission": receipt.marginal_pass,
                "signal_band": "positive" if receipt.signal_pass else "nonpositive",
                "marginal_band": "positive" if receipt.marginal_pass else "nonpositive",
            },
            success=True,
        )

    def _record_universe_rank(
        self,
        ctx: CapabilityExecutionContext,
        *,
        artifact_hash: str,
        scope: str,
        gain: float,
    ) -> None:
        recorder = ctx.recorder
        if recorder is None:
            return
        try:
            if scope == "probe":
                calibration = self._shard_null_calibration
                evidence_alias = ctx.episode_context.get("evidence_alias")
                if calibration is None or not isinstance(evidence_alias, str):
                    raise ValueError("probe universe reference is unavailable")
                percentile = calibration.universe_percentile(evidence_alias, gain)
            elif scope == "admission":
                attainability = self._attainability_reference
                if attainability is None:
                    raise ValueError("admission universe reference is unavailable")
                percentile = attainability.percentile(gain)
            else:
                raise ValueError("invalid materials universe-rank scope")
            recorder.log_observation(
                category="materials_universe_rank",
                payload={
                    "artifact_hash": artifact_hash,
                    "scope": scope,
                    "universe_percentile": percentile,
                },
            )
        except Exception as exc:  # noqa: BLE001 - telemetry cannot alter task results
            try:
                recorder.log_warning(
                    "materials_universe_rank_failed",
                    {
                        "artifact_hash": artifact_hash,
                        "scope": scope,
                        "detail": str(exc),
                    },
                )
            except Exception:  # noqa: BLE001 - recorder failures are also non-fatal
                pass

    @staticmethod
    def _public_hypothesis_record(
        *,
        hypothesis: Mapping[str, Any],
        evidence_plan: Mapping[str, Any],
        compiled_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "hypothesis": hypothesis.get("statement", ""),
            "rationale": hypothesis.get("rationale", ""),
            "applicability": hypothesis.get("applicability", ""),
            "evidence_aliases": list(evidence_plan.get("evidence_aliases") or []),
            "expression": compiled_state.get("source", ""),
            "uncertainty": compiled_state.get("uncertainty", "0.0"),
            "discriminating_test": hypothesis.get("discriminating_test", ""),
            "parent_hypothesis_ids": list(
                hypothesis.get("parent_hypothesis_ids") or []
            ),
        }

    @staticmethod
    def _placement_outcomes(loaded: LoadedDataset) -> tuple[PlacementOutcome, ...]:
        outcomes = []
        for record in loaded.records:
            if record.status is not EndpointStatus.VALID or record.energy is None:
                continue
            candidate_alias, system_alias = loaded.source_aliases[
                (record.source_system_id, record.source_placement_id)
            ]
            outcomes.append(
                PlacementOutcome(
                    candidate_alias=candidate_alias,
                    system_alias=system_alias,
                    energy=float(record.energy),
                    axis=record.axis,
                    chemistry_block=(
                        f"{record.adsorbate_group}:{record.catalyst_group}"
                    ),
                )
            )
        return tuple(outcomes)

    def measure_initial_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> MaterialsAdsorptionState:
        del containers, private_context
        return MaterialsAdsorptionState(
            dataset_hash=episode_context.get("dataset_hash"),
            candidate_universe_hash=episode_context.get("candidate_universe_hash"),
        )

    def compute_reward(
        self,
        initial: MaterialsAdsorptionState,
        final: MaterialsAdsorptionState,
        artifacts: EpisodeArtifacts,
    ) -> TaskReward:
        del initial, final
        verification_results = [
            result
            for result in artifacts.capability_results
            if result.name == "submit_hypothesis" and result.success
        ]
        training_success = any(
            result.output.get("training_success") is True
            for result in verification_results
        )
        requested_admission = any(
            result.output.get("archive_admission") is True
            for result in verification_results
        )
        rewritten = any(
            result.name == "submit_rewrite" and result.success
            for result in artifacts.capability_results
        )
        success = training_success and rewritten
        return TaskReward(
            value=float(success),
            success=success,
            breakdown={
                "training_success": success,
                "archive_admission": success and requested_admission,
            },
        )

    def finalize_episode(
        self,
        containers: list[Container],
        initial: MaterialsAdsorptionState,
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
        finalization_context: Any = None,
    ) -> TaskReward:
        del containers, initial, private_context, finalization_context
        verification = episode_context.get("_materials_verification")
        if not isinstance(verification, dict):
            return TaskReward(
                value=0.0,
                success=False,
                breakdown={
                    "training_success": False,
                    "archive_admission": False,
                    "admitted": False,
                    "rejection_stage": "incomplete",
                },
            )
        rewritten = any(
            result.name == "submit_rewrite" and result.success
            for result in artifacts.capability_results
        )
        training_success = bool(verification["training_success"] and rewritten)
        archive_admission = bool(training_success and verification["archive_admission"])
        return TaskReward(
            value=float(training_success),
            success=training_success,
            breakdown={
                **verification,
                "training_success": training_success,
                "archive_admission": archive_admission,
                "admitted": archive_admission,
            },
        )

    def training_data_transforms(self) -> tuple[MaterialsRewriteTransform, ...]:
        return (MaterialsRewriteTransform(),)

    def episode_commit_enabled(self) -> bool:
        return True

    def _archive_store(self):
        # Import and initialize only at a stateful lifecycle boundary. This
        # preserves dataset-free and archive-free task construction.
        from tasks.materials_adsorption.archive import ArchiveStore

        return ArchiveStore(self._archive_path, self._blob_dir)

    def prepare_episode_commit(
        self,
        identity: TrustedEpisodeIdentity,
        reward: TaskReward,
    ) -> TaskCommitPreparation:
        breakdown = reward.breakdown if isinstance(reward.breakdown, Mapping) else {}
        training_success = bool(breakdown.get("training_success", reward.success))
        requested_admission = bool(
            breakdown.get("archive_admission", breakdown.get("admitted", False))
        )
        raw_rule_hash = breakdown.get("canonical_rule_hash")
        materialized = isinstance(raw_rule_hash, str) and bool(raw_rule_hash)
        rule_hash = raw_rule_hash if materialized else None
        if not materialized:
            rule_hash = (
                "unmaterialized-"
                + hashlib.sha256(
                    (
                        f"{identity.run_id}\0{identity.generation_id}\0"
                        f"{identity.episode_index}\0{identity.episode_id}"
                    ).encode("utf-8")
                ).hexdigest()
            )
            requested_admission = False

        consequence_hash = breakdown.get("consequence_hash")
        behavior_hash = breakdown.get("behavior_hash")
        receipt_hash = breakdown.get("receipt_hash")
        return self._archive_store().prepare_episode_commit(
            identity=identity,
            training_success=training_success,
            requested_admission=requested_admission,
            canonical_rule_hash=rule_hash,
            behavior_hash=(behavior_hash if isinstance(behavior_hash, str) else None),
            consequence_hash=(
                consequence_hash if isinstance(consequence_hash, str) else None
            ),
            receipt_hash=receipt_hash if isinstance(receipt_hash, str) else None,
            outcome={
                "rejection_stage": breakdown.get("rejection_stage"),
                "materialized": materialized,
            },
            archive_record={
                "canonical_rule_hash": rule_hash,
                "behavior_hash": behavior_hash,
                "episode_id": identity.episode_id,
                **(
                    breakdown.get("hypothesis_record")
                    if isinstance(breakdown.get("hypothesis_record"), dict)
                    else {}
                ),
            },
        )

    def apply_episode_commit(self, token: str, intent_digest: str) -> None:
        self._archive_store().apply_episode_commit(token, intent_digest)

    def recover_episode_commits(self, journaled: Mapping[str, str]) -> None:
        self._archive_store().recover_episode_commits(journaled)

    def parallel_checkpoint_safe(self) -> bool:
        return True

    def generation_checkpoint_state(self) -> dict[str, Any]:
        return self._archive_store().generation_checkpoint_state()

    def load_generation_checkpoint_state(
        self,
        state: dict[str, Any] | None,
    ) -> None:
        self._archive_store().load_generation_checkpoint_state(state)
