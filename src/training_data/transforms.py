from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.typing.training import save_durable_json, save_generation_training_data


TARGET_COUNT_BASIS = "training_rows"
RowObservabilityHook = Callable[
    [Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]

REQUIRED_ROW_FIELDS = frozenset(
    {
        "row_id",
        "parent_row_id",
        "prompt",
        "raw_response",
        "interaction_type",
        "source_interaction_type",
        "timestamp",
        "success",
        "error_message",
        "episode_id",
        "episode_index",
        "generation_id",
        "episode_score",
        "episode_score_scope",
        "source_episode_id",
        "source_row_index",
        "workflow_step",
        "actor_role",
        "record_meta",
    }
)


@dataclass(frozen=True)
class TrainingDataExportContext:
    generation_id: int
    run_id: str | None
    task_name: str
    source_generation_dir: Path
    source_all_episodes_path: Path
    export_recipe_hash: str


@dataclass
class EpisodeTrainingRows:
    episode_id: str
    episode_index: int | None
    generation_id: int
    episode_score: float | None
    rows: list[dict[str, Any]]


@runtime_checkable
class TrainingDataTransform(Protocol):
    @property
    def name(self) -> str: ...

    def config(self) -> Mapping[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]: ...


@dataclass(frozen=True)
class ExportRecipe:
    transforms: list[dict[str, Any]]
    recipe_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "transforms": self.transforms,
            "recipe_hash": self.recipe_hash,
        }


@dataclass(frozen=True)
class TrainingDataExport:
    rows: list[dict[str, Any]]
    report: dict[str, Any]
    recipe: ExportRecipe
    export_id: str | None = None
    export_dir: Path | None = None


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_hash(payload: Any) -> str:
    return _sha256_text(_canonical_json(payload))


def _assert_json_serializable(payload: Any, *, label: str) -> None:
    try:
        _canonical_json(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not JSON-serializable: {exc}") from exc


def _transform_class_path(transform: TrainingDataTransform) -> str:
    cls = type(transform)
    return f"{cls.__module__}.{cls.__qualname__}"


def build_export_recipe(
    transforms: Sequence[TrainingDataTransform],
) -> ExportRecipe:
    transform_specs: list[dict[str, Any]] = []
    for index, transform in enumerate(transforms):
        name = transform.name
        if not isinstance(name, str) or not name:
            raise ValueError(f"transform at index {index} has invalid name")
        config = transform.config()
        if not isinstance(config, Mapping):
            raise ValueError(f"transform {name} config() must return a mapping")
        config_payload = dict(config)
        _assert_json_serializable(config_payload, label=f"transform {name} config")
        transform_specs.append(
            {
                "index": index,
                "class_path": _transform_class_path(transform),
                "name": name,
                "config": config_payload,
                "config_hash": _json_hash(config_payload),
            }
        )
    recipe_payload = {"schema_version": 1, "transforms": transform_specs}
    return ExportRecipe(
        transforms=transform_specs,
        recipe_hash=_json_hash(recipe_payload),
    )


def _count_rows(groups: list[EpisodeTrainingRows]) -> int:
    return sum(len(group.rows) for group in groups)


def _canonical_row_hash(row: dict[str, Any]) -> str:
    return _json_hash(row)


def _snapshot_row_index(groups: list[EpisodeTrainingRows]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for group in groups:
        for row in group.rows:
            row_id = row["row_id"]
            if row_id in snapshot:
                raise ValueError(f"duplicate row_id: {row_id}")
            snapshot[row_id] = _canonical_row_hash(row)
    return snapshot


def _compute_transform_delta(
    transform: TrainingDataTransform,
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, Any]:
    before_ids = set(before)
    after_ids = set(after)
    retained = before_ids & after_ids
    return {
        "name": transform.name,
        "class_path": _transform_class_path(transform),
        "rows_in": len(before),
        "rows_out": len(after),
        "rows_removed": len(before_ids - after_ids),
        "rows_added": len(after_ids - before_ids),
        "rows_modified": sum(
            1 for row_id in retained if before[row_id] != after[row_id]
        ),
    }


def _flatten_episode_groups(groups: list[EpisodeTrainingRows]) -> list[dict[str, Any]]:
    return [row for group in groups for row in group.rows]


def _validate_group(value: Any, index: int) -> EpisodeTrainingRows:
    if not isinstance(value, EpisodeTrainingRows):
        raise ValueError(f"group {index} is not EpisodeTrainingRows")
    if not isinstance(value.episode_id, str) or not value.episode_id:
        raise ValueError(f"group {index} has invalid episode_id")
    if value.episode_index is not None and not isinstance(value.episode_index, int):
        raise ValueError(f"group {index} has invalid episode_index")
    if not isinstance(value.generation_id, int):
        raise ValueError(f"group {index} has invalid generation_id")
    if value.episode_score is not None and not isinstance(
        value.episode_score,
        int | float,
    ):
        raise ValueError(f"group {index} has invalid episode_score")
    if not isinstance(value.rows, list):
        raise ValueError(f"group {index} rows must be a list")
    return value


def validate_training_row_groups(
    groups: Any,
    *,
    source_row_ids: set[str] | None = None,
) -> None:
    if not isinstance(groups, list):
        raise ValueError("training row groups must be a list")

    seen_row_ids: set[str] = set()
    parent_row_ids: list[str] = []
    allowed_parent_ids = set(source_row_ids or set())
    for group_index, raw_group in enumerate(groups):
        group = _validate_group(raw_group, group_index)
        for row_index, row in enumerate(group.rows):
            if not isinstance(row, dict):
                raise ValueError(f"group {group_index} row {row_index} is not a dict")
            missing = REQUIRED_ROW_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"group {group_index} row {row_index} missing required fields: "
                    f"{sorted(missing)}"
                )
            _assert_json_serializable(
                row,
                label=f"group {group_index} row {row_index}",
            )
            row_id = row["row_id"]
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(
                    f"group {group_index} row {row_index} has invalid row_id"
                )
            if row_id in seen_row_ids:
                raise ValueError(f"duplicate row_id: {row_id}")
            seen_row_ids.add(row_id)
            if row.get("episode_score_scope") != "whole_episode":
                raise ValueError(
                    f"row {row_id} has invalid episode_score_scope: "
                    f"{row.get('episode_score_scope')!r}"
                )
            if row.get("episode_id") != group.episode_id:
                raise ValueError(
                    f"row {row_id} episode_id does not match group episode_id"
                )
            parent_row_id = row.get("parent_row_id")
            if parent_row_id is not None:
                if not isinstance(parent_row_id, str) or not parent_row_id:
                    raise ValueError(f"row {row_id} has invalid parent_row_id")
                parent_row_ids.append(parent_row_id)

    allowed_parent_ids.update(seen_row_ids)
    for parent_row_id in parent_row_ids:
        if parent_row_id not in allowed_parent_ids:
            raise ValueError(
                f"parent_row_id references unknown row_id: {parent_row_id}"
            )


def _validate_sft_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row.get("prompt"), str):
            raise ValueError(f"SFT row {index} prompt must be a string")
        if not isinstance(row.get("raw_response"), str):
            raise ValueError(f"SFT row {index} raw_response must be a string")
        row_id = row.get("row_id")
        if row_id in seen:
            raise ValueError(f"duplicate row_id: {row_id}")
        seen.add(str(row_id))


def _context_to_report_paths(context: TrainingDataExportContext) -> dict[str, str]:
    return {
        "source_all_episodes_path": str(context.source_all_episodes_path),
    }


def _source_group_order(groups: list[EpisodeTrainingRows]) -> str:
    if all(group.episode_index is not None for group in groups):
        return "episode_index"
    return "collector_order"


def _row_observability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_kinds: Counter[str] = Counter()
    artifact_routes: Counter[str] = Counter()

    for row in rows:
        meta = (
            row.get("record_meta") if isinstance(row.get("record_meta"), dict) else {}
        )
        pair_kind = meta.get("pair_kind") or meta.get("task_kind") or "unknown"
        pair_kinds[str(pair_kind)] += 1
        route = meta.get("artifact_route")
        if isinstance(route, str) and route:
            artifact_routes[route] += 1

    return {
        "rows_by_pair_kind": dict(sorted(pair_kinds.items())),
        "rows_by_artifact_route": dict(sorted(artifact_routes.items())),
        "rows_skipped_by_reason": {},
    }


def _task_row_observability(
    rows: list[dict[str, Any]],
    callback: RowObservabilityHook | None,
    *,
    reserved_keys: set[str],
) -> dict[str, Any]:
    if callback is None:
        return {}
    raw_metrics = callback(rows)
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("row observability hook must return a mapping")
    metrics = dict(raw_metrics)
    if any(not isinstance(key, str) or not key for key in metrics):
        raise ValueError("row observability keys must be non-empty strings")
    collisions = sorted(reserved_keys & set(metrics))
    if collisions:
        raise ValueError(
            f"row observability hook returned reserved report keys: {collisions}"
        )
    _assert_json_serializable(metrics, label="row observability metrics")
    return metrics


def build_training_export(
    generation_data: Any,
    transforms: Sequence[TrainingDataTransform],
    context: TrainingDataExportContext,
    *,
    row_observability: RowObservabilityHook | None = None,
) -> TrainingDataExport:
    recipe = build_export_recipe(transforms)

    groups = generation_data.get_successful_training_row_groups()
    source_group_order = _source_group_order(groups)
    validate_training_row_groups(groups)
    raw_count = _count_rows(groups)
    source_row_ids = set(_snapshot_row_index(groups))

    transform_reports: list[dict[str, Any]] = []
    for transform in transforms:
        before = _snapshot_row_index(groups)
        groups = transform.transform_export_rows(context, groups)
        validate_training_row_groups(groups, source_row_ids=source_row_ids)
        after = _snapshot_row_index(groups)
        transform_reports.append(_compute_transform_delta(transform, before, after))

    rows = _flatten_episode_groups(groups)
    _validate_sft_rows(rows)
    training_row_count = len(rows)
    observability = _row_observability(rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "task_name": context.task_name,
        "generation_id": context.generation_id,
        "run_id": context.run_id,
        "target_count_basis": TARGET_COUNT_BASIS,
        "successful_episodes_only": True,
        "raw_successful_row_count": raw_count,
        "training_row_count": training_row_count,
        "rows_removed": sum(item["rows_removed"] for item in transform_reports),
        "rows_added": sum(item["rows_added"] for item in transform_reports),
        "rows_modified": sum(item["rows_modified"] for item in transform_reports),
        "per_transform": transform_reports,
        "export_recipe_hash": recipe.recipe_hash,
        "export_recipe_path": "export_recipe.json",
        "source_group_order": source_group_order,
        "sft_training_rows_sha256": None,
        "exported_at": datetime.now(UTC).isoformat(),
        **_context_to_report_paths(context),
        **observability,
    }
    report.update(
        _task_row_observability(
            rows,
            row_observability,
            reserved_keys=set(report),
        )
    )
    return TrainingDataExport(rows=rows, report=report, recipe=recipe)


def count_training_rows(
    generation_data: Any,
    transforms: Sequence[TrainingDataTransform],
    context: TrainingDataExportContext,
) -> int:
    return len(build_training_export(generation_data, transforms, context).rows)


def _jsonl_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _new_export_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _validate_export_id(export_id: Any) -> str:
    if (
        not isinstance(export_id, str)
        or not export_id
        or not export_id[0].isascii()
        or not export_id[0].isalnum()
        or any(
            not character.isascii()
            or (not character.isalnum() and character not in "._-")
            for character in export_id
        )
    ):
        raise ValueError(
            "export_id must start with an ASCII letter or digit and contain only "
            "ASCII letters, digits, '.', '_', or '-'"
        )
    return export_id


def _relative_to_generation(path: Path, generation_dir: Path) -> str:
    return str(path.relative_to(generation_dir))


def publish_training_export(
    generation_dir: Path | str,
    export: TrainingDataExport,
    *,
    export_id: str | None = None,
) -> TrainingDataExport:
    generation_dir = Path(generation_dir)
    active_export_id = _validate_export_id(
        _new_export_id() if export_id is None else export_id
    )
    export_root = generation_dir / "exports" / "sft"
    export_root.mkdir(parents=True, exist_ok=True)
    final_dir = export_root / active_export_id
    if final_dir.exists():
        raise FileExistsError(f"SFT export already exists: {final_dir}")

    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".tmp-{active_export_id}-", dir=str(export_root))
    )
    renamed = False
    try:
        rows_path = tmp_dir / "sft_training_rows.jsonl"
        recipe_path = tmp_dir / "export_recipe.json"
        report_path = tmp_dir / "training_data_export_report.json"

        save_generation_training_data(export.rows, rows_path)
        recipe_payload = export.recipe.to_dict()
        save_durable_json(recipe_payload, recipe_path)

        rows_hash = _jsonl_sha256(rows_path)
        report_payload = dict(export.report)
        report_payload["sft_training_rows_sha256"] = rows_hash
        report_payload["export_recipe_path"] = "export_recipe.json"
        save_durable_json(report_payload, report_path)

        if report_payload["sft_training_rows_sha256"] != _jsonl_sha256(rows_path):
            raise RuntimeError("SFT row hash changed during export staging")
        _assert_json_serializable(recipe_payload, label="export recipe")
        _assert_json_serializable(report_payload, label="export report")

        tmp_dir.rename(final_dir)
        renamed = True
        _fsync_directory(export_root)
        latest_payload = {
            "schema_version": 1,
            "export_id": active_export_id,
            "export_path": _relative_to_generation(final_dir, generation_dir),
            "sft_training_rows_path": _relative_to_generation(
                final_dir / "sft_training_rows.jsonl",
                generation_dir,
            ),
            "training_data_export_report_path": _relative_to_generation(
                final_dir / "training_data_export_report.json",
                generation_dir,
            ),
            "export_recipe_path": _relative_to_generation(
                final_dir / "export_recipe.json",
                generation_dir,
            ),
            "export_recipe_hash": export.recipe.recipe_hash,
            "training_row_count": len(export.rows),
            "sft_training_rows_sha256": rows_hash,
        }
        save_durable_json(latest_payload, export_root / "latest.json")
        return TrainingDataExport(
            rows=export.rows,
            report=report_payload,
            recipe=export.recipe,
            export_id=active_export_id,
            export_dir=final_dir,
        )
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        elif renamed and final_dir.exists():
            latest_path = export_root / "latest.json"
            try:
                latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                export_is_active = False
            except (OSError, UnicodeError, json.JSONDecodeError):
                export_is_active = True
            else:
                export_is_active = (
                    isinstance(latest_payload, Mapping)
                    and latest_payload.get("export_id") == active_export_id
                )
            if not export_is_active:
                shutil.rmtree(final_dir)
        raise


def _read_generation_metadata(generation_dir: Path) -> dict[str, Any]:
    metadata_path = generation_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def regenerate_sft_export(
    generation_dir: Path | str,
    task: Any,
    *,
    export_id: str | None = None,
) -> TrainingDataExport:
    from src.typing.trajectory import EpisodeTrajectory, GenerationData

    generation_dir = Path(generation_dir)
    all_episodes_path = generation_dir / "all_episodes.jsonl"
    generation_data: GenerationData | None = None
    with all_episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = EpisodeTrajectory.from_dict(json.loads(line))
            if generation_data is None:
                generation_data = GenerationData(generation_id=episode.generation_id)
            generation_data.add_episode(episode)
    if generation_data is None:
        metadata = _read_generation_metadata(generation_dir)
        generation_data = GenerationData(
            generation_id=int(metadata.get("generation_id", 0))
        )

    metadata = _read_generation_metadata(generation_dir)
    transforms = tuple(task.training_data_transforms())
    recipe = build_export_recipe(transforms)
    context = TrainingDataExportContext(
        generation_id=generation_data.generation_id,
        run_id=metadata.get("run_id"),
        task_name=str(getattr(task, "name", type(task).__name__)),
        source_generation_dir=generation_dir,
        source_all_episodes_path=all_episodes_path,
        export_recipe_hash=recipe.recipe_hash,
    )
    export = build_training_export(
        generation_data,
        transforms,
        context,
        row_observability=task.training_data_observability,
    )
    return publish_training_export(generation_dir, export, export_id=export_id)


def resolve_latest_sft_training_rows_path(generation_dir: Path | str) -> Path:
    generation_dir = Path(generation_dir).resolve()
    latest_path = generation_dir / "exports" / "sft" / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"Active SFT export not found: {latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    export_id = _validate_export_id(latest.get("export_id"))
    relative_row_path = Path(
        "exports", "sft", export_id, "sft_training_rows.jsonl"
    )
    if latest.get("sft_training_rows_path") != str(relative_row_path):
        raise ValueError("sft_training_rows_path is not the active SFT export path")

    export_root = generation_dir / "exports" / "sft"
    active_export_dir = export_root / export_id
    row_path = active_export_dir / "sft_training_rows.jsonl"
    try:
        if export_root.resolve(strict=True) != export_root:
            raise ValueError("SFT export root is not a canonical active SFT export path")
        if active_export_dir.resolve(strict=True) != active_export_dir:
            raise ValueError("SFT export directory is not a canonical active SFT export path")
        resolved_row_path = row_path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data not found: {row_path}") from None
    if resolved_row_path != row_path or not row_path.is_file():
        raise ValueError("Training data is not a canonical active SFT export path")

    expected_hash = latest.get("sft_training_rows_sha256")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError("latest.json requires a valid sft_training_rows_sha256")
    actual_hash = _jsonl_sha256(row_path)
    if expected_hash != actual_hash:
        raise ValueError("SFT training rows SHA-256 mismatch")
    return row_path


__all__ = [
    "EpisodeTrainingRows",
    "ExportRecipe",
    "RowObservabilityHook",
    "TARGET_COUNT_BASIS",
    "TrainingDataExport",
    "TrainingDataExportContext",
    "TrainingDataTransform",
    "build_export_recipe",
    "build_training_export",
    "count_training_rows",
    "publish_training_export",
    "regenerate_sft_export",
    "resolve_latest_sft_training_rows_path",
    "validate_training_row_groups",
]
