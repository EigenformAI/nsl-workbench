from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import src.training_data.transforms as transforms_module
from src.harness.recorder import TrajectoryRecord
from src.harness.training_row_adapter import (
    enrich_training_rows_for_episode,
    records_to_rows,
)
from src.training_data.transforms import (
    EpisodeTrainingRows,
    TrainingDataExportContext,
    build_export_recipe,
    build_training_export,
    publish_training_export,
    regenerate_sft_export,
    resolve_latest_sft_training_rows_path,
)
from src.typing.trajectory import EpisodeTrajectory, GenerationData


def _row(
    episode_id: str, row_index: int, *, workflow_step: str | None
) -> dict[str, Any]:
    return {
        "row_id": f"{episode_id}:{row_index}",
        "parent_row_id": None,
        "prompt": f"prompt-{episode_id}-{row_index}",
        "raw_response": f"response-{episode_id}-{row_index}",
        "interaction_type": "phase",
        "source_interaction_type": "phase",
        "timestamp": "2026-05-16T00:00:00",
        "success": True,
        "error_message": None,
        "episode_id": episode_id,
        "episode_index": int(episode_id.rsplit("-", 1)[1]),
        "generation_id": 3,
        "episode_score": 1.0,
        "episode_score_scope": "whole_episode",
        "source_episode_id": episode_id,
        "source_row_index": row_index,
        "workflow_step": workflow_step,
        "actor_role": None,
        "record_meta": {"workflow_step": workflow_step},
    }


def _episode(
    episode_index: int,
    *,
    workflow_steps: list[str | None],
    success: bool = True,
) -> EpisodeTrajectory:
    episode_id = f"ep-{episode_index}"
    raw_rows = [
        _row(episode_id, row_index, workflow_step=workflow_step)
        for row_index, workflow_step in enumerate(workflow_steps)
    ]
    return EpisodeTrajectory(
        episode_id=episode_id,
        generation_id=3,
        episode_index=episode_index,
        prompt_responses=[
            {
                "prompt": row["prompt"],
                "raw_response": row["raw_response"],
                "interaction_type": row["interaction_type"],
                "timestamp": row["timestamp"],
                "success": row["success"],
                "error_message": row["error_message"],
            }
            for row in raw_rows
        ],
        trajectory={},
        score=1.0 if success else 0.0,
        episode_runtime_success=success,
        success=success,
        llm_turns_count=len(raw_rows),
        container_variation="v1",
        started_at="2026-05-16T00:00:00",
        completed_at="2026-05-16T00:00:01",
        duration_seconds=1.0,
        raw_training_rows=raw_rows,
    )


def _generation_data() -> GenerationData:
    generation_data = GenerationData(generation_id=3)
    generation_data.add_episode(_episode(0, workflow_steps=["keep", "drop"]))
    generation_data.add_episode(_episode(1, workflow_steps=["keep"]))
    generation_data.add_episode(_episode(2, workflow_steps=["keep"], success=False))
    return generation_data


def _context(
    tmp_path: Path,
    transforms: tuple[Any, ...],
) -> TrainingDataExportContext:
    recipe = build_export_recipe(transforms)
    return TrainingDataExportContext(
        generation_id=3,
        run_id="run-123",
        task_name="dummy-task",
        source_generation_dir=tmp_path,
        source_all_episodes_path=tmp_path / "all_episodes.jsonl",
        export_recipe_hash=recipe.recipe_hash,
    )


class KeepWorkflowStep:
    def __init__(self, workflow_step: str) -> None:
        self.workflow_step = workflow_step

    @property
    def name(self) -> str:
        return "KeepWorkflowStep[v1]"

    def config(self) -> dict[str, Any]:
        return {"workflow_step": self.workflow_step}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        del context
        return [
            EpisodeTrainingRows(
                episode_id=episode.episode_id,
                episode_index=episode.episode_index,
                generation_id=episode.generation_id,
                episode_score=episode.episode_score,
                rows=[
                    row
                    for row in episode.rows
                    if row.get("workflow_step") == self.workflow_step
                ],
            )
            for episode in episodes
        ]


class RewritePrompt:
    @property
    def name(self) -> str:
        return "RewritePrompt[v1]"

    def config(self) -> dict[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        del context
        out: list[EpisodeTrainingRows] = []
        for episode in episodes:
            rows = []
            for row in episode.rows:
                rewritten = dict(row)
                rewritten["prompt"] = f"rewritten::{row['prompt']}"
                rows.append(rewritten)
            out.append(
                EpisodeTrainingRows(
                    episode_id=episode.episode_id,
                    episode_index=episode.episode_index,
                    generation_id=episode.generation_id,
                    episode_score=episode.episode_score,
                    rows=rows,
                )
            )
        return out


class DuplicateFirstRow:
    @property
    def name(self) -> str:
        return "DuplicateFirstRow[v1]"

    def config(self) -> dict[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        del context
        first = episodes[0].rows[0]
        return [
            EpisodeTrainingRows(
                episode_id=episodes[0].episode_id,
                episode_index=episodes[0].episode_index,
                generation_id=episodes[0].generation_id,
                episode_score=episodes[0].episode_score,
                rows=[first, dict(first)],
            )
        ]


class DummyTask:
    name = "dummy-task"

    def __init__(self, transforms: tuple[Any, ...]) -> None:
        self._transforms = transforms

    def training_data_transforms(self) -> tuple[Any, ...]:
        return self._transforms

    def generation_checkpoint_state(self) -> None:
        return None

    def training_data_observability(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"custom_rows": len(rows)}


def test_records_to_rows_preserves_generic_provenance() -> None:
    record = TrajectoryRecord(
        episode_id="ep-9",
        phase="orchestrator",
        messages=[{"role": "user", "content": "hello"}],
        response="world",
        usage=None,
        timestamp="2026-05-16T00:00:00",
        success=True,
        meta={
            "workflow_step": "explore",
            "actor_role": "proposer",
            "client": "unit-test",
            "not_jsonable": object(),
        },
    )

    rows = records_to_rows([record], run_id="run-123", version="abc123")
    enriched = enrich_training_rows_for_episode(
        rows,
        episode_id="ep-9",
        episode_index=9,
        generation_id=3,
        episode_score=0.75,
    )

    assert rows[0]["row_id"] == "ep-9:0"
    assert rows[0]["source_interaction_type"] == "orchestrator"
    assert rows[0]["workflow_step"] == "explore"
    assert rows[0]["actor_role"] == "proposer"
    assert rows[0]["record_meta"]["client"] == "unit-test"
    assert "not_jsonable" not in rows[0]["record_meta"]
    assert enriched[0]["episode_index"] == 9
    assert enriched[0]["generation_id"] == 3
    assert enriched[0]["episode_score"] == 0.75
    assert enriched[0]["episode_score_scope"] == "whole_episode"


def test_export_pipeline_filters_rows_and_reports_deltas(tmp_path: Path) -> None:
    transforms = (KeepWorkflowStep("keep"),)
    export = build_training_export(
        _generation_data(),
        transforms,
        _context(tmp_path, transforms),
    )

    assert len(export.rows) == 2
    assert {row["workflow_step"] for row in export.rows} == {"keep"}
    assert export.report["target_count_basis"] == "training_rows"
    assert export.report["raw_successful_row_count"] == 3
    assert export.report["training_row_count"] == 2
    assert export.report["rows_removed"] == 1
    assert export.report["rows_added"] == 0
    assert export.report["rows_modified"] == 0
    assert export.report["per_transform"][0]["name"] == "KeepWorkflowStep[v1]"
    assert export.report["per_transform"][0]["rows_in"] == 3
    assert export.report["per_transform"][0]["rows_out"] == 2


def test_export_merges_generic_and_task_observability(tmp_path: Path) -> None:
    export = build_training_export(
        _generation_data(),
        (),
        _context(tmp_path, ()),
        row_observability=lambda rows: {"custom_rows": len(rows)},
    )

    assert export.report["rows_by_pair_kind"] == {"unknown": 3}
    assert export.report["rows_by_artifact_route"] == {}
    assert export.report["custom_rows"] == 3


@pytest.mark.parametrize(
    ("callback", "message"),
    [
        (lambda _rows: [], "must return a mapping"),
        (lambda _rows: {"training_row_count": 99}, "reserved report keys"),
        (lambda _rows: {"custom": object()}, "not JSON-serializable"),
    ],
)
def test_export_rejects_invalid_task_observability(
    tmp_path: Path,
    callback: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_training_export(
            _generation_data(),
            (),
            _context(tmp_path, ()),
            row_observability=callback,
        )


def test_export_pipeline_tracks_modified_rows(tmp_path: Path) -> None:
    transforms = (RewritePrompt(),)
    export = build_training_export(
        _generation_data(),
        transforms,
        _context(tmp_path, transforms),
    )

    assert len(export.rows) == 3
    assert export.report["rows_modified"] == 3
    assert export.report["rows_removed"] == 0
    assert export.report["rows_added"] == 0


def test_export_validation_rejects_duplicate_row_ids(tmp_path: Path) -> None:
    transforms = (DuplicateFirstRow(),)
    with pytest.raises(ValueError, match="duplicate row_id"):
        build_training_export(
            _generation_data(),
            transforms,
            _context(tmp_path, transforms),
        )


def test_save_and_regenerate_publish_sft_bundle(tmp_path: Path) -> None:
    from src.execution.generation import save_generation_data

    transforms = (KeepWorkflowStep("keep"),)
    generation_dir = save_generation_data(
        generation_data=_generation_data(),
        output_dir=tmp_path,
        run_id="run-123",
        task=DummyTask(transforms),
    )

    latest_path = generation_dir / "exports" / "sft" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    first_rows_path = generation_dir / latest["sft_training_rows_path"]
    first_report_path = generation_dir / latest["training_data_export_report_path"]

    assert first_rows_path.exists()
    assert first_report_path.exists()
    assert latest["sft_training_rows_sha256"] == hashlib.sha256(
        first_rows_path.read_bytes()
    ).hexdigest()
    assert resolve_latest_sft_training_rows_path(generation_dir) == first_rows_path
    assert not (generation_dir / "sft_training_rows.jsonl").exists()
    assert len(first_rows_path.read_text(encoding="utf-8").splitlines()) == 2
    first_report = json.loads(first_report_path.read_text(encoding="utf-8"))
    assert first_report["custom_rows"] == 2

    replay = regenerate_sft_export(
        generation_dir,
        DummyTask(transforms),
        export_id="replay",
    )
    latest_after_replay = json.loads(latest_path.read_text(encoding="utf-8"))

    assert replay.export_id == "replay"
    assert latest_after_replay["export_id"] == "replay"
    assert (generation_dir / latest_after_replay["sft_training_rows_path"]).exists()
    assert replay.report["custom_rows"] == 2


@pytest.mark.parametrize(
    "export_id",
    ["", ".", "..", "../escape", "sibling/export", "/absolute"],
)
def test_publish_rejects_unsafe_export_id(tmp_path: Path, export_id: str) -> None:
    export = build_training_export(
        _generation_data(),
        (),
        _context(tmp_path, ()),
    )

    with pytest.raises(ValueError, match="export_id"):
        publish_training_export(tmp_path, export, export_id=export_id)


def test_publish_can_retry_same_id_after_post_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = build_training_export(_generation_data(), (), _context(tmp_path, ()))
    original_fsync_directory = transforms_module._fsync_directory

    def fail_fsync_directory(path: Path) -> None:
        raise OSError(f"injected fsync failure: {path}")

    monkeypatch.setattr(
        transforms_module, "_fsync_directory", fail_fsync_directory
    )
    with pytest.raises(OSError, match="injected fsync failure"):
        publish_training_export(tmp_path, export, export_id="retryable")

    export_dir = tmp_path / "exports" / "sft" / "retryable"
    assert not export_dir.exists()
    monkeypatch.setattr(
        transforms_module, "_fsync_directory", original_fsync_directory
    )

    published = publish_training_export(tmp_path, export, export_id="retryable")

    assert published.export_dir == export_dir
    assert export_dir.exists()


def test_publish_preserves_export_if_latest_was_replaced_before_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = build_training_export(_generation_data(), (), _context(tmp_path, ()))
    original_save_durable_json = transforms_module.save_durable_json

    def save_then_fail(payload: dict[str, Any], output_path: Path) -> None:
        original_save_durable_json(payload, output_path)
        if output_path.name == "latest.json":
            raise OSError("injected post-replace failure")

    monkeypatch.setattr(transforms_module, "save_durable_json", save_then_fail)
    with pytest.raises(OSError, match="injected post-replace failure"):
        publish_training_export(tmp_path, export, export_id="active")

    export_root = tmp_path / "exports" / "sft"
    assert (export_root / "active").exists()
    latest = json.loads((export_root / "latest.json").read_text(encoding="utf-8"))
    assert latest["export_id"] == "active"


def test_resolve_rejects_noncanonical_active_export_paths(tmp_path: Path) -> None:
    export = build_training_export(
        _generation_data(),
        (),
        _context(tmp_path, ()),
    )
    published = publish_training_export(tmp_path, export, export_id="active")
    assert published.export_dir is not None
    latest_path = tmp_path / "exports" / "sft" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    external_path = tmp_path / "external.jsonl"
    external_path.write_bytes(
        (published.export_dir / "sft_training_rows.jsonl").read_bytes()
    )
    sibling_dir = tmp_path / "exports" / "sft" / "sibling"
    sibling_dir.mkdir()
    sibling_path = sibling_dir / "sft_training_rows.jsonl"
    sibling_path.write_bytes(external_path.read_bytes())

    for row_path in (
        str(external_path),
        "exports/sft/sibling/sft_training_rows.jsonl",
        "exports/sft/active/../sibling/sft_training_rows.jsonl",
    ):
        latest["sft_training_rows_path"] = row_path
        latest_path.write_text(json.dumps(latest), encoding="utf-8")
        with pytest.raises(ValueError, match="active SFT export path"):
            resolve_latest_sft_training_rows_path(tmp_path)


def test_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    export_root = tmp_path / "exports" / "sft"
    export_root.mkdir(parents=True)
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    rows_path = external_dir / "sft_training_rows.jsonl"
    rows_path.write_text("{}\n", encoding="utf-8")
    (export_root / "active").symlink_to(external_dir, target_is_directory=True)
    (export_root / "latest.json").write_text(
        json.dumps(
            {
                "export_id": "active",
                "sft_training_rows_path": (
                    "exports/sft/active/sft_training_rows.jsonl"
                ),
                "sft_training_rows_sha256": hashlib.sha256(
                    rows_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="active SFT export path"):
        resolve_latest_sft_training_rows_path(tmp_path)


def test_resolve_requires_sft_training_rows_sha256(tmp_path: Path) -> None:
    export = build_training_export(_generation_data(), (), _context(tmp_path, ()))
    publish_training_export(tmp_path, export, export_id="active")
    latest_path = tmp_path / "exports" / "sft" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    del latest["sft_training_rows_sha256"]
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    with pytest.raises(ValueError, match="sft_training_rows_sha256"):
        resolve_latest_sft_training_rows_path(tmp_path)


def test_resolve_rejects_sft_training_rows_hash_mismatch(tmp_path: Path) -> None:
    export = build_training_export(_generation_data(), (), _context(tmp_path, ()))
    published = publish_training_export(tmp_path, export, export_id="active")
    assert published.export_dir is not None
    rows_path = published.export_dir / "sft_training_rows.jsonl"
    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "{}\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_latest_sft_training_rows_path(tmp_path)
