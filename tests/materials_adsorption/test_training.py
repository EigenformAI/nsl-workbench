from __future__ import annotations

import json
from pathlib import Path

from src.training_data.transforms import EpisodeTrainingRows, TrainingDataExportContext
from tasks.materials_adsorption.training import MaterialsRewriteTransform


def _row(row_id: str, step: str, *, private: bool = False) -> dict:
    return {
        "row_id": row_id,
        "parent_row_id": None,
        "prompt": "public prompt",
        "raw_response": "public target",
        "interaction_type": "inference",
        "source_interaction_type": "inference",
        "timestamp": "2026-07-20T00:00:00",
        "success": True,
        "error_message": None,
        "episode_id": "ep-1",
        "episode_index": 1,
        "generation_id": 0,
        "episode_score": 9.99,
        "episode_score_scope": "whole_episode",
        "source_episode_id": "ep-1",
        "source_row_index": 0,
        "workflow_step": step,
        "actor_role": "assistant",
        "record_meta": {
            "public_provenance": {"evidence_alias": "evidence-1"},
            "private_path": "/private/labels.sqlite" if private else None,
            "scores": {"stage1": 9.99},
            "nested": {"fold_id": 3},
        },
    }


def test_transform_keeps_only_rewrite_and_rebuilds_metadata_from_allowlist(tmp_path) -> None:
    context = TrainingDataExportContext(
        generation_id=0,
        run_id="run-a",
        task_name="materials-adsorption",
        source_generation_dir=tmp_path,
        source_all_episodes_path=Path(tmp_path / "all_episodes.jsonl"),
        export_recipe_hash="recipe",
    )
    groups = [
        EpisodeTrainingRows(
            episode_id="ep-1",
            episode_index=1,
            generation_id=0,
            episode_score=9.99,
            rows=[_row("code", "code"), _row("rewrite", "rewrite", private=True)],
        )
    ]

    transformed = MaterialsRewriteTransform().transform_export_rows(context, groups)

    assert len(transformed) == 1
    assert len(transformed[0].rows) == 1
    row = transformed[0].rows[0]
    assert row["row_id"] == "rewrite"
    assert row["record_meta"] == {
        "public_provenance": {"evidence_alias": "evidence-1"}
    }
    serialized = json.dumps(row, sort_keys=True)
    for forbidden in ("private_path", "labels.sqlite", "stage1", "fold_id", "9.99"):
        assert forbidden not in serialized
