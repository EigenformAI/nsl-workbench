"""Pure survivor-row export for the materials adsorption task."""

from __future__ import annotations

from typing import Any, Mapping

from src.training_data.transforms import (
    EpisodeTrainingRows,
    TrainingDataExportContext,
)


_PUBLIC_PROVENANCE_FIELDS = frozenset(
    {
        "candidate_universe_hash",
        "descriptor_registry_hash",
        "evidence_alias",
        "evidence_content_hash",
        "evidence_source_alias",
        "parent_hypothesis_ids",
        "parent_relationships",
        "parent_rule_hashes",
        "task_version",
    }
)


def _allowlisted_public_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    result: dict[str, Any] = {}
    scalar_fields = (
        "candidate_universe_hash",
        "descriptor_registry_hash",
        "evidence_alias",
        "evidence_content_hash",
        "evidence_source_alias",
        "task_version",
    )
    for key in scalar_fields:
        item = value.get(key)
        if isinstance(item, str):
            result[key] = item

    for key in ("parent_hypothesis_ids", "parent_rule_hashes"):
        item = value.get(key)
        if isinstance(item, list) and all(isinstance(entry, str) for entry in item):
            result[key] = list(item)

    relationships = value.get("parent_relationships")
    if isinstance(relationships, list):
        public_relationships = []
        for relationship in relationships:
            if not isinstance(relationship, Mapping):
                continue
            parent_id = relationship.get("parent_hypothesis_id")
            kind = relationship.get("relationship")
            if isinstance(parent_id, str) and isinstance(kind, str):
                public_relationships.append(
                    {
                        "parent_hypothesis_id": parent_id,
                        "relationship": kind,
                    }
                )
        result["parent_relationships"] = public_relationships

    return {key: result[key] for key in sorted(result)}


class MaterialsRewriteTransform:
    """Keep only real rewrite inference rows and rebuild their public schema."""

    @property
    def name(self) -> str:
        return "materials-rewrite-v1"

    def config(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "workflow_step": "rewrite",
            "record_meta_allowlist": {
                "public_provenance": sorted(_PUBLIC_PROVENANCE_FIELDS)
            },
        }

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        del context
        transformed: list[EpisodeTrainingRows] = []
        for episode in episodes:
            rows = [
                self._rebuild_row(row)
                for row in episode.rows
                if row.get("workflow_step") == "rewrite"
                and row.get("interaction_type") == "inference"
                and row.get("source_interaction_type") == "inference"
                and row.get("actor_role") == "assistant"
            ]
            if not rows:
                continue
            transformed.append(
                EpisodeTrainingRows(
                    episode_id=episode.episode_id,
                    episode_index=episode.episode_index,
                    generation_id=episode.generation_id,
                    episode_score=None,
                    rows=rows,
                )
            )
        return transformed

    @staticmethod
    def _rebuild_row(row: dict[str, Any]) -> dict[str, Any]:
        raw_meta = row.get("record_meta")
        meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        provenance = _allowlisted_public_provenance(meta.get("public_provenance"))

        # Rebuild every required field instead of copying then deleting known
        # secrets. New private metadata is therefore denied by default.
        return {
            "row_id": row["row_id"],
            "parent_row_id": row.get("parent_row_id"),
            "prompt": row["prompt"],
            "raw_response": row["raw_response"],
            "interaction_type": "inference",
            "source_interaction_type": "inference",
            "timestamp": row["timestamp"],
            "success": bool(row.get("success")),
            "error_message": None,
            "episode_id": row["episode_id"],
            "episode_index": row.get("episode_index"),
            "generation_id": row["generation_id"],
            "episode_score": None,
            "episode_score_scope": "whole_episode",
            "source_episode_id": row["source_episode_id"],
            "source_row_index": row["source_row_index"],
            "workflow_step": "rewrite",
            "actor_role": "assistant",
            "record_meta": {"public_provenance": provenance},
        }
