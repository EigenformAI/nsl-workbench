"""Adapter: TrajectoryRecord → legacy training-row dict.

The framework's authoritative training-data source is
:attr:`EventRecorder.inference_records`. This adapter renders each record
as the flat dict shape ``scripts/run_episode.py`` previously wrote.

Invariant: ``len(recorder.inference_records) == len(legacy prompt_responses)``
on pinned seeds (regression test ``test_row_count_parity``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from src.harness.recorder import TrajectoryRecord
from src.typing.message import Message


def _render_prompt(messages: list[Message]) -> str:
    """Render a messages list into a single prompt string.

    Mirrors the minimal concatenation shape used by the old flow
    (role: content, separated by blank lines). Downstream training
    pipelines re-template anyway — this is a human-readable form for
    per-row inspection.
    """
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if content is None and tool_calls:
            content = json.dumps(tool_calls)
        parts.append(f"[{role}]\n{content or ''}")
    return "\n\n".join(parts)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for key, nested in value.items():
                try:
                    json.dumps(key)
                except (TypeError, ValueError):
                    continue
                filtered = _jsonable(nested)
                if filtered is not _SKIP:
                    out[str(key)] = filtered
            return out
        if isinstance(value, list | tuple):
            out = []
            for nested in value:
                filtered = _jsonable(nested)
                if filtered is not _SKIP:
                    out.append(filtered)
            return out
        return _SKIP
    return value


class _Skip:
    pass


_SKIP = _Skip()


def _jsonable_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        filtered = _jsonable(value)
        if filtered is not _SKIP:
            out[str(key)] = filtered
    return out


def record_to_row(
    record: TrajectoryRecord,
    *,
    run_id: str,
    version: str,
    source_row_index: int = 0,
) -> dict[str, Any]:
    record_meta = _jsonable_meta(record.meta)
    workflow_step = record_meta.get("workflow_step")
    actor_role = record_meta.get("actor_role")
    return {
        "run_id": run_id,
        "version": version,
        "row_id": f"{record.episode_id}:{source_row_index}",
        "parent_row_id": None,
        "prompt": _render_prompt(record.messages),
        "raw_response": record.response,
        "interaction_type": record.phase,
        "source_interaction_type": record.phase,
        "timestamp": record.timestamp,
        "success": record.success,
        "error_message": record.error_message,
        "episode_id": record.episode_id,
        "source_episode_id": record.episode_id,
        "source_row_index": source_row_index,
        "workflow_step": workflow_step if isinstance(workflow_step, str) else None,
        "actor_role": actor_role if isinstance(actor_role, str) else None,
        "record_meta": record_meta,
    }


def enrich_training_rows_for_episode(
    rows: list[dict[str, Any]],
    *,
    episode_id: str,
    episode_index: int | None,
    generation_id: int,
    episode_score: float | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for fallback_index, row in enumerate(rows):
        source_row_index = int(row.get("source_row_index", fallback_index))
        interaction_type = str(row.get("interaction_type", ""))
        record_meta = row.get("record_meta")
        if not isinstance(record_meta, Mapping):
            record_meta = {}
        workflow_step = row.get("workflow_step")
        actor_role = row.get("actor_role")
        raw_response = row.get("raw_response", row.get("completion", ""))
        enriched.append(
            {
                **row,
                "row_id": row.get("row_id") or f"{episode_id}:{source_row_index}",
                "parent_row_id": row.get("parent_row_id"),
                "prompt": row["prompt"],
                "raw_response": raw_response,
                "interaction_type": interaction_type,
                "source_interaction_type": row.get(
                    "source_interaction_type",
                    interaction_type,
                ),
                "timestamp": row.get("timestamp", ""),
                "success": bool(row.get("success", True)),
                "error_message": row.get("error_message"),
                "episode_id": episode_id,
                "episode_index": episode_index,
                "generation_id": generation_id,
                "episode_score": episode_score,
                "episode_score_scope": "whole_episode",
                "source_episode_id": row.get("source_episode_id") or episode_id,
                "source_row_index": source_row_index,
                "workflow_step": workflow_step if isinstance(workflow_step, str) else None,
                "actor_role": actor_role if isinstance(actor_role, str) else None,
                "record_meta": _jsonable_meta(record_meta),
            }
        )
    return enriched


def records_to_rows(
    records: list[TrajectoryRecord],
    *,
    run_id: str,
    version: str,
) -> list[dict[str, Any]]:
    return [
        record_to_row(r, run_id=run_id, version=version, source_row_index=index)
        for index, r in enumerate(records)
    ]
