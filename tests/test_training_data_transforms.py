"""TDD tests for the training data export transform pipeline.

All imports from new/changed modules will fail until implementation is done.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from src.training_data.transforms import (
    EpisodeTrainingRows,
    TrainingDataExportContext,
    build_training_export,
    count_training_rows,
    validate_training_row_groups,
)
from src.typing.trajectory import EpisodeTrajectory, GenerationData

REQUIRED_ROW_FIELDS = {
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


# ---------------------------------------------------------------------------
# Row / group helpers
# ---------------------------------------------------------------------------


def _make_row(
    *,
    episode_id: str = "ep-0",
    episode_index: int = 0,
    generation_id: int = 1,
    row_index: int = 0,
    workflow_step: str | None = "explore",
    actor_role: str | None = None,
) -> dict[str, Any]:
    row_id = f"{episode_id}:{row_index}"
    return {
        "row_id": row_id,
        "parent_row_id": None,
        "prompt": f"prompt-{row_index}",
        "raw_response": f"response-{row_index}",
        "interaction_type": "orchestrator",
        "source_interaction_type": "orchestrator",
        "timestamp": "2026-05-16T00:00:00",
        "success": True,
        "error_message": None,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "generation_id": generation_id,
        "episode_score": 1.0,
        "episode_score_scope": "whole_episode",
        "source_episode_id": episode_id,
        "source_row_index": row_index,
        "workflow_step": workflow_step,
        "actor_role": actor_role,
        "record_meta": {},
    }


def _make_episode_rows(
    episode_id: str = "ep-0",
    *,
    num_rows: int = 2,
    episode_index: int = 0,
    generation_id: int = 1,
    episode_score: float = 1.0,
    workflow_step: str | None = "explore",
) -> EpisodeTrainingRows:
    rows = [
        _make_row(
            episode_id=episode_id,
            episode_index=episode_index,
            generation_id=generation_id,
            row_index=i,
            workflow_step=workflow_step,
        )
        for i in range(num_rows)
    ]
    return EpisodeTrainingRows(
        episode_id=episode_id,
        episode_index=episode_index,
        generation_id=generation_id,
        episode_score=episode_score,
        rows=rows,
    )


def _make_context(generation_id: int = 1) -> TrainingDataExportContext:
    return TrainingDataExportContext(
        generation_id=generation_id,
        run_id="run-test",
        task_name="test_task",
        source_generation_dir=Path("/tmp/gen"),
        source_all_episodes_path=Path("/tmp/gen/all_episodes.jsonl"),
        export_recipe_hash="abc123",
    )


def _make_trajectory(
    episode_id: str,
    *,
    episode_index: int = 0,
    generation_id: int = 1,
    score: float = 1.0,
    success: bool = True,
    raw_training_rows: list[dict[str, Any]] | None = None,
) -> EpisodeTrajectory:
    if raw_training_rows is None:
        raw_training_rows = [
            _make_row(episode_id=episode_id, episode_index=episode_index, row_index=i)
            for i in range(2)
        ]
    return EpisodeTrajectory(
        episode_id=episode_id,
        generation_id=generation_id,
        episode_index=episode_index,
        prompt_responses=[],
        trajectory={},
        score=score,
        episode_runtime_success=success,
        success=success,
        llm_turns_count=2,
        container_variation="v1",
        started_at="2026-05-16T00:00:00",
        completed_at="2026-05-16T00:01:00",
        duration_seconds=60.0,
        raw_training_rows=raw_training_rows,
    )


# ---------------------------------------------------------------------------
# Stub transforms used across tests
# ---------------------------------------------------------------------------


class _IdentityTransform:
    @property
    def name(self) -> str:
        return "Identity[v1]"

    def config(self) -> Mapping[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        return episodes


class _DropAllTransform:
    @property
    def name(self) -> str:
        return "DropAll[v1]"

    def config(self) -> Mapping[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        return [
            EpisodeTrainingRows(
                episode_id=ep.episode_id,
                episode_index=ep.episode_index,
                generation_id=ep.generation_id,
                episode_score=ep.episode_score,
                rows=[],
            )
            for ep in episodes
        ]


class _KeepByWorkflowStep:
    def __init__(self, allowed: set[str]) -> None:
        self._allowed = allowed

    @property
    def name(self) -> str:
        return "KeepByStep[v1]"

    def config(self) -> Mapping[str, Any]:
        return {"allowed": sorted(self._allowed)}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        return [
            EpisodeTrainingRows(
                episode_id=ep.episode_id,
                episode_index=ep.episode_index,
                generation_id=ep.generation_id,
                episode_score=ep.episode_score,
                rows=[r for r in ep.rows if r.get("workflow_step") in self._allowed],
            )
            for ep in episodes
        ]


class _RaisingTransform:
    @property
    def name(self) -> str:
        return "Raiser[v1]"

    def config(self) -> Mapping[str, Any]:
        return {}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        raise ValueError("intentional failure from Raiser transform")


class _TaggingTransform:
    """Appends a tag suffix to raw_response so chaining order is observable."""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    @property
    def name(self) -> str:
        return f"Tagger[{self._tag}]"

    def config(self) -> Mapping[str, Any]:
        return {"tag": self._tag}

    def transform_export_rows(
        self,
        context: TrainingDataExportContext,
        episodes: list[EpisodeTrainingRows],
    ) -> list[EpisodeTrainingRows]:
        result = []
        for ep in episodes:
            rows = [
                {**r, "raw_response": r["raw_response"] + f"|{self._tag}"}
                for r in ep.rows
            ]
            result.append(
                EpisodeTrainingRows(
                    episode_id=ep.episode_id,
                    episode_index=ep.episode_index,
                    generation_id=ep.generation_id,
                    episode_score=ep.episode_score,
                    rows=rows,
                )
            )
        return result


# ---------------------------------------------------------------------------
# EpisodeTrainingRows
# ---------------------------------------------------------------------------


class TestEpisodeTrainingRows(unittest.TestCase):
    def test_construction_variants(self) -> None:
        rows = [_make_row(row_index=0), _make_row(row_index=1)]
        cases = [
            (
                "populated",
                EpisodeTrainingRows("ep-0", 0, 1, 0.75, rows),
                ("ep-0", 0, 1, 0.75, rows),
            ),
            (
                "empty-rows-null-score",
                EpisodeTrainingRows("ep-empty", 0, 1, None, []),
                ("ep-empty", 0, 1, None, []),
            ),
            (
                "null-episode-index",
                EpisodeTrainingRows("ep-0", None, 1, 0.5, []),
                ("ep-0", None, 1, 0.5, []),
            ),
        ]

        for name, group, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    (
                        group.episode_id,
                        group.episode_index,
                        group.generation_id,
                        group.episode_score,
                        group.rows,
                    ),
                    expected,
                )


# ---------------------------------------------------------------------------
# validate_training_row_groups
# ---------------------------------------------------------------------------


class TestValidateTrainingRowGroups(unittest.TestCase):
    def test_valid_inputs(self) -> None:
        cases = [
            (
                "multiple-groups",
                [_make_episode_rows("ep-0"), _make_episode_rows("ep-1", episode_index=1)],
            ),
            ("empty-group-list", []),
            ("empty-rows", [EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[])]),
        ]

        for name, groups in cases:
            with self.subTest(name=name):
                validate_training_row_groups(groups)

    def test_invalid_inputs(self) -> None:
        missing_workflow = _make_row()
        del missing_workflow["workflow_step"]
        missing_row_id = _make_row()
        del missing_row_id["row_id"]
        invalid_scope = _make_row()
        invalid_scope["episode_score_scope"] = "per_row"
        non_json_meta = _make_row()
        non_json_meta["record_meta"] = {"obj": object()}
        duplicate_a = _make_row(episode_id="ep-0", row_index=0)
        duplicate_b = _make_row(episode_id="ep-0", row_index=0)
        within_a = _make_row(row_index=0)
        within_b = _make_row(row_index=0)
        cases = [
            (
                "missing-required-field",
                [EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[missing_workflow])],
            ),
            (
                "missing-row-id",
                [EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[missing_row_id])],
            ),
            (
                "duplicate-across-groups",
                [
                    EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[duplicate_a]),
                    EpisodeTrainingRows("ep-1", 1, 1, 1.0, rows=[duplicate_b]),
                ],
            ),
            (
                "invalid-score-scope",
                [EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[invalid_scope])],
            ),
            (
                "non-json-meta",
                [EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[non_json_meta])],
            ),
            (
                "duplicate-within-group",
                [EpisodeTrainingRows("ep-0", 0, 1, 1.0, rows=[within_a, within_b])],
            ),
        ]

        for name, groups in cases:
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError)):
                validate_training_row_groups(groups)


# ---------------------------------------------------------------------------
# build_training_export
# ---------------------------------------------------------------------------


def _make_generation_data(
    *, num_episodes: int = 2, rows_per_episode: int = 2
) -> GenerationData:
    gd = GenerationData(generation_id=1)
    for i in range(num_episodes):
        raw_rows = [
            _make_row(episode_id=f"ep-{i}", episode_index=i, row_index=j)
            for j in range(rows_per_episode)
        ]
        gd.add_episode(
            _make_trajectory(f"ep-{i}", episode_index=i, raw_training_rows=raw_rows)
        )
    return gd


class TestBuildSftTrainingExport(unittest.TestCase):
    def test_identity_export_report_and_ordering(self) -> None:
        gd = _make_generation_data(num_episodes=2, rows_per_episode=3)
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        self.assertEqual(
            [row["episode_id"] for row in result.rows],
            ["ep-0", "ep-0", "ep-0", "ep-1", "ep-1", "ep-1"],
        )
        self.assertEqual(result.report["training_row_count"], 6)
        self.assertEqual(result.report["raw_successful_row_count"], 6)
        self.assertEqual(result.report["rows_removed"], 0)
        self.assertEqual(result.report["rows_added"], 0)
        required_report_keys = {
            "training_row_count",
            "raw_successful_row_count",
            "rows_removed",
            "rows_added",
            "per_transform",
            "successful_episodes_only",
            "target_count_basis",
        }
        self.assertEqual(required_report_keys - set(result.report), set())
        self.assertEqual(result.report["target_count_basis"], "training_rows")

    def test_transform_accounting_and_score_provenance(self) -> None:
        gd = _make_generation_data(num_episodes=2, rows_per_episode=3)
        ctx = _make_context()
        dropped = build_training_export(
            gd, transforms=[_DropAllTransform()], context=ctx
        )
        self.assertEqual(dropped.report["training_row_count"], 0)
        self.assertEqual(dropped.report["raw_successful_row_count"], 6)
        self.assertEqual(dropped.report["rows_removed"], 6)
        self.assertEqual(dropped.report["rows_added"], 0)

        raw_rows = [
            _make_row(episode_id="ep-0", row_index=0, workflow_step="explore"),
            _make_row(episode_id="ep-0", row_index=1, workflow_step="execute"),
        ]
        filtered_gd = GenerationData(generation_id=1)
        filtered_gd.add_episode(
            _make_trajectory("ep-0", score=0.9, raw_training_rows=raw_rows)
        )
        filtered = build_training_export(
            filtered_gd,
            transforms=[_KeepByWorkflowStep({"explore"})],
            context=ctx,
        )
        self.assertEqual(len(filtered.rows), 1)
        self.assertEqual(filtered.rows[0]["episode_score"], 0.9)
        self.assertEqual(filtered.rows[0]["episode_score_scope"], "whole_episode")
        self.assertEqual(filtered.rows[0]["workflow_step"], "explore")

        delta_gd = _make_generation_data(num_episodes=1, rows_per_episode=4)
        delta = build_training_export(
            delta_gd,
            transforms=[_IdentityTransform(), _DropAllTransform()],
            context=ctx,
        )
        per_transform = delta.report["per_transform"]
        self.assertEqual(len(per_transform), 2)
        self.assertEqual(per_transform[0]["rows_removed"], 0)
        self.assertEqual(per_transform[1]["rows_removed"], 4)

    def test_transform_order_is_load_bearing(self) -> None:
        gd = _make_generation_data(num_episodes=1, rows_per_episode=1)
        ctx = _make_context()

        result_ab = build_training_export(
            gd,
            transforms=[_TaggingTransform("A"), _TaggingTransform("B")],
            context=ctx,
        )
        result_ba = build_training_export(
            gd,
            transforms=[_TaggingTransform("B"), _TaggingTransform("A")],
            context=ctx,
        )

        self.assertEqual(len(result_ab.rows), 1)
        self.assertIn("|A|B", result_ab.rows[0]["raw_response"])
        self.assertIn("|B|A", result_ba.rows[0]["raw_response"])

    def test_raising_transform_fails_strictly(self) -> None:
        gd = _make_generation_data(num_episodes=1, rows_per_episode=1)
        ctx = _make_context()

        with self.assertRaises(Exception):
            build_training_export(
                gd, transforms=[_RaisingTransform()], context=ctx
            )

    def test_failed_episodes_excluded_from_export(self) -> None:
        gd = GenerationData(generation_id=1)
        raw_rows = [_make_row(episode_id="ep-fail")]
        gd.add_episode(
            _make_trajectory(
                "ep-fail", score=0.0, success=False, raw_training_rows=raw_rows
            )
        )
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        self.assertEqual(result.report["training_row_count"], 0)
        self.assertTrue(result.report["successful_episodes_only"])

    def test_empty_generation_data_produces_zero_rows(self) -> None:
        gd = GenerationData(generation_id=1)
        ctx = _make_context()

        result = build_training_export(gd, transforms=[], context=ctx)

        self.assertEqual(result.rows, [])
        self.assertEqual(result.report["training_row_count"], 0)

# ---------------------------------------------------------------------------
# count_training_rows
# ---------------------------------------------------------------------------


class TestCountExportedSftRows(unittest.TestCase):
    def _make_gd_with_steps(self, workflow_steps: list[str]) -> GenerationData:
        gd = GenerationData(generation_id=1)
        raw_rows = [
            _make_row(episode_id="ep-0", row_index=i, workflow_step=step)
            for i, step in enumerate(workflow_steps)
        ]
        gd.add_episode(_make_trajectory("ep-0", raw_training_rows=raw_rows))
        return gd

    def test_count_training_rows_matrix(self) -> None:
        cases = [
            (
                "identity",
                ["explore", "hypothesise", "execute"],
                [],
                3,
            ),
            (
                "filtered",
                ["explore", "hypothesise", "execute", "explore"],
                [_KeepByWorkflowStep({"explore", "hypothesise"})],
                3,
            ),
            ("drop-all", ["explore", "execute"], [_DropAllTransform()], 0),
        ]

        for name, steps, transforms, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    count_training_rows(
                        self._make_gd_with_steps(steps),
                        transforms=transforms,
                        context=_make_context(),
                    ),
                    expected,
                )


# ---------------------------------------------------------------------------
# record_to_row enrichment
# ---------------------------------------------------------------------------


class TestRecordToRowEnrichment(unittest.TestCase):
    """Fails until training_row_adapter.py is updated."""

    def _make_record(
        self,
        *,
        episode_id: str = "ep-0",
        phase: str = "orchestrator",
        workflow_step: str | None = "explore",
        actor_role: str | None = None,
    ):
        from src.harness.recorder import TrajectoryRecord

        return TrajectoryRecord(
            episode_id=episode_id,
            phase=phase,
            messages=[{"role": "user", "content": "hello"}],
            response="response text",
            usage=None,
            timestamp="2026-05-16T00:00:00",
            success=True,
            error_message=None,
            meta={"workflow_step": workflow_step, "actor_role": actor_role},
        )

    def test_record_to_row_complete_mapping(self) -> None:
        from src.harness.training_row_adapter import (
            enrich_training_rows_for_episode,
            record_to_row,
        )

        record = self._make_record(
            episode_id="ep-42",
            phase="mcp_explore_call",
            workflow_step="hypothesise",
            actor_role="proposer",
        )
        record.meta["extra_key"] = "extra_val"
        row = record_to_row(record, run_id="run-1", version="v1", source_row_index=3)
        enriched = enrich_training_rows_for_episode(
            [row],
            episode_id="ep-42",
            episode_index=42,
            generation_id=1,
            episode_score=0.9,
        )

        self.assertEqual(row["workflow_step"], "hypothesise")
        self.assertEqual(row["actor_role"], "proposer")
        self.assertEqual(row["source_episode_id"], "ep-42")
        self.assertEqual(row["source_row_index"], 3)
        self.assertEqual(row["source_interaction_type"], "mcp_explore_call")
        self.assertEqual(row["row_id"], "ep-42:3")
        self.assertIsNone(row["parent_row_id"])
        self.assertIsInstance(row["record_meta"], dict)
        self.assertEqual(row["record_meta"]["extra_key"], "extra_val")
        missing = REQUIRED_ROW_FIELDS - set(enriched[0])
        self.assertEqual(missing, set(), f"enriched row missing fields: {missing}")

        nullable = record_to_row(
            self._make_record(workflow_step=None),
            run_id="run-1",
            version="v1",
            source_row_index=0,
        )
        self.assertIn("workflow_step", nullable)
        self.assertIsNone(nullable["workflow_step"])

    def test_records_to_rows_stable_ids_and_indices(self) -> None:
        from src.harness.training_row_adapter import records_to_rows

        records = [self._make_record() for _ in range(3)]
        rows = records_to_rows(records, run_id="run-1", version="v1")

        self.assertEqual([row["source_row_index"] for row in rows], [0, 1, 2])
        self.assertEqual([row["row_id"] for row in rows], ["ep-0:0", "ep-0:1", "ep-0:2"])


# ---------------------------------------------------------------------------
# EventRecorder workflow_step stamping
# ---------------------------------------------------------------------------


class TestRecorderWorkflowStepStamping(unittest.TestCase):
    """Fails until recorder.py stamps last_workflow_step into record.meta."""

    def _make_recorder(self, episode_id: str = "ep-test") -> tuple[Any, str]:
        from src.harness.recorder import EventRecorder

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        return EventRecorder(episode_id, Path(tmp.name)), tmp.name

    def test_workflow_step_label_transitions(self) -> None:
        from src.harness.recorder import TrajectoryRecord

        recorder, _ = self._make_recorder()

        def _record(phase: str) -> TrajectoryRecord:
            return TrajectoryRecord(
                episode_id="ep-test",
                phase=phase,
                messages=[],
                response="r",
                usage=None,
                timestamp="2026-05-16T00:00:00",
                success=True,
            )

        recorder.record_inference(_record("unlabelled"))
        recorder.set_label("last_workflow_step", "explore")
        recorder.record_inference(_record("p1"))
        recorder.set_label("last_workflow_step", "hypothesise")
        recorder.record_inference(_record("p2"))

        records = recorder.inference_records
        self.assertIsNone(records[0].meta.get("workflow_step"))
        self.assertEqual(records[1].meta.get("workflow_step"), "explore")
        self.assertEqual(records[2].meta.get("workflow_step"), "hypothesise")


if __name__ == "__main__":
    unittest.main()
