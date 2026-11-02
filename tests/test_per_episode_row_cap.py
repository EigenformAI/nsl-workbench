"""Per-episode selection and composition with the existing export pipeline."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.training_data import (
    EpisodeTrainingRows,
    PerEpisodeRowCap,
    TrainingDataExportContext,
    build_export_recipe,
    build_training_export,
    validate_training_row_groups,
)


def _context(**overrides):
    return replace(
        TrainingDataExportContext(7, "run", "test", Path("."), Path("all.jsonl"), ""),
        **overrides,
    )


def _group(count, episode_id="ep-0", episode_index=0):
    return EpisodeTrainingRows(
        episode_id, episode_index, 7, 1.0,
        [
            {
                "row_id": f"{episode_id}:{index}",
                "parent_row_id": None,
                "prompt": f"prompt-{index}",
                "raw_response": f"response-{index}",
                "interaction_type": "orchestrator",
                "source_interaction_type": "orchestrator",
                "timestamp": "2026-08-04T00:00:00Z",
                "success": True,
                "error_message": None,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "generation_id": 7,
                "episode_score": 1.0,
                "episode_score_scope": "whole_episode",
                "source_episode_id": episode_id,
                "source_row_index": index,
                "workflow_step": "execute",
                "actor_role": None,
                "record_meta": {},
            }
            for index in range(count)
        ],
    )


class _Generation:
    def __init__(self, groups):
        self.groups = groups

    def get_successful_training_row_groups(self):
        return self.groups


def _canonical_digest(groups):
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
        for group in groups for row in group.rows
    )
    return hashlib.sha256(payload).hexdigest()


def _expected_random_ids(group, digest, cap):
    def rank(row):
        parts = (cap.namespace, str(cap.seed), digest, group.episode_id, row["row_id"])
        return hashlib.sha256(b"\0".join(part.encode() for part in parts)).hexdigest()

    selected = sorted(group.rows, key=lambda row: (rank(row), row["row_id"]))[:cap.max_rows]
    return [row["row_id"] for row in sorted(selected, key=lambda row: (row["source_row_index"], row["row_id"]))]


def test_head_cap_keeps_only_the_first_source_rows():
    context = TrainingDataExportContext(1, "run", "test", Path("."), Path("all.jsonl"), "")
    group = EpisodeTrainingRows(
        "ep-0", 0, 1, 1.0,
        [{"row_id": f"r{i}", "source_row_index": i, "parent_row_id": None} for i in range(4)],
    )
    selected = PerEpisodeRowCap(2, keep="head").transform_export_rows(context, [group])

    assert [row["row_id"] for row in selected[0].rows] == ["r0", "r1"]
    assert len(group.rows) == 4


@pytest.mark.parametrize(
    ("keep", "max_rows", "expected"),
    [("head", 3, [0, 1, 2]), ("tail", 3, [4, 5, 6]),
     ("even", 3, [0, 3, 6]), ("even", 1, [0]), ("even", 5, [0, 2, 3, 5, 6])],
)
def test_ordered_strategies_use_source_order(keep, max_rows, expected):
    group = _group(7)
    group.rows.reverse()
    result = PerEpisodeRowCap(max_rows, keep=keep).transform_export_rows(_context(), [group])
    assert [row["source_row_index"] for row in result[0].rows] == expected


def test_groups_within_limit_and_empty_groups_are_preserved():
    groups = [_group(0), _group(2, "ep-1", 1), _group(5, "ep-2", 2)]
    original = copy.deepcopy(groups)
    result = PerEpisodeRowCap(2, keep="head").transform_export_rows(_context(), groups)
    assert [len(group.rows) for group in result] == [0, 2, 2]
    assert result[:2] == original[:2]
    assert groups == original
    validate_training_row_groups(result)


def test_random_known_answer_uses_canonical_candidate_digest_and_source_order():
    groups = [_group(4), _group(1, "ep-1", 1)]
    digest = _canonical_digest(groups)
    assert digest == "1f6bf47a118ac74f98c8c44698943aa863a7bd1ee3b4aa3f0e3aa9ce9721a00e"
    cap = PerEpisodeRowCap(2)
    export = build_training_export(_Generation(groups), (cap,), _context())
    expected = _expected_random_ids(groups[0], digest, cap) + ["ep-1:0"]
    assert expected == ["ep-0:1", "ep-0:3", "ep-1:0"]

    assert [row["row_id"] for row in export.rows] == expected
    assert export.report["raw_successful_row_count"] == 5
    assert export.report["training_row_count"] == 3
    assert export.report["per_transform"][0]["rows_removed"] == 2
    assert expected[:2] != ["ep-0:0", "ep-0:1"]


@pytest.mark.parametrize(("seed", "namespace"), [(0, "episode-row-cap-v1"), (17, "custom-cap"), (-3, "unicode-λ")])
def test_random_is_repeatable_nonmutating_and_uses_all_selection_overrides(seed, namespace):
    groups = [_group(20), _group(20, "ep-1", 1)]
    original = copy.deepcopy(groups)
    cap = PerEpisodeRowCap(4, seed=seed, namespace=namespace)
    first = cap.transform_export_rows(_context(), groups)
    second = cap.transform_export_rows(_context(), groups)
    assert first == second
    assert groups == original
    assert [[row["row_id"] for row in group.rows] for group in first] == [
        _expected_random_ids(group, _canonical_digest(groups), cap) for group in groups
    ]


def test_parent_links_retain_provenance_only_when_the_parent_is_dropped():
    group = _group(4)
    group.rows[2]["parent_row_id"] = "ep-0:0"
    group.rows[3]["parent_row_id"] = "ep-0:2"
    selected = PerEpisodeRowCap(2, keep="tail").transform_export_rows(_context(), [group])
    assert selected[0].rows[0]["parent_row_id"] is None
    assert selected[0].rows[0]["source_parent_row_id"] == "ep-0:0"
    assert selected[0].rows[1]["parent_row_id"] == "ep-0:2"
    assert "source_parent_row_id" not in selected[0].rows[1]
    assert group.rows[2]["parent_row_id"] == "ep-0:0"
    validate_training_row_groups(selected)


class _FilterAndRewrite:
    name = "FilterAndRewrite[v1]"

    def __init__(self):
        self.seen_digest = None

    def config(self):
        return {}

    def transform_export_rows(self, context, groups):
        self.seen_digest = context.raw_generation_digest
        return [replace(group, rows=[
            {**row, "prompt": "rewritten"}
            for row in group.rows if row["source_row_index"] % 2
        ]) for group in groups]


@pytest.mark.parametrize("unicode_content", [False, True])
def test_cap_after_filter_uses_pretransform_digest_not_filtered_or_rewritten_rows(unicode_content):
    groups = [_group(16)]
    if unicode_content:
        groups[0].rows[0]["prompt"] = "adsorção λ\n"
    original = copy.deepcopy(groups)
    filtering = _FilterAndRewrite()
    cap = PerEpisodeRowCap(3, seed=5)
    context = _context(raw_generation_digest="must-not-use-caller-digest")
    raw_digest = _canonical_digest(groups)
    filtered = filtering.transform_export_rows(context, groups)
    expected = _expected_random_ids(filtered[0], raw_digest, cap)
    wrong = _expected_random_ids(filtered[0], _canonical_digest(filtered), cap)
    assert expected != wrong, "fixture must reject rehashing transformed input"

    export = build_training_export(_Generation(groups), (filtering, cap), context)

    assert filtering.seen_digest == raw_digest
    assert [row["row_id"] for row in export.rows] == expected
    assert all(row["prompt"] == "rewritten" for row in export.rows)
    assert groups == original
    assert context.raw_generation_digest == "must-not-use-caller-digest"
    assert [item["rows_out"] for item in export.report["per_transform"]] == [8, 3]


def test_standalone_context_digest_takes_precedence_over_input_fallback():
    group = _group(20)
    cap = PerEpisodeRowCap(4)
    digest = "a" * 64
    expected = _expected_random_ids(group, digest, cap)
    assert expected != _expected_random_ids(group, _canonical_digest([group]), cap)
    result = cap.transform_export_rows(_context(raw_generation_digest=digest), [group])
    assert [row["row_id"] for row in result[0].rows] == expected


@pytest.mark.parametrize("override", [{"max_rows": 3}, {"keep": "tail"}, {"seed": 7}, {"namespace": "different"}])
def test_recipe_hash_changes_for_every_selection_override(override):
    baseline = PerEpisodeRowCap(2)
    changed = replace(baseline, **override)
    assert build_export_recipe((baseline,)).recipe_hash != build_export_recipe((changed,)).recipe_hash


def test_recipe_carries_algorithm_and_namespace_identity():
    recipe = build_export_recipe((PerEpisodeRowCap(2),))
    assert recipe.transforms[0]["selection_recipe"] == {
        "algorithm": "sha256-rank-v1", "K": 2,
        "transform_seed": 0, "namespace": "episode-row-cap-v1",
    }


@pytest.mark.parametrize("override", [
    {"max_rows": True}, {"max_rows": 0}, {"max_rows": -1}, {"max_rows": 1.5},
    {"keep": "unknown"}, {"seed": True}, {"seed": 1.5},
    {"algorithm": "unknown"}, {"namespace": ""}, {"namespace": None}, {"namespace": "a\0b"},
])
def test_invalid_selection_config_is_rejected(override):
    with pytest.raises(ValueError):
        PerEpisodeRowCap(**{"max_rows": 2, **override})


@pytest.mark.parametrize("field", ["episode_id", "row_id"])
def test_rank_identity_rejects_ambiguous_nul_delimiters(field):
    group = _group(1)
    if field == "episode_id":
        group.episode_id = "bad\0episode"
    else:
        group.rows[0]["row_id"] = "bad\0row"
    with pytest.raises(ValueError, match="NUL"):
        PerEpisodeRowCap(1).transform_export_rows(_context(), [group])
