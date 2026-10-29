from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.typing.training as training
from src.execution.generation_persistence import (
    GENERATION_COMPLETION_FILENAME,
    GenerationIdentity,
    ensure_generation_manifest,
    load_generation_journal,
    publish_episode_journal,
    validate_resume_checkpoint,
    write_generation_completion,
)
from src.typing.training import save_generation_checkpoint
from src.typing.trajectory import EpisodeTrajectory


def _episode(index: int) -> EpisodeTrajectory:
    return EpisodeTrajectory(
        episode_id=f"ep-{index}",
        generation_id=4,
        episode_index=index,
        prompt_responses=[],
        trajectory={},
        score=0.0,
        episode_runtime_success=False,
        success=False,
        llm_turns_count=0,
        container_variation="v0",
        started_at="2026-07-15T00:00:00",
        completed_at="2026-07-15T00:00:01",
        duration_seconds=1.0,
        task_breakdown={"bootstrap_active": False, "admitted": False},
    )


def _identity(
    *,
    recipe: str = "recipe-a",
    served_artifact_path: str | None = "/models/after_generation_3",
) -> GenerationIdentity:
    return GenerationIdentity(
        generation_id=4,
        run_id="run-a",
        task_name="task-a",
        task_identity="tests.TaskA",
        export_recipe_hash=recipe,
        execution_mode="sequential",
        backend="vllm",
        model_name="vllm:base",
        request_model="adapter",
        served_artifact_kind="lora",
        served_artifact_path=served_artifact_path,
        predecessor_generation_id=3,
        external_lora_attested=False,
    )


def test_manifest_is_create_if_absent_and_mismatch_leaves_bytes_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generation_4" / "generation_manifest.json"
    ensure_generation_manifest(path, _identity())
    original = path.read_bytes()

    with pytest.raises(RuntimeError, match="manifest identity mismatch"):
        ensure_generation_manifest(path, _identity(recipe="recipe-b"))

    assert path.read_bytes() == original


def test_manifest_rejects_serving_lineage_mismatch_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generation_4" / "generation_manifest.json"
    ensure_generation_manifest(path, _identity())
    original = path.read_bytes()

    with pytest.raises(RuntimeError, match="manifest identity mismatch"):
        ensure_generation_manifest(path, _identity(served_artifact_path="/models/other"))

    assert path.read_bytes() == original


def test_checkpoint_rejects_serving_lineage_mismatch() -> None:
    from src.execution.generation_persistence import checkpoint_payload_identity

    checkpoint = {
        **checkpoint_payload_identity(_identity()),
        "committed_episode_indices": [0],
    }

    with pytest.raises(RuntimeError, match="checkpoint identity differs"):
        validate_resume_checkpoint(
            checkpoint=checkpoint,
            journal_indices={0},
            identity=_identity(served_artifact_path="/models/other"),
        )


def test_vllm_local_model_lineage_matches_managed_request_name(tmp_path: Path) -> None:
    from src.execution.generation_persistence import serving_lineage_from_config
    from src.typing.config import AppConfig

    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    config = AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name="vllm:base",
        code_host_cache_path="/tmp/code",
        container_ids=[],
        train_data_save_folder="/tmp/train",
        vllm={"local_model_path": str(model_dir)},
    )

    lineage = serving_lineage_from_config(config, 1)

    assert lineage["request_model"] == "/models/local"
    assert lineage["predecessor_generation_id"] is None

    recursive_lineage = serving_lineage_from_config(
        config,
        1,
        predecessor_generation_id=0,
    )
    assert recursive_lineage["predecessor_generation_id"] == 0


def test_single_external_lora_lineage_requires_runtime_attestation(
    tmp_path: Path,
) -> None:
    from src.execution.generation_persistence import serving_lineage_from_config
    from src.typing.config import AppConfig

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name="vllm:base",
        code_host_cache_path="/tmp/code",
        container_ids=[],
        train_data_save_folder="/tmp/train",
        vllm={
            "endpoint": "https://external.example",
            "lora_adapter_path": str(adapter_dir),
        },
    )

    lineage = serving_lineage_from_config(config, 1)

    assert lineage["request_model"] == "adapter"
    assert lineage["external_lora_attested"] is True


def test_journal_publication_only_appends_missing_episodes(tmp_path: Path) -> None:
    path = tmp_path / "all_episodes.jsonl"
    first = _episode(0)
    second = _episode(1)
    publish_episode_journal([first], path)
    prefix = path.read_bytes()

    publish_episode_journal([first, second], path)

    assert path.read_bytes().startswith(prefix)
    assert [json.loads(line)["episode_index"] for line in path.read_text().splitlines()] == [
        0,
        1,
    ]


def test_empty_publication_creates_durable_journal(tmp_path: Path) -> None:
    path = tmp_path / "all_episodes.jsonl"

    publish_episode_journal([], path)

    assert path.exists()
    assert path.read_bytes() == b""


def test_sparse_sequential_rejected_but_parallel_resumes_at_max_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "all_episodes.jsonl"
    publish_episode_journal([_episode(1), _episode(4)], path)

    with pytest.raises(RuntimeError, match="sparse sequential journal"):
        load_generation_journal(path, generation_id=4, execution_mode="sequential")

    data, next_index = load_generation_journal(
        path,
        generation_id=4,
        execution_mode="parallel",
    )
    assert data.total_episodes_run == 2
    assert next_index == 5


def test_reverse_completion_order_resumes_after_max_index(tmp_path: Path) -> None:
    path = tmp_path / "all_episodes.jsonl"
    publish_episode_journal([_episode(1), _episode(0)], path)

    data, next_index = load_generation_journal(
        path,
        generation_id=4,
        execution_mode="parallel",
    )

    assert [episode.episode_index for episode in data.all_episodes] == [1, 0]
    assert next_index == 2


def test_journal_without_checkpoint_fails_with_preserved_history_guidance(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "all_episodes.jsonl"
    publish_episode_journal([_episode(0)], journal)

    with pytest.raises(RuntimeError, match="history was preserved"):
        validate_resume_checkpoint(
            checkpoint=None,
            journal_indices={0},
            identity=_identity(),
        )


def test_checkpoint_fsyncs_file_before_replace_and_parent_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_fsync = training.os.fsync
    real_replace = training.os.replace

    def fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def replace(source: str | Path, target: str | Path) -> None:
        events.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(training.os, "fsync", fsync)
    monkeypatch.setattr(training.os, "replace", replace)

    save_generation_checkpoint({"generation_id": 4}, tmp_path / "checkpoint.json")

    assert events == ["fsync", "replace", "fsync"]


def test_completion_marker_is_written_last_by_explicit_publication(tmp_path: Path) -> None:
    marker = write_generation_completion(tmp_path, generation_id=4)

    assert marker.name == GENERATION_COMPLETION_FILENAME
    assert json.loads(marker.read_text()) == {
        "schema_version": 2,
        "generation_id": 4,
    }
