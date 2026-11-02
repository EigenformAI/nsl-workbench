from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import uuid
from typing import Any, Callable, Iterable, Literal

from src.typing.training import (
    append_episode_jsonl,
    ensure_durable_file,
    save_durable_json,
)
from src.typing.trajectory import EpisodeTrajectory, GenerationData
from src.typing.config import AppConfig, _is_managed_vllm_endpoint


GENERATION_MANIFEST_FILENAME = "generation_manifest.json"
GENERATION_COMPLETION_FILENAME = "COMPLETED.json"
GENERATION_STATE_SCHEMA_VERSION = 2
ExecutionMode = Literal["sequential", "parallel"]


class JournaledTaskCommitError(RuntimeError):
    """Task apply failed after the episode journal became durable."""


def journal_apply_and_account(
    *,
    append_journal: Callable[[], None],
    apply_task: Callable[[], None],
    account_episode: Callable[[], None],
) -> None:
    """Publish one episode without abandoning a durably journaled commit."""
    append_journal()
    try:
        apply_task()
    except Exception as exc:
        try:
            account_episode()
        except Exception as accounting_exc:
            raise JournaledTaskCommitError(
                f"task apply and episode accounting failed: {exc}; {accounting_exc}"
            ) from accounting_exc
        raise JournaledTaskCommitError(f"task commit apply failed: {exc}") from exc
    try:
        account_episode()
    except Exception as exc:
        raise JournaledTaskCommitError(f"episode accounting failed: {exc}") from exc


@dataclass(frozen=True)
class GenerationIdentity:
    generation_id: int
    run_id: str
    task_name: str
    task_identity: str
    export_recipe_hash: str
    execution_mode: ExecutionMode
    backend: str | None = None
    model_name: str | None = None
    request_model: str | None = None
    served_artifact_kind: str | None = None
    served_artifact_path: str | None = None
    predecessor_generation_id: int | None = None
    external_lora_attested: bool | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": GENERATION_STATE_SCHEMA_VERSION,
            **asdict(self),
        }


def serving_lineage_from_config(
    config: AppConfig | None,
    generation_id: int,
    *,
    request_model: str | None = None,
    external_lora_attested: bool | None = None,
    predecessor_generation_id: int | None = None,
) -> dict[str, Any]:
    if config is None:
        return {
            "backend": None,
            "model_name": None,
            "request_model": request_model,
            "served_artifact_kind": None,
            "served_artifact_path": None,
            "predecessor_generation_id": predecessor_generation_id,
            "external_lora_attested": external_lora_attested,
        }

    backend = config.model_name.strip().split(":", 1)[0]
    artifact_kind = None
    artifact_path = None
    if backend == "vllm" and config.vllm is not None:
        if config.vllm.lora_adapter_path:
            artifact_kind = "lora"
            artifact_path = config.vllm.lora_adapter_path
        elif config.vllm.local_model_path:
            artifact_kind = "merged_16bit"
            artifact_path = config.vllm.local_model_path
        if request_model is None:
            if config.vllm.lora_adapter_path:
                request_model = config.vllm.external_request_model or "adapter"
            elif config.vllm.served_model_name:
                request_model = config.vllm.served_model_name
            elif config.vllm.local_model_path:
                request_model = "/models/local"
    elif backend == "llama":
        if config.llama is not None and config.llama.lora_adapter_path:
            artifact_kind = "lora"
            artifact_path = config.llama.lora_adapter_path
        else:
            source = config.model_name.split(":", 1)[1].strip()
            source_path = Path(source).expanduser()
            if source.lower().endswith(".gguf") and source_path.exists():
                artifact_kind = "gguf"
                artifact_path = source
        if request_model is None:
            request_model = config.model_name.split(":", 1)[1].strip()

    normalized_artifact_path = (
        str(Path(artifact_path).expanduser().resolve()) if artifact_path else None
    )
    if request_model is None:
        request_model = config.model_name.split(":", 1)[-1].strip()
    if external_lora_attested is None and backend == "vllm":
        external_vllm = config.vllm is not None and (
            bool(config.vllm.endpoints)
            or (
                config.vllm.endpoint is not None
                and not _is_managed_vllm_endpoint(config.vllm.endpoint)
            )
        )
        external_lora_attested = (
            True
            if config.vllm is not None
            and external_vllm
            and config.vllm.lora_adapter_path
            else None
        )
    return {
        "backend": backend,
        "model_name": config.model_name,
        "request_model": request_model,
        "served_artifact_kind": artifact_kind,
        "served_artifact_path": normalized_artifact_path,
        "predecessor_generation_id": predecessor_generation_id,
        "external_lora_attested": external_lora_attested,
    }


def validate_manifest_serving_lineage(
    path: Path,
    expected_lineage: dict[str, Any],
) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"generation manifest is unreadable: {path}") from exc
    actual = {key: manifest.get(key) for key in expected_lineage}
    if actual != expected_lineage:
        raise RuntimeError(
            f"generation manifest serving lineage mismatch at {path}: "
            f"existing={actual!r}, requested={expected_lineage!r}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_generation_manifest(
    path: Path,
    identity: GenerationIdentity,
) -> bool:
    """Create an immutable manifest, or validate the existing bytes."""
    expected_dir_name = f"generation_{identity.generation_id}"
    if path.name != GENERATION_MANIFEST_FILENAME or path.parent.name != expected_dir_name:
        raise RuntimeError(
            "generation manifest path does not match its generation identity: "
            f"{path} != */{expected_dir_name}/{GENERATION_MANIFEST_FILENAME}"
        )
    expected = identity.to_manifest()
    if path.exists():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"generation manifest is unreadable: {path}") from exc
        if actual != expected:
            raise RuntimeError(
                f"generation manifest identity mismatch at {path}: "
                f"existing={actual!r}, requested={expected!r}"
            )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            return ensure_generation_manifest(path, identity)
        _fsync_directory(path.parent)
        return True
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_generation_journal(
    path: Path,
    *,
    generation_id: int,
    execution_mode: ExecutionMode,
) -> tuple[GenerationData, int]:
    data = GenerationData(generation_id=generation_id)
    if not path.exists():
        return data, 0

    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episode = EpisodeTrajectory.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"malformed generation journal record at line {line_number}: {path}"
                ) from exc
            if episode.generation_id != generation_id:
                raise RuntimeError(
                    f"journal generation identity mismatch: {episode.generation_id} "
                    f"!= {generation_id}"
                )
            if episode.episode_id in seen_ids:
                raise RuntimeError(f"duplicate episode id in journal: {episode.episode_id}")
            if episode.episode_index in seen_indices:
                raise RuntimeError(
                    f"duplicate episode index in journal: {episode.episode_index}"
                )
            seen_ids.add(episode.episode_id)
            seen_indices.add(episode.episode_index)
            data.add_episode(episode)

    next_index = max(seen_indices, default=-1) + 1
    if execution_mode == "sequential" and seen_indices != set(range(next_index)):
        raise RuntimeError(
            f"sparse sequential journal is corrupt: indices={sorted(seen_indices)}"
        )
    return data, next_index


def publish_episode_journal(
    episodes: Iterable[EpisodeTrajectory],
    path: Path,
) -> None:
    desired = list(episodes)
    ensure_durable_file(path)
    existing_by_index: dict[int, dict[str, Any]] = {}
    existing_ids: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    index = int(payload["episode_index"])
                    episode_id = str(payload["episode_id"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"malformed generation journal record at line {line_number}: {path}"
                    ) from exc
                if index in existing_by_index or episode_id in existing_ids:
                    raise RuntimeError("duplicate episode identity in generation journal")
                existing_by_index[index] = payload
                existing_ids.add(episode_id)

    desired_indices = {episode.episode_index for episode in desired}
    if not set(existing_by_index).issubset(desired_indices):
        raise RuntimeError("generation journal contains episodes outside publication snapshot")
    for episode in desired:
        payload = json.loads(json.dumps(episode.to_dict(), default=str))
        existing = existing_by_index.get(episode.episode_index)
        if existing is not None:
            if existing != payload:
                raise RuntimeError(
                    f"journal episode {episode.episode_index} differs from committed snapshot"
                )
            continue
        if episode.episode_id in existing_ids:
            raise RuntimeError(f"duplicate episode id in publication: {episode.episode_id}")
        append_episode_jsonl(payload, path)
        existing_by_index[episode.episode_index] = payload
        existing_ids.add(episode.episode_id)


def task_commit_intents(
    episodes: Iterable[EpisodeTrajectory],
) -> dict[str, str]:
    """Return validated opaque task intents from durable episode records."""
    intents: dict[str, str] = {}
    for episode in episodes:
        token = episode.task_commit_token
        digest = episode.task_commit_digest
        if (token is None) != (digest is None):
            raise RuntimeError(
                f"episode {episode.episode_id} has incomplete task commit metadata"
            )
        if token is None:
            continue
        if not isinstance(token, str) or not isinstance(digest, str):
            raise RuntimeError(
                f"episode {episode.episode_id} has non-string task commit metadata"
            )
        if not token or not digest:
            raise RuntimeError(
                f"episode {episode.episode_id} has empty task commit metadata"
            )
        if token in intents:
            raise RuntimeError(f"duplicate task commit token in journal: {token}")
        intents[token] = digest
    return intents


def checkpoint_payload_identity(identity: GenerationIdentity) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": GENERATION_STATE_SCHEMA_VERSION,
        **asdict(identity),
    }


def validate_resume_checkpoint(
    *,
    checkpoint: dict[str, Any] | None,
    journal_indices: set[int],
    identity: GenerationIdentity,
    allow_journal_ahead: bool = False,
) -> None:
    if checkpoint is None:
        if allow_journal_ahead:
            return
        raise RuntimeError(
            "generation journal exists without a matching checkpoint; automatic "
            "task-state replay is unavailable, so history was preserved for manual recovery"
        )
    expected_identity = checkpoint_payload_identity(identity)
    actual_identity = {key: checkpoint.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise RuntimeError(
            "checkpoint identity differs from generation manifest: "
            f"{actual_identity!r} != {expected_identity!r}"
        )
    try:
        raw_checkpoint_indices = [
            int(value) for value in checkpoint["committed_episode_indices"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint has invalid committed episode indices") from exc
    checkpoint_indices = set(raw_checkpoint_indices)
    if len(checkpoint_indices) != len(raw_checkpoint_indices):
        raise RuntimeError("checkpoint contains duplicate committed episode indices")
    indices_match = checkpoint_indices == journal_indices
    checkpoint_safely_trails = (
        allow_journal_ahead and checkpoint_indices < journal_indices
    )
    if not indices_match and not checkpoint_safely_trails:
        relation = "trails" if checkpoint_indices < journal_indices else "does not match"
        raise RuntimeError(
            f"checkpoint {relation} durable journal; history was preserved for manual recovery"
        )


def write_generation_completion(generation_dir: Path, generation_id: int) -> Path:
    path = generation_dir / GENERATION_COMPLETION_FILENAME
    save_durable_json(
        {
            "schema_version": GENERATION_STATE_SCHEMA_VERSION,
            "generation_id": generation_id,
        },
        path,
    )
    return path


__all__ = [
    "GENERATION_COMPLETION_FILENAME",
    "GENERATION_MANIFEST_FILENAME",
    "GENERATION_STATE_SCHEMA_VERSION",
    "GenerationIdentity",
    "checkpoint_payload_identity",
    "ensure_generation_manifest",
    "load_generation_journal",
    "publish_episode_journal",
    "task_commit_intents",
    "serving_lineage_from_config",
    "validate_manifest_serving_lineage",
    "validate_resume_checkpoint",
    "write_generation_completion",
]
