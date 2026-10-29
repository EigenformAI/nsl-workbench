from __future__ import annotations

# Environment variables are loaded before importing backend modules.
# ruff: noqa: E402

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import docker
import pydantic
import toml
from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv()

from src.execution import open_backend_runtime, run_generation, save_generation_data
from src.execution.generation_persistence import (
    GENERATION_COMPLETION_FILENAME,
    GENERATION_MANIFEST_FILENAME,
    GENERATION_STATE_SCHEMA_VERSION,
    serving_lineage_from_config,
    validate_manifest_serving_lineage,
)
from src.backend.utils import is_http_ready
from src.backend.vllm import wait_for_gpu_memory_release
from src.harness.loader import resolve_harness_class
from src.harness.provisioning import ensure_configured_harness
from src.helper import generate_readable_run_id, unflatten_toml_dict, validate_run_id
from src.observability import MetricsCollector
from src.task import TaskSpec, load_task
from src.train.artifacts import (
    resolve_training_export_format as _resolve_training_export_format,
    training_artifact_path as _training_artifact_path,
    training_metadata_path,
    validate_training_artifact,
)
from src.training_data.transforms import resolve_latest_sft_training_rows_path
from src.typing.config import AppConfig
from src.typing.trajectory import GenerationData

RUN_FILENAME = "run.json"
GENERATION_METADATA_FILENAME = "metadata.json"
TRAINING_INFO_FILENAME = "training_info.json"
DEFAULT_VLLM_MODELS_URL = "http://127.0.0.1:8000/v1/models"


class CompletedGenerationPublicationError(RuntimeError):
    """A completion marker exists but its consumed publication is invalid."""


def _apply_training_artifact(
    generation_config: AppConfig,
    artifact_path: Path | None,
    export_format: str,
) -> None:
    if artifact_path is None:
        return

    backend = generation_config.model_name.strip().split(":", 1)[0]
    if backend == "vllm":
        if generation_config.vllm is None:
            generation_config.vllm = AppConfig.VllmConfig()

        if export_format == "lora":
            generation_config.vllm.lora_adapter_path = str(artifact_path)
            return

        if export_format == "merged_16bit":
            generation_config.vllm.lora_adapter_path = None
            generation_config.vllm.local_model_path = str(artifact_path)
            return

        raise ValueError("vLLM training loop does not consume GGUF artifacts")

    if backend == "llama":
        if generation_config.llama is None:
            generation_config.llama = AppConfig.LlamaConfig()

        if export_format == "lora":
            generation_config.llama.lora_adapter_path = str(artifact_path)
            return

        if export_format == "gguf":
            generation_config.llama.lora_adapter_path = None
            generation_config.model_name = f"llama:{artifact_path}"
            return

        raise ValueError("llama backend only supports LoRA adapters or GGUF artifacts")

    raise ValueError(
        f"Training loop does not know how to apply '{export_format}' artifacts to backend '{backend}'"
    )


def _served_adapter_path(config: AppConfig) -> str | None:
    if config.vllm is not None:
        return config.vllm.lora_adapter_path
    if config.llama is not None:
        return config.llama.lora_adapter_path
    return None


def _served_training_artifact_path(config: AppConfig) -> str | None:
    backend = config.model_name.strip().split(":", 1)[0]
    if backend == "vllm" and config.vllm is not None:
        return config.vllm.lora_adapter_path or config.vllm.local_model_path
    if (
        backend == "llama"
        and config.llama is not None
        and config.llama.lora_adapter_path
    ):
        return config.llama.lora_adapter_path
    if backend == "llama" and config.model_name.startswith("llama:"):
        return config.model_name.split(":", 1)[1].strip()
    return None


def _collect_training_window_paths(
    generation_root: Path,
    end_generation_id: int,
    window_size: int,
) -> list[Path]:
    if window_size < 1:
        raise ValueError("training window size must be positive")

    start_generation_id = max(0, end_generation_id - window_size + 1)
    training_paths: list[Path] = []
    for generation_id in range(start_generation_id, end_generation_id + 1):
        generation_dir = generation_root / f"generation_{generation_id}"
        training_paths.append(resolve_latest_sft_training_rows_path(generation_dir))
    return training_paths


def _run_doc_path(generation_root: Path) -> Path:
    return generation_root / RUN_FILENAME


@contextmanager
def _generation_run_lock(
    generation_root: Path,
    run_id: str,
) -> Iterator[None]:
    generation_root.mkdir(parents=True, exist_ok=True)
    lock_path = generation_root / ".run.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Training run {run_id!r} is already active; lock held at {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _generation_phase_complete(
    generation_root: Path,
    generation_id: int,
    *,
    expected_lineage: dict[str, Any] | None = None,
) -> bool:
    """Complete only after final generation publication writes its marker."""
    generation_dir = generation_root / f"generation_{generation_id}"
    marker_path = generation_dir / GENERATION_COMPLETION_FILENAME
    if not marker_path.exists():
        return False

    def _read_object(path: Path, artifact: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompletedGenerationPublicationError(
                f"completed generation {generation_id} has unreadable {artifact}: {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise CompletedGenerationPublicationError(
                f"completed generation {generation_id} has non-object {artifact}: {path}"
            )
        return payload

    try:
        marker = _read_object(marker_path, "completion marker")
        if marker.get("schema_version") != GENERATION_STATE_SCHEMA_VERSION:
            raise CompletedGenerationPublicationError(
                f"completed generation {generation_id} has unsupported marker schema: "
                f"{marker_path}"
            )
        if marker.get("generation_id") != generation_id:
            raise CompletedGenerationPublicationError(
                f"completed generation ID mismatch in {marker_path}"
            )

        if expected_lineage is not None:
            validate_manifest_serving_lineage(
                generation_dir / GENERATION_MANIFEST_FILENAME,
                expected_lineage,
            )

        metadata_path = generation_dir / GENERATION_METADATA_FILENAME
        metadata = _read_object(metadata_path, "metadata")
        if metadata.get("generation_id") != generation_id:
            raise CompletedGenerationPublicationError(
                f"completed generation ID mismatch in {metadata_path}"
            )

        latest_path = generation_dir / "exports" / "sft" / "latest.json"
        latest = _read_object(latest_path, "latest SFT pointer")
        rows_path = resolve_latest_sft_training_rows_path(generation_dir)
        if not rows_path.is_file():
            raise CompletedGenerationPublicationError(
                f"completed generation {generation_id} has no training rows file: "
                f"{rows_path}"
            )
        metadata_export_id = metadata.get("active_sft_export_id")
        latest_export_id = latest.get("export_id")
        if (
            metadata_export_id is not None
            and latest_export_id is not None
            and metadata_export_id != latest_export_id
        ):
            raise CompletedGenerationPublicationError(
                f"completed generation {generation_id} has inconsistent active SFT "
                f"export IDs in {metadata_path} and {latest_path}"
            )
    except CompletedGenerationPublicationError:
        logger.error(f"Corrupt completed generation publication: {generation_dir}")
        raise
    except Exception as exc:
        logger.error(f"Corrupt completed generation publication: {generation_dir}")
        raise CompletedGenerationPublicationError(
            f"completed generation {generation_id} has invalid publication artifacts "
            f"under {generation_dir}"
        ) from exc
    return True


def _load_generation_metadata(
    generation_root: Path, generation_id: int
) -> dict[str, Any]:
    metadata_path = (
        generation_root / f"generation_{generation_id}" / GENERATION_METADATA_FILENAME
    )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _training_artifact_complete(artifact_path: Path, export_format: str) -> bool:
    metadata_path = training_metadata_path(artifact_path, export_format)
    if not metadata_path.exists():
        return False
    validate_training_artifact(
        artifact_path,
        export_format,
        expected_path=artifact_path,
    )
    return True


def _vllm_server_is_ready(models_url: str = DEFAULT_VLLM_MODELS_URL) -> bool:
    return is_http_ready(models_url)


def _invoke_train_sft(
    config: AppConfig,
    *,
    training_paths: Sequence[Path],
    output_dir: Path,
    export_format: str,
) -> Path:
    """Run training as a child process so its CUDA memory is reclaimed on exit.

    In-process training leaves torch's CUDA context and caching allocator
    attached for the life of the orchestrator; the next generation's vLLM
    container then sees a busy GPU. A subprocess returns the memory to the
    driver when it exits.
    """
    assert config.training is not None, "training config is required"
    training = config.training

    cmd = [
        sys.executable,
        "-m",
        "src.train.qlora",
        "--base-model",
        training.base_model,
        "--output",
        str(output_dir),
        "--max-seq-length",
        str(training.max_seq_length),
        "--max-steps",
        str(training.max_steps),
        "--num-train-epochs",
        str(training.num_train_epochs),
        "--per-device-train-batch-size",
        str(training.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(training.gradient_accumulation_steps),
        "--learning-rate",
        str(training.learning_rate),
        "--warmup-steps",
        str(training.warmup_steps),
        "--warmup-ratio",
        str(training.warmup_ratio),
        "--lr-scheduler-type",
        training.lr_scheduler_type,
        "--weight-decay",
        str(training.weight_decay),
        "--inner-loss",
        training.inner_loss,
        "--dft-impl",
        training.dft_impl,
        "--lora-rank",
        str(training.lora_rank),
        "--lora-alpha",
        str(training.lora_alpha),
        "--lora-dropout",
        str(training.lora_dropout),
        "--seed",
        str(training.seed),
        "--export-format",
        export_format,
        "--quantize",
        training.gguf_quantize,
    ]
    if training.rehearsal_dataset:
        cmd += ["--rehearsal-dataset", training.rehearsal_dataset]
        cmd += ["--rehearsal-split", training.rehearsal_split]
        cmd += ["--rehearsal-text-field", training.rehearsal_text_field]
        cmd += [
            "--rehearsal-rows-per-epoch",
            str(training.rehearsal_rows_per_epoch),
        ]
        cmd += ["--rehearsal-prompt", training.rehearsal_prompt]
        cmd += ["--rehearsal-prompt-chars", str(training.rehearsal_prompt_chars)]
        cmd += ["--rehearsal-max-chars", str(training.rehearsal_max_chars)]
        if training.rehearsal_seed is not None:
            cmd += ["--rehearsal-seed", str(training.rehearsal_seed)]
    if training.wandb_project:
        cmd += ["--wandb-project", training.wandb_project]
    if training.inner_loss == "asft":
        cmd += ["--asft-mode", training.asft_mode]
        cmd += ["--asft-kl-weight", str(training.asft_kl_weight)]
        cmd += ["--asft-kl-direction", training.asft_kl_direction]
        cmd += ["--asft-reference-policy", training.asft_reference_policy]
        cmd += ["--asft-streaming", training.asft_streaming]
        if training.asft_ref_microbatch_size is not None:
            cmd += [
                "--asft-ref-microbatch-size",
                str(training.asft_ref_microbatch_size),
            ]
        if training.asft_seq_chunk_size is not None:
            cmd += ["--asft-seq-chunk-size", str(training.asft_seq_chunk_size)]
        cmd += ["--asft-normalize-by", training.asft_normalize_by]
    for path in training_paths:
        cmd += ["--training-data", str(path)]

    logger.info("Launching training subprocess: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )

    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        stdout_lines.append(line)
    return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Training subprocess exited with status {return_code}. "
            f"See above for details."
        )

    for line in reversed(stdout_lines):
        stripped = line.strip()
        if stripped:
            return Path(stripped)
    raise RuntimeError("Training subprocess produced no output; cannot locate artifact")


def _should_wait_for_vllm_gpu_release(
    config: AppConfig,
    generation_id: int,
) -> bool:
    if config.training is None or config.orchestration is None:
        return False

    if generation_id >= config.orchestration.num_generations - 1:
        return False

    if config.training.gpu_wait_timeout_seconds <= 0:
        return False

    backend = config.model_name.strip().split(":", 1)[0]
    if backend != "vllm":
        return False

    return not _vllm_server_is_ready()


def _write_run_doc(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = re.sub(r"[^a-z0-9]+", "-", item.strip().lower()).strip("-")
        if not tag or tag in seen:
            continue
        normalized.append(tag)
        seen.add(tag)
    return normalized


def _git_short_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    commit_id = result.stdout.strip()
    return commit_id or None


def _git_dirty() -> bool | None:
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if unstaged.returncode not in (0, 1) or staged.returncode not in (0, 1):
        return None
    return unstaged.returncode == 1 or staged.returncode == 1


def detect_hardware_tags() -> list[str]:
    from src.observability.gpu import detect_hardware_tags as _detect_hardware_tags

    return _detect_hardware_tags()


def run_generation_phase(
    config: AppConfig,
    generation_id: int,
    run_id: str,
    docker_client: Any,
    task: TaskSpec,
    metrics_collector: MetricsCollector | None = None,
    predecessor_generation_id: int | None = None,
) -> tuple[Path, GenerationData]:
    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()

    served_model_name = (
        config.vllm.served_model_name
        if config.vllm and config.vllm.served_model_name
        else config.model_name
    )
    served_adapter_dir = _served_adapter_path(config)
    logger.info(
        f"Starting generation {generation_id} with model={served_model_name} "
        f"adapter={served_adapter_dir or '<base>'}"
    )

    with open_backend_runtime(
        config,
        run_id=run_id,
        docker_client=docker_client,
        task=task,
        metrics_collector=metrics_collector,
        predecessor_generation_id=predecessor_generation_id,
    ) as runtime:
        generation_data = run_generation(runtime, generation_id=generation_id)
        generation_dir = save_generation_data(
            generation_data,
            Path(config.generation.generation_output_dir),
            runtime.run_id,
            runtime.task,
            runtime.config,
            request_model=runtime.request_model,
            external_lora_attested=runtime.external_lora_attested,
            predecessor_generation_id=runtime.predecessor_generation_id,
        )

    return generation_dir, generation_data


def _run_loop_locked(
    config: AppConfig,
    *,
    run_id: str | None = None,
    docker_client: Any = None,
    task: TaskSpec,
    metrics_collector: MetricsCollector | None = None,
    config_path: str | None = None,
    cli_hardware_tags: Sequence[str] | None = None,
    cli_load_tags: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()
    if config.training is None:
        config.training = AppConfig.TrainingConfig()
    if config.orchestration is None:
        config.orchestration = AppConfig.OrchestrationConfig()

    if config.orchestration.num_generations < 1:
        raise ValueError("num_generations must be at least 1")
    if config.orchestration.training_window_size < 1:
        raise ValueError("training_window_size must be at least 1")

    active_run_id = run_id or generate_readable_run_id()
    active_docker_client = (
        docker_client if docker_client is not None else docker.from_env()
    )
    generation_root = Path(config.generation.generation_output_dir) / active_run_id
    adapter_root = Path(config.training.adapter_output_dir) / active_run_id
    training_export_format = _resolve_training_export_format(
        config.model_name,
        config.training.export_format,
    )
    loop_started_at = time.perf_counter()

    latest_artifact_path: Path | None = None
    results: list[dict[str, Any]] = []
    run_doc_path = _run_doc_path(generation_root)

    hardware_tags: list[str] = []
    if config.observability.detect_hardware:
        hardware_tags.extend(detect_hardware_tags())
    hardware_tags = _normalize_tags(
        [
            *hardware_tags,
            *config.observability.hardware_tags,
            *(cli_hardware_tags or []),
        ]
    )
    load_tags = _normalize_tags(
        [*config.observability.load_tags, *(cli_load_tags or [])]
    )

    harness_class = resolve_harness_class(config.harness)
    harness_profile = None
    if config.harness.name == "container" and config.harness.container is not None:
        harness_profile = config.harness.container.profile
    task_name_raw = getattr(task, "name", None)
    task_name = task_name_raw if isinstance(task_name_raw, str) else "unknown"
    started_at = _utc_now_iso()

    identity_fields = {
        "run_id": active_run_id,
        "started_at": started_at,
        "harness_type": config.harness.name,
        "harness_profile": harness_profile,
        "harness_class": f"{harness_class.__module__}.{harness_class.__qualname__}",
        "task_class": config.task.class_,
        "task_name": task_name,
        "config_path": config_path,
        "model_name": config.model_name,
        "commit_id": _git_short_sha(),
        "git_dirty": _git_dirty(),
        "hardware_tags": hardware_tags,
        "load_tags": load_tags,
    }

    def _build_run_doc(
        status: str,
        *,
        ended_at: str | None,
        total_wall_clock_seconds: float | None = None,
    ) -> dict[str, Any]:
        if total_wall_clock_seconds is None:
            total_wall_clock_seconds = time.perf_counter() - loop_started_at
        return {
            **identity_fields,
            "status": status,
            "ended_at": ended_at,
            "num_generations": config.orchestration.num_generations,
            "training_window_size": config.orchestration.training_window_size,
            "training_export_format": training_export_format,
            "total_wall_clock_seconds": total_wall_clock_seconds,
            "total_agent_seconds": sum(
                float(result["metrics"].get("total_agent_seconds", 0.0))
                for result in results
                if isinstance(result.get("metrics"), dict)
            ),
            "total_tokens": sum(
                int(result["metrics"].get("total_tokens", 0))
                for result in results
                if isinstance(result.get("metrics"), dict)
            ),
            "generations": results,
        }

    def _try_write_run_doc(status: str, *, ended_at: str | None = None) -> None:
        try:
            _write_run_doc(run_doc_path, _build_run_doc(status, ended_at=ended_at))
            logger.info(f"run.json status={status} -> {run_doc_path}")
        except Exception:
            logger.exception(
                f"Failed to write run.json status={status} path={run_doc_path}"
            )

    _write_run_doc(
        run_doc_path,
        _build_run_doc(
            "in_progress",
            ended_at=None,
            total_wall_clock_seconds=0.0,
        ),
    )
    logger.info(f"run.json status=in_progress -> {run_doc_path}")

    num_gens = config.orchestration.num_generations
    try:
        for generation_id in range(num_gens):
            logger.info(
                f"{'=' * 60}\n"
                f"  Generation {generation_id + 1}/{num_gens} -- {active_run_id}\n"
                f"{'=' * 60}"
            )
            generation_config = config.model_copy(deep=True)
            assert generation_config.generation is not None
            generation_config.generation.generation_output_dir = str(generation_root)
            generation_task = task
            resolve_task_config = getattr(
                type(task), "resolve_generation_task_config", None
            )
            if callable(resolve_task_config):
                generation_config.task.config = resolve_task_config(
                    generation_config.task.config,
                    generation_id,
                    num_gens,
                )
                generation_task = load_task(
                    generation_config.task.class_, generation_config.task.config
                )
            if latest_artifact_path is not None:
                expected_served_artifact = _training_artifact_path(
                    adapter_root,
                    generation_id - 1,
                    training_export_format,
                )
                latest_artifact_path = validate_training_artifact(
                    latest_artifact_path,
                    training_export_format,
                    expected_path=expected_served_artifact,
                )
            _apply_training_artifact(
                generation_config,
                latest_artifact_path,
                training_export_format,
            )
            predecessor_generation_id = (
                generation_id - 1 if latest_artifact_path is not None else None
            )
            should_wait_for_gpu_release = _should_wait_for_vllm_gpu_release(
                generation_config,
                generation_id,
            )

            expected_lineage = serving_lineage_from_config(
                generation_config,
                generation_id,
                predecessor_generation_id=predecessor_generation_id,
            )
            if _generation_phase_complete(
                generation_root,
                generation_id,
                expected_lineage=expected_lineage,
            ):
                generation_dir = generation_root / f"generation_{generation_id}"
                generation_metrics = _load_generation_metadata(
                    generation_root, generation_id
                )
                logger.info(
                    f"Generation {generation_id} resumed from disk: "
                    f"rows={generation_metrics.get('training_row_count', 0)}, "
                    f"episodes={generation_metrics.get('total_episodes_run', 0)}"
                )
            else:
                generation_dir, generation_data = run_generation_phase(
                    generation_config,
                    generation_id,
                    active_run_id,
                    active_docker_client,
                    metrics_collector=metrics_collector,
                    task=generation_task,
                    predecessor_generation_id=predecessor_generation_id,
                )
                generation_metrics = generation_data.to_metadata_dict(
                    run_id=active_run_id
                )
                logger.info(
                    f"Generation {generation_id} complete: "
                    f"rows={generation_data.training_row_count}, "
                    f"episodes={generation_data.total_episodes_run}"
                )

            generation_result = {
                "generation_id": generation_id,
                "generation_dir": str(generation_dir),
                "served_adapter_dir": _served_adapter_path(generation_config),
                "served_artifact_path": _served_training_artifact_path(
                    generation_config
                ),
                "training_export_format": training_export_format,
                "trained_adapter_dir": None,
                "trained_artifact_path": None,
                "training_data_paths": [],
                "metrics": generation_metrics,
            }
            results.append(generation_result)

            if generation_id < num_gens - 1:
                _try_write_run_doc("in_progress")

                artifact_output_dir = _training_artifact_path(
                    adapter_root,
                    generation_id,
                    training_export_format,
                )
                training_paths = _collect_training_window_paths(
                    generation_root,
                    end_generation_id=generation_id,
                    window_size=config.orchestration.training_window_size,
                )
                generation_result["training_data_paths"] = [
                    str(path) for path in training_paths
                ]

                if _training_artifact_complete(
                    artifact_output_dir, training_export_format
                ):
                    trained_artifact_path = artifact_output_dir
                    logger.info(
                        f"Training resumed from disk: artifact={trained_artifact_path}"
                    )
                else:
                    if should_wait_for_gpu_release:
                        logger.info(
                            "Waiting for GPU memory to be released before training"
                        )
                        wait_for_gpu_memory_release(
                            min_free_memory_fraction=(
                                config.training.gpu_wait_min_free_memory_fraction
                            ),
                            timeout_s=config.training.gpu_wait_timeout_seconds,
                        )

                    logger.info(
                        f"{'─' * 60}\n"
                        f"  Training on {len(training_paths)} generation(s), "
                        f"max_steps={config.training.max_steps}\n"
                        f"{'─' * 60}"
                    )
                    trained_artifact_path = _invoke_train_sft(
                        config,
                        training_paths=training_paths,
                        output_dir=artifact_output_dir,
                        export_format=training_export_format,
                    )
                    trained_artifact_path = validate_training_artifact(
                        trained_artifact_path,
                        training_export_format,
                        expected_path=artifact_output_dir,
                    )
                    logger.info(f"Training complete: artifact={trained_artifact_path}")

                latest_artifact_path = trained_artifact_path
                generation_result["trained_adapter_dir"] = (
                    str(trained_artifact_path)
                    if training_export_format == "lora"
                    else None
                )
                generation_result["trained_artifact_path"] = str(trained_artifact_path)
                _try_write_run_doc("in_progress")
    except Exception:
        _try_write_run_doc("failed", ended_at=_utc_now_iso())
        raise

    _try_write_run_doc("completed", ended_at=_utc_now_iso())
    return results


def run_loop(
    config: AppConfig,
    *,
    run_id: str | None = None,
    docker_client: Any = None,
    task: TaskSpec,
    metrics_collector: MetricsCollector | None = None,
    config_path: str | None = None,
    cli_hardware_tags: Sequence[str] | None = None,
    cli_load_tags: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()

    active_run_id = validate_run_id(run_id or generate_readable_run_id())
    generation_root = Path(config.generation.generation_output_dir) / active_run_id
    with _generation_run_lock(generation_root, active_run_id):
        return _run_loop_locked(
            config,
            run_id=active_run_id,
            docker_client=docker_client,
            task=task,
            metrics_collector=metrics_collector,
            config_path=config_path,
            cli_hardware_tags=cli_hardware_tags,
            cli_load_tags=cli_load_tags,
        )


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hardware-tag", action="append", default=[])
    parser.add_argument("--load-tag", action="append", default=[])
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume an existing run by its safe single-component id. Generations whose "
        "metadata.json + SFT export rows are on disk are skipped, and "
        "training steps whose adapter (with training_info.json) exists "
        "are reused. Omit to start a fresh run with a generated id. "
        "Only one process may run the same id at a time.",
    )
    parser.add_argument(
        "--rebuild-harness",
        action="store_true",
        help="Force a no-cache harness image rebuild. Source-content drift "
        "also triggers automatic rebuilding. Without a "
        "[harness.container.build] block, the image is pulled rather than built.",
    )
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as handle:
        config_dict = toml.load(handle)
    try:
        config = AppConfig(**unflatten_toml_dict(config_dict))
    except pydantic.ValidationError as exc:
        raise RuntimeError(f"Config validation error: {exc}") from exc

    ensure_configured_harness(config, rebuild=args.rebuild_harness)

    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()

    # Load task at startup
    task = load_task(config.task.class_, config.task.config)
    logger.info(f"Loaded task: {task.name} ({task.description})")

    run_id = validate_run_id(args.run_id or generate_readable_run_id())
    metrics_collector = MetricsCollector.from_config(config, run_id)
    run_loop(
        config,
        run_id=run_id,
        metrics_collector=metrics_collector,
        task=task,
        config_path=args.config,
        cli_hardware_tags=args.hardware_tag,
        cli_load_tags=args.load_tag,
    )
    return _run_doc_path(Path(config.generation.generation_output_dir) / run_id)


if __name__ == "__main__":
    print(main())
