from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import random
import time
from typing import Any, Optional

from loguru import logger
from tqdm import tqdm

from src.container import (
    ContainerManager,
    compose_services,
    project_name_from_compose,
    resolve_compose_file,
)
from src.execution.backend_runtime import BackendRuntime
from src.execution.episode_runner import run_single_episode
from src.execution.generation_persistence import (
    GENERATION_COMPLETION_FILENAME,
    GENERATION_MANIFEST_FILENAME,
    GenerationIdentity,
    checkpoint_payload_identity,
    ensure_generation_manifest,
    journal_apply_and_account,
    load_generation_journal,
    publish_episode_journal,
    task_commit_intents,
    validate_resume_checkpoint,
    write_generation_completion,
    serving_lineage_from_config,
)
from src.execution.generation_state import (
    IntrinsicBudgetController,
    committed_episode_accounting,
)
from src.task.base import (
    SaturationDecision,
    TaskEnvironmentError,
)
from src.training_data.transforms import (
    TARGET_COUNT_BASIS,
    TrainingDataExportContext,
    build_export_recipe,
    build_training_export,
    count_training_rows,
    publish_training_export,
)
from src.typing.config import AppConfig
from src.typing.training import (
    append_episode_jsonl,
    load_generation_checkpoint,
    save_durable_json,
    save_generation_checkpoint,
    save_stop_reason_sentinel,
)
from src.typing.trajectory import GenerationData


class _NullProgressBar:
    def __init__(self, total: Optional[int] = None, initial: int = 0) -> None:
        self.total = total
        self.n = initial

    def update(self, increment: int = 1) -> None:
        self.n += increment

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


def update_harness_error_breaker(
    *,
    error_category: str | None,
    consecutive: int,
    limit: int,
) -> tuple[int, bool]:
    if error_category == "harness_error":
        new_count = consecutive + 1
        should_trip = limit > 0 and new_count >= limit
        return new_count, should_trip
    return 0, False


def select_variation_index(
    strategy: str,
    episode_index: int,
    variation_count: int,
    rng: Optional[random.Random] = None,
) -> int:
    if variation_count <= 0:
        raise ValueError("variation_count must be positive")
    if strategy == "round_robin":
        return episode_index % variation_count
    if strategy == "random":
        active_rng = rng or random.Random()
        return active_rng.randrange(variation_count)
    raise ValueError(f"Unsupported variation strategy: {strategy}")


def _make_progress_bar(
    enabled: bool,
    total: Optional[int],
    description: str,
    initial: int = 0,
    position: int = 0,
):
    if not enabled:
        return _NullProgressBar(total=total, initial=initial)
    return tqdm(total=total, desc=description, initial=initial, position=position)


def _generation_dir(output_dir: Path, generation_id: int) -> Path:
    return output_dir / f"generation_{generation_id}"


_ROW_COUNT_REFRESH_EVERY_EPISODES = 10
_ROW_COUNT_NEAR_TARGET_MARGIN = 2


def _task_training_data_transforms(task: Any) -> tuple[Any, ...]:
    transforms = task.training_data_transforms()
    return tuple(transforms)


def _task_episode_commit_enabled(task: Any) -> bool:
    declaration = getattr(type(task), "episode_commit_enabled", None)
    return bool(declaration(task)) if callable(declaration) else False


def _recover_task_commits(task: Any, generation_data: GenerationData) -> None:
    if _task_episode_commit_enabled(task):
        task.recover_episode_commits(task_commit_intents(generation_data.all_episodes))


def _apply_task_commit(task: Any, episode: Any) -> None:
    if not _task_episode_commit_enabled(task):
        return
    token = episode.task_commit_token
    digest = episode.task_commit_digest
    if token is None and digest is None:
        return
    if token is None or digest is None:
        raise RuntimeError("episode has incomplete task commit metadata")
    task.apply_episode_commit(token, digest)


def _export_context(
    *,
    generation_id: int,
    run_id: str | None,
    task: Any,
    generation_dir: Path,
    all_episodes_path: Path,
    export_recipe_hash: str,
) -> TrainingDataExportContext:
    return TrainingDataExportContext(
        generation_id=generation_id,
        run_id=run_id,
        task_name=str(getattr(task, "name", type(task).__name__)),
        source_generation_dir=generation_dir,
        source_all_episodes_path=all_episodes_path,
        export_recipe_hash=export_recipe_hash,
    )


def _refresh_training_row_count(
    generation_data: GenerationData,
    transforms: tuple[Any, ...],
    context: TrainingDataExportContext,
) -> int:
    count = count_training_rows(generation_data, transforms, context)
    generation_data.set_training_row_count(count)
    return count


def _should_refresh_training_row_count(
    generation_data: GenerationData,
    target_training_rows: int,
) -> bool:
    if generation_data.training_row_count_is_exact:
        return False
    episodes_since_refresh = (
        generation_data.total_episodes_run
        - generation_data.training_row_count_last_refreshed_episode
    )
    if episodes_since_refresh >= _ROW_COUNT_REFRESH_EVERY_EPISODES:
        return True
    return generation_data.training_row_count >= max(
        0,
        target_training_rows - _ROW_COUNT_NEAR_TARGET_MARGIN,
    )


def _build_checkpoint_payload(
    generation_data: GenerationData,
    next_episode_index: int,
    run_id: str,
    *,
    target_training_rows: int,
    export_recipe_hash: str,
    identity: GenerationIdentity,
    task: Any | None = None,
) -> dict[str, Any]:
    accounting = committed_episode_accounting(generation_data.all_episodes)
    payload = generation_data.to_metadata_dict(run_id=run_id)
    payload.update(checkpoint_payload_identity(identity))
    payload["next_episode_index"] = next_episode_index
    payload["committed_episode_indices"] = list(
        accounting.committed_episode_indices
    )
    payload["bootstrap_episodes_run"] = accounting.bootstrap_episodes_run
    payload["regular_episodes_run"] = accounting.regular_episodes_run
    payload["has_admission"] = accounting.has_admission
    payload["target_count_basis"] = TARGET_COUNT_BASIS
    payload["target_training_rows"] = target_training_rows
    payload["export_recipe_hash"] = export_recipe_hash
    payload["training_row_count"] = generation_data.training_row_count
    if task is not None:
        _recover_task_commits(task, generation_data)
        task_state = task.generation_checkpoint_state()
        if task_state is not None:
            payload["task_checkpoint_state"] = task_state
    return payload


def run_generation_sequential(
    rt: BackendRuntime,
    *,
    generation_id: int,
    target_rows: int | None = None,
    max_episodes: int | None = None,
) -> GenerationData:
    generation_config = rt.config.generation or AppConfig.GenerationConfig()
    target_training_rows = (
        target_rows
        if target_rows is not None
        else generation_config.target_training_rows
    )
    max_episode_count = (
        max_episodes if max_episodes is not None else generation_config.max_episodes
    )
    max_bootstrap_count = generation_config.max_bootstrap_episodes
    output_dir = Path(generation_config.generation_output_dir)
    generation_dir = _generation_dir(output_dir, generation_id)
    checkpoint_path = generation_dir / "checkpoint.json"
    all_episodes_path = generation_dir / "all_episodes.jsonl"
    transforms = _task_training_data_transforms(rt.task)
    recipe = build_export_recipe(transforms)
    identity = GenerationIdentity(
        generation_id=generation_id,
        run_id=rt.run_id,
        task_name=str(getattr(rt.task, "name", type(rt.task).__name__)),
        task_identity=f"{type(rt.task).__module__}.{type(rt.task).__qualname__}",
        export_recipe_hash=recipe.recipe_hash,
        execution_mode="sequential",
        **serving_lineage_from_config(
            rt.config,
            generation_id,
            request_model=rt.request_model,
            external_lora_attested=rt.external_lora_attested,
            predecessor_generation_id=rt.predecessor_generation_id,
        ),
    )
    commit_enabled = _task_episode_commit_enabled(rt.task)
    export_context = _export_context(
        generation_id=generation_id,
        run_id=rt.run_id,
        task=rt.task,
        generation_dir=generation_dir,
        all_episodes_path=all_episodes_path,
        export_recipe_hash=recipe.recipe_hash,
    )

    generation_data = GenerationData(generation_id=generation_id)
    manifest_path = generation_dir / GENERATION_MANIFEST_FILENAME
    completion_path = generation_dir / GENERATION_COMPLETION_FILENAME
    if completion_path.exists():
        raise RuntimeError(
            f"generation {generation_id} is already published; "
            "the training loop must inspect its completion marker"
        )
    if not manifest_path.exists() and (
        all_episodes_path.exists() or checkpoint_path.exists()
    ):
        raise RuntimeError(
            "generation artifacts exist without a manifest; history was preserved"
        )
    manifest_created = ensure_generation_manifest(manifest_path, identity)
    checkpoint = load_generation_checkpoint(checkpoint_path)
    if manifest_created:
        if all_episodes_path.exists() or checkpoint is not None:
            raise RuntimeError("generation artifacts exist without a manifest")
    elif generation_config.resume_from_checkpoint and (
        all_episodes_path.exists() or checkpoint is not None
    ):
        if not all_episodes_path.exists():
            raise RuntimeError(
                "generation checkpoint exists without its episode journal"
            )
        generation_data, _ = load_generation_journal(
            all_episodes_path,
            generation_id=generation_id,
            execution_mode="sequential",
        )
        validate_resume_checkpoint(
            checkpoint=checkpoint,
            journal_indices={ep.episode_index for ep in generation_data.all_episodes},
            identity=identity,
            allow_journal_ahead=commit_enabled,
        )
        if checkpoint is not None:
            rt.task.load_generation_checkpoint_state(
                checkpoint.get("task_checkpoint_state")
            )
        _refresh_training_row_count(generation_data, transforms, export_context)
        if checkpoint is not None:
            generation_data.started_at = (
                checkpoint.get("started_at") or generation_data.started_at
            )
    elif (
        not manifest_created
        and not all_episodes_path.exists()
        and checkpoint is None
    ):
        logger.info(
            f"Reusing pristine generation manifest for generation {generation_id}"
        )
    elif all_episodes_path.exists() or checkpoint is not None:
        raise RuntimeError(
            "generation artifacts already exist; enable resume_from_checkpoint"
        )

    _recover_task_commits(rt.task, generation_data)

    if generation_data.started_at is None:
        generation_data.started_at = datetime.now().isoformat()

    compose_dir_str = rt.config.docker_compose_dir or rt.task.docker_compose_dir
    project_name_pattern = None
    expected_services = None
    if compose_dir_str and Path(compose_dir_str).exists():
        compose_file = resolve_compose_file(compose_dir_str)
        project_name_pattern = project_name_from_compose(compose_file)
        expected_services = compose_services(compose_file)

    container_manager = ContainerManager(
        docker_client=rt.docker_client,
        container_ids=rt.config.container_ids,
        docker_compose_dir=compose_dir_str,
        post_rebuild_wait_seconds=generation_config.post_rebuild_wait_seconds,
        project_name_pattern=project_name_pattern,
        expected_services=expected_services,
        task=rt.task,
    )
    try:
        ready = container_manager.verify_ready()
    except Exception:
        ready = False
    if not ready:
        if rt.config.dynamic_container and compose_dir_str:
            container_manager.rebuild()
        else:
            raise RuntimeError("Containers failed readiness check at startup")

    variations = rt.task.list_variations()
    variation_count = len(variations)
    rng = (
        random.Random(generation_config.variation_random_seed)
        if generation_config.variation_strategy == "random"
        else None
    )
    if rng is not None and generation_config.variation_random_seed is not None:
        for prior_episode_index in range(generation_data.total_episodes_run):
            select_variation_index(
                "random",
                prior_episode_index,
                variation_count,
                rng=rng,
            )
    progress_episode_total = (
        max_episode_count + max_bootstrap_count
        if max_bootstrap_count is not None
        else max_episode_count
    )
    episode_progress = _make_progress_bar(
        generation_config.show_progress,
        progress_episode_total,
        "episodes",
        initial=generation_data.total_episodes_run,
    )
    rows_progress = _make_progress_bar(
        generation_config.show_progress,
        target_training_rows,
        "training rows",
        initial=generation_data.training_row_count,
    )
    generation_started_at = time.perf_counter()
    consecutive_verification_failures = 0
    consecutive_environment_failures = 0
    consecutive_harness_errors = 0
    harness_error_limit = rt.config.harness.consecutive_harness_error_limit
    pending_variation_index: Optional[int] = None
    active_episode_index: int | None = None
    sequential_harness_session: dict[str, Any] = {}
    budget = IntrinsicBudgetController(
        max_episodes=max_episode_count,
        max_bootstrap_episodes=max_bootstrap_count,
        committed=generation_data.all_episodes,
    )
    saturation_stopped = False

    def _confirm_target_reached() -> bool:
        if generation_data.training_row_count < target_training_rows:
            return False
        if not generation_data.training_row_count_is_exact:
            _refresh_training_row_count(
                generation_data,
                transforms,
                export_context,
            )
        return generation_data.training_row_count >= target_training_rows

    def _recover_environment(
        exc: TaskEnvironmentError,
        *,
        episode_index: int,
        phase: str,
    ) -> bool:
        nonlocal consecutive_environment_failures
        consecutive_environment_failures += 1
        logger.info(
            f"Episode {episode_index}: environment error during {phase} "
            f"({consecutive_environment_failures} consecutive): {exc}. "
            "Rebuilding containers."
        )
        try:
            if exc.container_ids:
                container_manager.rebuild_containers(exc.container_ids)
            else:
                container_manager.rebuild()
        except Exception as rebuild_exc:
            logger.warning(
                f"Episode {episode_index}: container recovery failed: {rebuild_exc}"
            )
        if consecutive_environment_failures < 3:
            return True
        logger.error(
            "Circuit breaker tripped: 3 consecutive environment recovery "
            "attempts without a completed episode. Aborting generation."
        )
        generation_data.termination_reason = "force_stop"
        return False

    try:
        while True:
            if _should_refresh_training_row_count(
                generation_data,
                target_training_rows,
            ):
                _refresh_training_row_count(
                    generation_data,
                    transforms,
                    export_context,
                )
            if generation_data.training_row_count >= target_training_rows:
                if not generation_data.training_row_count_is_exact:
                    _refresh_training_row_count(
                        generation_data,
                        transforms,
                        export_context,
                    )
                if generation_data.training_row_count >= target_training_rows:
                    generation_data.termination_reason = "target_reached"
                    break

            if active_episode_index is None:
                active_episode_index = budget.reserve()
                if active_episode_index is None:
                    break
            episode_index = active_episode_index

            if pending_variation_index is None:
                variation_index = select_variation_index(
                    generation_config.variation_strategy,
                    episode_index,
                    variation_count,
                    rng=rng,
                )
            else:
                variation_index = pending_variation_index

            selected_variation = variations[variation_index % variation_count]
            try:
                population_outcome, verified = container_manager.populate_with_task(
                    container_manager.get_containers(),
                    selected_variation,
                )
            except TaskEnvironmentError as exc:
                pending_variation_index = variation_index
                if not _recover_environment(
                    exc,
                    episode_index=episode_index,
                    phase="population",
                ):
                    break
                continue

            pending_variation_index = None
            previous_rows = generation_data.training_row_count
            try:
                episode = run_single_episode(
                    rt,
                    container_manager=container_manager,
                    generation_id=generation_id,
                    episode_index=episode_index,
                    variation_index=variation_index,
                    population_outcome=population_outcome,
                    verified=verified,
                    variation=selected_variation,
                    harness_session=sequential_harness_session,
                )
            except TaskEnvironmentError as exc:
                pending_variation_index = variation_index
                if not _recover_environment(
                    exc,
                    episode_index=episode_index,
                    phase="execution",
                ):
                    break
                continue
            consecutive_environment_failures = 0

            def account_episode() -> None:
                nonlocal active_episode_index
                generation_data.add_episode(episode)
                budget.commit(episode)
                active_episode_index = None

            journal_apply_and_account(
                append_journal=lambda: append_episode_jsonl(
                    episode.to_dict(), all_episodes_path
                ),
                apply_task=lambda: _apply_task_commit(rt.task, episode),
                account_episode=account_episode,
            )
            if transforms:
                generation_data.mark_training_row_count_stale()
            else:
                generation_data.set_training_row_count(
                    generation_data.training_row_count,
                )
            if episode.error_message == "container population verification failed":
                consecutive_verification_failures += 1
                logger.warning(
                    f"Episode {episode_index}: container population verification "
                    f"failed ({consecutive_verification_failures} consecutive). "
                    f"Variation: {episode.container_variation}"
                )
                if (
                    generation_config.max_consecutive_verification_failures > 0
                    and consecutive_verification_failures
                    >= generation_config.max_consecutive_verification_failures
                ):
                    logger.error(
                        f"Circuit breaker tripped: "
                        f"{consecutive_verification_failures} consecutive "
                        f"container population verification failures. "
                        f"This indicates a systematic issue with the container setup. "
                        f"Aborting generation."
                    )
                    generation_data.termination_reason = "force_stop"
                    break
            else:
                consecutive_verification_failures = 0

            consecutive_harness_errors, should_trip = update_harness_error_breaker(
                error_category=episode.error_category,
                consecutive=consecutive_harness_errors,
                limit=harness_error_limit,
            )
            if episode.error_category == "harness_error":
                logger.warning(
                    f"Episode {episode_index}: harness error "
                    f"({consecutive_harness_errors} consecutive): "
                    f"{episode.error_message}"
                )
            if should_trip:
                logger.error(
                    f"HARNESS CIRCUIT BREAKER: {consecutive_harness_errors} "
                    f"consecutive HarnessError occurrences. Aborting generation - "
                    f"this is distinct from environment failure and indicates the "
                    f"harness implementation or its configuration is broken."
                )
                generation_data.termination_reason = "harness_error"
                break

            if _should_refresh_training_row_count(
                generation_data,
                target_training_rows,
            ):
                _refresh_training_row_count(
                    generation_data,
                    transforms,
                    export_context,
                )

            if (
                rt.task.saturation_enabled()
                and not generation_data.training_row_count_is_exact
            ):
                _refresh_training_row_count(
                    generation_data,
                    transforms,
                    export_context,
                )
            target_reached_after_episode = _confirm_target_reached()
            saturation_outcome = None
            if not target_reached_after_episode and rt.task.saturation_enabled():
                saturation_outcome = rt.task.evaluate_saturation(generation_data)

            episode_progress.update(1)
            rows_progress.update(generation_data.training_row_count - previous_rows)
            elapsed_hours = (time.perf_counter() - generation_started_at) / 3600
            if elapsed_hours > 0:
                episode_postfix: dict[str, str] = {}
                if episode.average_output_tokens_per_second is not None:
                    episode_postfix["tok/s"] = (
                        f"{episode.average_output_tokens_per_second:.1f}"
                    )
                if episode.inference_duty_cycle is not None:
                    episode_postfix["duty"] = f"{episode.inference_duty_cycle:.0%}"
                if episode.peak_gpu_utilization_pct is not None:
                    episode_postfix["gpu"] = f"{episode.peak_gpu_utilization_pct:.0f}%"
                if episode.peak_cpu_utilization_pct is not None:
                    episode_postfix["cpu"] = f"{episode.peak_cpu_utilization_pct:.0f}%"
                if episode_postfix:
                    episode_progress.set_postfix(episode_postfix)
                rows_progress.set_postfix(
                    {
                        "rows/hr": (
                            f"{generation_data.training_row_count / elapsed_hours:.1f}"
                        )
                    }
                )

            if generation_config.checkpoint_every_episode:
                save_generation_checkpoint(
                    _build_checkpoint_payload(
                        generation_data,
                        episode_index + 1,
                        rt.run_id,
                        target_training_rows=target_training_rows,
                        export_recipe_hash=recipe.recipe_hash,
                        identity=identity,
                        task=rt.task,
                    ),
                    checkpoint_path,
                )

            if target_reached_after_episode:
                generation_data.termination_reason = "target_reached"
                break

            saturation_decision = getattr(
                saturation_outcome,
                "decision",
                SaturationDecision.CONTINUE,
            )
            saturation_decision_value = getattr(
                saturation_decision,
                "value",
                saturation_decision,
            )
            if saturation_decision_value == SaturationDecision.STOP.value:
                saturation_stopped = True
                generation_data.termination_reason = (
                    getattr(saturation_outcome, "reason", None)
                    or "saturation_exhausted"
                )
                break
            if saturation_decision_value == SaturationDecision.ROLLOVER.value:
                rt.task.on_generation_rollover(
                    getattr(saturation_outcome, "reason", None) or "saturation",
                    generation_data,
                    rolled_at=datetime.now().isoformat(timespec="seconds"),
                )
                if generation_config.checkpoint_every_episode:
                    save_generation_checkpoint(
                        _build_checkpoint_payload(
                            generation_data,
                            episode_index + 1,
                            rt.run_id,
                            target_training_rows=target_training_rows,
                            export_recipe_hash=recipe.recipe_hash,
                            identity=identity,
                            task=rt.task,
                        ),
                        checkpoint_path,
                    )

            completed_episodes = generation_data.total_episodes_run
            episodes_remaining = budget.exhaustion_reason is None
            if (
                generation_config.container_rebuild_interval > 0
                and episodes_remaining
                and completed_episodes % generation_config.container_rebuild_interval
                == 0
            ):
                container_manager.rebuild()
                consecutive_verification_failures = 0
                consecutive_environment_failures = 0
            elif (
                generation_config.container_restart_interval > 0
                and episodes_remaining
                and completed_episodes % generation_config.container_restart_interval
                == 0
            ):
                container_manager.restart()
                consecutive_verification_failures = 0
                consecutive_environment_failures = 0
    finally:
        episode_progress.close()
        rows_progress.close()

    if generation_data.termination_reason is None:
        if not generation_data.training_row_count_is_exact:
            _refresh_training_row_count(
                generation_data,
                transforms,
                export_context,
            )
        if generation_data.training_row_count >= target_training_rows:
            generation_data.termination_reason = "target_reached"
        else:
            generation_data.termination_reason = budget.stop_reason

    generation_data.completed_at = datetime.now().isoformat()
    if saturation_stopped:
        publish_episode_journal(generation_data.all_episodes, all_episodes_path)
        save_stop_reason_sentinel(
            generation_data.termination_reason,
            generation_dir / "STOP_REASON",
        )
        save_generation_checkpoint(
            _build_checkpoint_payload(
                generation_data,
                generation_data.total_episodes_run,
                rt.run_id,
                target_training_rows=target_training_rows,
                export_recipe_hash=recipe.recipe_hash,
                identity=identity,
                task=rt.task,
            ),
            checkpoint_path,
        )
    return generation_data


def save_generation_data(
    generation_data: GenerationData,
    output_dir: Path | str,
    run_id: str,
    task: Any,
    config: AppConfig | None = None,
    *,
    request_model: str | None = None,
    external_lora_attested: bool | None = None,
    predecessor_generation_id: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    generation_dir = _generation_dir(output_dir, generation_data.generation_id)
    transforms = _task_training_data_transforms(task)
    recipe = build_export_recipe(transforms)
    manifest_path = generation_dir / GENERATION_MANIFEST_FILENAME
    execution_mode = "sequential"
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        execution_mode = manifest_payload.get("execution_mode", execution_mode)
    identity = GenerationIdentity(
        generation_id=generation_data.generation_id,
        run_id=run_id,
        task_name=str(getattr(task, "name", type(task).__name__)),
        task_identity=f"{type(task).__module__}.{type(task).__qualname__}",
        export_recipe_hash=recipe.recipe_hash,
        execution_mode=execution_mode,
        **serving_lineage_from_config(
            config,
            generation_data.generation_id,
            request_model=request_model,
            external_lora_attested=external_lora_attested,
            predecessor_generation_id=predecessor_generation_id,
        ),
    )
    ensure_generation_manifest(manifest_path, identity)

    generation_dir.mkdir(parents=True, exist_ok=True)
    successful_dir = generation_dir / "successful"
    failed_dir = generation_dir / "failed"
    successful_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    all_episodes_path = generation_dir / "all_episodes.jsonl"
    publish_episode_journal(generation_data.all_episodes, all_episodes_path)

    for episode in generation_data.successful_episodes:
        save_durable_json(
            episode.to_dict(), successful_dir / f"{episode.episode_id}.json"
        )

    for episode in generation_data.failed_episodes:
        save_durable_json(episode.to_dict(), failed_dir / f"{episode.episode_id}.json")

    export_context = _export_context(
        generation_id=generation_data.generation_id,
        run_id=run_id,
        task=task,
        generation_dir=generation_dir,
        all_episodes_path=all_episodes_path,
        export_recipe_hash=recipe.recipe_hash,
    )
    export = build_training_export(
        generation_data,
        transforms,
        export_context,
        row_observability=task.training_data_observability,
    )
    published_export = publish_training_export(generation_dir, export)
    generation_data.set_training_row_count(len(published_export.rows))

    metadata_payload = generation_data.to_metadata_dict(run_id=run_id)
    metadata_payload["active_sft_export_id"] = published_export.export_id
    metadata_payload["target_count_basis"] = TARGET_COUNT_BASIS
    metadata_payload["export_recipe_hash"] = recipe.recipe_hash
    metadata_path = generation_dir / "metadata.json"
    save_durable_json(metadata_payload, metadata_path)

    save_generation_checkpoint(
        _build_checkpoint_payload(
            generation_data,
            max(
                (episode.episode_index for episode in generation_data.all_episodes),
                default=-1,
            )
            + 1,
            run_id,
            target_training_rows=generation_data.training_row_count,
            export_recipe_hash=recipe.recipe_hash,
            identity=identity,
            task=task,
        ),
        generation_dir / "checkpoint.json",
    )

    write_generation_completion(generation_dir, generation_data.generation_id)

    return generation_dir


__all__ = [
    "run_generation_sequential",
    "save_generation_data",
    "select_variation_index",
    "update_harness_error_breaker",
]
