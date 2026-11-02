from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import random
import signal
import sys
import threading
import time
from typing import Any

from loguru import logger

import src.parallel as parallel_runtime
from src.backend.endpoint_pool import EndpointPoolUnavailable
from src.execution.backend_runtime import BackendRuntime, _coerce_runtime
from src.execution.episode_runner import run_single_episode
from src.execution.generation_persistence import (
    GENERATION_COMPLETION_FILENAME,
    GENERATION_MANIFEST_FILENAME,
    GenerationIdentity,
    JournaledTaskCommitError,
    checkpoint_payload_identity,
    ensure_generation_manifest,
    journal_apply_and_account,
    load_generation_journal,
    publish_episode_journal,
    task_commit_intents,
    serving_lineage_from_config,
    validate_resume_checkpoint,
)
from src.execution.generation_state import (
    IntrinsicBudgetController,
    committed_episode_accounting,
)
from src.execution.telemetry import (
    initial_framework_telemetry,
    telemetry_columns_for_harness,
)
from src.harness.loader import construct_harness
from src.parallel import (
    StopReason,
    ThreadSafeGenerationCollector,
    WorkerSlot,
)
from src.task.base import SaturationDecision, TaskEnvironmentError
from src.training_data.transforms import (
    TARGET_COUNT_BASIS,
    TrainingDataExportContext,
    build_export_recipe,
    count_training_rows,
)
from src.typing.config import AppConfig
from src.typing.training import (
    append_episode_jsonl,
    load_generation_checkpoint,
    save_generation_checkpoint,
    save_stop_reason_sentinel,
)
from src.typing.trajectory import GenerationData


_ROW_COUNT_REFRESH_EVERY_EPISODES = 10
_ROW_COUNT_NEAR_TARGET_MARGIN = 2


def _task_episode_commit_enabled(task: Any) -> bool:
    declaration = getattr(type(task), "episode_commit_enabled", None)
    return bool(declaration(task)) if callable(declaration) else False


def _task_parallel_checkpoint_safe(task: Any) -> bool:
    declaration = getattr(type(task), "parallel_checkpoint_safe", None)
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


def _episode_triggers_endpoint_quarantine(error_category: str | None) -> bool:
    """Whether an episode's failure should quarantine its inference endpoint.

    Only a genuine outage (``endpoint_unavailable``) does. A request timeout
    (``inference_timeout``) is a benign, retryable failure — the endpoint is
    healthy, the model was just slow (e.g. decode starvation). Quarantining the
    (possibly sole) endpoint on a timeout would breach the capacity floor and
    abort the whole run, so it must not. Mirrors ``EndpointAwareGenner``, which
    keys quarantine on the ``inference_unavailable`` prefix, not on timeouts.
    """
    return error_category == "endpoint_unavailable"


@contextmanager
def _null_context():
    yield


@contextmanager
def _scoped_parallel_logging(log_path: Path):
    logger.remove()
    try:
        logger.add(sys.stderr, level="WARNING")
        logger.add(str(log_path), level="DEBUG", enqueue=True)
        yield
    finally:
        logger.remove()
        logger.add(sys.stderr)


def _save_parallel_checkpoint(
    collector: ThreadSafeGenerationCollector,
    checkpoint_path: Path,
    run_id: str,
    *,
    target_training_rows: int,
    export_recipe_hash: str,
    identity: GenerationIdentity,
    generation_data: GenerationData | None = None,
    task: Any | None = None,
) -> None:
    generation_data = generation_data or collector.snapshot()
    accounting = committed_episode_accounting(generation_data.all_episodes)
    payload = generation_data.to_metadata_dict(run_id=run_id)
    payload.update(checkpoint_payload_identity(identity))
    payload["total_episodes_completed"] = generation_data.total_episodes_run
    payload["next_episode_index"] = accounting.next_episode_index
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
    payload["training_row_count_is_exact"] = generation_data.training_row_count_is_exact
    payload["training_row_count_last_refreshed_episode"] = (
        generation_data.training_row_count_last_refreshed_episode
    )
    if task is not None:
        _recover_task_commits(task, generation_data)
        task_state = task.generation_checkpoint_state()
        if task_state is not None:
            if not _task_parallel_checkpoint_safe(task):
                raise RuntimeError(
                    "parallel resumability requires commit-safe task state; "
                    "tasks with checkpoint-owned mutable state must run sequentially"
                )
            payload["task_checkpoint_state"] = task_state
    save_generation_checkpoint(payload, checkpoint_path)


def _generation_dir(output_dir: Path, generation_id: int) -> Path:
    return output_dir / f"generation_{generation_id}"


def select_variation_index(
    strategy: str,
    episode_index: int,
    variation_count: int,
    rng: random.Random | None = None,
) -> int:
    if variation_count <= 0:
        raise ValueError("variation_count must be positive")
    if strategy == "round_robin":
        return episode_index % variation_count
    if strategy == "random":
        active_rng = rng or random.Random()
        return active_rng.randrange(variation_count)
    raise ValueError(f"Unsupported variation strategy: {strategy}")


def _run_generation_parallel(
    rt: BackendRuntime, *, generation_id: int
) -> GenerationData:
    generation_config = rt.config.generation or AppConfig.GenerationConfig()
    output_dir = Path(generation_config.generation_output_dir)
    generation_dir = _generation_dir(output_dir, generation_id)
    checkpoint_path = generation_dir / "checkpoint.json"
    all_episodes_path = generation_dir / "all_episodes.jsonl"

    n_workers = generation_config.parallel_episodes
    target_rows = generation_config.target_training_rows
    transforms = tuple(rt.task.training_data_transforms())
    recipe = build_export_recipe(transforms)
    identity = GenerationIdentity(
        generation_id=generation_id,
        run_id=rt.run_id,
        task_name=str(getattr(rt.task, "name", type(rt.task).__name__)),
        task_identity=f"{type(rt.task).__module__}.{type(rt.task).__qualname__}",
        export_recipe_hash=recipe.recipe_hash,
        execution_mode="parallel",
        **serving_lineage_from_config(
            rt.config,
            generation_id,
            request_model=rt.request_model,
            external_lora_attested=rt.external_lora_attested,
            predecessor_generation_id=rt.predecessor_generation_id,
        ),
    )
    commit_enabled = _task_episode_commit_enabled(rt.task)
    initial_task_state = rt.task.generation_checkpoint_state()
    checkpoint_requires_quiescence = commit_enabled or initial_task_state is not None
    if initial_task_state is not None and not _task_parallel_checkpoint_safe(rt.task):
        raise RuntimeError(
            "parallel resumability requires commit-safe task state; "
            "tasks with checkpoint-owned mutable state must run sequentially"
        )
    export_context = TrainingDataExportContext(
        generation_id=generation_id,
        run_id=rt.run_id,
        task_name=str(getattr(rt.task, "name", type(rt.task).__name__)),
        source_generation_dir=generation_dir,
        source_all_episodes_path=all_episodes_path,
        export_recipe_hash=recipe.recipe_hash,
    )

    def _count_training_rows(data: GenerationData) -> int:
        return count_training_rows(data, transforms, export_context)

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
            execution_mode="parallel",
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
        generation_data.set_training_row_count(_count_training_rows(generation_data))
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

    base_compose_dir = (
        Path(rt.config.docker_compose_dir or rt.task.docker_compose_dir)
        if (rt.config.docker_compose_dir or rt.task.docker_compose_dir)
        else None
    )
    if base_compose_dir is None:
        raise ValueError(
            "docker_compose_dir is required for parallel episode execution"
        )

    slots = parallel_runtime.create_worker_slots(
        n_slots=n_workers,
        base_compose_dir=base_compose_dir,
        generation_dir=generation_dir,
        run_id=rt.run_id,
        code_host_cache_path=Path(rt.config.code_host_cache_path),
        post_rebuild_wait_seconds=generation_config.post_rebuild_wait_seconds,
        task=rt.task,
    )
    if len(slots) < n_workers:
        logger.warning(
            f"Generation running with {len(slots)} worker slots instead of "
            f"{n_workers} requested - Docker network pool capacity was "
            f"insufficient to create all slots."
        )

    collector = parallel_runtime.ThreadSafeGenerationCollector(
        generation_data,
        training_row_count_fn=_count_training_rows if transforms else None,
    )
    global_circuit = parallel_runtime.GlobalCircuitBreaker(
        [s.circuit_breaker for s in slots],
        threshold=generation_config.global_circuit_breaker_threshold,
    )
    endpoint_pool = rt.endpoint_pool
    budget = IntrinsicBudgetController(
        max_episodes=generation_config.max_episodes,
        max_bootstrap_episodes=generation_config.max_bootstrap_episodes,
        committed=generation_data.all_episodes,
    )
    file_lock = budget.lock
    stop_event = threading.Event()
    stop_reason = StopReason()
    row_count_refresh_lock = threading.Lock()
    saturation_lock = threading.Lock()
    variations = rt.task.list_variations()
    variation_count = len(variations)
    rollover_lock = threading.Lock()
    quiescence_lock = threading.Lock()
    drain_condition = threading.Condition()
    draining = threading.Event()
    saturation_stop_requested = threading.Event()
    in_flight = 0
    checkpoint_condition = threading.Condition()
    checkpoint_active = False
    checkpoint_completed_episodes = generation_data.total_episodes_run

    from rich.console import Console as RichConsole

    from src.display import ParallelProgressDisplay, scoped_loguru_to_rich

    display_console = RichConsole(stderr=True)
    display: ParallelProgressDisplay | None = None
    if generation_config.show_progress:
        display = ParallelProgressDisplay(
            n_slots=len(slots),
            target_rows=target_rows,
            max_episodes=generation_config.max_episodes,
            run_id=rt.run_id,
            generation_id=generation_id,
            console=display_console,
            metrics_collector=rt.metrics,
            max_context_tokens=(
                rt.config.vllm.max_model_len
                if rt.config.vllm
                else (rt.config.sglang.max_model_len if rt.config.sglang else None)
            ),
        )
        display.set_telemetry_columns(
            telemetry_columns_for_harness(construct_harness(rt.config.harness))
        )

    if rt.metrics is not None:
        rt.metrics.start_utilization_sampling(
            inference_metrics_url=rt.metrics.inference_metrics_url,
            inference_metrics_backend=rt.metrics.inference_metrics_backend,
            inference_metrics_api_key=getattr(
                rt.metrics, "inference_metrics_api_key", None
            ),
        )

    def _display_update_slot(slot_id: int, **kwargs: Any) -> None:
        if display is not None:
            display.update_slot(slot_id, **kwargs)

    def _signal_stop(reason: str) -> None:
        stop_reason.set(reason)
        stop_event.set()

    def _endpoint_capacity_floor_tripped() -> bool:
        return bool(endpoint_pool is not None and endpoint_pool.below_capacity_floor())

    def _mark_endpoint_unavailable(lease: Any, detail: str | None) -> None:
        if endpoint_pool is None or lease is None:
            return
        if endpoint_pool.is_healthy(lease.endpoint_id):
            endpoint_pool.mark_unhealthy(lease.endpoint_id, detail)
        if endpoint_pool.below_capacity_floor():
            _signal_stop("endpoint_capacity_floor")

    def _annotate_endpoint_metadata(episode: Any, lease: Any) -> None:
        if lease is None:
            return
        trajectory = getattr(episode, "trajectory", None)
        if not isinstance(trajectory, dict):
            return
        extra = trajectory.setdefault("extra", {})
        if isinstance(extra, dict):
            extra["inference_endpoint"] = dict(lease.metadata)

    def _make_telemetry_observer(slot_id: int):
        active_display = display
        if active_display is None:
            return None

        def _observer(
            prompt_tokens: int | None,
            telemetry: dict[str, str],
        ) -> None:
            active_display.update_slot(
                slot_id,
                last_prompt_tokens=prompt_tokens,
                telemetry=dict(telemetry),
            )

        return _observer

    def _display_update_progress() -> None:
        if display is not None:
            episodes, rows = collector.progress_snapshot()
            display.update_progress(rows=rows, episodes=episodes)

    def _should_refresh_training_row_count() -> bool:
        state = collector.row_count_state()
        if state.training_row_count_is_exact:
            return False
        episodes_since_refresh = (
            state.total_episodes_run - state.training_row_count_last_refreshed_episode
        )
        if episodes_since_refresh >= _ROW_COUNT_REFRESH_EVERY_EPISODES:
            return True
        return state.training_row_count >= max(
            0,
            target_rows - _ROW_COUNT_NEAR_TARGET_MARGIN,
        )

    def _maybe_refresh_training_row_count() -> None:
        if not _should_refresh_training_row_count():
            return
        if not row_count_refresh_lock.acquire(blocking=False):
            return
        try:
            if _should_refresh_training_row_count():
                collector.refresh_training_row_count()
        finally:
            row_count_refresh_lock.release()

    def _confirm_target_reached() -> bool:
        if not collector.should_stop(target_rows):
            return False
        state = collector.row_count_state()
        if not state.training_row_count_is_exact:
            with row_count_refresh_lock:
                collector.refresh_training_row_count()
        return collector.should_stop(target_rows)

    def _begin_episode_in_flight() -> bool:
        nonlocal in_flight
        with drain_condition:
            while draining.is_set() and not stop_event.is_set():
                drain_condition.wait(timeout=0.1)
            if stop_event.is_set():
                return False
            in_flight += 1
            return True

    def _finish_episode_in_flight() -> None:
        nonlocal in_flight
        with drain_condition:
            in_flight = max(0, in_flight - 1)
            if in_flight == 0:
                drain_condition.notify_all()

    def _set_draining() -> None:
        with drain_condition:
            draining.set()
            drain_condition.notify_all()

    def _clear_draining() -> None:
        with drain_condition:
            draining.clear()
            drain_condition.notify_all()

    def _wait_for_in_flight_zero() -> bool:
        with drain_condition:
            while in_flight > 0:
                if stop_event.is_set():
                    return False
                drain_condition.wait(timeout=0.1)
        return True

    def _request_parallel_checkpoint() -> None:
        nonlocal checkpoint_active, checkpoint_completed_episodes
        target_episodes = collector.progress_snapshot()[0]
        while True:
            with checkpoint_condition:
                if checkpoint_completed_episodes >= target_episodes:
                    return
                if not checkpoint_active:
                    checkpoint_active = True
                    break
                checkpoint_condition.wait(timeout=0.1)

        checkpoint_snapshot: GenerationData | None = None
        checkpoint_saved = False
        try:
            if checkpoint_requires_quiescence:
                with quiescence_lock:
                    _set_draining()
                    try:
                        if not _wait_for_in_flight_zero():
                            return
                        with file_lock:
                            checkpoint_snapshot = collector.snapshot()
                            _save_parallel_checkpoint(
                                collector,
                                checkpoint_path,
                                rt.run_id,
                                target_training_rows=target_rows,
                                export_recipe_hash=recipe.recipe_hash,
                                identity=identity,
                                generation_data=checkpoint_snapshot,
                                task=rt.task,
                            )
                            checkpoint_saved = True
                    finally:
                        _clear_draining()
            else:
                # Stateless tasks need journal/snapshot consistency, not a
                # generation-wide inference barrier.
                with file_lock:
                    checkpoint_snapshot = collector.snapshot()
                    _save_parallel_checkpoint(
                        collector,
                        checkpoint_path,
                        rt.run_id,
                        target_training_rows=target_rows,
                        export_recipe_hash=recipe.recipe_hash,
                        identity=identity,
                        generation_data=checkpoint_snapshot,
                        task=rt.task,
                    )
                    checkpoint_saved = True
        finally:
            with checkpoint_condition:
                if checkpoint_saved and checkpoint_snapshot is not None:
                    checkpoint_completed_episodes = max(
                        checkpoint_completed_episodes,
                        checkpoint_snapshot.total_episodes_run,
                    )
                checkpoint_active = False
                checkpoint_condition.notify_all()

    def _maybe_handle_saturation() -> None:
        if stop_event.is_set():
            return
        if not rt.task.saturation_enabled():
            return
        with saturation_lock:
            if stop_event.is_set():
                return
            with row_count_refresh_lock:
                if not collector.row_count_state().training_row_count_is_exact:
                    collector.refresh_training_row_count()
                saturation_snapshot = collector.snapshot()
            if saturation_snapshot.training_row_count >= target_rows:
                _signal_stop("target_reached")
                return
            outcome = rt.task.evaluate_saturation(saturation_snapshot)
            decision = getattr(outcome, "decision", SaturationDecision.CONTINUE)
            decision_value = getattr(decision, "value", decision)
            if decision_value == SaturationDecision.STOP.value:
                saturation_stop_requested.set()
                _signal_stop(
                    getattr(outcome, "reason", None) or "saturation_exhausted"
                )
                return
            if decision_value != SaturationDecision.ROLLOVER.value:
                return
            if stop_event.is_set() or not rollover_lock.acquire(blocking=False):
                return
            try:
                if stop_event.is_set():
                    return
                with quiescence_lock:
                    _set_draining()
                    try:
                        if not _wait_for_in_flight_zero():
                            return
                        if stop_event.is_set():
                            return
                        with row_count_refresh_lock:
                            if not collector.row_count_state().training_row_count_is_exact:
                                collector.refresh_training_row_count()
                            rollover_snapshot = collector.snapshot()
                        if rollover_snapshot.training_row_count >= target_rows:
                            _signal_stop("target_reached")
                            return
                        with file_lock:
                            _recover_task_commits(rt.task, rollover_snapshot)
                            rt.task.on_generation_rollover(
                                getattr(outcome, "reason", None) or "saturation",
                                rollover_snapshot,
                                rolled_at=datetime.now().isoformat(timespec="seconds"),
                            )
                            _save_parallel_checkpoint(
                                collector,
                                checkpoint_path,
                                rt.run_id,
                                target_training_rows=target_rows,
                                export_recipe_hash=recipe.recipe_hash,
                                identity=identity,
                                generation_data=rollover_snapshot,
                                task=rt.task,
                            )
                    finally:
                        _clear_draining()
            finally:
                rollover_lock.release()

    def worker_loop(slot: WorkerSlot) -> None:
        slot_config = rt.config.model_copy(
            update={"code_host_cache_path": str(slot.cache_dir)}
        )
        slot_runtime = BackendRuntime(
            config=slot_config,
            run_id=rt.run_id,
            task=rt.task,
            genner=rt.genner,
            docker_client=slot.docker_client,
            metrics=rt.metrics,
            predecessor_generation_id=rt.predecessor_generation_id,
        )
        slot_rng = (
            random.Random((generation_config.variation_random_seed or 0) + slot.slot_id)
            if generation_config.variation_strategy == "random"
            else None
        )
        slot_episodes_since_rebuild = 0
        slot_successes = 0
        slot_failures = 0
        while (
            not stop_event.is_set()
            and not collector.should_stop(target_rows)
            and not global_circuit.is_tripped()
        ):
            if _endpoint_capacity_floor_tripped():
                _signal_stop("endpoint_capacity_floor")
                break
            episode_in_flight = False
            active_budget_reservation = False
            episode_index: int | None = None
            try:
                episode_in_flight = _begin_episode_in_flight()
                if not episode_in_flight:
                    break
                episode_index = budget.reserve()
                if episode_index is None:
                    episode_in_flight = False
                    _finish_episode_in_flight()
                    stop_reason.set(budget.stop_reason or "max_episodes")
                    break
                active_budget_reservation = True

                _display_update_slot(
                    slot.slot_id,
                    status="running",
                    episode=episode_index,
                    telemetry=initial_framework_telemetry(),
                    last_prompt_tokens=None,
                )
                variation_index = select_variation_index(
                    generation_config.variation_strategy,
                    episode_index,
                    variation_count,
                    rng=slot_rng,
                )
                selected_variation = variations[variation_index % variation_count]
                try:
                    population_outcome, verified = (
                        slot.container_manager.populate_with_task(
                            slot.container_manager.get_containers(),
                            selected_variation,
                        )
                    )
                except Exception as exc:
                    if not hasattr(exc, "container_ids"):
                        raise
                    slot.circuit_breaker.record_environment_failure()
                    logger.warning(
                        f"Slot {slot.slot_id} episode {episode_index}: "
                        f"environment error during population: {exc}"
                    )
                    _display_update_slot(slot.slot_id, status="rebuilding")
                    try:
                        if exc.container_ids:
                            slot.container_manager.rebuild_containers(exc.container_ids)
                        else:
                            slot.container_manager.rebuild()
                    except Exception:
                        slot.circuit_breaker.record_failure()
                        slot_failures += 1
                        _display_update_slot(slot.slot_id, failures=slot_failures)
                        if slot.circuit_breaker.is_tripped():
                            logger.error(
                                f"Slot {slot.slot_id}: circuit breaker tripped "
                                f"after rebuild failures"
                            )
                            _display_update_slot(
                                slot.slot_id,
                                status="tripped",
                                cb_tripped=True,
                            )
                        active_budget_reservation = False
                        budget.abandon(episode_index)
                        episode_in_flight = False
                        _finish_episode_in_flight()
                        if slot.circuit_breaker.is_tripped():
                            break
                        continue
                    _display_update_slot(slot.slot_id, status="running")
                    active_budget_reservation = False
                    budget.abandon(episode_index)
                    episode_in_flight = False
                    _finish_episode_in_flight()
                    if slot.circuit_breaker.is_environment_tripped():
                        logger.error(
                            f"Slot {slot.slot_id}: retiring after repeated "
                            "environment recovery attempts"
                        )
                        _display_update_slot(
                            slot.slot_id,
                            status="tripped",
                            cb_tripped=True,
                        )
                        break
                    continue

                lease = None
                try:
                    if endpoint_pool is not None:
                        lease = endpoint_pool.lease(
                            home_key=slot.slot_id,
                            stop_event=stop_event,
                        )
                        episode_runtime = replace(slot_runtime, genner=lease.genner)
                    else:
                        episode_runtime = slot_runtime
                except EndpointPoolUnavailable as exc:
                    logger.warning(
                        f"Slot {slot.slot_id}: endpoint lease unavailable - {exc}"
                    )
                    _signal_stop(
                        "endpoint_capacity_floor"
                        if _endpoint_capacity_floor_tripped()
                        else "endpoint_pool_unavailable"
                    )
                    active_budget_reservation = False
                    budget.abandon(episode_index)
                    episode_in_flight = False
                    _finish_episode_in_flight()
                    break

                try:
                    episode = run_single_episode(
                        episode_runtime,
                        container_manager=slot.container_manager,
                        generation_id=generation_id,
                        episode_index=episode_index,
                        variation_index=variation_index,
                        population_outcome=population_outcome,
                        verified=verified,
                        parallel_episodes=n_workers,
                        stop_event=stop_event,
                        stop_reason=stop_reason,
                        variation=selected_variation,
                        telemetry_observer=_make_telemetry_observer(slot.slot_id),
                        harness_session=slot.harness_session,
                    )
                    _annotate_endpoint_metadata(episode, lease)
                    if _episode_triggers_endpoint_quarantine(episode.error_category):
                        _mark_endpoint_unavailable(lease, episode.error_message)
                except TaskEnvironmentError as exc:
                    slot.circuit_breaker.record_environment_failure()
                    logger.warning(
                        f"Slot {slot.slot_id} episode {episode_index}: "
                        f"environment error during execution: {exc}"
                    )
                    _display_update_slot(slot.slot_id, status="rebuilding")
                    try:
                        if exc.container_ids:
                            slot.container_manager.rebuild_containers(exc.container_ids)
                        else:
                            slot.container_manager.rebuild()
                    except Exception as rebuild_exc:
                        logger.warning(
                            f"Slot {slot.slot_id}: execution recovery rebuild failed: "
                            f"{rebuild_exc}"
                        )
                        slot.circuit_breaker.record_failure()
                    active_budget_reservation = False
                    budget.abandon(episode_index)
                    episode_in_flight = False
                    _finish_episode_in_flight()
                    if slot.circuit_breaker.is_tripped():
                        logger.error(
                            f"Slot {slot.slot_id}: retiring after repeated "
                            "environment recovery attempts"
                        )
                        _display_update_slot(
                            slot.slot_id,
                            status="tripped",
                            cb_tripped=True,
                        )
                        break
                    _display_update_slot(slot.slot_id, status="running")
                    continue
                finally:
                    if endpoint_pool is not None:
                        endpoint_pool.release(lease)

                slot.circuit_breaker.record_environment_success()
                if episode.error_message == "container population verification failed":
                    slot.circuit_breaker.record_verification_failure()
                    if slot.circuit_breaker.is_verification_tripped():
                        logger.error(
                            f"Slot {slot.slot_id}: verification circuit breaker tripped"
                        )
                elif episode.error_category in {
                    "context_overflow",
                    "repetition_collapse",
                    "inference_timeout",
                }:
                    slot.circuit_breaker.record_benign_abort()
                elif episode.error_category == "harness_error":
                    slot.circuit_breaker.record_failure()
                    logger.warning(
                        f"Slot {slot.slot_id} episode {episode_index}: "
                        f"harness error - {episode.error_message}"
                    )
                elif episode.error_category == "endpoint_unavailable":
                    slot.circuit_breaker.record_benign_abort()
                    logger.warning(
                        f"Slot {slot.slot_id} episode {episode_index}: "
                        f"endpoint unavailable - {episode.error_message}"
                    )
                else:
                    slot.circuit_breaker.record_success()

                def account_episode() -> None:
                    nonlocal active_budget_reservation
                    collector.add_episode(episode)
                    budget.commit(episode)
                    active_budget_reservation = False

                with file_lock:
                    journal_apply_and_account(
                        append_journal=lambda: append_episode_jsonl(
                            episode.to_dict(), all_episodes_path
                        ),
                        apply_task=lambda: _apply_task_commit(rt.task, episode),
                        account_episode=account_episode,
                    )
                episode_in_flight = False
                _finish_episode_in_flight()
                if generation_config.checkpoint_every_episode:
                    _request_parallel_checkpoint()
                if episode.success:
                    slot_successes += len(episode.prompt_responses)
                _display_update_slot(slot.slot_id, successes=slot_successes)
                _maybe_refresh_training_row_count()
                _display_update_progress()

                if _confirm_target_reached():
                    _signal_stop("target_reached")
                else:
                    _maybe_handle_saturation()
                slot_episodes_since_rebuild += 1
                if (
                    generation_config.container_rebuild_interval > 0
                    and slot_episodes_since_rebuild
                    >= generation_config.container_rebuild_interval
                ):
                    _display_update_slot(slot.slot_id, status="rebuilding")
                    try:
                        slot.container_manager.rebuild()
                        slot.circuit_breaker.reset()
                        slot_episodes_since_rebuild = 0
                    except Exception as exc:
                        logger.warning(
                            f"Slot {slot.slot_id}: randomization rebuild failed: {exc}"
                        )
                        slot.circuit_breaker.record_failure()
                    _display_update_slot(slot.slot_id, status="running")

                if slot.circuit_breaker.is_tripped():
                    logger.warning(
                        f"Slot {slot.slot_id}: circuit breaker tripped, attempting rebuild"
                    )
                    _display_update_slot(
                        slot.slot_id, status="rebuilding", cb_tripped=True
                    )
                    try:
                        slot.container_manager.rebuild()
                    except Exception:
                        logger.error(
                            f"Slot {slot.slot_id}: recovery rebuild failed, slot retiring"
                        )
                        _display_update_slot(slot.slot_id, status="done")
                        break
                    slot.circuit_breaker.reset()
                    _display_update_slot(
                        slot.slot_id, status="running", cb_tripped=False
                    )
            except Exception as exc:
                if isinstance(exc, JournaledTaskCommitError):
                    _signal_stop("task_commit_apply_failed")
                try:
                    if active_budget_reservation and not isinstance(
                        exc, JournaledTaskCommitError
                    ):
                        active_budget_reservation = False
                        if episode_index is None:
                            raise RuntimeError(
                                "active budget reservation has no episode index"
                            )
                        budget.abandon(episode_index)
                finally:
                    if episode_in_flight:
                        episode_in_flight = False
                        _finish_episode_in_flight()
                logger.exception(
                    f"Slot {slot.slot_id} episode {episode_index}: unhandled exception"
                )
                slot.circuit_breaker.record_failure()
                slot_failures += 1
                _display_update_slot(slot.slot_id, failures=slot_failures)
                if slot_failures >= 3:
                    logger.error(
                        f"Slot {slot.slot_id}: retiring after 3 unhandled episode errors"
                    )
                    _display_update_slot(slot.slot_id, status="done")
                    break
                if slot.circuit_breaker.is_tripped():
                    logger.error(
                        f"Slot {slot.slot_id}: circuit breaker tripped after unhandled "
                        f"exceptions, attempting rebuild"
                    )
                    _display_update_slot(
                        slot.slot_id, status="rebuilding", cb_tripped=True
                    )
                    try:
                        slot.container_manager.rebuild()
                    except Exception:
                        logger.error(
                            f"Slot {slot.slot_id}: recovery rebuild failed, slot retiring"
                        )
                        _display_update_slot(slot.slot_id, status="done")
                        break
                    slot.circuit_breaker.reset()
                    _display_update_slot(
                        slot.slot_id, status="running", cb_tripped=False
                    )

        _display_update_slot(slot.slot_id, status="done")

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(slot,),
            name=f"worker-slot-{slot.slot_id}",
            daemon=False,
        )
        for slot in slots
    ]
    original_handler = signal.getsignal(signal.SIGINT)

    def _graceful_shutdown(signum, frame):
        del signum, frame
        logger.info("SIGINT received, signaling workers to stop...")
        _signal_stop("sigint")

    signal.signal(signal.SIGINT, _graceful_shutdown)

    deadline_timer: threading.Timer | None = None
    if generation_config.generation_timeout_s:

        def _on_deadline() -> None:
            logger.warning(
                f"Generation deadline ({generation_config.generation_timeout_s}s) "
                "reached; signalling workers to stop"
            )
            _signal_stop("deadline_exceeded")

        deadline_timer = threading.Timer(
            generation_config.generation_timeout_s, _on_deadline
        )
        deadline_timer.daemon = True
        deadline_timer.start()

    inference_timeout = rt.config.inference.timeout if rt.config.inference else 300
    worker_join_timeout_s = inference_timeout + 30
    log_path = generation_dir / "run.log"
    log_context = (
        scoped_loguru_to_rich(display_console, log_path)
        if display is not None
        else _scoped_parallel_logging(log_path)
    )

    result: GenerationData | None = None
    started_threads: list[threading.Thread] = []
    display_context = display if display is not None else _null_context()
    with log_context:
        try:
            with display_context:
                try:
                    try:
                        for thread in threads:
                            thread.start()
                            started_threads.append(thread)
                    except Exception:
                        _signal_stop("worker_start_failed")
                        raise

                    while True:
                        alive = [
                            thread for thread in started_threads if thread.is_alive()
                        ]
                        if not alive or stop_event.is_set():
                            break
                        for thread in alive:
                            thread.join(timeout=0.2)

                    if stop_event.is_set():
                        join_deadline = time.monotonic() + worker_join_timeout_s
                        for thread in started_threads:
                            remaining = max(0.0, join_deadline - time.monotonic())
                            thread.join(timeout=remaining)

                        timed_out = [
                            thread for thread in started_threads if thread.is_alive()
                        ]
                        if timed_out:
                            logger.warning(
                                f"{len(timed_out)} worker threads did not exit within "
                                f"{worker_join_timeout_s}s after stop signal; waiting up to "
                                "30 additional seconds"
                            )
                            stop_reason.set("force_stop")
                            stop_event.set()
                            final_deadline = time.monotonic() + 30
                            for thread in timed_out:
                                thread.join(
                                    timeout=max(0.0, final_deadline - time.monotonic())
                                )
                finally:
                    if any(thread.is_alive() for thread in started_threads):
                        if not stop_event.is_set():
                            _signal_stop("worker_shutdown")
                        logger.warning(
                            "Waiting without a deadline for worker threads to exit"
                        )
                    for thread in started_threads:
                        thread.join()
                    if deadline_timer is not None:
                        deadline_timer.cancel()
                    signal.signal(signal.SIGINT, original_handler)
                    _display_update_progress()

            with row_count_refresh_lock:
                if not collector.row_count_state().training_row_count_is_exact:
                    collector.refresh_training_row_count()
            result = collector.snapshot()
            result.completed_at = datetime.now().isoformat()
            target_reached = (
                result.training_row_count_is_exact
                and result.training_row_count >= target_rows
            )
            result.termination_reason = (
                "target_reached"
                if target_reached
                else stop_reason.get()
                or ("force_stop" if global_circuit.is_tripped() else "max_episodes")
            )
            if (
                saturation_stop_requested.is_set()
                and result.termination_reason != "target_reached"
            ):
                save_stop_reason_sentinel(
                    result.termination_reason,
                    generation_dir / "STOP_REASON",
                )
            with file_lock:
                publish_episode_journal(result.all_episodes, all_episodes_path)
                _save_parallel_checkpoint(
                    collector,
                    checkpoint_path,
                    rt.run_id,
                    target_training_rows=target_rows,
                    export_recipe_hash=recipe.recipe_hash,
                    identity=identity,
                    generation_data=result,
                    task=rt.task,
                )
        finally:
            try:
                if rt.metrics is not None:
                    utilization_summary = rt.metrics.stop_utilization_sampling()
                    logger.info(
                        f"Parallel generation utilization: "
                        f"peak_gpu={utilization_summary.peak_gpu_utilization_pct}, "
                        f"peak_kv_cache={utilization_summary.peak_kv_cache_usage_pct}, "
                        f"peak_requests_running={utilization_summary.peak_num_requests_running}, "
                        f"peak_requests_waiting={utilization_summary.peak_num_requests_waiting}"
                    )
                    delta = utilization_summary.inference_metrics_delta
                    if delta is not None:
                        logger.info(
                            "Parallel generation vLLM deltas (window="
                            f"{delta.get('window_seconds')}s): "
                            f"preemptions={delta.get('preemptions')}, "
                            f"prefix_cache_hit_rate={delta.get('prefix_cache_hit_rate')}, "
                            f"prompt_tokens_cached_rate={delta.get('prompt_tokens_cached_rate')}, "
                            f"prompt_tok/s={delta.get('prompt_tokens_per_second')}, "
                            f"gen_tok/s={delta.get('generation_tokens_per_second')}"
                        )
                    rt.metrics.flush()
            finally:
                parallel_runtime.teardown_worker_slots(slots)

    assert result is not None
    logger.info(
        f"Parallel generation complete: rows={result.training_row_count}, "
        f"episodes={result.total_episodes_run}"
    )
    for slot in slots:
        logger.info(
            f"Slot {slot.slot_id} final state: "
            f"{'tripped' if slot.circuit_breaker.is_tripped() else 'ok'}"
        )
    return result


def run_generation_parallel(*args: Any, **kwargs: Any) -> GenerationData:
    if args and isinstance(args[0], BackendRuntime):
        rt = args[0]
        generation_id = kwargs.pop("generation_id", args[1] if len(args) > 1 else None)
        if generation_id is None:
            raise TypeError("run_generation_parallel requires generation_id")
        return _run_generation_parallel(rt, generation_id=int(generation_id))

    genner = kwargs.pop("genner", args[0] if len(args) > 0 else None)
    docker_client = kwargs.pop("docker_client", args[1] if len(args) > 1 else None)
    config = kwargs.pop("config", args[2] if len(args) > 2 else None)
    generation_id = kwargs.pop("generation_id", args[3] if len(args) > 3 else None)
    run_id = kwargs.pop("run_id", args[4] if len(args) > 4 else None)
    metrics_collector = kwargs.pop("metrics_collector", None)
    task = kwargs.pop("task", None)

    if None in (genner, docker_client, config, generation_id, run_id, task):
        raise TypeError("run_generation_parallel missing required legacy arguments")

    rt = _coerce_runtime(
        config=config,
        run_id=run_id,
        task=task,
        genner=genner,
        docker_client=docker_client,
        metrics=metrics_collector,
    )
    return _run_generation_parallel(rt, generation_id=int(generation_id))


__all__ = ["_scoped_parallel_logging", "run_generation_parallel"]
