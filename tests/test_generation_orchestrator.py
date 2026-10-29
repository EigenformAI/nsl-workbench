import json
import random
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

from src.execution import BackendRuntime, run_generation
from src.execution.episode import EpisodeOutcome, EpisodeRequest
from src.execution.episode_runner import run_single_episode
from src.execution.generation import (
    save_generation_data,
    select_variation_index,
)
from src.execution.generation_persistence import (
    GENERATION_COMPLETION_FILENAME,
    GENERATION_MANIFEST_FILENAME,
    GenerationIdentity,
    checkpoint_payload_identity,
    ensure_generation_manifest,
    publish_episode_journal,
    serving_lineage_from_config,
)
from src.execution.generation_state import IntrinsicBudgetController
from src.execution.parallel import run_generation_parallel
from src.observability.types import InferenceMetric, UsageInfo
from src.parallel import SlotCircuitBreaker, StopReason, WorkerSlot
from src.task.base import (
    EpisodeClassification,
    SaturationDecision,
    SaturationOutcome,
    TaskEnvironmentError,
)
from src.task.types import PopulationOutcome, PopulationResult, TaskReward, Variation
from src.training_data.transforms import build_export_recipe
from src.typing.config import AppConfig
from src.typing.training import save_generation_checkpoint
from src.typing.trajectory import EpisodeTrajectory, GenerationData
from tests.fixtures.stub_task import ValidStubTask


class GenerationOrchestratorTests(unittest.TestCase):
    def make_task(self, *, verification_result: bool = True) -> MagicMock:
        base_task = ValidStubTask({})
        task = MagicMock()
        task.list_variations.return_value = [
            Variation(name=f"v{index}", description=f"Test variation {index}")
            for index in range(1, 6)
        ]
        task.verify_population.return_value = verification_result
        task.prompt_spec.side_effect = base_task.prompt_spec
        task.parse_response.side_effect = base_task.parse_response
        task.execute_capability.side_effect = base_task.execute_capability
        task.measure_initial_state.return_value = {}
        task.finalize_episode.return_value = TaskReward(
            value=0.0,
            success=False,
            breakdown={},
        )
        task.metric_name = base_task.metric_name
        task.metric_unit = base_task.metric_unit
        task.higher_is_better = base_task.higher_is_better
        task.agent_service_name = base_task.agent_service_name
        task.name = base_task.name
        task.training_data_transforms.return_value = ()
        task.training_data_observability.return_value = {}
        task.evaluate_saturation.return_value = SaturationOutcome(
            SaturationDecision.CONTINUE
        )
        task.saturation_enabled.return_value = False
        task.classify_episode.return_value = EpisodeClassification()
        task.generation_checkpoint_state.return_value = None
        task.on_generation_rollover.return_value = None
        return task

    def make_config(self, base_dir: Path, **generation_overrides: object) -> AppConfig:
        generation_config = {
            "target_training_rows": 10,
            "max_episodes": 5,
            "container_restart_interval": 50,
            "container_rebuild_interval": 50,
            "variation_strategy": "round_robin",
            "variation_random_seed": None,
            "post_rebuild_wait_seconds": 1,
            "checkpoint_every_episode": False,
            "resume_from_checkpoint": False,
            "show_progress": False,
            "resource_snapshot_interval_episodes": 1,
            "generation_output_dir": str(base_dir / "generations"),
            "max_consecutive_verification_failures": 0,
        }
        generation_config.update(generation_overrides)
        return AppConfig(
            task={"class": "tests.fixtures.stub_task.ValidStubTask"},
            model_name="claude",
            code_host_cache_path=str(base_dir / "code-host-cache"),
            container_ids=["container-a", "container-b"],
            main_container_idx=0,
            dynamic_container=False,
            docker_compose_dir=str(base_dir / "compose"),
            train_data_save_folder=str(base_dir / "train-data"),
            harness={
                "name": "orchestrator_modes",
                "orchestrator_modes": {
                    "max_harness_iterations": 2,
                    "scratchpad_max_chars": 1000,
                    # Phase 2: orchestrator_prompt is required — tests that
                    # mock the harness don't exercise the template but the
                    # config layer still validates it.
                    "orchestrator_prompt": "test prompt {scratchpad_content}",
                },
            },
            generation=generation_config,
            observability={
                "enabled": True,
                "record_inference": False,
                "record_phases": False,
                "record_resources": True,
            },
        )

    def make_backend_runtime(
        self,
        config: AppConfig,
        *,
        task: MagicMock | None = None,
        genner: MagicMock | None = None,
        docker_client: MagicMock | None = None,
        metrics: MagicMock | None = None,
        endpoint_pool: object | None = None,
        run_id: str = "run-123",
    ) -> BackendRuntime:
        return BackendRuntime(
            config=config,
            run_id=run_id,
            task=task or self.make_task(),
            genner=genner or MagicMock(),
            docker_client=docker_client or MagicMock(),
            metrics=metrics,
            endpoint_pool=endpoint_pool,
        )

    def make_episode_request(
        self,
        *,
        variation,
        container_manager: MagicMock,
        containers: list[MagicMock] | None = None,
        agent_container: MagicMock | None = None,
        episode_id: str = "ep-0",
        episode_context: dict[str, object] | None = None,
        private_context: dict[str, object] | None = None,
        stop_event: threading.Event | None = None,
        stop_reason: StopReason | None = None,
        telemetry_observer: Any = None,
        harness_session: dict[str, object] | None = None,
    ) -> EpisodeRequest:
        active_containers = containers or [MagicMock(id="container-a")]
        active_agent_container = agent_container or MagicMock(id="agent-container")
        return EpisodeRequest(
            episode_id=episode_id,
            containers=active_containers,
            container_manager=container_manager,
            agent_container=active_agent_container,
            variation=variation,
            episode_context=episode_context or {},
            private_context=private_context,
            harness_session=harness_session,
            stop_event=stop_event,
            stop_reason=stop_reason,
            telemetry_observer=telemetry_observer,
        )

    def make_episode(
        self,
        episode_index: int,
        *,
        row_count: int = 1,
        success: bool = True,
        episode_runtime_success: bool | None = None,
        partial: bool = False,
        error_message: str | None = None,
        score: float = 64.0,
        container_overhead_seconds: float | None = None,
        episode_execution_seconds: float | None = None,
        total_inference_ms: float | None = None,
        inference_call_count: int | None = None,
        average_output_tokens_per_second: float | None = None,
        inference_duty_cycle: float | None = None,
        peak_gpu_utilization_pct: float | None = None,
        peak_cpu_utilization_pct: float | None = None,
        avg_gpu_utilization_pct: float | None = None,
        avg_cpu_utilization_pct: float | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        peak_context_tokens: int | None = None,
        avg_context_tokens: float | None = None,
        median_context_tokens: float | None = None,
        error_category: str | None = None,
        peak_kv_cache_usage_pct: float | None = None,
        avg_kv_cache_usage_pct: float | None = None,
        peak_num_requests_running: int | None = None,
        peak_num_requests_waiting: int | None = None,
        bootstrap_active: bool = False,
        admitted: bool = False,
    ) -> EpisodeTrajectory:
        return EpisodeTrajectory(
            episode_id=f"ep-{episode_index}",
            generation_id=0,
            episode_index=episode_index,
            prompt_responses=[
                {
                    "prompt": f"prompt-{episode_index}-{idx}",
                    "raw_response": f"response-{episode_index}-{idx}",
                    "interaction_type": "orchestrator",
                    "timestamp": "2026-04-06T00:00:00",
                    "success": True,
                    "error_message": None,
                }
                for idx in range(row_count)
            ],
            trajectory={"mode_history": []},
            score=score,
            episode_runtime_success=(
                episode_runtime_success
                if episode_runtime_success is not None
                else score > 0.0
            ),
            success=success,
            llm_turns_count=2,
            container_variation="variation_1_heavy",
            started_at="2026-04-06T00:00:00",
            completed_at="2026-04-06T00:00:01",
            duration_seconds=1.0,
            partial=partial,
            error_message=error_message,
            error_category=error_category,
            task_breakdown={
                "space_measurements": {"container-a": (100.0, 164.0)},
                "filesystem_groups": [
                    {
                        "filesystem_id": "fs-shared",
                        "container_ids": ["container-a", "container-b"],
                    }
                ],
                "measurement_errors": [],
                "bootstrap_active": bootstrap_active,
                "admitted": admitted,
            },
            container_overhead_seconds=container_overhead_seconds,
            episode_execution_seconds=episode_execution_seconds,
            total_inference_ms=total_inference_ms,
            inference_call_count=inference_call_count,
            average_output_tokens_per_second=average_output_tokens_per_second,
            inference_duty_cycle=inference_duty_cycle,
            peak_gpu_utilization_pct=peak_gpu_utilization_pct,
            peak_cpu_utilization_pct=peak_cpu_utilization_pct,
            avg_gpu_utilization_pct=avg_gpu_utilization_pct,
            avg_cpu_utilization_pct=avg_cpu_utilization_pct,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            peak_context_tokens=peak_context_tokens,
            avg_context_tokens=avg_context_tokens,
            median_context_tokens=median_context_tokens,
            peak_kv_cache_usage_pct=peak_kv_cache_usage_pct,
            avg_kv_cache_usage_pct=avg_kv_cache_usage_pct,
            peak_num_requests_running=peak_num_requests_running,
            peak_num_requests_waiting=peak_num_requests_waiting,
        )

    def make_inference_metric(
        self,
        episode_id: str,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: float,
        success: bool = True,
    ) -> InferenceMetric:
        total_tokens = None
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        return InferenceMetric(
            inference_id=f"inf-{episode_id}-{latency_ms}",
            run_id="run-123",
            backend="claude",
            phase="orchestrator",
            success=success,
            episode_id=episode_id,
            usage=(
                UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                if prompt_tokens is not None or completion_tokens is not None
                else None
            ),
            latency_ms=latency_ms,
        )

    def make_episode_outcome(
        self,
        *,
        episode_id: str = "episode-1",
        row_count: int = 1,
        success: bool = True,
        partial: bool = False,
        error_message: str | None = None,
        error_category: str | None = None,
        score: float = 64.0,
    ) -> EpisodeOutcome:
        prompt_responses = [
            {
                "prompt": f"prompt-{idx}",
                "raw_response": f"response-{idx}",
                "interaction_type": "explorer",
                "timestamp": "2026-04-06T00:00:00",
                "success": True,
                "error_message": None,
            }
            for idx in range(row_count)
        ]
        return EpisodeOutcome(
            trajectory={"mode_history": []},
            prompt_responses=prompt_responses,
            train_rows=prompt_responses,
            score=score,
            success=success,
            episode_id=episode_id,
            llm_turns_count=2,
            partial=partial,
            error_message=error_message,
            error_category=error_category,
            reward_breakdown={
                "space_measurements": {"container-a": (100.0, 164.0)},
                "filesystem_groups": [
                    {
                        "filesystem_id": "fs-shared",
                        "container_ids": ["container-a", "container-b"],
                    }
                ],
                "measurement_errors": [],
            },
            harness_error=False,
        )

    def configure_manager_mock(self, manager: MagicMock) -> None:
        manager.populate_with_task.return_value = (
            PopulationOutcome(
                results=[
                    PopulationResult(
                        container_id="container-a",
                        variation_name="variation_1_heavy",
                        description="heavy variation",
                        success=True,
                    )
                ],
            ),
            True,
        )
        manager.rebuild.return_value = None

    def test_run_generation_stops_at_target_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=3,
                max_episodes=10,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(0, row_count=2, success=True),
                    self.make_episode(1, row_count=2, success=True),
                    self.make_episode(2, row_count=2, success=True),
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(run_single_episode_mock.call_count, 2)
        self.assertEqual(generation_data.training_row_count, 4)
        self.assertEqual(generation_data.total_successful, 2)

    def test_run_generation_stops_at_max_episodes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=2,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(0, row_count=0, success=False, score=0.0),
                    self.make_episode(1, row_count=0, success=False, score=0.0),
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(run_single_episode_mock.call_count, 2)
        self.assertEqual(generation_data.total_episodes_run, 2)
        self.assertEqual(generation_data.training_row_count, 0)

    def test_variation_round_robin_strategy(self) -> None:
        indices = [select_variation_index("round_robin", idx, 5) for idx in range(8)]

        self.assertEqual(indices, [0, 1, 2, 3, 4, 0, 1, 2])

    def test_variation_random_strategy_with_seed(self) -> None:
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        picks_a = [
            select_variation_index("random", idx, 5, rng=rng_a) for idx in range(6)
        ]
        picks_b = [
            select_variation_index("random", idx, 5, rng=rng_b) for idx in range(6)
        ]

        self.assertEqual(picks_a, picks_b)
        self.assertTrue(all(0 <= pick < 5 for pick in picks_a))

    def test_container_rebuild_at_interval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=5,
                container_restart_interval=99,
                container_rebuild_interval=2,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(idx, row_count=0, success=False, score=0.0)
                    for idx in range(5)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(manager.rebuild.call_count, 2)

    def test_container_restart_at_interval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=5,
                container_restart_interval=2,
                container_rebuild_interval=99,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(idx, row_count=0, success=False, score=0.0)
                    for idx in range(5)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(manager.restart.call_count, 2)

    def test_max_bootstrap_episodes_extends_budget_for_bootstrap_episodes(self) -> None:
        """After framework admission, the regular budget takes over. Three
        bootstrap episodes plus two regular episodes exhaust the regular cap.
        """
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=2,
                max_bootstrap_episodes=3,
                container_restart_interval=50,
                container_rebuild_interval=50,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        idx,
                        row_count=0,
                        success=(idx == 2),
                        score=0.0,
                        bootstrap_active=(idx < 3),
                        admitted=(idx == 2),
                    )
                    for idx in range(5)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(run_single_episode_mock.call_count, 5)

    def test_max_bootstrap_episodes_stops_without_progress(self) -> None:
        """If no bootstrap episode is admitted, the regular budget is
        irrelevant. The run must
        stop at max_bootstrap_episodes instead of looping until the regular cap
        (which it can never reach).
        """
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=20,
                max_bootstrap_episodes=3,
                container_restart_interval=50,
                container_rebuild_interval=50,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        idx,
                        row_count=0,
                        success=False,
                        score=0.0,
                        bootstrap_active=True,
                        admitted=False,
                    )
                    for idx in range(20)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(run_single_episode_mock.call_count, 3)

    def test_pre_admission_non_bootstrap_episodes_stop_at_regular_cap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=2,
                max_bootstrap_episodes=10,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(index, row_count=0, success=False, score=0.0)
                    for index in range(3)
                ]

                result = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(run_single_episode_mock.call_count, 2)
        self.assertEqual(result.termination_reason, "max_episodes")

    def test_rebuild_interval_takes_precedence_over_restart_overlap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=6,
                container_restart_interval=2,
                container_rebuild_interval=4,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(idx, row_count=0, success=False, score=0.0)
                    for idx in range(6)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(manager.rebuild.call_count, 1)
        self.assertEqual(manager.restart.call_count, 1)

    @patch("src.execution.episode_runner.run_episode")
    def test_run_single_episode_passes_design_format_episode_id(
        self,
        run_episode_mock: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="variation_1_heavy",
                            description="heavy variation",
                            success=True,
                        )
                    ],
                ),
                True,
            )

            def _run_episode_side_effect(
                *args: object, **kwargs: object
            ) -> dict[str, object]:
                req = kwargs.get("req") or args[1]
                return self.make_episode_outcome(episode_id=str(req.episode_id))

            run_episode_mock.side_effect = _run_episode_side_effect

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=3,
                episode_index=42,
                variation_index=0,
                run_id="run-123",
                task=self.make_task(),
            )

        actual_req = (
            run_episode_mock.call_args.kwargs.get("req")
            or run_episode_mock.call_args.args[1]
        )
        actual_episode_id = str(actual_req.episode_id)
        self.assertRegex(actual_episode_id, r"^ep_gen3_0042_\d{10,}$")
        self.assertEqual(trajectory.episode_id, actual_episode_id)

    @patch("src.execution.episode_runner.run_episode")
    def test_container_populate_every_episode(
        self, run_episode_mock: MagicMock
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="variation_4_large_sparse",
                            description="large sparse variation",
                            success=True,
                        )
                    ],
                ),
                True,
            )

            run_episode_mock.return_value = self.make_episode_outcome(
                episode_id="ep-0",
                row_count=2,
                success=True,
                score=64.0,
            )

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=3,
                run_id="run-123",
                task=self.make_task(),
            )

        manager.populate_with_task.assert_called_once()
        self.assertEqual(manager.get_containers.call_count, 2)
        manager.verify_population.assert_not_called()
        self.assertEqual(trajectory.container_variation, "variation_4_large_sparse")
        self.assertEqual(len(trajectory.prompt_responses), 2)

    def test_run_single_episode_records_utilization_on_verification_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="variation_1_heavy",
                            description="heavy variation",
                            success=True,
                        )
                    ],
                ),
                False,
            )
            metrics_collector = MagicMock()
            metrics_collector.inference_metrics = []
            metrics_collector._lock = threading.Lock()
            metrics_collector.inference_metrics_url = "http://backend/metrics"
            metrics_collector.inference_metrics_backend = "vllm"
            metrics_collector.inference_metrics_api_key = "metrics-secret"
            metrics_collector.stop_utilization_sampling.return_value = MagicMock(
                peak_gpu_utilization_pct=34.0,
                peak_cpu_utilization_pct=15.0,
                avg_gpu_utilization_pct=21.0,
                avg_cpu_utilization_pct=10.0,
                peak_kv_cache_usage_pct=81.0,
                avg_kv_cache_usage_pct=77.5,
                peak_num_requests_running=4,
                peak_num_requests_waiting=2,
            )

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
                metrics_collector=metrics_collector,
                task=self.make_task(verification_result=False),
            )

        self.assertFalse(trajectory.success)
        self.assertEqual(
            trajectory.error_message, "container population verification failed"
        )
        self.assertEqual(trajectory.peak_gpu_utilization_pct, 34.0)
        self.assertEqual(trajectory.peak_cpu_utilization_pct, 15.0)
        self.assertEqual(trajectory.avg_gpu_utilization_pct, 21.0)
        self.assertEqual(trajectory.avg_cpu_utilization_pct, 10.0)
        self.assertEqual(trajectory.peak_kv_cache_usage_pct, 81.0)
        self.assertEqual(trajectory.avg_kv_cache_usage_pct, 77.5)
        self.assertEqual(trajectory.peak_num_requests_running, 4)
        self.assertEqual(trajectory.peak_num_requests_waiting, 2)
        metrics_collector.start_utilization_sampling.assert_called_once_with(
            inference_metrics_url="http://backend/metrics",
            inference_metrics_backend="vllm",
            inference_metrics_api_key="metrics-secret",
        )
        metrics_collector.stop_utilization_sampling.assert_called_once_with()

    @patch("src.execution.episode_runner.run_episode")
    def test_run_single_episode_requires_population_results(
        self, run_episode_mock: MagicMock
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(results=[]),
                True,
            )

            with self.assertRaisesRegex(RuntimeError, "populate returned no results"):
                run_single_episode(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    container_manager=manager,
                    config=config,
                    generation_id=0,
                    episode_index=0,
                    variation_index=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        manager.verify_population.assert_not_called()
        run_episode_mock.assert_not_called()

    @patch("src.execution.episode_runner.run_episode")
    def test_partial_episode_from_mode_exception_is_recorded(
        self,
        run_episode_mock: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="variation_1_heavy",
                            description="heavy variation",
                            success=True,
                        )
                    ],
                ),
                True,
            )

            run_episode_mock.return_value = self.make_episode_outcome(
                episode_id="ep-0",
                success=False,
                partial=True,
                error_message="mode error",
                score=0.0,
            )

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
                task=self.make_task(),
            )

        self.assertTrue(trajectory.partial)
        self.assertFalse(trajectory.episode_runtime_success)
        self.assertEqual(trajectory.error_message, "mode error")

    @patch("src.execution.episode_runner.run_episode")
    def test_run_single_episode_converts_episode_exception_to_failed_trajectory(
        self,
        run_episode_mock: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="variation_1_heavy",
                            description="heavy variation",
                            success=True,
                        )
                    ],
                ),
                True,
            )

            run_episode_mock.side_effect = RuntimeError("df -k / failed")

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
                task=self.make_task(),
            )

        self.assertFalse(trajectory.success)
        self.assertFalse(trajectory.episode_runtime_success)
        self.assertIn("df -k / failed", trajectory.error_message)
        self.assertEqual(trajectory.prompt_responses, [])

    def test_checkpoint_written_after_each_episode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=2,
                checkpoint_every_episode=True,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
                patch(
                    "src.execution.generation.append_episode_jsonl"
                ) as append_episode_jsonl,
                patch(
                    "src.execution.generation.save_generation_checkpoint"
                ) as save_generation_checkpoint,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(0, row_count=0, success=False, score=0.0),
                    self.make_episode(1, row_count=0, success=False, score=0.0),
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(append_episode_jsonl.call_count, 2)
        self.assertEqual(save_generation_checkpoint.call_count, 2)
        checkpoint_payload = save_generation_checkpoint.call_args_list[-1].args[0]
        self.assertIn("training_row_count_is_exact", checkpoint_payload)
        self.assertIn(
            "training_row_count_last_refreshed_episode",
            checkpoint_payload,
        )
        self.assertTrue(checkpoint_payload["training_row_count_is_exact"])

    def test_resume_from_checkpoint_restores_progress(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                max_episodes=3,
                resume_from_checkpoint=True,
            )
            task = self.make_task()
            generation_dir = base_dir / "generations" / "generation_0"
            recipe_hash = build_export_recipe(()).recipe_hash
            identity = GenerationIdentity(
                generation_id=0,
                run_id="run-123",
                task_name=task.name,
                task_identity=f"{type(task).__module__}.{type(task).__qualname__}",
                export_recipe_hash=recipe_hash,
                execution_mode="sequential",
                **serving_lineage_from_config(config, 0),
            )
            ensure_generation_manifest(
                generation_dir / GENERATION_MANIFEST_FILENAME,
                identity,
            )
            existing = [
                self.make_episode(index, row_count=0, success=False, score=0.0)
                for index in range(2)
            ]
            publish_episode_journal(existing, generation_dir / "all_episodes.jsonl")
            save_generation_checkpoint(
                {
                    **checkpoint_payload_identity(identity),
                    "next_episode_index": 2,
                    "committed_episode_indices": [0, 1],
                    "export_recipe_hash": recipe_hash,
                },
                generation_dir / "checkpoint.json",
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.return_value = self.make_episode(
                    2,
                    row_count=0,
                    success=False,
                    score=0.0,
                )

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        first_call = run_single_episode_mock.call_args_list[0]
        self.assertEqual(first_call.kwargs["episode_index"], 2)

    def test_sequential_journal_without_checkpoint_fails_explicitly(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, resume_from_checkpoint=True)
            task = self.make_task()
            generation_dir = base_dir / "generations" / "generation_0"
            identity = GenerationIdentity(
                generation_id=0,
                run_id="run-123",
                task_name=task.name,
                task_identity=f"{type(task).__module__}.{type(task).__qualname__}",
                export_recipe_hash=build_export_recipe(()).recipe_hash,
                execution_mode="sequential",
                **serving_lineage_from_config(config, 0),
            )
            ensure_generation_manifest(
                generation_dir / GENERATION_MANIFEST_FILENAME,
                identity,
            )
            publish_episode_journal(
                [self.make_episode(0, row_count=0, success=False)],
                generation_dir / "all_episodes.jsonl",
            )

            with (
                patch("src.execution.generation.ContainerManager") as container_manager,
                self.assertRaisesRegex(RuntimeError, "history was preserved"),
            ):
                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        container_manager.assert_not_called()

    @patch("src.execution.episode_runner.run_episode")
    def test_run_single_episode_records_background_utilization_summary(
        self,
        run_episode_mock: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="variation_1_heavy",
                            description="heavy variation",
                            success=True,
                        )
                    ],
                ),
                True,
            )

            manager.get_containers.return_value = [MagicMock(), MagicMock()]
            metrics_collector = MagicMock()
            metrics_collector.inference_metrics = []
            metrics_collector._lock = threading.Lock()
            metrics_collector.stop_utilization_sampling.return_value = MagicMock(
                peak_gpu_utilization_pct=82.0,
                peak_cpu_utilization_pct=37.0,
                avg_gpu_utilization_pct=51.5,
                avg_cpu_utilization_pct=24.0,
                peak_kv_cache_usage_pct=88.0,
                avg_kv_cache_usage_pct=70.0,
                peak_num_requests_running=6,
                peak_num_requests_waiting=3,
            )
            run_episode_mock.return_value = self.make_episode_outcome(episode_id="ep-0")

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
                metrics_collector=metrics_collector,
                task=self.make_task(),
            )

        metrics_collector.start_utilization_sampling.assert_called_once()
        metrics_collector.stop_utilization_sampling.assert_called_once_with()
        self.assertEqual(trajectory.peak_gpu_utilization_pct, 82.0)
        self.assertEqual(trajectory.peak_cpu_utilization_pct, 37.0)
        self.assertEqual(trajectory.avg_gpu_utilization_pct, 51.5)
        self.assertEqual(trajectory.avg_cpu_utilization_pct, 24.0)
        self.assertEqual(trajectory.peak_kv_cache_usage_pct, 88.0)
        self.assertEqual(trajectory.avg_kv_cache_usage_pct, 70.0)
        self.assertEqual(trajectory.peak_num_requests_running, 6)
        self.assertEqual(trajectory.peak_num_requests_waiting, 3)

    def test_run_generation_does_not_snapshot_resources_for_utilization(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=5,
                resource_snapshot_interval_episodes=2,
            )
            metrics_collector = MagicMock()
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(idx, row_count=0, success=False, score=0.0)
                    for idx in range(5)
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    metrics_collector=metrics_collector,
                    task=self.make_task(),
                )

        metrics_collector.snapshot_resources.assert_not_called()

    @patch("src.execution.episode_runner.run_episode")
    @patch("src.execution.episode_runner.time.perf_counter")
    def test_run_single_episode_records_observability_metrics(
        self,
        perf_counter_mock: MagicMock,
        run_episode_mock: MagicMock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="variation_1_heavy",
                            description="heavy variation",
                            success=True,
                        )
                    ],
                ),
                True,
            )

            manager.get_containers.return_value = [MagicMock(), MagicMock()]
            metrics_collector = MagicMock()
            metrics_collector.inference_metrics = [
                self.make_inference_metric(
                    "ep-0", prompt_tokens=100, completion_tokens=80, latency_ms=1200.0
                ),
                self.make_inference_metric(
                    "ep-0", prompt_tokens=120, completion_tokens=120, latency_ms=1800.0
                ),
                self.make_inference_metric(
                    "other-episode",
                    prompt_tokens=999,
                    completion_tokens=999,
                    latency_ms=9000.0,
                ),
            ]
            metrics_collector._lock = threading.Lock()
            metrics_collector.summary.side_effect = [
                {
                    "inference_calls": 3,
                    "total_latency_ms": 12000.0,
                    "total_output_tokens": 2198,
                },
            ]
            metrics_collector.stop_utilization_sampling.return_value = MagicMock(
                peak_gpu_utilization_pct=82.0,
                peak_cpu_utilization_pct=37.0,
                avg_gpu_utilization_pct=51.5,
                avg_cpu_utilization_pct=24.0,
            )
            run_episode_mock.return_value = self.make_episode_outcome(episode_id="ep-0")
            perf_counter_mock.side_effect = [10.0, 12.0, 12.0, 20.0]

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
                metrics_collector=metrics_collector,
                task=self.make_task(),
            )

        self.assertEqual(trajectory.container_overhead_seconds, 2.0)
        self.assertEqual(trajectory.episode_execution_seconds, 8.0)
        self.assertEqual(trajectory.total_inference_ms, 3000.0)
        self.assertEqual(trajectory.inference_call_count, 2)
        self.assertEqual(trajectory.average_output_tokens_per_second, 200 / 3)
        self.assertEqual(trajectory.total_input_tokens, 220)
        self.assertEqual(trajectory.total_output_tokens, 200)
        self.assertEqual(trajectory.peak_context_tokens, 120)
        self.assertEqual(trajectory.avg_context_tokens, 110.0)
        self.assertEqual(trajectory.median_context_tokens, 110.0)
        self.assertAlmostEqual(trajectory.inference_duty_cycle, 0.3)
        self.assertEqual(trajectory.peak_gpu_utilization_pct, 82.0)
        self.assertEqual(trajectory.peak_cpu_utilization_pct, 37.0)
        self.assertEqual(trajectory.avg_gpu_utilization_pct, 51.5)
        self.assertEqual(trajectory.avg_cpu_utilization_pct, 24.0)
        metrics_collector.start_utilization_sampling.assert_called_once()
        metrics_collector.stop_utilization_sampling.assert_called_once_with()

    def test_run_generation_updates_progress_postfix_with_observability_metrics(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=1,
                resource_snapshot_interval_episodes=1,
            )
            metrics_collector = MagicMock()
            episode_progress = MagicMock()
            rows_progress = MagicMock()
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
                patch(
                    "src.execution.generation._make_progress_bar",
                    side_effect=[episode_progress, rows_progress],
                ),
                patch(
                    "src.execution.generation.time.perf_counter",
                    side_effect=[0.0, 1800.0],
                ),
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.return_value = self.make_episode(
                    0,
                    row_count=20,
                    success=True,
                    average_output_tokens_per_second=240.0,
                    inference_duty_cycle=0.75,
                    peak_gpu_utilization_pct=82.0,
                    peak_cpu_utilization_pct=37.0,
                )

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    metrics_collector=metrics_collector,
                    task=self.make_task(),
                )

        postfix = episode_progress.set_postfix.call_args.args[0]
        row_postfix = rows_progress.set_postfix.call_args.args[0]
        self.assertEqual(postfix["tok/s"], "240.0")
        self.assertEqual(postfix["duty"], "75%")
        self.assertEqual(postfix["gpu"], "82%")
        self.assertEqual(postfix["cpu"], "37%")
        self.assertEqual(row_postfix["rows/hr"], "40.0")

    def test_run_generation_parallel_uses_rich_display_and_scoped_logging(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                parallel_episodes=2,
                max_episodes=7,
                target_training_rows=11,
                show_progress=True,
            )

            fake_slots = [
                MagicMock(slot_id=i, circuit_breaker=MagicMock()) for i in range(2)
            ]
            with (
                patch("src.parallel.create_worker_slots", return_value=fake_slots),
                patch("src.parallel.teardown_worker_slots"),
                patch("src.execution.parallel.construct_harness") as construct_harness,
                patch(
                    "src.display.ParallelProgressDisplay",
                ) as mock_display_cls,
                patch(
                    "src.display.scoped_loguru_to_rich",
                    return_value=nullcontext(),
                ) as scoped_loguru,
            ):
                mock_display = MagicMock()
                mock_display.__enter__ = MagicMock(return_value=mock_display)
                mock_display.__exit__ = MagicMock(return_value=False)
                mock_display_cls.return_value = mock_display
                construct_harness.return_value = MagicMock(
                    telemetry_columns=MagicMock(return_value=["step", "budget_left"])
                )

                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(result.total_episodes_run, 0)
        self.assertEqual(result.training_row_count, 0)
        # Rich display should be created with correct params
        mock_display_cls.assert_called_once()
        call_kwargs = mock_display_cls.call_args.kwargs
        # Display n_slots reflects actual created slots, not requested count
        self.assertEqual(call_kwargs["n_slots"], 2)
        self.assertEqual(call_kwargs["target_rows"], 11)
        self.assertEqual(call_kwargs["max_episodes"], 7)
        self.assertEqual(call_kwargs["run_id"], "run-123")
        self.assertEqual(call_kwargs["generation_id"], 0)
        mock_display.set_telemetry_columns.assert_any_call(
            ["tool_calls", "step", "budget_left"]
        )
        # scoped_loguru_to_rich should be used instead of _scoped_parallel_logging
        scoped_loguru.assert_called_once()

    def test_parallel_generation_rejects_unserialized_task_checkpoint_state(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, parallel_episodes=2)
            task = self.make_task()
            task.generation_checkpoint_state.return_value = {"eager_state": 1}

            with patch("src.parallel.create_worker_slots") as create_worker_slots:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "parallel resumability requires commit-safe task state",
                ):
                    run_generation_parallel(
                        genner=MagicMock(),
                        docker_client=MagicMock(),
                        config=config,
                        generation_id=0,
                        run_id="run-123",
                        task=task,
                    )

            create_worker_slots.assert_not_called()
            self.assertFalse((base_dir / "generations" / "generation_0").exists())

    def test_parallel_resume_recomputes_rows_before_worker_slots(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=2,
                resume_from_checkpoint=True,
                target_training_rows=200,
            )
            generation_dir = base_dir / "generations" / "generation_0"
            task = self.make_task()
            recipe_hash = build_export_recipe(()).recipe_hash
            identity = GenerationIdentity(
                generation_id=0,
                run_id="run-123",
                task_name=task.name,
                task_identity=f"{type(task).__module__}.{type(task).__qualname__}",
                export_recipe_hash=recipe_hash,
                execution_mode="parallel",
                **serving_lineage_from_config(config, 0),
            )
            ensure_generation_manifest(
                generation_dir / GENERATION_MANIFEST_FILENAME,
                identity,
            )
            all_episodes_path = generation_dir / "all_episodes.jsonl"
            publish_episode_journal(
                [self.make_episode(0, row_count=1, success=True)],
                all_episodes_path,
            )
            checkpoint_path = generation_dir / "checkpoint.json"
            save_generation_checkpoint(
                {
                    **checkpoint_payload_identity(identity),
                    "next_episode_index": 1,
                    "committed_episode_indices": [0],
                    "training_row_count": 999,
                    "export_recipe_hash": recipe_hash,
                },
                checkpoint_path,
            )
            events: list[str] = []

            def fake_count_training_rows(*args: object, **kwargs: object) -> int:
                events.append("count")
                return 123

            def fake_create_worker_slots(
                *args: object, **kwargs: object
            ) -> list[WorkerSlot]:
                events.append("create_slots")
                return []

            with (
                patch(
                    "src.execution.parallel.count_training_rows",
                    side_effect=fake_count_training_rows,
                ) as count_training_rows,
                patch(
                    "src.parallel.create_worker_slots",
                    side_effect=fake_create_worker_slots,
                ),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        count_training_rows.assert_called_once()
        self.assertEqual(events[:2], ["count", "create_slots"])
        self.assertEqual(result.training_row_count, 123)

    def test_parallel_journal_without_checkpoint_fails_before_slots(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                parallel_episodes=2,
                resume_from_checkpoint=True,
            )
            task = self.make_task()
            generation_dir = base_dir / "generations" / "generation_0"
            identity = GenerationIdentity(
                generation_id=0,
                run_id="run-123",
                task_name=task.name,
                task_identity=f"{type(task).__module__}.{type(task).__qualname__}",
                export_recipe_hash=build_export_recipe(()).recipe_hash,
                execution_mode="parallel",
                **serving_lineage_from_config(config, 0),
            )
            ensure_generation_manifest(
                generation_dir / GENERATION_MANIFEST_FILENAME,
                identity,
            )
            publish_episode_journal(
                [self.make_episode(4, row_count=0, success=False)],
                generation_dir / "all_episodes.jsonl",
            )

            with (
                patch("src.parallel.create_worker_slots") as create_worker_slots,
                self.assertRaisesRegex(RuntimeError, "history was preserved"),
            ):
                run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        create_worker_slots.assert_not_called()

    def test_sequential_matching_manifest_only_retries_as_pristine(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                resume_from_checkpoint=True,
                max_episodes=1,
                target_training_rows=1,
            )
            task = self.make_task()
            generation_dir = base_dir / "generations" / "generation_0"
            identity = GenerationIdentity(
                generation_id=0,
                run_id="run-123",
                task_name=task.name,
                task_identity=f"{type(task).__module__}.{type(task).__qualname__}",
                export_recipe_hash=build_export_recipe(()).recipe_hash,
                execution_mode="sequential",
                **serving_lineage_from_config(config, 0),
            )
            manifest_path = generation_dir / GENERATION_MANIFEST_FILENAME
            ensure_generation_manifest(manifest_path, identity)
            manifest_bytes = manifest_path.read_bytes()
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode",
                    return_value=self.make_episode(0),
                ),
            ):
                self.configure_manager_mock(ContainerManager.return_value)
                result = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

            self.assertEqual(result.total_episodes_run, 1)
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)

    def test_parallel_matching_manifest_only_retries_as_pristine(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=2,
                resume_from_checkpoint=True,
            )
            task = self.make_task()
            generation_dir = base_dir / "generations" / "generation_0"
            identity = GenerationIdentity(
                generation_id=0,
                run_id="run-123",
                task_name=task.name,
                task_identity=f"{type(task).__module__}.{type(task).__qualname__}",
                export_recipe_hash=build_export_recipe(()).recipe_hash,
                execution_mode="parallel",
                **serving_lineage_from_config(config, 0),
            )
            manifest_path = generation_dir / GENERATION_MANIFEST_FILENAME
            ensure_generation_manifest(manifest_path, identity)
            manifest_bytes = manifest_path.read_bytes()
            with (
                patch("src.parallel.create_worker_slots", return_value=[]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

            self.assertEqual(result.total_episodes_run, 0)
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)

    def test_sequential_completed_generation_refuses_mutation_before_setup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            generation_dir = base_dir / "generations" / "generation_0"
            generation_dir.mkdir(parents=True)
            (generation_dir / GENERATION_COMPLETION_FILENAME).write_text(
                "{}", encoding="utf-8"
            )

            with (
                patch("src.execution.generation.ContainerManager") as manager,
                self.assertRaisesRegex(RuntimeError, "already published"),
            ):
                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

            manager.assert_not_called()
            self.assertFalse(
                (generation_dir / GENERATION_MANIFEST_FILENAME).exists()
            )

    def test_parallel_completed_generation_refuses_mutation_before_setup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, parallel_episodes=2)
            generation_dir = base_dir / "generations" / "generation_0"
            generation_dir.mkdir(parents=True)
            (generation_dir / GENERATION_COMPLETION_FILENAME).write_text(
                "{}", encoding="utf-8"
            )

            with (
                patch("src.parallel.create_worker_slots") as create_worker_slots,
                self.assertRaisesRegex(RuntimeError, "already published"),
            ):
                run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

            create_worker_slots.assert_not_called()
            self.assertFalse(
                (generation_dir / GENERATION_MANIFEST_FILENAME).exists()
            )

    def test_parallel_near_target_refreshes_before_target_reached(self) -> None:
        class IdentityTransform:
            @property
            def name(self) -> str:
                return "IdentityTransform[test]"

            def config(self) -> dict[str, object]:
                return {}

            def transform_export_rows(self, context: object, episodes: list[object]):
                return episodes

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=2,
                max_episodes=5,
            )
            task = self.make_task()
            task.training_data_transforms.return_value = (IdentityTransform(),)
            manager = MagicMock()
            self.configure_manager_mock(manager)
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=SlotCircuitBreaker(),
                cache_dir=base_dir / "slot-0",
            )
            exact_refresh_episode_counts: list[int] = []

            def fake_count_training_rows(data: GenerationData, *args: object) -> int:
                exact_refresh_episode_counts.append(data.total_episodes_run)
                return 2 if data.total_episodes_run >= 3 else 1

            def fake_run_single_episode(
                *args: object, **kwargs: object
            ) -> EpisodeTrajectory:
                return self.make_episode(
                    int(kwargs["episode_index"]),
                    row_count=1,
                    success=True,
                )

            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    side_effect=fake_run_single_episode,
                ) as run_single_episode_mock,
                patch(
                    "src.execution.parallel.count_training_rows",
                    side_effect=fake_count_training_rows,
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        self.assertEqual(run_single_episode_mock.call_count, 3)
        self.assertEqual(exact_refresh_episode_counts, [1, 2, 3])
        self.assertEqual(result.termination_reason, "target_reached")
        self.assertEqual(result.training_row_count, 2)

    def test_parallel_finalization_refreshes_exact_non_target_count(self) -> None:
        class IdentityTransform:
            @property
            def name(self) -> str:
                return "IdentityTransform[finalization-test]"

            def config(self) -> dict[str, object]:
                return {}

            def transform_export_rows(self, context: object, episodes: list[object]):
                return episodes

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=100,
                max_episodes=1,
            )
            task = self.make_task()
            task.training_data_transforms.return_value = (IdentityTransform(),)
            manager = MagicMock()
            self.configure_manager_mock(manager)
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=SlotCircuitBreaker(),
                cache_dir=base_dir / "slot-0",
            )
            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    return_value=self.make_episode(0, row_count=1),
                ),
                patch(
                    "src.execution.parallel.count_training_rows",
                    return_value=7,
                ) as count_training_rows,
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )
            checkpoint = json.loads(
                (
                    base_dir
                    / "generations"
                    / "generation_0"
                    / "checkpoint.json"
                ).read_text(encoding="utf-8")
            )

        count_training_rows.assert_called_once()
        self.assertEqual(result.training_row_count, 7)
        self.assertTrue(result.training_row_count_is_exact)
        self.assertEqual(checkpoint["training_row_count"], 7)
        self.assertTrue(checkpoint["training_row_count_is_exact"])

    def test_parallel_exact_target_precedes_stale_saturation_stop(self) -> None:
        class ExpandingTransform:
            @property
            def name(self) -> str:
                return "ExpandingTransform[parallel-saturation-test]"

            def config(self) -> dict[str, object]:
                return {}

            def transform_export_rows(self, context: object, episodes: list[object]):
                return episodes

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=10,
                max_episodes=2,
            )
            task = self.make_task()
            task.training_data_transforms.return_value = (ExpandingTransform(),)
            task.saturation_enabled.return_value = True
            task.evaluate_saturation.return_value = SaturationOutcome(
                SaturationDecision.STOP, "saturation_exhausted"
            )
            manager = MagicMock()
            self.configure_manager_mock(manager)
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=SlotCircuitBreaker(),
                cache_dir=base_dir / "slot-0",
            )
            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    return_value=self.make_episode(0, row_count=1),
                ),
                patch("src.execution.parallel.count_training_rows", return_value=10),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

            stop_reason_path = (
                base_dir / "generations" / "generation_0" / "STOP_REASON"
            )
            stop_reason_written = stop_reason_path.exists()

        task.evaluate_saturation.assert_not_called()
        self.assertEqual(result.termination_reason, "target_reached")
        self.assertEqual(result.training_row_count, 10)
        self.assertTrue(result.training_row_count_is_exact)
        self.assertFalse(stop_reason_written)

    def test_save_generation_data_output_structure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            generation_data = GenerationData(generation_id=4)
            generation_data.add_episode(
                self.make_episode(0, row_count=2, success=True, score=64.0)
            )
            generation_data.add_episode(
                self.make_episode(1, row_count=1, success=False, score=0.0)
            )
            generation_data.set_training_row_count(999)
            generation_data.mark_training_row_count_stale()

            generation_dir = save_generation_data(
                generation_data=generation_data,
                output_dir=base_dir / "generations",
                run_id="run-123",
                task=self.make_task(),
            )

            metadata_path = generation_dir / "metadata.json"
            latest_path = generation_dir / "exports" / "sft" / "latest.json"
            all_episodes_path = generation_dir / "all_episodes.jsonl"
            checkpoint_path = generation_dir / "checkpoint.json"
            completion_path = generation_dir / GENERATION_COMPLETION_FILENAME
            successful_episode_path = generation_dir / "successful" / "ep-0.json"
            failed_episode_path = generation_dir / "failed" / "ep-1.json"

            self.assertTrue(metadata_path.exists())
            self.assertTrue(latest_path.exists())
            self.assertTrue(all_episodes_path.exists())
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(completion_path.exists())
            self.assertTrue(successful_episode_path.exists())
            self.assertTrue(failed_episode_path.exists())

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["generation_id"], 4)
            self.assertEqual(metadata["training_row_count"], 2)
            self.assertEqual(metadata["target_count_basis"], "training_rows")

            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            sft_rows_path = generation_dir / latest["sft_training_rows_path"]
            sft_rows = [
                json.loads(line)
                for line in sft_rows_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(sft_rows), 2)
            self.assertEqual({row["episode_id"] for row in sft_rows}, {"ep-0"})

    def test_circuit_breaker_breaks_on_consecutive_verification_failures(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=20,
                max_consecutive_verification_failures=3,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                # All episodes fail with verification error
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        idx,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    )
                    for idx in range(20)
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        # Should have stopped after 3 consecutive failures, not run all 20
        self.assertEqual(run_single_episode_mock.call_count, 3)
        self.assertEqual(generation_data.total_episodes_run, 3)

    def test_target_reached_precedes_saturation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=1,
                max_episodes=5,
            )
            task = self.make_task()
            task.saturation_enabled.return_value = True
            task.evaluate_saturation.return_value = SaturationOutcome(
                SaturationDecision.ROLLOVER,
                "test saturation",
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.return_value = self.make_episode(
                    0,
                    row_count=1,
                    success=True,
                )

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        self.assertEqual(generation_data.termination_reason, "target_reached")
        task.on_generation_rollover.assert_not_called()

    def test_sequential_refreshes_expanding_transform_before_saturation(self) -> None:
        class ExpandingTransform:
            @property
            def name(self) -> str:
                return "ExpandingTransform[test]"

            def config(self) -> dict[str, object]:
                return {}

            def transform_export_rows(self, context: object, episodes: list[object]):
                return episodes

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir, target_training_rows=10, max_episodes=2
            )
            task = self.make_task()
            task.training_data_transforms.return_value = (ExpandingTransform(),)
            task.saturation_enabled.return_value = True
            task.evaluate_saturation.return_value = SaturationOutcome(
                SaturationDecision.STOP, "saturation_exhausted"
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode",
                    return_value=self.make_episode(0, row_count=1),
                ),
                patch(
                    "src.execution.generation.count_training_rows", return_value=10
                ) as count_training_rows,
            ):
                self.configure_manager_mock(ContainerManager.return_value)
                result = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        count_training_rows.assert_called_once()
        task.evaluate_saturation.assert_not_called()
        self.assertEqual(result.training_row_count, 10)
        self.assertTrue(result.training_row_count_is_exact)
        self.assertEqual(result.termination_reason, "target_reached")

    def test_sequential_saturation_receives_exact_below_target_count(self) -> None:
        class FilteringTransform:
            @property
            def name(self) -> str:
                return "FilteringTransform[test]"

            def config(self) -> dict[str, object]:
                return {}

            def transform_export_rows(self, context: object, episodes: list[object]):
                return episodes

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir, target_training_rows=10, max_episodes=1
            )
            task = self.make_task()
            task.training_data_transforms.return_value = (FilteringTransform(),)
            task.saturation_enabled.return_value = True
            observed: list[tuple[int, bool]] = []

            def evaluate(data: GenerationData) -> SaturationOutcome:
                observed.append(
                    (data.training_row_count, data.training_row_count_is_exact)
                )
                return SaturationOutcome(SaturationDecision.CONTINUE)

            task.evaluate_saturation.side_effect = evaluate
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode",
                    return_value=self.make_episode(0, row_count=1),
                ),
                patch("src.execution.generation.count_training_rows", return_value=5),
            ):
                self.configure_manager_mock(ContainerManager.return_value)
                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        self.assertEqual(observed, [(5, True)])

    def test_saturation_exhausted_writes_sentinel_and_checkpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=1,
                checkpoint_every_episode=False,
            )
            task = self.make_task()
            task.saturation_enabled.return_value = True
            task.evaluate_saturation.return_value = SaturationOutcome(
                SaturationDecision.STOP,
                "saturation_exhausted",
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.return_value = self.make_episode(
                    0,
                    row_count=0,
                    success=False,
                    score=0.0,
                )

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

            generation_dir = base_dir / "generations" / "generation_0"
            sentinel = generation_dir / "STOP_REASON"
            sentinel_text = sentinel.read_text(encoding="utf-8")
            checkpoint = json.loads(
                (generation_dir / "checkpoint.json").read_text(encoding="utf-8")
            )

        self.assertEqual(generation_data.termination_reason, "saturation_exhausted")
        self.assertEqual(sentinel_text, "saturation_exhausted")
        self.assertEqual(checkpoint["termination_reason"], "saturation_exhausted")

    def test_run_climbs_across_rollover(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=1,
                max_episodes=2,
            )
            task = self.make_task()
            task.saturation_enabled.return_value = True
            task.evaluate_saturation.return_value = SaturationOutcome(
                SaturationDecision.ROLLOVER,
                "test saturation",
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(0, row_count=0, success=False, score=0.0),
                    self.make_episode(1, row_count=1, success=True),
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        task.on_generation_rollover.assert_called_once()
        self.assertEqual(generation_data.training_row_count, 1)
        self.assertEqual(generation_data.termination_reason, "target_reached")

    def test_rollover_checkpoint_includes_updated_task_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=1,
                checkpoint_every_episode=True,
            )
            task = self.make_task()
            task.saturation_enabled.return_value = True
            before_state = {"saturation_epoch_start_count": 0}
            after_state = {"saturation_epoch_start_count": 1}
            task.generation_checkpoint_state.return_value = before_state
            task.evaluate_saturation.return_value = SaturationOutcome(
                SaturationDecision.ROLLOVER,
                "test saturation",
            )

            def rollover(*_args: object, **_kwargs: object) -> None:
                task.generation_checkpoint_state.return_value = after_state

            task.on_generation_rollover.side_effect = rollover
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.return_value = self.make_episode(
                    0,
                    row_count=0,
                    success=False,
                    score=0.0,
                )

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

            checkpoint = json.loads(
                (
                    base_dir / "generations" / "generation_0" / "checkpoint.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(checkpoint["task_checkpoint_state"], after_state)

    def test_circuit_breaker_resets_on_successful_episode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=10,
                max_consecutive_verification_failures=3,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                # 2 failures, then a success, then 2 more failures — should NOT trip
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        0,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(
                        1,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(2, row_count=1, success=True),
                    self.make_episode(
                        3,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(
                        4,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(5, row_count=1, success=True),
                    self.make_episode(
                        6,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(
                        7,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(
                        8,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(9, row_count=1, success=True),
                ]

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        # Trips at episode 8 (3 consecutive failures: 6, 7, 8)
        self.assertEqual(run_single_episode_mock.call_count, 9)

    def test_run_generation_recovers_from_broken_container_without_recording_it(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, max_episodes=1)
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                populated = (
                    PopulationOutcome(
                        results=[
                            PopulationResult(
                                container_id="container-a",
                                variation_name="variation_1_heavy",
                                description="heavy variation",
                                success=True,
                            )
                        ],
                    ),
                    True,
                )
                manager.populate_with_task.side_effect = [
                    TaskEnvironmentError(
                        "environment error",
                        container_ids=["container-b"],
                    ),
                    populated,
                ]
                run_single_episode_mock.return_value = self.make_episode(
                    0, row_count=1, success=True
                )

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        # Targeted rebuild when container_ids are provided
        manager.rebuild_containers.assert_called_once_with(["container-b"])
        manager.rebuild.assert_not_called()
        self.assertEqual(manager.populate_with_task.call_count, 2)
        self.assertEqual(run_single_episode_mock.call_count, 1)
        self.assertEqual(run_single_episode_mock.call_args.kwargs["episode_index"], 0)
        self.assertEqual(
            run_single_episode_mock.call_args.kwargs["population_outcome"], populated[0]
        )
        self.assertEqual(
            run_single_episode_mock.call_args.kwargs["verified"], populated[1]
        )
        self.assertEqual(generation_data.total_episodes_run, 1)

    def test_run_generation_aborts_after_three_consecutive_rebuild_failures(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, max_episodes=5)
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                manager.populate_with_task.side_effect = [
                    TaskEnvironmentError("environment error"),
                    TaskEnvironmentError("environment error"),
                    TaskEnvironmentError("environment error"),
                ]
                manager.rebuild.side_effect = [
                    RuntimeError("first rebuild failed"),
                    RuntimeError("second rebuild failed"),
                    RuntimeError("third rebuild failed"),
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(manager.rebuild.call_count, 3)
        self.assertEqual(run_single_episode_mock.call_count, 0)
        self.assertEqual(generation_data.total_episodes_run, 0)

    def test_run_generation_bounds_persistent_population_environment_errors(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, max_episodes=5)
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                manager.populate_with_task.side_effect = TaskEnvironmentError(
                    "persistent environment error"
                )

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(manager.populate_with_task.call_count, 3)
        self.assertEqual(manager.rebuild.call_count, 3)
        run_single_episode_mock.assert_not_called()
        self.assertEqual(generation_data.termination_reason, "force_stop")

    def test_run_generation_guards_execution_rebuild_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, max_episodes=1)
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                manager.rebuild.side_effect = [RuntimeError("transient rebuild failure")]
                run_single_episode_mock.side_effect = [
                    TaskEnvironmentError("execution environment error"),
                    self.make_episode(0, row_count=1, success=True),
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(run_single_episode_mock.call_count, 2)
        self.assertEqual(generation_data.total_episodes_run, 1)

    def test_parallel_persistent_execution_environment_errors_retire_slot(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=100,
                max_episodes=5,
            )
            manager = MagicMock()
            self.configure_manager_mock(manager)
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=SlotCircuitBreaker(),
                cache_dir=base_dir / "slot-0",
            )

            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    side_effect=TaskEnvironmentError("persistent environment error"),
                ) as run_single_episode_mock,
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(run_single_episode_mock.call_count, 3)
        self.assertEqual(manager.rebuild.call_count, 3)
        self.assertTrue(slot.circuit_breaker.is_tripped())
        self.assertEqual(result.total_episodes_run, 0)
        self.assertEqual(result.termination_reason, "force_stop")

    def test_run_generation_reuses_variation_after_recovery_retry(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                max_episodes=1,
                variation_strategy="random",
                variation_random_seed=7,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
                patch(
                    "src.execution.generation.select_variation_index",
                    side_effect=[4, 1],
                ) as select_variation_index_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                manager.populate_with_task.side_effect = [
                    TaskEnvironmentError("environment error"),
                    manager.populate_with_task.return_value,
                ]
                run_single_episode_mock.return_value = self.make_episode(
                    0, row_count=1, success=True
                )

                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(select_variation_index_mock.call_count, 1)
        self.assertEqual(run_single_episode_mock.call_args.kwargs["variation_index"], 4)

    def test_seeded_sequential_resume_advances_past_committed_draws(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=6,
                variation_strategy="random",
                variation_random_seed=731,
                resume_from_checkpoint=True,
            )
            task = self.make_task()
            generation_dir = base_dir / "generations" / "generation_0"
            recipe_hash = build_export_recipe(()).recipe_hash
            identity = GenerationIdentity(
                generation_id=0,
                run_id="run-123",
                task_name=task.name,
                task_identity=f"{type(task).__module__}.{type(task).__qualname__}",
                export_recipe_hash=recipe_hash,
                execution_mode="sequential",
                **serving_lineage_from_config(config, 0),
            )
            ensure_generation_manifest(
                generation_dir / GENERATION_MANIFEST_FILENAME,
                identity,
            )
            existing = [
                self.make_episode(index, row_count=0, success=False, score=0.0)
                for index in range(4)
            ]
            publish_episode_journal(existing, generation_dir / "all_episodes.jsonl")
            save_generation_checkpoint(
                {
                    **checkpoint_payload_identity(identity),
                    "committed_episode_indices": [0, 1, 2, 3],
                    "next_episode_index": 4,
                    "export_recipe_hash": recipe_hash,
                },
                generation_dir / "checkpoint.json",
            )

            captured: list[int] = []

            def run_episode(*args: object, **kwargs: object) -> EpisodeTrajectory:
                captured.append(int(kwargs["variation_index"]))
                return self.make_episode(
                    int(kwargs["episode_index"]),
                    row_count=0,
                    success=False,
                    score=0.0,
                )

            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode",
                    side_effect=run_episode,
                ),
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        rng = random.Random(731)
        expected = [rng.randrange(len(task.list_variations())) for _ in range(6)]
        self.assertEqual(captured, expected[4:])

    def test_scheduled_restart_resets_verification_failure_counter(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(
                base_dir,
                target_training_rows=100,
                max_episodes=4,
                container_restart_interval=2,
                container_rebuild_interval=99,
                max_consecutive_verification_failures=3,
            )
            with (
                patch("src.execution.generation.ContainerManager") as ContainerManager,
                patch(
                    "src.execution.generation.run_single_episode"
                ) as run_single_episode_mock,
            ):
                manager = ContainerManager.return_value
                self.configure_manager_mock(manager)
                run_single_episode_mock.side_effect = [
                    self.make_episode(
                        0,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(
                        1,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(
                        2,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                    self.make_episode(
                        3,
                        row_count=0,
                        success=False,
                        score=0.0,
                        error_message="container population verification failed",
                    ),
                ]

                generation_data = run_generation(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(run_single_episode_mock.call_count, 4)
        self.assertEqual(manager.restart.call_count, 1)
        self.assertEqual(generation_data.total_episodes_run, 4)

    def test_run_episode_exits_early_when_stop_event_is_set(self) -> None:
        """When stop_event is set, run_episode should exit mid-episode
        with partial=True instead of running the full action budget."""
        from src.execution.episode import run_episode

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            base_task = ValidStubTask({})
            task = self.make_task()
            task.prompt_spec.side_effect = base_task.prompt_spec
            task.parse_response.side_effect = base_task.parse_response
            task.execute_capability.side_effect = base_task.execute_capability
            task.measure_initial_state.return_value = {}
            task.finalize_episode.return_value = TaskReward(
                value=0.0,
                success=False,
                breakdown={},
            )
            variation = base_task.list_variations()[0]

            genner = MagicMock(collector=None)
            genner.plist_completion.return_value = MagicMock(
                is_ok=lambda: True,
                ok=lambda: MagicMock(content="delegate to explorer"),
            )

            stop_event = threading.Event()
            stop_event.set()  # Pre-set: episode should exit immediately

            containers = [MagicMock()]
            containers[0].id = "container-a"
            containers[0].name = "wrong-service-name"
            agent_container = MagicMock()
            agent_container.id = "agent-container"
            container_manager = MagicMock()
            container_manager.get_service.return_value = agent_container

            with patch(
                "src.execution.episode.construct_harness"
            ) as MockConstructHarness:
                from src.harness.transcript import HarnessTranscript
                from src.task.types import EpisodeArtifacts

                mock_harness = MagicMock()
                mock_harness.run_episode.return_value = HarnessTranscript(
                    artifacts=EpisodeArtifacts(),
                    llm_turns=0,
                    termination_reason="cancel_event set",
                    termination_category="wall_clock",
                )
                MockConstructHarness.return_value = mock_harness

                rt = self.make_backend_runtime(
                    config,
                    task=task,
                    genner=genner,
                    docker_client=MagicMock(),
                )
                req = self.make_episode_request(
                    variation=variation,
                    container_manager=container_manager,
                    containers=containers,
                    agent_container=agent_container,
                    stop_event=stop_event,
                )
                result = run_episode(rt, req)

            self.assertTrue(result.partial)
            self.assertIn("cancelled", str(result.error_message))

    def test_run_episode_uses_stop_reason_when_stop_event_is_set(self) -> None:
        from src.execution.episode import run_episode

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            base_task = ValidStubTask({})
            task = self.make_task()
            task.prompt_spec.side_effect = base_task.prompt_spec
            task.parse_response.side_effect = base_task.parse_response
            task.execute_capability.side_effect = base_task.execute_capability
            task.measure_initial_state.return_value = {}
            task.finalize_episode.return_value = TaskReward(
                value=0.0,
                success=False,
                breakdown={},
            )
            variation = base_task.list_variations()[0]

            stop_event = threading.Event()
            stop_reason = StopReason()
            stop_reason.set("deadline_exceeded")
            stop_event.set()

            containers = [MagicMock()]
            containers[0].id = "container-a"
            agent_container = MagicMock()
            agent_container.id = "agent-container"
            container_manager = MagicMock()
            container_manager.get_service.return_value = agent_container

            with patch(
                "src.execution.episode.construct_harness"
            ) as MockConstructHarness:
                from src.harness.transcript import HarnessTranscript
                from src.task.types import EpisodeArtifacts

                mock_harness = MagicMock()
                mock_harness.run_episode.return_value = HarnessTranscript(
                    artifacts=EpisodeArtifacts(),
                    llm_turns=0,
                    termination_reason="cancel_event set",
                    termination_category="wall_clock",
                )
                MockConstructHarness.return_value = mock_harness

                rt = self.make_backend_runtime(
                    config,
                    task=task,
                    genner=MagicMock(collector=None),
                    docker_client=MagicMock(),
                )
                req = self.make_episode_request(
                    variation=variation,
                    container_manager=container_manager,
                    containers=containers,
                    agent_container=agent_container,
                    stop_event=stop_event,
                    stop_reason=stop_reason,
                )
                result = run_episode(rt, req)

        self.assertIn("deadline_exceeded", str(result.error_message))

    def test_run_episode_marks_context_overflow_benignly(self) -> None:
        from src.execution.episode import run_episode

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            base_task = ValidStubTask({})
            task = self.make_task()
            task.prompt_spec.side_effect = base_task.prompt_spec
            task.parse_response.side_effect = base_task.parse_response
            task.execute_capability.side_effect = base_task.execute_capability
            task.measure_initial_state.return_value = {}
            task.finalize_episode.return_value = TaskReward(
                value=0.0,
                success=False,
                breakdown={},
            )
            variation = base_task.list_variations()[0]

            containers = [MagicMock()]
            containers[0].id = "container-a"
            agent_container = MagicMock()
            agent_container.id = "agent-container"
            container_manager = MagicMock()
            container_manager.get_service.return_value = agent_container

            with patch(
                "src.execution.episode.construct_harness"
            ) as MockConstructHarness:
                from src.harness.transcript import HarnessTranscript
                from src.task.types import EpisodeArtifacts

                mock_harness = MagicMock()
                mock_harness.run_episode.return_value = HarnessTranscript(
                    artifacts=EpisodeArtifacts(),
                    llm_turns=0,
                    termination_reason=(
                        "context_overflow: maximum context length exceeded"
                    ),
                    termination_category="context_overflow",
                )
                MockConstructHarness.return_value = mock_harness

                rt = self.make_backend_runtime(
                    config,
                    task=task,
                    genner=MagicMock(collector=None),
                    docker_client=MagicMock(),
                )
                req = self.make_episode_request(
                    variation=variation,
                    container_manager=container_manager,
                    containers=containers,
                    agent_container=agent_container,
                )
                result = run_episode(rt, req)

        self.assertTrue(result.partial)
        self.assertEqual(result.error_category, "context_overflow")
        self.assertIn("context_overflow:", str(result.error_message))

    @patch("src.execution.episode_runner.run_episode")
    def test_run_single_episode_retains_error_category(
        self, run_episode_mock: MagicMock
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            manager = MagicMock()
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="variation_1_heavy",
                            description="heavy variation",
                            success=True,
                        )
                    ],
                ),
                True,
            )

            run_episode_mock.return_value = self.make_episode_outcome(
                episode_id="ep-0",
                success=False,
                partial=True,
                error_message="context_overflow: maximum context length exceeded",
                error_category="context_overflow",
                score=0.0,
            )

            trajectory = run_single_episode(
                genner=MagicMock(),
                docker_client=MagicMock(),
                container_manager=manager,
                config=config,
                generation_id=0,
                episode_index=0,
                variation_index=0,
                run_id="run-123",
                task=self.make_task(),
            )

        self.assertEqual(trajectory.error_category, "context_overflow")

    def test_run_generation_parallel_records_deadline_termination_reason(self) -> None:
        class FakeThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.join_calls: list[float | None] = []
                self._alive = True

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                self.join_calls.append(timeout)
                self._alive = False

            def is_alive(self) -> bool:
                return self._alive

        class ImmediateTimer:
            def __init__(self, _interval: float, callback) -> None:
                self._callback = callback
                self.daemon = False
                self.cancelled = False

            def start(self) -> None:
                self._callback()

            def cancel(self) -> None:
                self.cancelled = True

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=2,
                generation_timeout_s=1,
            )
            fake_slots = [
                MagicMock(slot_id=i, circuit_breaker=MagicMock()) for i in range(2)
            ]

            with (
                patch("src.parallel.create_worker_slots", return_value=fake_slots),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.threading.Thread",
                    side_effect=[FakeThread(), FakeThread()],
                ),
                patch(
                    "src.execution.parallel.threading.Timer",
                    side_effect=lambda interval, callback: ImmediateTimer(
                        interval, callback
                    ),
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(result.termination_reason, "deadline_exceeded")

    def test_parallel_waits_for_workers_before_finalization(self) -> None:
        class LifecycleThread:
            def __init__(self) -> None:
                self.join_calls: list[float | None] = []
                self._alive = False

            def start(self) -> None:
                self._alive = True

            def join(self, timeout: float | None = None) -> None:
                self.join_calls.append(timeout)
                if timeout is None:
                    self._alive = False

            def is_alive(self) -> bool:
                return self._alive

        class ImmediateTimer:
            def __init__(self, _interval: float, callback) -> None:
                self._callback = callback
                self.daemon = False

            def start(self) -> None:
                self._callback()

            def cancel(self) -> None:
                return None

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                generation_timeout_s=1,
            )
            slot = MagicMock(slot_id=0, circuit_breaker=MagicMock())
            worker_thread = LifecycleThread()
            finalization_observed_alive: list[bool] = []

            def record_finalization(*args: object, **kwargs: object) -> None:
                del args, kwargs
                finalization_observed_alive.append(worker_thread.is_alive())

            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch(
                    "src.parallel.teardown_worker_slots",
                    side_effect=record_finalization,
                ),
                patch(
                    "src.execution.parallel.threading.Thread",
                    return_value=worker_thread,
                ),
                patch(
                    "src.execution.parallel.threading.Timer",
                    side_effect=lambda interval, callback: ImmediateTimer(
                        interval, callback
                    ),
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
                patch(
                    "src.execution.parallel.publish_episode_journal",
                    side_effect=record_finalization,
                ),
                patch(
                    "src.execution.parallel._save_parallel_checkpoint",
                    side_effect=record_finalization,
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertIn(None, worker_thread.join_calls)
        self.assertFalse(any(finalization_observed_alive))
        self.assertEqual(result.termination_reason, "deadline_exceeded")

    def test_run_generation_parallel_cleans_up_when_thread_start_fails(self) -> None:
        class StartedThread:
            def __init__(self) -> None:
                self.join_calls: list[float | None] = []
                self._alive = False

            def start(self) -> None:
                self._alive = True

            def join(self, timeout: float | None = None) -> None:
                self.join_calls.append(timeout)
                if timeout is None:
                    self._alive = False

            def is_alive(self) -> bool:
                return self._alive

        class FailingStartThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("thread start failed")

            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return False

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, parallel_episodes=2)
            fake_slots = [
                MagicMock(slot_id=i, circuit_breaker=MagicMock()) for i in range(2)
            ]
            started_thread = StartedThread()
            failing_thread = FailingStartThread()
            teardown_observed_alive: list[bool] = []
            metrics = MagicMock()
            utilization_summary = metrics.stop_utilization_sampling.return_value
            utilization_summary.peak_gpu_utilization_pct = None
            utilization_summary.peak_kv_cache_usage_pct = None
            utilization_summary.peak_num_requests_running = None
            utilization_summary.peak_num_requests_waiting = None

            with (
                patch("src.parallel.create_worker_slots", return_value=fake_slots),
                patch(
                    "src.parallel.teardown_worker_slots",
                    side_effect=lambda _slots: teardown_observed_alive.append(
                        started_thread.is_alive()
                    ),
                ) as teardown_worker_slots,
                patch(
                    "src.execution.parallel.threading.Thread",
                    side_effect=[started_thread, failing_thread],
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                    run_generation_parallel(
                        genner=MagicMock(),
                        docker_client=MagicMock(),
                        config=config,
                        generation_id=0,
                        run_id="run-123",
                        metrics_collector=metrics,
                        task=self.make_task(),
                    )

        metrics.stop_utilization_sampling.assert_called_once()
        metrics.flush.assert_called_once()
        self.assertEqual(started_thread.join_calls, [None])
        self.assertEqual(teardown_observed_alive, [False])
        teardown_worker_slots.assert_called_once_with(fake_slots)

    def test_parallel_checkpoint_failure_after_commit_keeps_commit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=100,
                max_episodes=1,
                checkpoint_every_episode=True,
            )
            task = self.make_task()
            manager = MagicMock()
            self.configure_manager_mock(manager)
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=SlotCircuitBreaker(max_consecutive_failures=10),
                cache_dir=base_dir / "slot-0",
            )
            budget = IntrinsicBudgetController(
                max_episodes=1,
                max_bootstrap_episodes=None,
                committed=[],
            )
            budget.commit = MagicMock(wraps=budget.commit)
            budget.abandon = MagicMock(wraps=budget.abandon)
            checkpoint_calls = 0

            def fail_first_checkpoint(*args: object, **kwargs: object) -> None:
                nonlocal checkpoint_calls
                del args, kwargs
                checkpoint_calls += 1
                if checkpoint_calls == 1:
                    raise RuntimeError("checkpoint write failed")

            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.IntrinsicBudgetController",
                    return_value=budget,
                ),
                patch(
                    "src.execution.parallel.run_single_episode",
                    return_value=self.make_episode(0, row_count=0, success=False),
                ),
                patch(
                    "src.execution.parallel._save_parallel_checkpoint",
                    side_effect=fail_first_checkpoint,
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        budget.commit.assert_called_once()
        budget.abandon.assert_not_called()
        self.assertEqual(budget.in_flight, 0)
        self.assertEqual(result.total_episodes_run, 1)

    def test_stateless_parallel_checkpoint_does_not_drain_in_flight_episode(
        self,
    ) -> None:
        class StatelessTask:
            def __init__(self, delegate: MagicMock) -> None:
                self._delegate = delegate

            def __getattr__(self, name: str):
                return getattr(self._delegate, name)

            def episode_commit_enabled(self) -> bool:
                return False

            def generation_checkpoint_state(self) -> None:
                return None

        both_running = threading.Barrier(2)
        slow_started = threading.Event()
        release_slow = threading.Event()
        slow_finished = threading.Event()
        slow_wait_timed_out = threading.Event()
        checkpoint_observations: list[tuple[bool, bool]] = []

        def fake_run_single_episode(
            *args: object, **kwargs: object
        ) -> EpisodeTrajectory:
            del args
            episode_index = int(kwargs["episode_index"])
            if episode_index == 1:
                slow_started.set()
            both_running.wait(timeout=5)
            if episode_index == 1:
                if not release_slow.wait(timeout=5):
                    slow_wait_timed_out.set()
                slow_finished.set()
            return self.make_episode(
                episode_index,
                row_count=0,
                success=False,
                score=0.0,
            )

        def observe_checkpoint(*args: object, **kwargs: object) -> None:
            del args
            snapshot = kwargs["generation_data"]
            if snapshot.total_episodes_run == 1:
                checkpoint_observations.append(
                    (slow_started.is_set(), slow_finished.is_set())
                )
                release_slow.set()

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=2,
                target_training_rows=100,
                max_episodes=1,
                checkpoint_every_episode=True,
            )
            slots: list[WorkerSlot] = []
            for slot_id in range(2):
                manager = MagicMock()
                self.configure_manager_mock(manager)
                slots.append(
                    WorkerSlot(
                        slot_id=slot_id,
                        container_manager=manager,
                        docker_client=MagicMock(),
                        circuit_breaker=SlotCircuitBreaker(),
                        cache_dir=base_dir / f"slot-{slot_id}",
                    )
                )

            try:
                with (
                    patch("src.parallel.create_worker_slots", return_value=slots),
                    patch("src.parallel.teardown_worker_slots"),
                    patch(
                        "src.execution.parallel.run_single_episode",
                        side_effect=fake_run_single_episode,
                    ),
                    patch(
                        "src.execution.parallel._save_parallel_checkpoint",
                        side_effect=observe_checkpoint,
                    ) as save_checkpoint,
                    patch(
                        "src.execution.parallel._scoped_parallel_logging",
                        return_value=nullcontext(),
                    ),
                ):
                    result = run_generation_parallel(
                        genner=MagicMock(),
                        docker_client=MagicMock(),
                        config=config,
                        generation_id=0,
                        run_id="run-123",
                        task=StatelessTask(self.make_task()),
                    )
            finally:
                release_slow.set()

        self.assertFalse(
            slow_wait_timed_out.is_set(),
            "stateless checkpoint waited for the in-flight episode",
        )
        self.assertEqual(checkpoint_observations, [(True, False)])
        self.assertEqual(result.total_episodes_run, 2)
        self.assertGreaterEqual(save_checkpoint.call_count, 2)

    def test_parallel_apply_failure_accounts_journal_and_stops(self) -> None:
        class CommitTask:
            def __init__(self, delegate: MagicMock) -> None:
                self._delegate = delegate
                self.apply_calls = 0
                self.recovered: list[dict[str, str]] = []

            def __getattr__(self, name: str):
                return getattr(self._delegate, name)

            def episode_commit_enabled(self) -> bool:
                return True

            def parallel_checkpoint_safe(self) -> bool:
                return True

            def generation_checkpoint_state(self):
                return None

            def apply_episode_commit(self, token: str, digest: str) -> None:
                self.apply_calls += 1
                raise RuntimeError(f"transient apply failure for {token}:{digest}")

            def recover_episode_commits(self, journaled: dict[str, str]) -> None:
                self.recovered.append(dict(journaled))

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=100,
                max_episodes=2,
                checkpoint_every_episode=False,
            )
            task = CommitTask(self.make_task())
            manager = MagicMock()
            self.configure_manager_mock(manager)
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=SlotCircuitBreaker(max_consecutive_failures=10),
                cache_dir=base_dir / "slot-0",
            )
            budget = IntrinsicBudgetController(
                max_episodes=2,
                max_bootstrap_episodes=None,
                committed=[],
            )
            budget.commit = MagicMock(wraps=budget.commit)
            budget.abandon = MagicMock(wraps=budget.abandon)
            episode = self.make_episode(0, row_count=0, success=False)
            episode.task_commit_token = "token-0"
            episode.task_commit_digest = "digest-0"

            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.IntrinsicBudgetController",
                    return_value=budget,
                ),
                patch(
                    "src.execution.parallel.run_single_episode",
                    return_value=episode,
                ) as run_episode,
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        run_episode.assert_called_once()
        budget.commit.assert_called_once_with(episode)
        budget.abandon.assert_not_called()
        self.assertEqual(result.total_episodes_run, 1)
        self.assertEqual(result.termination_reason, "task_commit_apply_failed")
        self.assertIn({"token-0": "digest-0"}, task.recovered)

    def test_run_generation_parallel_drains_reserved_workers_at_max_episodes(
        self,
    ) -> None:
        """A cap of one permits two completions when two workers reserved work."""
        observed_stop_events: list[bool] = []

        def fake_run_single_episode(
            *args: object, **kwargs: object
        ) -> EpisodeTrajectory:
            stop_event = kwargs["stop_event"]
            assert isinstance(stop_event, threading.Event)
            observed_stop_events.append(stop_event.wait(timeout=0.5))
            return self.make_episode(
                int(kwargs["episode_index"]),
                row_count=0,
                success=False,
                score=0.0,
            )

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=2,
                target_training_rows=100,
                max_episodes=1,
            )
            slots: list[WorkerSlot] = []
            for slot_id in range(2):
                manager = MagicMock()
                self.configure_manager_mock(manager)
                manager.get_containers.return_value = [
                    MagicMock(id=f"container-{slot_id}")
                ]
                slots.append(
                    WorkerSlot(
                        slot_id=slot_id,
                        container_manager=manager,
                        docker_client=MagicMock(),
                        circuit_breaker=SlotCircuitBreaker(),
                        cache_dir=base_dir / f"slot-{slot_id}",
                    )
                )

            with (
                patch("src.parallel.create_worker_slots", return_value=slots),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    side_effect=fake_run_single_episode,
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

        self.assertEqual(result.termination_reason, "max_episodes")
        self.assertEqual(result.total_episodes_run, 2)
        self.assertLessEqual(result.total_episodes_run, 1 + 2 - 1)
        self.assertEqual(observed_stop_events, [False, False])

    def test_parallel_bootstrap_cap_drains_only_reserved_workers(self) -> None:
        """A bootstrap cap of one permits three pre-reserved completions."""
        barrier = threading.Barrier(3)

        def fake_run_single_episode(
            *args: object, **kwargs: object
        ) -> EpisodeTrajectory:
            barrier.wait(timeout=2)
            return self.make_episode(
                int(kwargs["episode_index"]),
                row_count=0,
                success=False,
                score=0.0,
                bootstrap_active=True,
            )

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=3,
                target_training_rows=100,
                max_episodes=20,
                max_bootstrap_episodes=1,
                checkpoint_every_episode=True,
            )
            slots: list[WorkerSlot] = []
            for slot_id in range(3):
                manager = MagicMock()
                self.configure_manager_mock(manager)
                slots.append(
                    WorkerSlot(
                        slot_id=slot_id,
                        container_manager=manager,
                        docker_client=MagicMock(),
                        circuit_breaker=SlotCircuitBreaker(),
                        cache_dir=base_dir / f"slot-{slot_id}",
                    )
                )

            with (
                patch("src.parallel.create_worker_slots", return_value=slots),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    side_effect=fake_run_single_episode,
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=self.make_task(),
                )

            generation_dir = base_dir / "generations" / "generation_0"
            checkpoint = json.loads(
                (generation_dir / "checkpoint.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.termination_reason, "max_bootstrap_episodes")
        self.assertEqual(result.total_episodes_run, 3)
        self.assertLessEqual(result.total_episodes_run, 1 + 3 - 1)
        self.assertEqual(checkpoint["bootstrap_episodes_run"], 3)
        self.assertEqual(checkpoint["committed_episode_indices"], [0, 1, 2])

    def test_run_generation_parallel_uses_endpoint_pool_lease_and_releases(
        self,
    ) -> None:
        from src.backend.endpoint_pool import EndpointPool, EndpointState
        from src.genner.Base import Genner

        class TinyGenner(Genner):
            def plist_completion(self, messages, *, tools=None, tool_choice=None):
                raise AssertionError("not used by this test")

            @staticmethod
            def get_usage_info(response):
                return None

        observed_endpoint_ids: list[str | None] = []

        def fake_run_single_episode(runtime: BackendRuntime, **kwargs: object):
            observed_endpoint_ids.append(getattr(runtime.genner, "endpoint_id", None))
            return self.make_episode(
                int(kwargs["episode_index"]),
                row_count=0,
                success=False,
                score=0.0,
            )

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=100,
                max_episodes=1,
            )
            manager = MagicMock()
            self.configure_manager_mock(manager)
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=SlotCircuitBreaker(),
                cache_dir=base_dir / "slot-0",
            )
            pool = EndpointPool(
                [
                    EndpointState(
                        endpoint_id="ep0",
                        base_url="http://ep0:8000/v1",
                        models_url="http://ep0:8000/v1/models",
                        metrics_url="http://ep0:8000/metrics",
                        capacity=1,
                        genner=TinyGenner("vllm"),
                    )
                ]
            )
            rt = self.make_backend_runtime(config, endpoint_pool=pool)

            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    side_effect=fake_run_single_episode,
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(rt, generation_id=0)

        self.assertEqual(result.total_episodes_run, 1)
        self.assertEqual(observed_endpoint_ids, ["ep0"])
        self.assertEqual(pool.total_in_flight(), 0)

    def test_parallel_rollover_waits_for_in_flight_episode(self) -> None:
        slow_started = threading.Event()
        slow_finished = threading.Event()
        rollover_observed_slow_finished: list[bool] = []

        def fake_run_single_episode(
            *args: object, **kwargs: object
        ) -> EpisodeTrajectory:
            episode_index = int(kwargs["episode_index"])
            if episode_index == 0:
                slow_started.set()
                time.sleep(0.2)
                slow_finished.set()
            else:
                slow_started.wait(timeout=1.0)
            return self.make_episode(
                episode_index,
                row_count=0,
                success=False,
                score=0.0,
            )

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=2,
                target_training_rows=100,
                max_episodes=3,
            )
            task = self.make_task()
            task.saturation_enabled.return_value = True

            def evaluate_saturation(data: GenerationData) -> SaturationOutcome:
                if rollover_observed_slow_finished:
                    return SaturationOutcome(SaturationDecision.CONTINUE)
                if data.total_episodes_run >= 1:
                    return SaturationOutcome(
                        SaturationDecision.ROLLOVER,
                        "test saturation",
                    )
                return SaturationOutcome(SaturationDecision.CONTINUE)

            def on_generation_rollover(*_args: object, **_kwargs: object) -> None:
                rollover_observed_slow_finished.append(slow_finished.is_set())

            task.evaluate_saturation.side_effect = evaluate_saturation
            task.on_generation_rollover.side_effect = on_generation_rollover
            slots: list[WorkerSlot] = []
            for slot_id in range(2):
                manager = MagicMock()
                self.configure_manager_mock(manager)
                manager.get_containers.return_value = [
                    MagicMock(id=f"container-{slot_id}")
                ]
                slots.append(
                    WorkerSlot(
                        slot_id=slot_id,
                        container_manager=manager,
                        docker_client=MagicMock(),
                        circuit_breaker=SlotCircuitBreaker(),
                        cache_dir=base_dir / f"slot-{slot_id}",
                    )
                )

            with (
                patch("src.parallel.create_worker_slots", return_value=slots),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    side_effect=fake_run_single_episode,
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        self.assertEqual(rollover_observed_slow_finished, [True])

    def test_parallel_disabled_saturation_does_not_snapshot(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=100,
                max_episodes=1,
            )
            task = self.make_task()
            task.saturation_enabled.return_value = False
            task.evaluate_saturation.side_effect = AssertionError(
                "disabled saturation should not evaluate"
            )
            manager = MagicMock()
            self.configure_manager_mock(manager)
            manager.get_containers.return_value = [MagicMock(id="container-0")]
            slots = [
                WorkerSlot(
                    slot_id=0,
                    container_manager=manager,
                    docker_client=MagicMock(),
                    circuit_breaker=SlotCircuitBreaker(),
                    cache_dir=base_dir / "slot-0",
                )
            ]

            with (
                patch("src.parallel.create_worker_slots", return_value=slots),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    return_value=self.make_episode(0, row_count=0, success=False),
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                run_generation_parallel(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    config=config,
                    generation_id=0,
                    run_id="run-123",
                    task=task,
                )

        task.evaluate_saturation.assert_not_called()

    def test_run_generation_parallel_releases_endpoint_lease_on_exception(self) -> None:
        from src.backend.endpoint_pool import EndpointPool, EndpointState
        from src.genner.Base import Genner

        class TinyGenner(Genner):
            def plist_completion(self, messages, *, tools=None, tool_choice=None):
                raise AssertionError("not used by this test")

            @staticmethod
            def get_usage_info(response):
                return None

        def fake_run_single_episode(runtime: BackendRuntime, **kwargs: object):
            raise RuntimeError("episode crashed")

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=100,
                max_episodes=1,
            )
            manager = MagicMock()
            self.configure_manager_mock(manager)
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=SlotCircuitBreaker(max_consecutive_failures=10),
                cache_dir=base_dir / "slot-0",
            )
            pool = EndpointPool(
                [
                    EndpointState(
                        endpoint_id="ep0",
                        base_url="http://ep0:8000/v1",
                        models_url="http://ep0:8000/v1/models",
                        metrics_url="http://ep0:8000/metrics",
                        capacity=1,
                        genner=TinyGenner("vllm"),
                    )
                ]
            )
            rt = self.make_backend_runtime(config, endpoint_pool=pool)

            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    side_effect=fake_run_single_episode,
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                run_generation_parallel(rt, generation_id=0)

        self.assertEqual(pool.total_in_flight(), 0)

    def test_endpoint_unavailable_bypasses_slot_breaker_and_marks_endpoint(
        self,
    ) -> None:
        from src.backend.endpoint_pool import EndpointPool, EndpointState
        from src.genner.Base import Genner

        class TinyGenner(Genner):
            def plist_completion(self, messages, *, tools=None, tool_choice=None):
                raise AssertionError("not used by this test")

            @staticmethod
            def get_usage_info(response):
                return None

        def fake_run_single_episode(runtime: BackendRuntime, **kwargs: object):
            return self.make_episode(
                int(kwargs["episode_index"]),
                row_count=0,
                success=False,
                score=0.0,
                error_message="inference endpoint unavailable: connection refused",
                error_category="endpoint_unavailable",
            )

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "compose").mkdir()
            config = self.make_config(
                base_dir,
                parallel_episodes=1,
                target_training_rows=100,
                max_episodes=1,
            )
            manager = MagicMock()
            self.configure_manager_mock(manager)
            breaker = MagicMock()
            breaker.is_tripped.return_value = False
            breaker.is_verification_tripped.return_value = False
            slot = WorkerSlot(
                slot_id=0,
                container_manager=manager,
                docker_client=MagicMock(),
                circuit_breaker=breaker,
                cache_dir=base_dir / "slot-0",
            )
            pool = EndpointPool(
                [
                    EndpointState(
                        endpoint_id="ep0",
                        base_url="http://ep0:8000/v1",
                        models_url="http://ep0:8000/v1/models",
                        metrics_url="http://ep0:8000/metrics",
                        capacity=1,
                        genner=TinyGenner("vllm"),
                    )
                ]
            )
            rt = self.make_backend_runtime(config, endpoint_pool=pool)

            with (
                patch("src.parallel.create_worker_slots", return_value=[slot]),
                patch("src.parallel.teardown_worker_slots"),
                patch(
                    "src.execution.parallel.run_single_episode",
                    side_effect=fake_run_single_episode,
                ),
                patch(
                    "src.execution.parallel._scoped_parallel_logging",
                    return_value=nullcontext(),
                ),
            ):
                result = run_generation_parallel(rt, generation_id=0)

        self.assertEqual(result.total_episodes_run, 1)
        self.assertFalse(pool.is_healthy("ep0"))
        breaker.record_failure.assert_not_called()

if __name__ == "__main__":
    unittest.main()
