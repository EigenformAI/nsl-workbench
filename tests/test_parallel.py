"""Tests for parallel episode execution infrastructure.

TDD: These tests are written before the implementation in src/parallel.py.
"""

import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.task.types import PopulationOutcome, PopulationResult
from src.typing.trajectory import EpisodeTrajectory, GenerationData
from src.execution.episode import EpisodeOutcome


def _make_episode(
    episode_id: str = "ep-1",
    generation_id: int = 0,
    episode_index: int = 0,
    success: bool = True,
    score: float = 100.0,
    prompt_responses: list | None = None,
) -> EpisodeTrajectory:
    return EpisodeTrajectory(
        episode_id=episode_id,
        generation_id=generation_id,
        episode_index=episode_index,
        prompt_responses=prompt_responses or [{"prompt": "p", "completion": "c"}],
        trajectory={},
        score=score,
        episode_runtime_success=success,
        success=success,
        llm_turns_count=3,
        container_variation="variation_1",
        started_at="2026-04-08T00:00:00",
        completed_at="2026-04-08T00:01:00",
        duration_seconds=60.0,
    )


def _write_base_compose(root: Path) -> Path:
    base = root / "base"
    base.mkdir()
    (base / "docker-compose.yml").write_text(
        "services:\n  service-a:\n    image: foo\n  service-b:\n    image: bar\n"
    )
    return base


class TestEndpointQuarantineDecision(unittest.TestCase):
    """Only a genuine endpoint outage quarantines the endpoint.

    A request timeout (``inference_timeout``) is a benign, retryable failure —
    quarantining the (possibly sole) endpoint on a timeout breaches the capacity
    floor and aborts the whole run. This is the exact decision the worker loop
    makes at the ``episode.error_category == "endpoint_unavailable"`` check.
    """

    def test_only_endpoint_unavailable_triggers_quarantine(self) -> None:
        from src.execution.parallel import _episode_triggers_endpoint_quarantine

        self.assertTrue(_episode_triggers_endpoint_quarantine("endpoint_unavailable"))
        self.assertFalse(_episode_triggers_endpoint_quarantine("inference_timeout"))
        self.assertFalse(_episode_triggers_endpoint_quarantine("context_overflow"))
        self.assertFalse(_episode_triggers_endpoint_quarantine("agent_failure"))
        self.assertFalse(_episode_triggers_endpoint_quarantine("harness_error"))
        self.assertFalse(_episode_triggers_endpoint_quarantine("success"))
        self.assertFalse(_episode_triggers_endpoint_quarantine(None))


class TestSlotCircuitBreaker(unittest.TestCase):
    def test_trips_on_consecutive_failures(self):
        from src.parallel import SlotCircuitBreaker

        cb = SlotCircuitBreaker(max_consecutive_failures=3)
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.is_tripped())
        cb.record_failure()
        self.assertTrue(cb.is_tripped())

    def test_resets_on_success(self):
        from src.parallel import SlotCircuitBreaker

        cb = SlotCircuitBreaker(max_consecutive_failures=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        self.assertFalse(cb.is_tripped())

    def test_reset_only_after_successful_rebuild(self):
        from src.parallel import SlotCircuitBreaker

        cb = SlotCircuitBreaker(max_consecutive_failures=2)
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_tripped())
        # Without reset(), breaker remains tripped
        cb.record_failure()
        self.assertTrue(cb.is_tripped())
        # Reset clears tripped state
        cb.reset()
        self.assertFalse(cb.is_tripped())

    def test_verification_failures_tracked_separately(self):
        from src.parallel import SlotCircuitBreaker

        cb = SlotCircuitBreaker(
            max_consecutive_failures=10,
            max_consecutive_verification_failures=2,
        )
        cb.record_verification_failure()
        self.assertFalse(cb.is_verification_tripped())
        cb.record_verification_failure()
        self.assertTrue(cb.is_verification_tripped())
        # General breaker not tripped
        self.assertFalse(cb.is_tripped())

    def test_verification_disabled_when_zero(self):
        from src.parallel import SlotCircuitBreaker

        cb = SlotCircuitBreaker(
            max_consecutive_failures=3,
            max_consecutive_verification_failures=0,
        )
        cb.record_verification_failure()
        cb.record_verification_failure()
        cb.record_verification_failure()
        # Should never trip when threshold is 0 (disabled)
        self.assertFalse(cb.is_verification_tripped())

    def test_success_resets_verification_counter(self):
        from src.parallel import SlotCircuitBreaker

        cb = SlotCircuitBreaker(
            max_consecutive_failures=10,
            max_consecutive_verification_failures=3,
        )
        cb.record_verification_failure()
        cb.record_verification_failure()
        cb.record_success()
        cb.record_verification_failure()
        self.assertFalse(cb.is_verification_tripped())

    def test_benign_abort_preserves_existing_failure_history(self):
        from src.parallel import SlotCircuitBreaker

        cb = SlotCircuitBreaker(max_consecutive_failures=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_benign_abort()
        self.assertFalse(cb.is_tripped())
        cb.record_failure()
        self.assertTrue(cb.is_tripped())


class TestGlobalCircuitBreaker(unittest.TestCase):
    def test_not_tripped_single_slot_failure(self):
        from src.parallel import GlobalCircuitBreaker, SlotCircuitBreaker

        breakers = [SlotCircuitBreaker(max_consecutive_failures=1) for _ in range(4)]
        breakers[0].record_failure()  # Only 1 of 4 tripped
        gcb = GlobalCircuitBreaker(breakers, threshold=0.5)
        self.assertFalse(gcb.is_tripped())

    def test_trips_on_majority_failure(self):
        from src.parallel import GlobalCircuitBreaker, SlotCircuitBreaker

        breakers = [SlotCircuitBreaker(max_consecutive_failures=1) for _ in range(4)]
        breakers[0].record_failure()
        breakers[1].record_failure()
        breakers[2].record_failure()  # 3 of 4 tripped = 75% > 50%
        gcb = GlobalCircuitBreaker(breakers, threshold=0.5)
        self.assertTrue(gcb.is_tripped())

    def test_exactly_at_threshold_trips(self):
        from src.parallel import GlobalCircuitBreaker, SlotCircuitBreaker

        breakers = [SlotCircuitBreaker(max_consecutive_failures=1) for _ in range(4)]
        breakers[0].record_failure()
        breakers[1].record_failure()  # 2 of 4 = 50% = threshold
        gcb = GlobalCircuitBreaker(breakers, threshold=0.5)
        self.assertTrue(gcb.is_tripped())


class TestThreadSafeGenerationCollector(unittest.TestCase):
    def test_progress_snapshot_does_not_refresh_training_rows(self):
        from src.parallel import ThreadSafeGenerationCollector

        count_fn = MagicMock(return_value=10)
        collector = ThreadSafeGenerationCollector(
            GenerationData(generation_id=0),
            training_row_count_fn=count_fn,
        )

        self.assertEqual(collector.progress_snapshot(), (0, 0))
        count_fn.assert_not_called()

    def test_should_stop_does_not_refresh_training_rows(self):
        from src.parallel import ThreadSafeGenerationCollector

        count_fn = MagicMock(return_value=10)
        generation_data = GenerationData(generation_id=0)
        generation_data.set_training_row_count(3)
        collector = ThreadSafeGenerationCollector(
            generation_data,
            training_row_count_fn=count_fn,
        )

        self.assertTrue(collector.should_stop(target_rows=3))
        count_fn.assert_not_called()

    def test_training_row_count_does_not_refresh_training_rows(self):
        from src.parallel import ThreadSafeGenerationCollector

        count_fn = MagicMock(return_value=10)
        generation_data = GenerationData(generation_id=0)
        generation_data.set_training_row_count(3)
        collector = ThreadSafeGenerationCollector(
            generation_data,
            training_row_count_fn=count_fn,
        )

        self.assertEqual(collector.training_row_count(), 3)
        count_fn.assert_not_called()

    def test_add_episode_marks_row_count_stale_without_refreshing(self):
        from src.parallel import ThreadSafeGenerationCollector

        count_fn = MagicMock(return_value=10)
        collector = ThreadSafeGenerationCollector(
            GenerationData(generation_id=0),
            training_row_count_fn=count_fn,
        )

        collector.add_episode(_make_episode(episode_id="ep-0"))

        count_fn.assert_not_called()
        state = collector.row_count_state()
        self.assertEqual(state.training_row_count, 1)
        self.assertFalse(state.training_row_count_is_exact)
        self.assertEqual(state.training_row_count_last_refreshed_episode, 0)

    def test_explicit_refresh_counts_rows_once_from_snapshot(self):
        from src.parallel import ThreadSafeGenerationCollector

        snapshots = []

        def count_rows(data):
            snapshots.append(data)
            return 7

        generation_data = GenerationData(generation_id=0)
        collector = ThreadSafeGenerationCollector(
            generation_data,
            training_row_count_fn=count_rows,
        )
        collector.add_episode(_make_episode(episode_id="ep-0"))

        self.assertEqual(collector.refresh_training_row_count(), 7)

        self.assertEqual(len(snapshots), 1)
        self.assertIsNot(snapshots[0], generation_data)
        self.assertEqual(snapshots[0].total_episodes_run, 1)
        state = collector.row_count_state()
        self.assertEqual(state.training_row_count, 7)
        self.assertTrue(state.training_row_count_is_exact)
        self.assertEqual(state.training_row_count_last_refreshed_episode, 1)

    def test_concurrent_add_all_present(self):
        from src.parallel import ThreadSafeGenerationCollector

        collector = ThreadSafeGenerationCollector(GenerationData(generation_id=0))
        barrier = threading.Barrier(8)

        def add_episode(idx):
            barrier.wait()
            ep = _make_episode(
                episode_id=f"ep-{idx}",
                episode_index=idx,
                success=True,
                prompt_responses=[{"prompt": "p", "completion": "c"}],
            )
            collector.add_episode(ep)

        threads = [threading.Thread(target=add_episode, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(collector.training_row_count(), 8)

    def test_should_stop_at_target(self):
        from src.parallel import ThreadSafeGenerationCollector

        collector = ThreadSafeGenerationCollector(GenerationData(generation_id=0))
        self.assertFalse(collector.should_stop(target_rows=2))

        collector.add_episode(
            _make_episode(episode_id="ep-0", prompt_responses=[{"p": "a", "c": "b"}])
        )
        self.assertFalse(collector.should_stop(target_rows=2))

        collector.add_episode(
            _make_episode(episode_id="ep-1", prompt_responses=[{"p": "a", "c": "b"}])
        )
        self.assertTrue(collector.should_stop(target_rows=2))

    def test_failed_episodes_dont_count_toward_rows(self):
        from src.parallel import ThreadSafeGenerationCollector

        collector = ThreadSafeGenerationCollector(GenerationData(generation_id=0))
        collector.add_episode(
            _make_episode(
                episode_id="ep-fail",
                success=False,
                prompt_responses=[{"p": "a", "c": "b"}],
            )
        )
        self.assertEqual(collector.training_row_count(), 0)
        self.assertFalse(collector.should_stop(target_rows=1))

    def test_snapshot_returns_consistent_copy(self):
        from src.parallel import ThreadSafeGenerationCollector

        collector = ThreadSafeGenerationCollector(GenerationData(generation_id=0))
        collector.add_episode(_make_episode(episode_id="ep-0", episode_index=0))

        snapshot = collector.snapshot()
        collector.add_episode(_make_episode(episode_id="ep-1", episode_index=1))

        self.assertEqual(snapshot.total_episodes_run, 1)
        self.assertEqual(len(snapshot.all_episodes), 1)
        self.assertEqual(collector.training_row_count(), 2)


class TestWorkerSlot(unittest.TestCase):
    def test_unique_slot_ids(self):
        from src.parallel import WorkerSlot

        slots = []
        for i in range(4):
            cm = MagicMock()
            cm.container_ids = [f"container_{i}"]
            slots.append(
                WorkerSlot(
                    slot_id=i,
                    container_manager=cm,
                    docker_client=MagicMock(),
                    circuit_breaker=MagicMock(),
                    cache_dir=Path(f"/tmp/slot_{i}"),
                )
            )
        slot_ids = [s.slot_id for s in slots]
        self.assertEqual(slot_ids, [0, 1, 2, 3])
        cache_dirs = [s.cache_dir for s in slots]
        self.assertEqual(len(set(cache_dirs)), 4)


class TestGenerateSlotCompose(unittest.TestCase):
    def test_strips_explicit_container_name(self):
        import tempfile
        from src.parallel import _generate_slot_compose

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            (base / "docker-compose.yml").write_text(
                "services:\n"
                "  service-a:\n"
                "    container_name: crypto-exploit-compose_service-a_1\n"
                "    image: foo\n"
            )
            slot_dir = Path(tmp) / "slot_0"

            _generate_slot_compose(base, slot_dir, "gen_run1_slot_0", 0)

            generated = (slot_dir / "docker-compose.yml").read_text()
            self.assertNotIn("container_name", generated)
            self.assertIn("service-a:", generated)
            self.assertIn("image: foo", generated)

    def test_copies_nested_build_context_files(self):
        from src.parallel import _generate_slot_compose

        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            (base / "proxy-src").mkdir(parents=True)
            (base / "staging" / "rpc_caches").mkdir(parents=True)
            (base / "docker-compose.yml").write_text(
                "services:\n"
                "  service-a:\n"
                "    build:\n"
                "      context: .\n"
                "      dockerfile: Dockerfile\n"
            )
            (base / "Dockerfile").write_text("FROM alpine\n")
            (base / "proxy-src" / "server.js").write_text("console.log('ok')\n")
            (base / "staging" / "rpc_caches" / "cache.json").write_text("{}\n")
            slot_dir = Path(tmp) / "slot_0"

            _generate_slot_compose(base, slot_dir, "gen_run1_slot_0", 0)

            self.assertTrue((slot_dir / "proxy-src" / "server.js").exists())
            self.assertTrue(
                (slot_dir / "staging" / "rpc_caches" / "cache.json").exists()
            )

    def test_absolutizes_parent_build_context(self):
        """Build contexts that walk above the compose dir (e.g. ``../..``)
        must be rewritten to absolute paths anchored at the base compose
        dir; otherwise ``docker compose up --build`` from the slot dir
        resolves them against the wrong anchor and fails with
        ``lstat ...: no such file or directory``. Regression test for nested
        build contexts in parallel mode.
        """
        import yaml

        from src.parallel import _generate_slot_compose

        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            base = repo_root / "docker" / "nested-context-compose"
            base.mkdir(parents=True)
            (base / "docker-compose.yml").write_text(
                "services:\n"
                "  agent:\n"
                "    build:\n"
                "      context: ../..\n"
                "      dockerfile: docker/nested-context-compose/Dockerfile.agent\n"
                "  analysis:\n"
                "    build:\n"
                "      context: .\n"
                "      dockerfile: Dockerfile.analysis\n"
            )
            slot_dir = Path(tmp) / "generations" / "run1" / "compose" / "slot_0"

            _generate_slot_compose(base, slot_dir, "gen_run1_slot_0", 0)

            generated = yaml.safe_load((slot_dir / "docker-compose.yml").read_text())
            self.assertEqual(
                generated["services"]["agent"]["build"]["context"],
                str(repo_root.resolve()),
            )
            self.assertEqual(
                generated["services"]["analysis"]["build"]["context"],
                str(base.resolve()),
            )

    def test_absolutizes_relative_volume_sources(self):
        """Volume sources with relative paths — both bare and inside
        ``${VAR:-default}`` substitutions — must be rewritten so bind
        mounts resolve correctly from the slot compose dir."""
        import yaml

        from src.parallel import _generate_slot_compose

        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            base = repo_root / "docker" / "task-compose"
            base.mkdir(parents=True)
            (base / "docker-compose.yml").write_text(
                "services:\n"
                "  g2v:\n"
                "    image: foo\n"
                "    volumes:\n"
                "      - ${POOL_DIR:-../../tasks/pools}:/workspace/pool:ro\n"
                "      - ./local:/workspace/local\n"
                "      - named-volume:/workspace/named\n"
            )
            slot_dir = Path(tmp) / "generations" / "run1" / "compose" / "slot_0"

            _generate_slot_compose(base, slot_dir, "gen_run1_slot_0", 0)

            generated = yaml.safe_load((slot_dir / "docker-compose.yml").read_text())
            volumes = generated["services"]["g2v"]["volumes"]
            expected_pool = (repo_root / "tasks" / "pools").resolve()
            expected_local = (base / "local").resolve()
            self.assertEqual(
                volumes[0], f"${{POOL_DIR:-{expected_pool}}}:/workspace/pool:ro"
            )
            self.assertEqual(volumes[1], f"{expected_local}:/workspace/local")
            self.assertEqual(volumes[2], "named-volume:/workspace/named")

    def test_slot_local_bind_sources_are_isolated_per_worker(self):
        import yaml

        from src.parallel import _generate_slot_compose

        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            (base / "docker-compose.yml").write_text(
                "services:\n"
                "  compute:\n"
                "    image: python\n"
                "    x-nsl-slot-local-bind-sources: ['./public-evidence']\n"
                "    volumes:\n"
                "      - type: bind\n"
                "        source: ./public-evidence\n"
                "        target: /data\n"
            )
            first = Path(tmp) / "slot_0"
            second = Path(tmp) / "slot_1"

            _generate_slot_compose(base, first, "slot_0", 0)
            _generate_slot_compose(base, second, "slot_1", 1)

            first_service = yaml.safe_load(
                (first / "docker-compose.yml").read_text()
            )["services"]["compute"]
            second_service = yaml.safe_load(
                (second / "docker-compose.yml").read_text()
            )["services"]["compute"]
            first_source = first_service["volumes"][0]["source"]
            second_source = second_service["volumes"][0]["source"]
            self.assertNotEqual(first_source, second_source)
            self.assertEqual(first_source, str((first / "public-evidence").resolve()))
            self.assertNotIn("x-nsl-slot-local-bind-sources", first_service)


class TestCreateWorkerSlots(unittest.TestCase):
    @patch("src.parallel.ContainerManager.refresh_container_ids")
    @patch("src.container.subprocess.run")
    @patch("src.parallel._launch_slot_compose")
    @patch("src.parallel.docker")
    def test_creates_n_slots(
        self,
        mock_docker,
        mock_launch,
        compose_subprocess_run,
        refresh_container_ids,
    ):
        from src.parallel import create_worker_slots

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_compose_dir = _write_base_compose(root)
            generation_dir = root / "gen"
            cache_dir = root / "cache"

            mock_client = MagicMock()
            mock_client.containers.list.return_value = []
            mock_docker.from_env.return_value = mock_client
            compose_subprocess_run.return_value = SimpleNamespace(
                stdout="service-a\nservice-b\n",
                stderr="",
                returncode=0,
            )

            slots = create_worker_slots(
                n_slots=3,
                base_compose_dir=base_compose_dir,
                generation_dir=generation_dir,
                run_id="run1",
                code_host_cache_path=cache_dir,
            )
            self.assertEqual(len(slots), 3)
            for i, slot in enumerate(slots):
                self.assertEqual(slot.slot_id, i)
                self.assertEqual(
                    slot.container_manager.expected_services,
                    ["service-a", "service-b"],
                )

    @patch("src.parallel.ContainerManager.refresh_container_ids")
    @patch("src.container.subprocess.run")
    @patch("src.parallel._launch_slot_compose")
    @patch("src.parallel.docker")
    def test_separate_compose_projects(
        self,
        mock_docker,
        mock_launch,
        compose_subprocess_run,
        refresh_container_ids,
    ):
        from src.parallel import create_worker_slots

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_compose_dir = _write_base_compose(root)
            generation_dir = root / "gen"
            cache_dir = root / "cache"

            mock_client = MagicMock()
            mock_client.containers.list.return_value = []
            mock_docker.from_env.return_value = mock_client
            compose_subprocess_run.return_value = SimpleNamespace(
                stdout="service-a\nservice-b\n",
                stderr="",
                returncode=0,
            )

            slots = create_worker_slots(
                n_slots=2,
                base_compose_dir=base_compose_dir,
                generation_dir=generation_dir,
                run_id="run1",
                code_host_cache_path=cache_dir,
            )
            self.assertEqual(len(slots), 2)
            self.assertEqual(mock_launch.call_count, 2)
            self.assertEqual(
                slots[0].container_manager.project_name_pattern,
                "gen_run1_slot_0",
            )
            self.assertEqual(
                slots[1].container_manager.project_name_pattern,
                "gen_run1_slot_1",
            )

    @patch("src.parallel.ContainerManager.refresh_container_ids")
    @patch("src.container.subprocess.run")
    @patch("src.parallel._launch_slot_compose")
    @patch("src.parallel.docker")
    def test_per_slot_cache_dirs(
        self,
        mock_docker,
        mock_launch,
        compose_subprocess_run,
        refresh_container_ids,
    ):
        from src.parallel import create_worker_slots

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_compose_dir = _write_base_compose(root)
            generation_dir = root / "gen"
            cache_dir = root / "cache"

            mock_client = MagicMock()
            mock_client.containers.list.return_value = []
            mock_docker.from_env.return_value = mock_client
            compose_subprocess_run.return_value = SimpleNamespace(
                stdout="service-a\nservice-b\n",
                stderr="",
                returncode=0,
            )

            slots = create_worker_slots(
                n_slots=3,
                base_compose_dir=base_compose_dir,
                generation_dir=generation_dir,
                run_id="run1",
                code_host_cache_path=cache_dir,
            )
            cache_dirs = {slot.cache_dir for slot in slots}
            self.assertEqual(len(cache_dirs), 3)
            for slot in slots:
                self.assertIn(f"slot_{slot.slot_id}", str(slot.cache_dir))


class TestParallelConfigParsing(unittest.TestCase):
    def test_parallel_episodes_config_parses(self):
        from src.typing.config import AppConfig

        config = AppConfig(
            task={"class": "tests.fixtures.stub_task.ValidStubTask"},
            model_name="test",
            code_host_cache_path="/tmp",
            container_ids=["c1"],
            train_data_save_folder="/tmp",
            generation={"parallel_episodes": 4},
        )
        self.assertEqual(config.generation.parallel_episodes, 4)

    def test_parallel_episodes_defaults_to_1(self):
        from src.typing.config import AppConfig

        config = AppConfig(
            task={"class": "tests.fixtures.stub_task.ValidStubTask"},
            model_name="test",
            code_host_cache_path="/tmp",
            container_ids=["c1"],
            train_data_save_folder="/tmp",
            generation={},
        )
        self.assertEqual(config.generation.parallel_episodes, 1)


class TestParallelDispatch(unittest.TestCase):
    def _make_config(self, base_dir, parallel_episodes=1):
        from src.typing.config import AppConfig

        return AppConfig(
            task={"class": "tests.fixtures.stub_task.ValidStubTask"},
            model_name="claude",
            code_host_cache_path=str(base_dir / "code-host-cache"),
            container_ids=["container-a"],
            main_container_idx=0,
            dynamic_container=False,
            docker_compose_dir=str(base_dir / "compose"),
            train_data_save_folder=str(base_dir / "train-data"),
            harness={
                "name": "orchestrator_modes",
                "orchestrator_modes": {
                    "max_harness_iterations": 2,
                    "scratchpad_max_chars": 1000,
                    "orchestrator_prompt": "p",
                },
            },
            generation={
                "target_training_rows": 2,
                "max_episodes": 10,
                "container_restart_interval": 50,
                "container_rebuild_interval": 50,
                "show_progress": False,
                "checkpoint_every_episode": False,
                "resume_from_checkpoint": False,
                "generation_output_dir": str(base_dir / "generations"),
                "parallel_episodes": parallel_episodes,
            },
            observability={"enabled": False},
        )

    @patch("src.execution.generation.run_single_episode")
    @patch("src.execution.generation.ContainerManager")
    def test_run_generation_sequential_when_1(self, MockCM, mock_episode):
        """parallel_episodes=1 should use the existing sequential path."""
        from tempfile import TemporaryDirectory

        from src.execution import run_generation

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self._make_config(base_dir, parallel_episodes=1)

            manager = MockCM.return_value
            manager.populate_with_task.return_value = (
                PopulationOutcome(
                    results=[
                        PopulationResult(
                            container_id="container-a",
                            variation_name="v1",
                            description="t",
                            success=True,
                        )
                    ],
                ),
                True,
            )
            manager.get_containers.return_value = [MagicMock(id="container-a")]

            ep = _make_episode(
                episode_id="ep-0",
                prompt_responses=[
                    {"prompt": "p1", "completion": "c1"},
                    {"prompt": "p2", "completion": "c2"},
                ],
            )
            mock_episode.return_value = ep

            mock_task = MagicMock()
            mock_task.list_variations.return_value = [MagicMock()]
            result = run_generation(
                genner=MagicMock(),
                docker_client=MagicMock(),
                config=config,
                generation_id=0,
                run_id="test-run",
                task=mock_task,
            )
            # Sequential path should have been called (not parallel)
            self.assertGreater(mock_episode.call_count, 0)
            self.assertEqual(result.total_episodes_run, 1)

    @patch("src.execution.run_generation_parallel")
    @patch("src.execution.generation.ContainerManager")
    def test_run_generation_dispatches_parallel(self, MockCM, mock_parallel):
        """parallel_episodes > 1 should dispatch to run_generation_parallel."""
        from tempfile import TemporaryDirectory

        from src.execution import run_generation

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self._make_config(base_dir, parallel_episodes=2)

            mock_parallel.return_value = GenerationData(generation_id=0)

            run_generation(
                genner=MagicMock(),
                docker_client=MagicMock(),
                config=config,
                generation_id=0,
                run_id="test-run",
                task=MagicMock(),
            )
            mock_parallel.assert_called_once()


class TestParallelSampling(unittest.TestCase):
    """Verify sampling lifecycle is at generation level, not per-episode."""

    def test_run_single_episode_skips_sampling_when_parallel(self):
        """run_single_episode should NOT call start/stop_utilization_sampling
        when parallel_episodes > 1."""
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            from src.typing.config import AppConfig

            config = AppConfig(
                task={"class": "tests.fixtures.stub_task.ValidStubTask"},
                model_name="claude",
                code_host_cache_path=str(base_dir / "code-host-cache"),
                container_ids=["container-a"],
                main_container_idx=0,
                dynamic_container=False,
                docker_compose_dir=str(base_dir / "compose"),
                train_data_save_folder=str(base_dir / "train-data"),
                harness={
                    "name": "orchestrator_modes",
                    "orchestrator_modes": {
                        "max_harness_iterations": 2,
                        "scratchpad_max_chars": 1000,
                        "orchestrator_prompt": "p",
                    },
                },
                generation={
                    "target_training_rows": 2,
                    "max_episodes": 10,
                    "show_progress": False,
                    "generation_output_dir": str(base_dir / "generations"),
                    "parallel_episodes": 2,
                },
                observability={"enabled": True, "record_resources": True},
            )

            metrics_collector = MagicMock()
            metrics_collector.vllm_metrics_url = None
            metrics_collector.inference_metrics = []
            metrics_collector._lock = threading.Lock()
            metrics_collector.get_metrics_for_episode.return_value = []

            from src.observability.types import UtilizationSummary

            metrics_collector.stop_utilization_sampling.return_value = (
                UtilizationSummary()
            )

            manager = MagicMock()
            population_outcome = PopulationOutcome(
                results=[
                    PopulationResult(
                        container_id="container-a",
                        variation_name="v1",
                        description="t",
                        success=True,
                    )
                ],
            )
            manager.get_containers.return_value = [MagicMock(id="container-a")]

            from src.execution.episode_runner import run_single_episode

            with patch("src.execution.episode_runner.run_episode") as mock_ep:
                mock_ep.return_value = EpisodeOutcome(
                    episode_id="ep-0",
                    score=100.0,
                    success=True,
                    partial=False,
                    llm_turns_count=2,
                    train_rows=[{"prompt": "p", "raw_response": "c"}],
                    prompt_responses=[{"prompt": "p", "completion": "c"}],
                    trajectory={},
                    error_message=None,
                    error_category=None,
                    reward_breakdown={},
                    harness_error=False,
                )

                mock_task = MagicMock()
                mock_task.list_variations.return_value = [MagicMock()]
                run_single_episode(
                    genner=MagicMock(),
                    docker_client=MagicMock(),
                    container_manager=manager,
                    config=config,
                    generation_id=0,
                    episode_index=0,
                    variation_index=0,
                    run_id="test-run",
                    metrics_collector=metrics_collector,
                    population_outcome=population_outcome,
                    verified=True,
                    parallel_episodes=2,
                    task=mock_task,
                )

                # Should NOT have called start_utilization_sampling
                metrics_collector.start_utilization_sampling.assert_not_called()


class TestMetricsCollectorAccessor(unittest.TestCase):
    def test_get_metrics_for_episode_filters_correctly(self):
        from src.observability.collector import MetricsCollector
        from src.observability.types import InferenceMetric
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(run_id="run-1", output_dir=Path(temp_dir))

            m1 = InferenceMetric(
                episode_id="ep-0",
                inference_id="inf-1",
                run_id="run-1",
                backend="vllm",
                phase="orchestrator",
                latency_ms=100.0,
                success=True,
            )
            m2 = InferenceMetric(
                episode_id="ep-1",
                inference_id="inf-2",
                run_id="run-1",
                backend="vllm",
                phase="orchestrator",
                latency_ms=200.0,
                success=True,
            )
            collector.record_inference(m1)
            collector.record_inference(m2)

            result = collector.get_metrics_for_episode("ep-0")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].inference_id, "inf-1")


class TestTeardownWorkerSlots(unittest.TestCase):
    @patch("subprocess.run")
    def test_teardown_uses_volumes_and_remove_orphans(self, mock_run):
        from src.parallel import WorkerSlot, teardown_worker_slots

        cm = MagicMock()
        cm.docker_compose_dir = "/some/compose/dir"
        cm.project_name_pattern = "gen_run1_slot_0"
        slot = WorkerSlot(
            slot_id=0,
            container_manager=cm,
            docker_client=MagicMock(),
            circuit_breaker=MagicMock(),
            cache_dir=Path("/tmp/slot_0"),
        )

        teardown_worker_slots([slot])

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("down", cmd)
        self.assertIn("--volumes", cmd)
        self.assertIn("--remove-orphans", cmd)


class TestCreateWorkerSlotsPartialFailure(unittest.TestCase):
    """New contract: on mid-loop failure, keep successful slots, clean up the
    failed slot, warn, and return partial list. Only re-raise if zero slots
    succeeded.
    """

    @patch("src.parallel._cleanup_partial_slot")
    @patch("src.parallel.teardown_worker_slots")
    @patch("src.parallel.ContainerManager.refresh_container_ids")
    @patch("src.container.subprocess.run")
    @patch("src.parallel._launch_slot_compose")
    @patch("src.parallel.estimate_slot_capacity")
    @patch("src.parallel.docker")
    def test_returns_partial_slots_on_midloop_failure(
        self,
        mock_docker,
        mock_estimate,
        mock_launch,
        compose_subprocess_run,
        refresh_container_ids,
        mock_teardown,
        mock_cleanup,
    ):
        from src.parallel import create_worker_slots

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_compose_dir = _write_base_compose(root)
            generation_dir = root / "gen"
            cache_dir = root / "cache"

            mock_client = MagicMock()
            mock_client.containers.list.return_value = []
            mock_docker.from_env.return_value = mock_client
            compose_subprocess_run.return_value = SimpleNamespace(
                stdout="service-a\nservice-b\n",
                stderr="",
                returncode=0,
            )
            mock_estimate.return_value = 4  # preflight allows all 4

            mock_launch.side_effect = [None, None, RuntimeError("network exhausted")]

            slots = create_worker_slots(
                n_slots=4,
                base_compose_dir=base_compose_dir,
                generation_dir=generation_dir,
                run_id="run1",
                code_host_cache_path=cache_dir,
            )

            self.assertEqual(len(slots), 2)
            mock_teardown.assert_not_called()
            mock_cleanup.assert_called_once()
            # Cleanup was for the 3rd slot (index 2)
            cleanup_args = mock_cleanup.call_args[0]
            self.assertEqual(cleanup_args[0], "gen_run1_slot_2")

    @patch("src.parallel._cleanup_partial_slot")
    @patch("src.parallel.teardown_worker_slots")
    @patch("src.parallel._launch_slot_compose")
    @patch("src.parallel.estimate_slot_capacity")
    @patch("src.parallel.docker")
    def test_reraises_when_zero_slots_succeed(
        self,
        mock_docker,
        mock_estimate,
        mock_launch,
        mock_teardown,
        mock_cleanup,
    ):
        from src.parallel import create_worker_slots

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_compose_dir = _write_base_compose(root)
            generation_dir = root / "gen"
            cache_dir = root / "cache"

            mock_docker.from_env.return_value = MagicMock()
            mock_estimate.return_value = 4
            mock_launch.side_effect = RuntimeError("daemon down")

            with self.assertRaises(RuntimeError):
                create_worker_slots(
                    n_slots=4,
                    base_compose_dir=base_compose_dir,
                    generation_dir=generation_dir,
                    run_id="run1",
                    code_host_cache_path=cache_dir,
                )

            mock_teardown.assert_not_called()
            mock_cleanup.assert_called_once()

    @patch("src.parallel._cleanup_partial_slot")
    @patch("src.parallel.ContainerManager.refresh_container_ids")
    @patch("src.container.subprocess.run")
    @patch("src.parallel._launch_slot_compose")
    @patch("src.parallel.estimate_slot_capacity")
    @patch("src.parallel.docker")
    def test_preflight_caps_n_slots(
        self,
        mock_docker,
        mock_estimate,
        mock_launch,
        compose_subprocess_run,
        refresh_container_ids,
        mock_cleanup,
    ):
        from src.parallel import create_worker_slots

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_compose_dir = _write_base_compose(root)
            generation_dir = root / "gen"
            cache_dir = root / "cache"

            mock_docker.from_env.return_value = MagicMock()
            compose_subprocess_run.return_value = SimpleNamespace(
                stdout="service-a\n",
                stderr="",
                returncode=0,
            )
            mock_launch.return_value = None
            mock_estimate.return_value = 3  # cap to 3 even though 8 requested

            slots = create_worker_slots(
                n_slots=8,
                base_compose_dir=base_compose_dir,
                generation_dir=generation_dir,
                run_id="run1",
                code_host_cache_path=cache_dir,
            )

            self.assertEqual(len(slots), 3)
            self.assertEqual(mock_launch.call_count, 3)
            mock_cleanup.assert_not_called()

    @patch("src.parallel._cleanup_partial_slot")
    @patch("src.parallel.ContainerManager.refresh_container_ids")
    @patch("src.container.subprocess.run")
    @patch("src.parallel._launch_slot_compose")
    @patch("src.parallel.estimate_slot_capacity")
    @patch("src.parallel.docker")
    def test_preflight_zero_still_attempts_one(
        self,
        mock_docker,
        mock_estimate,
        mock_launch,
        compose_subprocess_run,
        refresh_container_ids,
        mock_cleanup,
    ):
        from src.parallel import create_worker_slots

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_compose_dir = _write_base_compose(root)
            generation_dir = root / "gen"
            cache_dir = root / "cache"

            mock_docker.from_env.return_value = MagicMock()
            compose_subprocess_run.return_value = SimpleNamespace(
                stdout="service-a\n",
                stderr="",
                returncode=0,
            )
            mock_launch.return_value = None
            mock_estimate.return_value = 0  # heuristic says no capacity

            slots = create_worker_slots(
                n_slots=4,
                base_compose_dir=base_compose_dir,
                generation_dir=generation_dir,
                run_id="run1",
                code_host_cache_path=cache_dir,
            )

            self.assertEqual(len(slots), 1)
            self.assertEqual(mock_launch.call_count, 1)


class TestCleanupPartialSlot(unittest.TestCase):
    def test_cleanup_never_raises(self):
        from src.parallel import _cleanup_partial_slot

        with TemporaryDirectory() as temp_dir:
            slot_dir = Path(temp_dir) / "slot_0"
            slot_dir.mkdir()
            (slot_dir / "docker-compose.yml").write_text("services: {}\n")

            with patch("src.parallel.subprocess.run") as mock_run:
                mock_run.side_effect = RuntimeError("docker gone")
                # Must not raise
                _cleanup_partial_slot("some_project", slot_dir)

            # Directory was still removed despite subprocess failure
            self.assertFalse(slot_dir.exists())

    def test_cleanup_handles_missing_directory(self):
        from src.parallel import _cleanup_partial_slot

        with patch("src.parallel.subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                stdout="",
                stderr="",
                returncode=0,
            )
            # Missing directory must not crash
            _cleanup_partial_slot("some_project", Path("/nonexistent/path"))


class TestEstimateSlotCapacity(unittest.TestCase):
    def test_subtracts_existing_networks(self):
        from src.parallel import estimate_slot_capacity

        # 31 capacity, 9 existing user networks, 3 nets/slot
        # affordable = (31 - 9) // 3 = 7; request 10 -> 7
        result = estimate_slot_capacity(
            requested_n_slots=10,
            existing_user_networks=9,
            pool_capacity=31,
        )
        self.assertEqual(result, 7)

    def test_capacity_bounds(self):
        from src.parallel import estimate_slot_capacity

        cases = [
            ("pool_exhausted", 4, 30, 31, 0),
            ("cap_at_requested", 4, 0, 31, 4),
        ]
        for name, requested, existing, pool_capacity, expected in cases:
            with self.subTest(name=name):
                result = estimate_slot_capacity(
                    requested_n_slots=requested,
                    existing_user_networks=existing,
                    pool_capacity=pool_capacity,
                )
                self.assertEqual(result, expected)

    def test_queries_docker_when_existing_is_none(self):
        from src.parallel import estimate_slot_capacity

        with patch("src.parallel.docker") as mock_docker:
            nets = [
                MagicMock(name="bridge"),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]
            nets[0].name = "bridge"  # should be excluded
            nets[1].name = "other_1"
            nets[2].name = "other_2"
            nets[3].name = "other_3"
            mock_docker.from_env.return_value.networks.list.return_value = nets

            # 3 user networks; affordable = (31-3)//3 = 9; request 4 -> 4
            result = estimate_slot_capacity(
                requested_n_slots=4,
                pool_capacity=31,
            )
            self.assertEqual(result, 4)

    def test_uses_daemon_pools_when_capacity_unspecified(self):
        """pool_capacity=None -> derive from docker info DefaultAddressPools."""
        from src.parallel import estimate_slot_capacity

        with patch("src.parallel.docker") as mock_docker:
            client = mock_docker.from_env.return_value
            # 10.0.0.0/8 carved into /24 = 2^(24-8) = 65536 subnets
            client.info.return_value = {
                "DefaultAddressPools": [{"Base": "10.0.0.0/8", "Size": 24}]
            }
            client.networks.list.return_value = []  # zero existing

            # affordable = 65536 // 3 = 21845; request 16 -> 16
            result = estimate_slot_capacity(requested_n_slots=16)
            self.assertEqual(result, 16)


class TestPoolCapacityFromDocker(unittest.TestCase):
    def test_sums_pools(self):
        from src.parallel import _pool_capacity_from_docker

        with patch("src.parallel.docker") as mock_docker:
            mock_docker.from_env.return_value.info.return_value = {
                "DefaultAddressPools": [
                    {"Base": "10.0.0.0/8", "Size": 24},  # 65536
                    {"Base": "172.16.0.0/12", "Size": 20},  # 256
                ]
            }
            self.assertEqual(_pool_capacity_from_docker(), 65536 + 256)

    def test_falls_back_when_pools_absent(self):
        """Unconfigured daemons don't enumerate their built-ins via info."""
        from src.parallel import (
            _DEFAULT_POOL_CAPACITY_NETWORKS,
            _pool_capacity_from_docker,
        )

        with patch("src.parallel.docker") as mock_docker:
            mock_docker.from_env.return_value.info.return_value = {}
            self.assertEqual(
                _pool_capacity_from_docker(), _DEFAULT_POOL_CAPACITY_NETWORKS
            )

    def test_falls_back_on_error(self):
        from src.parallel import (
            _DEFAULT_POOL_CAPACITY_NETWORKS,
            _pool_capacity_from_docker,
        )

        with patch("src.parallel.docker") as mock_docker:
            mock_docker.from_env.side_effect = RuntimeError("no docker")
            self.assertEqual(
                _pool_capacity_from_docker(), _DEFAULT_POOL_CAPACITY_NETWORKS
            )


if __name__ == "__main__":
    unittest.main()
