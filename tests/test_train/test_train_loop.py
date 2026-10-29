import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import toml

from src.typing.trajectory import EpisodeTrajectory, GenerationData


class TestTrainLoop(unittest.TestCase):
    def make_config(
        self,
        base_dir: Path,
        **orchestration_overrides: object,
    ):
        from src.typing.config import AppConfig

        orchestration_config = {
            "num_generations": 2,
            "training_window_size": 3,
        }
        orchestration_config.update(orchestration_overrides)

        return AppConfig(
            task={"class": "tests.fixtures.stub_task.ValidStubTask"},
            model_name="vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            code_host_cache_path=str(base_dir / "code-host-cache"),
            container_ids=["container-a", "container-b"],
            main_container_idx=0,
            dynamic_container=False,
            docker_compose_dir=str(base_dir / "compose"),
            train_data_save_folder=str(base_dir / "train-data"),
            vllm={
                "served_model_name": "nsl-test-loop",
            },
            generation={
                "target_training_rows": 2,
                "max_episodes": 2,
                "container_restart_interval": 10,
                "container_rebuild_interval": 10,
                "show_progress": False,
                "checkpoint_every_episode": False,
                "resume_from_checkpoint": False,
                "generation_output_dir": str(base_dir / "generations"),
            },
            training={
                "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "max_steps": 5,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 2e-4,
                "warmup_steps": 1,
                "max_seq_length": 1024,
                "adapter_output_dir": str(base_dir / "adapters"),
                "gpu_wait_timeout_seconds": 0,
                "gpu_wait_min_free_memory_fraction": 0.9,
            },
            observability={"detect_hardware": False},
            orchestration=orchestration_config,
        )

    def _write_training_rows(self, path: Path, generation_id: int) -> None:
        generation_dir = path.parent if path.name == "sft_training_rows.jsonl" else path
        export_dir = generation_dir / "exports" / "sft" / "test-export"
        export_dir.mkdir(parents=True, exist_ok=True)
        rows_path = export_dir / "sft_training_rows.jsonl"
        rows = [
            {
                "prompt": f"prompt-{generation_id}",
                "raw_response": f"response-{generation_id}",
                "timestamp": "2026-04-07T00:00:00",
                "interaction_type": "orchestrator",
                "success": True,
                "error_message": None,
                "episode_id": f"ep-gen-{generation_id}",
                "generation_id": generation_id,
                "episode_score": 64.0,
            }
        ]
        with open(rows_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        latest_path = generation_dir / "exports" / "sft" / "latest.json"
        latest_path.write_text(
            json.dumps(
                {
                    "export_id": "test-export",
                    "sft_training_rows_path": "exports/sft/test-export/sft_training_rows.jsonl",
                    "sft_training_rows_sha256": hashlib.sha256(
                        rows_path.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

    def _make_generation_data(
        self,
        generation_id: int,
        *,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        duration_seconds: float = 5.0,
    ) -> GenerationData:
        generation_data = GenerationData(
            generation_id=generation_id,
            started_at="2026-04-07T00:00:00",
            completed_at="2026-04-07T00:10:00",
        )
        generation_data.add_episode(
            EpisodeTrajectory(
                episode_id=f"ep-gen-{generation_id}",
                generation_id=generation_id,
                episode_index=0,
                prompt_responses=[
                    {
                        "prompt": f"prompt-{generation_id}",
                        "raw_response": f"response-{generation_id}",
                        "timestamp": "2026-04-07T00:00:00",
                        "interaction_type": "orchestrator",
                        "success": True,
                        "error_message": None,
                    }
                ],
                trajectory={},
                score=64.0,
                episode_runtime_success=True,
                success=True,
                llm_turns_count=1,
                container_variation="variation-a",
                started_at="2026-04-07T00:00:00",
                completed_at="2026-04-07T00:00:05",
                duration_seconds=duration_seconds,
                episode_execution_seconds=duration_seconds - 1.0,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )
        )
        return generation_data

    def _write_training_artifact(
        self,
        path: Path,
        export_format: str = "lora",
    ) -> Path:
        if export_format == "gguf":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"gguf")
            metadata_path = path.with_suffix(".training_info.json")
        else:
            path.mkdir(parents=True, exist_ok=True)
            metadata_path = path / "training_info.json"
            if export_format == "lora":
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_bytes(b"weights")
            else:
                (path / "model.safetensors").write_bytes(b"weights")
        metadata_path.write_text(
            json.dumps({"export_format": export_format}),
            encoding="utf-8",
        )
        return path

    def test_orchestration_config_in_app_config(self) -> None:
        from src.helper import unflatten_toml_dict
        from src.typing.config import AppConfig

        config_dict = toml.loads(
            """
model_name = "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
code_host_cache_path = "./code"
container_ids = ["container-a"]
train_data_save_folder = "./train-data"

[task]
class = "tests.fixtures.stub_task.ValidStubTask"

[generation]
generation_output_dir = "./data/generations"

[training]
base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter_output_dir = "./models/adapters"
max_steps = -1
num_train_epochs = 2
gradient_accumulation_steps = 16
learning_rate = 1e-4
warmup_ratio = 0.03
lr_scheduler_type = "linear"
weight_decay = 0.001
lora_rank = 32
lora_alpha = 32
seed = 3407
        rehearsal_dataset = "example/rehearsal-corpus"
rehearsal_rows_per_epoch = 500
inner_loss = "dft"
gpu_wait_timeout_seconds = 90
gpu_wait_min_free_memory_fraction = 0.85

[orchestration]
num_generations = 2
training_window_size = 3
"""
        )

        config = AppConfig(**unflatten_toml_dict(config_dict))

        self.assertIsNotNone(config.orchestration)
        self.assertEqual(config.orchestration.num_generations, 2)
        self.assertEqual(config.orchestration.training_window_size, 3)
        self.assertEqual(config.training.gpu_wait_timeout_seconds, 90)
        self.assertEqual(config.training.gpu_wait_min_free_memory_fraction, 0.85)
        self.assertEqual(config.training.max_steps, -1)
        self.assertEqual(config.training.num_train_epochs, 2)
        self.assertEqual(config.training.gradient_accumulation_steps, 16)
        self.assertEqual(config.training.learning_rate, 1e-4)
        self.assertEqual(config.training.warmup_ratio, 0.03)
        self.assertEqual(config.training.lr_scheduler_type, "linear")
        self.assertEqual(config.training.weight_decay, 0.001)
        self.assertEqual(config.training.lora_rank, 32)
        self.assertEqual(config.training.lora_alpha, 32)
        self.assertEqual(config.training.seed, 3407)
        self.assertEqual(
            config.training.rehearsal_dataset,
            "example/rehearsal-corpus",
        )
        self.assertEqual(config.training.rehearsal_rows_per_epoch, 500)
        self.assertEqual(config.training.inner_loss, "dft")

    def test_training_config_accepts_asft_fields(self) -> None:
        from src.helper import unflatten_toml_dict
        from src.typing.config import AppConfig

        config_dict = toml.loads(
            """
model_name = "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
code_host_cache_path = "./code"
container_ids = ["container-a"]
train_data_save_folder = "./train-data"

[task]
class = "tests.fixtures.stub_task.ValidStubTask"

[training]
inner_loss = "asft"
asft_mode = "sft+kl"
asft_kl_weight = 0.2
asft_kl_direction = "reverse"
asft_reference_policy = "frozen_copy"
asft_streaming = "hybrid"
asft_ref_microbatch_size = 3
asft_seq_chunk_size = 256
asft_normalize_by = "weights"
"""
        )

        config = AppConfig(**unflatten_toml_dict(config_dict))

        self.assertEqual(config.training.inner_loss, "asft")
        self.assertEqual(config.training.asft_mode, "sft+kl")
        self.assertEqual(config.training.asft_kl_weight, 0.2)
        self.assertEqual(config.training.asft_kl_direction, "reverse")
        self.assertEqual(config.training.asft_reference_policy, "frozen_copy")
        self.assertEqual(config.training.asft_streaming, "hybrid")
        self.assertEqual(config.training.asft_ref_microbatch_size, 3)
        self.assertEqual(config.training.asft_seq_chunk_size, 256)
        self.assertEqual(config.training.asft_normalize_by, "weights")

    def test_collect_training_window_uses_latest_n_generations(self) -> None:
        from scripts.run_train_loop import _collect_training_window_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            generation_root = Path(temp_dir)
            for generation_id in range(4):
                self._write_training_rows(
                    generation_root
                    / f"generation_{generation_id}"
                    / "sft_training_rows.jsonl",
                    generation_id,
                )

            training_paths = _collect_training_window_paths(
                generation_root,
                end_generation_id=3,
                window_size=3,
            )

        self.assertEqual(
            training_paths,
            [
                generation_root
                / "generation_1"
                / "exports"
                / "sft"
                / "test-export"
                / "sft_training_rows.jsonl",
                generation_root
                / "generation_2"
                / "exports"
                / "sft"
                / "test-export"
                / "sft_training_rows.jsonl",
                generation_root
                / "generation_3"
                / "exports"
                / "sft"
                / "test-export"
                / "sft_training_rows.jsonl",
            ],
        )

    def test_generation_phase_requires_completion_marker(self) -> None:
        from scripts.run_train_loop import _generation_phase_complete
        from src.execution.generation_persistence import (
            GENERATION_COMPLETION_FILENAME,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            generation_root = Path(temp_dir)
            generation_dir = generation_root / "generation_0"
            generation_dir.mkdir()
            (generation_dir / "metadata.json").write_text(
                json.dumps(
                    {"generation_id": 0, "active_sft_export_id": "test-export"}
                ),
                encoding="utf-8",
            )
            self._write_training_rows(generation_dir, 0)

            self.assertFalse(_generation_phase_complete(generation_root, 0))
            (generation_dir / GENERATION_COMPLETION_FILENAME).write_text(
                json.dumps({"schema_version": 2, "generation_id": 0}),
                encoding="utf-8",
            )
            self.assertTrue(_generation_phase_complete(generation_root, 0))

    def test_completed_generation_publication_corruption_fails_closed(self) -> None:
        from scripts.run_train_loop import (
            CompletedGenerationPublicationError,
            _generation_phase_complete,
        )
        from src.execution.generation_persistence import (
            GENERATION_COMPLETION_FILENAME,
        )

        cases = (
            "malformed_marker",
            "missing_metadata",
            "malformed_metadata",
            "malformed_latest",
            "missing_rows",
            "mismatched_export",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                generation_root = Path(temp_dir)
                generation_dir = generation_root / "generation_0"
                generation_dir.mkdir()
                journal_path = generation_dir / "all_episodes.jsonl"
                checkpoint_path = generation_dir / "checkpoint.json"
                journal_path.write_bytes(b"authoritative journal\n")
                checkpoint_path.write_bytes(b"authoritative checkpoint\n")
                authoritative_bytes = {
                    journal_path: journal_path.read_bytes(),
                    checkpoint_path: checkpoint_path.read_bytes(),
                }
                marker_path = generation_dir / GENERATION_COMPLETION_FILENAME
                marker_path.write_text(
                    "{" if case == "malformed_marker" else json.dumps(
                        {"schema_version": 2, "generation_id": 0}
                    ),
                    encoding="utf-8",
                )
                if case != "missing_metadata":
                    (generation_dir / "metadata.json").write_text(
                        "{" if case == "malformed_metadata" else json.dumps(
                            {
                                "generation_id": 0,
                                "active_sft_export_id": (
                                    "other-export"
                                    if case == "mismatched_export"
                                    else "test-export"
                                ),
                            }
                        ),
                        encoding="utf-8",
                    )
                if case not in {
                    "malformed_marker",
                    "missing_metadata",
                    "malformed_metadata",
                }:
                    self._write_training_rows(generation_dir, 0)
                    latest_path = generation_dir / "exports" / "sft" / "latest.json"
                    if case == "malformed_latest":
                        latest_path.write_text("{", encoding="utf-8")
                    elif case == "missing_rows":
                        rows_path = (
                            generation_dir
                            / "exports"
                            / "sft"
                            / "test-export"
                            / "sft_training_rows.jsonl"
                        )
                        rows_path.unlink()

                with self.assertRaises(CompletedGenerationPublicationError):
                    _generation_phase_complete(generation_root, 0)
                self.assertEqual(
                    {path: path.read_bytes() for path in authoritative_bytes},
                    authoritative_bytes,
                )

    def test_completed_generation_rejects_serving_lineage_mismatch(self) -> None:
        from scripts.run_train_loop import (
            CompletedGenerationPublicationError,
            _generation_phase_complete,
        )
        from src.execution.generation_persistence import serving_lineage_from_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generation_dir = root / "generation_0"
            generation_dir.mkdir()
            (generation_dir / "COMPLETED.json").write_text(
                json.dumps({"schema_version": 2, "generation_id": 0}),
                encoding="utf-8",
            )
            (generation_dir / "metadata.json").write_text(
                json.dumps({"generation_id": 0, "active_sft_export_id": "test-export"}),
                encoding="utf-8",
            )
            self._write_training_rows(generation_dir, 0)
            config = self.make_config(root, num_generations=1)
            expected = serving_lineage_from_config(config, 0)
            manifest = {"schema_version": 2, **expected}
            manifest["request_model"] = "wrong-model"
            (generation_dir / "generation_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                CompletedGenerationPublicationError,
                "invalid publication artifacts",
            ):
                _generation_phase_complete(
                    root,
                    0,
                    expected_lineage=expected,
                )

    @patch("scripts.run_train_loop.subprocess.Popen")
    def test_invoke_train_sft_passes_sft_and_rehearsal_flags(
        self,
        mock_popen: MagicMock,
    ) -> None:
        from scripts.run_train_loop import _invoke_train_sft

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            config.training.max_steps = -1
            config.training.num_train_epochs = 2
            config.training.gradient_accumulation_steps = 16
            config.training.learning_rate = 1e-4
            config.training.warmup_ratio = 0.03
            config.training.lr_scheduler_type = "linear"
            config.training.weight_decay = 0.001
            config.training.lora_rank = 32
            config.training.lora_alpha = 32
            config.training.lora_dropout = 0.0
            config.training.seed = 3407
            config.training.rehearsal_dataset = "example/rehearsal-corpus"
            config.training.rehearsal_rows_per_epoch = 500
            config.training.rehearsal_seed = 2026
            config.training.rehearsal_prompt = "Continue this reference passage"
            config.training.inner_loss = "dft"
            config.training.dft_impl = "fused"

            proc = MagicMock()
            proc.stdout = iter([str(base_dir / "adapter") + "\n"])
            proc.wait.return_value = 0
            mock_popen.return_value = proc

            result = _invoke_train_sft(
                config,
                training_paths=[base_dir / "rows.jsonl"],
                output_dir=base_dir / "adapter",
                export_format="lora",
            )

        self.assertEqual(result, base_dir / "adapter")
        cmd = mock_popen.call_args.args[0]
        self.assertIn("--num-train-epochs", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--warmup-ratio", cmd)
        self.assertIn("0.03", cmd)
        self.assertIn("--lr-scheduler-type", cmd)
        self.assertIn("linear", cmd)
        self.assertIn("--weight-decay", cmd)
        self.assertIn("0.001", cmd)
        self.assertIn("--lora-rank", cmd)
        self.assertIn("32", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("3407", cmd)
        self.assertIn("--rehearsal-dataset", cmd)
        self.assertIn("example/rehearsal-corpus", cmd)
        self.assertIn("--rehearsal-rows-per-epoch", cmd)
        self.assertIn("500", cmd)
        self.assertIn("--rehearsal-prompt", cmd)
        self.assertIn("Continue this reference passage", cmd)
        self.assertIn("--inner-loss", cmd)
        self.assertIn("dft", cmd)
        self.assertIn("--dft-impl", cmd)
        self.assertIn("fused", cmd)

    @patch("scripts.run_train_loop.subprocess.Popen")
    def test_invoke_train_sft_passes_asft_flags_only_for_asft(
        self,
        mock_popen: MagicMock,
    ) -> None:
        from scripts.run_train_loop import _invoke_train_sft

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            config.training.inner_loss = "asft"
            config.training.asft_mode = "asft"
            config.training.asft_kl_weight = 0.25
            config.training.asft_kl_direction = "reverse"
            config.training.asft_reference_policy = "disable_adapter"
            config.training.asft_streaming = "batch"
            config.training.asft_ref_microbatch_size = 2
            config.training.asft_seq_chunk_size = 128
            config.training.asft_normalize_by = "weights"

            proc = MagicMock()
            proc.stdout = iter([str(base_dir / "adapter") + "\n"])
            proc.wait.return_value = 0
            mock_popen.return_value = proc

            _invoke_train_sft(
                config,
                training_paths=[base_dir / "rows.jsonl"],
                output_dir=base_dir / "adapter",
                export_format="lora",
            )

        cmd = mock_popen.call_args.args[0]
        self.assertIn("--inner-loss", cmd)
        self.assertIn("asft", cmd)
        self.assertIn("--asft-mode", cmd)
        self.assertIn("asft", cmd)
        self.assertIn("--asft-kl-weight", cmd)
        self.assertIn("0.25", cmd)
        self.assertIn("--asft-kl-direction", cmd)
        self.assertIn("reverse", cmd)
        self.assertIn("--asft-reference-policy", cmd)
        self.assertIn("disable_adapter", cmd)
        self.assertIn("--asft-streaming", cmd)
        self.assertIn("batch", cmd)
        self.assertIn("--asft-ref-microbatch-size", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--asft-seq-chunk-size", cmd)
        self.assertIn("128", cmd)
        self.assertIn("--asft-normalize-by", cmd)
        self.assertIn("weights", cmd)

        config.training.inner_loss = "sft"
        proc = MagicMock()
        proc.stdout = iter([str(base_dir / "adapter") + "\n"])
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        _invoke_train_sft(
            config,
            training_paths=[base_dir / "rows.jsonl"],
            output_dir=base_dir / "adapter",
            export_format="lora",
        )

        sft_cmd = mock_popen.call_args.args[0]
        self.assertNotIn("--asft-mode", sft_cmd)
        self.assertNotIn("--asft-kl-weight", sft_cmd)
        self.assertNotIn("--asft-reference-policy", sft_cmd)

    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_first_generation_without_adapter(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            served_adapters: list[str | None] = []

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                served_adapters.append(generation_config.vllm.lora_adapter_path)
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = lambda *_args, **kwargs: self._write_training_artifact(
                kwargs["output_dir"], kwargs["export_format"]
            )

            run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )

        self.assertEqual(served_adapters[0], None)

    def test_run_loop_rejects_unsafe_run_id_before_creating_output(self) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, num_generations=1)

            with self.assertRaisesRegex(ValueError, "safe path component"):
                run_loop(config, run_id="../escape", task=MagicMock())

            self.assertFalse((base_dir / "escape").exists())

    @patch(
        "scripts.run_train_loop.generate_readable_run_id",
        return_value="nested/run",
    )
    def test_run_loop_validates_generated_run_id(
        self,
        _mock_generate_run_id: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_config(Path(temp_dir), num_generations=1)

            with self.assertRaisesRegex(ValueError, "safe path component"):
                run_loop(config, task=MagicMock())

    def test_generation_run_lock_is_lifetime_scoped_and_persistent(self) -> None:
        from scripts.run_train_loop import _generation_run_lock

        with tempfile.TemporaryDirectory() as temp_dir:
            generation_root = Path(temp_dir) / "generations" / "run-123"
            lock_path = generation_root / ".run.lock"

            with _generation_run_lock(generation_root, "run-123"):
                self.assertTrue(lock_path.exists())
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with _generation_run_lock(generation_root, "run-123"):
                        self.fail("contending lock unexpectedly acquired")

            self.assertTrue(lock_path.exists())
            with _generation_run_lock(generation_root, "run-123"):
                pass
            self.assertTrue(lock_path.exists())

    def test_generation_run_locks_allow_distinct_run_ids(self) -> None:
        from scripts.run_train_loop import _generation_run_lock

        with tempfile.TemporaryDirectory() as temp_dir:
            generation_base = Path(temp_dir) / "generations"
            with _generation_run_lock(generation_base / "run-a", "run-a"):
                with _generation_run_lock(generation_base / "run-b", "run-b"):
                    self.assertTrue((generation_base / "run-a" / ".run.lock").exists())
                    self.assertTrue((generation_base / "run-b" / ".run.lock").exists())

    def test_run_loop_holds_generation_lock_for_locked_body(self) -> None:
        from scripts.run_train_loop import _generation_run_lock, run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, num_generations=1)
            generation_root = base_dir / "generations" / "run-123"

            def assert_lock_held(*_args, **_kwargs) -> list[dict]:
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with _generation_run_lock(generation_root, "run-123"):
                        self.fail("run_loop did not hold its generation lock")
                return []

            with patch(
                "scripts.run_train_loop._run_loop_locked",
                side_effect=assert_lock_held,
            ):
                self.assertEqual(
                    run_loop(config, run_id="run-123", task=MagicMock()),
                    [],
                )

            self.assertTrue((generation_root / ".run.lock").exists())

    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_restarts_backend_with_new_adapter(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            served_adapters: list[str | None] = []
            adapter_dir = base_dir / "adapters" / "run-123" / "after_generation_0"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                served_adapters.append(generation_config.vllm.lora_adapter_path)
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = lambda *_args, **kwargs: self._write_training_artifact(
                kwargs["output_dir"], kwargs["export_format"]
            )

            run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )

        self.assertEqual(served_adapters, [None, str(adapter_dir)])
        self.assertEqual(mock_train_sft.call_args.kwargs["export_format"], "lora")
        self.assertIsNone(
            mock_run_generation_phase.call_args_list[0].kwargs[
                "predecessor_generation_id"
            ]
        )
        self.assertEqual(
            mock_run_generation_phase.call_args_list[1].kwargs[
                "predecessor_generation_id"
            ],
            0,
        )

    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_uses_merged_export_as_vllm_local_model(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            config.training.export_format = "merged_16bit"
            served_adapters: list[str | None] = []
            served_local_models: list[str | None] = []
            merged_model_dir = base_dir / "adapters" / "run-123" / "after_generation_0"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                served_adapters.append(generation_config.vllm.lora_adapter_path)
                served_local_models.append(generation_config.vllm.local_model_path)
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = lambda *_args, **kwargs: self._write_training_artifact(
                kwargs["output_dir"], kwargs["export_format"]
            )

            run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )

        self.assertEqual(served_adapters, [None, None])
        self.assertEqual(served_local_models, [None, str(merged_model_dir)])
        self.assertEqual(
            mock_train_sft.call_args.kwargs["export_format"],
            "merged_16bit",
        )

    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_rejects_peft_lora_for_llama_backend(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            config.model_name = "llama:unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
            config.training.export_format = "lora"
            served_adapters: list[str | None] = []

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                served_adapters.append(
                    generation_config.llama.lora_adapter_path
                    if generation_config.llama is not None
                    else None
                )
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = lambda *_args, **kwargs: self._write_training_artifact(
                kwargs["output_dir"], kwargs["export_format"]
            )

            with self.assertRaisesRegex(ValueError, "PEFT LoRA"):
                run_loop(
                    config,
                    run_id="run-123",
                    docker_client=MagicMock(),
                    task=MagicMock(),
                )

        self.assertEqual(served_adapters, [])
        mock_train_sft.assert_not_called()

    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_uses_explicit_gguf_export_for_llama_backend(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            config.model_name = "llama:unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
            config.training.export_format = "gguf"
            served_models: list[str] = []
            merged_model_path = (
                base_dir / "adapters" / "run-123" / "after_generation_0.gguf"
            )

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                served_models.append(generation_config.model_name)
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = lambda *_args, **kwargs: self._write_training_artifact(
                kwargs["output_dir"], kwargs["export_format"]
            )

            run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )

        self.assertEqual(
            served_models,
            [
                "llama:unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
                f"llama:{merged_model_path}",
            ],
        )
        self.assertEqual(mock_train_sft.call_args.kwargs["export_format"], "gguf")

    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    @patch(
        "scripts.run_train_loop.time.perf_counter",
        side_effect=[100.0, 130.0, 155.0, 160.0],
    )
    def test_run_loop_two_generations(
        self,
        _mock_perf_counter: MagicMock,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            events: list[str] = []
            adapter_dir = base_dir / "adapters" / "run-123" / "after_generation_0"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                events.append(
                    f"generation:{generation_id}:{generation_config.vllm.lora_adapter_path}"
                )
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(
                    generation_id,
                    total_input_tokens=100 + generation_id,
                    total_output_tokens=50 + generation_id,
                    duration_seconds=5.0 + generation_id,
                )

            def train_side_effect(*args, **kwargs) -> Path:
                training_paths = [str(p) for p in kwargs["training_paths"]]
                events.append(
                    f"train:{Path(kwargs['output_dir']).name}:{training_paths}"
                )
                return self._write_training_artifact(
                    kwargs["output_dir"], kwargs["export_format"]
                )

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = train_side_effect

            results = run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )
            summary_path = (
                Path(config.generation.generation_output_dir) / "run-123" / "run.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(
            events,
            [
                "generation:0:None",
                f"train:after_generation_0:['{base_dir / 'generations' / 'run-123' / 'generation_0' / 'exports' / 'sft' / 'test-export' / 'sft_training_rows.jsonl'}']",
                f"generation:1:{adapter_dir}",
            ],
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(summary["run_id"], "run-123")
        self.assertEqual(summary["status"], "completed")
        self.assertIsNotNone(summary["ended_at"])
        self.assertEqual(summary["num_generations"], 2)
        self.assertEqual(summary["training_window_size"], 3)
        self.assertEqual(summary["total_wall_clock_seconds"], 60.0)
        self.assertEqual(summary["total_agent_seconds"], 11.0)
        self.assertEqual(summary["total_tokens"], 302)
        self.assertEqual(
            summary["generations"][0]["metrics"]["total_input_tokens"], 100
        )
        self.assertEqual(
            summary["generations"][0]["metrics"]["total_output_tokens"], 50
        )
        self.assertEqual(summary["generations"][1]["metrics"]["total_tokens"], 152)
        self.assertEqual(
            summary["generations"][1]["served_adapter_dir"], str(adapter_dir)
        )

    @patch("scripts.run_train_loop._git_dirty", return_value=False)
    @patch("scripts.run_train_loop._git_short_sha", return_value="abc1234")
    @patch(
        "scripts.run_train_loop.detect_hardware_tags",
        return_value=["gpu-model", "24gb", "1x"],
    )
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_writes_run_json_at_start_with_identity_and_tags(
        self,
        mock_run_generation_phase: MagicMock,
        _mock_detect_hardware_tags: MagicMock,
        _mock_git_short_sha: MagicMock,
        _mock_git_dirty: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, num_generations=1)
            config.observability.detect_hardware = True
            config.observability.hardware_tags = ["Cloud Spot", "24gb"]
            config.observability.load_tags = ["Nightly Bench"]
            task = MagicMock()
            task.name = "valid-stub"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                run_path = base_dir / "generations" / run_id / "run.json"
                self.assertTrue(run_path.exists())
                initial = json.loads(run_path.read_text(encoding="utf-8"))
                self.assertEqual(initial["status"], "in_progress")
                self.assertIsNone(initial["ended_at"])
                self.assertEqual(initial["generations"], [])
                self.assertEqual(initial["task_name"], "valid-stub")
                self.assertEqual(initial["config_path"], "config/config-test.toml")

                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect

            run_loop(
                config,
                run_id="run-123",
                docker_client=MagicMock(),
                task=task,
                config_path="config/config-test.toml",
                cli_hardware_tags=["Manual Trial"],
                cli_load_tags=["Ad Hoc"],
            )
            run_doc = json.loads(
                (base_dir / "generations" / "run-123" / "run.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(run_doc["status"], "completed")
        self.assertEqual(run_doc["commit_id"], "abc1234")
        self.assertFalse(run_doc["git_dirty"])
        self.assertEqual(
            run_doc["hardware_tags"],
            ["gpu-model", "24gb", "1x", "cloud-spot", "manual-trial"],
        )
        self.assertEqual(run_doc["load_tags"], ["nightly-bench", "ad-hoc"])

    @patch(
        "scripts.run_train_loop.run_generation_phase", side_effect=RuntimeError("boom")
    )
    def test_run_loop_finalizes_run_json_on_failure(
        self,
        mock_run_generation_phase: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, num_generations=1)
            task = MagicMock()
            task.name = "valid-stub"

            with self.assertRaises(RuntimeError):
                run_loop(
                    config,
                    run_id="run-123",
                    docker_client=MagicMock(),
                    task=task,
                )
            run_doc = json.loads(
                (base_dir / "generations" / "run-123" / "run.json").read_text(
                    encoding="utf-8"
                )
            )

        mock_run_generation_phase.assert_called_once()
        self.assertEqual(run_doc["status"], "failed")
        self.assertIsNotNone(run_doc["ended_at"])

    @patch("scripts.run_train_loop._write_run_doc", side_effect=OSError("no write"))
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_json_initial_write_failure_aborts(
        self,
        mock_run_generation_phase: MagicMock,
        _mock_write_run_doc: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, num_generations=1)
            task = MagicMock()
            task.name = "valid-stub"

            with self.assertRaises(OSError):
                run_loop(
                    config,
                    run_id="run-123",
                    docker_client=MagicMock(),
                    task=task,
                )

        mock_run_generation_phase.assert_not_called()

    def test_write_run_doc_writes_atomically(self) -> None:
        from scripts.run_train_loop import _write_run_doc

        with tempfile.TemporaryDirectory() as temp_dir:
            run_path = Path(temp_dir) / "run.json"
            replace_calls: list[tuple[Path, Path]] = []
            original_replace = Path.replace

            def spy_replace(self_path: Path, target: Path) -> Path:
                replace_calls.append((self_path, target))
                return original_replace(self_path, target)

            with patch.object(Path, "replace", new=spy_replace):
                _write_run_doc(run_path, {"run_id": "run-123"})

            payload = json.loads(run_path.read_text(encoding="utf-8"))
            temp_files = list(run_path.parent.glob("*.tmp"))

        self.assertEqual(payload, {"run_id": "run-123"})
        self.assertEqual(replace_calls[0][1], run_path)
        self.assertEqual(temp_files, [])

    @patch("scripts.run_train_loop.resolve_harness_class")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_harness_identity_records_type_profile_and_class(
        self,
        mock_run_generation_phase: MagicMock,
        mock_resolve_harness_class: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop
        from src.typing.config import HarnessConfig

        class DummyHarness:
            pass

        DummyHarness.__module__ = "tests.fake_harness"
        DummyHarness.__qualname__ = "DummyHarness"
        mock_resolve_harness_class.return_value = DummyHarness

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, num_generations=1)
            config.harness = HarnessConfig(
                name="container",
                container={
                    "profile": "ms_agent",
                    "image": "nsl/test:latest",
                    "profile_config": {"model": "nsl-test"},
                },
            )
            task = MagicMock()
            task.name = "valid-stub"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect
            run_loop(
                config,
                run_id="run-123",
                docker_client=MagicMock(),
                task=task,
            )
            run_doc = json.loads(
                (base_dir / "generations" / "run-123" / "run.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(run_doc["harness_type"], "container")
        self.assertEqual(run_doc["harness_profile"], "ms_agent")
        self.assertEqual(run_doc["harness_class"], "tests.fake_harness.DummyHarness")

    def test_commit_id_falls_back_to_null(self) -> None:
        from scripts.run_train_loop import _git_short_sha

        with patch(
            "scripts.run_train_loop.subprocess.run", side_effect=OSError("git missing")
        ):
            self.assertIsNone(_git_short_sha())

    def test_git_dirty_records_tracked_changes(self) -> None:
        from scripts.run_train_loop import _git_dirty

        unstaged = MagicMock(returncode=1)
        staged = MagicMock(returncode=0)

        with patch(
            "scripts.run_train_loop.subprocess.run", side_effect=[unstaged, staged]
        ):
            self.assertTrue(_git_dirty())

    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_skips_training_after_final_generation(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = lambda *_args, **kwargs: self._write_training_artifact(
                kwargs["output_dir"], kwargs["export_format"]
            )

            run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )

        mock_train_sft.assert_called_once()

    @patch("scripts.run_train_loop.wait_for_gpu_memory_release")
    @patch("scripts.run_train_loop._vllm_server_is_ready", return_value=False)
    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_waits_for_gpu_release_after_managed_vllm_generation(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
        _mock_vllm_server_is_ready: MagicMock,
        mock_wait_for_gpu_memory_release: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            config.training.gpu_wait_timeout_seconds = 60
            events: list[str] = []
            adapter_dir = base_dir / "adapters" / "after_generation_0"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                events.append(f"generation:{generation_id}")
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            def train_side_effect(*args, **kwargs) -> Path:
                events.append("train")
                return self._write_training_artifact(
                    kwargs["output_dir"], kwargs["export_format"]
                )

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = train_side_effect
            mock_wait_for_gpu_memory_release.side_effect = lambda **kwargs: (
                events.append("wait")
            )

            run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )

        self.assertEqual(events, ["generation:0", "wait", "train", "generation:1"])
        mock_wait_for_gpu_memory_release.assert_called_once_with(
            min_free_memory_fraction=config.training.gpu_wait_min_free_memory_fraction,
            timeout_s=config.training.gpu_wait_timeout_seconds,
        )

    @patch("scripts.run_train_loop.wait_for_gpu_memory_release")
    @patch("scripts.run_train_loop._vllm_server_is_ready", return_value=True)
    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_skips_gpu_wait_when_existing_server_was_reused(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
        _mock_vllm_server_is_ready: MagicMock,
        mock_wait_for_gpu_memory_release: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir)
            config.training.gpu_wait_timeout_seconds = 60
            adapter_dir = base_dir / "adapters" / "after_generation_0"

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect
            mock_train_sft.side_effect = lambda *_args, **kwargs: self._write_training_artifact(
                kwargs["output_dir"], kwargs["export_format"]
            )

            run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )

        mock_wait_for_gpu_memory_release.assert_not_called()
        mock_train_sft.assert_called_once()

    @patch("scripts.run_train_loop.wait_for_gpu_memory_release")
    @patch("scripts.run_train_loop._vllm_server_is_ready", return_value=False)
    @patch("scripts.run_train_loop._invoke_train_sft")
    @patch("scripts.run_train_loop.run_generation_phase")
    def test_run_loop_skips_gpu_wait_after_final_generation(
        self,
        mock_run_generation_phase: MagicMock,
        mock_train_sft: MagicMock,
        _mock_vllm_server_is_ready: MagicMock,
        mock_wait_for_gpu_memory_release: MagicMock,
    ) -> None:
        from scripts.run_train_loop import run_loop

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config = self.make_config(base_dir, num_generations=1)
            config.training.gpu_wait_timeout_seconds = 60

            def run_generation_side_effect(
                generation_config,
                generation_id: int,
                run_id: str,
                docker_client,
                metrics_collector=None,
                **_kwargs,
            ) -> tuple[Path, GenerationData]:
                generation_dir = (
                    Path(generation_config.generation.generation_output_dir)
                    / f"generation_{generation_id}"
                )
                self._write_training_rows(
                    generation_dir / "sft_training_rows.jsonl",
                    generation_id,
                )
                return generation_dir, self._make_generation_data(generation_id)

            mock_run_generation_phase.side_effect = run_generation_side_effect

            run_loop(
                config, run_id="run-123", docker_client=MagicMock(), task=MagicMock()
            )

        mock_wait_for_gpu_memory_release.assert_not_called()
        mock_train_sft.assert_not_called()


if __name__ == "__main__":
    unittest.main()
