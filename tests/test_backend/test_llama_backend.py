import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.typing.config import AppConfig


def make_app_config(model_name: str, **llama_config) -> AppConfig:
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name=model_name,
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
        llama=llama_config or None,
    )


class LlamaBackendTests(unittest.TestCase):
    def test_config_accepts_long_context_and_single_slot(self) -> None:
        config = make_app_config(
            "llama:example.gguf",
            ctx_size=65536,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            flash_attention=True,
            parallel_slots=1,
            kv_unified=False,
            ctx_checkpoints=0,
            cache_ram_mib=0,
        )

        self.assertEqual(config.llama.ctx_size, 65536)
        self.assertEqual(config.llama.cache_type_k, "q4_0")
        self.assertEqual(config.llama.cache_type_v, "q4_0")
        self.assertTrue(config.llama.flash_attention)
        self.assertEqual(config.llama.parallel_slots, 1)
        self.assertFalse(config.llama.kv_unified)
        self.assertEqual(config.llama.ctx_checkpoints, 0)
        self.assertEqual(config.llama.cache_ram_mib, 0)

    @patch("src.backend.llama.OpenAI")
    @patch("src.backend.llama.is_http_ready", return_value=True)
    def test_ready_server_does_not_hide_missing_local_adapter(
        self,
        _mock_ready: MagicMock,
        mock_openai_cls: MagicMock,
    ) -> None:
        from src.backend.llama import setup_llama

        app_config = make_app_config(
            "llama:repo/model-GGUF",
            lora_adapter_path="/missing/adapter.gguf",
        )

        with self.assertRaises(FileNotFoundError):
            with setup_llama(app_config):
                pass

        mock_openai_cls.assert_not_called()

    @patch("src.backend.llama.terminate_process")
    @patch("src.backend.llama.get_genner")
    @patch("src.backend.llama.OpenAI")
    @patch("src.backend.llama._wait_for_llama_server_ready")
    @patch("src.backend.llama.subprocess.Popen")
    @patch("src.backend.llama._ensure_llama_server")
    @patch("src.backend.llama.is_http_ready", return_value=False)
    def test_setup_llama_passes_lora_adapter_to_server(
        self,
        _mock_is_http_ready: MagicMock,
        mock_ensure_llama_server: MagicMock,
        mock_popen: MagicMock,
        _mock_wait_for_ready: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
        mock_terminate_process: MagicMock,
    ) -> None:
        from src.backend.llama import setup_llama

        mock_ensure_llama_server.return_value = Path("/usr/bin/llama-server")
        mock_popen.return_value = MagicMock()
        mock_openai_cls.return_value = MagicMock()
        mock_get_genner.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            model_path = base_dir / "model.gguf"
            adapter_path = base_dir / "adapter.gguf"
            model_path.write_text("model", encoding="utf-8")
            adapter_path.write_text("adapter", encoding="utf-8")
            app_config = make_app_config(
                f"llama:{model_path}",
                lora_adapter_path=str(adapter_path),
            )

            with setup_llama(app_config):
                pass

        command = mock_popen.call_args.args[0]
        self.assertIn("--lora", command)
        self.assertEqual(
            command[command.index("--lora") + 1],
            str(adapter_path.resolve()),
        )
        mock_terminate_process.assert_called_once()

    def test_build_llama_config_uses_repo_managed_context_and_kv_cache(self) -> None:
        from src.backend.llama import _build_llama_config

        app_config = make_app_config(
            "llama:/tmp/model.gguf",
            ctx_size=65536,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            flash_attention=True,
            parallel_slots=1,
            kv_unified=False,
            ctx_checkpoints=0,
            cache_ram_mib=0,
        )

        config = _build_llama_config(app_config, "http://localhost:8000")

        self.assertEqual(config.ctx_size, 65536)
        self.assertEqual(config.cache_type_k, "q4_0")
        self.assertEqual(config.cache_type_v, "q4_0")
        self.assertTrue(config.flash_attention)
        self.assertEqual(config.parallel_slots, 1)
        self.assertFalse(config.kv_unified)
        self.assertEqual(config.ctx_checkpoints, 0)
        self.assertEqual(config.cache_ram_mib, 0)

    @patch("src.backend.llama.terminate_process")
    @patch("src.backend.llama.get_genner")
    @patch("src.backend.llama.OpenAI")
    @patch("src.backend.llama._wait_for_llama_server_ready")
    @patch("src.backend.llama.subprocess.Popen")
    @patch("src.backend.llama._ensure_llama_server")
    @patch("src.backend.llama.is_http_ready", return_value=False)
    def test_setup_llama_passes_quantized_kv_cache_to_server(
        self,
        _mock_is_http_ready: MagicMock,
        mock_ensure_llama_server: MagicMock,
        mock_popen: MagicMock,
        _mock_wait_for_ready: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
        _mock_terminate_process: MagicMock,
    ) -> None:
        from src.backend.llama import setup_llama

        mock_ensure_llama_server.return_value = Path("/usr/bin/llama-server")
        mock_popen.return_value = MagicMock()
        mock_openai_cls.return_value = MagicMock()
        mock_get_genner.return_value = MagicMock()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.gguf"
            model_path.write_text("model", encoding="utf-8")
            app_config = make_app_config(
                f"llama:{model_path}",
                ctx_size=65536,
                cache_type_k="q4_0",
                cache_type_v="q4_0",
                flash_attention=True,
                parallel_slots=1,
                kv_unified=False,
                ctx_checkpoints=0,
                cache_ram_mib=0,
            )

            with setup_llama(app_config):
                pass

        command = mock_popen.call_args.args[0]
        self.assertEqual(command[command.index("--ctx-size") + 1], "65536")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q4_0")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q4_0")
        self.assertEqual(command[command.index("--flash-attn") + 1], "on")
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertIn("--no-kv-unified", command)
        self.assertEqual(command[command.index("--ctx-checkpoints") + 1], "0")
        self.assertEqual(command[command.index("--cache-ram") + 1], "0")


if __name__ == "__main__":
    unittest.main()
