"""Tests that OpenAI clients are created with the configured inference timeout."""

import unittest
from unittest.mock import MagicMock, patch

from src.typing.config import AppConfig


class VllmClientTimeoutTests(unittest.TestCase):
    def _make_app_config(self, inference_timeout: int = 120) -> AppConfig:
        return AppConfig(
            task={"class": "tests.fixtures.stub_task.ValidStubTask"},
            model_name="vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            code_host_cache_path="/tmp/code",
            container_ids=["c1"],
            main_container_idx=0,
            dynamic_container=False,
            docker_compose_dir="/tmp/compose",
            train_data_save_folder="/tmp/data",
            inference={"timeout": inference_timeout},
            vllm={"served_model_name": "test-model"},
        )

    @patch("src.backend.vllm.is_http_ready", return_value=True)
    @patch("src.backend.vllm.OpenAI")
    def test_existing_server_client_uses_inference_timeout(
        self, MockOpenAI: MagicMock, is_http_ready: MagicMock
    ) -> None:
        """When connecting to an existing vLLM server, the OpenAI client
        should be created with timeout= from inference config."""
        from src.backend.vllm import setup_vllm

        config = self._make_app_config(inference_timeout=90)

        with setup_vllm(config):
            pass

        MockOpenAI.assert_called_once()
        call_kwargs = MockOpenAI.call_args
        self.assertEqual(call_kwargs.kwargs.get("timeout"), 90)

class LlamaClientTimeoutTests(unittest.TestCase):
    def _make_app_config(self, inference_timeout: int = 120) -> AppConfig:
        return AppConfig(
            task={"class": "tests.fixtures.stub_task.ValidStubTask"},
            model_name="llama:test-model",
            code_host_cache_path="/tmp/code",
            container_ids=["c1"],
            main_container_idx=0,
            dynamic_container=False,
            docker_compose_dir="/tmp/compose",
            train_data_save_folder="/tmp/data",
            inference={"timeout": inference_timeout},
        )

    @patch("src.backend.llama.is_http_ready", return_value=True)
    @patch("src.backend.llama.OpenAI")
    def test_llama_client_uses_inference_timeout(
        self, MockOpenAI: MagicMock, is_http_ready: MagicMock
    ) -> None:
        from src.backend.llama import setup_llama

        config = self._make_app_config(inference_timeout=60)

        with setup_llama(config):
            pass

        MockOpenAI.assert_called_once()
        call_kwargs = MockOpenAI.call_args
        self.assertEqual(call_kwargs.kwargs.get("timeout"), 60)


if __name__ == "__main__":
    unittest.main()
