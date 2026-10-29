import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.genner.Base import Genner
from src.typing.config import AppConfig


def make_app_config(
    model_name: str = "sglang:Qwen/Qwen3-8B",
    *,
    sglang: AppConfig.SglangConfig | None = None,
    vllm: AppConfig.VllmConfig | None = None,
) -> AppConfig:
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name=model_name,
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
        sglang=sglang,
        vllm=vllm,
    )


def write_mock_lora_adapter(adapter_dir: str | Path) -> Path:
    resolved_dir = Path(adapter_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    (resolved_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (resolved_dir / "adapter_model.safetensors").write_text(
        "weights",
        encoding="utf-8",
    )
    return resolved_dir


class TestBuildSglangDockerConfig(unittest.TestCase):
    def test_build_command_minimal(self) -> None:
        from src.backend.sglang import _build_sglang_container_command

        cmd = _build_sglang_container_command(
            "Qwen/Qwen3-8B",
            port=30000,
            served_model_name="Qwen/Qwen3-8B",
        )

        self.assertEqual(cmd[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertIn("--model-path", cmd)
        self.assertEqual(cmd[cmd.index("--model-path") + 1], "Qwen/Qwen3-8B")
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")
        self.assertEqual(cmd[cmd.index("--port") + 1], "30000")
        self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1], "0.85")

    def test_build_command_chunked_prefill_modes(self) -> None:
        from src.backend.sglang import _build_sglang_container_command

        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                cmd = _build_sglang_container_command(
                    "Qwen/Qwen3-8B",
                    port=30000,
                    served_model_name="Qwen/Qwen3-8B",
                    tensor_parallel_size=2,
                    enable_chunked_prefill=enabled,
                )
                self.assertEqual(cmd[cmd.index("--tp") + 1], "2")
                self.assertNotIn("--enable-chunked-prefill", cmd)
                if enabled:
                    self.assertNotIn("--chunked-prefill-size", cmd)
                else:
                    self.assertEqual(
                        cmd[cmd.index("--chunked-prefill-size") + 1], "-1"
                    )

    def test_build_command_forwards_all_configured_options(self) -> None:
        from src.backend.sglang import _build_sglang_container_command

        cmd = _build_sglang_container_command(
            "Qwen/Qwen3-8B",
            port=32123,
            served_model_name="Qwen/Qwen3-8B",
            cuda_graph_bs=[1, 2, 4, 8, 16],
            lora_paths={"a": "/adapters/a", "b": "/adapters/b"},
            max_lora_rank=64,
            lora_target_modules=["q_proj", "v_proj"],
            quantization="fp8",
            kv_cache_dtype="fp8_e5m2",
        )

        self.assertEqual(cmd[cmd.index("--port") + 1], "32123")
        graph_idx = cmd.index("--cuda-graph-bs")
        self.assertEqual(
            cmd[graph_idx + 1 : graph_idx + 6], ["1", "2", "4", "8", "16"]
        )
        self.assertEqual(cmd[cmd.index("--quantization") + 1], "fp8")
        self.assertEqual(cmd[cmd.index("--kv-cache-dtype") + 1], "fp8_e5m2")
        idx = cmd.index("--lora-paths")
        self.assertEqual(cmd[idx + 1 : idx + 3], ["a=/adapters/a", "b=/adapters/b"])
        self.assertEqual(cmd[cmd.index("--max-lora-rank") + 1], "64")
        modules_idx = cmd.index("--lora-target-modules")
        self.assertEqual(cmd[modules_idx + 1 : modules_idx + 3], ["q_proj", "v_proj"])

    def test_sglang_section_validation(self) -> None:
        invalid_cases = (
            (
                "missing-lora-rank",
                {"lora_adapters": {"a": "/tmp/a"}},
                "max_lora_rank",
            ),
            ("torchao-cuda-graph", {"quantization": "torchao"}, "torchao"),
            (
                "pinned-starvation",
                {
                    "lora_adapters": {"a": "/tmp/a"},
                    "max_lora_rank": 64,
                    "max_loras_per_batch": 1,
                    "pinned_lora_names": ["a"],
                },
                "pinned",
            ),
            (
                "speculative-chunked-prefill",
                {"speculative_algorithm": "EAGLE"},
                "chunked prefill",
            ),
        )
        for name, kwargs, message in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    AppConfig.SglangConfig(**kwargs)

        AppConfig.SglangConfig(quantization="torchao", disable_cuda_graph=True)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            AppConfig.SglangConfig(quantization="bitsandbytes")
        self.assertTrue(any("bitsandbytes" in str(r.message).lower() for r in records))

    def test_sglang_application_model_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cases = (
                (
                    "deepseek-native-fp8",
                    "sglang:deepseek-ai/DeepSeek-V3",
                    AppConfig.SglangConfig(quantization="fp8"),
                    "DeepSeek",
                ),
                (
                    "local-lora-served-name",
                    "sglang:local",
                    AppConfig.SglangConfig(
                        local_model_path=tmpdir,
                        lora_adapters={"policy": tmpdir},
                        max_lora_rank=64,
                    ),
                    "served_model_name",
                ),
                (
                    "mtp-non-deepseek",
                    "sglang:Qwen/Qwen3-8B",
                    AppConfig.SglangConfig(
                        speculative_algorithm="MTP",
                        enable_chunked_prefill=False,
                    ),
                    "DeepSeek",
                ),
            )
            for name, model, config, message in cases:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, message):
                        make_app_config(model, sglang=config)

    def test_build_command_tool_parser_inference(self) -> None:
        from src.backend.sglang import _build_sglang_config

        config = _build_sglang_config(
            make_app_config("sglang:Qwen/Qwen3-8B"),
            endpoint="http://127.0.0.1:30000",
        )

        self.assertEqual(config.tool_call_parser, "qwen25")

    def test_compile_cache_namespace_separates_engines(self) -> None:
        from src.backend._container_runtime import compile_cache_namespace

        vllm_ns = compile_cache_namespace(
            engine="vllm",
            model="Qwen/Qwen3-8B",
            tp=1,
            max_model_len=8192,
            quantization="fp8",
        )
        sglang_ns = compile_cache_namespace(
            engine="sglang",
            model="Qwen/Qwen3-8B",
            tp=1,
            max_model_len=8192,
            quantization="fp8",
        )

        self.assertNotEqual(vllm_ns, sglang_ns)
        self.assertTrue(vllm_ns.startswith("vllm/"))
        self.assertTrue(sglang_ns.startswith("sglang/"))

    def test_lora_adapter_path_validation(self) -> None:
        from src.backend._container_runtime import validate_lora_adapter_path

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "adapter_config.json"):
                validate_lora_adapter_path(tmpdir)

    def test_smoke_test_payload_shape(self) -> None:
        from src.backend.sglang import _build_sglang_smoke_test
        from src.genner.config import SglangServerConfig

        message = MagicMock()
        message.content = "pong"
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        client = MagicMock()
        client.chat.completions.create.return_value = response

        smoke = _build_sglang_smoke_test(
            client,
            SglangServerConfig(model="Qwen/Qwen3-8B", tool_call_parser="qwen25"),
        )
        self.assertEqual(smoke(), "pong")

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "auto")
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(len(kwargs["tools"]), 1)

    def test_session_extras_lora_helpers_call_correct_endpoints(self) -> None:
        from src.backend.sglang import SglangSessionExtras

        response = MagicMock()
        response.json.return_value = {"ok": True}
        with patch("src.backend.sglang.requests.post", return_value=response) as post:
            extras = SglangSessionExtras("http://127.0.0.1:30000/v1", timeout=7)
            self.assertEqual(extras.load_lora("policy", "/adapters/policy"), {"ok": True})
            extras.unload_lora("policy")
            extras.update_weights_from_disk("/models/policy")

        self.assertEqual(post.call_args_list[0].args[0], "http://127.0.0.1:30000/load_lora_adapter")
        self.assertEqual(
            post.call_args_list[0].kwargs["json"],
            {"lora_name": "policy", "lora_path": "/adapters/policy", "pinned": False},
        )
        self.assertEqual(post.call_args_list[1].args[0], "http://127.0.0.1:30000/unload_lora_adapter")
        self.assertEqual(post.call_args_list[2].args[0], "http://127.0.0.1:30000/update_weights_from_disk")

    def test_network_mode_loopback_vs_host(self) -> None:
        from src.backend._container_runtime import resolve_network_mode

        with patch.dict(os.environ, {"NSL_SGLANG_NETWORK_MODE": "loopback"}, clear=True):
            self.assertEqual(resolve_network_mode("NSL_SGLANG_NETWORK_MODE"), "loopback")
        with patch.dict(os.environ, {"NSL_SGLANG_NETWORK_MODE": "hostip"}, clear=True):
            self.assertEqual(resolve_network_mode("NSL_SGLANG_NETWORK_MODE"), "hostip")

    @patch("src.backend.sglang.get_genner")
    @patch("src.backend.sglang.OpenAI")
    @patch("src.backend.sglang._wait_for_sglang_ready", return_value="http://127.0.0.1:30000/v1/models")
    @patch("src.backend.sglang._start_sglang_container")
    @patch("src.backend.sglang.DockerClient")
    @patch("src.backend.sglang.is_http_ready", return_value=False)
    def test_setup_sglang_mounts_caches_and_chat_template(
        self,
        _http: MagicMock,
        docker_cls: MagicMock,
        start: MagicMock,
        _wait: MagicMock,
        _openai: MagicMock,
        get_genner: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.sglang import setup_sglang

        get_genner.return_value = MagicMock(spec=Genner)
        container = MagicMock()
        docker = MagicMock()
        docker.containers.get.side_effect = [NotFound(""), container]
        docker_cls.from_env.return_value = docker

        with tempfile.NamedTemporaryFile(suffix=".jinja") as handle:
            Path(handle.name).write_text(
                "{{ messages[0]['content'] }}", encoding="utf-8"
            )
            config = make_app_config(
                sglang=AppConfig.SglangConfig(chat_template_path=handle.name),
            )
            with setup_sglang(config):
                extra_volumes = start.call_args.kwargs["extra_volumes"]
                command = start.call_args.args[2]

        container_paths = [container_path for _host_path, container_path in extra_volumes]
        self.assertTrue(any(path.startswith("/root/.cache/sglang/") for path in container_paths))
        self.assertIn("/root/.cache/torch/inductor", container_paths)
        self.assertFalse(any(path.startswith("/tmp/") for path in container_paths))
        template_mounts = [
            container_path
            for _host_path, container_path in extra_volumes
            if container_path.startswith("/templates/")
        ]
        self.assertEqual(template_mounts, ["/templates/chat.jinja"])
        self.assertEqual(command[command.index("--chat-template") + 1], "/templates/chat.jinja")

    @patch("src.backend.sglang.subprocess.run")
    def test_start_sglang_container_forwards_gpu_and_hf_credentials(
        self, run: MagicMock
    ) -> None:
        from src.backend.sglang import _start_sglang_container

        run.return_value = MagicMock(returncode=0)
        with patch.dict(os.environ, {"HF_TOKEN": "secret"}, clear=True):
            _start_sglang_container(
                "nsl-sglang-30000",
                30000,
                ["python3", "-m", "sglang.launch_server", "--model-path", "x"],
                image="lmsysorg/sglang:test",
            )

        call_args = run.call_args.args[0]
        self.assertIn("HF_TOKEN=secret", call_args)
        self.assertIn("HUGGING_FACE_HUB_TOKEN=secret", call_args)
        self.assertIn("--device", call_args)
        self.assertIn("nvidia.com/gpu=all", call_args)
        self.assertNotIn("--gpus", call_args)

    def test_concurrent_vllm_sglang_same_gpu_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
            make_app_config(
                sglang=AppConfig.SglangConfig(
                    extra_env={"CUDA_VISIBLE_DEVICES": "1"},
                ),
                vllm=AppConfig.VllmConfig(
                    extra_env={"CUDA_VISIBLE_DEVICES": "0,1"},
                ),
            )


class TestSglangGenner(unittest.TestCase):
    def test_lora_routing_uses_message_meta_and_strips_meta(self) -> None:
        from src.genner.SglangServer import SglangServerGenner
        from src.genner.config import SglangServerConfig

        message = MagicMock()
        message.content = "ok"
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        response.usage = None
        client = MagicMock()
        client.chat.completions.create.return_value = response

        genner = SglangServerGenner(
            client,
            SglangServerConfig(
                model="base-model",
                lora_routing_enabled=True,
                default_lora_name="default",
            ),
        )
        result = genner.plist_completion(
            [
                {
                    "role": "user",
                    "content": "hello",
                    "meta": {"lora_adapter": "policy"},
                }
            ]
        )

        self.assertTrue(result.is_ok())
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "base-model:policy")
        self.assertNotIn("meta", kwargs["messages"][0])


if __name__ == "__main__":
    unittest.main()
