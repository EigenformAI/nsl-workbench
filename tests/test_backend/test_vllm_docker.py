import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.genner.Base import Genner
from src.typing.config import AppConfig


def make_app_config(
    model_name: str,
    gpu_mem: float = 0.85,
    chat_template_path: str | None = None,
    local_model_path: str | None = None,
    lora_adapter_path: str | None = None,
    served_model_name: str | None = None,
    compile_cache_dir: str | None = None,
    enable_auto_tool_choice: bool | None = None,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    cdi_devices: list[str] | None = None,
) -> AppConfig:
    has_vllm = any(
        v is not None
        for v in (
            chat_template_path,
            local_model_path,
            lora_adapter_path,
            served_model_name,
            compile_cache_dir,
            enable_auto_tool_choice,
            tool_call_parser,
            reasoning_parser,
            cdi_devices,
        )
    )
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name=model_name,
        gpu_memory_utilization=gpu_mem,
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
        vllm=AppConfig.VllmConfig(
            chat_template_path=chat_template_path,
            local_model_path=local_model_path,
            lora_adapter_path=lora_adapter_path,
            served_model_name=served_model_name,
            compile_cache_dir=compile_cache_dir,
            enable_auto_tool_choice=enable_auto_tool_choice,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            cdi_devices=cdi_devices or ["all"],
        )
        if has_vllm
        else None,
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


class TestBuildVllmDockerConfig(unittest.TestCase):
    def test_model_inventory_attests_only_matching_lora_card(self) -> None:
        from types import SimpleNamespace

        from src.backend.vllm import _model_inventory_attests_lora

        inventory = SimpleNamespace(
            data=[
                SimpleNamespace(id="base", root="base", parent=None),
                SimpleNamespace(id="adapter", root="/adapters/local", parent="base"),
                SimpleNamespace(id="other", root="/adapters/other", parent="base"),
            ]
        )

        self.assertTrue(_model_inventory_attests_lora(inventory, "adapter"))
        self.assertFalse(_model_inventory_attests_lora(inventory, "base"))
        self.assertFalse(_model_inventory_attests_lora(inventory, "missing"))

    def test_model_inventory_rejects_duplicate_or_malformed_lora_cards(self) -> None:
        from types import SimpleNamespace

        from src.backend.vllm import _model_inventory_attests_lora

        cases = [
            SimpleNamespace(data=[]),
            SimpleNamespace(data=[SimpleNamespace(id="adapter", parent="")]),
            SimpleNamespace(
                data=[
                    SimpleNamespace(id="adapter", parent="base"),
                    SimpleNamespace(id="adapter", parent="base"),
                ]
            ),
            SimpleNamespace(data="not-a-list"),
        ]
        for inventory in cases:
            with self.subTest(inventory=inventory):
                self.assertFalse(
                    _model_inventory_attests_lora(inventory, "adapter")
                )

    def test_vllm_cdi_devices_defaults_and_explicit_values(self) -> None:
        for devices, expected in (
            (None, ["all"]),
            (["all"], ["all"]),
            (["0", "1"], ["0", "1"]),
        ):
            with self.subTest(devices=devices):
                kwargs = {} if devices is None else {"cdi_devices": devices}
                self.assertEqual(AppConfig.VllmConfig(**kwargs).cdi_devices, expected)

    def test_vllm_cdi_devices_reject_malformed_lists(self) -> None:
        for devices in ([], [""], [" "], ["0,1"], ["0 1"], ["all", "0"]):
            with self.subTest(devices=devices):
                with self.assertRaises(ValueError):
                    AppConfig.VllmConfig(cdi_devices=devices)

    def test_network_mode_defaults_and_invalid_values_fall_back_to_auto(self) -> None:
        from src.backend.vllm import _get_network_mode

        for env in ({}, {"NSL_VLLM_NETWORK_MODE": "invalid"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertEqual(_get_network_mode(), "auto")

    def test_build_vllm_config_loads_chat_template_from_path(self) -> None:
        from src.backend.vllm import _build_vllm_config

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write("{{ messages[0]['content'] }}")
            handle.flush()

            config = make_app_config(
                "vllm:unsloth/Qwen2.5-Coder-7B-bnb-4bit",
                chat_template_path=handle.name,
            )

            vllm_config = _build_vllm_config(
                config,
                endpoint="http://127.0.0.1:8000",
            )

        self.assertEqual(vllm_config.chat_template, "{{ messages[0]['content'] }}")

    def test_build_vllm_config_resolves_legacy_model_alias(self) -> None:
        from src.backend.vllm import _build_vllm_config

        config = make_app_config("vllm:qwen2.5:7b-instruct")

        vllm_config = _build_vllm_config(
            config,
            endpoint="http://127.0.0.1:8000",
        )

        self.assertEqual(vllm_config.model, "Qwen/Qwen2.5-7B-Instruct")

    def test_build_vllm_config_targets_adapter_when_lora_present(self) -> None:
        from src.backend.vllm import ADAPTER_MODEL_NAME, _build_vllm_config

        config = make_app_config(
            "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            served_model_name="nsl-stub-sft",
            lora_adapter_path="/adapters/after_generation_0",
        )

        vllm_config = _build_vllm_config(config, endpoint="http://127.0.0.1:8000")

        self.assertEqual(vllm_config.model, ADAPTER_MODEL_NAME)

    def test_build_vllm_config_uses_served_name_when_no_adapter(self) -> None:
        from src.backend.vllm import _build_vllm_config

        config = make_app_config(
            "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            served_model_name="nsl-stub-sft",
        )

        vllm_config = _build_vllm_config(config, endpoint="http://127.0.0.1:8000")

        self.assertEqual(vllm_config.model, "nsl-stub-sft")

    def test_vllm_endpoint_configuration_accepts_supported_forms(self) -> None:
        endpoints = AppConfig.VllmConfig(
            endpoints=[
                {
                    "base_url": "https://ep0.example/v1",
                    "capacity": 6,
                    "api_key_env": "VLLM_EP0_KEY",
                },
                {"base_url": "https://ep1.example", "capacity": 12},
            ]
        ).endpoints
        self.assertEqual(len(endpoints), 2)
        self.assertEqual(endpoints[0].base_url, "https://ep0.example/v1")
        self.assertEqual(endpoints[0].capacity, 6)
        self.assertEqual(endpoints[0].api_key_env, "VLLM_EP0_KEY")
        self.assertEqual(
            AppConfig.VllmConfig(endpoint="http://127.0.0.1:8000").endpoint,
            "http://127.0.0.1:8000",
        )

    def test_vllm_endpoint_configuration_rejects_effective_id_collisions(self) -> None:
        cases = {
            "explicit-explicit": [
                {"id": "shared", "base_url": "https://ep0.example"},
                {"id": "shared", "base_url": "https://ep1.example"},
            ],
            "explicit-generated": [
                {"id": "vllm-1", "base_url": "https://ep0.example"},
                {"base_url": "https://ep1.example"},
            ],
        }
        for name, endpoints in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, "duplicate effective endpoint ID"
                ):
                    AppConfig.VllmConfig(endpoints=endpoints)

    def test_build_vllm_config_infers_hermes_parser_for_qwen_models(self) -> None:
        from src.backend.vllm import _build_vllm_config

        config = make_app_config("vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")

        vllm_config = _build_vllm_config(
            config,
            endpoint="http://127.0.0.1:8000",
        )

        self.assertTrue(vllm_config.enable_auto_tool_choice)
        self.assertEqual(vllm_config.tool_call_parser, "hermes")

    def test_build_vllm_config_propagates_optional_reasoning_parser(self) -> None:
        from src.backend.vllm import _build_vllm_config

        for configured, expected in ((None, None), ("qwen3", "qwen3")):
            with self.subTest(reasoning_parser=configured):
                config = make_app_config(
                    "vllm:Qwen/Qwen3-30B-A3B-AWQ",
                    reasoning_parser=configured,
                )
                vllm_config = _build_vllm_config(
                    config,
                    endpoint="http://127.0.0.1:8000",
                )
                self.assertEqual(vllm_config.reasoning_parser, expected)

    def test_build_container_command_model_modes(self) -> None:
        from src.backend.vllm import LOCAL_MODEL_CONTAINER_PATH, _build_container_command

        cases = (
            ("standard", "Qwen/Qwen2.5-7B-Instruct", False),
            ("bnb", "unsloth/Qwen2.5-Coder-7B-bnb-4bit", True),
            ("local", LOCAL_MODEL_CONTAINER_PATH, False),
        )
        for name, model, needs_bnb in cases:
            with self.subTest(name=name):
                cmd = _build_container_command(model, 0.85, needs_bnb=needs_bnb)
                self.assertEqual(cmd[cmd.index("--model") + 1], model)
                self.assertEqual(
                    cmd[cmd.index("--gpu-memory-utilization") + 1], "0.85"
                )
                if needs_bnb:
                    self.assertEqual(
                        cmd[cmd.index("--quantization") + 1], "bitsandbytes"
                    )
                    self.assertEqual(
                        cmd[cmd.index("--load-format") + 1], "bitsandbytes"
                    )
                else:
                    self.assertNotIn("--quantization", cmd)
                    self.assertNotIn("--load-format", cmd)

    def test_build_container_command_emits_all_optional_model_and_protocol_flags(
        self,
    ) -> None:
        from src.backend.vllm import LOCAL_ADAPTER_CONTAINER_PATH, _build_container_command

        template = "{{ messages[0]['content'] }}"
        cmd = _build_container_command(
            "Qwen/Qwen3-30B-A3B-AWQ",
            0.85,
            chat_template=template,
            lora_adapter_path=LOCAL_ADAPTER_CONTAINER_PATH,
            served_model_name="nsl-qwen-v1",
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            needs_bnb=False,
        )

        self.assertEqual(cmd[cmd.index("--chat-template") + 1], template)
        self.assertIn("--enable-lora", cmd)
        self.assertEqual(
            cmd[cmd.index("--lora-modules") + 1],
            f"adapter={LOCAL_ADAPTER_CONTAINER_PATH}",
        )
        self.assertEqual(cmd[cmd.index("--max-lora-rank") + 1], "64")
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "nsl-qwen-v1")
        self.assertIn("--enable-auto-tool-choice", cmd)
        self.assertEqual(cmd[cmd.index("--tool-call-parser") + 1], "hermes")
        self.assertEqual(cmd[cmd.index("--reasoning-parser") + 1], "qwen3")

    def test_build_container_command_omits_optional_model_and_protocol_flags_by_default(
        self,
    ) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command("Qwen/Qwen3-30B-A3B-AWQ", 0.85, needs_bnb=False)
        for flag in (
            "--chat-template",
            "--enable-lora",
            "--lora-modules",
            "--served-model-name",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "--reasoning-parser",
        ):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, cmd)

    def test_resolve_model_for_container_local_path(self) -> None:
        from src.backend.vllm import (
            LOCAL_MODEL_CONTAINER_PATH,
            _resolve_model_for_container,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_arg, host_mount, needs_bnb = _resolve_model_for_container(
                "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                tmpdir,
            )

            self.assertEqual(model_arg, LOCAL_MODEL_CONTAINER_PATH)
            self.assertEqual(host_mount, str(Path(tmpdir).resolve()))
            self.assertFalse(needs_bnb)

    def test_resolve_model_for_container_hf_model(self) -> None:
        from src.backend.vllm import _resolve_model_for_container

        model_arg, host_mount, needs_bnb = _resolve_model_for_container(
            "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
            None,
        )

        self.assertEqual(model_arg, "unsloth/Qwen2.5-Coder-7B-bnb-4bit")
        self.assertIsNone(host_mount)
        self.assertTrue(needs_bnb)

    def test_build_container_command_emits_all_tuning_flags_with_exact_values(
        self,
    ) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
            max_model_len=8192,
            max_num_seqs=10,
            max_num_batched_tokens=8192,
            enable_chunked_prefill=True,
            tensor_parallel_size=2,
            data_parallel_size=3,
            pipeline_parallel_size=4,
            kv_cache_dtype="fp8",
            disable_custom_all_reduce=True,
            enforce_eager=True,
        )

        expected_values = {
            "--max-model-len": "8192",
            "--max-num-seqs": "10",
            "--max-num-batched-tokens": "8192",
            "--tensor-parallel-size": "2",
            "--data-parallel-size": "3",
            "--pipeline-parallel-size": "4",
            "--kv-cache-dtype": "fp8",
        }
        for flag, value in expected_values.items():
            with self.subTest(flag=flag):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn("--enable-chunked-prefill", cmd)
        self.assertIn("--disable-custom-all-reduce", cmd)
        self.assertIn("--enforce-eager", cmd)

    def test_build_container_command_omits_tuning_flags_by_default(self) -> None:
        from src.backend.vllm import _build_container_command

        cmd = _build_container_command(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            0.85,
            needs_bnb=False,
        )

        self.assertNotIn("--max-model-len", cmd)
        self.assertNotIn("--max-num-seqs", cmd)
        self.assertNotIn("--max-num-batched-tokens", cmd)
        self.assertNotIn("--enable-chunked-prefill", cmd)
        self.assertNotIn("--tensor-parallel-size", cmd)
        self.assertNotIn("--data-parallel-size", cmd)
        self.assertNotIn("--pipeline-parallel-size", cmd)
        self.assertNotIn("--kv-cache-dtype", cmd)
        self.assertNotIn("--disable-custom-all-reduce", cmd)
        self.assertNotIn("--enforce-eager", cmd)

    def test_build_container_command_emits_exact_merged_compilation_configuration(
        self,
    ) -> None:
        from src.backend.vllm import _build_container_command

        overrides = {
            "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
            }
        }
        cases = (
            ("default", {}, overrides),
            (
                "cudagraph",
                {"cudagraph_mode": "piecewise"},
                {"cudagraph_mode": "PIECEWISE", **overrides},
            ),
            (
                "cache",
                {"compile_cache_dir": "/tmp/vllm_compile_cache"},
                {"cache_dir": "/tmp/vllm_compile_cache", **overrides},
            ),
            (
                "merged",
                {
                    "cudagraph_mode": "piecewise",
                    "compile_cache_dir": "/tmp/vllm_compile_cache",
                },
                {
                    "cache_dir": "/tmp/vllm_compile_cache",
                    "cudagraph_mode": "PIECEWISE",
                    **overrides,
                },
            ),
        )
        for name, kwargs, expected in cases:
            with self.subTest(name=name):
                cmd = _build_container_command(
                    "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                    0.85,
                    needs_bnb=False,
                    **kwargs,
                )
                self.assertEqual(cmd.count("--compilation-config"), 1)
                payload = cmd[cmd.index("--compilation-config") + 1]
                self.assertEqual(json.loads(payload), expected)

    def test_compile_cache_namespace_is_deterministic_path_safe_and_discriminating(
        self,
    ) -> None:
        from src.backend.vllm import _compile_cache_namespace

        baseline_kwargs = dict(
            model="QuantTrio/gemma-4-31B-it-AWQ",
            tp=2,
            pp=2,
            kv_cache_dtype="auto",
            max_model_len=32768,
            cudagraph_mode="piecewise",
            enforce_eager=False,
            lora_enabled=False,
        )
        gemma = _compile_cache_namespace(**baseline_kwargs)
        gemma_again = _compile_cache_namespace(**baseline_kwargs)
        qwen = _compile_cache_namespace(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            tp=2,
            pp=2,
            kv_cache_dtype="auto",
            max_model_len=32768,
            cudagraph_mode="piecewise",
            enforce_eager=False,
            lora_enabled=False,
        )

        self.assertNotEqual(gemma, qwen)
        self.assertEqual(gemma, gemma_again)
        self.assertIn("gemma-4-31B-it-AWQ", gemma.split("/", 1)[0])
        self.assertIn("Qwen2.5-Coder-7B-Instruct", qwen.split("/", 1)[0])
        self.assertEqual(gemma.count("/"), 1)
        self.assertEqual(qwen.count("/"), 1)

        variants = {
            "tensor-parallel": {"tp": 1},
            "pipeline-parallel": {"pp": 1},
            "max-model-length": {"max_model_len": 8192},
            "cudagraph-mode": {"cudagraph_mode": "full"},
            "eager": {"enforce_eager": True},
            "lora": {"lora_enabled": True},
            "kv-cache-dtype": {"kv_cache_dtype": "fp8"},
        }
        for name, override in variants.items():
            with self.subTest(name=name):
                kwargs = {**baseline_kwargs, **override}
                variant = _compile_cache_namespace(**kwargs)
                self.assertNotEqual(variant, gemma)
                if name in {"cudagraph-mode", "eager", "lora", "kv-cache-dtype"}:
                    self.assertEqual(variant.split("/")[0], gemma.split("/")[0])


class TestVllmContainerLifecycle(unittest.TestCase):
    """Tests that setup_vllm correctly manages the Docker container lifecycle."""

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm.is_http_ready", return_value=True)
    def test_reuses_existing_server(
        self,
        mock_http: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """When a server is already running, no container should be created."""
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config) as session:
            self.assertIsNotNone(session.genner)
            self.assertIsNone(session.process)
            self.assertIn("endpoint_pool", session.extras)
            pool = session.extras["endpoint_pool"]
            self.assertEqual(pool.endpoint_ids(), ["vllm-0"])

    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm.is_http_ready", return_value=True)
    def test_invalid_local_adapter_is_not_hidden_by_ready_server(
        self,
        _mock_http: MagicMock,
        mock_openai_cls: MagicMock,
    ) -> None:
        from src.backend.vllm import setup_vllm

        config = make_app_config(
            "vllm:Qwen/Qwen2.5-7B-Instruct",
            lora_adapter_path="/missing/adapter",
        )

        with self.assertRaises(FileNotFoundError):
            with setup_vllm(config):
                pass

        mock_openai_cls.assert_not_called()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._is_http_ready_with_auth", return_value=True)
    def test_external_recursive_endpoint_requires_matching_lora_card(
        self,
        _mock_ready: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from types import SimpleNamespace

        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        client = mock_openai_cls.return_value
        client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="adapter", parent=None)]
        )
        config = make_app_config(
            "vllm:Qwen/Qwen2.5-7B-Instruct",
            lora_adapter_path="/local/trained/adapter",
        )
        config.vllm.endpoints = [
            AppConfig.VllmConfig.EndpointConfig(base_url="https://external.example")
        ]

        with self.assertRaisesRegex(RuntimeError, "LoRA attestation"):
            with setup_vllm(config):
                pass

        client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="adapter", parent="base")]
        )
        with setup_vllm(config) as session:
            pool = session.extras["endpoint_pool"]
            metadata = pool.endpoint_metadata("vllm-0")
            self.assertTrue(metadata["lora_attested"])
            pool.mark_unhealthy("vllm-0", "transient outage")
            pool._recover_unhealthy_once()
            self.assertTrue(pool.is_healthy("vllm-0"))

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm.is_http_ready", return_value=True)
    def test_single_external_endpoint_requires_matching_lora_card(
        self,
        _mock_ready: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from types import SimpleNamespace

        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        client = mock_openai_cls.return_value
        with tempfile.TemporaryDirectory() as adapter_dir:
            write_mock_lora_adapter(adapter_dir)
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-7B-Instruct",
                lora_adapter_path=adapter_dir,
            )
            config.vllm.endpoint = "https://external.example"
            client.models.list.return_value = SimpleNamespace(
                data=[SimpleNamespace(id="adapter", parent=None)]
            )

            with self.assertRaisesRegex(RuntimeError, "LoRA attestation"):
                with setup_vllm(config):
                    pass

            client.models.list.return_value = SimpleNamespace(
                data=[SimpleNamespace(id="adapter", parent="base")]
            )
            with setup_vllm(config) as session:
                metadata = session.extras["endpoint_pool"].endpoint_metadata(
                    "vllm-0"
                )
                self.assertTrue(metadata["lora_attested"])

    @patch("src.backend.vllm._attach_endpoint_pool")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm.is_http_ready", return_value=True)
    def test_existing_server_monitor_stops_when_context_body_raises(
        self,
        mock_http: MagicMock,
        mock_openai_cls: MagicMock,
        attach_pool: MagicMock,
    ) -> None:
        from src.backend.vllm import setup_vllm

        pool = MagicMock()
        session = MagicMock()
        session.extras = {"endpoint_pool": pool}
        attach_pool.return_value = session
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with self.assertRaisesRegex(RuntimeError, "context failed"):
            with setup_vllm(config):
                raise RuntimeError("context failed")

        pool.start_health_monitor.assert_called_once_with()
        pool.stop_health_monitor.assert_called_once_with()

    @patch("src.backend.vllm._attach_endpoint_pool")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm.is_http_ready", return_value=True)
    def test_existing_server_monitor_stops_when_start_raises(
        self,
        mock_http: MagicMock,
        mock_openai_cls: MagicMock,
        attach_pool: MagicMock,
    ) -> None:
        from src.backend.vllm import setup_vllm

        pool = MagicMock()
        pool.start_health_monitor.side_effect = RuntimeError("monitor start failed")
        session = MagicMock()
        session.extras = {"endpoint_pool": pool}
        attach_pool.return_value = session
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with self.assertRaisesRegex(RuntimeError, "monitor start failed"):
            with setup_vllm(config):
                pass

        pool.stop_health_monitor.assert_called_once_with()

    @patch.dict(os.environ, {"VLLM_EP0_KEY": "secret-0"}, clear=False)
    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._is_http_ready_with_auth", return_value=True)
    def test_configured_external_endpoints_build_endpoint_pool(
        self,
        mock_ready: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from src.backend.vllm import setup_vllm

        fake0 = MagicMock(spec=Genner)
        fake0.identifier = "vllm"
        fake1 = MagicMock(spec=Genner)
        fake1.identifier = "vllm"
        mock_get_genner.side_effect = [fake0, fake1]
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")
        config.vllm = AppConfig.VllmConfig(
            endpoints=[
                {
                    "id": "local",
                    "base_url": "http://127.0.0.1:8000",
                    "capacity": 3,
                    "api_key_env": "VLLM_EP0_KEY",
                },
                {
                    "id": "remote",
                    "base_url": "https://remote.example/v1",
                    "capacity": 5,
                },
            ]
        )

        with setup_vllm(config) as session:
            pool = session.extras["endpoint_pool"]
            self.assertEqual(pool.endpoint_ids(), ["local", "remote"])
            self.assertEqual(pool.healthy_capacity(), 8)
            self.assertEqual(session.extras["metrics_api_key"], "secret-0")
            self.assertIsNotNone(pool._health_monitor_thread)
            self.assertTrue(pool._health_monitor_thread.is_alive())

        self.assertIsNone(pool._health_monitor_thread)

        self.assertEqual(mock_openai_cls.call_count, 2)
        self.assertEqual(
            mock_openai_cls.call_args_list[0].kwargs["api_key"], "secret-0"
        )
        self.assertEqual(mock_openai_cls.call_args_list[1].kwargs["api_key"], "dummy")
        mock_ready.assert_any_call(
            "http://127.0.0.1:8000/v1/models",
            api_key="secret-0",
        )

    @patch("src.backend.vllm._build_external_endpoint_pool_session")
    def test_external_endpoint_monitor_stops_when_context_body_raises(
        self,
        build_session: MagicMock,
    ) -> None:
        from src.backend.vllm import setup_vllm

        pool = MagicMock()
        session = MagicMock()
        session.extras = {"endpoint_pool": pool}
        build_session.return_value = session
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")
        config.vllm = AppConfig.VllmConfig(
            endpoints=[{"base_url": "https://external.example"}]
        )

        with self.assertRaisesRegex(RuntimeError, "context failed"):
            with setup_vllm(config):
                raise RuntimeError("context failed")

        pool.start_health_monitor.assert_called_once_with()
        pool.stop_health_monitor.assert_called_once_with()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_wildcard_endpoint_creates_and_removes_managed_container(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """When no server exists, a Docker container should be created and removed on exit."""
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_container.id = "abc123"
        mock_docker = MagicMock()
        # First get() is stale check (no stale), second is post-creation
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config(
            "vllm:Qwen/Qwen2.5-7B-Instruct", cdi_devices=["0", "1"]
        )
        assert config.vllm is not None
        config.vllm.endpoint = "http://0.0.0.0:8000"

        with patch.dict(os.environ, {"NSL_VLLM_NETWORK_MODE": "auto"}, clear=False):
            with setup_vllm(config) as session:
                self.assertIsNotNone(session.genner)
                mock_start.assert_called_once()

        self.assertEqual(mock_start.call_args.kwargs["cdi_devices"], ["0", "1"])

        wait_urls = mock_wait.call_args.args[0]
        self.assertEqual(
            wait_urls,
            [
                "http://0.0.0.0:8000/v1/models",
                "http://10.0.0.8:8000/v1/models",
            ],
        )
        self.assertEqual(mock_openai_cls.call_count, 1)

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("src.backend.vllm._attach_endpoint_pool")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_managed_container_stops_monitor_before_teardown_on_exception(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        attach_pool: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        cleanup_events: list[str] = []
        pool = MagicMock()
        pool.stop_health_monitor.side_effect = lambda: cleanup_events.append(
            "monitor.stop"
        )
        session = MagicMock()
        session.extras = {"endpoint_pool": pool}
        attach_pool.return_value = session

        container = MagicMock()
        container.name = "nsl-vllm-8000"
        container.stop.side_effect = lambda **kwargs: cleanup_events.append(
            "container.stop"
        )
        container.remove.side_effect = lambda **kwargs: cleanup_events.append(
            "container.remove"
        )
        docker_client = MagicMock()
        docker_client.containers.get.side_effect = [NotFound("not found"), container]
        mock_docker_cls.from_env.return_value = docker_client
        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with self.assertRaisesRegex(RuntimeError, "context failed"):
            with setup_vllm(config):
                raise RuntimeError("context failed")

        pool.start_health_monitor.assert_called_once_with()
        pool.stop_health_monitor.assert_called_once_with()
        self.assertEqual(
            cleanup_events,
            ["monitor.stop", "container.stop", "container.remove"],
        )

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._get_host_ip", return_value="10.0.0.8")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_hostip_mode_uses_only_host_ip_endpoint(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_host_ip: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_container.id = "abc123"
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker
        mock_wait.return_value = "http://10.0.0.8:8000/v1/models"

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with patch.dict(os.environ, {"NSL_VLLM_NETWORK_MODE": "hostip"}, clear=True):
            with setup_vllm(config) as session:
                self.assertIsNotNone(session.genner)

        self.assertEqual(
            mock_wait.call_args.args[0], ["http://10.0.0.8:8000/v1/models"]
        )
        self.assertEqual(
            mock_openai_cls.call_args.kwargs["base_url"], "http://10.0.0.8:8000/v1"
        )
        self.assertEqual(
            mock_wait.call_args.kwargs["primary_timeout_s"],
            max(config.inference.timeout, 500),
        )

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_removes_stale_container_before_creating(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """If a stale container with the same name exists, it should be removed first."""
        from src.backend.vllm import setup_vllm

        mock_get_genner.return_value = MagicMock(spec=Genner)

        # First get() call is for stale removal, second is for getting the new container
        stale_container = MagicMock()
        stale_container.name = "nsl-vllm-8000"
        new_container = MagicMock()
        new_container.name = "nsl-vllm-8000"
        new_container.id = "new123"

        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [stale_container, new_container]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config):
            stale_container.remove.assert_called_once_with(force=True)
            mock_start.assert_called_once()

        new_container.stop.assert_called_once()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_setup_vllm_with_local_model_path(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        from src.backend.vllm import (
            ADAPTER_MODEL_NAME,
            COMPILE_CACHE_CONTAINER_PATH,
            DEFAULT_COMPILE_CACHE_HOST,
            LOCAL_ADAPTER_CONTAINER_PATH,
            LOCAL_MODEL_CONTAINER_PATH,
            setup_vllm,
        )

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        mock_docker_cls.from_env.return_value = mock_docker

        with (
            tempfile.TemporaryDirectory() as model_dir,
            tempfile.TemporaryDirectory() as adapter_dir,
        ):
            write_mock_lora_adapter(adapter_dir)
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                local_model_path=model_dir,
                lora_adapter_path=adapter_dir,
                served_model_name="nsl-qwen2.5-v1",
            )

            with setup_vllm(config) as session:
                self.assertEqual(session.config.model, ADAPTER_MODEL_NAME)

            start_call = mock_start.call_args
            vllm_command = start_call.args[2]
            extra_volumes = start_call.kwargs["extra_volumes"]

            self.assertIn(LOCAL_MODEL_CONTAINER_PATH, vllm_command)
            self.assertIn("--enable-lora", vllm_command)
            self.assertIn(f"adapter={LOCAL_ADAPTER_CONTAINER_PATH}", vllm_command)
            self.assertIn("--served-model-name", vllm_command)
            self.assertIn("nsl-qwen2.5-v1", vllm_command)
            self.assertNotIn("--quantization", vllm_command)
            self.assertIn(
                (str(Path(model_dir).resolve()), LOCAL_MODEL_CONTAINER_PATH),
                extra_volumes,
            )
            self.assertIn(
                (str(Path(adapter_dir).resolve()), LOCAL_ADAPTER_CONTAINER_PATH),
                extra_volumes,
            )
            self.assertIn(
                (
                    str(DEFAULT_COMPILE_CACHE_HOST.resolve()),
                    COMPILE_CACHE_CONTAINER_PATH,
                ),
                extra_volumes,
            )

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_cleanup_on_startup_failure(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        """If readiness check fails, container should still be cleaned up."""
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_wait.side_effect = TimeoutError("Timed out")
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = [
            NotFound("not found"),
            mock_container,
        ]
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with self.assertRaises(TimeoutError):
            with setup_vllm(config):
                pass

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_setup_vllm_uses_persistent_compile_cache_by_default(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        import json

        from src.backend.vllm import (
            COMPILE_CACHE_CONTAINER_PATH,
            DEFAULT_COMPILE_CACHE_HOST,
            _compile_cache_namespace,
            setup_vllm,
        )

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        mock_docker_cls.from_env.return_value = mock_docker

        config = make_app_config("vllm:Qwen/Qwen2.5-7B-Instruct")

        with setup_vllm(config):
            start_call = mock_start.call_args
            vllm_command = start_call.args[2]
            extra_volumes = start_call.kwargs["extra_volumes"]

        self.assertIn(
            (str(DEFAULT_COMPILE_CACHE_HOST.resolve()), COMPILE_CACHE_CONTAINER_PATH),
            extra_volumes,
        )
        payload = vllm_command[vllm_command.index("--compilation-config") + 1]
        expected_namespace = _compile_cache_namespace(
            model="Qwen/Qwen2.5-7B-Instruct",
            tp=None,
            max_model_len=None,
            cudagraph_mode=None,
            enforce_eager=False,
            lora_enabled=False,
        )
        expected_cache_dir = f"{COMPILE_CACHE_CONTAINER_PATH}/{expected_namespace}"
        self.assertEqual(
            json.loads(payload),
            {
                "cache_dir": expected_cache_dir,
                "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            },
        )
        # Per-model bucket must be human-scannable: the first path segment
        # must contain the (sanitized) model identifier.
        first_segment = expected_namespace.split("/", 1)[0]
        self.assertIn("Qwen", first_segment)
        self.assertIn("Qwen2.5-7B-Instruct", first_segment)
        # Namespace subdir must be created on the host so the container mount
        # sees it on first start.
        self.assertTrue((DEFAULT_COMPILE_CACHE_HOST / expected_namespace).exists())

    @patch("src.backend.vllm.get_genner")
    @patch("src.backend.vllm.OpenAI")
    @patch("src.backend.vllm._wait_for_vllm_ready")
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_setup_vllm_respects_custom_compile_cache_dir(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
        mock_wait: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_genner: MagicMock,
    ) -> None:
        import json

        from src.backend.vllm import (
            COMPILE_CACHE_CONTAINER_PATH,
            _compile_cache_namespace,
            setup_vllm,
        )

        mock_get_genner.return_value = MagicMock(spec=Genner)
        mock_container = MagicMock()
        mock_container.name = "nsl-vllm-8000"
        mock_docker = MagicMock()
        mock_docker.containers.get.return_value = mock_container
        mock_docker_cls.from_env.return_value = mock_docker

        with tempfile.TemporaryDirectory() as cache_dir:
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-7B-Instruct",
                compile_cache_dir=cache_dir,
            )

            with setup_vllm(config):
                start_call = mock_start.call_args
                vllm_command = start_call.args[2]
                extra_volumes = start_call.kwargs["extra_volumes"]

        self.assertIn(
            (str(Path(cache_dir).resolve()), COMPILE_CACHE_CONTAINER_PATH),
            extra_volumes,
        )
        payload = vllm_command[vllm_command.index("--compilation-config") + 1]
        expected_namespace = _compile_cache_namespace(
            model="Qwen/Qwen2.5-7B-Instruct",
            tp=None,
            max_model_len=None,
            cudagraph_mode=None,
            enforce_eager=False,
            lora_enabled=False,
        )
        expected_cache_dir = f"{COMPILE_CACHE_CONTAINER_PATH}/{expected_namespace}"
        self.assertEqual(
            json.loads(payload),
            {
                "cache_dir": expected_cache_dir,
                "inductor_compile_config": {
                    "combo_kernels": False,
                    "benchmark_combo_kernel": False,
                },
            },
        )


class TestStartVllmContainer(unittest.TestCase):
    @patch("src.backend.vllm.subprocess.run")
    def test_start_container_uses_default_cdi_device_and_ignores_cuda_visible_devices(
        self, mock_run: MagicMock
    ) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=0)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            _start_vllm_container("nsl-vllm-8000", 8000, ["--model", "test"], [])

        call_args = mock_run.call_args[0][0]
        self.assertIn("docker", call_args)
        self.assertIn("nvidia.com/gpu=all", call_args)
        self.assertNotIn("nvidia.com/gpu=7", call_args)
        self.assertIn("--ipc=host", call_args)
        self.assertIn("--model", call_args)

    @patch("src.backend.vllm.subprocess.run")
    def test_start_container_emits_each_explicit_cdi_device(
        self, mock_run: MagicMock
    ) -> None:
        from src.backend.vllm import _start_vllm_container

        for devices in (["0"], ["0", "1"]):
            with self.subTest(devices=devices):
                mock_run.reset_mock()
                mock_run.return_value = MagicMock(returncode=0)
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
                    _start_vllm_container(
                        "nsl-vllm-8000",
                        8000,
                        ["--model", "test"],
                        cdi_devices=devices,
                    )

                call_args = mock_run.call_args.args[0]
                device_values = [
                    call_args[index + 1]
                    for index, argument in enumerate(call_args)
                    if argument == "--device"
                ]
                self.assertEqual(
                    device_values, [f"nvidia.com/gpu={device}" for device in devices]
                )

    @patch("src.backend.vllm.subprocess.run")
    def test_start_container_forwards_volumes_and_environment(
        self, mock_run: MagicMock
    ) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=0)
        _start_vllm_container(
            "nsl-vllm-8000",
            8000,
            ["--model", "test"],
            [("/host/model", "/models/local"), ("/host/adapter", "/adapters/local")],
            extra_env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        )

        call_args = mock_run.call_args.args[0]
        self.assertIn("/host/model:/models/local", call_args)
        self.assertIn("/host/adapter:/adapters/local", call_args)
        self.assertIn("PYTORCH_ALLOC_CONF=expandable_segments:True", call_args)

    @patch("src.backend.vllm.subprocess.run")
    def test_raises_on_docker_failure(self, mock_run: MagicMock) -> None:
        from src.backend.vllm import _start_vllm_container

        mock_run.return_value = MagicMock(returncode=1, stderr="no space left")
        with self.assertRaises(RuntimeError):
            _start_vllm_container(
                "nsl-vllm-8000", 8000, ["--model", "test"], []
            )


class TestWaitForVllmReady(unittest.TestCase):
    @patch("src.backend.vllm.is_http_ready")
    @patch("src.backend.vllm.time")
    def test_returns_when_http_ready(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        mock_time.time.side_effect = [0, 0, 0, 1]
        mock_http.return_value = True
        mock_container = MagicMock()
        mock_container.status = "running"

        _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)

    @patch("src.backend.vllm.is_http_ready", return_value=False)
    @patch("src.backend.vllm.time")
    def test_raises_on_timeout(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        # time.time() is called multiple times per loop iteration.
        # Use a counter to return values that eventually exceed the deadline.
        call_count = 0

        def fake_time():
            nonlocal call_count
            call_count += 1
            return 0 if call_count == 1 else (30 if call_count < 5 else 91)

        mock_time.time.side_effect = fake_time
        mock_time.sleep = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.reload = MagicMock()
        mock_container.logs.return_value = b"loading..."

        with self.assertRaises(TimeoutError):
            _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)

    @patch("src.backend.vllm.is_http_ready", return_value=False)
    @patch("src.backend.vllm.time")
    def test_raises_on_container_exit(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        mock_time.time.side_effect = [0, 0, 0, 1]
        mock_time.sleep = MagicMock()
        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_container.reload = MagicMock()
        mock_container.logs.return_value = b"some error"

        with self.assertRaises(RuntimeError, msg="container exited"):
            _wait_for_vllm_ready("http://localhost:8000/v1/models", mock_container, 60)

    @patch("src.backend.vllm.is_http_ready")
    @patch("src.backend.vllm.time")
    def test_tries_multiple_alternate_urls_after_primary_timeout(
        self, mock_time: MagicMock, mock_http: MagicMock
    ) -> None:
        from src.backend.vllm import _wait_for_vllm_ready

        urls = [
            "http://127.0.0.1:8000/v1/models",
            "http://10.0.0.8:8000/v1/models",
            "http://10.0.0.9:8000/v1/models",
        ]
        mock_time.time.side_effect = [0, 0, 0, 1, 1, 11, 11, 11, 61]
        mock_time.sleep = MagicMock()
        mock_http.side_effect = lambda url: url == urls[2]
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.reload = MagicMock()

        ready_url = _wait_for_vllm_ready(urls, mock_container, 60, primary_timeout_s=10)

        self.assertEqual(ready_url, urls[2])
        self.assertEqual(
            [call.args[0] for call in mock_http.call_args_list],
            [urls[0], urls[1], urls[2]],
        )


class TestWaitForGpuMemoryRelease(unittest.TestCase):
    @patch("src.backend.vllm._read_gpu_memory_info_mb")
    @patch("src.backend.vllm.time")
    def test_wait_for_gpu_memory_release_returns_when_threshold_met(
        self,
        mock_time: MagicMock,
        mock_read_gpu_memory_info_mb: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.side_effect = [0, 0, 2]
        mock_time.sleep = MagicMock()
        mock_read_gpu_memory_info_mb.side_effect = [
            (2000.0, 12000.0),
            (11000.0, 12000.0),
        ]

        wait_for_gpu_memory_release(
            min_free_memory_fraction=0.9, timeout_s=60, device_indices=[0]
        )

        self.assertEqual(mock_read_gpu_memory_info_mb.call_count, 2)
        mock_time.sleep.assert_called_once_with(2.0)

    @patch("src.backend.vllm._read_gpu_memory_info_mb", return_value=(2000.0, 12000.0))
    @patch("src.backend.vllm.time")
    def test_wait_for_gpu_memory_release_times_out_when_threshold_not_met(
        self,
        mock_time: MagicMock,
        _mock_read_gpu_memory_info_mb: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.side_effect = [0, 0, 61]
        mock_time.sleep = MagicMock()

        with self.assertRaises(TimeoutError):
            wait_for_gpu_memory_release(
                min_free_memory_fraction=0.9, timeout_s=60, device_indices=[0]
            )

        mock_time.sleep.assert_called_once_with(2.0)

    def test_gpu_memory_gate_early_exit_conditions(self) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        cases = (("probe-unavailable", [0], None), ("no-visible-gpus", None, []))
        for name, device_indices, visible_devices in cases:
            with self.subTest(name=name):
                with (
                    patch("src.backend.vllm.time") as mock_time,
                    patch(
                        "src.backend.vllm._read_gpu_memory_info_mb",
                        return_value=None,
                    ) as mock_read,
                    patch(
                        "src.backend.vllm.list_visible_gpu_indices",
                        return_value=visible_devices,
                    ),
                ):
                    mock_time.monotonic.return_value = 0
                    wait_for_gpu_memory_release(
                        min_free_memory_fraction=0.9,
                        timeout_s=60,
                        device_indices=device_indices,
                    )

                mock_time.sleep.assert_not_called()
                if visible_devices == []:
                    mock_read.assert_not_called()

    @patch("src.backend.vllm._read_gpu_memory_info_mb")
    @patch("src.backend.vllm.time")
    def test_gate_passes_only_when_all_devices_meet_threshold(
        self,
        mock_time: MagicMock,
        mock_read: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.side_effect = [0, 0, 2, 4]
        mock_time.sleep = MagicMock()
        # Round 1: device 0 free, device 1 still busy -> gate keeps waiting.
        # Round 2: both free -> gate passes.
        mock_read.side_effect = [
            (11000.0, 12000.0),
            (2000.0, 12000.0),
            (11000.0, 12000.0),
            (11000.0, 12000.0),
        ]

        wait_for_gpu_memory_release(
            min_free_memory_fraction=0.9, timeout_s=60, device_indices=[0, 1]
        )

        self.assertEqual(mock_read.call_count, 4)
        self.assertEqual(
            [
                c.kwargs.get("device_index", c.args[0] if c.args else None)
                for c in mock_read.call_args_list
            ],
            [0, 1, 0, 1],
        )

    @patch("src.backend.vllm.list_visible_gpu_indices", return_value=[0, 1])
    @patch("src.backend.vllm._read_gpu_memory_info_mb")
    @patch("src.backend.vllm.time")
    def test_gate_auto_enumerates_when_device_indices_none(
        self,
        mock_time: MagicMock,
        mock_read: MagicMock,
        _mock_list: MagicMock,
    ) -> None:
        from src.backend.vllm import wait_for_gpu_memory_release

        mock_time.monotonic.side_effect = [0, 0]
        mock_time.sleep = MagicMock()
        mock_read.return_value = (11000.0, 12000.0)

        wait_for_gpu_memory_release(min_free_memory_fraction=0.9, timeout_s=60)

        self.assertEqual(mock_read.call_count, 2)


class TestResolveModelForContainer(unittest.TestCase):
    def test_hf_awq_model_no_bnb_flags(self) -> None:
        from src.backend.vllm import _resolve_model_for_container

        model_arg, host_path, needs_bnb = _resolve_model_for_container(
            "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", None
        )
        self.assertEqual(model_arg, "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
        self.assertIsNone(host_path)
        self.assertFalse(needs_bnb)


class TestSetupVllmWithLocalModel(unittest.TestCase):
    @patch("src.backend.vllm._start_vllm_container")
    @patch("src.backend.vllm.DockerClient")
    @patch("src.backend.vllm.is_http_ready", return_value=False)
    def test_invalid_lora_adapter_path_fails_before_startup(
        self,
        mock_http: MagicMock,
        mock_docker_cls: MagicMock,
        mock_start: MagicMock,
    ) -> None:
        from docker.errors import NotFound
        from src.backend.vllm import setup_vllm

        mock_docker = MagicMock()
        mock_docker.containers.get.side_effect = NotFound("")
        mock_docker_cls.from_env.return_value = mock_docker

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_app_config(
                "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                lora_adapter_path=tmpdir,
            )

            with self.assertRaisesRegex(ValueError, "adapter_config.json"):
                with setup_vllm(config):
                    pass

        mock_start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
