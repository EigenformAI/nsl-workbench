import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests
from docker import DockerClient
from docker.errors import NotFound
from loguru import logger
from openai import OpenAI

from src.backend._container_runtime import (
    build_models_urls,
    compile_cache_namespace,
    resolve_host_ip,
    resolve_network_mode,
    validate_lora_adapter_path,
    wait_for_http_ready,
)
from src.backend._tool_parsers import infer_tool_call_parser
from src.backend.utils import is_http_ready
from src.genner import get_genner
from src.genner.config import SglangServerConfig
from src.typing.config import AppConfig

from .session import BackendSession


SGLANG_IMAGE = "lmsysorg/sglang:v0.5.10.post1"
CONTAINER_NAME_PREFIX = "nsl-sglang"
SGLANG_NETWORK_MODE_ENV = "NSL_SGLANG_NETWORK_MODE"

HF_CACHE_HOST = Path.home() / ".cache" / "huggingface"
HF_CACHE_CONTAINER = "/root/.cache/huggingface"
HOME_CONTAINER = "/root"

DEFAULT_SGLANG_COMPILE_CACHE_HOST = Path.home() / ".cache" / "sglang_compile_cache"
DEFAULT_INDUCTOR_CACHE_HOST = Path.home() / ".cache" / "torch_inductor_cache"
SGLANG_CACHE_CONTAINER_ROOT = "/root/.cache/sglang"
INDUCTOR_CACHE_CONTAINER_PATH = "/root/.cache/torch/inductor"

LOCAL_MODEL_CONTAINER_PATH = "/models/local"
LOCAL_CHAT_TEMPLATE_CONTAINER_PATH = "/templates/chat.jinja"
ADAPTERS_CONTAINER_ROOT = "/adapters"
SGLANG_LOG_DIR = Path("logs/sglang")


def _source_model_name(app_config: AppConfig) -> str:
    if app_config.model_name.startswith("sglang:"):
        return app_config.model_name.split(":", 1)[1].strip()
    return app_config.model_name.strip()


def _build_sglang_config(app_config: AppConfig, endpoint: str) -> SglangServerConfig:
    cfg = app_config.sglang
    source_model = _source_model_name(app_config)
    served_model_name = cfg.served_model_name if cfg and cfg.served_model_name else source_model

    config = SglangServerConfig()
    config.model = served_model_name
    config.endpoint = endpoint
    config.timeout = app_config.inference.timeout
    config.temperature = app_config.inference.temperature
    config.max_tokens = app_config.inference.max_tokens
    config.frequency_penalty = app_config.inference.frequency_penalty
    config.presence_penalty = app_config.inference.presence_penalty
    if cfg is not None:
        config.mem_fraction_static = cfg.mem_fraction_static or config.mem_fraction_static
        if cfg.chat_template_path:
            config.chat_template = Path(cfg.chat_template_path).read_text(encoding="utf-8")
        config.tool_call_parser = cfg.tool_call_parser or infer_tool_call_parser(
            "sglang",
            source_model,
        )
        config.reasoning_parser = cfg.reasoning_parser
        config.lora_routing_enabled = cfg.lora_routing_enabled
        config.default_lora_name = cfg.default_lora_name
    else:
        config.tool_call_parser = infer_tool_call_parser("sglang", source_model)
    return config


def _build_sglang_container_command(
    model: str,
    *,
    port: int,
    served_model_name: str,
    mem_fraction_static: float = 0.85,
    chat_template: str | None = None,
    lora_paths: dict[str, str] | None = None,
    pinned_lora_names: list[str] | None = None,
    max_loras_per_batch: int | None = None,
    max_loaded_loras: int | None = None,
    max_lora_rank: int | None = None,
    lora_target_modules: list[str] | None = None,
    lora_backend: str | None = None,
    enable_lora_overlap_loading: bool = False,
    lora_eviction_policy: str | None = None,
    quantization: str | None = None,
    fp8_gemm_backend: str | None = None,
    attention_backend: str | None = None,
    tensor_parallel_size: int | None = None,
    data_parallel_size: int | None = None,
    max_model_len: int | None = None,
    max_running_requests: int | None = None,
    kv_cache_dtype: str | None = None,
    disable_radix_cache: bool = False,
    radix_eviction_policy: str | None = None,
    disable_cuda_graph: bool = False,
    cuda_graph_max_bs: int | None = None,
    cuda_graph_bs: list[int] | None = None,
    enable_torch_compile: bool = False,
    torch_compile_max_bs: int | None = None,
    enable_chunked_prefill: bool = True,
    enable_mixed_chunk: bool = False,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    grammar_backend: str = "xgrammar",
    speculative_algorithm: str | None = None,
    speculative_draft_model_path: str | None = None,
    speculative_num_steps: int | None = None,
    speculative_eagle_topk: int | None = None,
    speculative_num_draft_tokens: int | None = None,
) -> list[str]:
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--mem-fraction-static",
        str(mem_fraction_static),
    ]

    if kv_cache_dtype:
        cmd.extend(["--kv-cache-dtype", kv_cache_dtype])
    if tensor_parallel_size is not None:
        cmd.extend(["--tp", str(tensor_parallel_size)])
    if data_parallel_size is not None:
        cmd.extend(["--dp", str(data_parallel_size)])
    if max_model_len is not None:
        cmd.extend(["--context-length", str(max_model_len)])
    if max_running_requests is not None:
        cmd.extend(["--max-running-requests", str(max_running_requests)])
    if quantization:
        cmd.extend(["--quantization", quantization])
    if fp8_gemm_backend:
        cmd.extend(["--fp8-gemm-backend", fp8_gemm_backend])
    if attention_backend:
        cmd.extend(["--attention-backend", attention_backend])
    if chat_template:
        cmd.extend(["--chat-template", chat_template])
    if not enable_chunked_prefill:
        cmd.extend(["--chunked-prefill-size", "-1"])
    if enable_mixed_chunk:
        cmd.append("--enable-mixed-chunk")
    if disable_radix_cache:
        cmd.append("--disable-radix-cache")
    if radix_eviction_policy:
        cmd.extend(["--radix-eviction-policy", radix_eviction_policy])
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if cuda_graph_max_bs is not None:
        cmd.extend(["--cuda-graph-max-bs", str(cuda_graph_max_bs)])
    if cuda_graph_bs:
        cmd.append("--cuda-graph-bs")
        cmd.extend(str(bs) for bs in cuda_graph_bs)
    if enable_torch_compile:
        cmd.append("--enable-torch-compile")
    if torch_compile_max_bs is not None:
        cmd.extend(["--torch-compile-max-bs", str(torch_compile_max_bs)])

    cmd.extend(["--grammar-backend", grammar_backend])
    if tool_call_parser:
        cmd.extend(["--tool-call-parser", tool_call_parser])
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])

    if lora_paths:
        cmd.append("--lora-paths")
        cmd.extend(f"{name}={path}" for name, path in lora_paths.items())
    if pinned_lora_names:
        cmd.append("--pinned-lora-names")
        cmd.extend(pinned_lora_names)
    if max_loras_per_batch is not None:
        cmd.extend(["--max-loras-per-batch", str(max_loras_per_batch)])
    if max_loaded_loras is not None:
        cmd.extend(["--max-loaded-loras", str(max_loaded_loras)])
    if max_lora_rank is not None:
        cmd.extend(["--max-lora-rank", str(max_lora_rank)])
    if lora_target_modules:
        cmd.append("--lora-target-modules")
        cmd.extend(lora_target_modules)
    if lora_backend:
        cmd.extend(["--lora-backend", lora_backend])
    if enable_lora_overlap_loading:
        cmd.append("--enable-lora-overlap-loading")
    if lora_eviction_policy:
        cmd.extend(["--lora-eviction-policy", lora_eviction_policy])

    if speculative_algorithm:
        cmd.extend(["--speculative-algorithm", speculative_algorithm])
    if speculative_draft_model_path:
        cmd.extend(["--speculative-draft-model-path", speculative_draft_model_path])
    if speculative_num_steps is not None:
        cmd.extend(["--speculative-num-steps", str(speculative_num_steps)])
    if speculative_eagle_topk is not None:
        cmd.extend(["--speculative-eagle-topk", str(speculative_eagle_topk)])
    if speculative_num_draft_tokens is not None:
        cmd.extend(["--speculative-num-draft-tokens", str(speculative_num_draft_tokens)])

    return cmd


def _container_name(port: int) -> str:
    return f"{CONTAINER_NAME_PREFIX}-{port}"


def _remove_stale_container(docker_client: DockerClient, name: str) -> None:
    try:
        stale = docker_client.containers.get(name)
        logger.info(f"Removing stale container '{name}'...")
        stale.remove(force=True)
    except NotFound:
        pass


def _start_sglang_container(
    name: str,
    port: int,
    sglang_command: list[str],
    extra_volumes: list[tuple[str, str]] | None = None,
    extra_env: dict[str, str] | None = None,
    image: str | None = None,
) -> None:
    del port  # The server binds the configured port inside the command itself.
    HF_CACHE_HOST.mkdir(parents=True, exist_ok=True)

    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--device",
        "nvidia.com/gpu=all",
        "--ipc=host",
        "--network=host",
        "-e",
        f"HF_HOME={HF_CACHE_CONTAINER}",
        "-e",
        f"HOME={HOME_CONTAINER}",
        "-v",
        f"{HF_CACHE_HOST}:{HF_CACHE_CONTAINER}",
    ]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        docker_cmd.extend(
            [
                "-e",
                f"HF_TOKEN={hf_token}",
                "-e",
                f"HUGGING_FACE_HUB_TOKEN={hf_token}",
            ]
        )
    for env_name, env_value in (extra_env or {}).items():
        docker_cmd.extend(["-e", f"{env_name}={env_value}"])
    for host_path, container_path in extra_volumes or []:
        docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
    docker_cmd.extend([image or SGLANG_IMAGE, *sglang_command])

    logger.info(f"Docker command: {' '.join(docker_cmd)}")
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start SGLang container: {result.stderr.strip()}")


def _wait_for_sglang_ready(
    models_urls: str | list[str],
    container,
    timeout_s: int,
    primary_timeout_s: int = 60,
) -> str:
    return wait_for_http_ready(
        models_urls,
        container,
        timeout_s,
        backend_name="SGLang",
        log_dir=SGLANG_LOG_DIR,
        primary_timeout_s=primary_timeout_s,
    )


def _build_sglang_smoke_test(client: OpenAI, config: SglangServerConfig):
    def _liveness_text(message: Any) -> str | None:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return "tool_call"
        return None

    def run_smoke_test() -> str:
        if config.tool_call_parser:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {
                        "role": "user",
                        "content": "Use the ping tool if tools are available.",
                    }
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "ping",
                            "description": "Return pong.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                tool_choice="auto",
                max_tokens=64,
                temperature=0.0,
            )
        else:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": "Who are you?"}],
                max_tokens=64,
                temperature=0.0,
            )
        content = _liveness_text(response.choices[0].message)
        if content is None:
            raise RuntimeError("SGLang smoke test returned a non-text response")
        return content

    return run_smoke_test


@dataclass
class SglangSessionExtras:
    base_url: str
    timeout: float = 10.0

    def load_lora(self, name: str, path: str, pinned: bool = False) -> Any:
        return self._post(
            "load_lora_adapter",
            {"lora_name": name, "lora_path": path, "pinned": pinned},
        )

    def unload_lora(self, name: str) -> Any:
        return self._post("unload_lora_adapter", {"lora_name": name})

    def update_weights_from_disk(self, path: str) -> Any:
        return self._post("update_weights_from_disk", {"model_path": path})

    def _post(self, endpoint: str, payload: dict[str, Any]) -> Any:
        response = requests.post(
            f"{self._server_root()}/{endpoint}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return response.text

    def _server_root(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1")


def _build_sglang_session(
    client: OpenAI,
    config: SglangServerConfig,
    base_url: str,
    models_url: str,
    process=None,
) -> BackendSession:
    genner = get_genner("sglang", server_config=config, oai_client=client)
    server_root = base_url.rsplit("/v1", 1)[0]
    return BackendSession(
        genner=genner,
        smoke_test=_build_sglang_smoke_test(client, config),
        client=client,
        config=config,
        base_url=base_url,
        models_url=models_url,
        process=process,
        metrics_url=f"{server_root}/metrics",
        extras=SglangSessionExtras(base_url, timeout=config.timeout),
    )


@contextmanager
def setup_sglang(
    app_config: AppConfig,
    *,
    endpoint: str = "http://127.0.0.1:30000",
) -> Iterator[BackendSession]:
    sglang_cfg = app_config.sglang or AppConfig.SglangConfig()
    startup_timeout = sglang_cfg.startup_timeout or 500
    config = _build_sglang_config(app_config, endpoint=endpoint)
    source_model = _source_model_name(app_config)

    endpoint = config.endpoint.rstrip("/")
    base_url = endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"
    models_url = f"{base_url}/models"
    api_key = config.api_key or "dummy"

    if is_http_ready(models_url):
        logger.info(f"Using existing SGLang server at {base_url}")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)
        yield _build_sglang_session(client, config, base_url, models_url)
        return

    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    scheme = parsed.scheme or "http"
    port = parsed.port or 30000
    network_mode = resolve_network_mode(SGLANG_NETWORK_MODE_ENV)
    host_ip = None if network_mode == "loopback" else resolve_host_ip()

    logger.info(
        f"No server detected at {base_url}. Starting SGLang Docker container "
        f"for {source_model} on port {port}..."
    )

    image = sglang_cfg.image or SGLANG_IMAGE
    effective_model = source_model
    model_host_path: str | None = None
    if sglang_cfg.local_model_path:
        effective_model = LOCAL_MODEL_CONTAINER_PATH
        model_host_path = str(Path(sglang_cfg.local_model_path).expanduser().resolve())

    resolved_lora_adapters = {
        name: validate_lora_adapter_path(path)
        for name, path in sglang_cfg.lora_adapters.items()
    }
    lora_container_paths = {
        name: f"{ADAPTERS_CONTAINER_ROOT}/{name}"
        for name in resolved_lora_adapters.keys()
    }

    docker_client = DockerClient.from_env()
    container_name = _container_name(port)
    _remove_stale_container(docker_client, container_name)

    extra_volumes: list[tuple[str, str]] = []
    if model_host_path:
        extra_volumes.append((model_host_path, LOCAL_MODEL_CONTAINER_PATH))
    for name, host_path in resolved_lora_adapters.items():
        extra_volumes.append((host_path, f"{ADAPTERS_CONTAINER_ROOT}/{name}"))
    if sglang_cfg.chat_template_path:
        extra_volumes.append(
            (
                str(Path(sglang_cfg.chat_template_path).expanduser().resolve()),
                LOCAL_CHAT_TEMPLATE_CONTAINER_PATH,
            )
        )

    cache_namespace = compile_cache_namespace(
        engine="sglang",
        model=effective_model,
        tp=sglang_cfg.tensor_parallel_size,
        max_model_len=sglang_cfg.max_model_len,
        image=image,
        quantization=sglang_cfg.quantization,
        cuda_graph_bs=sglang_cfg.cuda_graph_bs,
        enable_torch_compile=sglang_cfg.enable_torch_compile,
        lora_enabled=bool(resolved_lora_adapters),
        attention_backend=sglang_cfg.attention_backend,
        fp8_gemm_backend=sglang_cfg.fp8_gemm_backend,
    )
    sglang_cache_host = DEFAULT_SGLANG_COMPILE_CACHE_HOST / cache_namespace
    inductor_cache_host = DEFAULT_INDUCTOR_CACHE_HOST / cache_namespace
    sglang_cache_host.mkdir(parents=True, exist_ok=True)
    inductor_cache_host.mkdir(parents=True, exist_ok=True)
    extra_volumes.append(
        (str(sglang_cache_host), f"{SGLANG_CACHE_CONTAINER_ROOT}/{cache_namespace}")
    )
    extra_volumes.append((str(inductor_cache_host), INDUCTOR_CACHE_CONTAINER_PATH))

    command = _build_sglang_container_command(
        effective_model,
        port=port,
        served_model_name=config.model,
        mem_fraction_static=config.mem_fraction_static,
        chat_template=(
            LOCAL_CHAT_TEMPLATE_CONTAINER_PATH if sglang_cfg.chat_template_path else None
        ),
        lora_paths=lora_container_paths,
        pinned_lora_names=sglang_cfg.pinned_lora_names,
        max_loras_per_batch=sglang_cfg.max_loras_per_batch,
        max_loaded_loras=sglang_cfg.max_loaded_loras,
        max_lora_rank=sglang_cfg.max_lora_rank,
        lora_target_modules=sglang_cfg.lora_target_modules,
        lora_backend=sglang_cfg.lora_backend,
        enable_lora_overlap_loading=sglang_cfg.enable_lora_overlap_loading,
        lora_eviction_policy=sglang_cfg.lora_eviction_policy,
        quantization=sglang_cfg.quantization,
        fp8_gemm_backend=sglang_cfg.fp8_gemm_backend,
        attention_backend=sglang_cfg.attention_backend,
        tensor_parallel_size=sglang_cfg.tensor_parallel_size,
        data_parallel_size=sglang_cfg.data_parallel_size,
        max_model_len=sglang_cfg.max_model_len,
        max_running_requests=sglang_cfg.max_running_requests,
        kv_cache_dtype=sglang_cfg.kv_cache_dtype,
        disable_radix_cache=sglang_cfg.disable_radix_cache,
        radix_eviction_policy=sglang_cfg.radix_eviction_policy,
        disable_cuda_graph=sglang_cfg.disable_cuda_graph,
        cuda_graph_max_bs=sglang_cfg.cuda_graph_max_bs,
        cuda_graph_bs=sglang_cfg.cuda_graph_bs,
        enable_torch_compile=sglang_cfg.enable_torch_compile,
        torch_compile_max_bs=sglang_cfg.torch_compile_max_bs,
        enable_chunked_prefill=sglang_cfg.enable_chunked_prefill,
        enable_mixed_chunk=sglang_cfg.enable_mixed_chunk,
        tool_call_parser=config.tool_call_parser,
        reasoning_parser=config.reasoning_parser,
        grammar_backend=sglang_cfg.grammar_backend,
        speculative_algorithm=sglang_cfg.speculative_algorithm,
        speculative_draft_model_path=sglang_cfg.speculative_draft_model_path,
        speculative_num_steps=sglang_cfg.speculative_num_steps,
        speculative_eagle_topk=sglang_cfg.speculative_eagle_topk,
        speculative_num_draft_tokens=sglang_cfg.speculative_num_draft_tokens,
    )
    _start_sglang_container(
        container_name,
        port,
        command,
        extra_volumes=extra_volumes,
        extra_env=dict(sglang_cfg.extra_env),
        image=image,
    )

    container = docker_client.containers.get(container_name)
    logger.info(f"Started SGLang container '{container_name}' (id={container.short_id})")

    try:
        startup_timeout = max(config.timeout, startup_timeout)
        models_urls = build_models_urls(
            models_url,
            host_ip,
            port,
            network_mode,
            scheme=scheme,
        )
        wait_kwargs = {}
        if len(models_urls) == 1:
            wait_kwargs["primary_timeout_s"] = startup_timeout
        working_url = _wait_for_sglang_ready(
            models_urls,
            container,
            startup_timeout,
            **wait_kwargs,
        )
        base_url = working_url.rsplit("/models", 1)[0]
        models_url = working_url
        logger.info(f"SGLang server is ready at {base_url}")

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)
        yield _build_sglang_session(
            client,
            config,
            base_url,
            models_url,
            process=container,
        )
    finally:
        logger.info(f"Stopping SGLang container '{container_name}'...")
        try:
            container.stop(timeout=10)
        except Exception as exc:
            logger.warning(f"Error stopping container: {exc}")
        try:
            container.remove(force=True)
        except Exception as exc:
            logger.warning(f"Error removing container: {exc}")


__all__ = [
    "ADAPTERS_CONTAINER_ROOT",
    "DEFAULT_INDUCTOR_CACHE_HOST",
    "DEFAULT_SGLANG_COMPILE_CACHE_HOST",
    "INDUCTOR_CACHE_CONTAINER_PATH",
    "LOCAL_CHAT_TEMPLATE_CONTAINER_PATH",
    "LOCAL_MODEL_CONTAINER_PATH",
    "SGLANG_CACHE_CONTAINER_ROOT",
    "SGLANG_IMAGE",
    "SglangSessionExtras",
    "setup_sglang",
]
