import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import urlparse

from docker import DockerClient
from docker.errors import NotFound
from loguru import logger
from openai import OpenAI

from src.backend.endpoint_pool import EndpointPool, EndpointState
from src.backend.utils import is_http_ready
from src.genner import get_genner
from src.genner.config import VllmConfig
from src.observability.gpu import list_visible_gpu_indices
from src.typing.config import AppConfig, _is_managed_vllm_endpoint

from .session import BackendSession

VLLM_IMAGE = "vllm/vllm-openai:nightly-01d4d1ad375dc5854779c593eee093bcebb0cada"
# Pinned to the 2026-05-04 nightly because v0.20.1 (current `:latest`) was
# branched from v0.20.0 on 2026-04-27 and never received the gemma4 PP fix
# from PR #40786 (merged 2026-04-29). The fix replaces a broken
# `intermediate_tensors.get("per_layer_inputs")` call (gemma4.py L1320 in
# v0.20.1) with a subscript access.
CONTAINER_NAME_PREFIX = "nsl-vllm"
HF_CACHE_HOST = Path.home() / ".cache" / "huggingface"
HF_CACHE_CONTAINER = "/tmp/huggingface"
VLLM_LOG_DIR = Path("logs/vllm")
DEFAULT_COMPILE_CACHE_HOST = Path.home() / ".cache" / "vllm_compile_cache"
COMPILE_CACHE_CONTAINER_PATH = "/tmp/vllm_compile_cache"
# Inductor compile-config overrides we always inject. vLLM 0.19.x ships
# `combo_kernels=True`, which produces a "too many values to unpack (expected 6)"
# unpack mismatch in the inductor-compiled forward of quantized + heterogeneous-
# head-dim models (e.g. gemma-4 AWQ at TP>=2).
INDUCTOR_COMPILE_CONFIG_OVERRIDES: dict[str, object] = {
    "combo_kernels": False,
    "benchmark_combo_kernel": False,
}


_MODEL_PATH_SAFE_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _sanitize_model_for_path(model: str) -> str:
    cleaned = "".join(
        c if c in _MODEL_PATH_SAFE_CHARS else "_" for c in model.replace("/", "__")
    )
    if len(cleaned) <= 80:
        return cleaned
    suffix = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:71]}_{suffix}"


def _compile_cache_namespace(
    *,
    model: str,
    tp: int | None,
    pp: int | None = None,
    kv_cache_dtype: str | None = None,
    max_model_len: int | None,
    cudagraph_mode: str | None,
    enforce_eager: bool,
    lora_enabled: bool,
) -> str:
    """Two-segment cache subdir: ``<model-bucket>/v<inductor-fingerprint>``.

    Recovers what vLLM's auto-cache layout would have keyed on. We override
    `--compilation-config cache_dir`, which collapses vLLM's per-config hash
    subdir; without our own namespacing, two models share `rank_*_*/backbone/`
    and silently reload each other's graphs (vLLM writes `cache_key_factors.json`
    but never reads it back to verify).
    """
    bucket = (
        f"{_sanitize_model_for_path(model)}"
        f"__tp{tp if tp is not None else 'x'}"
        f"__pp{pp if pp is not None else 'x'}"
        f"__mml{max_model_len if max_model_len is not None else 'x'}"
    )
    inner_payload = json.dumps(
        {
            "inductor": INDUCTOR_COMPILE_CONFIG_OVERRIDES,
            "cudagraph_mode": cudagraph_mode,
            "enforce_eager": enforce_eager,
            "lora_enabled": lora_enabled,
            "kv_cache_dtype": kv_cache_dtype,
        },
        sort_keys=True,
    )
    inner = "v" + hashlib.sha256(inner_payload.encode("utf-8")).hexdigest()[:10]
    return f"{bucket}/{inner}"


LOCAL_MODEL_CONTAINER_PATH = "/models/local"
LOCAL_ADAPTER_CONTAINER_PATH = "/adapters/local"
# vLLM request name registered by --lora-modules; distinct from the base name.
ADAPTER_MODEL_NAME = "adapter"
VLLM_NETWORK_MODE_ENV = "NSL_VLLM_NETWORK_MODE"
LEGACY_VLLM_MODEL_ALIASES = {
    "qwen2.5-coder:7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5:7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}
LORA_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.safetensors.index.json",
    "adapter_model.bin",
)


def _infer_tool_call_parser(model: str) -> str | None:
    """Best-effort default parser selection for models we run today.

    vLLM requires an explicit parser when ``tool_choice='auto'`` is used.
    Our external harness currently relies on Qwen-family instruction models,
    whose tokenizer chat templates are compatible with the Hermes parser.
    """

    normalized = model.strip().lower()
    if "qwen" in normalized or "qwq" in normalized:
        return "hermes"
    return None


def _resolve_model_for_container(
    model: str, local_model_path: str | None
) -> tuple[str, str | None, bool]:
    """Returns (model_arg_for_vllm, host_mount_path, needs_bnb_flags)."""
    if local_model_path:
        return (
            LOCAL_MODEL_CONTAINER_PATH,
            str(Path(local_model_path).expanduser().resolve()),
            False,
        )
    lower = model.lower()
    return model, None, ("bnb-4bit" in lower or "bnb-8bit" in lower)


def _validate_lora_adapter_path(lora_adapter_path: str) -> str:
    resolved_path = Path(lora_adapter_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {resolved_path}")
    if not resolved_path.is_dir():
        raise ValueError(f"LoRA adapter path must be a directory, got: {resolved_path}")

    has_adapter_config = (resolved_path / "adapter_config.json").exists()
    has_adapter_weights = any(
        (resolved_path / filename).exists() for filename in LORA_WEIGHT_FILENAMES
    )
    if has_adapter_config and has_adapter_weights:
        try:
            adapter_config = json.loads(
                (resolved_path / "adapter_config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"LoRA adapter config is unreadable: {resolved_path / 'adapter_config.json'}"
            ) from exc
        if not isinstance(adapter_config, dict):
            raise ValueError(
                f"LoRA adapter config must be an object: {resolved_path / 'adapter_config.json'}"
            )
        return str(resolved_path)

    artifact_hint = ""
    if (resolved_path / "model.safetensors.index.json").exists() or any(
        resolved_path.glob("model-*.safetensors")
    ):
        artifact_hint = " The directory looks like a merged/full-model export, not a PEFT LoRA adapter."

    visible_entries = sorted(
        path.name for path in resolved_path.iterdir() if not path.name.startswith(".")
    )
    visible_entries_preview = ", ".join(visible_entries[:8])
    if len(visible_entries) > 8:
        visible_entries_preview += ", ..."

    raise ValueError(
        "Invalid LoRA adapter directory "
        f"'{resolved_path}': expected adapter_config.json and adapter_model weights."
        f"{artifact_hint}"
        + (
            f" Found: {visible_entries_preview}"
            if visible_entries_preview
            else " Directory is empty."
        )
    )


def _get_host_ip() -> str:
    """Get a routable host IP for container-to-host connectivity."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as exc:
        logger.debug(f"Failed to detect non-loopback host IP: {exc}")
        return "127.0.0.1"


def _get_network_mode() -> str:
    """Return the preferred vLLM network mode.

    Supported values:
    - auto: try the configured endpoint first, then a detected host IP
    - loopback: only use the configured endpoint
    - hostip: prefer the detected host IP endpoint
    """

    mode = os.getenv(VLLM_NETWORK_MODE_ENV, "auto").strip().lower()
    if mode in {"auto", "loopback", "hostip"}:
        return mode

    logger.warning(
        f"Invalid {VLLM_NETWORK_MODE_ENV} value '{mode}'; falling back to 'auto'"
    )
    return "auto"


def _build_models_urls(
    models_url: str,
    host_ip: str | None,
    port: int,
    network_mode: str,
    *,
    scheme: str,
) -> list[str]:
    if network_mode == "loopback":
        return [models_url]

    if host_ip in (None, "127.0.0.1"):
        if network_mode == "hostip":
            logger.warning(
                "Requested host IP networking, but no non-loopback host IP was "
                "detected. Falling back to the configured endpoint."
            )
        return [models_url]

    host_models_url = f"{scheme}://{host_ip}:{port}/v1/models"
    if network_mode == "hostip":
        return [host_models_url]

    if host_models_url == models_url:
        return [models_url]

    return [models_url, host_models_url]


def _read_gpu_memory_info_mb(device_index: int = 0) -> tuple[float, float] | None:
    """Return (free_mb, total_mb) via the shared GPU utility."""
    from src.observability.gpu import read_gpu_memory_info

    result = read_gpu_memory_info(device_index)
    if result is None:
        return None
    _used_mb, free_mb, total_mb = result
    return (free_mb, total_mb)


def _probe_devices_mb(
    device_indices: Sequence[int],
) -> list[tuple[float, float]] | None:
    """Probe each device; return None if any device's metrics are unreadable."""
    snapshots: list[tuple[float, float]] = []
    for index in device_indices:
        info = _read_gpu_memory_info_mb(index)
        if info is None:
            return None
        snapshots.append(info)
    return snapshots


def wait_for_gpu_memory_release(
    *,
    min_free_memory_fraction: float,
    timeout_s: int,
    poll_interval_s: float = 2.0,
    device_indices: Sequence[int] | None = None,
) -> None:
    """Block until every targeted GPU has at least `min_free_memory_fraction` of its
    own memory free.

    When ``device_indices`` is None, enumerate visible GPUs via NVML / nvidia-smi
    and wait on all of them. This matters when a previous vLLM session ran with
    pipeline_parallel_size > 1 or tensor_parallel_size > 1 — checking only GPU 0
    can falsely pass while GPU 1 still holds engine weights.
    """
    if timeout_s <= 0:
        return

    if device_indices is None:
        device_indices = list_visible_gpu_indices()
        if not device_indices:
            logger.warning(
                "Skipping GPU memory wait because no GPUs are visible to NVML / nvidia-smi"
            )
            return
    else:
        device_indices = list(device_indices)

    snapshots = _probe_devices_mb(device_indices)
    if snapshots is None:
        logger.warning(
            "Skipping GPU memory wait because GPU memory metrics are unavailable"
        )
        return

    deadline = time.monotonic() + timeout_s

    while True:
        blocking = [
            (idx, free_mb, total_mb)
            for idx, (free_mb, total_mb) in zip(device_indices, snapshots)
            if free_mb < total_mb * min_free_memory_fraction
        ]
        if not blocking:
            summary = ", ".join(
                f"GPU{idx} {free_mb:.0f}/{total_mb:.0f} MB free"
                for idx, (free_mb, total_mb) in zip(device_indices, snapshots)
            )
            logger.info(f"GPU memory gate passed: {summary}")
            return

        if time.monotonic() >= deadline:
            detail = ", ".join(
                f"GPU{idx}: {free_mb:.0f} MB free, needed {total_mb * min_free_memory_fraction:.0f}"
                for idx, free_mb, total_mb in blocking
            )
            raise TimeoutError(
                f"Timed out waiting for GPU memory to be released ({detail})"
            )

        time.sleep(poll_interval_s)
        snapshots = _probe_devices_mb(device_indices)
        if snapshots is None:
            logger.warning(
                "GPU memory metrics became unavailable while waiting; continuing without a wait gate"
            )
            return


def _build_vllm_config(
    app_config: AppConfig,
    endpoint: str,
    *,
    api_key: str | None = None,
) -> VllmConfig:
    config = VllmConfig()
    if app_config.model_name.startswith("vllm:"):
        config.model = app_config.model_name.split(":", 1)[1].strip()

    normalized_model = config.model.strip()
    resolved_model = LEGACY_VLLM_MODEL_ALIASES.get(normalized_model, normalized_model)
    if resolved_model != normalized_model:
        logger.info(f"Resolved model alias '{normalized_model}' to '{resolved_model}'")
    config.model = resolved_model
    parser_model_name = resolved_model
    config.endpoint = endpoint
    config.api_key = api_key
    config.timeout = app_config.inference.timeout
    config.gpu_memory_utilization = app_config.gpu_memory_utilization
    config.temperature = app_config.inference.temperature
    # When inference.max_tokens is None (the default) we propagate None
    # so OAIGenner omits the kwarg and vLLM uses its own per-request
    # budget of (max_model_len - prompt_tokens). When the user pins an
    # explicit number, only clamp it down to max_model_len here as a
    # static guard — the prompt-aware "max_tokens + prompt > max_model_len"
    # rejection still has to be handled at request time, but this at
    # least catches the obvious "configured higher than the engine
    # context" misconfiguration.
    requested_max_tokens = app_config.inference.max_tokens
    if (
        requested_max_tokens is not None
        and app_config.vllm is not None
        and app_config.vllm.max_model_len is not None
        and requested_max_tokens > app_config.vllm.max_model_len
    ):
        logger.info(
            f"inference.max_tokens={requested_max_tokens} exceeds "
            f"vllm.max_model_len={app_config.vllm.max_model_len}; "
            f"capping to max_model_len. Set inference.max_tokens=null "
            "(or omit it) to let vLLM size each request automatically."
        )
        requested_max_tokens = app_config.vllm.max_model_len
    config.max_tokens = requested_max_tokens
    config.frequency_penalty = app_config.inference.frequency_penalty
    config.presence_penalty = app_config.inference.presence_penalty
    if app_config.vllm is not None and app_config.vllm.chat_template_path:
        chat_template_path = Path(app_config.vllm.chat_template_path)
        config.chat_template = chat_template_path.read_text(encoding="utf-8")
    if app_config.vllm is not None:
        if app_config.vllm.lora_adapter_path:
            config.model = ADAPTER_MODEL_NAME
        elif app_config.vllm.served_model_name:
            config.model = app_config.vllm.served_model_name
        config.tool_call_parser = (
            app_config.vllm.tool_call_parser
            or _infer_tool_call_parser(parser_model_name)
        )
        # reasoning_parser is opt-in (no auto-inference): see VllmConfig.
        config.reasoning_parser = app_config.vllm.reasoning_parser
        if app_config.vllm.enable_auto_tool_choice is None:
            config.enable_auto_tool_choice = config.tool_call_parser is not None
        else:
            config.enable_auto_tool_choice = app_config.vllm.enable_auto_tool_choice
    else:
        config.tool_call_parser = _infer_tool_call_parser(parser_model_name)
        config.enable_auto_tool_choice = config.tool_call_parser is not None

    if config.enable_auto_tool_choice and not config.tool_call_parser:
        raise ValueError(
            "vLLM auto tool choice enabled but no tool_call_parser was configured "
            f"or inferred for model {parser_model_name!r}"
        )
    return config


def _endpoint_base_urls(endpoint: str) -> tuple[str, str, str, str]:
    root = endpoint.rstrip("/")
    base_url = root if root.endswith("/v1") else f"{root}/v1"
    models_url = f"{base_url}/models"
    server_root = base_url.rsplit("/v1", 1)[0]
    metrics_url = f"{server_root}/metrics"
    return root, base_url, models_url, metrics_url


def _default_endpoint_capacity(app_config: AppConfig) -> int:
    vllm_cfg = app_config.vllm
    if vllm_cfg is not None and vllm_cfg.max_num_seqs is not None:
        return max(1, int(vllm_cfg.max_num_seqs))
    if app_config.generation is not None:
        return max(1, int(app_config.generation.parallel_episodes))
    return 1


def _min_healthy_endpoint_capacity(app_config: AppConfig) -> int:
    vllm_cfg = app_config.vllm
    if vllm_cfg is None:
        return 1
    return vllm_cfg.min_healthy_endpoint_capacity


def _api_key_from_env(api_key_env: str | None) -> str | None:
    if not api_key_env:
        return None
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"Environment variable {api_key_env} is required for vLLM endpoint auth"
        )
    return api_key


def _is_http_ready_with_auth(
    url: str,
    *,
    api_key: str | None = None,
    timeout_sec: float = 2.0,
) -> bool:
    if not api_key:
        return is_http_ready(url, timeout_sec=timeout_sec)
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _model_inventory_attests_lora(inventory: Any, requested_model: str) -> bool:
    cards = getattr(inventory, "data", None)
    if not isinstance(cards, list):
        return False
    matches = [card for card in cards if getattr(card, "id", None) == requested_model]
    if len(matches) != 1:
        return False
    parent = getattr(matches[0], "parent", None)
    return isinstance(parent, str) and bool(parent.strip())


def _external_model_attestation(client: OpenAI, requested_model: str) -> bool:
    try:
        return _model_inventory_attests_lora(client.models.list(), requested_model)
    except Exception as exc:
        logger.warning(f"Unable to inspect external vLLM model inventory: {exc}")
        return False


def _attach_endpoint_pool(
    session: BackendSession,
    app_config: AppConfig,
    *,
    endpoint_id: str = "vllm-0",
    capacity: int | None = None,
    api_key: str | None = None,
    healthy: bool = True,
    metadata: dict[str, Any] | None = None,
    external_lora_attested: bool | None = None,
    recovery_probe: Callable[[], bool] | None = None,
) -> BackendSession:
    if session.base_url is None or session.models_url is None:
        return session
    endpoint_capacity = (
        capacity if capacity is not None else _default_endpoint_capacity(app_config)
    )
    pool = EndpointPool(
        [
            EndpointState(
                endpoint_id=endpoint_id,
                base_url=session.base_url,
                models_url=session.models_url,
                metrics_url=session.metrics_url,
                capacity=endpoint_capacity,
                genner=session.genner,
                api_key=api_key,
                healthy=healthy,
                metadata={"capacity": endpoint_capacity, **(metadata or {})},
                recovery_probe=recovery_probe,
            )
        ],
        min_healthy_capacity=_min_healthy_endpoint_capacity(app_config),
    )
    session.genner = pool.default_genner
    session.extras = {
        "endpoint_pool": pool,
        "metrics_api_key": pool.default_api_key,
        "metrics_urls": [pool.default_metrics_url] if pool.default_metrics_url else [],
        "external_lora_attested": external_lora_attested,
    }
    return session


def _build_pool_smoke_test(
    pool: EndpointPool,
    sessions_by_endpoint: dict[str, BackendSession],
):
    smoke_tests = {
        endpoint_id: session.smoke_test
        for endpoint_id, session in sessions_by_endpoint.items()
    }

    def run_smoke_test() -> str:
        errors: list[str] = []
        for endpoint_id in pool.endpoint_ids():
            if not pool.is_healthy(endpoint_id):
                continue
            smoke_test = smoke_tests[endpoint_id]
            if smoke_test is None:
                return f"{endpoint_id}: smoke test unavailable"
            try:
                return f"{endpoint_id}: {smoke_test()}"
            except Exception as exc:
                pool.mark_unhealthy(endpoint_id, str(exc))
                errors.append(f"{endpoint_id}: {exc}")
        detail = "; ".join(errors) if errors else "no healthy endpoints"
        raise RuntimeError(f"No healthy vLLM endpoint passed smoke test ({detail})")

    return run_smoke_test


def _build_external_endpoint_pool_session(app_config: AppConfig) -> BackendSession:
    vllm_cfg = app_config.vllm
    if vllm_cfg is None or not vllm_cfg.endpoints:
        raise ValueError("vllm.endpoints is required for external endpoint pool setup")

    states: list[EndpointState] = []
    sessions_by_endpoint: dict[str, BackendSession] = {}
    ordered_sessions: list[BackendSession] = []
    requires_lora_attestation = bool(vllm_cfg.lora_adapter_path)
    attested_capacity = 0
    for index, endpoint_cfg in enumerate(vllm_cfg.endpoints):
        endpoint_id = endpoint_cfg.effective_id(index)
        raw_endpoint, base_url, models_url, metrics_url = _endpoint_base_urls(
            endpoint_cfg.base_url
        )
        api_key = _api_key_from_env(endpoint_cfg.api_key_env)
        config = _build_vllm_config(app_config, endpoint=raw_endpoint, api_key=api_key)
        client = OpenAI(
            api_key=api_key or "dummy", base_url=base_url, timeout=config.timeout
        )
        session = _build_vllm_session(client, config, base_url, models_url)
        session.metrics_url = metrics_url
        transport_ready = _is_http_ready_with_auth(models_url, api_key=api_key)
        lora_attested = (
            _external_model_attestation(client, config.model)
            if transport_ready
            else False
        )
        healthy = transport_ready and (
            lora_attested or not requires_lora_attestation
        )
        if healthy and lora_attested:
            attested_capacity += endpoint_cfg.capacity
        states.append(
            EndpointState(
                endpoint_id=endpoint_id,
                base_url=base_url,
                models_url=models_url,
                metrics_url=metrics_url,
                capacity=endpoint_cfg.capacity,
                genner=session.genner,
                api_key=api_key,
                healthy=healthy,
                metadata={
                    "capacity": endpoint_cfg.capacity,
                    "api_key_env": endpoint_cfg.api_key_env,
                    "lora_attested": lora_attested,
                    "requires_lora_attestation": requires_lora_attestation,
                },
                recovery_probe=(
                    lambda client=client, requested_model=config.model: (
                        _external_model_attestation(client, requested_model)
                    )
                )
                if requires_lora_attestation
                else None,
            )
        )
        sessions_by_endpoint[endpoint_id] = session
        ordered_sessions.append(session)
        if healthy:
            logger.info(
                f"Configured vLLM endpoint {endpoint_id} at {base_url} "
                f"(capacity={endpoint_cfg.capacity})"
            )
        elif transport_ready and requires_lora_attestation:
            logger.warning(
                f"Configured vLLM endpoint {endpoint_id} did not attest requested "
                f"model {config.model!r} as a LoRA; it will remain quarantined"
            )
        else:
            logger.warning(
                f"Configured vLLM endpoint {endpoint_id} at {base_url} is not ready; "
                "it will remain quarantined until a probe succeeds"
            )

    if requires_lora_attestation and attested_capacity < max(
        1, _min_healthy_endpoint_capacity(app_config)
    ):
        raise RuntimeError(
            "External vLLM LoRA attestation failed: /models did not report the "
            f"requested model {ADAPTER_MODEL_NAME!r} with a non-empty parent"
        )

    pool = EndpointPool(
        states,
        min_healthy_capacity=_min_healthy_endpoint_capacity(app_config),
    )
    primary = ordered_sessions[0]
    primary.genner = pool.default_genner
    primary.smoke_test = _build_pool_smoke_test(pool, sessions_by_endpoint)
    primary.metrics_url = pool.default_metrics_url
    primary.extras = {
        "endpoint_pool": pool,
        "endpoint_sessions": sessions_by_endpoint,
        "metrics_api_key": pool.default_api_key,
        "metrics_urls": [
            session.metrics_url for session in ordered_sessions if session.metrics_url
        ],
        "external_lora_attested": True if requires_lora_attestation else None,
    }
    return primary


def _build_container_command(
    model: str,
    gpu_memory_utilization: float,
    chat_template: str | None = None,
    lora_adapter_path: str | None = None,
    served_model_name: str | None = None,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    compile_cache_dir: str | None = None,
    needs_bnb: bool | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    tensor_parallel_size: int | None = None,
    data_parallel_size: int | None = None,
    pipeline_parallel_size: int | None = None,
    kv_cache_dtype: str | None = None,
    disable_custom_all_reduce: bool = False,
    enforce_eager: bool = False,
    cudagraph_mode: str | None = None,
) -> list[str]:
    cmd = [
        "--model",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if chat_template:
        cmd.extend(["--chat-template", chat_template])
    use_bnb = (
        needs_bnb
        if needs_bnb is not None
        else ("bnb-4bit" in model.lower() or "bnb-8bit" in model.lower())
    )
    if use_bnb:
        cmd.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    if lora_adapter_path:
        cmd.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{ADAPTER_MODEL_NAME}={LOCAL_ADAPTER_CONTAINER_PATH}",
                "--max-lora-rank",
                "64",
            ]
        )
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if enable_auto_tool_choice:
        cmd.append("--enable-auto-tool-choice")
    if tool_call_parser:
        cmd.extend(["--tool-call-parser", tool_call_parser])
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    if max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
    if enable_chunked_prefill:
        cmd.extend(["--enable-chunked-prefill"])
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if tensor_parallel_size is not None:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if data_parallel_size is not None:
        cmd.extend(["--data-parallel-size", str(data_parallel_size)])
    if pipeline_parallel_size is not None:
        cmd.extend(["--pipeline-parallel-size", str(pipeline_parallel_size)])
    if kv_cache_dtype is not None:
        cmd.extend(["--kv-cache-dtype", kv_cache_dtype])
    if disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    if enforce_eager:
        cmd.append("--enforce-eager")
    compilation_config: dict[str, object] = {}
    if compile_cache_dir:
        compilation_config["cache_dir"] = compile_cache_dir
    if cudagraph_mode:
        # vLLM exposes `cudagraph_mode` only inside --compilation-config JSON
        # (there is no top-level --cudagraph-mode flag). The enum is uppercase.
        compilation_config["cudagraph_mode"] = cudagraph_mode.upper()
    compilation_config["inductor_compile_config"] = dict(
        INDUCTOR_COMPILE_CONFIG_OVERRIDES
    )
    cmd.extend(["--compilation-config", json.dumps(compilation_config)])
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


def _start_vllm_container(
    name: str,
    port: int,
    vllm_command: list[str],
    extra_volumes: list[tuple[str, str]] | None = None,
    extra_env: dict[str, str] | None = None,
    cdi_devices: list[str] | None = None,
) -> None:
    """Start a vLLM container using the docker CLI.

    We use subprocess instead of the Python SDK because the CLI supports the
    GPU device configuration used by the local environment.
    """
    HF_CACHE_HOST.mkdir(parents=True, exist_ok=True)

    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
    ]
    for device in cdi_devices if cdi_devices is not None else ["all"]:
        docker_cmd.extend(["--device", f"nvidia.com/gpu={device}"])
    docker_cmd.extend(
        [
            "--ipc=host",
            "--network=host",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"HF_HOME={HF_CACHE_CONTAINER}",
            "-e",
            "HOME=/tmp",
            "-v",
            "/etc/passwd:/etc/passwd:ro",
            "-v",
            "/etc/group:/etc/group:ro",
            "-v",
            f"{HF_CACHE_HOST}:{HF_CACHE_CONTAINER}",
        ]
    )
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
    docker_cmd.extend([VLLM_IMAGE, *vllm_command])

    logger.info(f"Docker command: {' '.join(docker_cmd)}")
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start vLLM container: {result.stderr.strip()}")


def _dump_container_logs(container, reason: str) -> Path | None:
    try:
        full_logs = container.logs().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning(f"Could not collect full container logs: {exc}")
        return None
    VLLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = VLLM_LOG_DIR / f"{container.name}-{timestamp}-{reason}.log"
    log_path.write_text(full_logs, encoding="utf-8")
    logger.info(f"Wrote full vLLM container logs to {log_path}")
    return log_path


def _wait_for_vllm_ready(
    models_urls: str | list[str],
    container,
    timeout_s: int,
    primary_timeout_s: int = 60,
) -> str:
    """Wait for vLLM to become ready, trying multiple endpoints.

    Returns the first URL that becomes ready.

    Args:
        models_urls: Candidate URLs to try in order
        container: Docker container handle
        timeout_s: Total timeout for all endpoints
        primary_timeout_s: Timeout for the first endpoint before trying alternates
    """
    if isinstance(models_urls, str):
        models_urls = [models_urls]

    deadline = time.time() + timeout_s
    start_time = time.time()
    next_log_at = 30.0

    primary_url = models_urls[0]
    alternate_urls = models_urls[1:]
    primary_deadline = time.time() + min(primary_timeout_s, timeout_s)
    last_checked_url = primary_url
    using_alternates = False
    while time.time() < deadline:
        if not using_alternates and alternate_urls and time.time() >= primary_deadline:
            using_alternates = True
            logger.info(
                "Primary endpoint unreachable, switching to alternate: "
                f"{alternate_urls[0]}"
            )

        candidate_urls = alternate_urls if using_alternates else [primary_url]
        if not candidate_urls:
            candidate_urls = [primary_url]

        for current_url in candidate_urls:
            last_checked_url = current_url
            if is_http_ready(current_url):
                return current_url

        container.reload()
        if container.status in ("exited", "dead"):
            logs = container.logs(tail=200).decode("utf-8", errors="replace")
            log_path = _dump_container_logs(container, reason="exited")
            path_msg = f" Full logs written to {log_path}." if log_path else ""
            raise RuntimeError(
                f"vLLM container exited unexpectedly (status={container.status})."
                f"{path_msg}\nRecent logs:\n{logs}"
            )

        elapsed = time.time() - start_time
        if elapsed >= next_log_at:
            logger.info(
                f"Waiting for vLLM readiness... ({int(elapsed)}s / {timeout_s}s)"
            )
            next_log_at += 30.0

        time.sleep(2)

    logs = container.logs(tail=200).decode("utf-8", errors="replace")
    log_path = _dump_container_logs(container, reason="timeout")
    path_msg = f" Full logs written to {log_path}." if log_path else ""
    raise TimeoutError(
        f"Timed out waiting for vLLM readiness at {last_checked_url} after {timeout_s}s."
        f"{path_msg}\nRecent logs:\n{logs}"
    )


def _build_vllm_smoke_test(client: OpenAI, config: VllmConfig):
    def _liveness_text(message: Any) -> str | None:
        """Return any non-empty text from the response.

        With --reasoning-parser, vLLM routes <think> content into
        message.reasoning_content and may leave message.content as None
        when generation truncates inside the think block. Either is
        proof-of-life for a smoke test.
        """
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        return None

    def run_smoke_test() -> str:
        # max_tokens needs to be high enough to clear a <think> block on
        # reasoning models. Kept well under any plausible max_model_len
        # so the request validates regardless of engine context size.
        test_response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Who are you?"}],
            max_tokens=512,
            temperature=0.5,
        )
        content = _liveness_text(test_response.choices[0].message)
        if content is None:
            raise RuntimeError("vLLM smoke test returned a non-text response")

        if config.enable_auto_tool_choice:
            client.chat.completions.create(
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
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                ],
                tool_choice="auto",
                max_tokens=512,
                temperature=0.0,
            )
        return content

    return run_smoke_test


def _build_vllm_session(
    client: OpenAI,
    config: VllmConfig,
    base_url: str,
    models_url: str,
    process=None,
) -> BackendSession:
    genner = get_genner("vllm", server_config=config, oai_client=client)
    # Derive metrics URL from the resolved base_url (which may differ from
    # the raw endpoint config due to Docker networking resolution).
    server_root = base_url.rsplit("/v1", 1)[0]
    metrics_url = f"{server_root}/metrics"
    return BackendSession(
        genner=genner,
        smoke_test=_build_vllm_smoke_test(client, config),
        client=client,
        config=config,
        base_url=base_url,
        models_url=models_url,
        process=process,
        metrics_url=metrics_url,
    )


@contextmanager
def _monitor_endpoint_pool(session: BackendSession) -> Iterator[BackendSession]:
    pool = session.extras["endpoint_pool"]
    try:
        pool.start_health_monitor()
        yield session
    finally:
        pool.stop_health_monitor()


@contextmanager
def setup_vllm(
    app_config: AppConfig,
    *,
    endpoint: str = "http://127.0.0.1:8000",
) -> Iterator[BackendSession]:
    vllm_cfg = app_config.vllm
    if vllm_cfg is not None and vllm_cfg.endpoints:
        session = _build_external_endpoint_pool_session(app_config)
        with _monitor_endpoint_pool(session) as monitored_session:
            yield monitored_session
        return

    if vllm_cfg is not None and vllm_cfg.endpoint:
        endpoint = vllm_cfg.endpoint

    startup_timeout = (
        vllm_cfg.startup_timeout if vllm_cfg and vllm_cfg.startup_timeout else 500
    )
    config = _build_vllm_config(app_config, endpoint=endpoint)
    source_model = (
        app_config.model_name.split(":", 1)[1].strip()
        if app_config.model_name.startswith("vllm:")
        else app_config.model_name
    )
    source_model = LEGACY_VLLM_MODEL_ALIASES.get(
        source_model.strip(), source_model.strip()
    )

    local_model_path = vllm_cfg.local_model_path if vllm_cfg else None
    lora_adapter_path = vllm_cfg.lora_adapter_path if vllm_cfg else None
    served_model_name = vllm_cfg.served_model_name if vllm_cfg else None
    resolved_lora_adapter_path = (
        _validate_lora_adapter_path(lora_adapter_path) if lora_adapter_path else None
    )
    if local_model_path:
        resolved_local_model_path = Path(local_model_path).expanduser().resolve()
        if not resolved_local_model_path.exists():
            raise FileNotFoundError(
                f"Local vLLM model path not found: {resolved_local_model_path}"
            )
        if not resolved_local_model_path.is_dir():
            raise ValueError(
                f"Local vLLM model path must be a directory: {resolved_local_model_path}"
            )

    endpoint = config.endpoint.rstrip("/")
    base_url = endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"
    models_url = f"{base_url}/models"
    api_key = config.api_key or "dummy"

    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    scheme = parsed.scheme or "http"
    port = parsed.port or 8000
    strict_local_artifact = bool(local_model_path or lora_adapter_path)
    configured_external_endpoint = bool(
        vllm_cfg is not None
        and vllm_cfg.endpoint
        and not _is_managed_vllm_endpoint(vllm_cfg.endpoint)
    )
    docker_client = None
    stale_container_removed = False
    server_ready = is_http_ready(models_url)
    if configured_external_endpoint and not server_ready:
        raise RuntimeError(f"Configured external vLLM endpoint is not ready: {base_url}")
    if configured_external_endpoint and server_ready:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)
        lora_attested = (
            _external_model_attestation(client, config.model)
            if lora_adapter_path
            else False
        )
        if lora_adapter_path and not lora_attested:
            raise RuntimeError(
                "External vLLM LoRA attestation failed: /models did not report "
                f"the requested model {config.model!r} with a non-empty parent"
            )
        session = _build_vllm_session(client, config, base_url, models_url)
        session = _attach_endpoint_pool(
            session,
            app_config,
            api_key=config.api_key,
            healthy=True,
            metadata={
                "lora_attested": lora_attested,
                "requires_lora_attestation": bool(lora_adapter_path),
            },
            external_lora_attested=True if lora_adapter_path else None,
            recovery_probe=(
                lambda: _external_model_attestation(client, config.model)
            )
            if lora_adapter_path
            else None,
        )
        with _monitor_endpoint_pool(session) as monitored_session:
            yield monitored_session
        return
    if server_ready and strict_local_artifact:
        docker_client = DockerClient.from_env()
        container_name = _container_name(port)
        try:
            stale = docker_client.containers.get(container_name)
        except NotFound as exc:
            raise RuntimeError(
                f"A server is already listening at {base_url}, but local artifact "
                "serving requires a framework-managed vLLM container"
            ) from exc
        stale.remove(force=True)
        stale_container_removed = True
    elif server_ready:
        logger.info(f"Using existing server at {base_url}")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)
        session = _build_vllm_session(client, config, base_url, models_url)
        session = _attach_endpoint_pool(
            session,
            app_config,
            api_key=config.api_key,
            healthy=True,
        )
        with _monitor_endpoint_pool(session) as monitored_session:
            yield monitored_session
        return

    network_mode = _get_network_mode()

    host_ip = None if network_mode == "loopback" else _get_host_ip()

    logger.info(
        f"No server detected at {base_url}. "
        f"Starting vLLM Docker container for {source_model} on port {port}..."
    )

    raw_compile_cache_dir = (
        vllm_cfg.compile_cache_dir
        if vllm_cfg and vllm_cfg.compile_cache_dir is not None
        else str(DEFAULT_COMPILE_CACHE_HOST)
    )
    compile_cache_dir = raw_compile_cache_dir.strip() or str(DEFAULT_COMPILE_CACHE_HOST)
    if docker_client is None:
        docker_client = DockerClient.from_env()
    container_name = _container_name(port)
    if not stale_container_removed:
        _remove_stale_container(docker_client, container_name)

    effective_model, model_host_path, needs_bnb = _resolve_model_for_container(
        source_model,
        local_model_path,
    )

    if model_host_path and not served_model_name and not lora_adapter_path:
        config.model = effective_model

    extra_volumes: list[tuple[str, str]] = []
    if model_host_path:
        extra_volumes.append((model_host_path, LOCAL_MODEL_CONTAINER_PATH))
    if resolved_lora_adapter_path:
        extra_volumes.append(
            (
                resolved_lora_adapter_path,
                LOCAL_ADAPTER_CONTAINER_PATH,
            )
        )
    compile_cache_host_path = Path(compile_cache_dir).expanduser().resolve()
    compile_cache_host_path.mkdir(parents=True, exist_ok=True)
    extra_volumes.append((str(compile_cache_host_path), COMPILE_CACHE_CONTAINER_PATH))
    cache_namespace = _compile_cache_namespace(
        model=effective_model,
        tp=vllm_cfg.tensor_parallel_size if vllm_cfg else None,
        pp=vllm_cfg.pipeline_parallel_size if vllm_cfg else None,
        kv_cache_dtype=vllm_cfg.kv_cache_dtype if vllm_cfg else None,
        max_model_len=vllm_cfg.max_model_len if vllm_cfg else None,
        cudagraph_mode=vllm_cfg.cudagraph_mode if vllm_cfg else None,
        enforce_eager=vllm_cfg.enforce_eager if vllm_cfg else False,
        lora_enabled=resolved_lora_adapter_path is not None,
    )
    (compile_cache_host_path / cache_namespace).mkdir(parents=True, exist_ok=True)
    namespaced_container_cache_dir = f"{COMPILE_CACHE_CONTAINER_PATH}/{cache_namespace}"

    vllm_command = _build_container_command(
        effective_model,
        config.gpu_memory_utilization,
        config.chat_template,
        lora_adapter_path=(
            LOCAL_ADAPTER_CONTAINER_PATH
            if resolved_lora_adapter_path is not None
            else None
        ),
        served_model_name=served_model_name,
        enable_auto_tool_choice=config.enable_auto_tool_choice,
        tool_call_parser=config.tool_call_parser,
        reasoning_parser=config.reasoning_parser,
        compile_cache_dir=namespaced_container_cache_dir,
        needs_bnb=needs_bnb,
        max_model_len=vllm_cfg.max_model_len if vllm_cfg else None,
        max_num_seqs=vllm_cfg.max_num_seqs if vllm_cfg else None,
        max_num_batched_tokens=(vllm_cfg.max_num_batched_tokens if vllm_cfg else None),
        enable_chunked_prefill=vllm_cfg.enable_chunked_prefill if vllm_cfg else False,
        enable_prefix_caching=vllm_cfg.enable_prefix_caching if vllm_cfg else False,
        tensor_parallel_size=vllm_cfg.tensor_parallel_size if vllm_cfg else None,
        data_parallel_size=vllm_cfg.data_parallel_size if vllm_cfg else None,
        pipeline_parallel_size=(vllm_cfg.pipeline_parallel_size if vllm_cfg else None),
        kv_cache_dtype=vllm_cfg.kv_cache_dtype if vllm_cfg else None,
        disable_custom_all_reduce=(
            vllm_cfg.disable_custom_all_reduce if vllm_cfg else True
        ),
        enforce_eager=vllm_cfg.enforce_eager if vllm_cfg else False,
        cudagraph_mode=vllm_cfg.cudagraph_mode if vllm_cfg else None,
    )
    _start_vllm_container(
        container_name,
        port,
        vllm_command,
        extra_volumes=extra_volumes,
        extra_env=dict(vllm_cfg.extra_env) if vllm_cfg else None,
        cdi_devices=list(vllm_cfg.cdi_devices) if vllm_cfg else None,
    )

    container = docker_client.containers.get(container_name)
    logger.info(f"Started container '{container_name}' (id={container.short_id})")

    try:
        startup_timeout = max(config.timeout, startup_timeout)

        models_urls = _build_models_urls(
            models_url,
            host_ip,
            port,
            network_mode,
            scheme=scheme,
        )
        if len(models_urls) > 1:
            logger.info(
                f"Will try alternate endpoint {models_urls[1]} if {models_urls[0]} fails"
            )
        elif models_urls[0] != models_url:
            logger.info(
                f"Using host IP endpoint {models_urls[0]} due to "
                f"{VLLM_NETWORK_MODE_ENV}={network_mode}"
            )

        wait_kwargs = {}
        if len(models_urls) == 1:
            wait_kwargs["primary_timeout_s"] = startup_timeout

        working_url = _wait_for_vllm_ready(
            models_urls,
            container,
            startup_timeout,
            **wait_kwargs,
        )
        base_url = working_url.rsplit("/models", 1)[0]
        models_url = working_url
        logger.info(f"vLLM server is ready at {base_url}")

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)

        session = _build_vllm_session(
            client,
            config,
            base_url,
            models_url,
            process=container,
        )
        session = _attach_endpoint_pool(
            session,
            app_config,
            api_key=config.api_key,
            healthy=True,
        )
        with _monitor_endpoint_pool(session) as monitored_session:
            yield monitored_session
    finally:
        logger.info(f"Stopping vLLM container '{container_name}'...")
        try:
            container.stop(timeout=10)
        except Exception as e:
            logger.warning(f"Error stopping container: {e}")
        try:
            container.remove(force=True)
        except Exception as e:
            logger.warning(f"Error removing container: {e}")


__all__ = [
    "COMPILE_CACHE_CONTAINER_PATH",
    "DEFAULT_COMPILE_CACHE_HOST",
    "INDUCTOR_COMPILE_CONFIG_OVERRIDES",
    "LOCAL_ADAPTER_CONTAINER_PATH",
    "ADAPTER_MODEL_NAME",
    "LOCAL_MODEL_CONTAINER_PATH",
    "_compile_cache_namespace",
    "setup_vllm",
    "wait_for_gpu_memory_release",
]
