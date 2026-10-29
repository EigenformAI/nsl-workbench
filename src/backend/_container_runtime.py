import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.backend.utils import is_http_ready


_MODEL_PATH_SAFE_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)

LORA_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.safetensors.index.json",
    "adapter_model.bin",
)


def sanitize_model_for_path(model: str) -> str:
    cleaned = "".join(
        c if c in _MODEL_PATH_SAFE_CHARS else "_" for c in model.replace("/", "__")
    )
    if len(cleaned) <= 80:
        return cleaned
    suffix = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:71]}_{suffix}"


def compile_cache_namespace(
    *,
    engine: str,
    model: str,
    tp: int | None = None,
    pp: int | None = None,
    max_model_len: int | None = None,
    **factors: Any,
) -> str:
    """Return ``<engine>/<model-bucket>/v<fingerprint>`` for compile caches."""

    bucket = (
        f"{sanitize_model_for_path(model)}"
        f"__tp{tp if tp is not None else 'x'}"
        f"__pp{pp if pp is not None else 'x'}"
        f"__mml{max_model_len if max_model_len is not None else 'x'}"
    )
    payload = json.dumps(factors, sort_keys=True, default=str)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{engine}/{bucket}/v{fingerprint}"


def validate_lora_adapter_path(lora_adapter_path: str) -> str:
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


def resolve_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception as exc:
        logger.debug(f"Failed to detect non-loopback host IP: {exc}")
        return "127.0.0.1"


def resolve_network_mode(env_var: str, default: str = "auto") -> str:
    mode = os.getenv(env_var, default).strip().lower()
    if mode in {"auto", "loopback", "hostip"}:
        return mode

    logger.warning(f"Invalid {env_var} value '{mode}'; falling back to '{default}'")
    return default


def build_models_urls(
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


def wait_for_http_ready(
    models_urls: str | list[str],
    container,
    timeout_s: int,
    *,
    backend_name: str,
    log_dir: Path,
    primary_timeout_s: int = 60,
) -> str:
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
            log_path = _dump_container_logs(container, backend_name, log_dir, "exited")
            path_msg = f" Full logs written to {log_path}." if log_path else ""
            raise RuntimeError(
                f"{backend_name} container exited unexpectedly (status={container.status})."
                f"{path_msg}\nRecent logs:\n{logs}"
            )

        elapsed = time.time() - start_time
        if elapsed >= next_log_at:
            logger.info(
                f"Waiting for {backend_name} readiness... ({int(elapsed)}s / {timeout_s}s)"
            )
            next_log_at += 30.0

        time.sleep(2)

    logs = container.logs(tail=200).decode("utf-8", errors="replace")
    log_path = _dump_container_logs(container, backend_name, log_dir, "timeout")
    path_msg = f" Full logs written to {log_path}." if log_path else ""
    raise TimeoutError(
        f"Timed out waiting for {backend_name} readiness at {last_checked_url} "
        f"after {timeout_s}s.{path_msg}\nRecent logs:\n{logs}"
    )


def _dump_container_logs(container, backend_name: str, log_dir: Path, reason: str) -> Path | None:
    try:
        full_logs = container.logs().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning(f"Could not collect full {backend_name} container logs: {exc}")
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{container.name}-{timestamp}-{reason}.log"
    log_path.write_text(full_logs, encoding="utf-8")
    logger.info(f"Wrote full {backend_name} container logs to {log_path}")
    return log_path
