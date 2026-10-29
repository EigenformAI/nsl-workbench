import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger
from openai import OpenAI

from src.backend.utils import get_recent_log_lines, is_http_ready, terminate_process
from src.genner import get_genner
from src.genner.config import LlamaConfig
from src.typing.config import AppConfig

from .session import BackendSession


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLAMA_CPP_TAG = "b8606"
LLAMA_DIR = PROJECT_ROOT / ".llama"
LLAMA_SERVER_BIN = LLAMA_DIR / "llama-server"
MODEL_CACHE_DIR = PROJECT_ROOT / ".model-cache"


LlamaBackendSession = BackendSession


def _is_gguf_model(model: str) -> bool:
    return "gguf" in model.lower() or model.lower().endswith(".gguf")


def _build_llama_config(
    app_config: AppConfig, endpoint: str
) -> LlamaConfig:
    if not app_config.model_name.startswith("llama:"):
        raise ValueError(
            f"setup_llama() requires a llama-prefixed model_name, got '{app_config.model_name}'"
        )

    config = LlamaConfig()
    config.model = app_config.model_name.split(":", 1)[1].strip()
    config.endpoint = endpoint
    config.timeout = app_config.inference.timeout
    config.temperature = app_config.inference.temperature
    config.max_tokens = app_config.inference.max_tokens
    config.frequency_penalty = app_config.inference.frequency_penalty
    config.presence_penalty = app_config.inference.presence_penalty
    if app_config.llama is not None:
        if app_config.llama.n_gpu_layers is not None:
            config.n_gpu_layers = app_config.llama.n_gpu_layers
        if app_config.llama.ctx_size is not None:
            config.ctx_size = app_config.llama.ctx_size
        config.cache_type_k = app_config.llama.cache_type_k
        config.cache_type_v = app_config.llama.cache_type_v
        config.flash_attention = app_config.llama.flash_attention
        config.parallel_slots = app_config.llama.parallel_slots
        config.kv_unified = app_config.llama.kv_unified
        config.ctx_checkpoints = app_config.llama.ctx_checkpoints
        config.cache_ram_mib = app_config.llama.cache_ram_mib
    return config


def _ensure_llama_server() -> Path:
    in_path = shutil.which("llama-server")
    if in_path:
        logger.info(f"llama-server found on PATH at {in_path}")
        return Path(in_path)

    if LLAMA_SERVER_BIN.exists():
        logger.info(f"llama-server found at {LLAMA_SERVER_BIN}")
        return LLAMA_SERVER_BIN

    logger.info(f"Downloading llama-server {LLAMA_CPP_TAG}...")
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine in ("x86_64", "amd64"):
        asset_name = f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz"
    else:
        raise RuntimeError(
            f"Unsupported platform: {system}/{machine}. "
            f"Install llama-cpp via your package manager (e.g., nixpkgs#llama-cpp) "
            f"or download a binary from "
            f"https://github.com/ggml-org/llama.cpp/releases/tag/{LLAMA_CPP_TAG} "
            f"and place it at {LLAMA_SERVER_BIN}"
        )

    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}/{asset_name}"
    archive_path = LLAMA_DIR / asset_name

    logger.info(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, archive_path)

    logger.info(f"Extracting {asset_name}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=LLAMA_DIR)

    extracted = list(LLAMA_DIR.glob("*/llama-server"))
    if not extracted:
        extracted = list(LLAMA_DIR.glob("llama-server"))
    if not extracted:
        raise RuntimeError("Could not find llama-server in extracted archive")

    src_bin = extracted[0]
    if src_bin != LLAMA_SERVER_BIN:
        src_bin.rename(LLAMA_SERVER_BIN)

    LLAMA_SERVER_BIN.chmod(0o755)
    archive_path.unlink(missing_ok=True)

    logger.info(f"llama-server ready at {LLAMA_SERVER_BIN}")
    return LLAMA_SERVER_BIN


def _download_gguf_model(repo_id: str) -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    repo_files = list_repo_files(repo_id=repo_id)
    gguf_files = [f for f in repo_files if f.endswith(".gguf")]

    if not gguf_files:
        raise RuntimeError(f"No GGUF files found in {repo_id}")

    # preferred_order = ["q4_k_m", "q5_k_m", "q8_0", "q4_k_s", "q5_k_s", "q2_k", "q3_k_m"]
    preferred_order = ["q8_0", "q4_k_m"]
    selected = None
    for suffix in preferred_order:
        for f in gguf_files:
            if suffix in f.lower():
                selected = f
                break
        if selected:
            break

    if not selected:
        selected = gguf_files[0]

    logger.info(f"Downloading GGUF file: {selected} from {repo_id}")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=selected,
        cache_dir=str(MODEL_CACHE_DIR),
    )
    logger.info(f"GGUF model cached at: {local_path}")
    return Path(local_path)


def _wait_for_llama_server_ready(
    models_url: str,
    process: subprocess.Popen,
    timeout_s: int,
    stderr_log_path: Optional[str] = None,
) -> None:
    deadline = time.time() + timeout_s
    last_output_size = 0
    last_output_change_time = time.time()
    freeze_threshold = 60

    while time.time() < deadline:
        if is_http_ready(models_url):
            return

        if process.poll() is not None:
            logs = ""
            if stderr_log_path:
                try:
                    logs = Path(stderr_log_path).read_text()[-2000:]
                except Exception:
                    pass
            raise RuntimeError(
                f"llama-server exited before becoming ready "
                f"(code={process.returncode}). Last logs:\n{logs}"
            )

        if stderr_log_path:
            try:
                current_size = Path(stderr_log_path).stat().st_size
                if current_size != last_output_size:
                    last_output_size = current_size
                    last_output_change_time = time.time()
                else:
                    seconds_silent = time.time() - last_output_change_time
                    if seconds_silent > freeze_threshold:
                        recent_lines = get_recent_log_lines(stderr_log_path, n=15)
                        logger.warning(
                            f"llama-server appears frozen (no output change for {seconds_silent:.0f}s). "
                            f"Recent log:\n{recent_lines}"
                        )
            except FileNotFoundError:
                pass

        time.sleep(1)

    raise TimeoutError(
        f"Timed out waiting for llama-server readiness at {models_url} after {timeout_s}s"
    )


def _build_llama_smoke_test(client: OpenAI, config: LlamaConfig):
    def run_smoke_test() -> str:
        test_response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Who are you?"}],
            max_tokens=50,
            temperature=0.5,
        )
        return test_response.choices[0].message.content

    return run_smoke_test


def _build_llama_session(
    client: OpenAI,
    config: LlamaConfig,
    base_url: str,
    models_url: str,
    stderr_log_path: Optional[str] = None,
    process: Optional[subprocess.Popen] = None,
) -> LlamaBackendSession:
    genner = get_genner("llama", server_config=config, oai_client=client)
    return LlamaBackendSession(
        genner=genner,
        smoke_test=_build_llama_smoke_test(client, config),
        client=client,
        config=config,
        base_url=base_url,
        models_url=models_url,
        stderr_log_path=stderr_log_path,
        process=process,
    )


@contextmanager
def setup_llama(
    app_config: AppConfig,
    *,
    endpoint: str = "http://localhost:8000",
) -> Iterator[LlamaBackendSession]:
    llama_cfg = app_config.llama
    startup_timeout = (llama_cfg.startup_timeout if llama_cfg and llama_cfg.startup_timeout else 500)
    config = _build_llama_config(app_config, endpoint=endpoint)

    resolved_lora_adapter_path = None
    if llama_cfg is not None and llama_cfg.lora_adapter_path:
        resolved_lora_adapter_path = (
            Path(llama_cfg.lora_adapter_path).expanduser().resolve()
        )
        if not resolved_lora_adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found: {resolved_lora_adapter_path}"
            )
        if not resolved_lora_adapter_path.is_file():
            raise ValueError(
                f"llama.cpp LoRA adapter must be a GGUF file: "
                f"{resolved_lora_adapter_path}"
            )

    configured_model_path = Path(config.model).expanduser()
    if configured_model_path.exists() and not configured_model_path.is_file():
        raise ValueError(
            f"Local llama.cpp model path must be a GGUF file: {configured_model_path}"
        )

    endpoint = config.endpoint.rstrip("/")
    base_url = endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"
    models_url = f"{base_url}/models"
    api_key = config.api_key or "dummy"
    if is_http_ready(models_url):
        local_model = configured_model_path.exists()
        if resolved_lora_adapter_path is not None or local_model:
            raise RuntimeError(
                f"A server is already listening at {base_url}, but local artifact "
                "serving requires a fresh framework-managed llama.cpp process"
            )
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)
        logger.info(f"Using existing llama.cpp server at {base_url}")
        yield _build_llama_session(client, config, base_url, models_url)
        return

    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000

    logger.info(
        f"No server detected at {base_url}. Starting llama.cpp backend for {config.model} on port {port}..."
    )

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)

    llama_bin = _ensure_llama_server()

    model_path_str = config.model
    if _is_gguf_model(config.model) and not Path(config.model).exists():
        repo_id = config.model
        if "/" not in repo_id:
            repo_id = f"unsloth/{repo_id}"
        local_gguf = _download_gguf_model(repo_id)
        model_path_str = str(local_gguf)
        logger.info(f"Using local GGUF: {model_path_str}")

    stderr_log = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"llama_stderr_{config.model.replace('/', '_')}_",
        suffix=".log",
        delete=False,
    )
    stderr_log_path = stderr_log.name
    stderr_log.close()
    logger.info(f"llama-server stderr log: {stderr_log_path}")

    command = [
        str(llama_bin),
        "--model",
        model_path_str,
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(config.ctx_size),
        "--temp",
        str(config.temperature),
        "--n-gpu-layers",
        str(config.n_gpu_layers),
        "--jinja",
    ]

    if config.cache_type_k is not None:
        command.extend(["--cache-type-k", config.cache_type_k])
    if config.cache_type_v is not None:
        command.extend(["--cache-type-v", config.cache_type_v])
    if config.flash_attention:
        command.extend(["--flash-attn", "on"])
    if config.parallel_slots is not None:
        command.extend(["--parallel", str(config.parallel_slots)])
    if config.kv_unified is not None:
        command.append("--kv-unified" if config.kv_unified else "--no-kv-unified")
    if config.ctx_checkpoints is not None:
        command.extend(["--ctx-checkpoints", str(config.ctx_checkpoints)])
    if config.cache_ram_mib is not None:
        command.extend(["--cache-ram", str(config.cache_ram_mib)])

    if resolved_lora_adapter_path is not None:
        command.extend(["--lora", str(resolved_lora_adapter_path)])

    logger.info(f"llama-server command: {' '.join(command)}")

    process = None
    try:
        with open(stderr_log_path, "a") as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )

        startup_timeout = max(config.timeout, startup_timeout)
        _wait_for_llama_server_ready(
            models_url, process, startup_timeout, stderr_log_path
        )

        logger.info(f"llama-server is ready at {base_url}")
        yield _build_llama_session(
            client,
            config,
            base_url,
            models_url,
            stderr_log_path=stderr_log_path,
            process=process,
        )
    finally:
        if process is not None:
            terminate_process(process, "llama-server")


__all__ = ["LlamaBackendSession", "setup_llama"]
