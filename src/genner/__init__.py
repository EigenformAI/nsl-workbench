from .config import (
    LlamaConfig,
    ServerConfig,
    SglangServerConfig,
    VllmConfig,
)
from .Base import Genner
from .OAI import OAIGenner
from .SglangServer import SglangServerGenner
from .Llama import LlamaGenner
from .Claude import ClaudeGenner, ClaudeConfig
from openai import OpenAI
import anthropic

__all__ = ["get_genner"]


class BackendException(Exception):
    pass


class ClaudeBackendException(Exception):
    pass


def get_genner(
    backend: str,
    server_config: ServerConfig | None = None,
    oai_client: OpenAI | None = None,
    claude_config: ClaudeConfig = ClaudeConfig(),
    claude_client: anthropic.Anthropic | None = None,
) -> Genner:
    """
    Get a genner instance based on the backend.

    Args:
        backend: The backend to use.
        server_config: Configuration for OpenAI-compatible server backends (vllm, llama).
        oai_client: OpenAI client (required for vllm/llama backends).
        claude_config: Configuration for the Claude backend.
        claude_client: Anthropic client (required for Claude backend).

    Returns:
        Genner: The genner instance.
    """
    available_backends = [
        "vllm",
        "sglang",
        "llama",
        "claude",
    ]

    if backend == "vllm":
        if not oai_client:
            raise Exception(
                "Using backend 'vllm', OpenAI client is required for vLLM backend"
            )
        if server_config is None:
            server_config = VllmConfig()
        return OAIGenner(oai_client, server_config, identifier="vllm")
    elif backend == "sglang":
        if not oai_client:
            raise Exception(
                "Using backend 'sglang', OpenAI client is required for SGLang backend"
            )
        if server_config is None:
            server_config = SglangServerConfig()
        return SglangServerGenner(oai_client, server_config)  # type: ignore[arg-type]
    elif backend == "llama":
        if not oai_client:
            raise Exception(
                "Using backend 'llama', OpenAI client is required for llama backend"
            )
        if server_config is None:
            server_config = LlamaConfig()
        return LlamaGenner(oai_client, server_config)
    elif backend == "claude":
        if not claude_client:
            raise ClaudeBackendException(
                "Using backend 'claude', Anthropic client is required for Claude backend"
            )
        return ClaudeGenner(claude_client, claude_config)

    raise BackendException(
        f"Unsupported backend: {backend}, available backends: {', '.join(available_backends)}"
    )
