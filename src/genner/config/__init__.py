from typing import Optional
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for OpenAI-compatible inference servers."""

    name: str = "server"
    model: str = "qwen2.5-coder:7b-instruct"
    endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None
    # None means "omit max_tokens from the request and let the backend
    # use its own default" — for vLLM that is (max_model_len - prompt
    # tokens), which is the right semantic for "effectively uncapped".
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 300
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None


@dataclass
class VllmConfig(ServerConfig):
    """Configuration for vLLM-based models."""

    name: str = "vllm"
    gpu_memory_utilization: float = 0.85
    chat_template: Optional[str] = None
    enable_auto_tool_choice: bool = False
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None


@dataclass
class SglangServerConfig(ServerConfig):
    """Configuration for SGLang OpenAI-compatible serving."""

    name: str = "sglang"
    mem_fraction_static: float = 0.85
    chat_template: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    lora_routing_enabled: bool = False
    default_lora_name: Optional[str] = None


@dataclass
class LlamaConfig(ServerConfig):
    """Configuration for llama.cpp-based models."""

    name: str = "llama"
    n_gpu_layers: int = 99
    ctx_size: int = 8192
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    flash_attention: bool = False
    parallel_slots: Optional[int] = None
    kv_unified: Optional[bool] = None
    ctx_checkpoints: Optional[int] = None
    cache_ram_mib: Optional[int] = None
