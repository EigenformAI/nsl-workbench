from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from src.typing.config import AppConfig

from .claude import setup_claude
from .llama import setup_llama
from .openrouter import setup_openrouter
from .sglang import setup_sglang
from .vllm import setup_vllm


BackendContextFactory = Callable[[AppConfig], AbstractContextManager[Any]]


_BACKEND_CONTEXT_FACTORIES: dict[str, BackendContextFactory] = {
    "claude": setup_claude,
    "llama": setup_llama,
    "openrouter": setup_openrouter,
    "sglang": setup_sglang,
    "vllm": setup_vllm,
}


def get_backend_context_factory(model_name: str) -> BackendContextFactory | None:
    normalized_model_name = model_name.strip()
    if normalized_model_name in _BACKEND_CONTEXT_FACTORIES:
        return _BACKEND_CONTEXT_FACTORIES[normalized_model_name]

    prefix = normalized_model_name.split(":", 1)[0]
    if prefix in _BACKEND_CONTEXT_FACTORIES:
        return _BACKEND_CONTEXT_FACTORIES[prefix]

    return None


def resolve_backend_context(
    app_config: AppConfig,
) -> AbstractContextManager[Any] | None:
    factory = get_backend_context_factory(app_config.model_name)
    if factory is None:
        return None
    return factory(app_config)
