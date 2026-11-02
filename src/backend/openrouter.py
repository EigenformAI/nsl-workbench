import os
from contextlib import contextmanager
from typing import Iterator

from openai import OpenAI

from src.genner import VllmConfig, get_genner
from src.typing.config import AppConfig

from .session import BackendSession


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@contextmanager
def setup_openrouter(app_config: AppConfig) -> Iterator[BackendSession]:
    """Set up an OpenRouter model through its OpenAI-compatible API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required for OpenRouter backend"
        )

    model_name = app_config.model_name.removeprefix("openrouter:").strip()
    if not model_name:
        raise ValueError("OpenRouter model_name must include a provider/model name")

    config = VllmConfig(
        name="openrouter",
        model=model_name,
        endpoint=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=app_config.inference.timeout,
        max_tokens=app_config.inference.max_tokens,
        temperature=app_config.inference.temperature,
        frequency_penalty=app_config.inference.frequency_penalty,
        presence_penalty=app_config.inference.presence_penalty,
        seed=app_config.inference.seed,
        enable_auto_tool_choice=True,
    )
    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=app_config.inference.timeout,
        max_retries=3,
    )
    genner = get_genner("vllm", server_config=config, oai_client=client)

    yield BackendSession(
        genner=genner,
        smoke_test=lambda: "OpenRouter connection ready",
        client=client,
        config=config,
        base_url=OPENROUTER_BASE_URL,
        models_url=f"{OPENROUTER_BASE_URL}/models",
    )


__all__ = ["setup_openrouter"]
