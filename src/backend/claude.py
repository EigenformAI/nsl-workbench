import os
from contextlib import contextmanager
from typing import Iterator

import anthropic

from src.genner import ClaudeConfig, get_genner
from src.typing.config import AppConfig

from .session import BackendSession


@contextmanager
def setup_claude(app_config: AppConfig) -> Iterator[BackendSession]:
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is required for Claude backend"
        )

    claude_client = anthropic.Anthropic(api_key=claude_api_key)
    # The Anthropic API requires a positive max_tokens; the shared
    # InferenceConfig default is None ("uncapped") which is a vLLM-only
    # idiom. Fall back to 8192 (claude-sonnet-4-6's per-call output cap)
    # when the config doesn't specify one.
    claude_max_tokens = app_config.inference.max_tokens or 8192
    claude_config = ClaudeConfig(
        model="claude-sonnet-4-6",
        max_tokens=claude_max_tokens,
        temperature=app_config.inference.temperature,
    )
    genner = get_genner(
        "claude",
        claude_client=claude_client,
        claude_config=claude_config,
    )
    yield BackendSession(
        genner=genner,
        client=claude_client,
        config=claude_config,
    )


__all__ = ["setup_claude"]
