from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.genner.Base import Genner


SmokeTest = Callable[[], str]


@dataclass
class BackendSession:
    genner: Genner
    smoke_test: SmokeTest | None = None
    client: Any = None
    config: Any = None
    base_url: str | None = None
    models_url: str | None = None
    stderr_log_path: str | None = None
    process: Any = None
    metrics_url: str | None = None  # Inference-server Prometheus /metrics endpoint
    extras: Any = None
