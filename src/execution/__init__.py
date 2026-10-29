from __future__ import annotations

from typing import Any

from src.execution.backend_runtime import (
    BackendRuntime,
    _coerce_runtime,
    open_backend_runtime,
)
from src.execution.episode import (
    EpisodeOutcome,
    EpisodeRequest,
    run_episode,
)
from src.execution.episode_runner import run_single_episode
from src.execution.generation import run_generation_sequential, save_generation_data
from src.execution.parallel import run_generation_parallel
from src.typing.config import AppConfig
from src.typing.trajectory import GenerationData


def run_generation(*args: Any, **kwargs: Any) -> GenerationData:
    target_rows = kwargs.pop("target_rows", None)
    max_episodes = kwargs.pop("max_episodes", None)

    if args and isinstance(args[0], BackendRuntime):
        rt = args[0]
        generation_id = kwargs.pop("generation_id", args[1] if len(args) > 1 else None)
        if generation_id is None:
            raise TypeError("run_generation requires generation_id")
    else:
        genner = kwargs.pop("genner", args[0] if len(args) > 0 else None)
        docker_client = kwargs.pop("docker_client", args[1] if len(args) > 1 else None)
        config = kwargs.pop("config", args[2] if len(args) > 2 else None)
        generation_id = kwargs.pop("generation_id", args[3] if len(args) > 3 else None)
        run_id = kwargs.pop("run_id", args[4] if len(args) > 4 else None)
        metrics_collector = kwargs.pop("metrics_collector", None)
        task = kwargs.pop("task", None)
        if None in (genner, docker_client, config, generation_id, run_id, task):
            raise TypeError("run_generation missing required legacy arguments")
        rt = _coerce_runtime(
            config=config,
            run_id=run_id,
            task=task,
            genner=genner,
            docker_client=docker_client,
            metrics=metrics_collector,
        )

    generation_config = rt.config.generation or AppConfig.GenerationConfig()
    if generation_config.parallel_episodes > 1:
        return run_generation_parallel(rt, generation_id=int(generation_id))
    return run_generation_sequential(
        rt,
        generation_id=int(generation_id),
        target_rows=target_rows,
        max_episodes=max_episodes,
    )


__all__ = [
    "BackendRuntime",
    "EpisodeOutcome",
    "EpisodeRequest",
    "open_backend_runtime",
    "run_episode",
    "run_generation",
    "run_generation_parallel",
    "run_generation_sequential",
    "run_single_episode",
    "save_generation_data",
]
