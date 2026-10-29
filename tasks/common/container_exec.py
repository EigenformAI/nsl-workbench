"""Shared Docker-exec result normalization and wall-clock timeout helpers."""

from __future__ import annotations

import threading
from typing import Any

from docker.models.containers import Container

from src.task.base import TaskEnvironmentError


EXEC_TIMEOUT_S = 120


def exec_run_with_timeout(
    container: Container,
    cmd: Any,
    timeout_s: float = EXEC_TIMEOUT_S,
    **kwargs: Any,
) -> Any:
    """Run container.exec_run with a wall-clock timeout.

    If the exec does not complete within timeout_s seconds, raises
    TaskEnvironmentError. The underlying Docker exec may still be running
    inside the container; only the calling thread is unblocked.
    """
    result: list[Any] = [None]
    exception: list[BaseException | None] = [None]

    def _run() -> None:
        try:
            result[0] = container.exec_run(cmd, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — re-raised
            exception[0] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TaskEnvironmentError(
            f"Docker exec timed out after {timeout_s}s on "
            f"{container.name or container.id}: {cmd}",
            container_ids=[container.id] if container.id else None,
        )
    if exception[0] is not None:
        raise exception[0]
    return result[0]


def coerce_exec_result(exec_result: Any) -> tuple[int, bytes]:
    """Normalize docker-py exec_run return shapes into (exit_code, output)."""
    if hasattr(exec_result, "exit_code") and hasattr(exec_result, "output"):
        exit_code = exec_result.exit_code if exec_result.exit_code is not None else -1
        output = exec_result.output if exec_result.output is not None else b""
        return int(exit_code), bytes(output)
    if isinstance(exec_result, tuple) and len(exec_result) >= 2:
        exit_code = exec_result[0] if exec_result[0] is not None else -1
        output = exec_result[1] if exec_result[1] is not None else b""
        return int(exit_code), bytes(output)
    raise TypeError(f"Unsupported exec result type: {type(exec_result)!r}")
