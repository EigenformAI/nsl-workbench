"""Execute task-owned Python capabilities inside the agent container."""

from __future__ import annotations

from typing import Any

from docker.models.containers import Container

from src.task.base import TaskEnvironmentError
from tasks.common.container_exec import coerce_exec_result, exec_run_with_timeout


_DEFAULT_TIMEOUT_S = 60


def run_python_in_agent(
    container: Container,
    code: str,
    *,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a Python script inside the agent container.

    The script is written via heredoc to ``/tmp/_nsl_script.py`` then
    executed with ``python3 -u``. stderr is captured separately when
    ``demux=True`` is supported.

    Returns a dict with ``stdout``, ``stderr``, ``return_code``,
    ``success``, ``executed_code``.
    """
    write_cmd = (
        "cat > /tmp/_nsl_script.py <<'__NSL_PY_EOF__'\n"
        f"{code}\n"
        "__NSL_PY_EOF__"
    )
    try:
        write_result = exec_run_with_timeout(
            container, ["sh", "-c", write_cmd], timeout_s=15
        )
        write_code, write_out = coerce_exec_result(write_result)
        if write_code != 0:
            return {
                "stdout": "",
                "stderr": (
                    f"failed to stage script: {write_out.decode(errors='replace')}"
                ),
                "return_code": write_code,
                "success": False,
                "executed_code": code,
            }

        run_result = exec_run_with_timeout(
            container,
            ["python3", "-u", "/tmp/_nsl_script.py"],
            timeout_s=timeout_s,
            demux=True,
        )
    except TaskEnvironmentError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface to MCP client
        return {
            "stdout": "",
            "stderr": f"agent exec failed: {exc}",
            "return_code": -1,
            "success": False,
            "executed_code": code,
        }

    exit_code = getattr(run_result, "exit_code", None)
    if exit_code is None and isinstance(run_result, tuple):
        exit_code = run_result[0]
    raw_output = getattr(run_result, "output", None)
    if raw_output is None and isinstance(run_result, tuple):
        raw_output = run_result[1]

    if isinstance(raw_output, tuple) and len(raw_output) == 2:
        stdout_bytes, stderr_bytes = raw_output
        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
    else:
        stdout = (
            raw_output.decode(errors="replace")
            if isinstance(raw_output, (bytes, bytearray))
            else ""
        )
        stderr = ""

    return {
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "return_code": int(exit_code) if exit_code is not None else -1,
        "success": exit_code == 0,
        "executed_code": code,
    }


__all__ = ["run_python_in_agent"]
