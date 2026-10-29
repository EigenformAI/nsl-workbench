from __future__ import annotations

from pathlib import Path
from typing import Any

from docker.models.containers import Container as DockerContainer
from result import Err, Ok

from src.tool.docker import run_code_in_con, write_code_in_con


def run_python_in_container(
    docker_client: Any,
    agent_container: DockerContainer,
    host_cache_folder: Path,
    code: str,
    *,
    timeout: int,
) -> dict[str, Any]:
    in_container_script_path: str | None = None
    reflected_code = code

    try:
        in_container_script_path, reflected_code = write_code_in_con(
            docker_client,
            agent_container,
            host_cache_folder=host_cache_folder,
            code=code,
            postfix="mode",
            in_container_path="/tmp",
        )
    except Exception as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Failed to write code to container: {exc}",
            "return_code": -1,
            "executed_code": code,
        }

    try:
        execution_result = run_code_in_con(
            agent_container,
            in_container_script_path,
            timeout_seconds=timeout,
        )
    except Exception as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Container execution failed: {exc}",
            "return_code": -1,
            "executed_code": reflected_code,
        }
    finally:
        if in_container_script_path is not None:
            try:
                agent_container.exec_run(
                    cmd=["/bin/sh", "-c", f"rm -f {in_container_script_path}"],
                )
            except Exception:
                pass

    match execution_result:
        case Ok(execution_output):
            return {
                "success": execution_output.exit_code == 0,
                "stdout": execution_output.stdout.strip(),
                "stderr": execution_output.stderr.strip(),
                "return_code": execution_output.exit_code,
                "executed_code": reflected_code,
            }
        case Err(error_message):
            return {
                "success": False,
                "stdout": "",
                "stderr": error_message,
                "return_code": -1,
                "executed_code": reflected_code,
            }

    return {
        "success": False,
        "stdout": "",
        "stderr": "Unexpected container execution result",
        "return_code": -1,
        "executed_code": reflected_code,
    }
