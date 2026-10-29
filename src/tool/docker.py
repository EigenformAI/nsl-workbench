import io
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, cast

from loguru import logger
from result import Err, Ok, Result

import docker
import docker.errors
from docker import DockerClient
from docker.models.containers import Container as DockerContainer
from src.helper import timeout


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int


def write_code_in_con(
    client: DockerClient,
    container: DockerContainer,
    host_cache_folder: Path,
    code: str,
    postfix: str,
    in_container_path: str = "/",
) -> Tuple[str, str]:
    """Write code into a temporary file in the host machine first then to the container.

    Algorithm:
    - Write code into a temporary file in the host machine
    - Create a tar archive containing the file
    - Copy the tar archive to the container's root directory
    - Check if the file exists in the container

    Args:
        code (str): The code to write into the container
        postfix (str): The type identifier for the agent, used in the file path
        in_container_path (str, optional): The base path in the container to write the code to. Defaults to "/".

    Raises:
        Exception: If the file cannot be written to the container or if verification fails

    Returns:
        Tuple[str, str]:
            - The path to the temporary file in the container
            - The reflected code (content of the file as read from the container)
    """
    # Create temp file name with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file_name = f"temp_script_{current_time}.py"
    temp_file_path = f"{in_container_path}/{temp_file_name}"

    # Create host file path and ensure directory exists
    # logger.info(f"Writing file {temp_file_name} into host machine")
    host_path = host_cache_folder / temp_file_name
    host_path.parent.mkdir(parents=True, exist_ok=True)
    host_path.write_text(code)

    # Create a tar archive in memory
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(host_path, arcname=temp_file_name)
    tar_stream.seek(0)

    # Copy the file to the container's root directory
    # logger.info(f"Writing file {temp_file_name} into container")

    succeed = container.put_archive(path=in_container_path, data=tar_stream.read())

    if not succeed:
        raise Exception("Failed to write code into the container")

    # Check if file exists in container
    check_exist_command = (
        f"test -f {temp_file_path} && echo 'File exists' || echo 'File does not exist'"
    )
    check_exist_result = container.exec_run(cmd=["/bin/sh", "-c", check_exist_command])

    if b"File exists" not in check_exist_result.output:
        logger.error(
            f"File verification failed: {check_exist_result.output.decode('utf-8')}"
        )
        raise Exception(
            f"File verification failed: {check_exist_result.output.decode('utf-8')}"
        )

    # Read the file content
    reflected_code = container.exec_run(cmd=["cat", temp_file_path]).output.decode(
        "utf-8"
    )
    assert isinstance(reflected_code, str)

    return temp_file_path, reflected_code


def run_code_in_con(
    container: DockerContainer,
    in_container_script_path: str,
    timeout_seconds: int = 600,
) -> Result[ExecutionResult, str]:
    """Run a Python script path inside a container and return its output.

    Args:
        container: The container to run the code in.
        in_container_script_path: The in-container path to the Python script.
        timeout_seconds: Maximum execution time before failure.

    Returns:
        Result[ExecutionResult, str]:
            - Ok: Execution result with stdout, stderr, and exit code.
            - Err: Error details when execution fails or times out.

    Note:
        - After execution, any remaining Python processes are killed.
    """
    command_str = f"python -u {in_container_script_path}"
    cmd = ["/bin/sh", "-c", command_str]  # Execute via shell
    timeout_bool = False
    timeout_checker = time.time()
    try:
        with timeout(seconds=timeout_seconds):
            python_exit_code, python_output = cast(
                Tuple[int, Tuple[bytes | None, bytes | None]],
                container.exec_run(
                    cmd=cmd,
                    demux=True,
                    stream=False,  # Wait for the command to finish and return all output at once
                ),
            )
            if time.time() - timeout_checker > timeout_seconds:
                timeout_bool = True
            stdout_bytes, stderr_bytes = python_output
            stdout = (
                stdout_bytes.decode("utf-8", errors="replace")
                if stdout_bytes is not None
                else ""
            )
            stderr = (
                stderr_bytes.decode("utf-8", errors="replace")
                if stderr_bytes is not None
                else ""
            )
    except TimeoutError as e:
        return Err(f"ContainerManager.run_code_in_con: Code ran too long, error: \n{e}")
    except docker.errors.ContainerError as e:
        return Err(f"ContainerManager.run_code_in_con: Container error, error: \n{e}")
    if timeout_bool:
        return Err(
            "ContainerManager.run_code_in_con: Code ran too long, output: \n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    container.exec_run(cmd="kill -9 $(pidof python)")

    return Ok(ExecutionResult(stdout=stdout, stderr=stderr, exit_code=python_exit_code))


def wait_and_get_container(
    client: DockerClient, container_name: str, timeout: int = 30
) -> DockerContainer:
    """Wait for container to be fully running and return the container object."""
    start_time = time.time()
    terminal_states = {"exited", "dead"}

    while time.time() - start_time < timeout:
        try:
            container = client.containers.get(container_name)
            container.reload()

            if container.status in terminal_states:
                logs = ""
                try:
                    logs_output = container.logs(tail=20)
                    logs = (
                        logs_output.decode("utf-8", errors="replace")
                        if isinstance(logs_output, bytes)
                        else str(logs_output)
                    ).strip()
                except Exception:
                    logs = "<unavailable>"

                detail = f" Last logs:\n{logs}" if logs else ""
                raise RuntimeError(
                    f"Container {container_name} is in terminal state '{container.status}'.{detail}"
                )

            if container.status == "running":
                # Test if Docker exec is usable without requiring procps in the image.
                exit_code, output = container.exec_run(["sh", "-c", "true"])
                if exit_code == 0:
                    logger.info(f"Container {container_name} is fully running")
                    return container

            logger.debug(f"Container status: {container.status}, waiting...")
            time.sleep(1)

        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} not found yet, waiting...")
            time.sleep(1)
            continue

    raise TimeoutError(
        f"Container {container_name} did not start properly within {timeout} seconds"
    )
