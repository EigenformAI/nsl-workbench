from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from docker import DockerClient
from docker.models.containers import Container as DockerContainer
from loguru import logger

from src.tool.docker import wait_and_get_container


# docker-py's Model.id returns attrs.get("Id"), typed Optional[str]. For a
# Container obtained from the daemon, attrs["Id"] is always populated — a
# None here would mean a framework invariant was violated. Patch the
# property to raise at read time so downstream code (task lifecycle,
# TaskEnvironmentError, PopulationResult) can treat container.id as str
# AT RUNTIME.
#
# Note for LSP/pyright: static type checkers do not execute this module, so
# they still see container.id as Optional[str]. Call sites that pass it to a
# typed `str` boundary (List[str] literal, typed param, Dict[str, ...] key)
# must still narrow locally — prefer `assert container.id is not None` right
# after the `for container in containers:` header. f-string interpolation and
# `container.id or ...` forms do not need narrowing.
def _validated_container_id(self: DockerContainer) -> str:
    cid = self.attrs.get("Id")
    if cid is None:
        state = self.attrs.get("State", {})
        status = state.get("Status", "?") if isinstance(state, dict) else "?"
        raise RuntimeError(
            f"Docker Container has no id (name={self.name!r}, status={status!r})"
        )
    return cid


DockerContainer.id = property(_validated_container_id)  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.task.base import TaskSpec
    from src.task.types import PopulationOutcome, Variation

# Default timeouts (seconds) for Docker operations
EXEC_TIMEOUT_S = 120
COMPOSE_TIMEOUT_S = 300


def resolve_compose_file(docker_compose_dir: str | Path) -> Path:
    compose_dir = Path(docker_compose_dir)
    default_compose = compose_dir / "docker-compose.yml"
    candidates = [default_compose, compose_dir / "compose.yml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    yaml_files = sorted(compose_dir.glob("*.yml")) + sorted(compose_dir.glob("*.yaml"))
    if yaml_files:
        return yaml_files[0]
    return default_compose


def _compose_command_prefix(
    compose_file: Path,
    project_name: str | None = None,
) -> list[str]:
    command = ["docker", "compose"]
    if project_name:
        command.extend(["-p", project_name])
    command.extend(["-f", str(compose_file)])
    return command


def compose_services(
    compose_file: Path,
    project_name: str | None = None,
) -> list[str]:
    result = subprocess.run(
        [*_compose_command_prefix(compose_file, project_name), "config", "--services"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def project_name_from_compose(compose_file: Path) -> str:
    result = subprocess.run(
        [*_compose_command_prefix(compose_file), "config", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = json.loads(result.stdout or "{}")
    if isinstance(resolved, dict):
        name = resolved.get("name")
        if isinstance(name, str) and name:
            return name

    if compose_file.parent.name:
        return compose_file.parent.name
    raise ValueError(f"Could not determine compose project name for {compose_file}")


def _parse_compose_ps_output(stdout: str) -> list[dict[str, Any]]:
    stripped = stdout.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("docker compose ps --format json did not return a list")
        return [entry for entry in parsed if isinstance(entry, dict)]

    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def container_to_service(container: DockerContainer) -> str:
    labels = {}
    attrs = getattr(container, "attrs", {})
    if isinstance(attrs, dict):
        config = attrs.get("Config", {})
        if isinstance(config, dict):
            raw_labels = config.get("Labels", {})
            if isinstance(raw_labels, dict):
                labels = raw_labels
        if not labels:
            raw_labels = attrs.get("Labels", {})
            if isinstance(raw_labels, dict):
                labels = raw_labels

    if not labels:
        raw_labels = getattr(container, "labels", {})
        if isinstance(raw_labels, dict):
            labels = raw_labels

    service = labels.get("com.docker.compose.service")
    if not service:
        raise ValueError(
            f"Container {container.name or container.id} has no "
            "com.docker.compose.service label"
        )
    return str(service)


class DockerExecTimeoutError(TimeoutError):
    """Raised when a container exec_run exceeds its timeout."""


def _exec_run_with_timeout(
    container: DockerContainer,
    cmd: Any,
    timeout_s: float = EXEC_TIMEOUT_S,
    **kwargs: Any,
) -> Any:
    """Run container.exec_run with a timeout.

    If the exec does not complete within *timeout_s* seconds the call raises
    ``DockerExecTimeoutError``.  The underlying Docker exec may still be
    running inside the container but the calling thread is unblocked.
    """
    result: list[Any] = [None]
    exception: list[BaseException | None] = [None]

    def _run() -> None:
        try:
            result[0] = container.exec_run(cmd, **kwargs)
        except BaseException as exc:
            exception[0] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise DockerExecTimeoutError(
            f"Docker exec timed out after {timeout_s}s on "
            f"{container.name or container.id}: {cmd}"
        )
    if exception[0] is not None:
        raise exception[0]
    return result[0]


@dataclass
class ContainerPopulationResult:
    container_id: str
    variation_name: str
    description: str
    expected_kb: int
    success: bool
    error_message: Optional[str] = None


class ContainerBrokenError(Exception):
    """Raised when baseline measurement fails for one or more containers."""

    def __init__(self, message: str, broken_ids: List[str]) -> None:
        super().__init__(message)
        self.broken_ids = list(broken_ids)


class ContainerManager:
    def __init__(
        self,
        docker_client: DockerClient,
        container_ids: List[str],
        docker_compose_dir: Optional[str] = None,
        post_rebuild_wait_seconds: int = 10,
        project_name_pattern: Optional[str] = None,
        expected_services: Optional[List[str]] = None,
        task: Optional[TaskSpec] = None,
    ) -> None:
        self.docker_client = docker_client
        self.container_ids = list(container_ids)
        self.docker_compose_dir = docker_compose_dir
        self.post_rebuild_wait_seconds = post_rebuild_wait_seconds
        self.project_name_pattern = project_name_pattern
        self.expected_services = (
            list(expected_services) if expected_services is not None else None
        )
        self.task = task

    def get_containers(self) -> List[DockerContainer]:
        return [
            wait_and_get_container(self.docker_client, container_id)
            for container_id in self.container_ids
        ]

    def populate_with_task(
        self,
        containers: List[DockerContainer],
        variation: Variation,
    ) -> tuple[PopulationOutcome, bool]:
        """Task-aware population: reset → populate → verify.

        Delegates environment lifecycle to self.task.

        Returns:
            Tuple of (PopulationOutcome, verified: bool). The caller
            should abort the episode if any PopulationResult.success is
            False or if verified is False.

            The caller should use ``outcome.episode_context`` as the
            authoritative episode context for downstream methods —
            it is produced by the task's ``populate()`` and may contain
            task-specific data (e.g. baseline measurements).

        Raises:
            RuntimeError: if no task is set on this ContainerManager.
            TaskEnvironmentError: if reset fails unrecoverably.
        """
        if self.task is None:
            raise RuntimeError(
                "populate_with_task requires a TaskSpec on ContainerManager"
            )
        self.task.reset(containers)
        outcome = self.task.populate(containers, variation)

        # Fail fast on partial population failures
        any_failed = any(not r.success for r in outcome.results)
        if any_failed:
            return outcome, False

        verified = self.task.verify_population(
            containers,
            variation,
            outcome.episode_context,
            private_context=outcome.private_context,
        )
        return outcome, verified

    def services(self) -> dict[str, DockerContainer]:
        out: dict[str, DockerContainer] = {}
        for container_id in self.container_ids:
            container = self.docker_client.containers.get(container_id)
            out[container_to_service(container)] = container
        return out

    def get_service(self, service_name: str) -> DockerContainer:
        services = self.services()
        if service_name not in services:
            raise KeyError(
                f"Service {service_name!r} not in project {self.project_name_pattern}. "
                f"Available: {sorted(services)}"
            )
        return services[service_name]

    def refresh_container_ids(self) -> List[str]:
        if not self.project_name_pattern:
            raise ValueError("project_name_pattern is required")
        self._require_compose_dir()

        assert self.docker_compose_dir is not None
        compose_file = resolve_compose_file(self.docker_compose_dir)
        result = subprocess.run(
            [
                *(_compose_command_prefix(compose_file, self.project_name_pattern)),
                "ps",
                "--format",
                "json",
                "--all",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        entries = _parse_compose_ps_output(result.stdout)

        if self.expected_services is not None:
            expected = list(self.expected_services)
            if not expected:
                raise RuntimeError(
                    f"Compose project {self.project_name_pattern} declared zero services"
                )
            discovered = {str(entry["Service"]) for entry in entries}
            missing = set(expected) - discovered
            unexpected = discovered - set(expected)
            if missing or unexpected:
                logger.error(
                    f"Compose project {self.project_name_pattern}: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
                raise RuntimeError(
                    f"Compose project {self.project_name_pattern}: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            entries.sort(key=lambda entry: expected.index(str(entry["Service"])))

        self.container_ids = [str(entry["Name"]) for entry in entries]
        logger.debug(
            f"Discovered {len(entries)} containers for project "
            f"{self.project_name_pattern}: {self.container_ids}"
        )
        return list(self.container_ids)

    def rebuild(self) -> None:
        self._require_compose_dir()
        self._run_compose(["docker", "compose", "down"])
        self._run_compose(["docker", "compose", "up", "-d", "--build"])
        self.refresh_container_ids()
        self._install_procps()
        if not self.verify_ready():
            raise RuntimeError("Containers failed readiness check after rebuild")

    def rebuild_containers(self, broken_ids: List[str]) -> None:
        self._require_compose_dir()
        if not broken_ids:
            return
        requested_ids = list(dict.fromkeys(broken_ids))
        if set(requested_ids) == set(self.container_ids):
            self.rebuild()
            return

        target_services: List[str] = []
        for container_id in requested_ids:
            service_name = self._container_to_service(container_id)
            if service_name not in target_services:
                target_services.append(service_name)
            self._run_compose(
                [
                    "docker",
                    "compose",
                    "up",
                    "-d",
                    "--build",
                    "--force-recreate",
                    "--no-deps",
                    service_name,
                ]
            )

        refreshed_container_ids = self.refresh_container_ids()
        rebuilt_container_ids = [
            container_id
            for container_id in refreshed_container_ids
            if self._container_to_service(container_id) in target_services
        ]
        self._install_procps_for(rebuilt_container_ids)
        for container_id in rebuilt_container_ids:
            if not self.verify_container_ready(container_id):
                raise RuntimeError(
                    f"Container {container_id} failed readiness check after rebuild"
                )

    def restart(self) -> None:
        self._require_compose_dir()
        self._run_compose(["docker", "compose", "down"])
        self._run_compose(["docker", "compose", "up", "-d"])
        self.refresh_container_ids()
        self._install_procps()
        if not self.verify_ready():
            raise RuntimeError("Containers failed readiness check after restart")

    def verify_ready(self) -> bool:
        return all(
            self.verify_container_ready(container_id)
            for container_id in self.container_ids
        )

    def verify_container_ready(self, container_id: str, timeout: int = 30) -> bool:
        try:
            container = wait_and_get_container(
                self.docker_client,
                container_id,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning(f"Container {container_id} failed to become ready: {exc}")
            return False

        health = container.attrs.get("State", {}).get("Health", {}).get("Status")
        if health == "unhealthy":
            logger.warning(f"Container {container_id} healthcheck reports unhealthy")
            return False
        return True

    def _install_procps(self) -> None:
        self._install_procps_for(self.container_ids)

    def _install_procps_for(self, container_ids: List[str]) -> None:
        for container_id in container_ids:
            container = self.docker_client.containers.get(container_id)
            _exec_run_with_timeout(
                container,
                ["sh", "-c", "apk add --no-cache procps"],
                stdout=False,
                stderr=False,
            )

    def _container_to_service(self, container_id: str) -> str:
        container = self.docker_client.containers.get(container_id)
        return container_to_service(container)

    def _run_compose(
        self,
        command: List[str],
        timeout_s: float = COMPOSE_TIMEOUT_S,
    ) -> None:
        # Inject -p <project_name> after "docker compose" so that
        # down/up/build target the correct compose project.
        if self.project_name_pattern and "-p" not in command:
            idx = command.index("compose") + 1
            command = command[:idx] + ["-p", self.project_name_pattern] + command[idx:]
        subprocess.run(
            command,
            cwd=self.docker_compose_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
        )

    def _require_compose_dir(self) -> None:
        if not self.docker_compose_dir:
            raise ValueError("docker_compose_dir is required for compose operations")

    @staticmethod
    def _coerce_exec_result(exec_result: Any) -> tuple[int, bytes]:
        if hasattr(exec_result, "exit_code") and hasattr(exec_result, "output"):
            return int(exec_result.exit_code), bytes(exec_result.output)
        if isinstance(exec_result, tuple) and len(exec_result) >= 2:
            return int(exec_result[0]), bytes(exec_result[1])
        raise TypeError(f"Unsupported exec result type: {type(exec_result)!r}")
