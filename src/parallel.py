"""Parallel episode execution infrastructure.

Thread-per-Slot pattern: each worker thread owns a dedicated WorkerSlot
with its own ContainerManager, DockerClient, and circuit breaker.
"""

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, List, Optional

import docker
from loguru import logger

from src.container import ContainerManager, compose_services
from src.typing.trajectory import EpisodeTrajectory, GenerationData

# Unconfigured Docker daemons carve /20 subnets and empirically afford ~31
# user bridge networks before `all predefined address pools have been fully
# subnetted`. Hosts with a custom `default-address-pools` in daemon.json (or
# `virtualisation.docker.daemon.settings` on NixOS) can have vastly more.
# `_pool_capacity_from_docker` asks docker for the real value and falls back
# to this heuristic if the query fails. Each slot claims NETWORKS_PER_SLOT.
_DEFAULT_POOL_CAPACITY_NETWORKS = 31
NETWORKS_PER_SLOT = 3


def _pool_capacity_from_docker() -> int:
    """Sum of user bridge subnets available from docker's configured
    default-address-pools. Falls back to the unconfigured-daemon heuristic
    when docker is unreachable or reports no pools (Moby's built-in defaults
    are not enumerated in `docker info`, so absence is indistinguishable
    from the default case)."""
    try:
        client = docker.from_env()
        info = client.info()
        pools = info.get("DefaultAddressPools") if isinstance(info, dict) else None
        if not pools or not isinstance(pools, list):
            return _DEFAULT_POOL_CAPACITY_NETWORKS
        total = 0
        for p in pools:
            base_prefix = int(p["Base"].split("/")[1])
            total += 2 ** (p["Size"] - base_prefix)
        return total or _DEFAULT_POOL_CAPACITY_NETWORKS
    except Exception as exc:
        logger.debug(
            f"Could not query docker default-address-pools ({exc}); "
            f"using heuristic {_DEFAULT_POOL_CAPACITY_NETWORKS}"
        )
        return _DEFAULT_POOL_CAPACITY_NETWORKS


@dataclass
class SlotCircuitBreaker:
    """Per-slot circuit breaker for container health.

    Preserves the sequential path's failure distinctions:
    - General failures (episode errors, unhandled exceptions)
    - Rebuild failures (container recovery failed)
    - Verification failures (systematic container setup issues)
    """

    max_consecutive_failures: int = 3
    max_consecutive_verification_failures: int = 0  # 0 = disabled

    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _consecutive_verification_failures: int = field(default=0, init=False, repr=False)
    _consecutive_environment_failures: int = field(default=0, init=False, repr=False)

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._consecutive_verification_failures = 0
        self._consecutive_environment_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1

    def record_verification_failure(self) -> None:
        self._consecutive_verification_failures += 1

    def record_environment_failure(self) -> None:
        self._consecutive_environment_failures += 1

    def record_environment_success(self) -> None:
        self._consecutive_environment_failures = 0

    def record_benign_abort(self) -> None:
        return None

    def is_tripped(self) -> bool:
        return (
            self._consecutive_failures >= self.max_consecutive_failures
            or self._consecutive_environment_failures
            >= self.max_consecutive_failures
        )

    def is_environment_tripped(self) -> bool:
        return (
            self._consecutive_environment_failures
            >= self.max_consecutive_failures
        )

    def is_verification_tripped(self) -> bool:
        if self.max_consecutive_verification_failures <= 0:
            return False
        return (
            self._consecutive_verification_failures
            >= self.max_consecutive_verification_failures
        )

    def reset(self) -> None:
        """Reset after a SUCCESSFUL rebuild."""
        self._consecutive_failures = 0
        self._consecutive_verification_failures = 0
        self._consecutive_environment_failures = 0


class StopReason:
    """Thread-safe first-writer-wins stop reason carrier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None

    def set(self, reason: str) -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason

    def get(self) -> str | None:
        with self._lock:
            return self._reason


class GlobalCircuitBreaker:
    """Trips only when a majority of slots are failing."""

    def __init__(
        self,
        slot_breakers: List[SlotCircuitBreaker],
        threshold: float = 0.5,
    ) -> None:
        self._slot_breakers = slot_breakers
        self._threshold = threshold

    def is_tripped(self) -> bool:
        if not self._slot_breakers:
            return False
        tripped_count = sum(1 for b in self._slot_breakers if b.is_tripped())
        return tripped_count / len(self._slot_breakers) >= self._threshold


@dataclass(frozen=True)
class RowCountState:
    total_episodes_run: int
    training_row_count: int
    training_row_count_is_exact: bool
    training_row_count_last_refreshed_episode: int


class ThreadSafeGenerationCollector:
    """Thread-safe wrapper for collecting episode results."""

    def __init__(
        self,
        generation_data: GenerationData,
        training_row_count_fn: Any | None = None,
    ) -> None:
        self._generation_data = generation_data
        self._training_row_count_fn = training_row_count_fn
        self._lock = threading.Lock()

    def _snapshot_generation_data_locked(self) -> GenerationData:
        return replace(
            self._generation_data,
            all_episodes=list(self._generation_data.all_episodes),
            successful_episodes=list(self._generation_data.successful_episodes),
            failed_episodes=list(self._generation_data.failed_episodes),
        )

    def snapshot(self) -> GenerationData:
        with self._lock:
            return self._snapshot_generation_data_locked()

    def add_episode(self, episode: EpisodeTrajectory) -> None:
        with self._lock:
            self._generation_data.add_episode(episode)
            if self._training_row_count_fn is None:
                self._generation_data.set_training_row_count(
                    self._generation_data.training_row_count,
                )
            else:
                self._generation_data.mark_training_row_count_stale()

    def refresh_training_row_count(self) -> int:
        if self._training_row_count_fn is None:
            with self._lock:
                self._generation_data.set_training_row_count(
                    self._generation_data.training_row_count,
                )
                return self._generation_data.training_row_count

        with self._lock:
            snapshot = self._snapshot_generation_data_locked()

        exact_count = int(self._training_row_count_fn(snapshot))

        with self._lock:
            raw_delta = max(
                0,
                self._generation_data.raw_successful_row_count
                - snapshot.raw_successful_row_count,
            )
            is_exact = (
                self._generation_data.total_episodes_run == snapshot.total_episodes_run
            )
            self._generation_data.set_training_row_count(
                exact_count if is_exact else exact_count + raw_delta,
                is_exact=is_exact,
                last_refreshed_episode=snapshot.total_episodes_run,
            )
            return self._generation_data.training_row_count

    def training_row_count(self) -> int:
        with self._lock:
            return self._generation_data.training_row_count

    def progress_snapshot(self) -> tuple[int, int]:
        with self._lock:
            return (
                self._generation_data.total_episodes_run,
                self._generation_data.training_row_count,
            )

    def should_stop(self, target_rows: int) -> bool:
        with self._lock:
            return self._generation_data.training_row_count >= target_rows

    def row_count_state(self) -> RowCountState:
        with self._lock:
            return RowCountState(
                total_episodes_run=self._generation_data.total_episodes_run,
                training_row_count=self._generation_data.training_row_count,
                training_row_count_is_exact=(
                    self._generation_data.training_row_count_is_exact
                ),
                training_row_count_last_refreshed_episode=(
                    self._generation_data.training_row_count_last_refreshed_episode
                ),
            )

@dataclass
class WorkerSlot:
    """A dedicated container slot owned by a single worker thread.

    ``harness_session`` persists across episodes within this slot. Each
    harness owns its own keys inside the dict.
    """

    slot_id: int
    container_manager: ContainerManager
    docker_client: object  # DockerClient
    circuit_breaker: SlotCircuitBreaker
    cache_dir: Path
    harness_session: dict[str, Any] = field(default_factory=dict)


_DEFAULT_SUBST_RE = re.compile(
    r"^\$\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)(?P<op>:?-)(?P<default>[^}]*)\}$"
)


def _is_relative_path_literal(value: str) -> bool:
    """True for compose path literals that resolve against the compose-file
    directory: ``.``, ``..``, or anything starting with ``./`` / ``../``.

    Compose's bind-vs-named-volume rule treats only these prefixes as host
    paths; bare identifiers (e.g. ``named-volume``) stay named volumes.
    Absolute (``/...``), home (``~``), and URL-style values are left alone.
    """
    return value in {".", ".."} or value.startswith(("./", "../"))


def _absolutize_path(value: str, anchor: Path) -> str:
    """Rewrite a possibly-substituted path to be absolute, anchored at *anchor*.

    Compose-file relative paths (``./foo``, ``../foo``) resolve against the
    compose file's directory. When the file is copied to a slot directory the
    anchor changes and the paths break. This rewrites the literal portion to
    an absolute path so the slot compose file is portable.

    Handles three shapes seen in our compose files:
      - plain relative path: ``.``, ``./foo``, ``../foo`` → absolute
      - substitution with relative default: ``${VAR:-../foo}`` →
        ``${VAR:-/abs/foo}``
      - substitution without default / absolute / named volume → unchanged
    """
    if not value:
        return value
    match = _DEFAULT_SUBST_RE.match(value)
    if match:
        default = match.group("default")
        if _is_relative_path_literal(default):
            absolute = str((anchor / default).resolve())
            return f"${{{match.group('var')}{match.group('op')}{absolute}}}"
        return value
    if _is_relative_path_literal(value):
        return str((anchor / value).resolve())
    return value


def _split_volume_short_syntax(spec: str) -> list[str]:
    """Split a compose short-syntax volume string on ``:`` while preserving
    ``${...}`` substitutions that may themselves contain ``:`` (e.g. ``:-``).
    """
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    i = 0
    while i < len(spec):
        char = spec[i]
        if char == "$" and i + 1 < len(spec) and spec[i + 1] == "{":
            depth += 1
            buffer.append(char)
            buffer.append(spec[i + 1])
            i += 2
            continue
        if char == "}" and depth > 0:
            depth -= 1
            buffer.append(char)
            i += 1
            continue
        if char == ":" and depth == 0:
            parts.append("".join(buffer))
            buffer = []
            i += 1
            continue
        buffer.append(char)
        i += 1
    parts.append("".join(buffer))
    return parts


def _rewrite_compose_paths(
    compose: dict,
    base_compose_dir: Path,
    slot_compose_dir: Path | None = None,
) -> None:
    """In-place rewrite of relative build contexts and bind-mount sources in
    a parsed compose dict, anchoring them at *base_compose_dir*.
    """
    anchor = base_compose_dir.resolve()
    services = compose.get("services")
    if not isinstance(services, dict):
        return

    for service in services.values():
        if not isinstance(service, dict):
            continue
        build = service.get("build")
        if isinstance(build, str):
            service["build"] = _absolutize_path(build, anchor)
        elif isinstance(build, dict):
            context = build.get("context")
            if isinstance(context, str):
                build["context"] = _absolutize_path(context, anchor)

        slot_local = set(service.pop("x-nsl-slot-local-bind-sources", []))
        volumes = service.get("volumes")
        if not isinstance(volumes, list):
            continue
        for idx, entry in enumerate(volumes):
            if isinstance(entry, str):
                parts = _split_volume_short_syntax(entry)
                if not parts:
                    continue
                source = parts[0]
                rewritten = _absolutize_path(source, anchor)
                if rewritten != source:
                    parts[0] = rewritten
                    volumes[idx] = ":".join(parts)
            elif isinstance(entry, dict):
                source = entry.get("source")
                if isinstance(source, str):
                    if source in slot_local and slot_compose_dir is not None:
                        local_source = (slot_compose_dir / source).resolve()
                        local_source.mkdir(parents=True, exist_ok=True)
                        entry["source"] = str(local_source)
                    else:
                        entry["source"] = _absolutize_path(source, anchor)


def _generate_slot_compose(
    base_compose_dir: Path,
    slot_compose_dir: Path,
    project_name: str,
    slot_id: int,
) -> None:
    """Generate a compose file from the base template with slot-specific names.

    Path rewriting: any relative ``build.context`` or volume source in the
    base compose file is rewritten to an absolute path anchored at
    *base_compose_dir*. Without this, copying the compose file under
    ``<generation_dir>/compose/slot_N/`` shifts the anchor of relative paths
    and breaks ``docker compose up --build`` for parent-relative contexts.
    """
    base_compose_file = base_compose_dir / "docker-compose.yml"
    if not base_compose_file.exists():
        base_compose_file = base_compose_dir / "compose.yml"
    if not base_compose_file.exists():
        yaml_files = list(base_compose_dir.glob("*.yml")) + list(
            base_compose_dir.glob("*.yaml")
        )
        if yaml_files:
            base_compose_file = yaml_files[0]
        else:
            raise FileNotFoundError(f"No compose file found in {base_compose_dir}")

    import yaml

    shutil.copytree(
        base_compose_dir,
        slot_compose_dir,
        dirs_exist_ok=True,
        symlinks=True,
    )

    compose = yaml.safe_load(base_compose_file.read_text()) or {}
    if not isinstance(compose, dict):
        raise ValueError(f"Compose file {base_compose_file} did not parse to a mapping")

    _rewrite_compose_paths(compose, base_compose_dir, slot_compose_dir)

    services = compose.get("services")
    if isinstance(services, dict):
        for service in services.values():
            if isinstance(service, dict):
                service.pop("container_name", None)

    slot_compose_path = slot_compose_dir / "docker-compose.yml"
    slot_compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))

    copied_compose_file = slot_compose_dir / base_compose_file.name
    if copied_compose_file != slot_compose_path and copied_compose_file.exists():
        copied_compose_file.unlink()


def _launch_slot_compose(
    slot_compose_dir: Path,
    project_name: str,
) -> None:
    """Launch a compose project for a slot."""
    compose_file = (slot_compose_dir / "docker-compose.yml").resolve()
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            project_name,
            "-f",
            str(compose_file),
            "up",
            "-d",
            "--build",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"docker compose up failed for {project_name}:\n{result.stderr}")
        result.check_returncode()


def estimate_slot_capacity(
    requested_n_slots: int,
    existing_user_networks: Optional[int] = None,
    pool_capacity: Optional[int] = None,
) -> int:
    """Upper bound on slots that can start without exhausting Docker's
    default address pools.

    If `existing_user_networks` is None, query docker for the current count
    of non-default bridge networks. If `pool_capacity` is None, derive it
    from docker's configured default-address-pools via
    `_pool_capacity_from_docker`.
    """
    if pool_capacity is None:
        pool_capacity = _pool_capacity_from_docker()

    if existing_user_networks is None:
        try:
            client = docker.from_env()
            nets = client.networks.list(filters={"driver": "bridge"})
            existing_user_networks = sum(
                1
                for n in nets
                if getattr(n, "name", None) not in {"bridge", "host", "none"}
            )
        except Exception as exc:
            logger.warning(
                f"Preflight network-capacity check failed ({exc}); assuming empty pool"
            )
            existing_user_networks = 0

    remaining = max(0, pool_capacity - existing_user_networks)
    affordable = remaining // NETWORKS_PER_SLOT
    if affordable <= 0:
        return 0
    return min(requested_n_slots, affordable)


def _cleanup_partial_slot(project_name: str, slot_compose_dir: Path) -> None:
    """Best-effort cleanup of a slot that failed to start. Never raises.

    `docker compose down` is a no-op against a nonexistent project and
    `shutil.rmtree(..., ignore_errors=True)` is a no-op against a missing
    dir, so this is safe to call regardless of how far setup progressed.
    """
    compose_file = slot_compose_dir / "docker-compose.yml"
    if compose_file.exists():
        try:
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "-p",
                    project_name,
                    "-f",
                    str(compose_file.resolve()),
                    "down",
                    "-v",
                    "--remove-orphans",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as exc:
            logger.debug(
                f"Cleanup of partial slot {project_name} compose down failed: {exc}"
            )
    else:
        # Dir may have no compose yet; still try a projectless down as a
        # safety net for any networks docker may have pre-created.
        try:
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "-p",
                    project_name,
                    "down",
                    "-v",
                    "--remove-orphans",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as exc:
            logger.debug(
                f"Cleanup of partial slot {project_name} projectless down failed: {exc}"
            )

    try:
        shutil.rmtree(slot_compose_dir, ignore_errors=True)
    except Exception as exc:
        logger.debug(f"Cleanup of partial slot {project_name} rmtree failed: {exc}")


def create_worker_slots(
    n_slots: int,
    base_compose_dir: Path,
    generation_dir: Path,
    run_id: str,
    code_host_cache_path: Path,
    post_rebuild_wait_seconds: int = 10,
    task=None,
) -> List[WorkerSlot]:
    """Create up to N isolated worker slots, each with its own compose project.

    Preflight: cap N against estimated Docker network-pool capacity.
    Fallback: on per-slot failure, clean up the partial slot and return the
    successful ones. Only re-raise when zero slots could be started.
    """
    effective_n_slots = estimate_slot_capacity(n_slots)
    if effective_n_slots < n_slots:
        logger.warning(
            f"Docker network pool preflight: capping slot count "
            f"{n_slots} -> {effective_n_slots} (existing networks on host "
            f"leave insufficient headroom for {n_slots * NETWORKS_PER_SLOT} "
            f"new networks). Configure daemon.json default-address-pools to "
            f"raise the ceiling."
        )
    if effective_n_slots == 0:
        # Attempt one slot so the surfaced error comes from docker itself
        effective_n_slots = 1

    slots: List[WorkerSlot] = []
    for i in range(effective_n_slots):
        project_name = f"gen_{run_id}_slot_{i}"
        slot_compose_dir = generation_dir / "compose" / f"slot_{i}"
        slot_cache_dir = code_host_cache_path / f"slot_{i}"
        try:
            _generate_slot_compose(base_compose_dir, slot_compose_dir, project_name, i)

            docker_client = docker.from_env()
            _launch_slot_compose(slot_compose_dir, project_name)

            expected_services = compose_services(
                slot_compose_dir / "docker-compose.yml",
                project_name=project_name,
            )
            container_manager = ContainerManager(
                docker_client=docker_client,
                container_ids=[],
                docker_compose_dir=str(slot_compose_dir),
                post_rebuild_wait_seconds=post_rebuild_wait_seconds,
                project_name_pattern=project_name,
                expected_services=expected_services,
                task=task,
            )
            container_manager.refresh_container_ids()
            logger.info(
                f"Slot {i} discovered containers: {container_manager.container_ids}"
            )

            slots.append(
                WorkerSlot(
                    slot_id=i,
                    container_manager=container_manager,
                    docker_client=docker_client,
                    circuit_breaker=SlotCircuitBreaker(),
                    cache_dir=slot_cache_dir,
                )
            )
        except Exception as exc:
            logger.error(f"Slot {i} creation failed ({type(exc).__name__}: {exc})")
            _cleanup_partial_slot(project_name, slot_compose_dir)
            if not slots:
                raise
            logger.warning(
                f"Proceeding with {len(slots)}/{n_slots} worker slots — "
                f"slot {i} failed to start; remaining slots skipped. "
                f"Throughput will be reduced."
            )
            return slots

    return slots


def teardown_worker_slots(
    slots: List[WorkerSlot],
    timeout_per_slot_s: float = 120,
) -> None:
    """Tear down all worker slot compose projects."""
    import subprocess

    for slot in slots:
        compose_dir = slot.container_manager.docker_compose_dir
        if compose_dir:
            project_name = slot.container_manager.project_name_pattern
            cmd = ["docker", "compose"]
            if project_name:
                cmd.extend(["-p", project_name])
            cmd.extend(["down", "--volumes", "--remove-orphans"])
            try:
                subprocess.run(
                    cmd,
                    cwd=compose_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_per_slot_s,
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Slot {slot.slot_id} teardown timed out after "
                    f"{timeout_per_slot_s}s — skipping"
                )
            except Exception as exc:
                logger.warning(f"Failed to teardown slot {slot.slot_id}: {exc}")
