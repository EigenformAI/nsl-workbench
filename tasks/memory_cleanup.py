"""Memory Cleanup Task — reference implementation.

Agents operate inside Alpine Linux containers and attempt to free
disk space by identifying and removing unnecessary files.

Extracted from the original hardcoded task logic in:
- src/container.py (variations, reset, populate, verify)
- src/mode_prompts.py (prompts)
- src/mode_parsers.py (mode output parsing)
- src/tool/docker.py (measurement helpers)
- scripts/run_episode.py (pre/post measurement, reward)
"""

import re
import secrets
import shlex
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from docker.models.containers import Container
from loguru import logger

from src.container import container_to_service
from src.task.base import TaskEnvironmentError, TaskSpec
from src.task.types import (
    BudgetConstraints,
    Capability,
    CapabilityExecutionContext,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
    EpisodeConstraints,
    PopulationOutcome,
    PopulationResult,
    StepConstraints,
    SuccessConstraints,
    TaskPromptSpec,
    TaskReward,
    Variation,
    Workflow,
    WorkflowStep,
)
from src.training_data.transforms import PerEpisodeRowCap, TrainingDataTransform
from tasks.common.agent_exec import run_python_in_agent


# ---------------------------------------------------------------------------
# Task-specific data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisposableObject:
    path: str
    size_kb: int


@dataclass
class MemoryCleanupVariation(Variation):
    """A structured fixture from which population shell is rendered."""

    disposable_objects: tuple[DisposableObject, ...] = ()

    @property
    def created_dirs(self) -> tuple[str, ...]:
        return tuple(sorted({item.path.rsplit("/", 1)[0] for item in self.disposable_objects}))

    @property
    def expected_kb(self) -> float:
        return float(sum(item.size_kb for item in self.disposable_objects))


@dataclass(frozen=True)
class ViabilityProbe:
    name: str
    command: str
    integrity: bool = False


@dataclass(frozen=True)
class ProbeResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class MemoryCleanupBaseline:
    service_name: str
    container_id: str
    instance_seed: int
    variation_name: str
    theta: float
    disposable_paths: tuple[str, ...]
    disposable_roots: tuple[str, ...]
    viability_probes: tuple[ViabilityProbe, ...]
    disposable_count: int
    populated_allocated_kb: float
    baseline_processes: tuple[str, ...]


@dataclass(frozen=True)
class MemoryCleanupState:
    baseline: MemoryCleanupBaseline
    fraction_cleared: float
    allocated_kb: float | None  # diagnostic only; None when the du probe failed
    viability_results: tuple[ProbeResult, ...]
    integrity_ok: bool
    process_set_ok: bool


# ---------------------------------------------------------------------------
# Docker exec timeout helper (absorbed from container.py)
# ---------------------------------------------------------------------------

_EXEC_TIMEOUT_S = 120
_MEASUREMENT_ATTEMPTS = 3
_MEASUREMENT_RETRY_BACKOFF_S = 0.25
_INTEGRITY_ANCHORS = (
    "/usr/sbin/sshd",
    "/etc/ssh/ssh_host_ed25519_key",
    "/usr/local/bin/python3",
    "/usr/local/bin/pip",
)


def _exec_run_with_timeout(
    container: Container,
    cmd: Any,
    timeout_s: float = _EXEC_TIMEOUT_S,
    **kwargs: Any,
) -> Any:
    """Run container.exec_run with a timeout.

    Harness measurement/probe execs use absolute paths, so they must not inherit
    the container's image ``WorkingDir`` (``/home/alice``): an agent can delete
    its own home during a cleanup episode, after which docker's ``chdir`` into a
    missing ``WorkingDir`` makes every subsequent exec fail with exit 127. Default
    to ``workdir="/"`` (a directory the agent cannot remove) so a destroyed home
    is *scored* as a viability failure rather than crashing the measurement.
    """
    kwargs.setdefault("workdir", "/")
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
        raise TaskEnvironmentError(
            f"Docker exec timed out after {timeout_s}s on "
            f"{container.name or container.id}: {cmd}",
            container_ids=[container.id] if container.id else None,
        )
    if exception[0] is not None:
        raise exception[0]
    return result[0]


def _coerce_exec_result(exec_result: Any) -> tuple[int, bytes]:
    if hasattr(exec_result, "exit_code") and hasattr(exec_result, "output"):
        exit_code = exec_result.exit_code if exec_result.exit_code is not None else -1
        output = exec_result.output if exec_result.output is not None else b""
        return int(exit_code), bytes(output)
    if isinstance(exec_result, tuple) and len(exec_result) >= 2:
        exit_code = exec_result[0] if exec_result[0] is not None else -1
        output = exec_result[1] if exec_result[1] is not None else b""
        return int(exit_code), bytes(output)
    raise TypeError(f"Unsupported exec result type: {type(exec_result)!r}")


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------


class MemoryCleanupTask(TaskSpec[MemoryCleanupState]):
    """Clear disk space in Linux containers.

    Agents operate inside Alpine Linux containers and attempt to free
    disk space by identifying and removing unnecessary files.

    Recommended action budget: 12 (default).
    """

    name = "memory-cleanup"
    description = "Free disk space in Linux containers by removing unnecessary files"
    metric_name = "fraction_cleared"
    metric_unit = "fraction"
    higher_is_better = True
    agent_service_name = "service-a"

    def __init__(self, task_config: dict[str, Any]) -> None:
        self._task_config = dict(task_config)
        self._docker_compose_dir = task_config.get(
            "docker_compose_dir", "docker/special-learn-compose"
        )
        self._cleanup_paths = task_config.get(
            "cleanup_paths",
            ["/tmp", "/var/log", "/var/cache", "/var/tmp", "/home/alice"],
        )
        self._validate_theta_config(task_config)
        self.theta = float(task_config.get("theta", task_config.get("theta_base", 0.05)))
        max_rows_per_episode = task_config.get("max_rows_per_episode")
        self._training_data_transforms: tuple[TrainingDataTransform, ...] = ()
        if max_rows_per_episode is not None:
            self._training_data_transforms = (
                PerEpisodeRowCap(
                    max_rows=max_rows_per_episode,
                    keep=task_config.get("keep", "random"),
                ),
            )

    def tracking_config(self) -> dict[str, Any]:
        max_rows_per_episode = self._task_config.get("max_rows_per_episode")
        return {
            "theta": self.theta,
            "theta_schedule": str(self._task_config.get("theta_schedule", "constant")),
            "theta_base": float(self._task_config.get("theta_base", 0.05)),
            "theta_step": float(self._task_config.get("theta_step", 0.0)),
            "theta_cap": float(self._task_config.get("theta_cap", 0.95)),
            "max_rows_per_episode": max_rows_per_episode,
            "row_cap_keep": (
                self._task_config.get("keep", "random")
                if max_rows_per_episode is not None
                else None
            ),
        }

    def training_data_transforms(self) -> tuple[TrainingDataTransform, ...]:
        return self._training_data_transforms

    @staticmethod
    def _validate_theta_config(config: dict[str, Any]) -> None:
        schedule = config.get("theta_schedule", "constant")
        if schedule not in {"constant", "linear", "explicit"}:
            raise ValueError(f"Unsupported theta_schedule: {schedule!r}")
        for key, default in (("theta_base", 0.05), ("theta_cap", 0.95)):
            value = float(config.get(key, default))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be between 0 and 1")
        theta = float(config.get("theta", config.get("theta_base", 0.05)))
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be between 0 and 1")
        if float(config.get("theta_step", 0.0)) < 0:
            raise ValueError("theta_step must be non-negative")
        explicit = config.get("theta_explicit")
        if schedule == "explicit" and (
            not isinstance(explicit, list) or not explicit
        ):
            raise ValueError("theta_explicit must be a non-empty list")
        if explicit is not None and any(not 0.0 <= float(value) <= 1.0 for value in explicit):
            raise ValueError("theta_explicit values must be between 0 and 1")

    @classmethod
    def resolve_generation_task_config(
        cls,
        task_config: dict[str, Any],
        generation_id: int,
        num_generations: int,
    ) -> dict[str, Any]:
        del num_generations
        cls._validate_theta_config(task_config)
        resolved = dict(task_config)
        schedule = resolved.get("theta_schedule", "constant")
        base = float(resolved.get("theta_base", 0.05))
        cap = float(resolved.get("theta_cap", 0.95))
        if schedule == "constant":
            theta = float(resolved.get("theta", base))
        elif schedule == "linear":
            theta = base + generation_id * float(resolved.get("theta_step", 0.0))
        else:
            explicit = resolved["theta_explicit"]
            if generation_id >= len(explicit):
                raise ValueError(
                    "theta_explicit does not contain generation "
                    f"{generation_id}"
                )
            theta = float(explicit[generation_id])
        resolved["theta"] = min(cap, theta)
        return resolved

    @property
    def docker_compose_dir(self) -> str:
        return self._docker_compose_dir

    def episode_constraints(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> EpisodeConstraints:
        return EpisodeConstraints(
            budgets=BudgetConstraints(
                max_task_tool_calls=15,
                max_llm_turns=18,
            ),
            success=SuccessConstraints(min_task_tool_calls_for_success=1),
            step_overrides={
                "plan_cleanup": StepConstraints(
                    success=SuccessConstraints(min_task_tool_calls_for_success=0),
                ),
            },
        )

    # --- Variations ---

    def list_variations(self) -> list[Variation]:
        def objects(
            groups: tuple[tuple[str, str, int, int, str], ...]
        ) -> tuple[DisposableObject, ...]:
            return tuple(
                DisposableObject(f"{root}/{prefix}_{index}{suffix}", size_kb)
                for root, prefix, count, size_kb, suffix in groups
                for index in range(1, count + 1)
            )

        definitions = (
            ("variation_1_heavy", "Heavy mix across cleanup categories", (("/tmp/cleanup", "temp", 40, 100, ".tmp"), ("/var/log/app", "app", 30, 100, ".log"), ("/var/cache/pkg", "cache", 25, 100, ".dat"), ("/tmp/db_temp", "table", 15, 200, ".tmp"), ("/home/alice/trash", "deleted", 20, 50, ".bak"), ("/var/tmp/build", "obj", 10, 200, ".o"))),
            ("variation_2_medium", "Balanced file sizes and counts", (("/tmp/work", "work", 50, 50, ".tmp"), ("/var/log/system", "sys", 40, 50, ".log"), ("/var/cache/app", "app", 35, 75, ".cache"), ("/tmp/downloads", "download", 15, 150, ".part"), ("/home/alice/.cache", "thumb", 30, 20, ".png"), ("/var/tmp/sql", "query", 20, 75, ".tmp"))),
            ("variation_3_many_small", "Numerous small files", (("/tmp/fragments", "frag", 200, 10, ".tmp"), ("/var/log/debug", "debug", 100, 20, ".log"), ("/var/cache/thumbnails", "thumb", 150, 10, ".jpg"), ("/tmp/sessions", "sess", 80, 5, ".lock"), ("/home/alice/temp", "temp", 60, 30, ".dat"), ("/var/tmp/locks", "lock", 100, 5, ".pid"))),
            ("variation_4_large_sparse", "Few large files", (("/tmp/backups", "backup", 5, 1024, ".tar"), ("/var/log/archives", "archive", 8, 500, ".log.gz"), ("/var/cache/packages", "pkg", 10, 300, ".deb"), ("/tmp/exports", "export", 6, 500, ".sql"), ("/home/alice/old", "old", 12, 200, ".bak"), ("/var/tmp/dumps", "dump", 4, 500, ".sql"))),
            ("variation_5_mixed_realistic", "Realistic mixed scenario", (("/tmp/app_temp", "temp", 25, 150, ".tmp"), ("/var/log/nginx", "access", 20, 150, ".log"), ("/var/cache/apt", "pkg", 15, 300, ".deb"), ("/tmp/.build", "obj", 30, 75, ".o"), ("/home/alice/.local/share/Trash", "file", 18, 100, ".old"), ("/var/tmp/mysql", "tmp_table", 10, 200, ".ibd"))),
        )
        return [
            MemoryCleanupVariation(name=name, description=description, disposable_objects=objects(groups))
            for name, description, groups in definitions
        ]

    # --- Prompts ---

    def prompt_spec(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> TaskPromptSpec:
        return TaskPromptSpec(
            system_instruction=(
                "You are a Linux storage administrator working inside a "
                "networked container. Your goal is to free non-volatile "
                "storage space by removing unnecessary files."
            ),
            capabilities=[
                Capability(
                    name="run_python",
                    description=(
                        "Execute a Python script inside the agent container. "
                        "Use Python's standard library + os/shutil/subprocess "
                        "to inspect or clean the filesystem. Returns stdout, "
                        "stderr, and exit code."
                    ),
                    runs_code=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python source to execute.",
                            },
                            "timeout_s": {
                                "type": "integer",
                                "description": (
                                    "Wall-clock timeout (seconds). Default 60."
                                ),
                            },
                        },
                        "required": ["code"],
                    },
                ),
                Capability(
                    name="report_result",
                    description=(
                        "Report the cleanup outcome: how much space was freed "
                        "and a short verification summary. Invoking this signals "
                        "that cleanup is complete."
                    ),
                    publishes_metric=True,
                    schema={
                        "type": "object",
                        "properties": {
                            "space_freed_kb": {
                                "type": "number",
                                "description": "Disk space freed, in KB.",
                            },
                            "operation_type": {
                                "type": "string",
                                "description": "e.g. deletion / compression.",
                            },
                        },
                        "required": ["space_freed_kb"],
                    },
                ),
            ],
        )

    def workflow(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> Workflow | None:
        cleanup_paths = ", ".join(self._cleanup_paths)
        return Workflow(
            steps=(
                WorkflowStep(
                    name="explore_container",
                    description="Scan the filesystem to understand disk usage",
                    prompt=(
                        "Explore the container filesystem to understand current disk usage. "
                        f"Inspect the cleanup paths ({cleanup_paths}) "
                        "and identify the largest files and directories, their sizes, "
                        "and what appears safe to remove. "
                        "Build a complete picture of space usage before making any changes. "
                        "Do not delete anything yet.\n"
                        "Write PYTHON ONLY — your code runs through the Python "
                        "interpreter. Use os, shutil, and subprocess.run([...]). "
                        "Raw shell commands (du, df, ls, find) are NOT valid Python "
                        "and will fail with SyntaxError; call them via "
                        "subprocess.run([\"du\", \"-sh\", path], capture_output=True, "
                        "text=True) or use os/shutil equivalents (os.scandir, "
                        "os.path.getsize)."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("run_python",),
                    is_entry=True,
                    next_steps=("plan_cleanup",),
                    context_mode="inherit",
                ),
                WorkflowStep(
                    name="plan_cleanup",
                    description="Reason through the cleanup strategy without executing",
                    prompt=(
                        "Based on your exploration findings, plan your cleanup strategy. "
                        "Reason through: which files and directories to remove, in what order, "
                        "what the expected space savings are, and any risks to avoid. "
                        "Do not use any tools — this is a reasoning step only. "
                        "Write out your complete plan before proceeding."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=(),
                    next_steps=("execute_cleanup",),
                    context_mode="inherit",
                ),
                WorkflowStep(
                    name="execute_cleanup",
                    description="Execute the cleanup plan and verify results",
                    prompt=(
                        "Execute your cleanup plan by removing the files and "
                        "directories you identified as safe to delete. "
                        "After cleanup, verify the freed space by measuring "
                        "disk usage in the cleanup paths again, then report the "
                        "space you freed.\n"
                        "Write PYTHON ONLY — your code runs through the Python "
                        "interpreter. Delete with shutil.rmtree(path) / os.remove(path) "
                        "and measure with os.path.getsize / os.scandir. Raw shell "
                        "commands (rm, du, df) are NOT valid Python and will fail "
                        "with SyntaxError; if you must shell out, use "
                        "subprocess.run([\"rm\", \"-rf\", path], check=True). Do not "
                        "use sudo (you already have the needed permissions)."
                    ),
                    inherit_all_capabilities=False,
                    capabilities=("run_python", "report_result"),
                    terminator_capabilities=("report_result",),
                    next_steps=(),
                    context_mode="inherit",
                ),
            )
        )

    # --- Response parsing ---

    def parse_response(
        self,
        raw_response: str,
        *,
        invoked_capability: str | None = None,
    ) -> list[CapabilityInvocation]:
        """Extract metric reports from the agent's response.

        ``invoked_capability`` is either a workflow *step name* (native
        workflow execution — the real WHAT-side identity) or a harness phase
        tag (legacy flat-loop modes). Both are task-owned strings:

        - ``execute_cleanup`` step → emit the ``report_result`` terminator
          capability (native workflow path).
        - ``explorer`` mode / ``None`` → emit the legacy metric echo.
        """
        payload = self._parse_explorer_output(raw_response)
        if not payload:
            return []
        if invoked_capability == "execute_cleanup":
            return [CapabilityInvocation(name="report_result", input=payload)]
        if invoked_capability in (None, "explorer"):
            name = (
                "explorer_metric_report"
                if invoked_capability == "explorer"
                else "metric_report"
            )
            return [CapabilityInvocation(name=name, input=payload)]
        return []

    def execute_capability(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: Variation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        if invocation.name in {
            "explorer_metric_report",
            "metric_report",
            "report_result",
        }:
            return CapabilityResult(
                name=invocation.name,
                output=invocation.input,
                success=True,
            )
        if invocation.name != "run_python":
            return super().execute_capability(invocation, containers, variation, ctx)
        code = invocation.input.get("code")
        if not isinstance(code, str) or not code.strip():
            return CapabilityResult(
                name=invocation.name,
                output={},
                success=False,
                error="run_python requires a non-empty 'code' string",
            )
        timeout_s = int(invocation.input.get("timeout_s") or 60)
        target = self._resolve_target(containers)
        out = run_python_in_agent(target, code, timeout_s=timeout_s)
        return CapabilityResult(
            name=invocation.name,
            output=out,
            success=bool(out["success"]),
            error=None if out["success"] else (out.get("stderr") or None),
        )

    @staticmethod
    def _parse_explorer_output(raw_response: str) -> dict[str, Any]:
        """Extract space_freed_kb and operation classification."""
        space_freed_kb = 0.0
        matched_metric = False

        # Look for explicit KB mentions
        kb_match = re.search(r"(\d+\.?\d*)\s*KB", raw_response, re.IGNORECASE)
        if kb_match:
            space_freed_kb = float(kb_match.group(1))
            matched_metric = True
        else:
            # Look for MB mentions and convert to KB
            mb_match = re.search(r"(\d+\.?\d*)\s*MB", raw_response, re.IGNORECASE)
            if mb_match:
                space_freed_kb = float(mb_match.group(1)) * 1024
                matched_metric = True
            else:
                # Look for GB mentions and convert to KB
                gb_match = re.search(r"(\d+\.?\d*)\s*GB", raw_response, re.IGNORECASE)
                if gb_match:
                    space_freed_kb = float(gb_match.group(1)) * 1024 * 1024
                    matched_metric = True
                else:
                    freed_match = re.search(
                        r"freed\s+(\d+\.?\d*)", raw_response, re.IGNORECASE
                    )
                    if freed_match:
                        space_freed_kb = float(freed_match.group(1))
                        matched_metric = True

        if not matched_metric:
            return {}

        result: dict[str, Any] = {"space_freed_kb": space_freed_kb}

        # Classify operation type
        if re.search(r"\b(deleted|removed|cleaned)\b", raw_response, re.IGNORECASE):
            result["operation_type"] = "deletion"
        elif re.search(r"\b(compressed|archived)\b", raw_response, re.IGNORECASE):
            result["operation_type"] = "compression"

        return result

    # --- Environment Lifecycle ---

    def reset(self, containers: list[Container]) -> None:
        """Validate the target; exact fixture cleanup occurs inside populate."""
        self._resolve_target(containers)

    def populate(
        self,
        containers: list[Container],
        variation: Variation,
    ) -> PopulationOutcome:
        if not isinstance(variation, MemoryCleanupVariation):
            raise TypeError("MemoryCleanupTask requires MemoryCleanupVariation")
        target = self._resolve_target(containers)
        assert target.id is not None
        nonce = secrets.token_hex(16)
        script = self._render_population_script(variation, nonce)
        try:
            raw = _exec_run_with_timeout(target, ["sh", "-c", script])
            exit_code, output = _coerce_exec_result(raw)
        except TaskEnvironmentError:
            raise
        except Exception as exc:
            raise TaskEnvironmentError(
                f"Population exec failed on {target.id}: {exc}",
                container_ids=[target.id],
            ) from exc

        text = output.decode(errors="replace")
        failure = re.search(r"^__POPULATE_FAILED__=(.+)$", text, re.MULTILINE)
        if exit_code != 0 or failure:
            detail = failure.group(1) if failure else text.strip() or "(no output)"
            raise TaskEnvironmentError(
                f"Population command failed on {target.id} (exit {exit_code}): {detail}",
                container_ids=[target.id],
            )
        try:
            baseline_kb = self._sentinel_number(text, "__BASELINE_KB__")
            populated_kb = self._sentinel_number(text, "__POPULATED_KB__")
            populated_count = int(self._sentinel_number(text, "__POPULATED_COUNT__"))
        except ValueError as exc:
            raise TaskEnvironmentError(
                f"Malformed population sentinels on {target.id}: {exc}",
                container_ids=[target.id],
            ) from exc
        if f"__POPULATE_COMPLETE__={nonce}" not in text:
            raise TaskEnvironmentError(
                f"Missing population completion nonce on {target.id}",
                container_ids=[target.id],
            )
        if populated_count != len(variation.disposable_objects):
            raise TaskEnvironmentError(
                f"Population count mismatch on {target.id}: "
                f"{populated_count} != {len(variation.disposable_objects)}",
                container_ids=[target.id],
            )

        anchor_candidates = [
            line for line in text.splitlines() if "__ANCHOR_SHA256__=" in line
        ]
        anchor_matches = [
            match.groups()
            for line in anchor_candidates
            if (
                match := re.fullmatch(r"__ANCHOR_SHA256__=([^:]+):([0-9a-f]{64})", line)
            )
        ]
        matched_paths = [path for path, _ in anchor_matches]
        anchor_hashes = dict(anchor_matches)
        expected_paths = set(_INTEGRITY_ANCHORS)
        duplicate_paths = sorted(
            {path for path in matched_paths if matched_paths.count(path) > 1}
        )
        missing_paths = sorted(expected_paths - set(matched_paths))
        unexpected_paths = sorted(set(matched_paths) - expected_paths)
        malformed_count = len(anchor_candidates) - len(anchor_matches)
        if (
            len(anchor_candidates) != len(_INTEGRITY_ANCHORS)
            or malformed_count
            or duplicate_paths
            or missing_paths
            or unexpected_paths
        ):
            candidate_preview = [line[:256] for line in anchor_candidates[:8]]
            raise TaskEnvironmentError(
                f"Malformed integrity anchor sentinels on {target.id}: "
                f"missing={missing_paths}, unexpected={unexpected_paths}, "
                f"duplicate={duplicate_paths}, malformed={malformed_count}, "
                f"candidate_count={len(anchor_candidates)}, "
                f"candidates={candidate_preview!r}",
                container_ids=[target.id],
            )
        probes = self._viability_probes(anchor_hashes)
        baseline = MemoryCleanupBaseline(
            service_name=self.agent_service_name,
            container_id=target.id,
            instance_seed=secrets.randbits(63),
            variation_name=variation.name,
            theta=self.theta,
            disposable_paths=tuple(item.path for item in variation.disposable_objects),
            disposable_roots=variation.created_dirs,
            viability_probes=probes,
            disposable_count=len(variation.disposable_objects),
            populated_allocated_kb=populated_kb,
            baseline_processes=tuple(
                sorted(
                    line.split("=", 1)[1]
                    for line in text.splitlines()
                    if line.startswith("__BASELINE_PROCESS__=")
                )
            ),
        )
        return PopulationOutcome(
            results=[
                PopulationResult(
                    container_id=target.id,
                    variation_name=variation.name,
                    description=variation.description,
                    success=True,
                    details={
                        "disposable_count": baseline.disposable_count,
                        "populated_allocated_kb": baseline.populated_allocated_kb,
                    },
                )
            ],
            episode_context={"variation_name": variation.name},
            private_context={"memory_cleanup_baseline": baseline},
        )

    def verify_population(
        self,
        containers: list[Container],
        variation: Variation,
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> bool:
        del episode_context
        target = self._resolve_target(containers)
        baseline = self._baseline(private_context)
        self._validate_baseline(baseline, target, variation)
        usable, _, integrity_ok, _ = self._run_probe_script(target, baseline)
        absent = self._count_absent(target, baseline.disposable_paths)
        self._measure_population_kb(target, list(baseline.disposable_roots))
        if not usable or not integrity_ok or absent != 0:
            raise TaskEnvironmentError(
                f"Population viability admission failed on {target.id}",
                container_ids=[target.id] if target.id else None,
            )
        return True

    # --- Measurement ---

    def measure_initial_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> MemoryCleanupState:
        del episode_context
        target = self._resolve_target(containers)
        baseline = self._baseline(private_context)
        self._validate_baseline(baseline, target)
        usable, results, integrity_ok, _ = self._run_probe_script(
            target, baseline
        )
        if not usable or not integrity_ok:
            raise TaskEnvironmentError(
                f"Initial viability measurement failed on {target.id}",
                container_ids=[target.id] if target.id else None,
            )
        return MemoryCleanupState(
            baseline=baseline,
            fraction_cleared=0.0,
            allocated_kb=baseline.populated_allocated_kb,
            viability_results=results,
            integrity_ok=True,
            process_set_ok=True,
        )

    def measure_final_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
    ) -> MemoryCleanupState:
        del episode_context, artifacts
        target = self._resolve_target(containers)
        baseline = self._baseline(private_context)
        self._validate_baseline(baseline, target)
        usable, results, integrity_ok, processes = self._run_probe_script(
            target, baseline
        )
        absent = self._count_absent(target, baseline.disposable_paths)
        allocated_kb = self._measure_population_kb(
            target, list(baseline.disposable_roots)
        )
        expected_processes = set(baseline.baseline_processes)
        process_set_ok = not expected_processes or set(processes) <= expected_processes
        return MemoryCleanupState(
            baseline=baseline,
            fraction_cleared=absent / baseline.disposable_count,
            allocated_kb=allocated_kb,
            viability_results=results,
            integrity_ok=integrity_ok,
            process_set_ok=process_set_ok and usable,
        )

    # --- Reward (pure) ---

    def compute_reward(
        self,
        initial: MemoryCleanupState,
        final: MemoryCleanupState,
        artifacts: EpisodeArtifacts,
    ) -> TaskReward:
        """Pure viability, integrity, and effort admission."""
        if initial.baseline != final.baseline:
            raise ValueError("initial and final memory-cleanup baselines do not match")
        probe_pass = all(result.passed for result in final.viability_results)
        usable = probe_pass and final.process_set_ok
        effort = final.fraction_cleared >= final.baseline.theta
        allocated_freed = (
            max(0.0, initial.allocated_kb - final.allocated_kb)
            if initial.allocated_kb is not None and final.allocated_kb is not None
            else None
        )
        return TaskReward(
            value=final.fraction_cleared,
            success=usable and final.integrity_ok and effort,
            breakdown={
                "fraction_cleared": final.fraction_cleared,
                "theta": final.baseline.theta,
                "usable": usable,
                "integrity_ok": final.integrity_ok,
                "effort_ok": effort,
                "process_set_ok": final.process_set_ok,
                "allocated_kb_freed": allocated_freed,
                "initial_allocated_kb": initial.allocated_kb,
                "final_allocated_kb": final.allocated_kb,
                "viability_probes": {
                    result.name: result.passed for result in final.viability_results
                },
            },
        )

    # --- Private helpers ---

    def _resolve_target(self, containers: list[Container]) -> Container:
        matches: list[Container] = []
        invalid_ids: list[str] = []
        for container in containers:
            try:
                if container_to_service(container) == self.agent_service_name:
                    matches.append(container)
            except ValueError:
                if container.id:
                    invalid_ids.append(container.id)
        if len(matches) != 1:
            ids = [container.id for container in matches if container.id] or invalid_ids
            raise TaskEnvironmentError(
                f"Expected exactly one {self.agent_service_name!r} target; "
                f"found {len(matches)}",
                container_ids=ids,
            )
        if not matches[0].id:
            raise TaskEnvironmentError("Resolved memory-cleanup target has no ID")
        return matches[0]

    @staticmethod
    def _baseline(
        private_context: dict[str, Any] | None,
    ) -> MemoryCleanupBaseline:
        baseline = (private_context or {}).get("memory_cleanup_baseline")
        if not isinstance(baseline, MemoryCleanupBaseline):
            raise TaskEnvironmentError("Missing immutable memory-cleanup baseline")
        return baseline

    def _validate_baseline(
        self,
        baseline: MemoryCleanupBaseline,
        target: Container,
        variation: Variation | None = None,
    ) -> None:
        if baseline.container_id != target.id or baseline.service_name != self.agent_service_name:
            raise TaskEnvironmentError(
                "Memory-cleanup baseline target identity mismatch",
                container_ids=[target.id] if target.id else None,
            )
        if variation is not None and baseline.variation_name != variation.name:
            raise TaskEnvironmentError(
                "Memory-cleanup baseline variation identity mismatch",
                container_ids=[target.id] if target.id else None,
            )

    @staticmethod
    def _viability_probes(
        anchor_hashes: dict[str, str],
    ) -> tuple[ViabilityProbe, ...]:
        probes = [
            ViabilityProbe("sshd", "pgrep -x sshd >/dev/null && /usr/sbin/sshd -T >/dev/null"),
            ViabilityProbe("python_venv", "python3 -c 'import sys; raise SystemExit(0 if sys.prefix == \"/home/alice/venv\" else 1)'", True),
            ViabilityProbe("alice_shell", "test -d /home/alice && id alice >/dev/null && su - alice -c true", True),
            ViabilityProbe("sshd_binary", "test -x /usr/sbin/sshd && test -d /etc/ssh && test -d /run/sshd", True),
            ViabilityProbe("python_links", "test -L /usr/local/bin/python3 && test -L /usr/local/bin/pip", True),
        ]
        probes.extend(
            ViabilityProbe(
                f"anchor_sha256:{path}",
                "test \"$(sha256sum -- "
                f"{shlex.quote(path)} | awk '{{print $1}}')\" = "
                f"{shlex.quote(digest)}",
                True,
            )
            for path, digest in anchor_hashes.items()
        )
        return tuple(probes)

    @staticmethod
    def _sentinel_number(text: str, name: str) -> float:
        matches = re.findall(rf"^{re.escape(name)}=([0-9]+(?:\.[0-9]+)?)$", text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"expected one {name} sentinel, found {len(matches)}")
        return float(matches[0])

    @staticmethod
    def _du_shell(paths: tuple[str, ...], sentinel: str) -> str:
        quoted = " ".join(shlex.quote(path) for path in paths)
        return (
            "set --\n"
            f"for d in {quoted}; do [ -e \"$d\" ] && set -- \"$@\" \"$d\"; done\n"
            f"if [ $# -eq 0 ]; then echo '{sentinel}=0'; else "
            "out=$(du -sk \"$@\" 2>/dev/null) || exit $?; "
            f"printf '%s\\n' \"$out\" | awk '{{t+=$1}} END{{printf \"{sentinel}=%d\\n\", t+0}}'; fi\n"
        )

    def _render_population_script(
        self, variation: MemoryCleanupVariation, nonce: str
    ) -> str:
        paths = tuple(item.path for item in variation.disposable_objects)
        owned_roots = tuple(sorted({
            root
            for candidate in self.list_variations()
            if isinstance(candidate, MemoryCleanupVariation)
            for root in candidate.created_dirs
        }))
        lines = ["set -u", "command -v du >/dev/null || exit 127"]
        for root in owned_roots:
            quoted_root = shlex.quote(root)
            lines.extend(
                (
                    f"mkdir -p -- {quoted_root} || exit $?",
                    f"find {quoted_root} -mindepth 1 -maxdepth 1 "
                    "-exec rm -rf -- {} + || exit $?",
                )
            )
        lines.append(self._du_shell(variation.created_dirs, "__BASELINE_KB__"))
        for root, prefix, first, last, suffix, size_kb in self._group_objects(
            variation.disposable_objects
        ):
            lines.extend(
                (
                    f"for i in $(seq {first} {last}); do",
                    "path=$(printf '%s/%s_%s%s' "
                    f"{shlex.quote(root)} {shlex.quote(prefix)} \"$i\" "
                    f"{shlex.quote(suffix)})",
                    f"dd if=/dev/zero of=\"$path\" bs=1024 count={size_kb} "
                    "status=none || { rc=$?; "
                    "echo \"__POPULATE_FAILED__=$i:$path\"; exit $rc; }",
                    "done",
                )
            )
        lines.append(f"echo '__POPULATED_COUNT__={len(paths)}'")
        lines.append(self._du_shell(variation.created_dirs, "__POPULATED_KB__"))
        lines.append(
            "ps -eo comm= 2>/dev/null | sort -u | "
            "sed 's/^/__BASELINE_PROCESS__=/'"
        )
        for path in _INTEGRITY_ANCHORS:
            quoted = shlex.quote(path)
            failure_marker = shlex.quote(f"__POPULATE_FAILED__=anchor:{path}")
            sentinel_prefix = shlex.quote(f"__ANCHOR_SHA256__={path}:")
            lines.append(
                f"hash_output=$(sha256sum -- {quoted}) || {{ rc=$?; "
                f"printf '%s\\n' {failure_marker}; exit \"$rc\"; }}"
            )
            lines.append("hash=${hash_output%% *}")
            lines.append(f"printf '%s%s\\n' {sentinel_prefix} \"$hash\"")
        lines.append(f"echo '__POPULATE_COMPLETE__={nonce}'")
        return "\n".join(lines)

    @staticmethod
    def _group_objects(
        objects: tuple[DisposableObject, ...],
    ) -> tuple[tuple[str, str, int, int, str, int], ...]:
        grouped: dict[tuple[str, str, str, int], list[int]] = {}
        for item in objects:
            match = re.fullmatch(r"(.+)/([^/]+)_(\d+)([^/]*)", item.path)
            if match is None:
                raise ValueError(f"Disposable path is not structured: {item.path}")
            root, prefix, index, suffix = match.groups()
            grouped.setdefault((root, prefix, suffix, item.size_kb), []).append(
                int(index)
            )
        result: list[tuple[str, str, int, int, str, int]] = []
        for (root, prefix, suffix, size_kb), indexes in grouped.items():
            ordered = sorted(indexes)
            if ordered != list(range(ordered[0], ordered[-1] + 1)):
                raise ValueError(
                    f"Disposable path indexes are not contiguous: {root}/{prefix}"
                )
            result.append(
                (root, prefix, ordered[0], ordered[-1], suffix, size_kb)
            )
        return tuple(result)

    @staticmethod
    def _population_success_output(
        variation: MemoryCleanupVariation,
        nonce: str,
        baseline_kb: float,
        populated_kb: float,
    ) -> str:
        return (
            f"__BASELINE_KB__={baseline_kb}\n"
            f"__POPULATED_COUNT__={len(variation.disposable_objects)}\n"
            f"__POPULATED_KB__={populated_kb}\n"
            + "".join(
                f"__ANCHOR_SHA256__={path}:{'0' * 64}\n"
                for path in _INTEGRITY_ANCHORS
            )
            + f"__POPULATE_COMPLETE__={nonce}\n"
        )

    def _measure_allocated_kb(
        self, container: Container, paths: tuple[str, ...]
    ) -> float:
        cmd = "command -v du >/dev/null 2>&1 || exit 127\n" + self._du_shell(
            paths, "__MEASURE_KB__"
        )
        try:
            raw = _exec_run_with_timeout(container, ["sh", "-c", cmd])
            exit_code, output = _coerce_exec_result(raw)
        except TaskEnvironmentError:
            raise
        except Exception as exc:
            raise TaskEnvironmentError(
                f"Allocated-space measurement failed on {container.id}: {exc}",
                container_ids=[container.id] if container.id else None,
            ) from exc
        text = output.decode(errors="replace")
        if exit_code != 0:
            raise TaskEnvironmentError(
                f"Allocated-space measurement failed on {container.id}: {text.strip()}",
                container_ids=[container.id] if container.id else None,
            )
        try:
            return self._sentinel_number(text, "__MEASURE_KB__")
        except ValueError as exc:
            raise TaskEnvironmentError(
                f"Malformed allocated-space measurement on {container.id}: {exc}",
                container_ids=[container.id] if container.id else None,
            ) from exc

    def _measure_population_kb(
        self, container: Container, dirs: list[str] | None = None
    ) -> float | None:
        """Return allocated KB, or None when the diagnostic probe fails."""
        paths = tuple(self._cleanup_paths if dirs is None else dirs)
        try:
            return self._measure_allocated_kb(container, paths)
        except TaskEnvironmentError as exc:
            logger.warning(str(exc))
            return None

    def _count_absent(self, container: Container, paths: tuple[str, ...]) -> int:
        lines = ["count=0"]
        for path in paths:
            lines.append(f"[ ! -e {shlex.quote(path)} ] && count=$((count+1))")
        lines.append("echo __ABSENT_COUNT__=$count")
        script = "\n".join(lines)
        # Retry transient exec/output failures; typed environment errors propagate.
        last_detail = "no attempt made"
        for attempt in range(_MEASUREMENT_ATTEMPTS):
            try:
                raw = _exec_run_with_timeout(container, ["sh", "-c", script])
                exit_code, output = _coerce_exec_result(raw)
            except TaskEnvironmentError:
                raise
            except Exception as exc:
                last_detail = f"exec error: {exc}"
            else:
                text = output.decode(errors="replace")
                if exit_code == 0:
                    try:
                        return int(self._sentinel_number(text, "__ABSENT_COUNT__"))
                    except ValueError as exc:
                        last_detail = f"no sentinel ({exc}); output={text.strip()[:500]!r}"
                else:
                    last_detail = f"exit={exit_code}: {text.strip()[:500]!r}"
            if attempt + 1 < _MEASUREMENT_ATTEMPTS:
                time.sleep(_MEASUREMENT_RETRY_BACKOFF_S * (attempt + 1))
        raise TaskEnvironmentError(
            f"Disposable manifest measurement failed on {container.id} after "
            f"{_MEASUREMENT_ATTEMPTS} attempts: {last_detail}",
            container_ids=[container.id] if container.id else None,
        )

    def _run_probe_script(
        self,
        container: Container,
        baseline: MemoryCleanupBaseline,
    ) -> tuple[bool, tuple[ProbeResult, ...], bool, tuple[str, ...]]:
        try:
            container.reload()
            running = bool(container.attrs.get("State", {}).get("Running"))
        except Exception:
            running = False
        results: list[ProbeResult] = [ProbeResult("container_running", running)]
        if not running:
            return False, tuple(results), False, ()
        lines: list[str] = []
        for index, probe in enumerate(baseline.viability_probes):
            lines.append(
                f"if {probe.command}; then echo '__PROBE_{index}__=1'; "
                f"else echo '__PROBE_{index}__=0'; fi"
            )
        lines.append("ps -eo comm= 2>/dev/null | sort -u | sed 's/^/__PROCESS__=/'")
        try:
            raw = _exec_run_with_timeout(container, ["sh", "-c", "\n".join(lines)])
            exit_code, output = _coerce_exec_result(raw)
        except TaskEnvironmentError:
            raise
        except Exception as exc:
            results.append(ProbeResult("probe_exec", False, str(exc)))
            return False, tuple(results), False, ()
        text = output.decode(errors="replace")
        if exit_code != 0:
            results.append(ProbeResult("probe_exec", False, text.strip()))
            return False, tuple(results), False, ()
        for index, probe in enumerate(baseline.viability_probes):
            passed = f"__PROBE_{index}__=1" in text
            results.append(ProbeResult(probe.name, passed))
        integrity_ok = all(
            result.passed
            for result, probe in zip(results[1:], baseline.viability_probes)
            if probe.integrity
        )
        processes = tuple(
            sorted(
                line.split("=", 1)[1]
                for line in text.splitlines()
                if line.startswith("__PROCESS__=")
            )
        )
        return all(result.passed for result in results), tuple(results), integrity_ok, processes
