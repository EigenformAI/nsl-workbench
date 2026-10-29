"""Leak-safe public-data export and isolated Python execution."""

from __future__ import annotations

import csv
import io
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from docker.models.containers import Container

from src.task.base import TaskEnvironmentError
from src.task.types import CapabilityResult
from tasks.common.container_exec import coerce_exec_result, exec_run_with_timeout
from tasks.materials_adsorption.dataset import LoadedDataset

MAX_SNIPPET_BYTES = 8 * 1024
MAX_OUTPUT_BYTES = 4 * 1024
OUTPUT_TRUNCATION_MARKER = "\n[output truncated]"
COMPUTE_SERVICE_NAME = "compute-service"


def _csv_bytes(fieldnames: list[str], rows: Sequence[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_public_export(
    loaded: LoadedDataset,
    evidence: Sequence[dict[str, Any]],
    evidence_alias: str,
    rule_contract: str,
    destination: Path,
) -> None:
    """Write only public aliases and descriptors in deterministic order."""
    descriptor_names = sorted(spec.name for spec in loaded.registry.descriptors)
    universe_rows = [
        {
            "candidate_alias": candidate.alias,
            **{name: candidate.descriptors[name] for name in descriptor_names},
        }
        for candidate in sorted(loaded.universe.candidates, key=lambda item: item.alias)
    ]
    evidence_rows = []
    for item in sorted(evidence, key=lambda row: str(row.get("alias", ""))):
        descriptors = item.get("descriptors")
        if not isinstance(descriptors, dict):
            continue
        evidence_rows.append(
            {
                "evidence_alias": evidence_alias,
                "candidate_alias": item.get("alias", ""),
                "system_alias": item.get("system_alias", ""),
                **{name: descriptors[name] for name in descriptor_names},
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    for child in destination.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (destination / "universe.csv").write_bytes(
        _csv_bytes(["candidate_alias", *descriptor_names], universe_rows)
    )
    (destination / "evidence.csv").write_bytes(
        _csv_bytes(
            ["evidence_alias", "candidate_alias", "system_alias", *descriptor_names],
            evidence_rows,
        )
    )
    (destination / "README.md").write_text(
        "# Public Materials Evidence\n\n"
        "`universe.csv` contains candidate aliases and public pre-DFT descriptors.\n"
        "`evidence.csv` contains the episode evidence shard using the same aliases.\n\n"
        "## Rule Contract\n\n"
        f"{rule_contract}\n",
        encoding="utf-8",
        newline="\n",
    )


def public_export_mount(container: Container) -> Path | None:
    attrs = getattr(container, "attrs", {})
    mounts = attrs.get("Mounts", []) if isinstance(attrs, dict) else []
    for mount in mounts:
        if (
            isinstance(mount, dict)
            and mount.get("Destination") == "/data"
            and isinstance(mount.get("Source"), str)
        ):
            return Path(mount["Source"])
    return None


def _truncate(raw: bytes) -> tuple[str, bool]:
    if len(raw) <= MAX_OUTPUT_BYTES:
        return raw.decode("utf-8", errors="replace"), False
    allowed = MAX_OUTPUT_BYTES - len(OUTPUT_TRUNCATION_MARKER.encode("utf-8"))
    return (
        raw[:allowed].decode("utf-8", errors="replace") + OUTPUT_TRUNCATION_MARKER,
        True,
    )


def execute_public_python(
    container: Container,
    snippet: Any,
    *,
    timeout_s: float = 120,
    executor: Callable[..., Any] = exec_run_with_timeout,
) -> CapabilityResult:
    if not isinstance(snippet, str) or not snippet.strip():
        return CapabilityResult("run_python", success=False, error="snippet_required")
    if len(snippet.encode("utf-8")) > MAX_SNIPPET_BYTES:
        return CapabilityResult("run_python", success=False, error="snippet_too_large")
    try:
        result = executor(
            container,
            ["timeout", "--signal=KILL", f"{timeout_s:g}s", "python", "-I", "-c", snippet],
            timeout_s=timeout_s + 5,
            demux=True,
            workdir="/data",
        )
    except (TimeoutError, TaskEnvironmentError):
        return CapabilityResult(
            "run_python",
            output={"timed_out": True, "stdout": "", "stderr": ""},
            success=False,
            error="execution_timeout",
        )

    if hasattr(result, "exit_code"):
        exit_code = int(result.exit_code if result.exit_code is not None else -1)
        output = result.output
    else:
        exit_code, output = coerce_exec_result(result)
    if isinstance(output, tuple):
        stdout_raw = bytes(output[0] or b"")
        stderr_raw = bytes(output[1] or b"")
    else:
        stdout_raw = bytes(output or b"")
        stderr_raw = b""
    if exit_code == 124 or exit_code == 137:
        return CapabilityResult(
            "run_python",
            output={"timed_out": True, "stdout": "", "stderr": ""},
            success=False,
            error="execution_timeout",
        )
    combined, truncated = _truncate(stdout_raw + stderr_raw)
    split = min(len(stdout_raw), len(combined))
    stdout = combined[:split] if stderr_raw else combined
    stderr = combined[split:] if stderr_raw else ""
    return CapabilityResult(
        "run_python",
        output={
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
            "timed_out": False,
        },
        success=exit_code == 0,
        error=None if exit_code == 0 else "execution_failed",
    )
