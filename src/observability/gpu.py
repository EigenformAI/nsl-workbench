"""Shared GPU reading utilities using pynvml with proper lifecycle management.

All functions accept a ``device_index`` parameter (default 0) to support
future multi-GPU extension.  Neither function falls back to
``torch.cuda.memory_allocated()`` — that measures client-process PyTorch
allocations, not device-level usage, which is misleading when the inference
server runs in a separate process or container.
"""

from __future__ import annotations

import importlib
import re
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass(frozen=True)
class VisibleGpuDetails:
    index: int
    name: str
    total_memory_mb: float


def read_gpu_memory_info(
    device_index: int = 0,
) -> Optional[tuple[float, float, float]]:
    """Return (used_mb, free_mb, total_mb) via pynvml or nvidia-smi.

    Returns None and logs a warning if neither pynvml nor nvidia-smi is
    available.
    """
    result = _read_gpu_memory_info_pynvml(device_index)
    if result is not None:
        return result

    result = _read_gpu_memory_info_nvidia_smi(device_index)
    if result is not None:
        return result

    logger.warning("GPU memory info unavailable (no pynvml or nvidia-smi)")
    return None


def list_visible_gpu_indices() -> list[int]:
    """Return indices of GPUs currently visible to NVML / nvidia-smi.

    Returns ``[]`` when neither source is available, so callers can
    distinguish "no GPUs / no probe" from "one GPU at index 0".
    """
    result = _list_visible_gpu_indices_pynvml()
    if result is not None:
        return result

    result = _list_visible_gpu_indices_nvidia_smi()
    if result is not None:
        return result

    return []


def list_visible_gpu_details() -> list[VisibleGpuDetails] | None:
    """Return visible GPU name/VRAM details, or None when probing fails."""
    result = _list_visible_gpu_details_pynvml()
    if result is not None:
        return result

    result = _list_visible_gpu_details_nvidia_smi()
    if result is not None:
        return result

    return None


def detect_hardware_tags() -> list[str]:
    """Best-effort tags for GPUs visible to this process/container."""
    devices = list_visible_gpu_details()
    if devices is None:
        logger.warning("GPU hardware tag detection unavailable (no pynvml or nvidia-smi)")
        return ["unknown-hardware"]
    if not devices:
        logger.info("GPU hardware tag detection found no visible GPUs")
        return ["cpu-only"]

    tags: list[str] = [f"{len(devices)}x"]
    profiles: OrderedDict[tuple[str, str], int] = OrderedDict()
    for device in devices:
        key = (_normalize_gpu_name(device.name), _vram_tag(device.total_memory_mb))
        profiles[key] = profiles.get(key, 0) + 1

    for (name_tag, vram_tag), count in profiles.items():
        tags.extend([name_tag, vram_tag, f"{name_tag}-{vram_tag}-{count}x"])
    return _normalize_tags(tags)


def read_gpu_utilization_pct(
    device_index: int = 0,
) -> Optional[float]:
    """Return GPU utilization percentage via pynvml or nvidia-smi.

    Returns None if neither source is available.
    """
    result = _read_gpu_utilization_pct_pynvml(device_index)
    if result is not None:
        return result

    result = _read_gpu_utilization_pct_nvidia_smi(device_index)
    if result is not None:
        return result

    return None


# ---------------------------------------------------------------------------
# pynvml helpers
# ---------------------------------------------------------------------------

def _read_gpu_memory_info_pynvml(
    device_index: int,
) -> Optional[tuple[float, float, float]]:
    pynvml = None
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (
            float(memory_info.used / (1024 * 1024)),
            float(memory_info.free / (1024 * 1024)),
            float(memory_info.total / (1024 * 1024)),
        )
    except Exception:
        return None
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _read_gpu_utilization_pct_pynvml(
    device_index: int,
) -> Optional[float]:
    pynvml = None
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(utilization.gpu)
    except Exception:
        return None
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _list_visible_gpu_indices_pynvml() -> Optional[list[int]]:
    pynvml = None
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        return list(range(count))
    except Exception:
        return None
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _list_visible_gpu_details_pynvml() -> Optional[list[VisibleGpuDetails]]:
    pynvml = None
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        devices: list[VisibleGpuDetails] = []
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                VisibleGpuDetails(
                    index=index,
                    name=name,
                    total_memory_mb=float(memory_info.total / (1024 * 1024)),
                )
            )
        return devices
    except Exception:
        return None
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# nvidia-smi fallback helpers
# ---------------------------------------------------------------------------

def _read_gpu_memory_info_nvidia_smi(
    device_index: int,
) -> Optional[tuple[float, float, float]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    lines = result.stdout.strip().splitlines()
    if not lines:
        return None

    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 3:
        return None

    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _list_visible_gpu_indices_nvidia_smi() -> Optional[list[int]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    indices: list[int] = []
    for line in result.stdout.strip().splitlines():
        try:
            indices.append(int(line.strip()))
        except ValueError:
            return None
    return indices


def _list_visible_gpu_details_nvidia_smi() -> Optional[list[VisibleGpuDetails]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    devices: list[VisibleGpuDetails] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            return None
        try:
            devices.append(
                VisibleGpuDetails(
                    index=int(parts[0]),
                    name=parts[1],
                    total_memory_mb=float(parts[2]),
                )
            )
        except ValueError:
            return None
    return devices


def _read_gpu_utilization_pct_nvidia_smi(
    device_index: int,
) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    lines = result.stdout.strip().splitlines()
    if not lines:
        return None

    try:
        return float(lines[0])
    except ValueError:
        return None


def _normalize_gpu_name(name: str) -> str:
    normalized = name.lower()
    normalized = re.sub(r"\b(nvidia|geforce|amd|radeon)\b", " ", normalized)
    normalized = re.sub(r"\b\d+\s*gb\b", " ", normalized)
    normalized = re.sub(r"\bhbm\d*\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized or "unknown-gpu"


def _vram_tag(total_memory_mb: float) -> str:
    return f"{int(round(total_memory_mb / 1024))}gb"


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in tags:
        tag = re.sub(r"[^a-z0-9]+", "-", item.strip().lower()).strip("-")
        if not tag or tag in seen:
            continue
        normalized.append(tag)
        seen.add(tag)
    return normalized
