from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from src.observability.collector import MetricsCollector


@dataclass
class SlotDisplayState:
    slot_id: int
    status: str = "starting"  # starting | running | rebuilding | tripped | done
    episode: Optional[int] = None
    successes: int = 0
    failures: int = 0
    cb_tripped: bool = False
    telemetry: dict[str, str] = field(default_factory=dict)
    last_prompt_tokens: Optional[int] = None


class ParallelProgressDisplay:
    """Thread-safe Rich Live display for run_generation_parallel()."""

    def __init__(
        self,
        n_slots: int,
        target_rows: int,
        max_episodes: int,
        run_id: str,
        generation_id: int,
        console: Console,
        metrics_collector: MetricsCollector | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self._target_rows = target_rows
        self._max_episodes = max_episodes
        self._run_id = run_id
        self._generation_id = generation_id
        self._console = console
        self._metrics_collector = metrics_collector
        self._max_context_tokens = max_context_tokens
        self._lock = threading.Lock()
        self._metrics: dict[str, Any] = {}
        self._last_metrics_read: float = 0.0
        self._telemetry_columns: list[str] = []

        self.slot_states: dict[int, SlotDisplayState] = {
            i: SlotDisplayState(slot_id=i) for i in range(n_slots)
        }

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.percentage:>5.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self._rows_task = self._progress.add_task("rows", total=target_rows)
        self._episodes_task = self._progress.add_task("episodes", total=max_episodes)
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=4,
            get_renderable=self._render,
        )

    def update_slot(self, slot_id: int, **kwargs: Any) -> None:
        """Thread-safe per-slot state update."""
        with self._lock:
            state = self.slot_states.get(slot_id)
            if state is None:
                return
            for k, v in kwargs.items():
                setattr(state, k, v)

    def update_progress(self, rows: int, episodes: int) -> None:
        """Update aggregate progress. Refreshes metrics if >1s since last read."""
        self._progress.update(self._rows_task, completed=rows)
        self._progress.update(self._episodes_task, completed=episodes)

        if self._metrics_collector is not None:
            now = time.monotonic()
            if now - self._last_metrics_read >= 1.0:
                self._last_metrics_read = now
                snap = self._metrics_collector.live_utilization_snapshot()
                with self._lock:
                    self._metrics = {}
                    if snap.avg_gpu_utilization_pct is not None:
                        self._metrics["gpu_pct"] = snap.avg_gpu_utilization_pct
                    if snap.peak_gpu_utilization_pct is not None:
                        self._metrics["gpu_peak"] = snap.peak_gpu_utilization_pct
                    if snap.avg_cpu_utilization_pct is not None:
                        self._metrics["cpu_pct"] = snap.avg_cpu_utilization_pct
                    if snap.avg_kv_cache_usage_pct is not None:
                        self._metrics["kv_pct"] = snap.avg_kv_cache_usage_pct
                    if snap.avg_output_tokens_per_second is not None:
                        self._metrics["tok_s"] = snap.avg_output_tokens_per_second

    def set_telemetry_columns(self, columns: list[str]) -> None:
        with self._lock:
            self._telemetry_columns = list(columns)

    def update_metrics(
        self,
        gpu_pct: float | None = None,
        gpu_peak: float | None = None,
        cpu_pct: float | None = None,
        kv_pct: float | None = None,
        tok_s: float | None = None,
    ) -> None:
        """Directly set metrics values (useful for testing without a MetricsCollector)."""
        with self._lock:
            self._metrics = {}
            if gpu_pct is not None:
                self._metrics["gpu_pct"] = gpu_pct
            if gpu_peak is not None:
                self._metrics["gpu_peak"] = gpu_peak
            if cpu_pct is not None:
                self._metrics["cpu_pct"] = cpu_pct
            if kv_pct is not None:
                self._metrics["kv_pct"] = kv_pct
            if tok_s is not None:
                self._metrics["tok_s"] = tok_s

    def _render(self) -> Group:
        """Build the display: title, progress bars, metrics line, per-slot table."""
        title = Text(
            f"Generation {self._generation_id} -- {self._run_id}",
            style="bold",
        )

        # Metrics line
        with self._lock:
            metrics = dict(self._metrics)
            telemetry_columns = list(self._telemetry_columns)
            slot_snapshot = [
                (
                    s.slot_id,
                    s.status,
                    s.episode,
                    s.successes,
                    s.failures,
                    s.cb_tripped,
                    dict(s.telemetry),
                    s.last_prompt_tokens,
                )
                for s in self.slot_states.values()
            ]

        parts: list[str] = []
        if "gpu_pct" in metrics:
            gpu_str = f"GPU: {metrics['gpu_pct']:.0f}%"
            if "gpu_peak" in metrics:
                gpu_str += f" (pk {metrics['gpu_peak']:.0f}%)"
            parts.append(gpu_str)
        if "cpu_pct" in metrics:
            parts.append(f"CPU: {metrics['cpu_pct']:.0f}%")
        if "kv_pct" in metrics:
            parts.append(f"KV: {metrics['kv_pct']:.0f}%")
        if "tok_s" in metrics:
            parts.append(f"tok/s: {metrics['tok_s']:.1f}")
        metrics_text = Text("  ".join(parts), style="dim") if parts else Text("")

        # Per-slot table
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Slot", justify="right", width=4)
        table.add_column("Status", width=12)
        table.add_column("Episode", justify="right", width=7)
        table.add_column("Rows", justify="right", width=5)
        table.add_column("Fails", justify="right", width=5)
        table.add_column("CB", width=3)
        for key in telemetry_columns:
            table.add_column(key, justify="right", width=10)
        if self._max_context_tokens is not None:
            table.add_column("Ctx", justify="right", width=12)

        for (
            slot_id,
            status,
            episode,
            successes,
            failures,
            cb_tripped,
            telemetry,
            last_prompt_tokens,
        ) in slot_snapshot:
            status_style = {
                "running": "green",
                "rebuilding": "yellow",
                "tripped": "red",
                "done": "dim",
                "starting": "dim",
            }.get(status, "")

            cb_icon = "[red]![/red]" if cb_tripped else "[green]ok[/green]"
            ep_str = str(episode) if episode is not None else "-"

            row = [
                str(slot_id),
                f"[{status_style}]{status}[/{status_style}]"
                if status_style
                else status,
                ep_str,
                str(successes),
                str(failures),
                cb_icon,
            ]
            for key in telemetry_columns:
                row.append(telemetry.get(key, "-"))
            if self._max_context_tokens is not None:
                if last_prompt_tokens is None:
                    row.append("-")
                else:
                    ratio = (
                        last_prompt_tokens / self._max_context_tokens
                        if self._max_context_tokens > 0
                        else 0.0
                    )
                    color = "green"
                    if ratio >= 0.9:
                        color = "red"
                    elif ratio >= 0.7:
                        color = "yellow"
                    row.append(
                        f"[{color}]"
                        f"{last_prompt_tokens}/{self._max_context_tokens}"
                        f"[/{color}]"
                    )
            table.add_row(*row)

        return Group(title, self._progress, metrics_text, table)

    def __enter__(self) -> ParallelProgressDisplay:
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._live.__exit__(*args)


@contextmanager
def scoped_loguru_to_rich(console: Console, log_file_path: Path):
    """WARNING+ -> Rich console, DEBUG+ -> log file. Restores loguru on exit."""
    logger.remove()
    try:
        logger.add(str(log_file_path), level="DEBUG", enqueue=True)

        def _rich_sink(message):
            console.print(message, end="")

        logger.add(
            _rich_sink,
            level="WARNING",
            colorize=False,
            format="{time:HH:mm:ss} [{level}] {message}",
        )
        yield
    finally:
        logger.remove()
        logger.add(sys.stderr)
