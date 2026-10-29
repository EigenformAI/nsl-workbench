from __future__ import annotations

import io
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from rich.console import Console

from src.display import (
    ParallelProgressDisplay,
    scoped_loguru_to_rich,
)


class ParallelProgressDisplayTests(unittest.TestCase):
    def _make_display(
        self,
        n_slots: int = 4,
        target_rows: int = 2000,
        max_episodes: int = 100,
        **kwargs,
    ):
        console = Console(file=io.StringIO(), force_terminal=False)
        return ParallelProgressDisplay(
            n_slots=n_slots,
            target_rows=target_rows,
            max_episodes=max_episodes,
            run_id="test-run-42",
            generation_id=0,
            console=console,
            **kwargs,
        )

    def test_update_slot_mutates_state(self) -> None:
        display = self._make_display()
        display.update_slot(0, status="running", episode=5)
        self.assertEqual(display.slot_states[0].status, "running")
        self.assertEqual(display.slot_states[0].episode, 5)
        # Other slots unchanged
        self.assertEqual(display.slot_states[1].status, "starting")

    def test_update_slot_multiple_fields(self) -> None:
        display = self._make_display()
        display.update_slot(2, status="tripped", cb_tripped=True, failures=4)
        self.assertEqual(display.slot_states[2].status, "tripped")
        self.assertTrue(display.slot_states[2].cb_tripped)
        self.assertEqual(display.slot_states[2].failures, 4)

    def test_update_slot_accepts_telemetry(self) -> None:
        display = self._make_display()
        display.update_slot(0, telemetry={"step": "3"})
        self.assertEqual(display.slot_states[0].telemetry, {"step": "3"})

    def test_update_progress_advances_tasks(self) -> None:
        display = self._make_display()
        display.update_progress(rows=500, episodes=10)
        rows_task = display._progress.tasks[0]
        episodes_task = display._progress.tasks[1]
        self.assertEqual(rows_task.completed, 500)
        self.assertEqual(episodes_task.completed, 10)

    def test_episodes_has_deterministic_total(self) -> None:
        display = self._make_display(max_episodes=200)
        episodes_task = display._progress.tasks[1]
        self.assertEqual(episodes_task.total, 200)

    def test_render_with_metrics(self) -> None:
        display = self._make_display()
        display.update_metrics(gpu_pct=85.0, cpu_pct=42.0, tok_s=23.4)
        console = Console(file=io.StringIO(), force_terminal=True, width=120)
        console.print(display._render())
        output = console.file.getvalue()
        self.assertIn("85%", output)
        self.assertIn("42%", output)
        self.assertIn("23.4", output)

    def test_render_metrics_none_fields_omitted(self) -> None:
        display = self._make_display()
        display.update_metrics(gpu_pct=50.0)
        console = Console(file=io.StringIO(), force_terminal=True, width=120)
        console.print(display._render())
        output = console.file.getvalue()
        self.assertIn("50", output)
        # tok/s should not appear when None
        self.assertNotIn("tok/s", output)

    def test_render_includes_context_column_when_configured(self) -> None:
        display = self._make_display(max_context_tokens=1000)
        display.update_slot(0, last_prompt_tokens=650)
        display.update_slot(1, last_prompt_tokens=920)
        console = Console(file=io.StringIO(), force_terminal=True, width=120)
        console.print(display._render())
        output = console.file.getvalue()
        self.assertIn("Ctx", output)
        self.assertIn("650/1000", output)
        self.assertIn("920/1000", output)

    def test_render_includes_telemetry_columns_when_configured(self) -> None:
        display = self._make_display()
        display.set_telemetry_columns(["step"])
        display.update_slot(0, telemetry={"step": "3"})
        console = Console(file=io.StringIO(), force_terminal=True, width=120)
        console.print(display._render())
        output = console.file.getvalue()
        self.assertIn("step", output)
        self.assertIn("3", output)
        self.assertNotIn("Pad", output)

    def test_set_telemetry_columns_thread_safe(self) -> None:
        display = self._make_display()

        def writer() -> None:
            for _ in range(50):
                display.set_telemetry_columns(["step", "budget_left"])

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(display._telemetry_columns, ["step", "budget_left"])

    def test_thread_safety_concurrent_update_slot(self) -> None:
        display = self._make_display(n_slots=8)
        errors: list[Exception] = []

        def updater(slot_id: int):
            try:
                for j in range(50):
                    display.update_slot(slot_id, successes=j, episode=j)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

    def test_metrics_from_collector(self) -> None:
        """When a metrics_collector is provided, update_progress should read live metrics."""
        from src.observability.types import LiveUtilizationSnapshot

        mock_collector = MagicMock()
        mock_collector.live_utilization_snapshot.return_value = LiveUtilizationSnapshot(
            avg_gpu_utilization_pct=77.0,
            peak_gpu_utilization_pct=92.0,
            avg_cpu_utilization_pct=35.0,
            peak_cpu_utilization_pct=55.0,
            avg_output_tokens_per_second=150.0,
            sample_count=10,
        )

        display = self._make_display(metrics_collector=mock_collector)
        display.update_progress(rows=100, episodes=5)

        mock_collector.live_utilization_snapshot.assert_called()
        # Metrics should be populated
        self.assertIsNotNone(display._metrics.get("gpu_pct"))


class ScopedLoguruToRichTests(unittest.TestCase):
    def test_warning_reaches_console(self) -> None:
        from loguru import logger

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False)

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "test.log"
            with scoped_loguru_to_rich(console, log_path):
                logger.warning("test warning message")

        output = buf.getvalue()
        self.assertIn("test warning message", output)

    def test_info_suppressed_on_console(self) -> None:
        from loguru import logger

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False)

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "test.log"
            with scoped_loguru_to_rich(console, log_path):
                logger.info("this should not appear on console")

        output = buf.getvalue()
        self.assertNotIn("this should not appear on console", output)

    def test_debug_reaches_file(self) -> None:
        from loguru import logger

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False)

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "test.log"
            with scoped_loguru_to_rich(console, log_path):
                logger.debug("debug file message")

            # loguru enqueue=True means async write; give it a moment
            import time

            time.sleep(0.1)
            contents = log_path.read_text()

        self.assertIn("debug file message", contents)


if __name__ == "__main__":
    unittest.main()
