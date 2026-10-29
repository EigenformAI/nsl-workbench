"""Tests for the timeout context manager in src/helper.py."""

import threading
import time
import unittest

from src.helper import timeout


class TimeoutHelperTests(unittest.TestCase):
    def test_timeout_raises_in_calling_thread(self) -> None:
        """The TimeoutError must be raised in the thread that entered the
        context manager, not in a separate timer thread."""
        with self.assertRaises(TimeoutError):
            with timeout(0.3):
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    pass

    def test_no_timeout_when_code_completes_in_time(self) -> None:
        with timeout(5):
            x = sum(range(1000))  # noqa: F841
        # Should reach here without error

    def test_callback_is_invoked_on_timeout(self) -> None:
        called = threading.Event()

        with self.assertRaises(TimeoutError):
            with timeout(0.3, callback=lambda: called.set()):
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    pass

        self.assertTrue(called.is_set())

    def test_timeout_works_in_worker_thread(self) -> None:
        """Timeout must also work when called from a non-main thread
        (the parallel generation workers are non-main threads).

        Note: PyThreadState_SetAsyncExc can only interrupt Python bytecode,
        not C-level blocking calls (time.sleep, socket.recv, etc.). We use
        a pure-Python busy loop to validate the mechanism.
        """
        result: dict[str, object] = {}

        def worker() -> None:
            try:
                with timeout(0.3):
                    # Pure-Python loop: bytecode is checked between iterations
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        pass
                result["error"] = None
            except TimeoutError:
                result["error"] = "TimeoutError"

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)

        self.assertEqual(result.get("error"), "TimeoutError")
        self.assertFalse(t.is_alive())


if __name__ == "__main__":
    unittest.main()
