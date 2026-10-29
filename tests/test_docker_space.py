import unittest
from unittest.mock import MagicMock, patch

from src.tool.docker import wait_and_get_container


class WaitForContainerTests(unittest.TestCase):
    @patch("src.tool.docker.time.sleep")
    def test_running_container_uses_shell_noop_probe(self, sleep: MagicMock) -> None:
        container = MagicMock()
        container.status = "running"
        container.exec_run.return_value = (0, b"")

        docker_client = MagicMock()
        docker_client.containers.get.return_value = container

        with patch("src.tool.docker.time.time", side_effect=[0.0, 0.0]):
            result = wait_and_get_container(docker_client, "container-a", timeout=1)

        self.assertIs(result, container)
        container.exec_run.assert_called_once_with(["sh", "-c", "true"])
        sleep.assert_not_called()

    @patch("src.tool.docker.time.sleep")
    def test_fails_fast_for_terminal_container_state(self, sleep: MagicMock) -> None:
        container = MagicMock()
        container.status = "exited"
        container.logs.return_value = b"process crashed"

        docker_client = MagicMock()
        docker_client.containers.get.return_value = container

        with patch("src.tool.docker.time.time", side_effect=[0.0, 0.0, 2.0]):
            with self.assertRaises(RuntimeError) as context:
                wait_and_get_container(docker_client, "container-a", timeout=1)

        self.assertIn("terminal state 'exited'", str(context.exception))
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
