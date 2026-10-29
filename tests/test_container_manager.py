import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import src.container as container_module
from src.container import ContainerManager
from tests.fixtures.stub_task import ValidStubTask


class ExecResult:
    def __init__(self, exit_code: int = 0, output: bytes = b"") -> None:
        self.exit_code = exit_code
        self.output = output

    def __iter__(self):
        yield self.exit_code
        yield self.output


class ContainerManagerTests(unittest.TestCase):
    def make_compose_result(self, *entries: dict[str, str]) -> SimpleNamespace:
        return SimpleNamespace(
            stdout="\n".join(json.dumps(entry) for entry in entries),
            stderr="",
            returncode=0,
        )

    def make_manager(
        self, exec_return: bytes = b"ok"
    ) -> tuple[ContainerManager, MagicMock, MagicMock, MagicMock]:
        docker_client = MagicMock()
        container_a = MagicMock()
        container_a.id = "container-a"
        container_a.name = "special-learn-compose-service-a-1"
        container_a.status = "running"
        container_a.attrs = {
            "Config": {
                "Labels": {"com.docker.compose.service": "service-a"},
            },
            "State": {"Health": {"Status": "healthy"}},
        }
        container_a.exec_run.return_value = ExecResult(0, exec_return)

        container_b = MagicMock()
        container_b.id = "container-b"
        container_b.name = "special-learn-compose-service-b-1"
        container_b.status = "running"
        container_b.attrs = {
            "Config": {
                "Labels": {"com.docker.compose.service": "service-b"},
            },
            "State": {"Health": {"Status": "healthy"}},
        }
        container_b.exec_run.return_value = ExecResult(0, exec_return)

        refreshed_a = MagicMock()
        refreshed_a.id = "container-a-new"
        refreshed_a.name = "special-learn-compose-service-a-9"
        refreshed_a.attrs = {
            "Config": {
                "Labels": {"com.docker.compose.service": "service-a"},
            },
            "State": {"Health": {"Status": "healthy"}},
        }

        refreshed_b = MagicMock()
        refreshed_b.id = "container-b-new"
        refreshed_b.name = "special-learn-compose-service-b-9"
        refreshed_b.attrs = {
            "Config": {
                "Labels": {"com.docker.compose.service": "service-b"},
            },
            "State": {"Health": {"Status": "healthy"}},
        }

        containers_by_id = {
            "container-a": container_a,
            "container-b": container_b,
            "special-learn-compose-service-a-1": container_a,
            "special-learn-compose-service-b-1": container_b,
            "special-learn-compose-service-a-9": refreshed_a,
            "special-learn-compose-service-b-9": refreshed_b,
        }
        docker_client.containers.get.side_effect = containers_by_id.__getitem__
        docker_client.containers.list.return_value = [container_b, container_a]

        manager = ContainerManager(
            docker_client=docker_client,
            container_ids=["container-a", "container-b"],
            docker_compose_dir="/tmp/docker-compose",
            post_rebuild_wait_seconds=7,
            project_name_pattern="special-learn-compose",
            task=ValidStubTask({}),
        )
        setattr(manager, "expected_services", ["service-a", "service-b"])
        return manager, docker_client, container_a, container_b

    @patch("src.container.subprocess.run")
    def test_rebuild_calls_compose_with_build_flag(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        with (
            patch.object(manager, "refresh_container_ids") as refresh_container_ids,
            patch.object(
                manager,
                "verify_ready",
                return_value=True,
            ),
        ):
            manager.rebuild()

        subprocess_run.assert_has_calls(
            [
                call(
                    ["docker", "compose", "-p", "special-learn-compose", "down"],
                    cwd="/tmp/docker-compose",
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,
                ),
                call(
                    [
                        "docker",
                        "compose",
                        "-p",
                        "special-learn-compose",
                        "up",
                        "-d",
                        "--build",
                    ],
                    cwd="/tmp/docker-compose",
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,
                ),
            ]
        )
        refresh_container_ids.assert_called_once()

    @patch("src.container.subprocess.run")
    def test_restart_calls_compose_without_build(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        with (
            patch.object(manager, "refresh_container_ids") as refresh_container_ids,
            patch.object(
                manager,
                "verify_ready",
                return_value=True,
            ),
        ):
            manager.restart()

        subprocess_run.assert_has_calls(
            [
                call(
                    ["docker", "compose", "-p", "special-learn-compose", "down"],
                    cwd="/tmp/docker-compose",
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,
                ),
                call(
                    ["docker", "compose", "-p", "special-learn-compose", "up", "-d"],
                    cwd="/tmp/docker-compose",
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,
                ),
            ]
        )
        refresh_container_ids.assert_called_once()

    @patch("src.container.subprocess.run")
    def test_refresh_container_ids_uses_compose_ps_and_preserves_expected_order(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        subprocess_run.return_value = self.make_compose_result(
            {
                "Name": "special-learn-compose-service-b-9",
                "Service": "service-b",
                "State": "exited",
            },
            {
                "Name": "special-learn-compose-service-a-9",
                "Service": "service-a",
                "State": "running",
            },
        )

        refreshed = manager.refresh_container_ids()

        self.assertEqual(
            refreshed,
            [
                "special-learn-compose-service-a-9",
                "special-learn-compose-service-b-9",
            ],
        )
        subprocess_run.assert_called_once_with(
            [
                "docker",
                "compose",
                "-p",
                "special-learn-compose",
                "-f",
                "/tmp/docker-compose/docker-compose.yml",
                "ps",
                "--format",
                "json",
                "--all",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("src.container.subprocess.run")
    def test_refresh_container_ids_raises_when_expected_service_missing(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        subprocess_run.return_value = self.make_compose_result(
            {
                "Name": "special-learn-compose-service-a-9",
                "Service": "service-a",
                "State": "running",
            }
        )

        with self.assertRaises(RuntimeError):
            manager.refresh_container_ids()

    def test_refresh_container_ids_requires_project_name_pattern(self) -> None:
        manager, _, _, _ = self.make_manager()
        manager.project_name_pattern = None

        with self.assertRaises(ValueError):
            manager.refresh_container_ids()

    def test_container_to_service_uses_compose_label(self) -> None:
        manager, _, _, _ = self.make_manager()

        self.assertEqual(manager._container_to_service("container-a"), "service-a")

    def test_container_to_service_raises_without_compose_label(self) -> None:
        manager, docker_client, _, _ = self.make_manager()
        unlabeled = MagicMock()
        unlabeled.attrs = {"Config": {"Labels": {}}, "State": {}}
        docker_client.containers.get.side_effect = {"broken": unlabeled}.__getitem__

        with self.assertRaises(ValueError):
            manager._container_to_service("broken")

    def test_public_container_to_service_reads_compose_label(self) -> None:
        _, _, container_a, _ = self.make_manager()

        self.assertEqual(
            getattr(container_module, "container_to_service")(container_a),
            "service-a",
        )

    def test_services_map_uses_exact_service_names(self) -> None:
        manager, _, container_a, container_b = self.make_manager()

        self.assertEqual(
            manager.services(),
            {"service-a": container_a, "service-b": container_b},
        )

    def test_get_service_raises_for_unknown_service(self) -> None:
        manager, _, _, _ = self.make_manager()

        with self.assertRaises(KeyError):
            manager.get_service("agent-service")

    def test_rebuild_containers_targets_specific_services(self) -> None:
        manager, _, _, _ = self.make_manager()

        with (
            patch.object(manager, "_run_compose") as mock_compose,
            patch.object(
                manager,
                "refresh_container_ids",
                return_value=[
                    "special-learn-compose-service-a-1",
                    "special-learn-compose-service-b-1",
                ],
            ),
            patch.object(manager, "_install_procps_for") as install_procps,
            patch.object(
                manager, "verify_container_ready", return_value=True
            ) as verify,
        ):
            manager.rebuild_containers(["container-b"])

        mock_compose.assert_called_once_with(
            [
                "docker",
                "compose",
                "up",
                "-d",
                "--build",
                "--force-recreate",
                "--no-deps",
                "service-b",
            ]
        )
        install_procps.assert_called_once_with(["special-learn-compose-service-b-1"])
        verify.assert_called_once_with("special-learn-compose-service-b-1")

    def test_rebuild_containers_falls_back_to_stack_rebuild_when_all_broken(
        self,
    ) -> None:
        manager, _, _, _ = self.make_manager()

        with (
            patch.object(manager, "rebuild") as rebuild,
            patch.object(manager, "_run_compose") as mock_compose,
        ):
            manager.rebuild_containers(["container-a", "container-b"])

        rebuild.assert_called_once_with()
        mock_compose.assert_not_called()

    @patch("src.container.subprocess.run")
    def test_rebuild_installs_procps_for_readiness_probe(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, container_a, container_b = self.make_manager()
        with (
            patch.object(
                manager, "refresh_container_ids", return_value=manager.container_ids
            ),
            patch.object(
                manager,
                "verify_ready",
                return_value=True,
            ),
        ):
            manager.rebuild()

        container_a.exec_run.assert_any_call(
            ["sh", "-c", "apk add --no-cache procps"],
            stdout=False,
            stderr=False,
        )
        container_b.exec_run.assert_any_call(
            ["sh", "-c", "apk add --no-cache procps"],
            stdout=False,
            stderr=False,
        )

    @patch("src.container.subprocess.run")
    def test_rebuild_does_not_use_fixed_sleep_before_verify_ready(
        self,
        subprocess_run: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        with (
            patch.object(manager, "refresh_container_ids"),
            patch.object(
                manager,
                "verify_ready",
                return_value=True,
            ),
        ):
            manager.rebuild()

    @patch("src.container.wait_and_get_container")
    def test_verify_ready_checks_all_containers(
        self,
        wait_and_get_container: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        ready_container = MagicMock()
        ready_container.attrs = {"State": {}}
        wait_and_get_container.return_value = ready_container

        ready = manager.verify_ready()

        self.assertTrue(ready)
        wait_and_get_container.assert_has_calls(
            [
                call(manager.docker_client, "container-a", timeout=30),
                call(manager.docker_client, "container-b", timeout=30),
            ]
        )

    def test_verify_ready_returns_false_when_any_container_fails(self) -> None:
        manager, _, _, _ = self.make_manager()

        with patch.object(
            manager,
            "verify_container_ready",
            side_effect=[True, False],
        ) as verify_container_ready:
            ready = manager.verify_ready()

        self.assertFalse(ready)
        self.assertEqual(verify_container_ready.call_count, 2)

    @patch("src.container.wait_and_get_container")
    def test_verify_container_ready_checks_single_container(
        self,
        wait_and_get_container: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()

        ready = manager.verify_container_ready("container-b")

        self.assertTrue(ready)
        wait_and_get_container.assert_called_once_with(
            manager.docker_client, "container-b", timeout=30
        )

    @patch("src.container.wait_and_get_container")
    def test_verify_container_ready_returns_false_when_wait_fails(
        self,
        wait_and_get_container: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        wait_and_get_container.side_effect = RuntimeError("boom")

        self.assertFalse(manager.verify_container_ready("container-b"))

    @patch("src.container.wait_and_get_container")
    def test_verify_container_ready_returns_false_when_container_unhealthy(
        self,
        wait_and_get_container: MagicMock,
    ) -> None:
        manager, _, _, _ = self.make_manager()
        unhealthy = MagicMock()
        unhealthy.attrs = {"State": {"Health": {"Status": "unhealthy"}}}
        wait_and_get_container.return_value = unhealthy

        self.assertFalse(manager.verify_container_ready("container-b"))


if __name__ == "__main__":
    unittest.main()
