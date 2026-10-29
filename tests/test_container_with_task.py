"""Tests for ContainerManager delegation to TaskSpec.

These tests verify that ContainerManager delegates environment lifecycle
methods to the task when a TaskSpec is provided.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.container import ContainerManager
from src.task.types import (
    PopulationOutcome,
    PopulationResult,
)
from tests.fixtures.stub_task import ValidStubTask


@pytest.fixture
def mock_docker_client():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.name = "special-learn-compose-service-a-1"
    mock_container.id = "abc123"
    client.containers.get.return_value = mock_container
    client.containers.list.return_value = [mock_container]
    return client


@pytest.fixture
def task():
    return ValidStubTask({})


class TestContainerManagerWithTask:
    def test_populate_with_task_delegates(self, mock_docker_client, task):
        """When task is set, populate_with_task delegates to task methods."""
        cm = ContainerManager(
            docker_client=mock_docker_client,
            container_ids=["abc123"],
            docker_compose_dir="docker/test",
            task=task,
        )
        variation = task.list_variations()[0]
        mock_containers = [MagicMock()]
        mock_containers[0].id = "abc123"

        # Mock the task's methods — outcome carries episode_context
        mock_episode_ctx = {"baseline_kb": {"abc123": 100.0}}
        mock_outcome = PopulationOutcome(
            results=[
                PopulationResult(
                    container_id="abc123",
                    variation_name=variation.name,
                    description="ok",
                    success=True,
                )
            ],
            episode_context=mock_episode_ctx,
        )
        with (
            patch.object(task, "reset") as mock_reset,
            patch.object(task, "populate", return_value=mock_outcome) as mock_populate,
            patch.object(task, "verify_population", return_value=True) as mock_verify,
        ):
            outcome, verified = cm.populate_with_task(
                mock_containers,
                variation,
            )
            mock_reset.assert_called_once_with(mock_containers)
            mock_populate.assert_called_once_with(mock_containers, variation)
            # verify_population receives outcome.episode_context, not caller-supplied
            mock_verify.assert_called_once_with(
                mock_containers,
                variation,
                mock_episode_ctx,
                private_context=None,
            )
            assert outcome is mock_outcome
            assert verified is True

    def test_populate_with_task_verify_failure(self, mock_docker_client, task):
        """When verification fails, verified=False is returned."""
        cm = ContainerManager(
            docker_client=mock_docker_client,
            container_ids=["abc123"],
            docker_compose_dir="docker/test",
            task=task,
        )
        variation = task.list_variations()[0]
        mock_containers = [MagicMock()]
        mock_outcome = PopulationOutcome(
            results=[
                PopulationResult(
                    container_id="abc123",
                    variation_name=variation.name,
                    description="ok",
                    success=True,
                )
            ],
        )
        with (
            patch.object(task, "reset"),
            patch.object(task, "populate", return_value=mock_outcome),
            patch.object(task, "verify_population", return_value=False),
        ):
            outcome, verified = cm.populate_with_task(
                mock_containers,
                variation,
            )
            assert outcome is mock_outcome
            assert verified is False

    def test_populate_with_task_partial_failure(self, mock_docker_client, task):
        """When any PopulationResult.success is False, fails fast without verifying."""
        cm = ContainerManager(
            docker_client=mock_docker_client,
            container_ids=["abc123"],
            docker_compose_dir="docker/test",
            task=task,
        )
        variation = task.list_variations()[0]
        mock_containers = [MagicMock()]
        mock_outcome = PopulationOutcome(
            results=[
                PopulationResult(
                    container_id="abc123",
                    variation_name=variation.name,
                    description="failed",
                    success=False,
                    error_message="populate error",
                )
            ],
        )
        with (
            patch.object(task, "reset"),
            patch.object(task, "populate", return_value=mock_outcome),
            patch.object(task, "verify_population") as mock_verify,
        ):
            outcome, verified = cm.populate_with_task(
                mock_containers,
                variation,
            )
            # verify_population should NOT be called on partial failure
            mock_verify.assert_not_called()
            assert verified is False

    def test_populate_with_task_threads_private_context(self, mock_docker_client, task):
        """verify_population receives private_context from PopulationOutcome."""
        cm = ContainerManager(
            docker_client=mock_docker_client,
            container_ids=["abc123"],
            docker_compose_dir="docker/test",
            task=task,
        )
        variation = task.list_variations()[0]
        mock_containers = [MagicMock()]
        mock_containers[0].id = "abc123"

        mock_private_ctx = {"admin_key": "secret123"}
        mock_outcome = PopulationOutcome(
            results=[
                PopulationResult(
                    container_id="abc123",
                    variation_name=variation.name,
                    description="ok",
                    success=True,
                )
            ],
            episode_context={"addr": "0x123"},
            private_context=mock_private_ctx,
        )
        with (
            patch.object(task, "reset"),
            patch.object(task, "populate", return_value=mock_outcome),
            patch.object(task, "verify_population", return_value=True) as mock_verify,
        ):
            outcome, verified = cm.populate_with_task(
                mock_containers,
                variation,
            )
            mock_verify.assert_called_once_with(
                mock_containers,
                variation,
                {"addr": "0x123"},
                private_context=mock_private_ctx,
            )
            assert verified is True
