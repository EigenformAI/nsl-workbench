"""Unit tests for the target-scoped memory-cleanup task."""

import re
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.task.base import TaskEnvironmentError
from src.task.types import EpisodeArtifacts, Variation
from src.training_data.transforms import PerEpisodeRowCap
from tasks import memory_cleanup
from tasks.common.agent_exec import run_python_in_agent
from tasks.memory_cleanup import (
    MemoryCleanupBaseline,
    MemoryCleanupState,
    MemoryCleanupTask,
    MemoryCleanupVariation,
    ProbeResult,
)


def _container(cid: str, service: str = "service-a") -> MagicMock:
    container = MagicMock()
    container.id = cid
    container.name = cid
    container.attrs = {
        "Config": {"Labels": {"com.docker.compose.service": service}},
        "State": {"Running": True},
    }
    return container


@pytest.fixture
def task() -> MemoryCleanupTask:
    return MemoryCleanupTask({})


def _baseline(*, cid: str = "c1", theta: float = 0.5) -> MemoryCleanupBaseline:
    return MemoryCleanupBaseline(
        service_name="service-a",
        container_id=cid,
        instance_seed=7,
        variation_name="v1",
        theta=theta,
        disposable_paths=("/tmp/a", "/tmp/b"),
        disposable_roots=("/tmp",),
        viability_probes=(),
        disposable_count=2,
        populated_allocated_kb=20.0,
        baseline_processes=("sshd",),
    )


def _state(
    baseline: MemoryCleanupBaseline,
    *,
    fraction: float,
    usable: bool = True,
    integrity: bool = True,
) -> MemoryCleanupState:
    return MemoryCleanupState(
        baseline=baseline,
        fraction_cleared=fraction,
        allocated_kb=10.0,
        viability_results=(ProbeResult("container_running", usable, ""),),
        integrity_ok=integrity,
        process_set_ok=usable,
    )


def test_agent_exec_propagates_environment_timeout():
    container = _container("timed-out")
    error = TaskEnvironmentError("timed out", container_ids=["timed-out"])
    with patch(
        "tasks.common.agent_exec.exec_run_with_timeout", side_effect=error
    ):
        with pytest.raises(TaskEnvironmentError) as exc_info:
            run_python_in_agent(container, "print('never runs')")
    assert exc_info.value.container_ids == ["timed-out"]


class TestTaskContract:
    def test_metadata_and_defaults(self, task):
        assert task.metric_name == "fraction_cleared"
        assert task.metric_unit == "fraction"
        assert task.theta == 0.05
        assert task.agent_service_name == "service-a"

    def test_row_cap_transform_is_gated_by_task_config(self):
        assert tuple(MemoryCleanupTask({}).training_data_transforms()) == ()

        transforms = tuple(
            MemoryCleanupTask(
                {"max_rows_per_episode": 12, "keep": "random"}
            ).training_data_transforms()
        )

        assert transforms == (PerEpisodeRowCap(max_rows=12, keep="random"),)

    def test_execute_terminator_capability_declared(self, task):
        """execute_cleanup declares report_result as its semantic terminator,
        and report_result is inside the step's own allowlist."""
        variation = task.list_variations()[0]
        workflow = task.workflow(variation, {})
        steps = {step.name: step for step in workflow.steps}
        execute = steps["execute_cleanup"]
        assert execute.terminator_capabilities == ("report_result",)
        assert "report_result" in execute.capabilities

        # report_result is a real declared capability, so the workflow validates.
        cap_names = {c.name for c in task.prompt_spec(variation, {}).capabilities}
        assert "report_result" in cap_names
        workflow.validate(cap_names)

    def test_code_steps_steer_to_python_not_shell(self, task):
        """Regression: the code-running phases must tell the model to write
        Python (run_python executes as Python). Without this, the model writes
        raw shell (rm/du/df) → SyntaxError → 0 disk freed. See dissection
        2026-07-28: 86% of run_python calls failed as shell-as-Python."""
        variation = task.list_variations()[0]
        steps = {s.name: s for s in task.workflow(variation, {}).steps}
        for name in ("explore_container", "execute_cleanup"):
            prompt = steps[name].prompt.lower()
            assert "python" in prompt, f"{name} must steer to Python"
            assert "subprocess" in prompt, f"{name} must mention subprocess for shell ops"

    def test_execute_step_parses_report_result_terminator(self, task):
        """In the execute phase, a freed-space report is parsed as report_result."""
        invs = task.parse_response(
            "Deleted temp files and freed 512 KB of disk space.",
            invoked_capability="execute_cleanup",
        )
        assert [inv.name for inv in invs] == ["report_result"]
        assert invs[0].input["space_freed_kb"] == 512.0
        # Non-terminal phases do not emit a terminator report.
        assert task.parse_response(
            "Found 512 KB of junk to remove.",
            invoked_capability="explore_container",
        ) == []

    def test_report_result_capability_echoes_payload(self, task):
        from src.task.types import CapabilityExecutionContext, CapabilityInvocation

        result = task.execute_capability(
            CapabilityInvocation(name="report_result", input={"space_freed_kb": 128.0}),
            [],
            task.list_variations()[0],
            CapabilityExecutionContext(
                episode_id="ep", workflow_step="execute_cleanup", episode_context={}
            ),
        )
        assert result.success
        assert result.output == {"space_freed_kb": 128.0}

    def test_variations_have_structured_manifests(self, task):
        for variation in task.list_variations():
            assert isinstance(variation, (MemoryCleanupVariation, Variation))
            assert variation.disposable_objects
            assert all(item.path.startswith("/") for item in variation.disposable_objects)
            assert len({item.path for item in variation.disposable_objects}) == len(
                variation.disposable_objects
            )


class TestThetaResolution:
    @pytest.mark.parametrize(
        ("config", "generation_id", "expected"),
        [
            ({"theta_base": 0.2, "theta_schedule": "constant"}, 9, 0.2),
            (
                {
                    "theta_base": 0.1,
                    "theta_schedule": "linear",
                    "theta_step": 0.2,
                    "theta_cap": 0.55,
                },
                3,
                0.55,
            ),
            (
                {
                    "theta_schedule": "explicit",
                    "theta_explicit": [0.1, 0.4, 0.8],
                    "theta_cap": 0.6,
                },
                2,
                0.6,
            ),
        ],
    )
    def test_schedule_resolution(self, config, generation_id, expected):
        resolved = MemoryCleanupTask.resolve_generation_task_config(
            config, generation_id, 10
        )
        assert resolved["theta"] == expected

    @pytest.mark.parametrize(
        "config",
        [
            {"theta_base": -0.1},
            {"theta_base": 1.1},
            {"theta_schedule": "unknown"},
            {"theta_schedule": "linear", "theta_step": -0.1},
            {"theta_schedule": "explicit", "theta_explicit": []},
        ],
    )
    def test_invalid_schedule_rejected(self, config):
        with pytest.raises(ValueError):
            MemoryCleanupTask(config)

    def test_constant_schedule_preserves_top_level_theta(self):
        resolved = MemoryCleanupTask.resolve_generation_task_config(
            {"theta": 0.8},
            generation_id=4,
            num_generations=5,
        )

        assert resolved["theta"] == 0.8


class TestTrackingConfig:
    def test_contains_complete_effective_theta_config(self):
        assert MemoryCleanupTask({}).tracking_config() == {
            "theta": 0.05,
            "theta_schedule": "constant",
            "theta_base": 0.05,
            "theta_step": 0.0,
            "theta_cap": 0.95,
            "max_rows_per_episode": None,
            "row_cap_keep": None,
        }

        assert MemoryCleanupTask(
            {"max_rows_per_episode": 12, "keep": "random"}
        ).tracking_config() == {
            "theta": 0.05,
            "theta_schedule": "constant",
            "theta_base": 0.05,
            "theta_step": 0.0,
            "theta_cap": 0.95,
            "max_rows_per_episode": 12,
            "row_cap_keep": "random",
        }

    @pytest.mark.parametrize(
        ("generation_id", "expected_theta"),
        [(0, 0.1), (1, 0.3), (2, 0.5), (3, 0.55)],
    )
    def test_reflects_resolved_theta_per_generation(
        self, generation_id, expected_theta
    ):
        resolved = MemoryCleanupTask.resolve_generation_task_config(
            {
                "theta_schedule": "linear",
                "theta_base": 0.1,
                "theta_step": 0.2,
                "theta_cap": 0.55,
            },
            generation_id=generation_id,
            num_generations=4,
        )

        assert MemoryCleanupTask(resolved).tracking_config()["theta"] == pytest.approx(
            expected_theta
        )

    def test_omits_paths_environment_and_arbitrary_values(self):
        tracking = MemoryCleanupTask(
            {
                "cleanup_paths": ["/secret/path"],
                "docker_compose_dir": "/private/compose",
                "credential": "do-not-track",
            }
        ).tracking_config()

        assert set(tracking) == {
            "theta",
            "theta_schedule",
            "theta_base",
            "theta_step",
            "theta_cap",
            "max_rows_per_episode",
            "row_cap_keep",
        }


class TestTargetResolution:
    def test_reversed_container_order_selects_agent_service(self, task):
        sibling = _container("sibling", "service-b")
        target = _container("target")
        assert task._resolve_target([sibling, target]) is target

    @pytest.mark.parametrize(
        "containers",
        [[], [_container("sibling", "service-b")], [_container("a"), _container("b")]],
    )
    def test_zero_or_duplicate_target_is_infra_error(self, task, containers):
        with pytest.raises(TaskEnvironmentError):
            task._resolve_target(containers)


class TestPopulation:
    @staticmethod
    def _anchor_fragment(task, variation) -> str:
        lines = task._render_population_script(variation, "nonce").splitlines()
        start = next(
            index for index, line in enumerate(lines) if "sha256sum --" in line
        )
        end = next(
            index
            for index, line in enumerate(lines[start:], start)
            if "__POPULATE_COMPLETE__" in line
        )
        return "set -u\n" + "\n".join(lines[start:end])

    def test_anchor_hash_failure_exits_nonzero_without_emitting_anchor(
        self, task, tmp_path: Path
    ):
        variation = task.list_variations()[0]
        missing = tmp_path / "missing-anchor"

        with patch.object(memory_cleanup, "_INTEGRITY_ANCHORS", (str(missing),)):
            fragment = self._anchor_fragment(task, variation)
            result = subprocess.run(
                ["sh", "-c", fragment],
                capture_output=True,
                text=True,
                check=False,
            )

        assert result.returncode != 0
        assert "__ANCHOR_SHA256__=" not in result.stdout

    def test_anchor_hash_success_emits_one_valid_sentinel(
        self, task, tmp_path: Path
    ):
        variation = task.list_variations()[0]
        anchor = tmp_path / "anchor"
        anchor.write_bytes(b"anchor contents")

        with patch.object(memory_cleanup, "_INTEGRITY_ANCHORS", (str(anchor),)):
            fragment = self._anchor_fragment(task, variation)
            result = subprocess.run(
                ["sh", "-c", fragment],
                capture_output=True,
                text=True,
                check=False,
            )

        assert result.returncode == 0
        assert re.fullmatch(
            rf"__ANCHOR_SHA256__={re.escape(str(anchor))}:[0-9a-f]{{64}}\n",
            result.stdout,
        )

    def test_population_is_one_ordered_exec_and_manifest_scoped(self, task):
        target = _container("target")
        sibling = _container("sibling", "service-b")
        variation = task.list_variations()[1]
        assert isinstance(variation, MemoryCleanupVariation)
        output = task._population_success_output(variation, "nonce", 0, 100)

        with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"):
            with patch(
                "tasks.memory_cleanup._exec_run_with_timeout",
                return_value=(0, output.encode()),
            ) as exec_run:
                outcome = task.populate([sibling, target], variation)

        exec_run.assert_called_once()
        assert exec_run.call_args.args[0] is target
        script = exec_run.call_args.args[1][-1]
        assert "conv=fdatasync" not in script
        assert "\nsync\n" not in script
        assert "/home/alice/*" not in script
        assert "/home/alice/venv" not in script
        assert len(script) < 16_000
        assert all(root in script for root in variation.created_dirs)
        assert "__POPULATE_FAILED__=$i:$path" in script
        assert outcome.results[0].success is True
        baseline = outcome.private_context["memory_cleanup_baseline"]
        assert baseline.container_id == "target"

    def test_population_clears_all_task_owned_roots_before_repopulation(self, task):
        variation = task.list_variations()[0]

        script = task._render_population_script(variation, "nonce")

        for candidate in task.list_variations():
            for root in candidate.created_dirs:
                assert f"find {root}" in script
        assert "-mindepth 1 -maxdepth 1 -exec rm -rf -- {} +" in script

    def test_failed_marker_raises_typed_error(self, task):
        target = _container("target")
        variation = task.list_variations()[0]
        with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"):
            with patch(
                "tasks.memory_cleanup._exec_run_with_timeout",
                return_value=(1, b"__POPULATE_FAILED__=12:/tmp/bad\n"),
            ):
                with pytest.raises(TaskEnvironmentError) as exc_info:
                    task.populate([target], variation)
        assert exc_info.value.container_ids == ["target"]
        assert "/tmp/bad" in str(exc_info.value)

    def test_duplicate_anchor_sentinel_is_rejected(self, task):
        target = _container("target")
        variation = task.list_variations()[0]
        output = task._population_success_output(variation, "nonce", 0, 100)
        duplicate = (
            f"__ANCHOR_SHA256__={memory_cleanup._INTEGRITY_ANCHORS[0]}:"
            f"{'1' * 64}\n"
        )

        with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"):
            with patch(
                "tasks.memory_cleanup._exec_run_with_timeout",
                return_value=(0, (output + duplicate).encode()),
            ):
                with pytest.raises(TaskEnvironmentError, match="duplicate"):
                    task.populate([target], variation)

    def test_malformed_anchor_error_identifies_candidate(self, task):
        target = _container("target")
        variation = task.list_variations()[0]
        path = memory_cleanup._INTEGRITY_ANCHORS[0]
        output = task._population_success_output(variation, "nonce", 0, 100)
        output = output.replace(
            f"__ANCHOR_SHA256__={path}:{'0' * 64}",
            f"__ANCHOR_SHA256__={path}:",
        )

        with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"):
            with patch(
                "tasks.memory_cleanup._exec_run_with_timeout",
                return_value=(0, output.encode()),
            ):
                with pytest.raises(TaskEnvironmentError) as exc_info:
                    task.populate([target], variation)

        assert path in str(exc_info.value)
        assert "candidates=" in str(exc_info.value)

    def test_missing_completion_nonce_replaces_target(self, task):
        target = _container("target")
        variation = task.list_variations()[0]
        with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"):
            with patch(
                "tasks.memory_cleanup._exec_run_with_timeout",
                return_value=(0, b"__BASELINE_KB__=0\n__POPULATED_KB__=10\n"),
            ):
                with pytest.raises(TaskEnvironmentError) as exc_info:
                    task.populate([target], variation)
        assert exc_info.value.container_ids == ["target"]

    def test_reset_does_not_mutate_filesystem(self, task):
        target = _container("target")
        with patch("tasks.memory_cleanup._exec_run_with_timeout") as exec_run:
            task.reset([target])
        exec_run.assert_not_called()


class TestReward:
    @pytest.mark.parametrize(
        ("fraction", "usable", "integrity", "success"),
        [
            (0.0, True, True, False),
            (0.49, True, True, False),
            (0.5, True, True, True),
            (1.0, False, True, False),
            (1.0, True, False, False),
        ],
    )
    def test_viability_effort_and_integrity_gate(
        self, task, fraction, usable, integrity, success
    ):
        baseline = _baseline()
        initial = _state(baseline, fraction=0.0)
        final = _state(
            baseline, fraction=fraction, usable=usable, integrity=integrity
        )
        reward = task.compute_reward(initial, final, EpisodeArtifacts())
        assert reward.value == fraction
        assert reward.success is success

    def test_mismatched_baseline_identity_rejected(self, task):
        with pytest.raises(ValueError, match="baseline"):
            task.compute_reward(
                _state(_baseline(cid="a"), fraction=0),
                _state(_baseline(cid="b"), fraction=1),
                EpisodeArtifacts(),
            )

    def test_baseline_is_immutable(self):
        baseline = _baseline()
        with pytest.raises(FrozenInstanceError):
            baseline.theta = 1.0  # type: ignore[misc]


class TestAllocatedKbIsBestEffort:
    """Allocated-space diagnostics must not gate admission or scoring."""

    def _healthy_probe(self):
        return (
            True,
            (ProbeResult("container_running", True, ""),),
            True,
            ("sshd",),
        )

    def test_measure_final_state_scores_when_du_fails(self, task):
        target = _container("target")
        baseline = _baseline(cid="target")
        with patch.object(task, "_run_probe_script", return_value=self._healthy_probe()), \
            patch.object(task, "_count_absent", return_value=2), \
            patch.object(
                task,
                "_measure_allocated_kb",
                side_effect=TaskEnvironmentError("du exited non-zero"),
            ):
            state = task.measure_final_state(
                [target],
                {},
                EpisodeArtifacts(),
                private_context={"memory_cleanup_baseline": baseline},
            )
        assert isinstance(state, MemoryCleanupState)
        assert state.allocated_kb is None
        assert state.fraction_cleared == 1.0
        assert state.integrity_ok is True

    def test_verify_population_admits_when_du_fails(self, task):
        target = _container("target")
        baseline = _baseline(cid="target")
        variation = task.list_variations()[0]
        baseline = MemoryCleanupBaseline(
            **{**baseline.__dict__, "variation_name": variation.name}
        )
        with patch.object(task, "_run_probe_script", return_value=self._healthy_probe()), \
            patch.object(task, "_count_absent", return_value=0), \
            patch.object(
                task,
                "_measure_allocated_kb",
                side_effect=TaskEnvironmentError("du exited non-zero"),
            ):
            assert (
                task.verify_population(
                    [target],
                    variation,
                    {},
                    private_context={"memory_cleanup_baseline": baseline},
                )
                is True
            )

    def test_compute_reward_tolerates_absent_allocated_diagnostic(self, task):
        baseline = _baseline()
        initial = _state(baseline, fraction=0.0)
        final = MemoryCleanupState(
            baseline=baseline,
            fraction_cleared=1.0,
            allocated_kb=None,
            viability_results=(ProbeResult("container_running", True, ""),),
            integrity_ok=True,
            process_set_ok=True,
        )
        reward = task.compute_reward(initial, final, EpisodeArtifacts())
        assert reward.success is True
        assert reward.breakdown["allocated_kb_freed"] is None


class TestCountAbsentRetriesTransientExecLoss:
    """Retry transient failures of the idempotent manifest count."""

    def test_retries_missing_sentinel_then_succeeds(self, task):
        target = _container("target")
        results = [(0, b""), (0, b"garbled partial output\n"), (0, b"__ABSENT_COUNT__=3\n")]
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout", side_effect=results
        ) as exec_run, patch("tasks.memory_cleanup.time.sleep"):
            count = task._count_absent(target, ("/tmp/a", "/tmp/b", "/tmp/c"))
        assert count == 3
        assert exec_run.call_count == 3

    def test_retries_transient_nonzero_exit_then_succeeds(self, task):
        target = _container("target")
        results = [(127, b"OCI runtime exec failed\n"), (0, b"__ABSENT_COUNT__=1\n")]
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout", side_effect=results
        ), patch("tasks.memory_cleanup.time.sleep"):
            assert task._count_absent(target, ("/tmp/x",)) == 1

    def test_first_attempt_success_does_not_retry(self, task):
        target = _container("target")
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(0, b"__ABSENT_COUNT__=2\n"),
        ) as exec_run, patch("tasks.memory_cleanup.time.sleep"):
            assert task._count_absent(target, ("/tmp/x", "/tmp/y")) == 2
        exec_run.assert_called_once()

    def test_raises_after_persistent_failure_with_captured_detail(self, task):
        target = _container("target")
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout", return_value=(0, b"")
        ) as exec_run, patch("tasks.memory_cleanup.time.sleep"):
            with pytest.raises(TaskEnvironmentError) as exc_info:
                task._count_absent(target, ("/tmp/x",))
        assert exc_info.value.container_ids == ["target"]
        assert "attempt" in str(exc_info.value).lower()
        assert exec_run.call_count >= 2

    def test_docker_exec_timeout_is_not_retried(self, task):
        target = _container("target")
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            side_effect=TaskEnvironmentError("Docker exec timed out", container_ids=["target"]),
        ) as exec_run, patch("tasks.memory_cleanup.time.sleep"):
            with pytest.raises(TaskEnvironmentError):
                task._count_absent(target, ("/tmp/x",))
        exec_run.assert_called_once()
