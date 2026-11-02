"""Behavioral tests for cleanup's fixture and scoring contract."""

import re
import threading
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from src.task.base import TaskEnvironmentError
from src.task.types import CapabilityExecutionContext, CapabilityInvocation, EpisodeArtifacts
from src.training_data import PerEpisodeRowCap
from tasks import memory_cleanup
from tasks.common.agent_exec import run_python_in_agent
from tasks.memory_cleanup import MemoryCleanupTask


def _container(cid="target"):
    target = MagicMock()
    target.id = target.name = cid
    target.attrs = {
        "Config": {"Labels": {"com.docker.compose.service": "service-a"}},
        "State": {"Running": True},
    }
    return target


def _populated(task, target, variation):
    output = task._population_success_output(variation, "nonce", 0, 100)
    with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"), patch(
        "tasks.memory_cleanup._exec_run_with_timeout", return_value=(0, output.encode())
    ):
        return task.populate([target], variation)


def test_all_five_disposable_manifests_have_the_expected_counts():
    from tasks.memory_cleanup import MemoryCleanupTask

    variations = MemoryCleanupTask({}).list_variations()

    assert [len(variation.disposable_objects) for variation in variations] == [
        140, 190, 690, 45, 118,
    ]
    assert all(
        len({item.path for item in variation.disposable_objects})
        == len(variation.disposable_objects)
        for variation in variations
    )


@pytest.mark.parametrize("keep", ["head", "tail", "even", "random"])
def test_optional_cap_forwards_the_requested_selection_strategy(keep):
    assert MemoryCleanupTask({"keep": keep}).training_data_transforms() == ()
    assert MemoryCleanupTask({"max_rows_per_episode": 2, "keep": keep}).training_data_transforms() == (
        PerEpisodeRowCap(2, keep=keep),
    )


def test_explicit_theta_schedule_requires_an_entry_for_the_requested_generation():
    with pytest.raises(ValueError, match="does not contain generation 2"):
        MemoryCleanupTask.resolve_generation_task_config(
            {"theta_schedule": "explicit", "theta_explicit": [0.1, 0.2]}, 2, 3
        )


@pytest.mark.parametrize("variation_index", range(5))
def test_population_and_partial_full_path_measurements_for_each_manifest(variation_index):
    task = MemoryCleanupTask({"theta": 0.5})
    target = _container()
    variation = task.list_variations()[variation_index]
    outcome = _populated(task, target, variation)
    baseline = outcome.private_context["memory_cleanup_baseline"]
    assert baseline.disposable_paths == tuple(item.path for item in variation.disposable_objects)
    assert baseline.theta == 0.5
    assert baseline.variation_name == variation.name
    assert "memory_cleanup_baseline" not in outcome.episode_context
    healthy = (True, (memory_cleanup.ProbeResult("healthy", True),), True, baseline.baseline_processes)
    with patch.object(task, "_run_probe_script", return_value=healthy), patch.object(
        task, "_measure_population_kb", return_value=None
    ):
        initial = task.measure_initial_state([target], {}, private_context=outcome.private_context)
        for absent in (0, len(variation.disposable_objects) // 2, len(variation.disposable_objects)):
            with patch.object(task, "_count_absent", return_value=absent) as count:
                final = task.measure_final_state([target], {}, EpisodeArtifacts(), private_context=outcome.private_context)
            count.assert_called_once_with(target, baseline.disposable_paths)
            reward = task.compute_reward(initial, final, EpisodeArtifacts(final_response="I freed 999999999 KB"))
            assert reward.value == absent / len(variation.disposable_objects)
            assert reward.success is (absent / len(variation.disposable_objects) >= 0.5)


@pytest.mark.parametrize(("mutation", "error"), [
    (lambda text: text.replace("__POPULATED_COUNT__=140", "__POPULATED_COUNT__=139"), "count mismatch"),
    (lambda text: text.replace("__POPULATE_COMPLETE__=nonce", "__POPULATE_COMPLETE__=wrong"), "completion nonce"),
    (lambda text: text.replace("__POPULATED_KB__=100", "__POPULATED_KB__=broken"), "Malformed population"),
    (lambda text: text + "__BASELINE_KB__=10\n", "Malformed population"),
    (lambda text: re.sub(r"__ANCHOR_SHA256__=/usr/sbin/sshd:[^\n]*\n", "", text), "missing="),
    (lambda text: text.replace("__ANCHOR_SHA256__=/usr/sbin/sshd:", "__ANCHOR_SHA256__=/unexpected:"), "unexpected="),
])
def test_population_rejects_incomplete_or_mismatched_sentinels(mutation, error):
    task = MemoryCleanupTask({})
    target = _container()
    variation = task.list_variations()[0]
    output = mutation(task._population_success_output(variation, "nonce", 0, 100))
    with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"), patch(
        "tasks.memory_cleanup._exec_run_with_timeout", return_value=(0, output.encode())
    ), pytest.raises(TaskEnvironmentError, match=error) as caught:
        task.populate([target], variation)
    assert caught.value.container_ids == [target.id]


def test_baseline_must_match_container_and_variation_before_probes():
    task = MemoryCleanupTask({})
    target = _container()
    variations = task.list_variations()
    outcome = _populated(task, target, variations[0])
    with patch.object(task, "_run_probe_script") as probes:
        with pytest.raises(TaskEnvironmentError, match="target identity"):
            task.measure_initial_state([_container("other")], {}, private_context=outcome.private_context)
        with pytest.raises(TaskEnvironmentError, match="variation identity"):
            task.verify_population([target], variations[1], {}, private_context=outcome.private_context)
        with pytest.raises(TaskEnvironmentError, match="Missing immutable"):
            task.measure_initial_state([target], {})
    probes.assert_not_called()


@pytest.mark.parametrize(("usable", "integrity", "absent"), [(False, True, 0), (True, False, 0), (True, True, 1)])
def test_population_admission_requires_viability_integrity_and_every_original_path(usable, integrity, absent):
    task = MemoryCleanupTask({})
    target = _container()
    variation = task.list_variations()[0]
    outcome = _populated(task, target, variation)
    with patch.object(task, "_run_probe_script", return_value=(usable, (), integrity, ())), patch.object(
        task, "_count_absent", return_value=absent
    ), patch.object(task, "_measure_population_kb", return_value=100):
        with pytest.raises(TaskEnvironmentError, match="viability admission"):
            task.verify_population([target], variation, {}, private_context=outcome.private_context)


def test_unexpected_processes_fail_reward_without_changing_path_fraction():
    task = MemoryCleanupTask({})
    target = _container()
    outcome = _populated(task, target, task.list_variations()[0])
    baseline = outcome.private_context["memory_cleanup_baseline"]
    baseline = replace(baseline, baseline_processes=("sshd",))
    context = {"memory_cleanup_baseline": baseline}
    with patch.object(task, "_run_probe_script", return_value=(True, (), True, ("sshd",))), patch.object(
        task, "_measure_population_kb", return_value=None
    ), patch.object(task, "_count_absent", return_value=baseline.disposable_count):
        initial = task.measure_initial_state([target], {}, private_context=context)
        with patch.object(task, "_run_probe_script", return_value=(True, (), True, ("sshd", "new-daemon"))):
            final = task.measure_final_state([target], {}, EpisodeArtifacts(), private_context=context)
    reward = task.compute_reward(initial, final, EpisodeArtifacts())
    assert reward.value == 1.0
    assert not reward.success
    assert not reward.breakdown["process_set_ok"]


def test_measurement_exec_uses_root_workdir_and_preserves_explicit_override():
    target = _container()
    target.exec_run.return_value = (0, b"done")
    assert memory_cleanup._exec_run_with_timeout(target, ["true"]) == (0, b"done")
    target.exec_run.assert_called_with(["true"], workdir="/")
    memory_cleanup._exec_run_with_timeout(target, ["true"], workdir="/var")
    target.exec_run.assert_called_with(["true"], workdir="/var")


def test_measurement_timeout_is_typed_and_reports_target_identity():
    target = _container()
    release = threading.Event()
    target.exec_run.side_effect = lambda *args, **kwargs: release.wait(5)
    try:
        with pytest.raises(TaskEnvironmentError, match="timed out") as caught:
            memory_cleanup._exec_run_with_timeout(target, ["sleep", "5"], timeout_s=0.01)
        assert caught.value.container_ids == [target.id]
    finally:
        release.set()


def test_manifest_count_retries_transport_exceptions_but_has_exactly_three_attempts():
    task = MemoryCleanupTask({})
    target = _container()
    with patch("tasks.memory_cleanup._exec_run_with_timeout", side_effect=RuntimeError("lost exec")) as execute, patch(
        "tasks.memory_cleanup.time.sleep"
    ), pytest.raises(TaskEnvironmentError, match="after 3 attempts"):
        task._count_absent(target, ("/tmp/example",))
    assert execute.call_count == 3


@pytest.mark.parametrize("raw_output", [(b"hello\n", b"warning\n"), b"hello\n"])
def test_agent_python_helper_stages_then_executes_python_without_measurement_workdir(raw_output):
    target = _container()
    with patch("tasks.common.agent_exec.exec_run_with_timeout", side_effect=[(0, b""), (0, raw_output)]) as execute:
        result = run_python_in_agent(target, "print('hello')", timeout_s=7)
    assert execute.call_args_list[1].args == (target, ["python3", "-u", "/tmp/_nsl_script.py"])
    assert execute.call_args_list[1].kwargs == {"timeout_s": 7, "demux": True}
    assert result["stdout"] == "hello"
    assert result["success"]
    assert result["executed_code"] == "print('hello')"
    assert result["stderr"] == ("warning" if isinstance(raw_output, tuple) else "")


@pytest.mark.parametrize("failure", [(1, b"read-only"), RuntimeError("transport failure")])
def test_agent_python_staging_failure_does_not_execute_script(failure):
    with patch("tasks.common.agent_exec.exec_run_with_timeout", side_effect=[failure]) as execute:
        result = run_python_in_agent(_container(), "print('never')")
    assert not result["success"]
    assert result["stderr"]
    execute.assert_called_once()


def test_task_python_tool_preserves_python_code_and_typed_environment_failures():
    task = MemoryCleanupTask({})
    target = _container()
    variation = task.list_variations()[0]
    context = CapabilityExecutionContext(episode_id="ep", workflow_step="execute_cleanup", episode_context={})
    invocation = CapabilityInvocation(name="run_python", input={"code": "print('ok')", "timeout_s": 9})
    error = TaskEnvironmentError("timeout", container_ids=[target.id])
    with patch("tasks.memory_cleanup.run_python_in_agent", side_effect=error) as execute:
        with pytest.raises(TaskEnvironmentError) as caught:
            task.execute_capability(invocation, [target], variation, context)
    assert caught.value is error
    execute.assert_called_once_with(target, "print('ok')", timeout_s=9)
    with patch("tasks.memory_cleanup.run_python_in_agent") as execute:
        result = task.execute_capability(replace(invocation, input={"code": "  "}), [target], variation, context)
    assert not result.success
    execute.assert_not_called()
