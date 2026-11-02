"""The memory-cleanup reward is pure and target-independent."""

from src.task.types import EpisodeArtifacts, TaskReward
from tasks.memory_cleanup import (
    MemoryCleanupBaseline,
    MemoryCleanupState,
    MemoryCleanupTask,
    ProbeResult,
)


def test_compute_reward_from_plain_frozen_dataclasses():
    baseline = MemoryCleanupBaseline(
        service_name="service-a",
        container_id="c1",
        instance_seed=1,
        variation_name="v",
        theta=0.25,
        disposable_paths=("/tmp/a", "/tmp/b", "/tmp/c", "/tmp/d"),
        disposable_roots=("/tmp",),
        viability_probes=(),
        disposable_count=4,
        populated_allocated_kb=16,
        baseline_processes=("sshd",),
    )
    initial = MemoryCleanupState(
        baseline, 0.0, 16, (ProbeResult("healthy", True, ""),), True, True
    )
    final = MemoryCleanupState(
        baseline, 0.5, 8, (ProbeResult("healthy", True, ""),), True, True
    )

    reward = MemoryCleanupTask({}).compute_reward(
        initial, final, EpisodeArtifacts()
    )

    assert isinstance(reward, TaskReward)
    assert reward.value == 0.5
    assert reward.success is True
    assert reward.breakdown["allocated_kb_freed"] == 8
    assert reward.breakdown["fraction_cleared"] == 0.5
