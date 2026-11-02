# Cleanup task reference

`tasks.memory_cleanup.MemoryCleanupTask` asks an agent to remove disposable files from an Alpine container without breaking its Python environment, SSH service, or integrity anchors. It has five variations, containing 140, 190, 690, 45, and 118 disposable paths.

## Workflow and scoring

The task declares three phases: explore, plan without tools, and execute. `run_python` accepts Python source, not shell text. Shell commands can run through Python's `subprocess` module. `report_result` signals completion but does not determine reward.

The score is the fraction of original disposable paths that no longer exist. Success requires that fraction to reach `theta`, with viability, integrity, and process checks passing. Moving a file removes its original path. Compressing it in place does not. Allocated KB are diagnostic only. A failed `du` probe produces a missing diagnostic, not an automatic unsuccessful reward.

Checks compare protected Python and SSH files with their initial hashes, verify that the environment still works, and reject process names absent from the baseline.

Population clears the roots owned by all five variations, creates the selected fixture, records a private container-specific baseline, and verifies that the populated environment is usable. Baselines are not shared between parallel slots. Population does not reset arbitrary changes elsewhere in a reused container.

## Task configuration

The task class remains explicit:

```toml
[task]
class = "tasks.memory_cleanup.MemoryCleanupTask"

[task.config]
docker_compose_dir = "./docker/special-learn-compose"
theta = 0.05
theta_schedule = "constant"
```

`theta_schedule` accepts `constant`, `linear`, and `explicit`. Constant mode honors `theta`, falling back to `theta_base` (default `0.05`). Linear mode uses `theta_base + generation_id * theta_step`. Explicit mode selects `theta_explicit[generation_id]`. All modes apply `theta_cap` (default `0.95`). Generation IDs start at zero.

`max_rows_per_episode` optionally caps successful training rows. `keep` accepts `head`, `tail`, `even`, and `random` (default). Without a cap, cleanup adds no training-data transform. Random selection uses a deterministic hash rank, then preserves source-row order. The cap does not write candidate manifests or selection audit files.

## Harnesses and examples

- [Native smoke](../config/config-memory-cleanup-smoke.toml) uses task-owned prompts and capabilities with one shared orchestrator-decision budget. A reasoning-only phase advances after one worker turn. Reservation and repetition rules can advance unfinished phases. Workflow completion alone is not task success.
- [MS-agent smoke](../config/config-memory-cleanup-msagent-smoke.toml) uses a separate harness container and the same task workflow.
- [Recursive training](../config/config-memory-cleanup-train.toml) uses MS-agent, three generations, and two training phases: ordinary supervised token loss (`sft`) after generation zero and probability-weighted token loss (`dft`) after generation one. Generation two collects with the final trained model. Output paths differ from both smoke examples.

Native workflows support inherited context and normal-success chains. Error routing and isolated context are rejected before inference. Materials continues to use MS-agent for its isolated phases.

All examples use the matching llama.cpp and Transformers inputs prepared by the [source setup guide](setup.md). A successful episode is not guaranteed. Repeated independent runs require fresh output paths. Rebuild and restart intervals remain configurable rather than requiring a fresh container per episode.

## Container behavior and limitations

The Compose service is `service-a`, with configured container name `special-learn-compose_service-a_1`. Its Python environment is `/home/alice/venv`. The task container is separate from an MS-agent harness container. Native orchestration runs on the framework host.

The fixture image is intentionally unchanged. Task actions execute as root. Its SSH setup permits passwordless access, generates host keys during the build, and uses floating dependencies. It is a disposable task environment, not the restricted materials compute sandbox. The source fixture scripts can fail during an image build. No general sandbox-hardening or complete-reset guarantee is provided.

Measurement uses working directory `/` and makes up to three attempts for transient manifest-count failures. Manifest-count and viability probes propagate typed execution timeouts. Allocated-space diagnostics instead catch those errors and return a missing diagnostic. Agent execution retains the helper's working-directory behavior. Deleting the home or breaking Python can fail viability checks. A stopped or unexecutable container can produce an infrastructure error instead of a score. Timeout recovery replaces the affected target. A timed-out Docker exec can remain alive until replacement.

## Run the examples

After preparing the model and host dependencies from the setup guide, build the task image:

```bash
docker compose -f docker/special-learn-compose/docker-compose.yml build
```

Run either smoke example:

```bash
uv run python scripts/generate_training_data.py --config config/config-memory-cleanup-smoke.toml
uv run python scripts/generate_training_data.py --config config/config-memory-cleanup-msagent-smoke.toml
```

Run recursive training:

```bash
uv run python scripts/run_train_loop.py --config config/config-memory-cleanup-train.toml
```

The container harness builds from `docker/ms-agent` when needed. Both smoke configurations permit only one episode. See [runtime and recovery](runtime.md) for interruption, credentials, and output-state behavior.
