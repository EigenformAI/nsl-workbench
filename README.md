# Negative Space Learning v2

`negative-space-learning-v2` is a framework for recursive agent learning: generate task trajectories, train on accepted rewrites, and generate the next round with the trained model. This snapshot includes a materials-adsorption task using MS-agent, local Docker Compose, llama.cpp inference, and GPU SFT training.

The materials task asks an agent to discover adsorption rules from pre-DFT descriptors without access to scoring labels.

## Run the example

Follow [Set up a source checkout](docs/setup.md). Choose one smoke entry point from the repository root.

To attempt one episode without final export publication:

```bash
uv run python scripts/run_episode.py config/config-materials-adsorption-smoke.toml
```

Alternatively, to attempt one episode and publish its export:

```bash
uv run python scripts/generate_training_data.py --config config/config-materials-adsorption-smoke.toml
```

Do not run both smoke entry points against the same existing generation directory. To repeat smoke, use fresh smoke output paths.

To attempt recursive training with its separate state:

```bash
uv run python scripts/run_train_loop.py --config config/config-materials-adsorption-train.toml
```

Smoke outputs and the smoke archive are separate from training outputs and the training archive. Read [Runtime and recovery](docs/runtime.md) before stopping or resuming a run.

## Read more

- [Materials task source index](docs/materials-task.md).
- [Runtime and recovery](docs/runtime.md): network access, credentials, and resume behavior.
- [Compact task assets](assets/materials-adsorption/oc20-dense-mini-v1/README.md): dataset provenance and checksums.

Generic task interfaces, the orchestration harness, other inference backends, and SFT, DFT, and ASFT objectives remain available. Installation is source-only on Linux x86-64 with Python 3.12. No wheel-installation workflow is provided.

## Licenses

No repository-wide first-party license has been selected. Public source availability does not grant general reuse rights. [Third-party notices](THIRD_PARTY_NOTICES.md) identify component-specific terms. The OC20Dense-derived data has separate attribution and CC-BY-4.0 terms. Model weights have their own upstream terms and are not included.
