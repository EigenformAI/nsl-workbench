# Negative Space Learning v2

An experimental framework for recursive agent learning. An agent attempts tasks, accepted rewrites become training data, and the trained model attempts the next round.

The included materials-adsorption task asks the agent to discover adsorption rules from descriptors of initial atomic structures. Scoring labels are hidden from the agent during an episode. The example uses MS-agent, local Docker Compose, llama.cpp inference, and GPU supervised fine-tuning (SFT).

## Example workflow

The [source setup guide](docs/setup.md) covers prerequisites, locked dependencies, model preparation, container builds, and run commands. The example uses Linux x86-64, Python 3.12, a local Docker Engine with Compose, and an NVIDIA GPU with a compatible CUDA toolchain. Installation is source-only.

The two example configurations serve different purposes:

- [Smoke configuration](config/config-materials-adsorption-smoke.toml): one episode to attempt a training-data export. An accepted rewrite is not guaranteed.
- [Training configuration](config/config-materials-adsorption-train.toml): two generations with one SFT phase between them. Smoke and training use separate output paths and task archives.

The [runtime and recovery reference](docs/runtime.md) describes network exposure, credentials in logs, output state, and recovery after interruption. Repeated smoke attempts need fresh output paths. A new training run identifier still uses the configured task archive.

## Source and data

The source and provenance indexes are:

- [Materials task source index](docs/materials-task.md): task workflow, scoring, archive, and training-row selection.
- [Compact task assets](assets/materials-adsorption/oc20-dense-mini-v1/README.md): dataset provenance, calibration limits, attribution, and checksums.

The framework also includes generic task interfaces, other inference backends, and other training objectives.

## Licenses

No repository-wide first-party license has been selected. Public source availability does not grant general reuse rights. [Third-party notices](THIRD_PARTY_NOTICES.md) identify component-specific terms. The OC20Dense-derived data has separate attribution requirements and CC BY 4.0 terms. Model weights are not included and have their own upstream terms.
