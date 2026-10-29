# Generated label-free evidence

This directory is empty scaffolding in the source checkout. No populated CSV files are shipped here.

Task population writes slot-local `universe.csv` and `evidence.csv` exports in the configured runtime export directory. `universe.csv` contains candidate aliases and initial-structure descriptors. `evidence.csv` contains the episode's evidence shard using the same aliases. The runtime supplies the rule contract for the configured scorer and calibration.

Only generated label-free exports are mounted read-only into the network-disabled compute sandbox. The host dataset, calibration bundle, repository root, model cache, and host Docker socket must not be mounted into the agent or compute sandbox. The host dataset and calibration files are public distribution assets, but they contain scoring targets, source identities, and successful calibration expressions that are not episode-visible inputs.

The build context excludes `public-evidence/` to avoid transmitting generated exports to the builder. Runtime export mounts, not this scaffolding directory, supply episode data.
