# Materials task

The materials-adsorption task asks an agent to propose a rule over pre-DFT descriptors of anonymized adsorption placements. The framework evaluates the rule against host-side scoring data. It trains on accepted rewrites rather than the complete exploration transcript.

## Compact inputs

The included `assets/materials-adsorption/oc20-dense-mini-v1/` bundle contains 248 placements across 32 systems. Its dataset version is `oc20-dense-phase0-structure-mini-v1`. This is a compact OC20Dense-derived example, not the full upstream dataset or a paper reproduction.

The runtime loads four matching files: `dataset.json`, `permutation-null-v4.json`, `shard-null-v2.json`, and `attainability-v2.json`. Frozen calibrations bind the scoring configuration to the dataset. Replacing the data requires compatible recalibration, not disabling validation. The [asset README](../assets/materials-adsorption/oc20-dense-mini-v1/README.md) gives provenance and checksums.

## Episode workflow

The task supplies five ordered workflow steps:

1. `explore`: inspect descriptor evidence and record an evidence plan.
2. `hypothesise`: propose a descriptor-based hypothesis.
3. `code`: compile a feature and select the compiled rule.
4. `verify`: evaluate the selected rule with host-side scoring.
5. `rewrite`: read the accepted hypothesis and submit a concise training rewrite.

Task capabilities and budgets constrain each step. A tool call that succeeds mechanically does not establish task success. Scoring applies support and calibrated signal checks before a hypothesis qualifies for rewriting. A valid terminal rewrite is also required.

The host archive stores hypotheses, rules, relationships, and receipts. Later episodes can use permitted archive records. Smoke and train configurations use different archives so smoke exploration does not change training state.

## Agent-visible data

MS-agent receives prompts and task capabilities through authenticated host bridges. The compute sandbox exposes generated label-free CSV evidence at `/data`. The host runs Python snippets there through Docker exec.

The host dataset includes scoring labels and source identities. Neither the dataset nor the calibration bundle is mounted into the agent or compute container. The repository root, model cache, and host Docker socket are not agent mounts. The network-disabled compute sandbox is distinct from the network-enabled MS-agent harness.

Public distribution of benchmark labels does not make them agent-visible during an episode. It also cannot prevent memorization or label recovery by a malicious operator who controls the host.

## Training-row selection

Exports begin with successful episodes. `MaterialsRewriteTransform` retains real assistant inference rows from the `rewrite` step. It does not convert tool results, receipts, or failed exploration into synthetic training responses. The transform rebuilds each row with allowlisted public provenance.

The generation target counts exported training rows, not episodes, proposals, or tool calls. The smoke target is one row within one episode. The train target is one row within at most 16 episodes per generation. Reaching the episode cap without usable rows produces no training data.

The train loop consumes the active SFT export, fits the configured Transformers base, and exports merged GGUF weights. Generation 1 must use that trained artifact. With two generations, the loop fits once after generation 0. It does not train after the final generation.

The example uses SFT. DFT and ASFT remain configurable.
