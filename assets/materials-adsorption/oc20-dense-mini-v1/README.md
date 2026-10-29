# OC20Dense compact materials example

This directory contains a small OC20Dense-derived example for adsorption-rule discovery and its frozen scoring references. It is not the full OC20Dense release and is not a representative scientific benchmark.

## Contents

| File | Purpose | Bytes |
| --- | --- | ---: |
| `dataset.json` | Source identities, initial-structure descriptors, grouping metadata, and DFT adsorption-energy targets | 279,199 |
| `permutation-null-v4.json` | Frozen stage-1 threshold and six control results | 18,909 |
| `shard-null-v2.json` | Evidence-probe null percentile bins and shard reference distributions | 212,462 |
| `attainability-v2.json` | Scores and expressions for 696 calibration candidates | 480,380 |

The dataset has 248 placements across 32 systems, fourteen numeric descriptors, and 32 source groups. Thirty systems have eight placements, one has six, and one has two. There are 182 randomly initialized and 66 heuristically initialized placements.

The dataset version is `oc20-dense-phase0-structure-mini-v1`. The descriptor registry version is `oc20-dense-initial-structure-v1`.

| Evaluation axis | Systems | Placements |
| --- | ---: | ---: |
| In domain (`id`) | 10 | 78 |
| Unseen adsorbate (`ood_adsorbate`) | 9 | 72 |
| Unseen catalyst (`ood_catalyst`) | 7 | 50 |
| Both unseen (`ood_both`) | 6 | 48 |

These axes retain upstream validation provenance. They do not establish statistical representativeness of this small selection.

## Source and attribution

The source is **Open Catalyst 2020 Dense (OC20Dense)**, released by the Open Catalyst Project and introduced in:

Janice Lan, Aini Palizhati, Muhammed Shuaibi, Brandon M. Wood, Brook Wander, Abhishek Das, Matt Uyttendaele, C. Lawrence Zitnick, and Zachary W. Ulissi. *AdsorbML: A Leap in Efficiency for Adsorption Energy Calculations using Generalizable Machine Learning Potentials* (2023). [Published paper](https://doi.org/10.1038/s41524-023-01121-5), [arXiv](https://arxiv.org/abs/2211.16486).

- [Official OC20Dense description and downloads](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20dense.md)
- [Official OC20 description and split mapping](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20.md)
- [OC20Dense mappings and targets](https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_mappings.tar.gz)
- [OC20Dense initial structures](https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_data.tar.gz)
- [OC20 metadata mapping](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl)

The upstream OC20Dense and OC20 data are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode). Retain attribution, the source links, the license link, and an indication of changes when sharing derived data. These dataset terms are separate from the licensing of framework source code.

Changes in this example are deterministic subsampling, conversion of mapping and target data to JSON, calculation of fourteen descriptors from initial structures, assignment of grouping metadata and evaluation axes, and generation of task-specific calibration references. This is an independent derived example. No endorsement by the original authors or the Open Catalyst Project is implied.

## Preparation and sampling

- [Structure preparation and descriptor formulas](../../../scripts/prepare_materials_adsorption_phase0_structures.py)
- [Target preparation and deterministic sampling](../../../scripts/prepare_materials_adsorption_phase0.py)

Targets are taken directly from the OC20Dense DFT adsorption-energy target mapping. Missing-target configurations are not included. This example does not classify the reasons those other configurations lack valid targets.

The preparation scripts check the five official mapping-member MD5 checksums before loading their pickle contents. The mappings archive served during preparation has MD5 `03227241400a465d7c6e9fa7980e480c`, whereas the upstream documentation lists `c18735c405ce6ce5761432b07287d8d9`. All five extracted members match the documented member checksums. The member checks, rather than an asserted archive-wrapper match, establish that mapping input identity.

## Frozen calibration references

- [Stage-1 permutation calibration](../../../scripts/calibrate_materials_adsorption.py)
- [Attainability calibration](../../../scripts/calibrate_materials_adsorption_attainability.py)
- [Evidence-probe calibration](../../../scripts/calibrate_materials_adsorption_shard_null.py)

Calibration expressions are not model-generated discoveries or evidence of scientific generalization. File checksums and internal binding hashes are different: `SHA256SUMS` hashes file bytes, while runtime checks use canonical content manifests.

Changing descriptors, labels, sampling, alias salt, compiler behavior, or scorer settings can invalidate these references. New datasets need new compatible calibrations. The supplied shard-calibration script currently assumes 696 candidates and eleven evidence shards. Larger custom datasets require an appropriate calibration configuration or script adaptation; their scores are not interchangeable with these files.

## Runtime visibility and limitations

These are public benchmark inputs, but they are host-side scoring assets during an episode. `dataset.json` contains targets and source identities. `attainability-v2.json` contains successful expressions and scores. Do not mount this directory, the repository root, or calibration files into the agent or compute sandbox. The compute sandbox receives generated label-free exports only.

The fixed alias salt is public fixture metadata, not a secret or an anti-memorization mechanism. Public availability of targets and successful rules means this example cannot establish resistance to memorization. It supports functional workflow checks, not claims of improved scientific performance or full OC20Dense reproduction.

## Checksums

`SHA256SUMS` covers the four JSON files. From this directory, `sha256sum -c SHA256SUMS` verifies their bytes. The quickstart uses these shipped files and does not require downloading the full upstream structures.
