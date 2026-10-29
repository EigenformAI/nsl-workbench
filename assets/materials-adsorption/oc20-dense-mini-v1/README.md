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

`scripts/prepare_materials_adsorption_phase0_structures.py` derives descriptors from the official initial-structure LMDB. It matches surface identities to OC20 validation metadata using adsorbate composition, Materials Project bulk ID, Miller indices, surface shift, and top/bottom identity. It maps the upstream validation categories to the four axes above.

`scripts/prepare_materials_adsorption_phase0.py` joins the official mapping and target files. Systems with fewer than two available targets are excluded. Eligible system IDs are ordered by the raw SHA-256 digest of their UTF-8 string, with the ID as a tie-breaker. The first 32 systems are selected. Placements within each system are ordered by the SHA-256 digest of `system_id + NUL + config_id`, with that string as a tie-breaker. The first eight available placements are retained. This produces 248 placements, not 256, because two selected systems have fewer than eight targets.

Targets are taken directly from the OC20Dense DFT adsorption-energy target mapping. Missing-target configurations are not included. This example does not classify the reasons those other configurations lack valid targets.

The preparation scripts check the five official mapping-member MD5 checksums before loading their pickle contents. The mappings archive served during preparation has MD5 `03227241400a465d7c6e9fa7980e480c`, whereas the upstream documentation lists `c18735c405ce6ce5761432b07287d8d9`. All five extracted members match the documented member checksums. The member checks, rather than an asserted archive-wrapper match, establish that mapping input identity.

### Descriptor definitions

Tag 2 atoms are adsorbate atoms. Tag 1 atoms are surface atoms. All atoms with tags other than 2 form the slab. Periodic displacements are wrapped using rounded fractional coordinates in the supplied cell. A contacted surface atom has nearest adsorbate distance at most 0.25 Å above the global minimum contact distance. Heights are measured along the normal to the first two cell vectors, relative to the highest surface atom. Values are rounded to twelve decimal places.

| Descriptor | Definition |
| --- | --- |
| `adsorbate_atom_count` | Number of adsorbate atoms |
| `surface_atom_count` | Number of surface atoms |
| `adsorbate_atomic_number_mean` | Mean atomic number of adsorbate atoms |
| `minimum_contact_distance` | Minimum adsorbate–surface atom distance, in Å |
| `mean_nearest_surface_distance` | Mean nearest-surface distance over adsorbate atoms, in Å |
| `contact_count_2_5a` | Number of adsorbate–surface atom pairs within 2.5 Å |
| `site_nuclearity` | Number of contacted surface atoms |
| `adsorbate_height_min` | Minimum adsorbate height, in Å |
| `adsorbate_height_mean` | Mean adsorbate height, in Å |
| `adsorbate_height_span` | Maximum minus minimum adsorbate height, in Å |
| `surface_roughness` | Population standard deviation of surface-atom heights, in Å |
| `contact_surface_atomic_number_mean` | Mean atomic number of contacted surface atoms |
| `contact_surface_coordination_mean` | Mean count of slab neighbors within 3.0 Å of contacted surface atoms, excluding zero-distance self pairs |
| `surface_area` | Magnitude of the cross product of the first two cell vectors, in Å² |

Descriptor formulas do not use the target energy.

## Frozen calibration references

`scripts/calibrate_materials_adsorption.py` generates the stage-1 permutation reference by shuffling target energies within each system while retaining descriptors and grouping. This reference uses 100 permutations, seed 41, six accepted controls, and the nearest-rank 95th percentile of 600 statistics. Its threshold is `0.002343568457146716`.

The six controls are `0.0`, `-minimum_contact_distance`, `-abs(adsorbate_height_mean)`, `contact_surface_coordination_mean`, `contact_surface_coordination_mean * minimum_contact_distance`, and `2.0 * minimum_contact_distance`. The final control is checked as an affine duplicate of minimum contact. A separate `candidate_id` expression is expected to fail compilation.

`scripts/calibrate_materials_adsorption_attainability.py` scores fourteen single descriptors, 91 pair products, 91 zero-safe pair ratios, and 500 unique valid random expressions using seed 41. The file contains 696 scored candidates, of which 23 pass its stage-1 signal test. These expressions are calibration inputs, not model-generated discoveries or evidence of scientific generalization.

`scripts/calibrate_materials_adsorption_shard_null.py` generates the probe reference. Runtime partition `materials-evidence-shard-v2` combines the 32 source groups into eleven evidence shards of three groups. A source group is repeated to fill the final shard. The calibration pools 11 × 696 × 100 = 765,600 permuted scores. It stores nineteen quantile edges and twenty percentile-bin labels, plus 696 unpermuted gains for each shard. It does not store every pooled null sample.

These references use five grouped folds, L2 regularization `0.1`, complexity coefficient `0.001`, fold seed `grouped-system-folds-v1`, and pair seed `grouped-hash-oriented-pairs-v1`. The task validates dataset, universe, registry, compiler, scorer, and partition bindings. File checksums and internal binding hashes are different: `SHA256SUMS` hashes file bytes, while runtime checks use canonical content manifests.

Changing descriptors, labels, sampling, alias salt, compiler behavior, or scorer settings can invalidate these references. New datasets need new compatible calibrations. The supplied shard-calibration script currently assumes 696 candidates and eleven evidence shards. Larger custom datasets require an appropriate calibration configuration or script adaptation; their scores are not interchangeable with these files.

## Runtime visibility and limitations

These are public benchmark inputs, but they are host-side scoring assets during an episode. `dataset.json` contains targets and source identities. `attainability-v2.json` contains successful expressions and scores. Do not mount this directory, the repository root, or calibration files into the agent or compute sandbox. The compute sandbox receives generated label-free exports only.

The fixed alias salt is public fixture metadata, not a secret or an anti-memorization mechanism. Public availability of targets and successful rules means this example cannot establish resistance to memorization. It supports functional workflow checks, not claims of improved scientific performance or full OC20Dense reproduction.

## Checksums

`SHA256SUMS` covers the four JSON files. From this directory, `sha256sum -c SHA256SUMS` verifies their bytes. The quickstart uses these shipped files and does not require downloading the full upstream structures.
