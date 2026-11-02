# Third-party notices

These notices apply to the components named below. They do not select a license for the repository as a whole.

## Unsloth-derived ASFT loss and trainer

`src/train/asft.py` is derived from `unsloth/losses/asft.py`.
`src/train/asft_trainer.py` adapts `ASFTTrainer` from `unsloth/trainer.py`.
Both upstream files are from Unsloth commit `5dc8ee0b3597c5c78de66db132499b431a0aa8c2`:

- [Upstream ASFT loss](https://github.com/unslothai/unsloth/blob/5dc8ee0b3597c5c78de66db132499b431a0aa8c2/unsloth/losses/asft.py)
- [Upstream trainer](https://github.com/unslothai/unsloth/blob/5dc8ee0b3597c5c78de66db132499b431a0aa8c2/unsloth/trainer.py)

Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

These upstream components are licensed under Apache License 2.0. The exact upstream license is reproduced in `LICENSES/Unsloth-LICENSE.txt`. That upstream license identifies `unsloth/*`, `tests/*`, and `scripts/*` as Apache-2.0 components, and separately identifies optional `studio/*` and `cli/*` as AGPLv3 components. The two adapted files named here come from the Apache-2.0 scope. No Studio or CLI source is included by this adaptation.

This project modifies the loss to:

- Rehome the module and add source-version metadata.
- Skip inactive KL and reference-model work, including zero KL weight.
- Support noncontiguous tensors in relevant KL reshapes.
- Log legacy sequence-KV streaming fallbacks.
- Compute active KL using one full reference forward and sequence chunks with optional non-reentrant gradient checkpointing. Legacy reference-streaming helpers remain defined but are not the active KL execution path.
- Separate token and detached probability-weight normalization, including optimizer-update-wide denominators with random-state replay.

The local trainer subclasses the installed `UnslothTrainer`, imports the modified local loss, requests full logits before parent initialization when ASFT is enabled, and creates a frozen reference only when KL is active. Types, documentation, and warnings are adapted for the local training integration.

## OC20Dense-derived materials example

`assets/materials-adsorption/oc20-dense-mini-v1/` contains a compact derivative of **Open Catalyst 2020 Dense (OC20Dense)** from the Open Catalyst Project. OC20 metadata also supplies the original validation-split provenance.

The upstream OC20Dense and OC20 datasets are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode):

- [Official OC20Dense documentation and downloads](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20dense.md)
- [Official OC20 documentation and metadata](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20.md)

OC20Dense was introduced by Janice Lan, Aini Palizhati, Muhammed Shuaibi, Brandon M. Wood, Brook Wander, Abhishek Das, Matt Uyttendaele, C. Lawrence Zitnick, and Zachary W. Ulissi in *AdsorbML: A Leap in Efficiency for Adsorption Energy Calculations using Generalizable Machine Learning Potentials* (2023). [Published paper](https://doi.org/10.1038/s41524-023-01121-5), [arXiv](https://arxiv.org/abs/2211.16486).

Changes in this example are deterministic subsampling to 248 placements across 32 systems, JSON conversion, fourteen descriptors calculated from initial structures, scientific grouping and evaluation-axis metadata, and task-specific calibration references. The asset README describes these transformations and limitations. Retain the upstream attribution, source links, license link, and indication of changes when sharing derived data. No endorsement by the original authors or the Open Catalyst Project is implied.

Dataset license terms are separate from source-code licensing. This notice does not apply Unsloth's Apache license to the dataset or select a license for other framework code. The public fixture contains labels, source identities, and successful calibration expressions. It supports functional workflow checks, not claims of scientific performance or resistance to memorization.
