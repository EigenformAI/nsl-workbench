from __future__ import annotations

import numpy as np

from scripts.prepare_materials_adsorption_phase0_structures import (
    _axis_manifest,
    _structure_descriptors,
)


def test_axis_manifest_joins_official_oc20_validation_splits() -> None:
    dense = {
        0: {
            "system_id": "dense-a",
            "adsorbate": "*CO",
            "mpid": "mp-1",
            "miller_idx": (1, 0, 0),
            "shift": 0.125,
            "top": True,
        },
        1: {
            "system_id": "dense-b",
            "adsorbate": "*OH",
            "mpid": "mp-2",
            "miller_idx": (1, 1, 1),
            "shift": 0.25,
            "top": False,
        },
    }
    oc20 = {
        "random-a": {
            "ads_symbols": "*CO",
            "bulk_mpid": "mp-1",
            "miller_index": (1, 0, 0),
            "shift": 0.125,
            "top": True,
            "split": "val_is",
        },
        "random-b": {
            "ads_symbols": "*OH",
            "bulk_mpid": "mp-2",
            "miller_index": (1, 1, 1),
            "shift": 0.25,
            "top": False,
            "split": "val_oos_ads_bulk",
        },
    }

    assert _axis_manifest(dense, oc20) == {
        "dense-a": "id",
        "dense-b": "ood_both",
    }


def test_structure_descriptors_are_periodic_translation_invariant() -> None:
    cell = np.diag([10.0, 10.0, 20.0])
    positions = np.array(
        [
            [0.2, 0.2, 4.0],
            [9.8, 0.2, 4.0],
            [0.0, 0.2, 5.5],
            [0.0, 0.2, 6.5],
        ]
    )
    atomic_numbers = np.array([29, 29, 6, 8])
    tags = np.array([1, 1, 2, 2])

    first = _structure_descriptors(positions, atomic_numbers, tags, cell)
    translated = positions.copy()
    translated[2:, 0] += 10.0
    second = _structure_descriptors(translated, atomic_numbers, tags, cell)

    assert second == first
    assert first["minimum_contact_distance"] == 1.513274595042
    assert first["site_nuclearity"] == 2.0
    assert first["adsorbate_height_span"] == 1.0
