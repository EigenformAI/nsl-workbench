#!/usr/bin/env python3
"""Generate the fixed synthetic materials-scorer capacity fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SYSTEMS = 963
PLACEMENTS_PER_SYSTEM = 67
MAX_PAIRS_PER_SYSTEM = 256
AXES = ("id", "ood_adsorbate", "ood_catalyst", "ood_both")
EXPECTED_DATASET_SHA256 = "3552b3617d7e42f3354c8b18ef7ae2d6e1da5bf103d0edf0ac77cef83e68f369"
EXPECTED_CONTROLS_SHA256 = "0f13d43e7ebae828ea16b7bbad2aa415cfe4426b223b13a6eaea462a34033491"

CONTROLS = [
    {"name": "minimum_contact", "expression": "-minimum_contact_distance"},
    {"name": "site_height", "expression": "-abs(adsorbate_height_mean)"},
    {
        "name": "contact_coordination",
        "expression": "contact_surface_coordination_mean",
    },
    {
        "name": "coordination_distance_interaction",
        "expression": (
            "contact_surface_coordination_mean * minimum_contact_distance"
        ),
    },
]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def generate(output_dir: Path) -> dict[str, object]:
    placements = SYSTEMS * PLACEMENTS_PER_SYSTEM
    retained_pairs = SYSTEMS * MAX_PAIRS_PER_SYSTEM
    if placements != 64_521 or retained_pairs != 246_528:
        raise RuntimeError("synthetic benchmark shape changed")
    if (
        PLACEMENTS_PER_SYSTEM * (PLACEMENTS_PER_SYSTEM - 1) // 2
        <= MAX_PAIRS_PER_SYSTEM
    ):
        raise RuntimeError("pair cap must bind for every synthetic system")

    records = []
    for system_index in range(SYSTEMS):
        system_id = f"synthetic-system-{system_index:04d}"
        for placement_index in range(PLACEMENTS_PER_SYSTEM):
            placement_id = f"placement-{placement_index:02d}"
            distance = 1 + ((placement_index + system_index) % PLACEMENTS_PER_SYSTEM)
            height = 1 + (
                (17 * placement_index + 3 * system_index) % PLACEMENTS_PER_SYSTEM
            )
            coordination = 1 + (
                (29 * placement_index + 5 * system_index) % PLACEMENTS_PER_SYSTEM
            )
            interaction = distance * coordination
            records.append(
                {
                    "source_system_id": system_id,
                    "source_placement_id": placement_id,
                    "source_filename": f"synthetic/{system_id}/{placement_id}.json",
                    "adsorbate_group": f"adsorbate-{system_index % 11:02d}",
                    "catalyst_group": f"catalyst-{(system_index // 11) % 13:02d}",
                    "structure_group": f"surface-{system_index:04d}",
                    "source_group": f"synthetic-shard-{system_index % 32:02d}",
                    "descriptors": {
                        "minimum_contact_distance": distance,
                        "adsorbate_height_mean": height,
                        "contact_surface_coordination_mean": coordination,
                    },
                    "energy": -(128 * interaction + placement_index),
                    "axis": AXES[system_index % len(AXES)],
                    "status": "valid",
                }
            )

    dataset = {
        "schema_version": 1,
        "version": "synthetic-materials-adsorption-benchmark-v1",
        "alias_salt": "synthetic-materials-adsorption-benchmark-v1",
        "descriptor_registry": {
            "schema_version": 1,
            "version": "synthetic-materials-adsorption-benchmark-descriptors-v1",
            "descriptors": [
                {
                    "name": "minimum_contact_distance",
                    "value_type": "number",
                    "unit": "angstrom",
                    "family": "contact",
                },
                {
                    "name": "adsorbate_height_mean",
                    "value_type": "number",
                    "unit": "angstrom",
                    "family": "orientation",
                },
                {
                    "name": "contact_surface_coordination_mean",
                    "value_type": "number",
                    "unit": "count",
                    "family": "coordination",
                },
            ],
        },
        "records": records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.json"
    controls_path = output_dir / "controls.json"
    _write_json(dataset_path, dataset)
    _write_json(controls_path, CONTROLS)
    result = {
        "systems": SYSTEMS,
        "placements": placements,
        "retained_pairs": retained_pairs,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "controls_sha256": hashlib.sha256(controls_path.read_bytes()).hexdigest(),
    }
    if result["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("synthetic benchmark dataset hash changed")
    if result["controls_sha256"] != EXPECTED_CONTROLS_SHA256:
        raise RuntimeError("synthetic benchmark controls hash changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
