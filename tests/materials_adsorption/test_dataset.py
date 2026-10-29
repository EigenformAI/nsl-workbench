from __future__ import annotations

import json

from tasks.materials_adsorption.dataset import (
    EndpointStatus,
    PlacementRecord,
    build_candidate_universe,
)


def _records() -> list[PlacementRecord]:
    return [
        PlacementRecord(
            source_system_id="oc20-system-1",
            source_placement_id="placement-a",
            source_filename="val_id/system-1-a.traj",
            adsorbate_group="CO",
            catalyst_group="Pt",
            structure_group="fcc-111",
            source_group="paper-a",
            descriptors={"binding_en": 2.2, "coordination": 3.0},
            energy=-1.2,
            axis="id",
            status=EndpointStatus.VALID,
        ),
        PlacementRecord(
            source_system_id="oc20-system-1",
            source_placement_id="placement-b",
            source_filename="val_id/system-1-b.traj",
            adsorbate_group="CO",
            catalyst_group="Pt",
            structure_group="fcc-111",
            source_group="paper-a",
            descriptors={"binding_en": 2.1, "coordination": 2.0},
            energy=-0.8,
            axis="id",
            status=EndpointStatus.DESORBED,
        ),
        PlacementRecord(
            source_system_id="oc20-system-2",
            source_placement_id="placement-c",
            source_filename="ood_cat/system-2-c.traj",
            adsorbate_group="N2",
            catalyst_group="Fe",
            structure_group="bcc-110",
            source_group="paper-b",
            descriptors={"binding_en": 1.8, "coordination": 4.0},
            energy=-0.4,
            axis="ood_catalyst",
            status=EndpointStatus.VALID,
        ),
    ]


def test_public_universe_filters_invalid_endpoints_and_strips_private_fields() -> None:
    universe = build_candidate_universe(_records(), version="v1", alias_salt="salt-a")

    assert len(universe.candidates) == 2
    serialized = json.dumps(universe.public_payload(), sort_keys=True)
    for forbidden in (
        "oc20-system",
        "placement-a",
        ".traj",
        "val_id",
        "ood_catalyst",
        "energy",
        "-1.2",
    ):
        assert forbidden not in serialized


def test_aliases_are_order_independent_versioned_and_salted() -> None:
    records = _records()
    first = build_candidate_universe(records, version="v1", alias_salt="salt-a")
    reordered = build_candidate_universe(
        list(reversed(records)), version="v1", alias_salt="salt-a"
    )
    resalted = build_candidate_universe(records, version="v1", alias_salt="salt-b")

    assert first.universe_hash == reordered.universe_hash
    assert [item.alias for item in first.candidates] == [
        item.alias for item in reordered.candidates
    ]
    assert {item.alias for item in first.candidates}.isdisjoint(
        item.alias for item in resalted.candidates
    )


def test_manifest_hash_changes_with_descriptor_data() -> None:
    records = _records()
    original = build_candidate_universe(records, version="v1", alias_salt="salt-a")
    changed = list(records)
    changed[0] = PlacementRecord(
        **{**changed[0].to_dict(), "descriptors": {"binding_en": 9.9, "coordination": 3.0}}
    )

    drifted = build_candidate_universe(changed, version="v1", alias_salt="salt-a")

    assert original.manifest_hash != drifted.manifest_hash
    assert original.universe_hash != drifted.universe_hash
