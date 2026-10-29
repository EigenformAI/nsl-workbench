#!/usr/bin/env python3
"""Verify OC20-Dense metadata and build a deterministic Phase 0 mini fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import statistics
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

DOCUMENTED_ARCHIVE_MD5 = "c18735c405ce6ce5761432b07287d8d9"
EXPECTED_MEMBERS = {
    "oc20dense_mapping.pkl": "3e26c3bcef01ccfc9b001931065ea6e6",
    "oc20dense_targets.pkl": "fd589b013b72e62e11a6b2a5bd1d323c",
    "oc20dense_compute.pkl": "78d25997e0aaf754df526ab37276bb89",
    "oc20dense_ref_energies.pkl": "b07c64158e4bfa5f7b9bf6263753ecc5",
    "oc20dense_tags.pkl": "1ba0bc266130f186850f5faa547b6a02",
}


def _digest(data: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, data).hexdigest()


def _hash_order(value: str) -> tuple[bytes, str]:
    return hashlib.sha256(value.encode("utf-8")).digest(), value


def _load_verified_members(
    archive_path: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    archive = archive_path.read_bytes()
    members: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as bundle:
        names = {member.name for member in bundle.getmembers() if member.isfile()}
        if names != set(EXPECTED_MEMBERS):
            raise RuntimeError(
                f"OC20-Dense bundle members differ from the frozen manifest: {sorted(names)}"
            )
        for name, expected_md5 in EXPECTED_MEMBERS.items():
            stream = bundle.extractfile(name)
            if stream is None:
                raise RuntimeError(f"OC20-Dense bundle member is unreadable: {name}")
            data = stream.read()
            actual_md5 = _digest(data, "md5")
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"OC20-Dense member checksum mismatch for {name}: {actual_md5}"
                )
            members[name] = data
    return members, {
        "path": str(archive_path),
        "bytes": len(archive),
        "md5": _digest(archive, "md5"),
        "sha256": _digest(archive, "sha256"),
        "documented_md5": DOCUMENTED_ARCHIVE_MD5,
        "documented_md5_matches": _digest(archive, "md5") == DOCUMENTED_ARCHIVE_MD5,
        "members": {
            name: {
                "bytes": len(data),
                "md5": _digest(data, "md5"),
                "sha256": _digest(data, "sha256"),
            }
            for name, data in sorted(members.items())
        },
    }


def _config_method(config_id: str) -> str:
    if config_id.startswith("heur"):
        return "heuristic"
    if config_id.startswith("rand"):
        return "random"
    return "other"


def prepare(
    archive_path: Path,
    descriptor_manifest_path: Path,
    output_dir: Path,
    *,
    system_limit: int,
    placement_limit: int | None,
) -> dict[str, Any]:
    members, source_manifest = _load_verified_members(archive_path)
    # Verified official pickles are converted immediately to JSON. Never load
    # an unverified pickle supplied through this interface.
    mapping = pickle.loads(members["oc20dense_mapping.pkl"])
    targets = pickle.loads(members["oc20dense_targets.pkl"])
    descriptor_manifest = json.loads(
        descriptor_manifest_path.read_text(encoding="utf-8")
    )
    descriptors_by_sid = descriptor_manifest.get("descriptors_by_sid")
    descriptor_registry = descriptor_manifest.get("descriptor_registry")
    if not isinstance(descriptors_by_sid, dict) or not isinstance(
        descriptor_registry, dict
    ):
        raise ValueError("invalid structure descriptor manifest")

    mapping_by_pair: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
    method_counts: Counter[str] = Counter()
    for sid, metadata in mapping.items():
        system_id = str(metadata["system_id"])
        config_id = str(metadata["config_id"])
        key = (system_id, config_id)
        if key in mapping_by_pair:
            raise RuntimeError(f"duplicate OC20-Dense mapping key: {key!r}")
        mapping_by_pair[key] = (sid, metadata)
        method_counts[_config_method(config_id)] += 1

    target_pairs = {
        (str(system_id), str(config_id))
        for system_id, rows in targets.items()
        for config_id, _energy in rows
    }
    missing_mapping = sorted(target_pairs - set(mapping_by_pair))
    if missing_mapping:
        raise RuntimeError(
            f"{len(missing_mapping)} target placements are missing mapping metadata"
        )
    finite_targets = all(
        math.isfinite(float(energy))
        for rows in targets.values()
        for _config_id, energy in rows
    )
    if not finite_targets:
        raise RuntimeError("OC20-Dense target bundle contains non-finite energies")

    eligible_systems = [
        str(system_id) for system_id, rows in targets.items() if len(rows) >= 2
    ]
    selected_systems = sorted(eligible_systems, key=_hash_order)[:system_limit]
    shard_count = min(32, len(selected_systems))
    evidence_shards = {
        system_id: f"evidence-shard-{index % shard_count + 1:03d}"
        for index, system_id in enumerate(selected_systems)
    }
    records: list[dict[str, Any]] = []
    selected_method_counts: Counter[str] = Counter()
    for system_id in selected_systems:
        ordered_rows = sorted(
            targets[system_id],
            key=lambda item: _hash_order(f"{system_id}\0{item[0]}"),
        )
        rows = (
            ordered_rows if placement_limit is None else ordered_rows[:placement_limit]
        )
        for config_id_value, energy in rows:
            config_id = str(config_id_value)
            sid, metadata = mapping_by_pair[(system_id, config_id)]
            structure_entry = descriptors_by_sid.get(str(sid))
            if not isinstance(structure_entry, dict):
                raise RuntimeError(f"missing structure descriptors for sid {sid}")
            if (
                str(structure_entry.get("system_id")) != system_id
                or str(structure_entry.get("config_id")) != config_id
            ):
                raise RuntimeError(
                    f"structure descriptor identity mismatch for sid {sid}"
                )
            structure_descriptors = structure_entry.get("values")
            if not isinstance(structure_descriptors, dict):
                raise RuntimeError(f"invalid structure descriptors for sid {sid}")
            method = _config_method(config_id)
            selected_method_counts[method] += 1
            miller = tuple(int(value) for value in metadata["miller_idx"])
            structure_group = json.dumps(
                {
                    "mpid": str(metadata["mpid"]),
                    "miller": miller,
                    "shift": float(metadata["shift"]),
                    "top": bool(metadata["top"]),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            records.append(
                {
                    "source_system_id": system_id,
                    "source_placement_id": config_id,
                    "source_filename": f"verified-oc20dense-mapping/{sid}",
                    "adsorbate_group": str(metadata["adsorbate"]),
                    "catalyst_group": str(metadata["mpid"]),
                    "structure_group": structure_group,
                    "source_group": evidence_shards[system_id],
                    "descriptors": dict(structure_descriptors),
                    "energy": float(energy),
                    "axis": str(structure_entry["axis"]),
                    "status": "valid",
                }
            )

    dataset = {
        "version": "oc20-dense-phase0-structure-mini-v1",
        "alias_salt": "public-phase0-fixture-v1-not-for-recursive-runs",
        "descriptor_registry": descriptor_registry,
        "records": records,
    }
    controls = [
        {"name": "null", "expression": "0.0", "expect_signal": False},
        {
            "name": "minimum_contact",
            "expression": "-minimum_contact_distance",
            "expect_signal": True,
        },
        {"name": "site_height", "expression": "-abs(adsorbate_height_mean)"},
        {
            "name": "contact_coordination",
            "expression": "contact_surface_coordination_mean",
        },
        {
            "name": "coordination_distance_interaction",
            "expression": "contact_surface_coordination_mean * minimum_contact_distance",
            "expect_signal": True,
        },
        {
            "name": "affine_minimum_contact",
            "expression": "2.0 * minimum_contact_distance",
            "duplicate_of": "minimum_contact",
            "expect_marginal": False,
        },
        {
            "name": "leaked_identifier",
            "expression": "candidate_id",
            "expect_compile_error": True,
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.json"
    controls_path = output_dir / "controls.json"
    evidence_shards_path = output_dir / "evidence-shards.json"
    report_path = output_dir / "phase0-report.json"
    dataset_path.write_text(
        json.dumps(dataset, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    controls_path.write_text(
        json.dumps(controls, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shard_records: dict[str, list[dict[str, Any]]] = {
        shard_id: [] for shard_id in sorted(set(evidence_shards.values()))
    }
    for record in records:
        shard_records[str(record["source_group"])].append(
            {
                "source_system_id": record["source_system_id"],
                "source_placement_id": record["source_placement_id"],
                "descriptors": record["descriptors"],
            }
        )
    evidence_shard_manifest = {
        "schema_version": 1,
        "shards": [
            {
                "source_group": shard_id,
                "systems": len(
                    {
                        str(record["source_system_id"])
                        for record in shard_records[shard_id]
                    }
                ),
                "placements": len(shard_records[shard_id]),
                "content_sha256": _digest(
                    json.dumps(
                        shard_records[shard_id],
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    "sha256",
                ),
            }
            for shard_id in sorted(shard_records)
        ],
    }
    evidence_shards_path.write_text(
        json.dumps(evidence_shard_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sizes = sorted(len(rows) for rows in targets.values())
    report = {
        "schema_version": 1,
        "source": source_manifest,
        "structure_descriptor_manifest": {
            "path": str(descriptor_manifest_path),
            "bytes": descriptor_manifest_path.stat().st_size,
            "sha256": _digest(descriptor_manifest_path.read_bytes(), "sha256"),
            "version": descriptor_registry.get("version"),
        },
        "official_documentation": "https://fair-chem.github.io/oc20dense/",
        "full_bundle": {
            "mapping_configurations": len(mapping),
            "mapping_systems": len(
                {str(item["system_id"]) for item in mapping.values()}
            ),
            "target_configurations": len(target_pairs),
            "target_systems": len(targets),
            "mapped_without_valid_target": len(mapping_by_pair) - len(target_pairs),
            "finite_targets": finite_targets,
            "configuration_methods": dict(sorted(method_counts.items())),
            "target_placements_per_system": {
                "min": sizes[0],
                "median": statistics.median(sizes),
                "max": sizes[-1],
            },
        },
        "mini_fixture": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": _digest(dataset_path.read_bytes(), "sha256"),
            "controls_path": str(controls_path),
            "controls_sha256": _digest(controls_path.read_bytes(), "sha256"),
            "systems": len(selected_systems),
            "placements": len(records),
            "placements_per_system": placement_limit,
            "configuration_methods": dict(sorted(selected_method_counts.items())),
            "selection": "ascending SHA-256(system_id), then SHA-256(system_id\\0config_id)",
            "evidence_shards": len(set(evidence_shards.values())),
            "evidence_shards_path": str(evidence_shards_path),
            "evidence_shards_sha256": _digest(
                evidence_shards_path.read_bytes(), "sha256"
            ),
        },
        "additional_open_items": [
            "Placements absent from targets are not separated by invalidity cause.",
        ],
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify OC20-Dense mappings and prepare a deterministic Phase 0 mini fixture."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("descriptor_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--systems", type=int, default=32)
    parser.add_argument("--placements", type=int, default=8)
    parser.add_argument("--all-placements", action="store_true")
    args = parser.parse_args()
    if args.systems < 2 or (not args.all_placements and args.placements < 2):
        parser.error("--systems and --placements must both be at least two")
    report = prepare(
        args.archive,
        args.descriptor_manifest,
        args.output_dir,
        system_limit=args.systems,
        placement_limit=None if args.all_placements else args.placements,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
