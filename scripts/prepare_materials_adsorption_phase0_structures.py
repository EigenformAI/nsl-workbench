#!/usr/bin/env python3
"""Freeze OC20-Dense structure descriptors and authoritative split provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.prepare_materials_adsorption_phase0 import _load_verified_members

STRUCTURE_ARCHIVE_MD5 = "0163b0e8c4df6d9c426b875a28d9178a"
OC20_MAPPING_MD5 = "6b5d485019861f6e7efca38338375b61"
_AXIS_BY_OC20_SPLIT = {
    "val_is": "id",
    "val_oos_ads": "ood_adsorbate",
    "val_oos_bulk": "ood_catalyst",
    "val_oos_ads_bulk": "ood_both",
}
DESCRIPTOR_REGISTRY = (
    ("adsorbate_atom_count", "count", "composition"),
    ("surface_atom_count", "count", "surface"),
    ("adsorbate_atomic_number_mean", "atomic_number", "composition"),
    ("minimum_contact_distance", "angstrom", "contact"),
    ("mean_nearest_surface_distance", "angstrom", "contact"),
    ("contact_count_2_5a", "count", "contact"),
    ("site_nuclearity", "count", "contact"),
    ("adsorbate_height_min", "angstrom", "orientation"),
    ("adsorbate_height_mean", "angstrom", "orientation"),
    ("adsorbate_height_span", "angstrom", "orientation"),
    ("surface_roughness", "angstrom", "surface"),
    ("contact_surface_atomic_number_mean", "atomic_number", "contact"),
    ("contact_surface_coordination_mean", "count", "coordination"),
    ("surface_area", "angstrom^2", "surface"),
)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _surface_key(metadata: Mapping[str, Any], *, dense: bool) -> tuple[Any, ...]:
    return (
        str(metadata["adsorbate" if dense else "ads_symbols"]),
        str(metadata["mpid" if dense else "bulk_mpid"]),
        tuple(
            int(value) for value in metadata["miller_idx" if dense else "miller_index"]
        ),
        round(float(metadata["shift"]), 6),
        bool(metadata["top"]),
    )


def _axis_manifest(
    dense_mapping: Mapping[Any, Mapping[str, Any]],
    oc20_mapping: Mapping[Any, Mapping[str, Any]],
) -> dict[str, str]:
    splits_by_surface: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for metadata in oc20_mapping.values():
        split = str(metadata.get("split", ""))
        if split in _AXIS_BY_OC20_SPLIT:
            splits_by_surface[_surface_key(metadata, dense=False)].add(split)

    systems: dict[str, Mapping[str, Any]] = {}
    for metadata in dense_mapping.values():
        systems.setdefault(str(metadata["system_id"]), metadata)

    result: dict[str, str] = {}
    for system_id, metadata in sorted(systems.items()):
        splits = splits_by_surface.get(_surface_key(metadata, dense=True), set())
        if len(splits) != 1:
            raise RuntimeError(
                f"OC20 validation provenance for {system_id} is not unique: "
                f"{sorted(splits)}"
            )
        result[system_id] = _AXIS_BY_OC20_SPLIT[next(iter(splits))]
    return result


def _minimum_image(deltas: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = deltas @ np.linalg.inv(cell)
    fractional -= np.rint(fractional)
    return fractional @ cell


def _canonical_float(value: float) -> float:
    result = round(float(value), 12)
    if not math.isfinite(result):
        raise ValueError("structure descriptor is not finite")
    return result


def _structure_descriptors(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    tags: np.ndarray,
    cell: np.ndarray,
) -> dict[str, float]:
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    atomic_numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    tags = np.asarray(tags, dtype=int).reshape(-1)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    if not (
        len(positions) == len(atomic_numbers) == len(tags)
        and np.isfinite(positions).all()
        and np.isfinite(cell).all()
        and abs(np.linalg.det(cell)) > 1e-12
    ):
        raise ValueError("invalid OC20-Dense structure arrays")

    adsorbate_indices = np.flatnonzero(tags == 2)
    surface_indices = np.flatnonzero(tags == 1)
    slab_indices = np.flatnonzero(tags != 2)
    if not len(adsorbate_indices) or not len(surface_indices):
        raise ValueError("structure must contain adsorbate and surface atoms")

    adsorbate_positions = positions[adsorbate_indices]
    surface_positions = positions[surface_indices]
    contact_vectors = _minimum_image(
        adsorbate_positions[:, None, :] - surface_positions[None, :, :], cell
    )
    contact_distances = np.linalg.norm(contact_vectors, axis=2)
    nearest_distances = contact_distances.min(axis=1)
    minimum_contact = float(contact_distances.min())
    surface_minima = contact_distances.min(axis=0)
    contact_surface_mask = surface_minima <= minimum_contact + 0.25
    contact_surface_indices = surface_indices[contact_surface_mask]

    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    surface_heights = surface_positions @ normal
    adsorbate_heights = adsorbate_positions @ normal - surface_heights.max()

    slab_positions = positions[slab_indices]
    contacted_positions = positions[contact_surface_indices]
    coordination_vectors = _minimum_image(
        contacted_positions[:, None, :] - slab_positions[None, :, :], cell
    )
    coordination_distances = np.linalg.norm(coordination_vectors, axis=2)
    coordination = (
        (coordination_distances > 1e-8) & (coordination_distances <= 3.0)
    ).sum(axis=1)

    return {
        "adsorbate_atom_count": _canonical_float(len(adsorbate_indices)),
        "surface_atom_count": _canonical_float(len(surface_indices)),
        "adsorbate_atomic_number_mean": _canonical_float(
            atomic_numbers[adsorbate_indices].mean()
        ),
        "minimum_contact_distance": _canonical_float(minimum_contact),
        "mean_nearest_surface_distance": _canonical_float(nearest_distances.mean()),
        "contact_count_2_5a": _canonical_float((contact_distances <= 2.5).sum()),
        "site_nuclearity": _canonical_float(contact_surface_mask.sum()),
        "adsorbate_height_min": _canonical_float(adsorbate_heights.min()),
        "adsorbate_height_mean": _canonical_float(adsorbate_heights.mean()),
        "adsorbate_height_span": _canonical_float(
            adsorbate_heights.max() - adsorbate_heights.min()
        ),
        "surface_roughness": _canonical_float(surface_heights.std()),
        "contact_surface_atomic_number_mean": _canonical_float(
            atomic_numbers[contact_surface_indices].mean()
        ),
        "contact_surface_coordination_mean": _canonical_float(coordination.mean()),
        "surface_area": _canonical_float(np.linalg.norm(np.cross(cell[0], cell[1]))),
    }


def prepare(
    mappings_archive: Path,
    structure_archive: Path,
    structure_lmdb: Path,
    oc20_mapping_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if _md5(structure_archive) != STRUCTURE_ARCHIVE_MD5:
        raise RuntimeError("OC20-Dense structure archive checksum mismatch")
    if _md5(oc20_mapping_path) != OC20_MAPPING_MD5:
        raise RuntimeError("OC20 mapping checksum mismatch")

    members, mappings_source = _load_verified_members(mappings_archive)
    dense_mapping = pickle.loads(members["oc20dense_mapping.pkl"])
    oc20_mapping = pickle.loads(oc20_mapping_path.read_bytes())
    axes = _axis_manifest(dense_mapping, oc20_mapping)

    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError(
            "LMDB extraction requires `uv run --with lmdb --with torch-geometric`"
        ) from exc

    descriptors: dict[str, dict[str, Any]] = {}
    environment = lmdb.open(
        str(structure_lmdb),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    try:
        with environment.begin(write=False) as transaction:
            for key, payload in transaction.cursor():
                try:
                    int(key.decode("ascii"))
                except (UnicodeDecodeError, ValueError):
                    continue
                record = pickle.loads(payload)
                sid_value = getattr(record, "sid")
                sid = int(sid_value.item() if hasattr(sid_value, "item") else sid_value)
                if sid in descriptors:
                    raise RuntimeError(f"duplicate structure sid: {sid}")
                metadata = dense_mapping.get(sid)
                if metadata is None:
                    raise RuntimeError(f"structure sid has no mapping metadata: {sid}")

                def array(name: str) -> np.ndarray:
                    value = getattr(record, name)
                    if hasattr(value, "detach"):
                        value = value.detach().cpu().numpy()
                    return np.asarray(value)

                descriptors[str(sid)] = {
                    "system_id": str(metadata["system_id"]),
                    "config_id": str(metadata["config_id"]),
                    "axis": axes[str(metadata["system_id"])],
                    "values": _structure_descriptors(
                        array("pos"),
                        array("atomic_numbers"),
                        array("tags"),
                        array("cell"),
                    ),
                }
    finally:
        environment.close()

    if len(descriptors) != len(dense_mapping):
        raise RuntimeError(
            f"structure/mapping count mismatch: {len(descriptors)} != {len(dense_mapping)}"
        )

    manifest = {
        "schema_version": 1,
        "descriptor_registry": {
            "version": "oc20-dense-initial-structure-v1",
            "descriptors": [
                {
                    "name": name,
                    "value_type": "number",
                    "unit": unit,
                    "family": family,
                }
                for name, unit, family in DESCRIPTOR_REGISTRY
            ],
        },
        "sources": {
            "mappings": mappings_source,
            "structure_archive": {
                "path": str(structure_archive),
                "md5": STRUCTURE_ARCHIVE_MD5,
                "sha256": _sha256(structure_archive),
            },
            "structure_lmdb": {
                "path": str(structure_lmdb),
                "bytes": structure_lmdb.stat().st_size,
                "sha256": _sha256(structure_lmdb),
            },
            "oc20_mapping": {
                "path": str(oc20_mapping_path),
                "md5": OC20_MAPPING_MD5,
                "sha256": _sha256(oc20_mapping_path),
            },
        },
        "axis_counts": dict(sorted(Counter(axes.values()).items())),
        "systems": len(axes),
        "placements": len(descriptors),
        "descriptors_by_sid": descriptors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mappings_archive", type=Path)
    parser.add_argument("structure_archive", type=Path)
    parser.add_argument("structure_lmdb", type=Path)
    parser.add_argument("oc20_mapping", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = prepare(
        args.mappings_archive,
        args.structure_archive,
        args.structure_lmdb,
        args.oc20_mapping,
        args.output,
    )
    print(
        json.dumps(
            {
                "axis_counts": manifest["axis_counts"],
                "placements": manifest["placements"],
                "systems": manifest["systems"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
