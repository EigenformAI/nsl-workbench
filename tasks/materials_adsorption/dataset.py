from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .descriptors import (
    DescriptorRegistry,
    DescriptorSpec,
    DescriptorValue,
    _PRIVATE_NAME_PARTS,
    _canonical_hash,
    _canonical_json,
)

__all__ = [
    "CandidateState",
    "CandidateUniverse",
    "EndpointStatus",
    "LoadedDataset",
    "PlacementRecord",
    "build_candidate_universe",
    "load_dataset",
]


class EndpointStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCONVERGED = "unconverged"
    DESORBED = "desorbed"
    DISSOCIATED = "dissociated"
    SURFACE_MISMATCH = "surface_mismatch"


def _immutable_descriptors(
    descriptors: Mapping[str, DescriptorValue],
) -> Mapping[str, DescriptorValue]:
    normalized: dict[str, DescriptorValue] = {}
    for name, value in sorted(descriptors.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("descriptor names must be non-empty strings")
        if set(name.split("_")) & _PRIVATE_NAME_PARTS:
            raise ValueError(f"private or label-derived descriptor name: {name!r}")
        if type(value) not in (bool, int, float):
            raise ValueError(f"descriptor {name!r} has a non-scalar value")
        if type(value) in (int, float) and not math.isfinite(float(value)):
            raise ValueError(f"descriptor {name!r} must be finite")
        normalized[name] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class PlacementRecord:
    """Private ingestion record. It must never be included in a public payload."""

    source_system_id: str
    source_placement_id: str
    source_filename: str
    adsorbate_group: str
    catalyst_group: str
    structure_group: str
    source_group: str
    descriptors: Mapping[str, DescriptorValue]
    energy: float | None
    axis: str
    status: EndpointStatus

    def __post_init__(self) -> None:
        for field_name in (
            "source_system_id",
            "source_placement_id",
            "source_filename",
            "adsorbate_group",
            "catalyst_group",
            "structure_group",
            "source_group",
            "axis",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        try:
            status = EndpointStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"unknown endpoint status: {self.status!r}") from exc
        if self.energy is not None and not math.isfinite(float(self.energy)):
            raise ValueError("energy must be finite when present")
        if status is EndpointStatus.VALID and self.energy is None:
            raise ValueError("valid endpoints require an energy")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "descriptors", _immutable_descriptors(self.descriptors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_system_id": self.source_system_id,
            "source_placement_id": self.source_placement_id,
            "source_filename": self.source_filename,
            "adsorbate_group": self.adsorbate_group,
            "catalyst_group": self.catalyst_group,
            "structure_group": self.structure_group,
            "source_group": self.source_group,
            "descriptors": dict(self.descriptors),
            "energy": self.energy,
            "axis": self.axis,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class CandidateState:
    """Agent-visible, label-free placement state."""

    alias: str
    system_alias: str
    descriptors: Mapping[str, DescriptorValue]

    def __post_init__(self) -> None:
        if not self.alias or not self.system_alias:
            raise ValueError("candidate and system aliases must be non-empty")
        object.__setattr__(self, "descriptors", _immutable_descriptors(self.descriptors))

    def public_payload(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "system_alias": self.system_alias,
            "descriptors": dict(self.descriptors),
        }


@dataclass(frozen=True, slots=True)
class CandidateUniverse:
    version: str
    candidates: tuple[CandidateState, ...]
    manifest_hash: str
    universe_hash: str

    def public_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "version": self.version,
            "manifest_hash": self.manifest_hash,
            "universe_hash": self.universe_hash,
            "candidates": [candidate.public_payload() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Host-side dataset view; ``records`` and source aliases are never public."""

    registry: DescriptorRegistry
    universe: CandidateUniverse
    records: tuple[PlacementRecord, ...]
    source_aliases: Mapping[tuple[str, str], tuple[str, str]]
    alias_salt: str


def _alias(kind: str, identity: tuple[str, ...], version: str, salt: str) -> str:
    payload = _canonical_json(
        {"kind": kind, "identity": identity, "version": version, "alias_schema": 1}
    )
    digest = hmac.new(salt.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256)
    prefix = "sys" if kind == "system" else "cfg"
    return f"{prefix}_{digest.hexdigest()[:24]}"


def build_candidate_universe(
    records: Iterable[PlacementRecord], *, version: str, alias_salt: str
) -> CandidateUniverse:
    if not version or not alias_salt:
        raise ValueError("version and alias_salt must be non-empty")
    ordered = sorted(
        tuple(records),
        key=lambda record: (record.source_system_id, record.source_placement_id),
    )
    identities = [(record.source_system_id, record.source_placement_id) for record in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("source placement identities must be unique")
    system_groups: dict[str, tuple[str, str, str, str, str]] = {}
    for record in ordered:
        grouping = (
            record.adsorbate_group,
            record.catalyst_group,
            record.structure_group,
            record.source_group,
            record.axis,
        )
        existing = system_groups.setdefault(record.source_system_id, grouping)
        if existing != grouping:
            raise ValueError("a source system cannot cross scientific grouping boundaries")

    manifest_payload = {
        "schema_version": 1,
        "version": version,
        "alias_salt_hash": hashlib.sha256(alias_salt.encode("utf-8")).hexdigest(),
        "records": [record.to_dict() for record in ordered],
    }
    manifest_hash = _canonical_hash(manifest_payload)

    candidates = tuple(
        sorted(
            (
                CandidateState(
                    alias=_alias(
                        "candidate",
                        (record.source_system_id, record.source_placement_id),
                        version,
                        alias_salt,
                    ),
                    system_alias=_alias(
                        "system", (record.source_system_id,), version, alias_salt
                    ),
                    descriptors=record.descriptors,
                )
                for record in ordered
                if record.status is EndpointStatus.VALID
            ),
            key=lambda candidate: candidate.alias,
        )
    )
    aliases = [candidate.alias for candidate in candidates]
    if len(set(aliases)) != len(aliases):
        raise ValueError("candidate alias collision")
    universe_hash = _canonical_hash(
        {
            "schema_version": 1,
            "version": version,
            "candidates": [candidate.public_payload() for candidate in candidates],
        }
    )
    return CandidateUniverse(version, candidates, manifest_hash, universe_hash)


def load_dataset(path: str | Path) -> LoadedDataset:
    """Load the versioned JSON interchange emitted by OC20 preprocessing.

    The production downloader/canonicalizer is intentionally separate from the
    online task. This loader accepts its small, checksummed interchange format
    and creates both the public candidate universe and the private label view.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("materials dataset must be a JSON object")
    registry_payload = payload.get("descriptor_registry")
    if not isinstance(registry_payload, dict):
        raise ValueError("materials dataset is missing descriptor_registry")
    descriptor_payloads = registry_payload.get("descriptors")
    if not isinstance(descriptor_payloads, list):
        raise ValueError("descriptor_registry.descriptors must be a list")
    registry = DescriptorRegistry(
        version=str(registry_payload.get("version") or ""),
        descriptors=tuple(
            DescriptorSpec(
                name=str(item["name"]),
                value_type=str(item["value_type"]),
                unit=str(item["unit"]),
                family=str(item["family"]),
            )
            for item in descriptor_payloads
            if isinstance(item, dict)
        ),
    )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("materials dataset records must be a list")
    records = tuple(
        PlacementRecord(
            source_system_id=str(item["source_system_id"]),
            source_placement_id=str(item["source_placement_id"]),
            source_filename=str(item["source_filename"]),
            adsorbate_group=str(item["adsorbate_group"]),
            catalyst_group=str(item["catalyst_group"]),
            structure_group=str(item["structure_group"]),
            source_group=str(item["source_group"]),
            descriptors=dict(item["descriptors"]),
            energy=None if item.get("energy") is None else float(item["energy"]),
            axis=str(item["axis"]),
            status=EndpointStatus(item["status"]),
        )
        for item in raw_records
        if isinstance(item, dict)
    )
    for record in records:
        registry.validate_values(record.descriptors)
    version = str(payload.get("version") or "")
    alias_salt = str(payload.get("alias_salt") or "")
    universe = build_candidate_universe(records, version=version, alias_salt=alias_salt)
    source_aliases = MappingProxyType(
        {
            (record.source_system_id, record.source_placement_id): (
                _alias(
                    "candidate",
                    (record.source_system_id, record.source_placement_id),
                    version,
                    alias_salt,
                ),
                _alias("system", (record.source_system_id,), version, alias_salt),
            )
            for record in records
            if record.status is EndpointStatus.VALID
        }
    )
    return LoadedDataset(registry, universe, records, source_aliases, alias_salt)
