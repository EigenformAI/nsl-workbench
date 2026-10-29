from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

__all__ = ["DescriptorRegistry", "DescriptorSpec"]

DescriptorType = Literal["number", "boolean"]
DescriptorValue = float | int | bool

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PRIVATE_NAME_PARTS = frozenset(
    {
        "candidate",
        "energy",
        "filename",
        "fold",
        "id",
        "label",
        "oc20",
        "path",
        "split",
        "system",
    }
)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DescriptorSpec:
    name: str
    value_type: DescriptorType
    unit: str
    family: str

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid descriptor name: {self.name!r}")
        parts = set(self.name.split("_"))
        if parts & _PRIVATE_NAME_PARTS:
            raise ValueError(f"private or label-derived descriptor name: {self.name!r}")
        if self.value_type not in ("number", "boolean"):
            raise ValueError(f"unsupported descriptor type: {self.value_type!r}")
        if not self.unit or not self.family:
            raise ValueError("descriptor unit and family must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "unit": self.unit,
            "family": self.family,
        }


@dataclass(frozen=True, slots=True)
class DescriptorRegistry:
    version: str
    descriptors: tuple[DescriptorSpec, ...]
    registry_hash: str = field(init=False)
    _by_name: Mapping[str, DescriptorSpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("descriptor registry version must be non-empty")
        descriptors = tuple(self.descriptors)
        by_name = {descriptor.name: descriptor for descriptor in descriptors}
        if len(by_name) != len(descriptors):
            raise ValueError("descriptor names must be unique")
        ordered = tuple(sorted(descriptors, key=lambda descriptor: descriptor.name))
        payload = {
            "schema_version": 1,
            "version": self.version,
            "descriptors": [descriptor.to_dict() for descriptor in ordered],
        }
        object.__setattr__(self, "descriptors", ordered)
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))
        object.__setattr__(self, "registry_hash", _canonical_hash(payload))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def get(self, name: str) -> DescriptorSpec | None:
        return self._by_name.get(name)

    def require(self, name: str) -> DescriptorSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown descriptor: {name}") from exc

    def validate_values(self, values: Mapping[str, DescriptorValue]) -> None:
        actual = set(values)
        expected = set(self._by_name)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(f"descriptor schema mismatch; missing={missing}, unknown={unknown}")
        for name, value in values.items():
            value_type = self._by_name[name].value_type
            if value_type == "boolean" and type(value) is not bool:
                raise ValueError(f"descriptor {name!r} must be boolean")
            if value_type == "number" and (
                type(value) not in (int, float) or not math.isfinite(float(value))
            ):
                raise ValueError(f"descriptor {name!r} must be finite and numeric")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "version": self.version,
            "descriptors": [descriptor.to_dict() for descriptor in self.descriptors],
            "registry_hash": self.registry_hash,
        }
