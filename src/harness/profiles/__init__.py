"""HarnessProfile registry.

External-container harnesses declare themselves here; the loader looks up
the concrete profile class by name when validating
``ContainerHarnessConfig.profile`` and when ``ContainerHarness`` resolves
the profile at episode start.

Adding a new profile: define a subclass of :class:`HarnessProfile` in its
own module, declare a ``profile_config_class``, and add it to
``REGISTRY``. The outer :class:`HarnessConfig` schema does not need
modification.
"""

from __future__ import annotations

from typing import Any

from src.harness.base import HarnessError
from src.harness.profiles.base import HarnessProfile
from src.harness.profiles.ms_agent import MsAgentProfile

REGISTRY: dict[str, type[HarnessProfile]] = {
    MsAgentProfile.name: MsAgentProfile,
}


def resolve_profile(name: str, profile_config: dict[str, Any]) -> HarnessProfile:
    if name not in REGISTRY:
        raise HarnessError(
            f"unknown harness profile {name!r}; registered: {sorted(REGISTRY)}"
        )
    cls = REGISTRY[name]
    config = cls.profile_config_class.model_validate(profile_config)
    return cls(config)


__all__ = ["REGISTRY", "resolve_profile", "HarnessProfile"]
