"""``ensure_harness_image`` contract.

Covers the entire decision tree for image provisioning so the single
bootstrap call at each entry point is trustworthy:

* Image already present → no pull/build.
* Image missing, no build → pull.
* Image missing, build set → build, never pull.
* Image present, build.force=True → rebuild with ``nocache=True``.
* Missing build context → HarnessError with a useful message.
* Docker daemon unreachable / pull fails / build fails → HarnessError
  that wraps the underlying cause without eating its details.
* Schema-level: registry-prefixed image + build set rejected at load.

The Docker SDK is faked end-to-end — these tests must never touch a
real daemon or filesystem context beyond ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from docker.errors import APIError, BuildError, DockerException, ImageNotFound
from pydantic import ValidationError

from src.harness.base import HarnessError
from src.harness.provisioning import ensure_harness_image
from src.typing.config import (
    AppConfig,
    ContainerHarnessBuildConfig,
    ContainerHarnessConfig,
)


def _min_app_payload() -> dict:
    return {
        "model_name": "test",
        "code_host_cache_path": "./code/",
        "container_ids": ["svc-1"],
        "train_data_save_folder": "./data/",
        "task": {"class": "tests.fixtures.stub_task.ValidStubTask"},
    }


class _FakeImages:
    """Records every images.get / images.pull / images.build call."""

    def __init__(
        self,
        *,
        present: bool,
        build_raises: Exception | None = None,
        pull_raises: Exception | None = None,
        labels: Dict[str, str] | None = None,
    ) -> None:
        self.present = present
        self.build_raises = build_raises
        self.pull_raises = pull_raises
        self.labels = labels or {}
        self.get_calls: List[str] = []
        self.pull_calls: List[str] = []
        self.build_calls: List[Dict[str, Any]] = []

    def get(self, tag: str) -> Any:
        self.get_calls.append(tag)
        if not self.present:
            raise ImageNotFound(tag)
        image = MagicMock()
        image.short_id = "sha256:deadbeef"
        # Provide labels through both common surfaces — Docker SDK exposes
        # image.labels (a dict) and the underlying attrs payload.
        image.labels = dict(self.labels)
        image.attrs = {"Config": {"Labels": dict(self.labels)}}
        return image

    def pull(self, tag: str) -> Any:
        self.pull_calls.append(tag)
        if self.pull_raises is not None:
            raise self.pull_raises
        self.present = True

    def build(self, **kwargs: Any):
        self.build_calls.append(kwargs)
        if self.build_raises is not None:
            raise self.build_raises
        image = MagicMock()
        image.short_id = "sha256:cafef00d"
        return image, iter([{"stream": "Step 1/2 : FROM scratch\n"}])


class _FakeClient:
    def __init__(self, images: _FakeImages) -> None:
        self.images = images


def _pull_cfg() -> ContainerHarnessConfig:
    return ContainerHarnessConfig(
        profile="ms_agent",
        image="nsl/ms-agent:0.1.0",
    )


def _build_cfg(
    *,
    context: str = "docker/ms-agent",
    force: bool = False,
    build_args: Dict[str, str] | None = None,
) -> ContainerHarnessConfig:
    return ContainerHarnessConfig(
        profile="ms_agent",
        image="nsl/ms-agent:0.1.0",
        build=ContainerHarnessBuildConfig(
            context=context,
            build_args=build_args or {},
            force=force,
        ),
    )


def test_ensure_present_skips() -> None:
    images = _FakeImages(present=True)
    ensure_harness_image(_pull_cfg(), client=_FakeClient(images))  # type: ignore[arg-type]
    assert images.get_calls == ["nsl/ms-agent:0.1.0"]
    assert images.pull_calls == []
    assert images.build_calls == []


def test_ensure_pulls_when_no_build() -> None:
    images = _FakeImages(present=False)
    ensure_harness_image(_pull_cfg(), client=_FakeClient(images))  # type: ignore[arg-type]
    assert images.pull_calls == ["nsl/ms-agent:0.1.0"]
    assert images.build_calls == []


def test_ensure_builds_when_build_set() -> None:
    images = _FakeImages(present=False)
    cfg = _build_cfg(build_args={"MS_AGENT_VERSION": "0.1"})
    ensure_harness_image(cfg, client=_FakeClient(images))  # type: ignore[arg-type]
    assert images.pull_calls == []
    assert len(images.build_calls) == 1
    call = images.build_calls[0]
    assert call["tag"] == "nsl/ms-agent:0.1.0"
    assert call["dockerfile"] == "Dockerfile"
    # User-supplied build args are passed through; provisioning auto-
    # injects NSL_BUILD_CONTEXT_SHA on top so the resulting image carries
    # a content-hash label for next-run drift detection.
    assert call["buildargs"]["MS_AGENT_VERSION"] == "0.1"
    assert "NSL_BUILD_CONTEXT_SHA" in call["buildargs"]
    assert call["nocache"] is False
    # Resolved path should end with the context directory.
    assert call["path"].endswith("docker/ms-agent")


def test_ensure_forces_rebuild_when_force_true() -> None:
    # Image is ALREADY present — force=True must still trigger build.
    images = _FakeImages(present=True)
    cfg = _build_cfg(force=True)
    ensure_harness_image(cfg, client=_FakeClient(images))  # type: ignore[arg-type]
    # Short-circuit skipped when force=True.
    assert images.get_calls == []
    assert len(images.build_calls) == 1
    assert images.build_calls[0]["nocache"] is True


def test_ensure_raises_on_missing_context(tmp_path: Path) -> None:
    images = _FakeImages(present=False)
    cfg = _build_cfg(context=str(tmp_path / "does-not-exist"))
    with pytest.raises(HarnessError) as exc_info:
        ensure_harness_image(cfg, client=_FakeClient(images))  # type: ignore[arg-type]
    assert "is not a directory" in str(exc_info.value)
    assert images.build_calls == []


def test_ensure_wraps_pull_failure() -> None:
    err = APIError("denied: registry auth required")
    images = _FakeImages(present=False, pull_raises=err)
    with pytest.raises(HarnessError) as exc_info:
        ensure_harness_image(_pull_cfg(), client=_FakeClient(images))  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "nsl/ms-agent:0.1.0" in msg
    assert "[harness.container.build]" in msg


def test_ensure_wraps_daemon_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**_kwargs: Any) -> Any:
        raise DockerException("cannot connect to docker daemon")

    monkeypatch.setattr("src.harness.provisioning.docker.from_env", _raise)
    with pytest.raises(HarnessError) as exc_info:
        ensure_harness_image(_pull_cfg())
    assert "docker info" in str(exc_info.value)


def test_ensure_wraps_build_failure() -> None:
    build_log = [
        {"stream": "Step 1/3 : FROM python:3.11-slim\n"},
        {"error": "pip install failed: no matching distribution\n"},
    ]
    err = BuildError(reason="pip install failed", build_log=iter(build_log))
    images = _FakeImages(present=False, build_raises=err)
    with pytest.raises(HarnessError) as exc_info:
        ensure_harness_image(_build_cfg(), client=_FakeClient(images))  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "nsl/ms-agent:0.1.0" in msg
    assert "pip install failed" in msg


def test_validator_rejects_image_prefix_with_build() -> None:
    payload = _min_app_payload()
    payload["harness"] = {
        "name": "container",
        "container": {
            "profile": "ms_agent",
            "image": "example.invalid/test-harness:0.1.0",
            "build": {"context": "docker/ms-agent"},
            "profile_config": {
                "model": "claude-sonnet-4-6",
                "max_chat_round": 60,
                "tool_call_timeout": 90,
                "transcript_tag": "episode",
            },
        },
    }
    with pytest.raises(ValidationError) as exc_info:
        AppConfig.model_validate(payload)
    assert "registry reference" in str(exc_info.value)


def test_validator_accepts_bare_tag_with_build() -> None:
    """Sanity: the validator must not break the happy path."""
    cfg = _build_cfg()
    assert cfg.build is not None
    assert cfg.build.context == "docker/ms-agent"


# ---------------------------------------------------------------------------
# Stale-image drift detection
# ---------------------------------------------------------------------------
#
# An image's presence does not establish that it matches the configured source.
#
# Mitigation: stamp the image with a content hash of the build context at
# build time (via Docker label nsl.build-context-sha), and on subsequent
# runs compute the current context hash and compare. On mismatch the
# image is auto-rebuilt with nocache=True so the new content actually
# lands in the image.


def _write_context(tmp_path: Path, files: Dict[str, str]) -> Path:
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    for name, content in files.items():
        path = ctx / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return ctx


def test_drift_detection_no_rebuild_when_label_matches(tmp_path: Path) -> None:
    """Image present, build set, label hash matches context → skip rebuild."""
    from src.harness.provisioning import _hash_build_context

    ctx = _write_context(tmp_path, {"Dockerfile": "FROM scratch\n", "run.py": "x = 1\n"})
    expected_sha = _hash_build_context(ctx)

    images = _FakeImages(
        present=True,
        labels={"nsl.build-context-sha": expected_sha},
    )
    cfg = _build_cfg(context=str(ctx))
    ensure_harness_image(cfg, client=_FakeClient(images))  # type: ignore[arg-type]

    assert images.build_calls == [], "no rebuild expected when hashes match"


def test_drift_detection_rebuilds_on_mismatch(tmp_path: Path) -> None:
    """Image present, build set, label hash differs from context → rebuild."""
    ctx = _write_context(tmp_path, {"Dockerfile": "FROM scratch\n", "run.py": "x = 2\n"})

    images = _FakeImages(
        present=True,
        labels={"nsl.build-context-sha": "0" * 64},  # stale
    )
    cfg = _build_cfg(context=str(ctx))
    ensure_harness_image(cfg, client=_FakeClient(images))  # type: ignore[arg-type]

    assert len(images.build_calls) == 1
    # Auto-rebuild forces nocache=True so cached layers don't mask the drift.
    assert images.build_calls[0]["nocache"] is True


def test_drift_detection_rebuilds_when_label_absent(tmp_path: Path) -> None:
    """Older image with no nsl.build-context-sha label: treat as drift."""
    ctx = _write_context(tmp_path, {"Dockerfile": "FROM scratch\n", "run.py": "x = 3\n"})

    images = _FakeImages(present=True, labels={})
    cfg = _build_cfg(context=str(ctx))
    ensure_harness_image(cfg, client=_FakeClient(images))  # type: ignore[arg-type]

    assert len(images.build_calls) == 1


def test_drift_rebuild_passes_current_sha_as_build_arg(tmp_path: Path) -> None:
    """When the rebuild fires, the build args must carry the current SHA so
    the new image gets a fresh label and won't trigger another rebuild on
    the next run."""
    from src.harness.provisioning import _hash_build_context

    ctx = _write_context(tmp_path, {"Dockerfile": "FROM scratch\n", "run.py": "x = 4\n"})
    current_sha = _hash_build_context(ctx)

    images = _FakeImages(present=True, labels={"nsl.build-context-sha": "stale"})
    cfg = _build_cfg(context=str(ctx))
    ensure_harness_image(cfg, client=_FakeClient(images))  # type: ignore[arg-type]

    assert len(images.build_calls) == 1
    buildargs = images.build_calls[0]["buildargs"]
    assert buildargs.get("NSL_BUILD_CONTEXT_SHA") == current_sha


def test_drift_detection_skipped_for_pull_only_configs(tmp_path: Path) -> None:
    """Pull-only configs (build=None) have no context to hash; drift
    detection is silently a no-op for them."""
    images = _FakeImages(present=True, labels={})
    ensure_harness_image(_pull_cfg(), client=_FakeClient(images))  # type: ignore[arg-type]

    assert images.build_calls == []
    assert images.pull_calls == []


def test_first_build_always_passes_sha_arg(tmp_path: Path) -> None:
    """When the image is missing and we build for the first time, the
    SHA build arg is included so the resulting image carries the label."""
    from src.harness.provisioning import _hash_build_context

    ctx = _write_context(tmp_path, {"Dockerfile": "FROM scratch\n", "run.py": "x = 5\n"})
    current_sha = _hash_build_context(ctx)

    images = _FakeImages(present=False)
    cfg = _build_cfg(context=str(ctx))
    ensure_harness_image(cfg, client=_FakeClient(images))  # type: ignore[arg-type]

    assert len(images.build_calls) == 1
    buildargs = images.build_calls[0]["buildargs"]
    assert buildargs.get("NSL_BUILD_CONTEXT_SHA") == current_sha


def test_hash_build_context_ignores_pycache(tmp_path: Path) -> None:
    """Cache directories must not affect the content hash, otherwise a
    pyc generated on first build would flip the hash on the next run
    and trigger a perpetual rebuild loop."""
    from src.harness.provisioning import _hash_build_context

    ctx = _write_context(tmp_path, {"Dockerfile": "FROM scratch\n", "run.py": "x = 1\n"})
    sha_clean = _hash_build_context(ctx)

    pycache = ctx / "__pycache__"
    pycache.mkdir()
    (pycache / "run.cpython-311.pyc").write_bytes(b"\x00\x01\x02\x03")
    sha_with_pycache = _hash_build_context(ctx)

    assert sha_clean == sha_with_pycache, (
        "context hash must ignore __pycache__ noise"
    )


def test_hash_build_context_changes_with_content(tmp_path: Path) -> None:
    """Sanity: changing a real build-input file must change the hash."""
    from src.harness.provisioning import _hash_build_context

    ctx = _write_context(tmp_path, {"Dockerfile": "FROM scratch\n", "run.py": "x = 1\n"})
    sha_before = _hash_build_context(ctx)

    (ctx / "run.py").write_text("x = 999\n")
    sha_after = _hash_build_context(ctx)

    assert sha_before != sha_after
