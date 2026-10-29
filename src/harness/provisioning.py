"""Idempotent bootstrap that ensures a harness image exists locally.

Called once per entry point from the main process, before worker slots
spawn. Safe to call multiple times: ``images.get`` short-circuits when
the image is already present (unless ``build.force`` is True).

Not a classmethod on :class:`ContainerHarness` — provisioning is an
orchestrator-level concern, while ContainerHarness is a per-episode
object. Keeping them separate avoids the classmethod-without-cls smell
and makes :func:`ensure_harness_image` trivially usable by
hermes / other future harnesses without subclassing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import docker
from docker.errors import APIError, BuildError, DockerException, ImageNotFound
from loguru import logger

from src.harness.base import HarnessError
from src.paths import PROJECT_ROOT
from src.typing.config import ContainerHarnessConfig

# Docker label that stamps the build-context content hash into the image
# so a later run can detect drift and auto-rebuild instead of silently
# running a stale image.
_BUILD_CONTEXT_SHA_LABEL = "nsl.build-context-sha"
_BUILD_CONTEXT_SHA_BUILD_ARG = "NSL_BUILD_CONTEXT_SHA"

# Filenames/dirs in the build context that are noise — including them in
# the content hash would flap on every build (e.g. __pycache__ generated
# during a python build, hidden editor swap files). Keep this list narrow:
# false positives mean spurious rebuilds, false negatives mean silent
# staleness.
_HASH_IGNORE_DIRS = frozenset({"__pycache__"})


def _hash_build_context(context: Path) -> str:
    """Return a deterministic content hash of ``context``'s build inputs.

    Walks the directory recursively, sorts by relative path, and feeds
    each (path, contents) pair into sha256. Skips noise directories
    (see ``_HASH_IGNORE_DIRS``) and any path component starting with
    ``.`` (hidden files / dirs — git metadata, editor backups). This
    matches the spirit of ``.dockerignore`` without parsing it: we
    intentionally over-include rather than rely on per-context dockerignore
    rules, since the consequence of a missing dockerignore is "spurious
    rebuild" (cheap), while the consequence of skipping a real input is
    "silent staleness" (the bug we're fixing).
    """
    h = hashlib.sha256()
    for path in sorted(context.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(context).parts
        if any(p in _HASH_IGNORE_DIRS or p.startswith(".") for p in rel_parts):
            continue
        rel = path.relative_to(context).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _image_label(image: Any, name: str) -> str | None:
    labels = getattr(image, "labels", None)
    if isinstance(labels, dict):
        value = labels.get(name)
        if isinstance(value, str):
            return value
    attrs = getattr(image, "attrs", None) or {}
    config = attrs.get("Config") if isinstance(attrs, dict) else None
    if isinstance(config, dict):
        attr_labels = config.get("Labels") or {}
        if isinstance(attr_labels, dict):
            value = attr_labels.get(name)
            if isinstance(value, str):
                return value
    return None


def ensure_harness_image(
    config: ContainerHarnessConfig,
    client: Any = None,
) -> None:
    """Make sure ``config.image`` exists locally.

    * ``config.build`` set → build from ``build.context`` when the image
      is missing OR ``build.force`` is True.
    * ``config.build`` unset → pull ``config.image``.

    Idempotent, NOT singleton: calling twice on a present image is a
    no-op (one ``images.get`` round-trip). A future parallel-runs
    scenario that needs strict singleton semantics would layer a file
    lock on top; not needed today.

    Wraps daemon/pull/build failures as :class:`HarnessError` so the
    caller sees one exception type regardless of whether the daemon
    died, the registry denied auth, or a Dockerfile step failed.
    """
    try:
        client = client or docker.from_env()
    except DockerException as exc:
        raise HarnessError(
            f"docker daemon unreachable ({type(exc).__name__}: {exc}); "
            f"ensure `docker info` succeeds on this host before running."
        ) from exc

    want_build = config.build is not None
    force = bool(want_build and config.build is not None and config.build.force)

    if not force:
        try:
            image = client.images.get(config.image)
        except ImageNotFound:
            image = None
        if image is not None:
            if config.build is not None:
                # Drift detection: compare the build-context hash baked
                # into the image at build time against the current
                # context's hash. Mismatch → auto-rebuild with nocache.
                drift_reason = _detect_context_drift(config.build, image)
                if drift_reason is None:
                    logger.info(
                        f"harness image {config.image} present locally "
                        f"(content hash matches build context)"
                    )
                    return
                logger.warning(
                    f"harness image {config.image} is stale: {drift_reason}; "
                    f"rebuilding with nocache=True"
                )
                # Fall through to _build, with force=True semantics applied
                # locally (we do not mutate config.build.force — the config
                # is shared across callers).
                _build(config, client, force_nocache=True)
                return
            logger.info(f"harness image {config.image} present locally")
            return

    if want_build:
        _build(config, client)
    else:
        _pull(config, client)


def _detect_context_drift(
    build: Any,  # ContainerHarnessBuildConfig
    image: Any,
) -> str | None:
    """Return a human-readable reason if the image is stale, else None.

    Stale = build context's content hash differs from the hash baked
    into the image's ``nsl.build-context-sha`` label. Older images
    without the label are also treated as stale (we cannot prove they
    match). This prevents silently running an image whose contents differ
    from the configured source.
    """
    context = (PROJECT_ROOT / build.context).resolve()
    if not context.is_dir():
        # Don't second-guess _build; it raises with the canonical message.
        return None
    try:
        current_sha = _hash_build_context(context)
    except OSError as exc:
        logger.warning(
            f"build context hash failed for {context}: {exc}; "
            f"skipping drift detection"
        )
        return None
    baked_sha = _image_label(image, _BUILD_CONTEXT_SHA_LABEL)
    if baked_sha is None:
        return (
            f"image carries no {_BUILD_CONTEXT_SHA_LABEL} label "
            f"(predates drift-detection support)"
        )
    if baked_sha != current_sha:
        return (
            f"build context {context} hash {current_sha[:12]}… "
            f"differs from image-baked {baked_sha[:12]}…"
        )
    return None


def ensure_configured_harness(app_config: Any, *, rebuild: bool = False) -> None:
    """Provision the harness image named by ``app_config`` if any.

    No-op for non-container harnesses. Safe to call on configs that
    don't set ``harness.container`` (e.g. ``orchestrator_modes``).

    ``rebuild=True`` mirrors the ``--rebuild-harness`` CLI flag: when
    the config declares a local build, flip ``build.force`` to True so
    the next build is nocache. Warns (but does not raise) when rebuild
    is requested for a pull-only config — there's nothing to rebuild.
    """
    harness = getattr(app_config, "harness", None)
    if harness is None:
        return
    if harness.name != "container" or harness.container is None:
        if rebuild:
            logger.warning(
                "rebuild requested but harness.container is not configured; ignoring."
            )
        return
    if rebuild:
        if harness.container.build is None:
            logger.warning(
                "rebuild requested but no [harness.container.build] in config; "
                "image will be pulled, not rebuilt."
            )
        else:
            harness.container.build.force = True
    ensure_harness_image(harness.container)


def _build(
    config: ContainerHarnessConfig,
    client: Any,
    *,
    force_nocache: bool = False,
) -> None:
    assert config.build is not None  # narrowed by caller
    build = config.build

    context = (PROJECT_ROOT / build.context).resolve()
    if not context.is_dir():
        raise HarnessError(
            f"harness build context {context} is not a directory "
            f"(resolved from build.context={build.context!r})"
        )

    # Always stamp the current context hash into a build arg so the
    # Dockerfile can write it as a label. Auto-injected — callers do not
    # need to know about the drift-detection plumbing.
    buildargs = dict(build.build_args)
    try:
        buildargs[_BUILD_CONTEXT_SHA_BUILD_ARG] = _hash_build_context(context)
    except OSError as exc:
        # Not fatal — proceed without the label. Drift detection on the
        # *next* run will then trigger another rebuild, which is the
        # safe direction.
        logger.warning(
            f"could not hash build context {context} for label: {exc}"
        )
    nocache = bool(build.force or force_nocache)

    logger.info(
        f"building harness image {config.image} from {context} "
        f"(force={build.force}, force_nocache={force_nocache}, args={buildargs})"
    )
    try:
        image, build_logs = client.images.build(
            path=str(context),
            dockerfile=build.dockerfile,
            tag=config.image,
            buildargs=buildargs,
            rm=True,
            nocache=nocache,
        )
    except BuildError as exc:
        tail = _tail_build_log(getattr(exc, "build_log", ()), max_entries=30)
        raise HarnessError(
            f"docker build failed for {config.image}: {exc}\n{tail}"
        ) from exc
    except APIError as exc:
        raise HarnessError(
            f"docker build failed for {config.image}: {exc}"
        ) from exc

    # Consume the generator so trailing warnings surface. ``build_logs``
    # is an iterable of dicts like ``{"stream": "Step 1/4 : FROM ...\n"}``.
    for entry in build_logs:
        if not isinstance(entry, dict):
            continue
        msg = (entry.get("stream") or "").rstrip()
        if msg:
            logger.debug(f"[build {config.image}] {msg}")
    logger.info(f"harness image {config.image} built (id={image.short_id})")


def _pull(config: ContainerHarnessConfig, client: Any) -> None:
    logger.info(f"harness image {config.image} missing; pulling")
    try:
        client.images.pull(config.image)
    except (APIError, ImageNotFound) as exc:
        raise HarnessError(
            f"pulling {config.image} failed: {exc}. "
            f"If this image is built locally, set "
            f"[harness.container.build] with context=<path-to-Dockerfile-dir> "
            f"in your config."
        ) from exc


def _tail_build_log(build_log: Iterable[Any], *, max_entries: int) -> str:
    try:
        entries = list(build_log)
    except Exception:  # noqa: BLE001
        return "<build log unreadable>"
    tail = entries[-max_entries:]
    lines: list[str] = []
    for entry in tail:
        if not isinstance(entry, dict):
            continue
        msg = entry.get("stream") or entry.get("error") or ""
        if isinstance(msg, str):
            stripped = msg.rstrip()
            if stripped:
                lines.append(stripped)
    return "\n".join(lines)


__all__ = ["ensure_harness_image", "ensure_configured_harness"]
