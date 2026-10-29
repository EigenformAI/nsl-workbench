"""Wrapper script baked into the ms-agent harness image.

Reads the rendered config files from /work and drives ms-agent via the
Python API. Avoids the CLI (``ms-agent run --config``), which:

- Has no ``--query-file`` flag — the prompt must come from argv or a
  Python-level object.
- Ignores top-level ``mcpServers`` in ``agent.yaml``; the CLI loader only
  reads ``tools.<name>`` entries. Passing ``mcp_config`` to the Python
  ``LLMAgent`` constructor is the documented, format-independent path.

Files the driver (``ContainerHarness`` + ``MsAgentProfile``) writes here:

- ``/work/agent.yaml``     — llm section only (openai + our shim URL).
- ``/work/mcp_config.json`` — ``{"mcpServers": {"nsl": {"type":
  "streamable_http", "url": ..., "headers": {...}}}}``.
- ``/work/query.txt``       — rendered user query (env + capability manifest).
  When ``/work/workflow.yaml`` is present, each step has its own agent config.

Transcript path: ms-agent's ``save_history`` writes
``<output_dir>/.memory/<tag>.json`` after each round (hidden directory —
verified against ms_agent/utils/utils.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


EXIT_OK = 0


def _load_inputs(scratch: Path = Path("/work")):
    cfg = OmegaConf.load(scratch / "agent.yaml")
    mcp_config = json.loads((scratch / "mcp_config.json").read_text())
    query = (scratch / "query.txt").read_text()
    return cfg, mcp_config, query


def _validate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict) or not spec:
        raise RuntimeError("workflow.yaml must contain at least one step")
    for name, entry in spec.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("workflow.yaml step names must be non-empty strings")
        if not isinstance(entry, dict):
            raise RuntimeError(f"step {name!r}: workflow entry must be a mapping")
        agent_config = entry.get("agent_config")
        if not isinstance(agent_config, str) or not agent_config.strip():
            raise RuntimeError(f"step {name!r}: missing agent_config")
        context_mode = entry.get("context_mode", "inherit")
        if context_mode not in {"inherit", "isolated"}:
            raise RuntimeError(
                f"step {name!r}: unsupported context_mode={context_mode!r}"
            )
        next_steps = entry.get("next", []) or []
        if not isinstance(next_steps, list):
            raise RuntimeError(f"step {name!r}: next must be a list")
        if len(next_steps) > 1:
            raise RuntimeError(f"step {name!r}: fan-out is not supported")
        for next_name in next_steps:
            if next_name not in spec:
                raise RuntimeError(
                    f"step {name!r}: next={next_name!r} not found in workflow"
                )
        error_target = entry.get("on_error")
        if error_target is not None and error_target not in spec:
            raise RuntimeError(
                f"step {name!r}: on_error={error_target!r} not found in workflow"
            )


def _topological_chain(spec: dict[str, Any]) -> list[str]:
    incoming = {name: 0 for name in spec}
    outgoing: dict[str, list[str]] = {}
    for name, entry in spec.items():
        targets = list(entry.get("next", []) or [])
        error_target = entry.get("on_error")
        if error_target is not None:
            targets.append(error_target)
        outgoing[name] = targets
        for target in targets:
            incoming[target] += 1

    queue = [name for name, degree in incoming.items() if degree == 0]
    visited_count = 0
    remaining = dict(incoming)
    while queue:
        name = queue.pop()
        visited_count += 1
        for target in outgoing[name]:
            remaining[target] -= 1
            if remaining[target] == 0:
                queue.append(target)
    if visited_count != len(spec):
        raise RuntimeError("workflow contains a cycle")

    roots = [name for name, degree in incoming.items() if degree == 0]
    if len(roots) != 1:
        if not roots:
            raise RuntimeError("workflow contains a cycle or has no root")
        raise RuntimeError(f"workflow.yaml must have exactly one root; got {roots}")

    order: list[str] = []
    current: str | None = roots[0]
    seen: set[str] = set()
    while current is not None:
        if current in seen:
            raise RuntimeError(f"workflow contains a cycle at step {current!r}")
        seen.add(current)
        order.append(current)
        next_steps = spec[current].get("next", []) or []
        current = next_steps[0] if next_steps else None

    reachable: set[str] = set()
    stack = [roots[0]]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        stack.extend(outgoing[name])
    missing = sorted(set(spec) - reachable)
    if missing:
        raise RuntimeError(f"workflow has unreachable steps: {missing}")
    return order


def _step_query(yaml_path: Path) -> str:
    cfg = OmegaConf.load(yaml_path)
    query = getattr(getattr(cfg, "prompt", None), "query", "") or ""
    if not isinstance(query, str) or not query.strip():
        raise RuntimeError(f"step config {yaml_path} has empty prompt.query")
    return query


def _emit(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(f"[nsl-event] {json.dumps(payload)}", file=sys.stderr, flush=True)


def _load_agent_class():
    from ms_agent import LLMAgent

    return LLMAgent


async def _close_agent(agent: Any) -> None:
    close = getattr(agent, "close", None)
    if close is None:
        close = getattr(agent, "cleanup_tools", None)
    if close is None:
        return
    with contextlib.suppress(Exception):
        result = close()
        if asyncio.iscoroutine(result):
            await result


async def _run_workflow(
    scratch: Path,
    mcp_config: dict[str, Any],
    entry_query: str,
) -> int:
    spec = yaml.safe_load((scratch / "workflow.yaml").read_text())
    _validate_spec(spec)
    order = _topological_chain(spec)

    inputs_by_step: dict[str, Any] = {order[0]: entry_query}
    visited: set[str] = set()
    current: str | None = order[0]
    last_outputs: list[Any] | None = None
    recovery_fired = False
    LLMAgent = _load_agent_class()

    while current is not None:
        if current in visited:
            raise RuntimeError(
                f"step {current!r} revisited; on_error must point forward"
            )
        visited.add(current)
        entry = spec[current]
        step_cfg = OmegaConf.load(scratch / entry["agent_config"])
        step_cfg.tag = current
        step_cfg.trust_remote_code = True
        step_cfg.local_dir = str(scratch)
        agent = LLMAgent(
            config=step_cfg,
            # The step config carries its own workflow header and tool allowlist.
            # Passing the global server entry here replaces that complete mapping
            # inside ms-agent and silently drops both fields.
            mcp_config={},
            tag=current,
            trust_remote_code=True,
        )
        step_input = inputs_by_step[current]
        t0 = time.monotonic()
        _emit(
            "workflow_step_enter",
            step=current,
            input_kind=("string" if isinstance(step_input, str) else "list"),
            context_mode=entry.get("context_mode", "inherit"),
        )
        try:
            try:
                outputs = await agent.run(step_input)
            except Exception as exc:  # noqa: BLE001
                error_target = entry.get("on_error")
                duration_s = round(time.monotonic() - t0, 3)
                _emit(
                    "workflow_step_exit",
                    step=current,
                    outcome="error",
                    exc_type=type(exc).__name__,
                    exc_msg=str(exc)[:512],
                    duration_s=duration_s,
                    recovery=(error_target is not None),
                )
                if error_target is None:
                    print(
                        f"[nsl] ms-agent step {current!r} failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    raise

                print(
                    f"[nsl] ms-agent step {current!r} failed: "
                    f"{type(exc).__name__}: {exc}; routing to {error_target!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                recovery_fired = True
                recovery_mode = spec[error_target].get("context_mode", "isolated")
                if recovery_mode == "inherit" and last_outputs is not None:
                    inputs_by_step[error_target] = last_outputs
                else:
                    inputs_by_step[error_target] = _step_query(
                        scratch / spec[error_target]["agent_config"]
                    )
                current = error_target
                continue
        finally:
            await _close_agent(agent)

        outputs = outputs or []
        _emit(
            "workflow_step_exit",
            step=current,
            outcome="ok",
            message_count=len(outputs),
            duration_s=round(time.monotonic() - t0, 3),
        )
        last_outputs = outputs
        next_steps = entry.get("next", []) or []
        if not next_steps:
            break
        next_step = next_steps[0]
        next_mode = spec[next_step].get("context_mode", "inherit")
        if next_mode == "isolated":
            inputs_by_step[next_step] = _step_query(
                scratch / spec[next_step]["agent_config"]
            )
        else:
            inputs_by_step[next_step] = outputs
        current = next_step

    _emit(
        "workflow_finished",
        message_count=(len(last_outputs) if last_outputs else 0),
        recovery_fired=recovery_fired,
        steps_visited=sorted(visited),
    )
    print(
        f"[nsl] ms-agent workflow finished with "
        f"{len(last_outputs) if last_outputs else 0} messages",
        file=sys.stderr,
    )
    return EXIT_OK


async def _main() -> int:
    scratch = Path("/work")
    cfg, mcp_config, query = _load_inputs(scratch)
    if (scratch / "workflow.yaml").exists():
        return await _run_workflow(scratch, mcp_config, query)
    LLMAgent = _load_agent_class()
    agent = LLMAgent(config=cfg, mcp_config=mcp_config)
    try:
        messages = await agent.run(query)
    finally:
        await _close_agent(agent)
    print(
        f"[nsl] ms-agent finished with {len(messages) if messages else 0} messages",
        file=sys.stderr,
    )
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(_main()))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        print(
            f"[nsl] ms-agent failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
