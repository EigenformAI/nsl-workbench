"""MsAgentProfile — adapter for ModelScope's ``ms-agent``.

ms-agent is driven via its Python API (``LLMAgent(config, mcp_config).run``)
rather than the CLI, which lacks a ``--query-file`` flag. A wrapper script
baked into ``docker/ms-agent/run.py`` reads the rendered config + query
and invokes the Python API, then ms-agent writes its own transcript into
``<output_dir>/.memory/<tag>.json`` (hidden directory — verified against
``ms_agent/utils/utils.py:save_history``).

Config files produced:

- ``agent.yaml``: LLM section only (``llm.service: openai`` + ``openai_base_url`` +
  ``openai_api_key`` pointing at our shim). Top-level ``mcpServers`` is
  deliberately absent — ms-agent's CLI loader ignores it; MCP config is
  passed programmatically via ``mcp_config.json``.
- ``mcp_config.json``: ``{"mcpServers": {"nsl": {"type": "streamable_http",
  "url": mcp_url, "headers": {"Authorization": "Bearer <token>"}}}}``.
- ``query.txt``: the harness-native query string.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.harness.context import project_step_constraints
from src.harness.profiles.base import HarnessProfile, MissingTranscriptError
from src.harness.tool_contract import ToolCallContractProbe
from src.task.types import (
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
    EpisodeConstraints,
    TaskPromptSpec,
    Workflow,
    WorkflowStep,
)


_QUERY_INJECTION_CALLBACK = "inject_query_callback.py"


class MsAgentProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_chat_round: int = 60
    tool_call_timeout: int = 90
    output_dir: str = "/work/output"
    transcript_tag: str = "episode"
    extra_yaml: dict[str, Any] = Field(default_factory=dict)


def _last_content_bearing_assistant(transcript: dict[str, Any]) -> str | None:
    """Return the most recent non-empty assistant content in the transcript.

    ms-agent synthesises a trailing empty assistant message when the
    ``max_chat_round`` cap fires; we skip over those to find the real
    final answer.
    """
    messages = transcript.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _assistant_step_count(transcript: dict[str, Any]) -> int:
    """Count content-bearing assistant turns.

    ms-agent appends an empty assistant trailer when ``max_chat_round`` is
    exhausted. Treating that synthetic cutoff marker as a real turn inflates
    ``llm_turns`` by one.
    """
    messages = transcript.get("messages") or []
    return sum(
        1
        for msg in messages
        if msg.get("role") == "assistant"
        and isinstance(msg.get("content"), str)
        and msg["content"].strip()
    )


def _merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _safe_config_filename(step_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", step_name).strip("._")
    if not safe:
        raise ValueError("workflow step name cannot render to an empty filename")
    return f"{safe}.yaml"


def _normalise_transcript_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, list):
        return {"messages": payload}
    if isinstance(payload, dict):
        return payload
    return None


class MsAgentProfile(HarnessProfile):
    name = "ms_agent"
    profile_config_class = MsAgentProfileConfig

    # Narrow the config type for the concrete profile — the ABC stores it
    # as BaseModel so declare it here for readability at call sites.
    config: MsAgentProfileConfig

    def __init__(self, config: MsAgentProfileConfig) -> None:
        super().__init__(config)

    # --- Query rendering ---

    def render_query(
        self,
        prompt_spec: TaskPromptSpec,
        *,
        constraints: EpisodeConstraints | None = None,
    ) -> str:
        # Without this preamble, models trained to be helpful emit a plain
        # text description of what they'd do — ms-agent's ReAct loop sees
        # no tool call and terminates on the first assistant turn.
        # Don't advertise an example name format here: ms-agent keys its
        # tool registry as f"{server}---{tool}" (TOOL_SPLITER='---', see
        # ms_agent/tools/tool_manager.py), and the model will see those
        # canonical names directly in the tool schema sent by ms-agent.
        # Naming a different format in prose can teach the model to emit a
        # name that ms-agent's _tool_index does not contain.
        preamble = (
            "You operate via MCP tools registered under the 'nsl' server. "
            "To take any action, CALL the appropriate tool from the "
            "schema you have been given — do not describe what you would "
            "do in plain text. Each capability the task lists below is "
            "exposed as one of those tools; pass your input as the tool "
            "arguments."
        )
        base = self.render_query_without_system(prompt_spec, constraints=constraints)
        return f"{preamble}\n\n{base}"

    # --- Container launch ---

    def default_args(self, scratch: Path) -> list[str]:
        # Wrapper script is baked into the image at /opt/nsl/run.py; it
        # loads /work/agent.yaml + /work/mcp_config.json + /work/query.txt
        # and calls LLMAgent(config=..., mcp_config=...).run(query).
        return ["python", "/opt/nsl/run.py"]

    def env(self, scratch: Path) -> dict[str, str]:
        # Don't export TOOL_CALL_TIMEOUT here. ms-agent's config loader
        # (ms_agent/config/config.py:173) merges env vars into the agent
        # config case-insensitively, replacing the int tool_call_timeout in
        # agent.yaml with the env var's *string* value. That string then
        # reaches asyncio.wait_for(timeout=...) which compares with `<=`,
        # crashing every tool call. agent.yaml is the source of truth.
        return {}

    def supports_native_workflow(self, workflow: Workflow) -> bool:
        return True

    def tool_call_contract_probe(self) -> ToolCallContractProbe:
        name = "nsl---contract_probe"
        return ToolCallContractProbe(
            tools=[_contract_probe_tool(name)],
            messages=[
                {
                    "role": "system",
                    "content": "You are a deterministic tool-call contract validator.",
                },
                {
                    "role": "user",
                    "content": (
                        "You are a contract-validation bot. You MUST call exactly "
                        f"the provided tool named {name} and set message to \"ok\". "
                        "Do not answer in plain text."
                    ),
                },
            ],
            expected_tool_name=name,
            expected_argument_keys={"message"},
            expected_arguments={"message": "ok"},
            tool_choice="auto",
        )

    # --- Config rendering ---

    def render_config(
        self,
        *,
        scratch: Path,
        query: str,
        capabilities: list[Capability],
        inference_url: str,
        mcp_url: str,
        token: str,
        prompt_spec: TaskPromptSpec | None = None,
        workflow: Workflow | None = None,
        constraints: EpisodeConstraints | None = None,
    ) -> None:
        agent_yaml = self._agent_yaml(
            inference_url=inference_url,
            token=token,
            prompt_spec=prompt_spec,
        )
        query_to_write = query
        if workflow is not None and workflow.entry_step is not None:
            # MCP preamble + capability manifest must lead the user message;
            # putting the step prompt first ("Use run_python to scan...")
            # primes code-biased models for markdown JSON output instead of
            # <tool_call> emission.
            query_to_write = _join_nonempty([query, workflow.entry_step.prompt])
            self._render_workflow(
                scratch=scratch,
                workflow=workflow,
                prompt_spec=prompt_spec,
                capabilities=capabilities,
                inference_url=inference_url,
                mcp_url=mcp_url,
                token=token,
                constraints=constraints,
            )
        (scratch / "agent.yaml").write_text(yaml.safe_dump(agent_yaml))
        (scratch / "query.txt").write_text(query_to_write)
        mcp_config = {
            "mcpServers": {
                "nsl": {
                    "type": "streamable_http",
                    "url": mcp_url,
                    "headers": {"Authorization": f"Bearer {token}"},
                },
            },
        }
        (scratch / "mcp_config.json").write_text(json.dumps(mcp_config))

    def _agent_yaml(
        self,
        *,
        inference_url: str,
        token: str,
        prompt_spec: TaskPromptSpec | None,
        query: str | None = None,
        workflow_step: str | None = None,
    ) -> dict[str, Any]:
        llm_config: dict[str, Any] = {
            "service": "openai",
            "model": self.config.model,
            "openai_base_url": inference_url,
            "openai_api_key": token,
        }
        if workflow_step is not None:
            llm_config["default_headers"] = {"X-NSL-Workflow-Step": workflow_step}
        agent_yaml: dict[str, Any] = {
            "llm": llm_config,
            "max_chat_round": self.config.max_chat_round,
            "tool_call_timeout": self.config.tool_call_timeout,
            "output_dir": self.config.output_dir,
        }
        if self.config.extra_yaml:
            agent_yaml = _merge_dicts(agent_yaml, self.config.extra_yaml)
        if prompt_spec is not None or query is not None:
            prompt = dict(agent_yaml.get("prompt") or {})
            if prompt_spec is not None:
                prompt["system"] = prompt_spec.system_instruction
            if query is not None:
                prompt["query"] = query
            agent_yaml["prompt"] = prompt
        return agent_yaml

    def _render_workflow(
        self,
        *,
        scratch: Path,
        workflow: Workflow,
        prompt_spec: TaskPromptSpec | None,
        capabilities: list[Capability],
        inference_url: str,
        mcp_url: str,
        token: str,
        constraints: EpisodeConstraints | None = None,
    ) -> None:
        filenames = {step.name: _safe_config_filename(step.name) for step in workflow.steps}
        if len(set(filenames.values())) != len(filenames):
            raise ValueError("workflow step names collide after filename sanitization")

        workflow_yaml: dict[str, Any] = {}
        for step in workflow.steps:
            entry: dict[str, Any] = {
                "agent_config": filenames[step.name],
                "context_mode": step.context_mode,
            }
            if step.next_steps:
                entry["next"] = list(step.next_steps)
            if step.on_error is not None:
                entry["on_error"] = step.on_error
            workflow_yaml[step.name] = entry

            step_constraints = project_step_constraints(
                constraints,
                step.name,
                has_capabilities=(
                    bool(capabilities) if step.inherit_all_capabilities else bool(step.capabilities)
                ),
            )
            step_query = _join_nonempty(
                [step.prompt, self.render_constraints_block(step_constraints)]
            )
            step_yaml = self._agent_yaml(
                inference_url=inference_url,
                token=token,
                prompt_spec=prompt_spec,
                query=step_query,
                workflow_step=step.name,
            )
            callbacks = step_yaml.get("callbacks") or []
            if isinstance(callbacks, str):
                callbacks = [callbacks]
            else:
                callbacks = list(callbacks)
            if _QUERY_INJECTION_CALLBACK not in callbacks:
                callbacks.append(_QUERY_INJECTION_CALLBACK)
            step_yaml["callbacks"] = callbacks
            step_yaml["tools"] = _merge_dicts(
                dict(step_yaml.get("tools") or {}),
                {
                    "nsl": _mcp_tool_config(
                        step=step,
                        capabilities=capabilities,
                        mcp_url=mcp_url,
                        token=token,
                    )
                },
            )
            (scratch / filenames[step.name]).write_text(yaml.safe_dump(step_yaml))

        (scratch / "workflow.yaml").write_text(yaml.safe_dump(workflow_yaml))

    # --- Transcript recovery ---

    def read_transcript(self, scratch: Path) -> dict[str, Any] | None:
        # ms-agent writes to <output_dir>/.memory/<tag>.json (hidden dir —
        # verified against ms_agent/utils/utils.py:save_history).
        output_dir = PurePosixPath(posixpath.normpath(self.config.output_dir))
        if output_dir.is_absolute():
            try:
                output_dir = output_dir.relative_to("/work")
            except ValueError:
                return None
        if ".." in output_dir.parts:
            return None
        memory_dir = scratch.joinpath(*output_dir.parts) / ".memory"
        path = memory_dir / f"{self.config.transcript_tag}.json"
        if not path.exists():
            workflow_transcript = self._read_workflow_transcript(scratch, memory_dir)
            if workflow_transcript is not None:
                return workflow_transcript
            if (scratch / "workflow.yaml").exists():
                raise MissingTranscriptError(
                    "ms-agent workflow transcript missing: expected per-step "
                    f"JSON files under {memory_dir}"
                )
            candidates = (
                sorted(memory_dir.glob("*.json")) if memory_dir.exists() else []
            )
            if len(candidates) != 1:
                return None
            path = candidates[0]
        return _normalise_transcript_payload(json.loads(path.read_text()))

    def _read_workflow_transcript(
        self,
        scratch: Path,
        memory_dir: Path,
    ) -> dict[str, Any] | None:
        order = _workflow_order(scratch / "workflow.yaml")
        if not order:
            return None
        messages: list[dict[str, Any]] = []
        by_step: dict[str, dict[str, Any]] = {}
        last_workflow_step: str | None = None
        for step_name in order:
            path = memory_dir / f"{step_name}.json"
            if not path.exists():
                continue
            payload = _normalise_transcript_payload(json.loads(path.read_text()))
            if payload is None:
                continue
            by_step[step_name] = payload
            last_workflow_step = step_name
            messages.extend(payload.get("messages") or [])
        if not by_step:
            return None
        transcript = {"messages": messages, "workflow": by_step}
        if last_workflow_step is not None:
            transcript["last_workflow_step"] = last_workflow_step
        return transcript

    # --- Artifact reconstruction ---

    def to_artifacts(
        self,
        *,
        transcript: dict[str, Any] | None,
        capability_pairs: list[tuple[CapabilityInvocation, CapabilityResult]],
    ) -> EpisodeArtifacts:
        invocations = [pair[0] for pair in capability_pairs]
        results = [pair[1] for pair in capability_pairs]
        final = _last_content_bearing_assistant(transcript) if transcript else None
        return EpisodeArtifacts(
            capability_invocations=invocations,
            capability_results=results,
            final_response=final,
        )

    def count_llm_turns(self, transcript: dict[str, Any] | None) -> int:
        if not transcript:
            return 0
        return _assistant_step_count(transcript)


__all__ = ["MsAgentProfile", "MsAgentProfileConfig"]


def _join_nonempty(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part.strip())


def _contract_probe_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Validate that the backend emits structured tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Set this to ok.",
                    }
                },
                "required": ["message"],
            },
        },
    }


def _mcp_tool_config(
    *,
    step: WorkflowStep,
    capabilities: list[Capability],
    mcp_url: str,
    token: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "type": "streamable_http",
        "url": mcp_url,
        "headers": {
            "Authorization": f"Bearer {token}",
            "X-NSL-Workflow-Step": step.name,
        },
    }
    if not step.inherit_all_capabilities:
        if step.capabilities:
            config["include"] = list(step.capabilities)
        else:
            config["exclude"] = [cap.name for cap in capabilities]
    return config


def _workflow_order(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        return []
    incoming_names = {
        next_name
        for value in payload.values()
        if isinstance(value, dict)
        for next_name in value.get("next", []) or []
    }
    incoming_names.update(
        value["on_error"]
        for value in payload.values()
        if isinstance(value, dict) and value.get("on_error") is not None
    )
    roots = [name for name in payload if name not in incoming_names]
    if len(roots) != 1:
        return []
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        order.append(name)
        step = payload.get(name) or {}
        if not isinstance(step, dict):
            return
        for next_name in step.get("next", []) or []:
            if next_name in payload:
                visit(next_name)
        error_name = step.get("on_error")
        if isinstance(error_name, str) and error_name in payload:
            visit(error_name)

    visit(roots[0])
    return order
