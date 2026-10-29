"""``MsAgentProfile`` renders ms-agent config + query, reads back transcript.

The profile is the per-harness adapter that ``ContainerHarness`` drives.
Phase 2 pins:

- ``agent.yaml`` carries ``llm.service: openai`` + ``openai_base_url`` +
  ``openai_api_key`` — the CLI-independent path that avoids top-level
  ``mcpServers`` (which ms-agent's CLI loader ignores).
- ``mcp_config.json`` uses ``streamable_http`` type with bearer header.
- ``read_transcript`` looks at ``<output_dir>/.memory/<tag>.json`` (hidden
  directory — ms-agent's actual path).
- ``to_artifacts`` reconstructs ``EpisodeArtifacts`` from the
  ``capability_pairs`` recorder query.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.harness.profiles.ms_agent import MsAgentProfile, MsAgentProfileConfig
from src.task.types import (
    BudgetConstraints,
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeConstraints,
    StepConstraints,
    SuccessConstraints,
    TaskPromptSpec,
    Workflow,
    WorkflowStep,
)


def _profile(**overrides) -> MsAgentProfile:
    cfg = MsAgentProfileConfig(
        model=overrides.pop("model", "claude-sonnet-4-6"),
        **overrides,
    )
    return MsAgentProfile(cfg)


def test_render_writes_single_agent_contract(tmp_path: Path) -> None:
    profile = _profile()
    prompt_spec = TaskPromptSpec(system_instruction="You are an exploit finder.")
    profile.render_config(
        scratch=tmp_path,
        query="solve this",
        capabilities=[Capability(name="analyzer", description="read")],
        inference_url="http://172.17.0.1:9001/v1",
        mcp_url="http://172.17.0.1:9002/mcp",
        token="tok",
        prompt_spec=prompt_spec,
    )
    cfg = yaml.safe_load((tmp_path / "agent.yaml").read_text())
    # LLM section points at our shim, not top-level OpenAI.
    assert cfg["llm"]["service"] == "openai"
    assert cfg["llm"]["openai_base_url"] == "http://172.17.0.1:9001/v1"
    assert cfg["llm"]["openai_api_key"] == "tok"
    assert cfg["llm"]["model"] == "claude-sonnet-4-6"
    assert cfg["prompt"]["system"] == "You are an exploit finder."
    # Top-level mcpServers is intentionally absent — ms-agent's CLI loader
    # ignores it; MCP config is passed programmatically via mcp_config.json.
    assert "mcpServers" not in cfg
    mcp_config = json.loads((tmp_path / "mcp_config.json").read_text())
    server = mcp_config["mcpServers"]["nsl"]
    assert server["type"] == "streamable_http"
    assert server["url"] == "http://172.17.0.1:9002/mcp"
    assert server["headers"]["Authorization"] == "Bearer tok"
    assert (tmp_path / "query.txt").read_text() == "solve this"


def test_render_query_excludes_system_prompt_from_query_txt() -> None:
    """ms-agent receives system text through agent.yaml prompt.system."""
    profile = _profile()
    spec = TaskPromptSpec(
        system_instruction="You are an exploit finder.",
        environment_context="RPC: http://anvil:8545",
        capabilities=[
            Capability(
                name="analyzer",
                description="read state",
            ),
            Capability(
                name="exploiter",
                description="write Attack.sol",
            ),
        ],
    )
    query = profile.render_query(spec)
    assert query.lstrip().lower().startswith("you operate via mcp tools")
    assert "You are an exploit finder." not in query
    assert "RPC: http://anvil:8545" in query
    assert "analyzer" in query and "read state" in query
    assert "exploiter" in query and "write Attack.sol" in query
    assert "nsl" in query.lower() and "tool" in query.lower()
    assert "ends the episode without making progress" not in query


def test_render_query_includes_static_constraints_block() -> None:
    profile = _profile()
    spec = TaskPromptSpec(
        system_instruction="You are an exploit finder.",
        capabilities=[Capability(name="deploy_attack_sol", description="deploy")],
    )
    constraints = EpisodeConstraints(
        budgets=BudgetConstraints(
            max_task_tool_calls=30,
            max_task_tool_calls_by_name={"deploy_attack_sol": 8},
            max_llm_turns=40,
        ),
        success=SuccessConstraints(min_task_tool_calls_for_success=1),
    )

    query = profile.render_query(spec, constraints=constraints)

    assert "Task constraints:" in query
    assert "task tool calls: at most 30" in query
    assert "deploy_attack_sol: at most 8" in query
    assert "llm turns: advisory limit 40" in query


def test_render_workflow_entry_step_query_starts_with_mcp_preamble(
    tmp_path: Path,
) -> None:
    """Entry-step query.txt must lead with the MCP preamble + capability
    manifest, then the workflow step prompt.

    Putting ``entry_step.prompt`` before ``query`` places step prompts before
    the "CALL the appropriate tool" instruction, priming the model for
    markdown code-block output instead of <tool_call> emission.
    """
    profile = _profile()
    spec = TaskPromptSpec(
        system_instruction="You are a task agent.",
        capabilities=[
            Capability(name="run_python", description="Execute Python."),
        ],
    )
    workflow = Workflow(
        steps=(
            WorkflowStep(
                name="explore",
                prompt="Explore the filesystem and use run_python to scan paths.",
                is_entry=True,
                next_steps=("act",),
            ),
            WorkflowStep(name="act", prompt="Now act on findings."),
        )
    )

    full_query = profile.render_query(spec)
    profile.render_config(
        scratch=tmp_path,
        query=full_query,
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
    )

    query_text = (tmp_path / "query.txt").read_text()
    preamble_idx = query_text.index("CALL the appropriate tool")
    step_idx = query_text.index("Explore the filesystem")
    assert preamble_idx < step_idx, (
        "MCP preamble must precede entry-step prompt to avoid priming "
        "code-biased models for markdown output"
    )
    # Both pieces must still be present.
    assert "run_python" in query_text and "Execute Python." in query_text


def test_render_workflow_files_for_ms_agent(tmp_path: Path) -> None:
    profile = _profile()
    spec = TaskPromptSpec(
        system_instruction="System persona",
        environment_context="Episode context",
        capabilities=[
            Capability(name="alpha", description="first"),
            Capability(name="beta", description="second"),
        ],
    )
    from src.task.types import Workflow, WorkflowStep

    workflow = Workflow(
        steps=(
            WorkflowStep(name="plan", prompt="Plan", next_steps=("act",)),
            WorkflowStep(
                name="act",
                prompt="Act",
                inherit_all_capabilities=False,
                capabilities=("beta",),
            ),
        )
    )

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
    )

    workflow_yaml = yaml.safe_load((tmp_path / "workflow.yaml").read_text())
    assert workflow_yaml == {
        "plan": {
            "agent_config": "plan.yaml",
            "context_mode": "inherit",
            "next": ["act"],
        },
        "act": {"agent_config": "act.yaml", "context_mode": "inherit"},
    }
    plan_yaml = yaml.safe_load((tmp_path / "plan.yaml").read_text())
    act_yaml = yaml.safe_load((tmp_path / "act.yaml").read_text())
    assert plan_yaml["prompt"] == {"system": "System persona", "query": "Plan"}
    assert plan_yaml["llm"]["default_headers"] == {"X-NSL-Workflow-Step": "plan"}
    assert plan_yaml["callbacks"] == ["inject_query_callback.py"]
    assert plan_yaml["tools"]["nsl"]["url"] == "http://h:2/mcp"
    assert plan_yaml["tools"]["nsl"]["headers"]["X-NSL-Workflow-Step"] == "plan"
    assert "include" not in plan_yaml["tools"]["nsl"]
    assert act_yaml["prompt"] == {"system": "System persona", "query": "Act"}
    assert act_yaml["llm"]["default_headers"] == {"X-NSL-Workflow-Step": "act"}
    assert act_yaml["callbacks"] == ["inject_query_callback.py"]
    assert act_yaml["tools"]["nsl"]["headers"]["X-NSL-Workflow-Step"] == "act"
    assert act_yaml["tools"]["nsl"]["include"] == ["beta"]


def test_render_workflow_files_preserve_step_query_without_prompt_spec(
    tmp_path: Path,
) -> None:
    profile = _profile()
    workflow = Workflow(
        steps=(
            WorkflowStep(name="plan", prompt="Plan", next_steps=("act",)),
            WorkflowStep(name="act", prompt="Act"),
        )
    )

    profile.render_config(
        scratch=tmp_path,
        query="base query",
        capabilities=[Capability(name="alpha", description="first")],
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        workflow=workflow,
    )

    plan_yaml = yaml.safe_load((tmp_path / "plan.yaml").read_text())
    act_yaml = yaml.safe_load((tmp_path / "act.yaml").read_text())
    assert plan_yaml["prompt"] == {"query": "Plan"}
    assert act_yaml["prompt"] == {"query": "Act"}


def test_render_workflow_files_include_context_mode_and_on_error(
    tmp_path: Path,
) -> None:
    profile = _profile()
    spec = TaskPromptSpec(system_instruction="System persona")
    workflow = Workflow(
        steps=(
            WorkflowStep(name="explore", prompt="Explore", next_steps=("execute",)),
            WorkflowStep(
                name="execute",
                prompt="Execute",
                context_mode="isolated",
                next_steps=("submit",),
                on_error="submit",
            ),
            WorkflowStep(name="submit", prompt="Submit"),
        )
    )

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec),
        capabilities=[],
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
    )

    workflow_yaml = yaml.safe_load((tmp_path / "workflow.yaml").read_text())
    assert workflow_yaml == {
        "explore": {
            "agent_config": "explore.yaml",
            "context_mode": "inherit",
            "next": ["execute"],
        },
        "execute": {
            "agent_config": "execute.yaml",
            "context_mode": "isolated",
            "next": ["submit"],
            "on_error": "submit",
        },
        "submit": {"agent_config": "submit.yaml", "context_mode": "inherit"},
    }

    execute_yaml = yaml.safe_load((tmp_path / "execute.yaml").read_text())
    assert execute_yaml["prompt"]["query"] == "Execute"
    assert execute_yaml["callbacks"] == ["inject_query_callback.py"]


def test_render_workflow_files_include_effective_step_constraints(
    tmp_path: Path,
) -> None:
    profile = _profile()
    spec = TaskPromptSpec(
        system_instruction="System persona",
        environment_context="Episode context",
        capabilities=[Capability(name="run_python", description="run")],
    )
    workflow = Workflow(
        steps=(
            WorkflowStep(name="plan", prompt="Plan", next_steps=("act",)),
            WorkflowStep(name="act", prompt="Act"),
        )
    )
    constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=15, max_llm_turns=18),
        success=SuccessConstraints(min_task_tool_calls_for_success=1),
        step_overrides={
            "plan": StepConstraints(
                budgets=BudgetConstraints(max_task_tool_calls=0, max_llm_turns=4),
                success=SuccessConstraints(min_task_tool_calls_for_success=0),
            )
        },
    )

    profile.render_config(
        scratch=tmp_path,
        query=profile.render_query(spec, constraints=constraints),
        capabilities=list(spec.capabilities),
        inference_url="http://h:1/v1",
        mcp_url="http://h:2/mcp",
        token="tok",
        prompt_spec=spec,
        workflow=workflow,
        constraints=constraints,
    )

    plan_yaml = yaml.safe_load((tmp_path / "plan.yaml").read_text())
    act_yaml = yaml.safe_load((tmp_path / "act.yaml").read_text())
    assert "task tool calls: at most 0" in plan_yaml["prompt"]["query"]
    assert "success requires at least 0 task tool calls" in plan_yaml["prompt"]["query"]
    assert "llm turns: advisory limit 4" in plan_yaml["prompt"]["query"]
    assert "task tool calls: at most 15" in act_yaml["prompt"]["query"]


def test_read_transcript_uses_configured_output_dir(tmp_path: Path) -> None:
    profile = _profile(output_dir="/work/custom-output", transcript_tag="episode")
    memory_dir = tmp_path / "custom-output" / ".memory"
    memory_dir.mkdir(parents=True)
    payload = {"messages": [{"role": "assistant", "content": "custom"}]}
    (memory_dir / "episode.json").write_text(json.dumps(payload))

    loaded = profile.read_transcript(tmp_path)
    assert loaded == payload


def test_to_artifacts_reconstructs_from_capability_pairs(tmp_path: Path) -> None:
    profile = _profile()
    transcript = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "final answer here"},
        ],
    }
    pairs = [
        (
            CapabilityInvocation(name="analyzer", input={"addr": "0xabc"}),
            CapabilityResult(
                name="analyzer",
                output={"balance": 10},
                success=True,
                error=None,
            ),
        ),
    ]
    artifacts = profile.to_artifacts(transcript=transcript, capability_pairs=pairs)
    assert artifacts.capability_invocations[0].name == "analyzer"
    assert artifacts.capability_results[0].success is True
    assert artifacts.final_response == "final answer here"


def test_profile_runtime_command_and_environment(tmp_path: Path) -> None:
    """Profile drives the wrapper script, not the ms-agent CLI — ``--query-file``
    does not exist in ms-agent."""
    profile = _profile(tool_call_timeout=45)
    args = profile.default_args(tmp_path)
    assert args[0] == "python"
    # The wrapper script lives at /opt/nsl/run.py inside the image — calling
    # ms-agent via the Python API rather than its CLI.
    assert any("/opt/nsl/run.py" in a for a in args)
    env = profile.env(tmp_path)
    assert "TOOL_CALL_TIMEOUT" not in env
    # Case-insensitive guard — ms-agent's matcher is case-insensitive.
    assert all(k.lower() != "tool_call_timeout" for k in env)
