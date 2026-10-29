from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUN_PY = ROOT / "docker" / "ms-agent" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("nsl_ms_agent_run", RUN_PY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topological_chain_returns_next_chain_from_single_root() -> None:
    run = _load_run_module()
    spec = {
        "plan": {"agent_config": "plan.yaml", "next": ["act"]},
        "act": {"agent_config": "act.yaml"},
    }

    run._validate_spec(spec)

    assert run._topological_chain(spec) == ["plan", "act"]


def test_topological_chain_counts_on_error_target_as_non_root() -> None:
    run = _load_run_module()
    spec = {
        "try": {"agent_config": "try.yaml", "on_error": "recover"},
        "recover": {"agent_config": "recover.yaml"},
    }

    run._validate_spec(spec)

    assert run._topological_chain(spec) == ["try"]


def test_validate_spec_rejects_missing_next_step() -> None:
    run = _load_run_module()
    spec = {"plan": {"agent_config": "plan.yaml", "next": ["missing"]}}

    with pytest.raises(RuntimeError, match="next='missing' not found"):
        run._validate_spec(spec)


def test_validate_spec_rejects_missing_on_error_step() -> None:
    run = _load_run_module()
    spec = {"plan": {"agent_config": "plan.yaml", "on_error": "missing"}}

    with pytest.raises(RuntimeError, match="on_error='missing' not found"):
        run._validate_spec(spec)


def test_topological_chain_rejects_multiple_roots() -> None:
    run = _load_run_module()
    spec = {
        "plan": {"agent_config": "plan.yaml"},
        "act": {"agent_config": "act.yaml"},
    }

    run._validate_spec(spec)

    with pytest.raises(RuntimeError, match="exactly one root"):
        run._topological_chain(spec)


def test_topological_chain_rejects_cycles() -> None:
    run = _load_run_module()
    spec = {
        "root": {"agent_config": "root.yaml", "next": ["a"]},
        "a": {"agent_config": "a.yaml", "next": ["b"]},
        "b": {"agent_config": "b.yaml", "next": ["a"]},
    }

    run._validate_spec(spec)
    with pytest.raises(RuntimeError, match="cycle"):
        run._topological_chain(spec)


@pytest.mark.parametrize(
    "spec",
    [
        {
            "root": {"agent_config": "root.yaml"},
            "a": {"agent_config": "a.yaml", "on_error": "b"},
            "b": {"agent_config": "b.yaml", "on_error": "a"},
        },
        {
            "root": {"agent_config": "root.yaml", "next": ["act"]},
            "act": {"agent_config": "act.yaml", "on_error": "recover"},
            "recover": {"agent_config": "recover.yaml", "next": ["finish"]},
            "finish": {"agent_config": "finish.yaml", "on_error": "act"},
        },
    ],
    ids=["disconnected-error-cycle", "reachable-mixed-cycle"],
)
def test_topological_chain_rejects_all_edge_cycles(
    spec: dict[str, dict[str, object]],
) -> None:
    run = _load_run_module()
    run._validate_spec(spec)

    with pytest.raises(RuntimeError, match="cycle"):
        run._topological_chain(spec)


def test_topological_chain_rejects_unreachable_step() -> None:
    run = _load_run_module()
    spec = {
        "root": {"agent_config": "root.yaml", "next": ["act"]},
        "act": {"agent_config": "act.yaml"},
        "orphan": {"agent_config": "orphan.yaml", "next": ["act"]},
    }
    run._validate_spec(spec)

    with pytest.raises(RuntimeError, match="exactly one root|unreachable"):
        run._topological_chain(spec)


def test_recovered_workflow_returns_success_and_emits_recovery_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _load_run_module()
    (tmp_path / "workflow.yaml").write_text(
        "try:\n  agent_config: try.yaml\n  on_error: recover\n"
        "recover:\n  agent_config: recover.yaml\n"
    )
    (tmp_path / "try.yaml").write_text("prompt:\n  query: try\n")
    (tmp_path / "recover.yaml").write_text("prompt:\n  query: recover\n")

    class FakeAgent:
        def __init__(self, *, tag: str, **_kwargs) -> None:
            self.tag = tag

        async def run(self, _step_input):
            if self.tag == "try":
                raise RuntimeError("expected failure")
            return [{"role": "assistant", "content": "recovered"}]

    monkeypatch.setattr(run, "_load_agent_class", lambda: FakeAgent)

    status = asyncio.run(run._run_workflow(tmp_path, {}, "entry query"))

    assert status == run.EXIT_OK
    event_lines = [
        line.removeprefix("[nsl-event] ")
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("[nsl-event] ")
    ]
    assert any(
        event["event"] == "workflow_finished" and event["recovery_fired"] is True
        for event in map(json.loads, event_lines)
    )


def test_workflow_agents_use_per_step_mcp_headers_and_includes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _load_run_module()
    (tmp_path / "workflow.yaml").write_text(
        "explore:\n  agent_config: explore.yaml\n  next: [code]\n"
        "code:\n  agent_config: code.yaml\n"
    )
    (tmp_path / "explore.yaml").write_text(
        "prompt:\n  query: explore\n"
        "tools:\n  nsl:\n    type: streamable_http\n    url: http://mcp\n"
        "    headers:\n      Authorization: Bearer token\n"
        "      X-NSL-Workflow-Step: explore\n"
        "    include: [inspect_evidence, record_evidence_plan]\n"
    )
    (tmp_path / "code.yaml").write_text(
        "prompt:\n  query: code\n"
        "tools:\n  nsl:\n    type: streamable_http\n    url: http://mcp\n"
        "    headers:\n      Authorization: Bearer token\n"
        "      X-NSL-Workflow-Step: code\n"
        "    include: [inspect_evidence, compile_feature, finalize_feature, abandon_feature]\n"
    )
    effective_configs = []

    class FakeAgent:
        def __init__(self, *, config, mcp_config, **_kwargs) -> None:
            effective_configs.append((config, mcp_config))

        async def run(self, _step_input):
            return [{"role": "assistant", "content": "ok"}]

    monkeypatch.setattr(run, "_load_agent_class", lambda: FakeAgent)

    status = asyncio.run(
        run._run_workflow(
            tmp_path,
            {
                "mcpServers": {
                    "nsl": {
                        "url": "http://mcp",
                        "headers": {"Authorization": "Bearer token"},
                    }
                }
            },
            "entry query",
        )
    )

    assert status == run.EXIT_OK
    assert [entry[0].tools.nsl.headers["X-NSL-Workflow-Step"] for entry in effective_configs] == [
        "explore",
        "code",
    ]
    assert [list(entry[0].tools.nsl.include) for entry in effective_configs] == [
        ["inspect_evidence", "record_evidence_plan"],
        [
            "inspect_evidence",
            "compile_feature",
            "finalize_feature",
            "abandon_feature",
        ],
    ]
    assert [entry[1] for entry in effective_configs] == [{}, {}]
