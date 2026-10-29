from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Message:
    role: str
    content: str


def _load_callback(monkeypatch):
    ms_agent_pkg = types.ModuleType("ms_agent")
    callbacks_pkg = types.ModuleType("ms_agent.callbacks")
    callbacks_base = types.ModuleType("ms_agent.callbacks.base")
    llm_pkg = types.ModuleType("ms_agent.llm")
    llm_utils = types.ModuleType("ms_agent.llm.utils")

    class Callback:
        def __init__(self, config):
            self.config = config

    callbacks_base.Callback = Callback
    llm_utils.Message = Message

    monkeypatch.setitem(sys.modules, "ms_agent", ms_agent_pkg)
    monkeypatch.setitem(sys.modules, "ms_agent.callbacks", callbacks_pkg)
    monkeypatch.setitem(sys.modules, "ms_agent.callbacks.base", callbacks_base)
    monkeypatch.setitem(sys.modules, "ms_agent.llm", llm_pkg)
    monkeypatch.setitem(sys.modules, "ms_agent.llm.utils", llm_utils)

    path = ROOT / "docker" / "ms-agent" / "inject_query_callback.py"
    spec = importlib.util.spec_from_file_location("inject_query_callback", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InjectStepQuery


def test_inject_appends_query_when_missing(monkeypatch):
    inject_step_query = _load_callback(monkeypatch)
    callback = inject_step_query(OmegaConf.create({"prompt": {"query": "do plan thing"}}))
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="explore"),
        Message(role="assistant", content="ok"),
    ]

    asyncio.run(callback.on_task_begin(runtime=None, messages=messages))

    assert messages[-1] == Message(role="user", content="do plan thing")


def test_inject_skips_when_query_already_last(monkeypatch):
    inject_step_query = _load_callback(monkeypatch)
    callback = inject_step_query(OmegaConf.create({"prompt": {"query": "do plan thing"}}))
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="do plan thing"),
    ]

    asyncio.run(callback.on_task_begin(runtime=None, messages=messages))

    assert messages == [
        Message(role="system", content="sys"),
        Message(role="user", content="do plan thing"),
    ]


def test_ms_agent_image_copies_callback_module():
    dockerfile = (ROOT / "docker" / "ms-agent" / "Dockerfile").read_text()

    assert "COPY inject_query_callback.py /opt/nsl/inject_query_callback.py" in dockerfile


def test_custom_workflow_runner_uses_per_step_config():
    run_py = (ROOT / "docker" / "ms-agent" / "run.py").read_text()

    assert "WorkflowLoader" not in run_py
    assert "step_cfg = OmegaConf.load(scratch / entry[\"agent_config\"])" in run_py
    assert "step_cfg.trust_remote_code = True" in run_py
    assert "step_cfg.local_dir = str(scratch)" in run_py
    assert "trust_remote_code=True" in run_py
