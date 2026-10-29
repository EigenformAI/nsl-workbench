"""Server startup helpers must pass pre-bound sockets into uvicorn.

These tests pin the regression that caused ``EADDRINUSE``: both helpers
pre-bound an ephemeral port but then called ``server.serve()`` without the
matching ``sockets=[sock]`` argument, so uvicorn attempted a second bind on
the same port.
"""

from __future__ import annotations

import socket
import sys
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

import src.harness.container as container_mod
import src.framework.capability_bridge as capability_bridge_mod
from src.harness.base import HarnessError


class _FakeSocket:
    def __init__(self, port: int = 43123) -> None:
        self._port = port
        self.bound_to: tuple[str, int] | None = None
        self.listen_backlog: int | None = None

    def bind(self, addr: tuple[str, int]) -> None:
        self.bound_to = addr

    def listen(self, backlog: int) -> None:
        self.listen_backlog = backlog

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", self._port)

    def close(self) -> None:
        return None


class _InlineThread:
    def __init__(self, *, target, name: str, daemon: bool) -> None:
        self._target = target
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        self._target()

    def join(self, timeout: float | None = None) -> None:
        return None


class _FakeConfig:
    def __init__(
        self,
        app,
        *,
        host: str,
        port: int,
        log_level: str,
        loop: str,
    ) -> None:
        self.app = app
        self.host = host
        self.port = port
        self.log_level = log_level
        self.loop = loop


class _FakeServer:
    def __init__(self, config: _FakeConfig) -> None:
        self.config = config
        self.should_exit = False
        self.serve_sockets = None
        self.startup = self._startup

    async def _startup(self, sockets=None) -> None:
        return None

    async def serve(self, sockets=None) -> None:
        self.serve_sockets = sockets
        await self.startup(sockets=sockets)


class _FailingServer(_FakeServer):
    async def serve(self, sockets=None) -> None:
        self.serve_sockets = sockets
        raise RuntimeError("startup failed")


def _install_fake_runtime(
    monkeypatch, module, *, port: int = 43123, server_cls=_FakeServer
):
    fake_socket = _FakeSocket(port=port)
    holder: dict[str, _FakeServer] = {}
    real_socket = socket.socket
    socket_calls = 0

    def _socket_factory(*_args, **_kwargs) -> Any:
        nonlocal socket_calls
        socket_calls += 1
        if socket_calls == 1:
            return fake_socket
        return real_socket(*_args, **_kwargs)

    def _thread_factory(*, target, name: str, daemon: bool) -> _InlineThread:
        return _InlineThread(target=target, name=name, daemon=daemon)

    def _server_factory(config: _FakeConfig) -> _FakeServer:
        server = server_cls(config)
        holder["server"] = server
        return server

    fake_uvicorn = SimpleNamespace(Config=_FakeConfig, Server=_server_factory)

    monkeypatch.setattr(socket, "socket", _socket_factory)
    monkeypatch.setattr(module.threading, "Thread", _thread_factory)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    return holder, fake_socket


def test_container_loopback_passes_bound_socket_to_uvicorn(monkeypatch) -> None:
    holder, fake_socket = _install_fake_runtime(monkeypatch, container_mod, port=43123)

    handle = container_mod._serve_host_shim(app=object())

    server = holder["server"]
    assert handle.port == 43123
    assert fake_socket.bound_to == ("0.0.0.0", 0)
    assert server.serve_sockets == [fake_socket]


def test_bridge_loopback_passes_bound_socket_to_uvicorn(monkeypatch) -> None:
    holder, fake_socket = _install_fake_runtime(
        monkeypatch, capability_bridge_mod, port=43124
    )
    ctx = SimpleNamespace(
        prompt_spec=SimpleNamespace(capabilities=[]),
        cancel_event=threading.Event(),
        recorder=SimpleNamespace(),
        task=SimpleNamespace(),
        containers=[],
        variation=None,
    )

    handle = capability_bridge_mod.CapabilityMcpBridge(
        cast(Any, ctx), token="secret"
    ).serve_http()

    server = holder["server"]
    assert handle.port == 43124
    assert fake_socket.bound_to == ("0.0.0.0", 0)
    assert server.serve_sockets == [fake_socket]


def test_bridge_loopback_raises_when_server_startup_fails(monkeypatch) -> None:
    _install_fake_runtime(
        monkeypatch,
        capability_bridge_mod,
        port=43125,
        server_cls=_FailingServer,
    )
    ctx = SimpleNamespace(
        prompt_spec=SimpleNamespace(capabilities=[]),
        cancel_event=threading.Event(),
        recorder=SimpleNamespace(),
        task=SimpleNamespace(),
        containers=[],
        variation=None,
    )

    with pytest.raises(HarnessError, match="MCP bridge server failed to start"):
        capability_bridge_mod.CapabilityMcpBridge(
            cast(Any, ctx), token="secret"
        ).serve_http()
