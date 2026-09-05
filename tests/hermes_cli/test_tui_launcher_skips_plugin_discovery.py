
"""Regression test: the TUI launcher must not spend time on plugin discovery.

`hermes --tui` just spawns a Node process; the spawned tui_gateway backend
performs its own plugin discovery. Running discover_plugins() in the
launcher added ~0.5s to every `hermes --tui` startup for work the backend
then redoes. Plain chat must still discover plugins.
"""

from __future__ import annotations

from argparse import Namespace
import sys
import types

from hermes_cli import main as main_mod


def _install_discover_spy(monkeypatch):
    calls = []

    def _discover():
        calls.append("discover")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(
            discover_plugins=_discover,
            # main.py now kicks discovery off in a background thread; both
            # entry points count as "discovery work happened in the launcher".
            start_background_plugin_discovery=_discover,
        ),
    )
    return calls


def _args(**overrides):
    base = {
        "accept_hooks": False,
        "yolo": False,
        "safe_mode": False,
        "command": None,
        "query": None,
        "image": None,
    }
    base.update(overrides)
    return Namespace(**base)


def test_plugin_discovery_skipped_for_tui_launch(monkeypatch):
    calls = _install_discover_spy(monkeypatch)
    main_mod._prepare_agent_startup(_args(tui=True))
    assert calls == [], (
        "Plugin discovery must not run in the TUI launcher: the spawned "
        "tui_gateway backend discovers plugins itself."
    )


def test_plugin_discovery_runs_for_plain_chat(monkeypatch):
    calls = _install_discover_spy(monkeypatch)
    main_mod._prepare_agent_startup(_args(tui=False, command="chat"))
    assert calls == ["discover"]


def test_no_tools_startup_skips_plugins_mcp_shell_hooks_and_webhooks(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(start_background_plugin_discovery=lambda: calls.append("plugins")),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        types.SimpleNamespace(
            set_mcp_server_filter=lambda *_args: calls.append("mcp-filter"),
            start_background_mcp_discovery=lambda **_kwargs: calls.append("mcp"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(load_config=lambda: {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=lambda *_args, **_kwargs: calls.append("shell")),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.outbound_webhooks",
        types.SimpleNamespace(register_from_config=lambda *_args, **_kwargs: calls.append("webhook")),
    )
    monkeypatch.setattr(main_mod, "_is_tui_chat_launch", lambda _args: False)

    main_mod._prepare_agent_startup(_args(command="chat", no_tools=True))

    assert calls == []
