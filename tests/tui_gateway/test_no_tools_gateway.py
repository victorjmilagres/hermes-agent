import types
import os
from pathlib import Path
import subprocess
import sys


def test_empty_configured_tui_toolsets_remain_explicitly_empty(monkeypatch):
    from tui_gateway import server

    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    monkeypatch.delenv("HERMES_TUI_NO_TOOLS", raising=False)
    monkeypatch.setattr("agent.coding_context.coding_selection", lambda **_kwargs: None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_args, **_kwargs: set())

    assert server._load_enabled_toolsets("tui") == []


def test_tui_gateway_propagates_no_tools_to_agent_boundary(monkeypatch):
    from tui_gateway import server
    import run_agent

    captured = {}
    monkeypatch.setenv("HERMES_TUI_NO_TOOLS", "1")
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_startup_system_prompt", lambda *_args: "")
    monkeypatch.setattr(
        server,
        "_resolve_agent_model_runtime",
        lambda *_args: ("test-model", {"provider": "test", "base_url": "u", "api_key": "k", "api_mode": "chat_completions"}),
    )
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})
    monkeypatch.setattr(server, "_get_db", lambda: object())
    monkeypatch.setattr(server.importlib, "import_module", lambda _name: types.SimpleNamespace(wait_for_mcp_discovery=lambda: None))
    monkeypatch.setattr(run_agent, "AIAgent", lambda **kwargs: captured.update(kwargs) or types.SimpleNamespace())

    server._make_agent("sid", "key")

    assert captured["no_tools"] is True
    assert captured["enabled_toolsets"] == []
    assert captured["session_db"] is None


def test_tui_background_agent_preserves_explicit_empty_toolsets(monkeypatch):
    from tui_gateway import server

    parent = types.SimpleNamespace(
        enabled_toolsets=[],
        _fallback_chain=[],
        no_tools=True,
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_args: ["terminal"])
    monkeypatch.setattr(server, "_get_db", lambda: object())

    kwargs = server._background_agent_kwargs(parent, "task")

    assert kwargs["enabled_toolsets"] == []
    assert kwargs["no_tools"] is True
    assert kwargs["session_db"] is None


def test_stateless_session_create_does_not_import_terminal_or_schedule_persistence(monkeypatch):
    import sys
    from tui_gateway import server

    monkeypatch.setenv("HERMES_TUI_NO_TOOLS", "1")
    monkeypatch.delitem(sys.modules, "tools.terminal_tool", raising=False)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: (_ for _ in ()).throw(
        AssertionError("session cap persistence scheduled")))
    response = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "session.create", "params": {}})

    assert response["result"]["session_id"]
    assert "tools.terminal_tool" not in sys.modules


def test_stateless_prompt_submit_skips_lease_and_session_row(monkeypatch):
    from tui_gateway import server

    monkeypatch.setenv("HERMES_TUI_NO_SESSION_PERSISTENCE", "1")
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    created = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "session.create", "params": {}})
    sid = created["result"]["session_id"]
    session = server._sessions[sid]
    session["agent"] = types.SimpleNamespace()
    session["agent_ready"].set()
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: (_ for _ in ()).throw(
        AssertionError("lease acquired")))
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *_args: (_ for _ in ()).throw(
        AssertionError("session row persisted")))
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *_args, **_kwargs: None)

    response = server.dispatch({
        "jsonrpc": "2.0", "id": 2, "method": "prompt.submit",
        "params": {"session_id": sid, "text": "hello"},
    })

    assert response["result"]["status"] == "streaming"


def test_fresh_stateless_gateway_import_is_read_only(tmp_path):
    hermes_home = tmp_path / "fresh"
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(hermes_home),
        "HERMES_TUI_NO_TOOLS": "1",
    }

    subprocess.run(
        [sys.executable, "-B", "-c", "import tui_gateway.server"],
        check=True, cwd=Path(__file__).parents[2], env=env, capture_output=True, text=True,
    )

    assert not hermes_home.exists()


def test_stateless_slash_exec_does_not_spawn_worker(monkeypatch):
    from tui_gateway import server

    monkeypatch.setenv("HERMES_TUI_NO_TOOLS", "1")
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    created = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "session.create", "params": {}})
    sid = created["result"]["session_id"]
    monkeypatch.setattr(
        server, "_SlashWorker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("slash worker spawned")),
    )

    response = server.handle_request({
        "jsonrpc": "2.0", "id": 2, "method": "slash.exec",
        "params": {"session_id": sid, "command": "/tools"},
    })

    assert response["error"]["code"] == 4018