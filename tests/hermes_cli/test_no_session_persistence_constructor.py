from unittest.mock import patch
import os
from pathlib import Path
import subprocess
import sys


def test_classic_cli_disables_state_before_runtime_initialization(monkeypatch, tmp_path):
    import cli

    calls = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli.HermesCLI, "_init_display_options", lambda self, *_args: None)

    def init_routing(self, *_args, **_kwargs):
        self.no_session_persistence = True

    monkeypatch.setattr(cli.HermesCLI, "_init_model_routing", init_routing)
    monkeypatch.setattr(cli.HermesCLI, "_write_terminal_breadcrumb", lambda self: calls.append("breadcrumb"))
    monkeypatch.setattr(cli, "_run_state_db_auto_maintenance", lambda *_args: calls.append("state-maintenance"))
    monkeypatch.setattr(cli, "_run_checkpoint_auto_maintenance", lambda: calls.append("checkpoint-maintenance"))

    with patch("hermes_state.SessionDB", side_effect=lambda: calls.append("SessionDB")):
        shell = cli.HermesCLI(no_session_persistence=True)

    assert shell._session_db is None
    assert calls == []
    assert not (tmp_path / "state.db").exists()
    assert not (tmp_path / "sessions" / "active").exists()
    assert not (tmp_path / "profiles").exists()

    monkeypatch.setattr(
        "hermes_cli.active_sessions.try_acquire_active_session",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("lock opened")),
    )
    assert shell._claim_active_session() is True


def test_fresh_no_tools_import_and_construction_are_read_only(tmp_path):
    hermes_home = tmp_path / "fresh"
    code = """
from unittest.mock import patch
import run_agent
with patch('agent.process_bootstrap.OpenAI'):
    run_agent.AIAgent(api_key='k', base_url='https://example.test/v1', model='m', no_tools=True, quiet_mode=True)
"""
    env = {**os.environ, "HERMES_HOME": str(hermes_home), "HOME": str(tmp_path / "home")}

    subprocess.run(
        [sys.executable, "-B", "-c", code], check=True, cwd=Path(__file__).parents[2], env=env,
        capture_output=True, text=True,
    )

    assert not hermes_home.exists()