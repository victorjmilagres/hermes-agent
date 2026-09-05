from types import SimpleNamespace

import pytest

from hermes_cli._parser import build_top_level_parser


def test_oneshot_no_tools_flags_parse_for_isolated_service():
    parser, _, _ = build_top_level_parser()
    args = parser.parse_args([
        "--no-tools",
        "--no-context-files",
        "--no-memory",
        "--no-background-review",
        "--no-fallbacks",
        "--no-session-persistence",
        "-z",
        "hello",
    ])

    assert args.no_tools is True
    assert args.no_context_files is True
    assert args.no_memory is True
    assert args.no_background_review is True
    assert args.no_fallbacks is True
    assert args.no_session_persistence is True


def test_toolsets_none_is_exclusive_and_empty_string_stays_legacy():
    from hermes_cli.oneshot import _resolve_zero_tools

    assert _resolve_zero_tools(False, "none") == (True, None)
    assert _resolve_zero_tools(True, None) == (True, None)
    assert _resolve_zero_tools(False, "") == (False, None)
    assert _resolve_zero_tools(False, ["none"]) == (True, None)
    assert _resolve_zero_tools(False, ("none",)) == (True, None)
    assert _resolve_zero_tools(False, ["NONE"]) == (True, None)
    enabled, error = _resolve_zero_tools(False, "none,file")
    assert enabled is False
    assert "exclusive" in error


@pytest.mark.parametrize("toolsets", ["NONE", ["NONE"], ("none",)])
def test_classic_uses_shared_case_insensitive_none_normalization(monkeypatch, toolsets):
    import cli as cli_mod

    captured = {}
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kwargs: captured.update(kwargs) or SimpleNamespace())
    cli_mod._build_cli_from_args(
        "m", toolsets, None, None, None, None, None, None, False, False,
        None, False, False, False, None,
    )

    assert captured["no_tools"] is True
    assert captured["toolsets"] == []


def test_parsed_toolsets_none_is_normalized_before_startup_discovery():
    from hermes_cli import main as main_mod

    parser, subparsers = main_mod._build_cli_parser()
    args = main_mod._parse_cli_args(parser, subparsers, ["--toolsets", "NONE", "-z", "hello"])
    main_mod._normalize_zero_tools_args(args)

    assert args.no_tools is True
    assert args.toolsets is None
    assert args.no_context_files is True
    assert args.no_session_persistence is True


def test_classic_zero_tools_diagnostic_names_chat_surface(capsys):
    from hermes_cli.main import cmd_chat

    args = SimpleNamespace(no_tools=True, toolsets=["file"], command="chat")
    with pytest.raises(SystemExit) as exc:
        cmd_chat(args)

    assert exc.value.code == 2
    assert capsys.readouterr().err.startswith("hermes chat:")


def test_oneshot_failure_text_goes_to_stderr_not_stdout(monkeypatch, capsys):
    from hermes_cli import oneshot

    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *args, **kwargs: (
            "Model generated invalid tool call: terminal",
            {
                "final_response": "Model generated invalid tool call: terminal",
                "completed": False,
                "failed": True,
                "error": "Zero-tools invariant violated: model emitted a tool call",
            },
        ),
    )

    rc = oneshot.run_oneshot("hello", no_tools=True)
    captured = capsys.readouterr()

    assert rc != 0
    assert captured.out == ""
    assert "tool call" in captured.err.lower()


def test_oneshot_no_tools_applies_fail_closed_umbrella_before_agent_build(monkeypatch):
    from hermes_cli import oneshot

    captured = {}
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *args, **kwargs: captured.update(kwargs) or ("ok", {"completed": True}),
    )

    assert oneshot.run_oneshot("hello", no_tools=True) == 0
    assert captured["no_tools"] is True
    assert captured["skip_context_files"] is True
    assert captured["skip_memory"] is True
    assert captured["skip_background_review"] is True
    assert captured["disable_fallbacks"] is True
    assert captured["disable_session_persistence"] is True


def test_oneshot_partial_response_is_diagnostic(monkeypatch, capsys):
    from hermes_cli import oneshot

    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *args, **kwargs: (
            "unfinished answer",
            {"final_response": "unfinished answer", "completed": False, "partial": True},
        ),
    )

    rc = oneshot.run_oneshot("hello", no_tools=True)
    captured = capsys.readouterr()

    assert rc != 0
    assert captured.out == ""
    assert "unfinished answer" in captured.err
