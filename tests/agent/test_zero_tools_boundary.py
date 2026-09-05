from types import SimpleNamespace
import inspect
from unittest.mock import MagicMock, patch

import pytest


def test_no_tools_initialization_skips_tool_discovery(monkeypatch):
    from agent.agent_init import _load_tools

    agent = SimpleNamespace(
        quiet_mode=True,
        save_trajectories=False,
        ephemeral_system_prompt=None,
        _use_prompt_caching=False,
    )

    def unexpected_discovery():
        raise AssertionError("zero-tools mode must not discover plugins")

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", unexpected_discovery)

    _load_tools(agent, enabled_toolsets=None, disabled_toolsets=None, no_tools=True)

    assert agent.no_tools is True
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._kanban_worker_guidance == ""


@pytest.mark.parametrize(
    "injected",
    [
        {"tools": [{"type": "function"}]},
        {"extra_body": {"tools": [{"type": "function"}]}},
        {"extra_body": {"tool_choice": "required"}},
        {"metadata": {"functions": [{"name": "terminal"}]}},
        {"extra_body": {"toolConfig": {"functionCallingConfig": {"mode": "ANY"}}}},
    ],
)
def test_zero_tools_final_provider_boundary_rejects_recursive_tool_fields(injected):
    from agent.turn_api_request import enforce_no_tools_request

    with pytest.raises(RuntimeError, match="Zero-tools invariant"):
        enforce_no_tools_request(SimpleNamespace(no_tools=True), injected)


def test_zero_tools_bypasses_execution_middleware_and_relay(monkeypatch):
    from agent.turn_api_call import perform_api_call

    provider_calls = []
    agent = SimpleNamespace(
        no_tools=True, api_mode="chat_completions", base_url="https://example.test/v1",
        provider="test", model="test-model", platform="cli", session_id="session",
        is_subagent=False, _fallback_index=0, _disable_streaming=True,
        _model_request_active=None, _pending_redirect_lock=None, _pending_redirect=None,
        _has_pending_redirect=lambda: False, _has_stream_consumers=lambda: False,
        _interruptible_api_call=lambda request: provider_calls.append(request) or "provider-response",
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("middleware ran")),
    )
    monkeypatch.setattr(
        "agent.relay_llm.execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("relay ran")),
    )

    verdict = perform_api_call(
        agent, api_kwargs={"model": "test-model", "messages": [{"role": "user", "content": "original"}]},
        _original_api_kwargs={}, _llm_middleware_trace=[], _moa_prepared_request=None,
        _retry=SimpleNamespace(), thinking_spinner=None, retry_count=0, api_call_count=0,
        api_request_id="request-1", effective_task_id="task-1", turn_id="turn-1", interrupted=False,
    )

    assert verdict.response == "provider-response"
    assert provider_calls[0]["messages"][0]["content"] == "original"


@pytest.mark.parametrize(
    ("api_mode", "provider", "streaming"),
    [
        ("chat_completions", "openai", True),
        ("anthropic_messages", "anthropic", False),
        ("bedrock_converse", "bedrock", False),
        ("chat_completions", "gemini", False),
        ("codex_responses", "openai-codex", False),
    ],
)
def test_zero_tools_reaches_each_supported_provider_boundary_without_tool_fields(
    monkeypatch, api_mode, provider, streaming,
):
    from agent.turn_api_call import perform_api_call
    from agent.turn_api_request import _provider_tool_field_path

    calls = []
    agent = SimpleNamespace(
        no_tools=True, api_mode=api_mode, provider=provider, model="model", platform="cli",
        base_url="https://example.test/v1", session_id="", is_subagent=False,
        _fallback_index=0, _disable_streaming=not streaming, _model_request_active=None,
        _pending_redirect_lock=None, _pending_redirect=None, _has_pending_redirect=lambda: False,
        _has_stream_consumers=lambda: streaming,
        _is_copilot_url=lambda: False, _is_codex_backend=lambda: False,
        _get_transport=lambda: SimpleNamespace(preflight_kwargs=lambda request, **_kwargs: request),
        _interruptible_api_call=lambda request: calls.append(request) or "response",
        _interruptible_streaming_api_call=lambda request, **_kwargs: calls.append(request) or "response",
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("middleware ran")),
    )

    verdict = perform_api_call(
        agent, api_kwargs={"model": "model", "messages": [{"role": "user", "content": "hello"}]},
        _original_api_kwargs={}, _llm_middleware_trace=[], _moa_prepared_request=None,
        _retry=SimpleNamespace(), thinking_spinner=None, retry_count=0, api_call_count=0,
        api_request_id="r", effective_task_id="t", turn_id="turn", interrupted=False,
    )

    assert verdict.response == "response"
    assert len(calls) == 1
    assert _provider_tool_field_path(calls[0]) is None


def test_gemini_native_translation_keeps_zero_tools_payload_tool_free():
    from agent.gemini_native_adapter import build_gemini_request
    from agent.turn_api_request import _provider_tool_field_path

    request = build_gemini_request(
        model="gemini-test", messages=[{"role": "user", "content": "hello"}],
    )

    assert _provider_tool_field_path(request) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_mode": "codex_app_server", "provider": "openai-codex"},
        {"api_mode": "chat_completions", "provider": "copilot-acp", "base_url": "acp://codex"},
    ],
)
def test_zero_tools_rejects_native_implicit_tool_providers_before_client_build(monkeypatch, kwargs):
    from run_agent import AIAgent

    monkeypatch.setattr(
        "agent.agent_init._build_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("client built")),
    )
    with pytest.raises(ValueError, match="implicit-tool"):
        AIAgent(model="model", no_tools=True, quiet_mode=True, **kwargs)


def test_zero_tools_boundary_rejects_model_tool_calls():
    from agent.turn_response_intake import reject_no_tools_response

    agent = SimpleNamespace(no_tools=True)
    assistant_message = SimpleNamespace(tool_calls=[object()])

    result = reject_no_tools_response(agent, assistant_message, [], api_call_count=1)

    assert result["failed"] is True
    assert result["completed"] is False
    assert "tool call" in result["error"].lower()


def test_zero_tools_forced_tool_call_fails_end_to_end_without_execution(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key", base_url="https://example.test/v1", model="test-model",
            no_tools=True, quiet_mode=True,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name="terminal", arguments='{"command":"id"}')],
    )
    agent._cached_system_prompt = "No tools are available."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent._execute_tool_calls = MagicMock(side_effect=AssertionError("tool execution reached"))

    result = agent.run_conversation("Run id")

    assert result["failed"] is True
    assert result["completed"] is False
    assert "tool call" in result["error"].lower()
    agent._execute_tool_calls.assert_not_called()
    sent = agent.client.chat.completions.create.call_args.kwargs
    assert "tools" not in sent


def test_zero_tools_prompt_states_no_tools_without_tool_guidance():
    from agent.system_prompt import _guidance_parts

    parts = _guidance_parts(SimpleNamespace(no_tools=True, valid_tool_names=set()))

    assert parts == ["No tools are available in this session. Answer using only the conversation context."]


def test_isolated_session_state_does_not_touch_disk_or_import_tool_stores(tmp_path, monkeypatch):
    from agent.agent_init import _init_session_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.agent_init._publish_session_id", lambda _session_id: None)
    agent = SimpleNamespace(max_iterations=1)

    _init_session_state(
        agent, None, None, None, None, None, False, 20, 500, 10,
        no_tools=True, disable_session_persistence=True,
    )

    assert agent._persist_disabled is True
    assert agent._session_db is None
    assert agent._checkpoint_mgr.enabled is False
    assert agent._todo_store.has_items() is False
    assert not (tmp_path / "state.db").exists()


def test_no_tools_is_fail_closed_umbrella_at_public_agent_boundary(tmp_path, monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    fallback = {"provider": "openai", "model": "fallback-model"}
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            no_tools=True,
            fallback_model=fallback,
            load_soul_identity=True,
            checkpoints_enabled=True,
            quiet_mode=True,
        )

    assert agent.no_tools is True
    assert agent._persist_disabled is True
    assert agent.skip_context_files is True
    assert agent.load_soul_identity is False
    assert agent.skip_background_review is True
    assert agent._memory_manager is None
    assert agent._memory_store is None
    assert agent._fallback_chain == []
    assert agent._fallback_model is None
    assert agent._checkpoint_mgr.enabled is False
    assert not (tmp_path / "state.db").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "profiles").exists()


def test_no_session_persistence_disables_checkpoint_and_db_before_open(tmp_path, monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key", base_url="https://example.test/v1", model="test-model",
            disable_session_persistence=True, checkpoints_enabled=True, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )

    assert agent._session_db is None
    assert agent._persist_disabled is True
    assert agent._checkpoint_mgr.enabled is False
    assert not (tmp_path / "state.db").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "profiles").exists()


def test_public_constructor_positional_slots_remain_compatible():
    from agent.agent_init import init_agent
    from run_agent import AIAgent

    expected_agent_prefix = [
        "self", "base_url", "api_key", "provider", "api_mode", "acp_command", "acp_args",
        "command", "args", "model", "max_iterations", "tool_delay", "enabled_toolsets",
        "disabled_toolsets", "save_trajectories", "verbose_logging", "quiet_mode",
    ]
    expected_init_prefix = [
        "agent", "base_url", "api_key", "provider", "api_mode", "acp_command", "acp_args",
        "command", "args", "model", "max_iterations", "enabled_toolsets", "disabled_toolsets",
        "save_trajectories", "verbose_logging", "quiet_mode",
    ]

    assert list(inspect.signature(AIAgent.__init__).parameters)[:len(expected_agent_prefix)] == expected_agent_prefix
    assert list(inspect.signature(init_agent).parameters)[:len(expected_init_prefix)] == expected_init_prefix


def test_no_tools_output_boundary_skips_transform_and_post_llm_hooks(monkeypatch):
    from agent.turn_finalizer import _apply_output_hooks

    hook_calls = []
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **_kwargs: hook_calls.append(name) or ["INJECTED_OUTPUT"],
    )
    agent = SimpleNamespace(no_tools=True, session_id="s", model="m")

    response, transformed, original = _apply_output_hooks(
        agent,
        "provider response",
        SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        platform="cli",
        effective_task_id="task",
        turn_id="turn",
        original_user_message="hello",
        messages=[],
    )

    assert response == "provider response"
    assert transformed is False
    assert original is None
    assert hook_calls == []


def test_no_tools_system_prompt_skips_plugin_sections(monkeypatch):
    from agent.system_prompt import _frozen_plugin_prompt_sections

    calls = []
    monkeypatch.setattr(
        "hermes_cli.plugins.render_system_prompt_sections",
        lambda *_args: calls.append(True) or ("PLUGIN PROMPT",),
    )
    agent = SimpleNamespace(no_tools=True, _cached_system_prompt=None)

    assert _frozen_plugin_prompt_sections(agent) == ()
    assert calls == []
