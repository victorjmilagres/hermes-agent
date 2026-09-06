"""Todo snapshots bypass optional tool-progress display settings."""

import json

import pytest

import tui_gateway.server as server
import tools.todo_tool as todo_tool
from tools.todo_tool import (
    MAX_TODO_DEFERRED_ENVELOPE_CHARS,
    MAX_TODO_RESULT_CHARS,
    latest_todo_snapshot_from_history,
)


def _history_call(call_id, name="todo", arguments=None):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments or {}),
                },
            }
        ],
    }


def _history_result(call_id, revision, *, todos=None, content=None):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content
        if content is not None
        else json.dumps(
            {
                "todos": todos
                if todos is not None
                else [{"id": call_id, "content": "Recovered", "status": "pending"}],
                "revision": revision,
            }
        ),
    }


def test_todo_completion_always_emits_snapshot_and_compat_event(monkeypatch):
    sid = "todo-state-test"
    events = []
    session = {
        "agent": None,
        "edit_snapshots": {},
        "tool_started_at": {},
        "tool_progress_mode": "off",
    }
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: False)
    monkeypatch.setattr(server, "_tool_lifecycle_required_for_ui", lambda _name: False)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, event_sid, payload=None: events.append(
            (event, event_sid, payload)
        ),
    )

    state = {
        "todos": [{"id": "1", "content": "Work", "status": "in_progress"}],
        "revision": 9,
    }
    server._on_tool_complete(sid, "call-1", "todo", {}, json.dumps(state))

    assert [event[0] for event in events] == ["tool.complete", "todo.updated"]
    assert events[-1] == ("todo.updated", sid, state)
    assert session["todo_state"] == state


def test_non_todo_completion_stays_suppressed_when_progress_is_off(monkeypatch):
    sid = "ordinary-tool-test"
    events = []
    monkeypatch.setitem(
        server._sessions,
        sid,
        {
            "agent": None,
            "edit_snapshots": {},
            "tool_started_at": {},
            "tool_progress_mode": "off",
        },
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: False)
    monkeypatch.setattr(server, "_tool_lifecycle_required_for_ui", lambda _name: False)
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))

    server._on_tool_complete(sid, "call-1", "terminal", {}, "ok")

    assert events == []


def test_live_snapshot_prefers_the_highest_revision():
    class Store:
        @staticmethod
        def snapshot():
            return {"todos": [], "revision": 4}

    class Agent:
        _todo_store = Store()

    session = {
        "agent": Agent(),
        "todo_state": {
            "todos": [{"id": "1", "content": "Current", "status": "pending"}],
            "revision": 5,
        },
    }

    payload = server._attach_todo_state({}, session)

    assert payload["todo_state"]["revision"] == 5


def test_unused_store_is_not_attached():
    class Store:
        @staticmethod
        def snapshot():
            return {"todos": [], "revision": 0}

    class Agent:
        _todo_store = Store()

    payload = server._attach_todo_state({}, {"agent": Agent()})

    assert "todo_state" not in payload


def test_empty_list_at_nonzero_revision_is_a_real_clear():
    state = server._normalize_todo_state({"todos": [], "revision": 2})

    assert state == {"todos": [], "revision": 2}


def test_history_snapshot_accepts_direct_and_deferred_todo_calls():
    for revision, name in enumerate(("todo", "todo_list", "tool_call"), start=1):
        arguments = (
            {"name": "todo_list", "arguments": {}}
            if name == "tool_call"
            else {}
        )
        call_id = f"call-{revision}"
        history = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {
                        "todos": [
                            {"id": name, "content": "Recovered", "status": "pending"}
                        ],
                        "revision": revision,
                    }
                ),
            },
        ]

        assert server._todo_state_from_history(history) == {
            "todos": [{"id": name, "content": "Recovered", "status": "pending"}],
            "revision": revision,
        }

    malformed_call = history[0]
    malformed_call["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"name": "todo_list", "arguments": []}
    )
    history[1]["content"] = json.dumps({"todos": [], "revision": 99})
    assert server._todo_state_from_history(history) is None

    history[0]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"name": "todo_list", "arguments": {}}
    )
    history[1]["tool_call_id"] = "unpaired"
    assert server._todo_state_from_history(history) is None


@pytest.mark.parametrize(
    "invalid_tail",
    [
        [_history_result("future", 99), _history_call("future")],
        [_history_call("boundary"), {"role": "user", "content": "next"}, _history_result("boundary", 99)],
        [_history_call("boundary"), {"role": "system", "content": "policy"}, _history_result("boundary", 99)],
        [_history_call("same", "terminal"), _history_result("same", 99)],
        [
            _history_call(
                "nested",
                "tool_call",
                {"name": "tool_call", "arguments": {"name": "todo_list", "arguments": {}}},
            ),
            _history_result("nested", 99),
        ],
        [_history_call("array"), _history_result("array", 99, content='["todos"]')],
        [_history_call("huge-result"), _history_result("huge-result", 99, content='{"todos":"' + "x" * MAX_TODO_RESULT_CHARS + '"}')],
        [
            _history_call(
                "huge-envelope",
                "tool_call",
                {"name": "todo_list", "arguments": {}, "padding": "x" * MAX_TODO_RESULT_CHARS},
            ),
            _history_result("huge-envelope", 99),
        ],
        [_history_call("malformed"), _history_result("malformed", 99, content='{"todos":')],
    ],
    ids=[
        "result-before-call",
        "user-boundary",
        "system-boundary",
        "reused-id",
        "nested-deferred-wrapper",
        "non-object-result",
        "oversized-result",
        "oversized-deferred-envelope",
        "newest-malformed-result",
    ],
)
def test_history_snapshot_uses_nearest_valid_temporal_pair(invalid_tail):
    older = [_history_call("same"), _history_result("same", 7)]

    assert server._todo_state_from_history(older + invalid_tail) == {
        "todos": [{"id": "same", "content": "Recovered", "status": "pending"}],
        "revision": 7,
    }


def test_history_snapshot_uses_newest_valid_pair_not_highest_revision():
    history = [
        _history_call("older"),
        _history_result("older", 7),
        _history_call("newer"),
        _history_result("newer", 2),
    ]

    assert server._todo_state_from_history(history)["revision"] == 2


def test_oversized_history_fields_are_rejected_before_json_parse(monkeypatch):
    original_loads = json.loads

    def bounded_loads(value, *args, **kwargs):
        assert not isinstance(value, str) or len(value) <= MAX_TODO_RESULT_CHARS
        return original_loads(value, *args, **kwargs)

    history = [
        _history_call("older"),
        _history_result("older", 3),
        _history_call("huge-result"),
        _history_result("huge-result", 99, content='{"todos":"' + "x" * MAX_TODO_RESULT_CHARS + '"}'),
        _history_call(
            "huge-envelope",
            "tool_call",
            {"name": "todo_list", "arguments": {}, "padding": "x" * MAX_TODO_RESULT_CHARS},
        ),
        _history_result("huge-envelope", 99),
    ]
    monkeypatch.setattr(todo_tool.json, "loads", bounded_loads)

    assert server._todo_state_from_history(history)["revision"] == 3


def test_history_walk_is_linear_and_tool_results_pair_one_to_one():
    class CountedMessage(dict):
        role_reads = 0

        def get(self, key, default=None):
            if key == "role":
                type(self).role_reads += 1
            return super().get(key, default)

    candidates = [
        CountedMessage(_history_result(f"unpaired-{index}", index + 1))
        for index in range(1_000)
    ]
    assert latest_todo_snapshot_from_history(candidates) is None
    assert CountedMessage.role_reads == len(candidates)

    reused_result_id = [
        _history_call("one-call"),
        _history_result("one-call", 2),
        _history_result("one-call", 99),
    ]
    assert server._todo_state_from_history(reused_result_id)["revision"] == 2

    duplicate_call_ids = [
        _history_call("older"),
        _history_result("older", 7),
        {
            "role": "assistant",
            "tool_calls": [
                _history_call("duplicate")["tool_calls"][0],
                _history_call("duplicate")["tool_calls"][0],
            ],
        },
        _history_result("duplicate", 99),
    ]
    assert server._todo_state_from_history(duplicate_call_ids)["revision"] == 7


@pytest.mark.parametrize(
    "bad_snapshot",
    [
        {"todos": [], "revision": True},
        {"todos": [], "revision": "8"},
        {"todos": [], "revision": -1},
        {"todos": []},
        {"todos": {}, "revision": 8},
        {"todos": [], "revision": 8, "unexpected": "field"},
        {"todos": [], "revision": 8, "summary": []},
    ],
)
def test_history_snapshot_validation_is_canonical_and_falls_back(bad_snapshot):
    history = [
        _history_call("older"),
        _history_result("older", 7),
        _history_call("invalid"),
        {"role": "tool", "tool_call_id": "invalid", "content": json.dumps(bad_snapshot)},
    ]

    expected = {
        "todos": [{"id": "older", "content": "Recovered", "status": "pending"}],
        "revision": 7,
    }
    assert latest_todo_snapshot_from_history(history) == expected
    assert server._todo_state_from_history(history) == expected


@pytest.mark.parametrize("bad_role", [None, 1, [], {}])
def test_history_snapshot_fails_closed_for_non_string_roles(bad_role):
    malformed = {"role": bad_role, "content": "malformed"}
    valid_pair = [_history_call("older"), _history_result("older", 7)]

    for history in ([malformed, *valid_pair], [*valid_pair, malformed]):
        assert latest_todo_snapshot_from_history(history) is None
        assert server._todo_state_from_history(history) is None


def test_already_decoded_deferred_arguments_and_results_are_bounded():
    oversized_arguments = {
        "name": "todo_list",
        "arguments": {"padding": "x" * MAX_TODO_DEFERRED_ENVELOPE_CHARS},
    }
    oversized_call = _history_call("oversized-args", "tool_call")
    oversized_call["tool_calls"][0]["function"]["arguments"] = oversized_arguments
    oversized_result = {
        "todos": [
            {
                "id": "oversized",
                "content": "x" * MAX_TODO_RESULT_CHARS,
                "status": "pending",
            }
        ],
        "revision": 99,
    }
    decoded_call = _history_call("oversized-result", "tool_call")
    decoded_call["tool_calls"][0]["function"]["arguments"] = {
        "name": "todo_list",
        "arguments": {},
    }
    older = [_history_call("older"), _history_result("older", 7)]

    oversized_args_history = older + [oversized_call, _history_result("oversized-args", 98)]
    oversized_result_history = older + [
        decoded_call,
        {"role": "tool", "tool_call_id": "oversized-result", "content": oversized_result},
    ]

    assert server._todo_state_from_history(oversized_args_history)["revision"] == 7
    assert server._todo_state_from_history(oversized_result_history)["revision"] == 7

    malformed_call = _history_call("malformed-decoded", "tool_call")
    malformed_call["tool_calls"][0]["function"]["arguments"] = [{"unhashable": True}]
    malformed_history = older + [malformed_call, _history_result("malformed-decoded", 99)]
    assert server._todo_state_from_history(malformed_history)["revision"] == 7

    malformed_name = _history_call("malformed-name")
    malformed_name["tool_calls"][0]["function"]["name"] = []
    malformed_name_history = older + [malformed_name, _history_result("malformed-name", 99)]
    assert server._todo_state_from_history(malformed_name_history)["revision"] == 7
