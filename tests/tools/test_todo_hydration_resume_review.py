"""Bounds apply to historical evidence without suppressing valid live state."""
import json
from unittest.mock import patch

from tools.todo_tool import TodoStore, latest_todo_snapshot_from_history, _decoded_json_within_limit
from tui_gateway.tool_progress import _normalize_todo_state


def test_history_pairing_and_limits_precede_untrusted_decoding():
    safe = {'todos': [{'id': '1', 'content': 'recover me', 'status': 'pending'}], 'revision': 2}
    call = {'role': 'assistant', 'tool_calls': [{'id': 'todo', 'function': {'name': 'todo_list'}}]}
    result = {'role': 'tool', 'tool_call_id': 'todo', 'content': safe}
    deep = '{"todos":' + '[' * 2000 + '0' + ']' * 2000 + '}'
    with patch('tools.todo_tool.json.loads', side_effect=AssertionError('unpaired result decoded')):
        assert latest_todo_snapshot_from_history([
            {'role': 'tool', 'tool_call_id': 'unpaired', 'content': deep}, call, result
        ]) == safe
    with patch('tools.todo_tool.json.loads', side_effect=AssertionError('obsolete result decoded')):
        assert latest_todo_snapshot_from_history([call, dict(result, content=deep), call, result]) == safe
    # A paired malformed candidate also cannot destroy a valid older snapshot.
    for malformed in (deep, '{"todos":[],"revision":' + '9' * 5000 + '}'):
        assert latest_todo_snapshot_from_history([call, result, call, dict(result, content=malformed)]) == safe
    class OversizedList(list):
        def __iter__(self):
            raise AssertionError('oversized list expanded')
    class OversizedDict(dict):
        def items(self):
            raise AssertionError('oversized dictionary expanded')
    assert not _decoded_json_within_limit(OversizedList([0] * 100), 10)
    assert not _decoded_json_within_limit(OversizedDict({str(i): 0 for i in range(100)}), 10)


def test_live_state_supports_full_store_capacity_without_relaxing_history():
    store = TodoStore()
    store.write([{'id': str(i), 'content': 'x' * 4000, 'status': 'pending'} for i in range(129)])
    snapshot = store.snapshot()
    assert _normalize_todo_state(snapshot) == snapshot
    history = [
        {'role': 'assistant', 'tool_calls': [{'id': 'todo', 'function': {'name': 'todo_list'}}]},
        {'role': 'tool', 'tool_call_id': 'todo', 'content': json.dumps(snapshot)},
    ]
    assert latest_todo_snapshot_from_history(history) is None
