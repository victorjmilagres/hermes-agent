"""Todo tool: in-memory, revisioned task list for multi-step work. State lives on the
AIAgent (one per session), is re-injected after context compression, and every write bumps
a monotonic revision so UI clients can reject stale updates. One ``todo_list`` tool: pass
``todos`` to write, omit to read; every call returns the full list. No system-prompt mutation."""

import json
from typing import Any, Dict, List, Optional

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
# The list is re-read after every compression (format_for_injection), so unbounded
# content/count would defeat the compression it rides through. Caps apply equally to
# model-authored items and caller-replayed API history.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
# Max single todo tool-result payload accepted during history hydration, so a forged
# oversized result is dropped before parsing (AIAgent._hydrate_todo_store).
MAX_TODO_RESULT_CHARS = 512_000
# Deferred-call arguments are persisted as JSON strings too. Bound them before
# decoding so history hydration cannot spend unbounded work recognizing a call.
MAX_TODO_DEFERRED_ENVELOPE_CHARS = 512_000
_TRUNCATION_MARKER = "… [truncated]"
# Persisted as ordinary message content; ContextCompressor keys on this stable header to
# tell the synthetic post-compaction row from a real user message.
TODO_INJECTION_HEADER = "[Your active task list was preserved across context compression]"
_STATUS_MARKERS = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]", "cancelled": "[~]"}
_ACTIVE_STATUSES = {"pending", "in_progress"}
_TODO_TOOL_NAMES = frozenset({"todo", "todo_list"})


def is_todo_tool_call(call: Any) -> bool:
    """Whether a persisted call invokes the todo tool directly or through ``tool_call``."""
    function = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
    if isinstance(name, str) and name in _TODO_TOOL_NAMES:
        return True
    if name != "tool_call":
        return False
    if isinstance(arguments, str):
        if len(arguments) > MAX_TODO_DEFERRED_ENVELOPE_CHARS:
            return False
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError, RecursionError):
            return False
    if (
        not isinstance(arguments, dict)
        or not _decoded_json_within_limit(arguments, MAX_TODO_DEFERRED_ENVELOPE_CHARS)
        or set(arguments) != {"name", "arguments"}
        or arguments.get("name") != "todo_list"
    ):
        return False
    underlying_arguments = arguments.get("arguments")
    if isinstance(underlying_arguments, str):
        if len(underlying_arguments) > MAX_TODO_DEFERRED_ENVELOPE_CHARS:
            return False
        try:
            underlying_arguments = json.loads(underlying_arguments)
        except (ValueError, TypeError, RecursionError):
            return False
    return (
        isinstance(underlying_arguments, dict)
        and _decoded_json_within_limit(
            underlying_arguments, MAX_TODO_DEFERRED_ENVELOPE_CHARS
        )
    )


def _call_id(call: Any) -> Any:
    return call.get("id") if isinstance(call, dict) else getattr(call, "id", None)


def _decoded_json_within_limit(value: Any, limit: int) -> bool:
    """Bound already-decoded JSON without re-encoding or recursive descent."""
    remaining = limit
    stack = [value]
    seen_containers = set()
    while stack:
        current = stack.pop()
        remaining -= 1
        if remaining < 0:
            return False
        if isinstance(current, str):
            remaining -= len(current)
        elif isinstance(current, dict):
            identity = id(current)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            if len(current) > remaining:
                return False
            for key, item in current.items():
                if not isinstance(key, str):
                    return False
                remaining -= len(key)
                if remaining < len(stack) + 1:
                    return False
                stack.append(item)
        elif isinstance(current, list):
            identity = id(current)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            if len(current) + len(stack) > remaining:
                return False
            stack.extend(current)
        elif current is not None and not isinstance(current, (bool, int, float)):
            return False
        if remaining < 0:
            return False
    return True


def normalize_todo_snapshot(value: Any) -> Optional[Dict[str, Any]]:
    """Project the shared snapshot shape; history applies its own input budget."""
    if (
        not isinstance(value, dict)
        or not {"todos", "revision"}.issubset(value)
        or not set(value).issubset({"todos", "revision", "summary"})
        or not isinstance(value["todos"], list)
        or len(value["todos"]) > MAX_TODO_ITEMS
        or ("summary" in value and not isinstance(value["summary"], dict))
    ):
        return None
    revision = value["revision"]
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        return None
    return {"todos": list(value["todos"]), "revision": revision}


def _snapshot_from_result_content(content: Any) -> Optional[Dict[str, Any]]:
    if isinstance(content, str):
        if len(content) > MAX_TODO_RESULT_CHARS or '\"todos\"' not in content:
            return None
        try:
            content = json.loads(content)
        except (ValueError, TypeError, RecursionError):
            return None
    if not _decoded_json_within_limit(content, MAX_TODO_RESULT_CHARS):
        return None
    return normalize_todo_snapshot(content)


def latest_todo_snapshot_from_history(history: Any) -> Optional[Dict[str, Any]]:
    """Newest valid todo result from one bounded reverse pass with one-to-one pairing."""
    if not isinstance(history, list):
        return None
    pending_results: Dict[str, tuple[int, Any]] = {}
    latest_snapshot = None
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str):
            return None
        if latest_snapshot is not None:
            # Keep structural validation of the transcript, but never decode
            # obsolete results after selecting the newest authoritative group.
            continue
        if role in {"user", "system"}:
            pending_results.clear()
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                # Traversing backward means replacement keeps the result nearest its call.
                pending_results[tool_call_id] = (
                    index,
                    message.get("content"),
                )
            continue
        if role != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            pending_results.clear()
            continue
        call_ids = [_call_id(call) for call in calls]
        seen_ids = set()
        duplicate_ids = set()
        for call_id in call_ids:
            if not isinstance(call_id, str):
                continue
            if call_id in seen_ids:
                duplicate_ids.add(call_id)
            else:
                seen_ids.add(call_id)
        matches = []
        for call, call_id in zip(calls, call_ids):
            if (
                isinstance(call_id, str)
                and call_id
                and call_id not in duplicate_ids
                and call_id in pending_results
                and is_todo_tool_call(call)
            ):
                result_index, content = pending_results[call_id]
                snapshot = _snapshot_from_result_content(content)
                if snapshot is not None:
                    matches.append((result_index, snapshot))
        if matches and latest_snapshot is None:
            latest_snapshot = max(matches, key=lambda match: match[0])[1]
        pending_results.clear()
    return latest_snapshot


class TodoStore:
    """In-memory todo list, one per AIAgent. List position is priority; items are
    ``{id, content, status, parent?}`` — ``parent`` nests a subtask."""

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        self._revision = 0

    def _fresh_items(self, todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Validate, dedupe and order a whole new list (replace / restore)."""
        return self._normalize_order([self._validate(t) for t in self._dedupe_by_id(todos)])

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """Replace the list (default) or merge by id; returns the full list after writing."""
        before = self.read()
        if merge:
            self._merge(todos)
        else:
            self._items = self._fresh_items(todos)
        del self._items[MAX_TODO_ITEMS:]  # keep the priority head; replays can't grow unbounded
        self._sanitize_parents(self._items)
        if self._items != before:
            self._revision += 1
        return self.read()

    def _merge(self, todos: List[Dict[str, Any]]) -> None:
        """Update existing items only in the fields provided; append new ones (validated)."""
        existing = {item["id"]: item for item in self._items}
        for t in self._dedupe_by_id(todos):
            item_id = str(t.get("id", "")).strip()
            if not item_id:
                continue  # can't merge without an id
            cur = existing.get(item_id)
            if cur is None:
                validated = self._validate(t)
                existing[validated["id"]] = validated
                self._items.append(validated)
                continue
            if t.get("content"):
                cur["content"] = self._cap_content(str(t["content"]).strip())
            if t.get("status") and str(t["status"]).strip().lower() in VALID_STATUSES:
                cur["status"] = str(t["status"]).strip().lower()
            if "parent" in t:
                parent = str(t["parent"] or "").strip()
                if parent:
                    cur["parent"] = parent
                else:
                    cur.pop("parent", None)
        # Rebuild preserving original order for existing items (first occurrence wins).
        rebuilt = {item["id"]: existing.get(item["id"], item) for item in self._items}
        self._items = self._normalize_order(list(rebuilt.values()))

    def read(self) -> List[Dict[str, str]]:
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        return bool(self._items)

    def snapshot(self) -> Dict[str, Any]:
        """Full state clients can reconcile atomically."""
        return {"todos": self.read(), "revision": self._revision}

    def restore(self, todos: List[Dict[str, Any]], *, revision: Any = 0) -> List[Dict[str, str]]:
        """Restore a trusted snapshot without manufacturing a new revision."""
        self._items = self._fresh_items(todos)[:MAX_TODO_ITEMS]
        try:
            self._revision = max(0, int(revision or 0))
        except (TypeError, ValueError):
            self._revision = 0
        return self.read()

    def format_for_injection(self) -> Optional[str]:
        """Render the list for post-compression injection, or None if nothing active. Only
        pending/in_progress items are injected — finished ones make the model re-do work after
        compression. A parent is kept (with its real status marker) when any descendant is
        active so subtasks keep context."""
        if not self._items:
            return None
        children: Dict[str, List[Dict[str, str]]] = {}
        for item in self._items:
            if item.get("parent"):
                children.setdefault(item["parent"], []).append(item)

        def render(item: Dict[str, str], depth: int, out: List[str]) -> bool:
            kid_lines: List[str] = []
            has_active_kid = False
            for kid in children.get(item["id"], []):
                has_active_kid |= render(kid, depth + 1, kid_lines)
            keep = item["status"] in _ACTIVE_STATUSES or has_active_kid
            if keep:
                marker = _STATUS_MARKERS.get(item["status"], "[?]")
                out.append(f"{'  ' * depth}- {marker} {item['id']}. "
                           f"{item['content']} ({item['status']})")
                out.extend(kid_lines)
            return keep

        lines = [TODO_INJECTION_HEADER]
        for item in self._items:
            if not item.get("parent"):
                render(item, 0, lines)
        return "\n".join(lines) if len(lines) > 1 else None

    @staticmethod
    def _cap_content(content: str) -> str:
        """Truncate to MAX_TODO_CONTENT_CHARS keeping the head (the actionable part) + marker."""
        if len(content) > MAX_TODO_CONTENT_CHARS:
            return content[:MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """Normalize one item to ``{id, content, status, parent?}`` (placeholders when missing)."""
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}
        item_id = str(item.get("id", "")).strip() or "?"
        content = str(item.get("content", "")).strip()
        status = str(item.get("status", "pending")).strip().lower()
        result = {"id": item_id,
                  "content": TodoStore._cap_content(content) if content else "(no description)",
                  "status": status if status in VALID_STATUSES else "pending"}
        parent = str(item.get("parent") or "").strip()
        if parent and parent != item_id:
            result["parent"] = parent
        return result

    @staticmethod
    def _sanitize_parents(items: List[Dict[str, str]]) -> None:
        """Drop dangling parent refs and break cycles in place (such items become roots)."""
        by_id = {item["id"]: item for item in items}
        for item in items:
            if item.get("parent") and item["parent"] not in by_id:
                item.pop("parent", None)
        for item in items:
            seen, node = {item["id"]}, item
            while node.get("parent"):
                if node["parent"] in seen:
                    item.pop("parent", None)
                    break
                seen.add(node["parent"])
                node = by_id[node["parent"]]

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):  # non-dicts get a synthetic key; _validate handles them
            key = str(item.get("id", "")).strip() if isinstance(item, dict) else f"__invalid_{i}"
            last_index[key or "?"] = i
        return [todos[i] for i in sorted(last_index.values())]

    @staticmethod
    def _normalize_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Lift the in_progress step ahead of any earlier pending placeholder. Nested lists
        keep authored order — reordering would tear a subtask from its siblings."""
        statuses = [item["status"] for item in items]
        if any(item.get("parent") for item in items) or "in_progress" not in statuses:
            return items
        active_index = statuses.index("in_progress")
        if "pending" not in statuses[:active_index]:
            return items
        normalized = items.copy()
        normalized.insert(statuses.index("pending"), normalized.pop(active_index))
        return normalized


def todo_tool(todos: Optional[List[Dict[str, Any]]] = None, merge: bool = False,
              store: Optional[TodoStore] = None) -> str:
    """Write ``todos`` (replace, or ``merge`` by id) or read when None -> list + summary JSON."""
    if store is None:
        return tool_error("TodoStore not initialized")
    if todos is None:
        items = store.read()
    else:
        if isinstance(todos, str):  # LLMs sometimes send a JSON string instead of a list
            try:
                todos = json.loads(todos)
            except (ValueError, TypeError, RecursionError):
                return tool_error("todos must be a list of objects, got unparseable string")
        if not isinstance(todos, list):
            return tool_error(f"todos must be a list, got {type(todos).__name__}")
        items = store.write(todos, merge)
    summary = {"total": len(items)}
    for status in ("pending", "in_progress", "completed", "cancelled"):
        summary[status] = sum(1 for i in items if i["status"] == status)
    return json.dumps({"todos": items, "revision": store.snapshot()["revision"],
                       "summary": summary}, ensure_ascii=False)


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


# Behavioral guidance is baked into the (static, cached) description; item shape and merge
# semantics live ONLY in the parameter schema.
TODO_SCHEMA = {
    "name": "todo_list",
    "description": (
        # See #95681.
        "Track a task list for multi-step work (3+ steps). Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "For 'all N items' tasks, enumerate every instance as its own checklist "
        "item so none are silently dropped. "
        "Call with no parameters to read the current list.\n"
        "List order is priority. Only ONE item in_progress at a time. "
        "Break large phases into subtasks via parent. "
        "Mark an item completed only after the work is verified done, never "
        "based on intent. If something fails, cancel it and add a revised "
        "item. Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"]
                        },
                        "parent": {
                            "type": "string",
                            "description": "Optional id of another item, making this a nested subtask. Omit for top-level."
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list with a fresh plan."
                ),
                "default": False
            }
        },
        "required": []
    }
}


from tools.registry import registry, tool_error

registry.register(
    name="todo_list", toolset="todo", schema=TODO_SCHEMA, check_fn=check_todo_requirements,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"), merge=args.get("merge", False), store=kw.get("store")),
    emoji="📋")
