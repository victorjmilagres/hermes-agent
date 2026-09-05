"""Construction-scoped guard for stateless agent startup."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_session_persistence_disabled: ContextVar[bool] = ContextVar(
    "hermes_session_persistence_disabled", default=False,
)


def session_persistence_disabled() -> bool:
    return _session_persistence_disabled.get()


@contextmanager
def without_session_persistence(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    token = _session_persistence_disabled.set(True)
    try:
        yield
    finally:
        _session_persistence_disabled.reset(token)
