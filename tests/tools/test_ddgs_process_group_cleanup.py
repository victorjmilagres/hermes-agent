"""Cancellation must also stop descendants that retain the search pipes."""
import os
import selectors
import signal
import subprocess
import sys
import time

import psutil
import pytest


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("leader_exits", [False, True])
def test_cleanup_stops_term_ignoring_descendant(tmp_path, leader_exits):
    from plugins.web.ddgs.provider import _terminate_and_reap

    # The child signals readiness only after installing its TERM handler.
    code = """
import os, signal, sys, time
reader, writer = os.pipe()
child = os.fork()
if child == 0:
    os.close(reader)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(writer, b'1')
    os.close(writer)
    time.sleep(30)
    os._exit(0)
os.close(writer)
os.read(reader, 1)
os.close(reader)
print(child, flush=True)
if sys.argv[1] == 'exit':
    sys.exit(0)
time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", code, "exit" if leader_exits else "wait"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"},
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        assert selector.select(5), "synthetic descendant did not become ready"
        child_pid = int(process.stdout.readline())
        child = psutil.Process(child_pid)
        if leader_exits:
            process.wait(timeout=5)
        _terminate_and_reap(process, grace=0.2)
        deadline = time.monotonic() + 3
        while child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "search descendant survived cleanup"
            time.sleep(0.05)
        assert process.poll() is not None
    finally:
        selector.close()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        process.stdout.close()
