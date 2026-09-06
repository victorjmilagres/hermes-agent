# Todo hydration verification

Resume uses the same paired tool-result history for the agent store and the
TUI/Desktop state projection. Direct `todo`, `todo_list`, and deferred
`tool_call` invocations of `todo_list` are recognized. No additional database
or persistence writer is introduced.

The inherited implementation was preserved. Independent review found and
corrected early parsing of unpaired results, uncaught JSON depth/numeric-limit
errors, container expansion before budget checks, suppression of valid live
snapshots by a history-only cap, and decoding obsolete historical snapshots.

Validation on 2026-09-06:

- The new run-agent regressions reproduced six failures on base `b0ab2e16` in
  an isolated temporary worktree. The TUI test file needs new symbols and did
  not collect on that base; this is not counted as a behavioral RED result.
- Two added invariant tests reproduced the review findings before fixes.
- 371 tests in seven relevant files passed using `scripts/run_tests.sh -j 2`: run-agent,
  todo tool, nested tasks, coercion, hydration regressions, TUI todo events,
  and session-resume database ownership. No failures or retries were reported.
- Final independent code review reported no actionable findings. The reviewer
  inspected supplied source; test execution was performed separately.
- Ruff, diff check, and the repository plugin-compat pointer check passed.
- No production gateway restart, migration, or model invocation was performed.

History retains its 512,000-character per-result budget. Valid live store
snapshots do not inherit that historical-input restriction. Structural history
validation remains fail-closed even after finding the latest snapshot, while
obsolete contents are no longer decoded.
