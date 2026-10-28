# Workspace, Application, and Browser Tool Adapters

This document details the architecture, safety constraints, and structured artifact interfaces for the SAIL Workspace, Application, and Browser tool adapters (`src/tool/workspace.rs`, `src/tool/application.rs`, and `src/tool/artifact.rs`).

## 1. Motivation and Design Goals

In a verified assistant architecture, invoking tools must not default to indiscriminate mouse-clicking or passing arbitrary raw strings to a shell interpreter:
1. **Scoped Workspace Boundaries**: Tools that search, read, or write code must enforce canonical path restrictions bounded by `allowed_read_roots` and `allowed_write_roots`. Traversal attempts (such as `../`) are rejected before filesystem access.
2. **Stale File Version & Concurrency Protection**: If a user or IDE edits a file on disk while the assistant is reasoning, any subsequent write with an outdated expected timestamp or hash is rejected with `StaleFileVersion`.
3. **Structured Process Execution vs. Shell Evaluation**: The `WorkspaceTestRunnerTool` runs targeted tests via direct argument vectors (`["test", "<test_name>", "--offline"]`) in `std::process::Command`, making shell injection impossible.
4. **Precondition-Verified UI Interactions**: Window focusing, control clicks, and shortcuts check the active UI tree and DPI scale factor against expected bounds. If controls have moved or the window has relocated across monitors, explicit errors (`ControlMovedError`, `DpiMismatchError`) prompt a UI refresh rather than clicking stale pixels.
5. **URL Scheme Restrictions**: The browser adapter strictly permits `http://` and `https://`, rejecting dangerous schemes like `file://` or `javascript:`.

---

## 2. Adapters Summary

| Tool Name | Safety Level | Idempotent? | Key Parameters | Output / Artifact |
|---|---|---|---|---|
| `workspace_file_search` | `ReadOnly` | Yes | `query`, `root_dir`, `extension` | JSON list of matching files & sizes |
| `workspace_file_read` | `ReadOnly` | Yes | `path`, `start_line`, `end_line`, `expected_mtime_ms` | Line slice, total lines, speech summary |
| `workspace_file_write` | `DestructiveOrExternal` | No | `path`, `content`, `expected_mtime_ms` | Atomic write verification, bytes written |
| `workspace_test_runner` | `ReadOnly` | Yes | `test_name`, `package`, `release` | Exit code, passed/failed count, duration, snippet |
| `application_control` | `DestructiveOrExternal` | No | `app_name`, `action`, `target_element_id`, `expected_bounds`, `expected_scale_factor` | Window/element action status, UI tree |
| `browser_interaction` | `DestructiveOrExternal` | No | `action`, `url`, `selector` | Loaded URL, DOM click, extracted content |

---

## 3. Structured Artifacts and Speech Summaries

Every tool adapter emits a `StructuredToolArtifact`:
- `tool_name`: Identifier of the executing adapter.
- `success`: Boolean indicating whether the action succeeded.
- `speech_summary`: Short, natural sentence designed for zero-latency TTS playback (e.g., *"Test 'test_bpe_save_and_load' passed in 0.31s: 1 passed, 0 failed."*).
- `detailed_payload`: Full structured JSON for deep model reasoning, debugging, and verification traces.
- `warnings`: Actionable warnings (e.g. truncated output, deprecation notices).

---

## 4. Verification

The test suite in `tests/test_workspace_and_app_adapters.rs` validates:
- Reading files and running real targeted `cargo test` processes with structured result parsing.
- Rejection of path traversal attempts outside workspace roots.
- Detection of stale file versions during concurrent writes.
- Explicit errors for unsupported desktop applications.
- Rejection of dangerous URL schemes in browser navigation.
- Detection of moved UI controls and DPI scale mismatches before dispatching clicks.

