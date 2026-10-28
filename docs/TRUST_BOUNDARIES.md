# Trust Boundaries and Capability Enforcement

This document formalizes the immutable security architecture and trust boundaries of the Sail framework, strictly enforcing the permanent foundational alignment rule from [`ALIGNMENT.md`](../ALIGNMENT.md).

---

## 1. Foundational Alignment Invariant

> **SAIL must never perform autonomous recursive self-improvement. This rule is permanent, non-negotiable, and has no exceptions.**

Learning within Sail is strictly limited to continuous, context-dependent training on externally sourced, human-approved data through a human-defined and human-executed training pipeline. Model outputs, assistant actions, tool traces, and runtime user sessions must never be automatically routed into training queues or reward channels.

---

## 2. Trust Domains & System Architecture

```
+-----------------------------------------------------------------------------------+
|                           HOST RUNTIME DOMAIN (TRUSTED)                           |
|                                                                                   |
|  - Owns configuration files, ALIGNMENT.md, and source code (`src/`, `Cargo.toml`) |
|  - Controls capability grants via `CapabilityPolicy` (human-configured only)      |
|  - Executes training (`sail stream-train`) and model promotion                     |
|  - Ingests human-approved external data (`scripts/hf_streamer.py`, `sqlite3`)      |
+------------------------------------------+----------------------------------------+
                                           |
                                  Host Capability Gate
                             (Typed Capability Enforcement)
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                        MODEL EXECUTION CONTEXT (UNTRUSTED)                        |
|                                                                                   |
|  - Autoregressive SSM / Transformer hybrid inference                              |
|  - Emits text generation & tool call requests: `<tool_call>name(...)`              |
|  - Strictly forbidden from invoking privileged actions or registering tools      |
+------------------------------------------+----------------------------------------+
                                           |
                                 Observation Ingestion
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     ENVIRONMENT OBSERVATION DOMAIN (UNTRUSTED)                     |
|                                                                                   |
|  - Screen frames (OCR text, UI elements)                                          |
|  - Microphone audio (speech-to-text transcripts)                                  |
|  - External web & tool outputs                                                    |
|  - Treated as `UntrustedContent`: sanitized and blocked from privilege escalation |
+-----------------------------------------------------------------------------------+
```

---

## 3. Host-Enforced Invariants

### A. Immutable Host-Side Boundary
The model execution context cannot:
1. Start or resume training (`StartTraining`).
2. Promote or deploy new model checkpoints (`PromoteModel`).
3. Alter or expand security policies (`AlterSecurityPolicy`).
4. Modify or rewrite framework implementation code or build files (`ModifyImplementationCode`).
5. Register new tools into the tool registry (`RegisterNewTool`).
6. Relax, bypass, or override alignment rules (`BypassAlignment`).

Any attempt by a model generation or tool string to invoke these actions is intercepted by the host runtime and results in an immediate `SecurityError::PrivilegedActionProhibited` or `SecurityError::AlignmentViolation`.

### B. Typed Capability API
Capabilities are explicitly configured by the human operator using `CapabilityPolicy`:
- `Capability::ToolExecution(String)`
- `Capability::ScreenCapture`
- `Capability::MicrophoneCapture`
- `Capability::CameraCapture`
- `Capability::AudioOutput`
- `Capability::FileSystemRead(PathBuf)`
- `Capability::FileSystemWrite(PathBuf)`
- `Capability::NetworkAccess(String)`

Privileged actions (training, tool registration, policy alteration) are **not representable** as capabilities and cannot be granted under any circumstance.

### C. Protected Implementation Paths & Canonical Path Confinement
The host enforces strict filesystem boundaries and canonical path confinement across all file operations:
1. **Target Identity Resolution**: Target paths are resolved to their canonical identities using `resolve_canonical_target` prior to any policy evaluation. For non-existent target files (e.g. pending writes), the longest existing ancestor path is canonicalized to defeat symlink aliasing, dot-dot traversal (`..`), and absolute path prefix evasion.
2. **Permanent Exclusions and Ancestor Protection**: Even if a tool is granted write access to a workspace directory, write, delete, and rename operations to the following paths are permanently blocked:
   - Framework source code: `src/`, `tests/`, `benches/`, `scripts/` under protected roots
   - Build definitions: `Cargo.toml`, `Cargo.lock`, `build.rs`
   - Governance and alignment rules: `ALIGNMENT.md`, `AGENTS.md`
   - Version control metadata: `.git/`
   - Active model checkpoints: `checkpoints/`
   - Any directory ancestor containing a protected framework root (mutations or deletions that would relocate or destroy protected framework subtrees are blocked with `ProtectedPathViolation`).
3. **Framework vs. User Project Separation & Permanent Protection**: Unrelated user projects can be explicitly authorized via `with_user_project_root` so that user code editing is permitted. However, `with_user_project_root` strictly prevents reclassifying or removing protection from designated SAIL framework roots, their ancestors, or any workspace root containing `ALIGNMENT.md`.
4. **End-to-End Tool & Verifier Confinement**: All file-facing tools (`TypedFileReadTool`, `TypedFileDeleteTool`, `WorkspaceFileReadTool`, `WorkspaceFileWriteTool`, `WorkspaceFileSearchTool`) and postcondition verifiers (`StepPostcondition`) enforce capability policy checks at the host/broker boundary and at tool execution time. Constructor roots cannot widen revoked or restricted host policy.
5. **Descendant Search Traversal**: Recursive workspace search (`WorkspaceFileSearchTool`) checks the authoritative host read policy on every traversed descendant entry and directory, detects symlink cycles with canonical `visited_dirs` tracking, and caps traversal budgets at 5000 entries.
6. **Platform & Concurrency Scope**: Path canonicalization resolves paths at check time; full TOCTOU race immunity across concurrent adversarial host filesystem processes requires platform-specific atomic directory descriptors (`openat2` on Linux / `FILE_FLAG_BACKUP_SEMANTICS` with handle operations on Windows).

### D. Segregation of Runtime Data from Training Pipelines
To prevent recursive feedback loops and self-training degradation:
- Assistant conversational answers, tool traces, and self-evaluations are flagged with `ProvenanceKind::RuntimeSyntheticOutput` and rejected by training watchers (`IncomingDirectoryWatcher`, `IncomingMultimodalWatcher`).
- Live screen and audio captures from interactive user sessions are flagged with `ProvenanceKind::LiveCapturedSession` and rejected until explicitly processed through a human-curated external dataset workflow.
- Ingestion gates check for runtime origin headers and verify cryptographic hashes before adding records to streaming buffers.

### E. Prompt Injection Resistance
Untrusted inputs originating from screen OCR, user speech transcripts, or third-party tool responses are wrapped in `UntrustedContent`.
1. Host-level pattern scanners detect and reject adversarial triggers attempting to command the host runtime (e.g. `system override:`, `grant capability`, `<tool_call>register_tool`, `alter_policy`).
2. In the event that novel prompt injection text reaches the model and influences its output, the host's typed capability gate drops the request because privileged actions do not exist in the dispatch table.

