# Live assistant product roadmap

Canonical product epic: [#1 — Live voice, screen/camera context and verified tools](https://github.com/PedroVictorCoding/sail-framework/issues/1).

Final product: a local-first assistant that understands live speech, uses fresh screen/camera evidence, preserves task context, executes only user-granted tools, verifies outcomes, and replies with interruptible intelligible speech. Deliver a usable pretrained-adapter mode first and validate the eventual Sail-native multimodal/speech mode separately.

**Architectural Strategy & Consensus:** See [Architectural Debate (LLM_DEBATE.md)](../LLM_DEBATE.md) for the agreed Staged Hybrid Architecture (decoupling streaming ASR/TTS to CPU and dedicating 100% of GPU compute to SAIL's reasoning and continual training).

**Permanent invariant:** no autonomous recursive self-improvement, ever, with no exceptions. Preserve [ALIGNMENT.md](../ALIGNMENT.md). Learning uses human-approved external data through a human-defined pipeline; the model cannot control its implementation, training policy, permissions or deployment. Generated answers/actions are never automatically training data or rewards.

Every linked issue includes a product outcome, current context, implementation requirements, explicit dependencies, acceptance tests and completion evidence. Priority describes importance; dependencies determine readiness. Verify current implementation before duplicating work. GitHub issue bodies are the detailed work specifications.

## Foundation and model adapters

| Issue | Priority | Depends on |
| --- | --- | --- |
| [#2: [P0] Enforce the permanent alignment boundary and human-controlled capabilities](https://github.com/PedroVictorCoding/sail-framework/issues/2) | P0 | Foundation |
| [#3: [P0] Build the native host runtime and bounded timestamped event bus](https://github.com/PedroVictorCoding/sail-framework/issues/3) | P0 | [#2](https://github.com/PedroVictorCoding/sail-framework/issues/2) |
| [#4: [P0] Add interchangeable pretrained and native Sail inference backends](https://github.com/PedroVictorCoding/sail-framework/issues/4) | P0 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3) |

## Real voice conversation

| Issue | Priority | Depends on |
| --- | --- | --- |
| [#5: [P0] Implement production microphone capture and continuous audio playback](https://github.com/PedroVictorCoding/sail-framework/issues/5) | P0 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3) |
| [#6: [P0] Add echo cancellation, noise handling and speech endpoint detection](https://github.com/PedroVictorCoding/sail-framework/issues/6) | P0 | [#5](https://github.com/PedroVictorCoding/sail-framework/issues/5) |
| [#7: [P0] Replace fixed transcription with real streaming speech recognition](https://github.com/PedroVictorCoding/sail-framework/issues/7) | P0 | [#4](https://github.com/PedroVictorCoding/sail-framework/issues/4), [#6](https://github.com/PedroVictorCoding/sail-framework/issues/6) |
| [#8: [P0] Replace harmonic tones with real streaming neural speech synthesis](https://github.com/PedroVictorCoding/sail-framework/issues/8) | P0 | [#4](https://github.com/PedroVictorCoding/sail-framework/issues/4), [#5](https://github.com/PedroVictorCoding/sail-framework/issues/5) |
| [#9: [P0] Implement full-duplex dialogue, barge-in and cancellation semantics](https://github.com/PedroVictorCoding/sail-framework/issues/9) | P0 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3), [#7](https://github.com/PedroVictorCoding/sail-framework/issues/7), [#8](https://github.com/PedroVictorCoding/sail-framework/issues/8) |

## Grounded screen/camera actions

| Issue | Priority | Depends on |
| --- | --- | --- |
| [#10: [P0] Add selected-window capture, UI Automation and OCR context](https://github.com/PedroVictorCoding/sail-framework/issues/10) | P0 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3), [#2](https://github.com/PedroVictorCoding/sail-framework/issues/2) |
| [#11: [P1] Add camera capture, bounded temporal history and task-driven sampling](https://github.com/PedroVictorCoding/sail-framework/issues/11) | P1 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3), [#2](https://github.com/PedroVictorCoding/sail-framework/issues/2) |
| [#12: [P1] Implement visual understanding, region selection and grounding](https://github.com/PedroVictorCoding/sail-framework/issues/12) | P1 | [#4](https://github.com/PedroVictorCoding/sail-framework/issues/4), [#10](https://github.com/PedroVictorCoding/sail-framework/issues/10), [#11](https://github.com/PedroVictorCoding/sail-framework/issues/11) |
| [#13: [P0] Build synchronized multimodal context assembly and reference resolution](https://github.com/PedroVictorCoding/sail-framework/issues/13) | P0 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3), [#7](https://github.com/PedroVictorCoding/sail-framework/issues/7), [#12](https://github.com/PedroVictorCoding/sail-framework/issues/12) |
| [#14: [P0] Implement explicit session/task state and bounded model memory](https://github.com/PedroVictorCoding/sail-framework/issues/14) | P0 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3), [#13](https://github.com/PedroVictorCoding/sail-framework/issues/13) |
| [#16: [P0] Implement a typed, capability-scoped tool execution broker](https://github.com/PedroVictorCoding/sail-framework/issues/16) | P0 | [#2](https://github.com/PedroVictorCoding/sail-framework/issues/2), [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3) |
| [#17: [P0] Implement workspace, application and browser tool adapters](https://github.com/PedroVictorCoding/sail-framework/issues/17) | P0 | [#16](https://github.com/PedroVictorCoding/sail-framework/issues/16), [#10](https://github.com/PedroVictorCoding/sail-framework/issues/10) |
| [#18: [P0] Verify tool outcomes and coordinate multi-step task execution](https://github.com/PedroVictorCoding/sail-framework/issues/18) | P0 | [#13](https://github.com/PedroVictorCoding/sail-framework/issues/13), [#14](https://github.com/PedroVictorCoding/sail-framework/issues/14), [#17](https://github.com/PedroVictorCoding/sail-framework/issues/17) |

## Persistent and reliable local operation

| Issue | Priority | Depends on |
| --- | --- | --- |
| [#15: [P1] Add provenance-aware long-term retrieval and user preferences](https://github.com/PedroVictorCoding/sail-framework/issues/15) | P1 | [#14](https://github.com/PedroVictorCoding/sail-framework/issues/14), [#2](https://github.com/PedroVictorCoding/sail-framework/issues/2) |
| [#19: [P0] Schedule GPU/CPU resources to prioritize live interaction](https://github.com/PedroVictorCoding/sail-framework/issues/19) | P0 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3), [#4](https://github.com/PedroVictorCoding/sail-framework/issues/4) |
| [#20: [P0] Implement a versioned model registry and human-controlled promotion](https://github.com/PedroVictorCoding/sail-framework/issues/20) | P0 | [#2](https://github.com/PedroVictorCoding/sail-framework/issues/2), [#4](https://github.com/PedroVictorCoding/sail-framework/issues/4), [#19](https://github.com/PedroVictorCoding/sail-framework/issues/19) |
| [#21: [P1] Build the live assistant controls and causal trace dashboard](https://github.com/PedroVictorCoding/sail-framework/issues/21) | P1 | [#3](https://github.com/PedroVictorCoding/sail-framework/issues/3), [#9](https://github.com/PedroVictorCoding/sail-framework/issues/9), [#18](https://github.com/PedroVictorCoding/sail-framework/issues/18) |

## Trustworthy learning and native multimodal understanding

| Issue | Priority | Depends on |
| --- | --- | --- |
| [#22: [P0] Prove durable data identity, deduplication and independent splits](https://github.com/PedroVictorCoding/sail-framework/issues/22) | P0 | [#2](https://github.com/PedroVictorCoding/sail-framework/issues/2) |
| [#23: [P0] Build versioned paired multimodal preprocessing and target masking](https://github.com/PedroVictorCoding/sail-framework/issues/23) | P0 | [#22](https://github.com/PedroVictorCoding/sail-framework/issues/22) |
| [#24: [P0] Bring multimodal training, recovery and retention to text-loop parity](https://github.com/PedroVictorCoding/sail-framework/issues/24) | P0 | [#22](https://github.com/PedroVictorCoding/sail-framework/issues/22), [#23](https://github.com/PedroVictorCoding/sail-framework/issues/23) |
| [#25: [P1] Develop and validate Sail’s timestamped multimodal backbone](https://github.com/PedroVictorCoding/sail-framework/issues/25) | P1 | [#4](https://github.com/PedroVictorCoding/sail-framework/issues/4), [#23](https://github.com/PedroVictorCoding/sail-framework/issues/23) |
| [#26: [P1] Implement human-defined staged training and regression-gated adaptation](https://github.com/PedroVictorCoding/sail-framework/issues/26) | P1 | [#24](https://github.com/PedroVictorCoding/sail-framework/issues/24), [#25](https://github.com/PedroVictorCoding/sail-framework/issues/25) |

## Native full-duplex speech

| Issue | Priority | Depends on |
| --- | --- | --- |
| [#27: [P1] Train native speech-token output and a streaming codec decoder](https://github.com/PedroVictorCoding/sail-framework/issues/27) | P1 | [#25](https://github.com/PedroVictorCoding/sail-framework/issues/25), [#23](https://github.com/PedroVictorCoding/sail-framework/issues/23), [#8](https://github.com/PedroVictorCoding/sail-framework/issues/8) |
| [#28: [P1] Train native full-duplex speech and turn-taking behavior](https://github.com/PedroVictorCoding/sail-framework/issues/28) | P1 | [#27](https://github.com/PedroVictorCoding/sail-framework/issues/27), [#26](https://github.com/PedroVictorCoding/sail-framework/issues/26), [#9](https://github.com/PedroVictorCoding/sail-framework/issues/9) |

## Independent evidence, measured optimization and release

| Issue | Priority | Depends on |
| --- | --- | --- |
| [#29: [P0] Create independent capability evaluation and regression gates](https://github.com/PedroVictorCoding/sail-framework/issues/29) | P0 | [#22](https://github.com/PedroVictorCoding/sail-framework/issues/22), [#23](https://github.com/PedroVictorCoding/sail-framework/issues/23), [#4](https://github.com/PedroVictorCoding/sail-framework/issues/4) |
| [#30: [P1] Benchmark end-to-end assistant latency, memory and alternative implementations](https://github.com/PedroVictorCoding/sail-framework/issues/30) | P1 | [#19](https://github.com/PedroVictorCoding/sail-framework/issues/19), [#29](https://github.com/PedroVictorCoding/sail-framework/issues/29), [#4](https://github.com/PedroVictorCoding/sail-framework/issues/4) |
| [#31: [P2] Validate production sparse MoE dispatch with measured quality and speed](https://github.com/PedroVictorCoding/sail-framework/issues/31) — See [Specification](research/BLOCK_SPARSE_MOE_DISPATCH.md) | P2 | [#30](https://github.com/PedroVictorCoding/sail-framework/issues/30) |
| [#32: [P2] Validate fused scan/recomputation only against measured assistant bottlenecks](https://github.com/PedroVictorCoding/sail-framework/issues/32) — See [SSD & Recomputation Specification](research/STATE_SPACE_DUALITY_AND_RECOMPUTATION.md) | P2 | [#30](https://github.com/PedroVictorCoding/sail-framework/issues/30) |
| [#35: [P2] Implement decay-aware chunkwise gated linear attention](https://github.com/PedroVictorCoding/sail-framework/issues/35) — See [Specification](research/CHUNKWISE_GATED_LINEAR_ATTENTION.md) | P2 | [#30](https://github.com/PedroVictorCoding/sail-framework/issues/30) |
| [#33: [P0] Automate the voice-screen-tool acceptance scenario and fault recovery suite](https://github.com/PedroVictorCoding/sail-framework/issues/33) | P0 | [#9](https://github.com/PedroVictorCoding/sail-framework/issues/9), [#13](https://github.com/PedroVictorCoding/sail-framework/issues/13), [#14](https://github.com/PedroVictorCoding/sail-framework/issues/14), [#18](https://github.com/PedroVictorCoding/sail-framework/issues/18), [#20](https://github.com/PedroVictorCoding/sail-framework/issues/20), [#29](https://github.com/PedroVictorCoding/sail-framework/issues/29), [#30](https://github.com/PedroVictorCoding/sail-framework/issues/30), [#28](https://github.com/PedroVictorCoding/sail-framework/issues/28) |
| [#34: [P1] Package and document a reproducible native Windows assistant release](https://github.com/PedroVictorCoding/sail-framework/issues/34) | P1 | [#33](https://github.com/PedroVictorCoding/sail-framework/issues/33) |

## First end-to-end milestone

A real user asks about a visible editor error; Sail identifies the correct window and error, reads permitted files, runs the appropriate test, verifies its result and explains it aloud. The user interrupts with “Only run that one test”; Sail stops obsolete speech/pending work, preserves completed side effects and continues under the revised objective.

Start with #2–#3 and then #4–#9 for real voice; establish #22/#29 early for valid learning/evaluation; add screen/camera/tool grounding through #10–#18. Complete resource management, registry and UX, then native learned speech. [#33](https://github.com/PedroVictorCoding/sail-framework/issues/33) and [#34](https://github.com/PedroVictorCoding/sail-framework/issues/34) define final product acceptance and release.

Latency values in the epic are engineering targets, not measured results. Placeholder behavior, tensor-shape checks, historical contaminated validation or unmatched external benchmarks do not establish product completion.

