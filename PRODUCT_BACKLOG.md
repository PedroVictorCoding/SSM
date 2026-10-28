# Product backlog

Foundational constraint: [ALIGNMENT.md](ALIGNMENT.md). **No autonomous recursive self-improvement, ever; no exceptions.** All learning features must use context-dependent external data within a human-defined training pipeline.

Current direction: prioritize measured training/inference performance and algorithmic
experiments in the Rust/Burn SSM/MoE model. GUI expansion is secondary. Brain-inspired
architectures remain deferred research proposals.

## Current product

The final live-assistant goal is tracked in [GitHub epic #1](https://github.com/PedroVictorCoding/sail-framework/issues/1), with 33 detailed implementation issues. See [the dependency-linked assistant roadmap](docs/ASSISTANT_ROADMAP.md) for delivery order, product acceptance, and links to each request.

The [MoE/local execution research plan](docs/research/MOE_LOCAL_OPTIMIZATION.md)
translates the user-supplied Ludziejewski thesis and additional primary research
into ordered reproduction tasks PERF-MOE-00 through PERF-MOE-08. It covers actual
token dispatch, component learning-rate schedules, expert granularity, memory-aware
SSM/MoE design, CPU/GPU placement, caching, quantization and speculative decoding.
Results are pending; implementation submissions and QA remain in [LLM_HANDOFF.md](LLM_HANDOFF.md).

| Priority | Item | Acceptance criteria |
| --- | --- | --- |
| P0 · Opt-in experiment | Work-efficient selective scan | Pair-reduction scan with linear tensor work; compare forward outputs, both input gradients, recurrent decoding, and complete training-step timings against the corrected chunked baseline. CPU results documented in docs/PERFORMANCE.md; GPU measurements pending. |
| P1 · Specification | Dropless Block-Sparse MoE Dispatch | Formulate dynamic expert dispatch as token gather-GEMM-scatter (MegaBlocks); eliminate padding overhead and reduce row evaluations to strictly $K \cdot N$ instead of $E \cdot N$. See [Specification](docs/research/BLOCK_SPARSE_MOE_DISPATCH.md). |
| P1 · Specification | State Space Duality (SSD) & Selective Recomputation | Replace elementwise associative scan with chunkwise Tensor Core GEMMs ($Y = (L \circ CB^\top)X$) via Mamba-2 duality; recompute intermediate $h_t$ states on backward pass to cut VRAM traffic by 75%. See [Specification](docs/research/STATE_SPACE_DUALITY_AND_RECOMPUTATION.md). |
| P2 · Specification | Chunkwise Gated Linear Attention (GLA) | Replace quadratic softmax attention layers with hardware-efficient decay-aware chunkwise linear attention (Yang et al. 2023), preserving subquadratic scaling and universal $O(1)$ constant-memory recurrent decoding. See [Specification](docs/research/CHUNKWISE_GATED_LINEAR_ATTENTION.md). |
| P0 · Prototype | Tool-Calling Local Knowledge Lookup | Native tool control tokens (<tool_call>...</tool_call>) in BPE engine with stateful recurrence pause/resume and local verified knowledge-base lookup. |
| P0 · Implemented | Predictive Output Entropy Heuristic | Per-token predictive Shannon entropy calculation in [0, 1] demonstrating confidence-based tool invocation thresholds. (Prototype demonstration; not calibrated factual hallucination detection). |
| P0 · Prototype | Live Audio Capture & Harmonic Tones | Real-time microphone capture with cpal, energy-based VAD turn detection, and harmonic tone synthesizer. (Prototype demonstration; not neural TTS or generalized cloud ASR). |
| P0 · Implemented | Native BPE Subword Engine | Pure Rust byte-level fallback BPE tokenizer with merge ranking tables, greedy encode/decode, and zero <unk> out-of-vocabulary tokens. Tested end-to-end. |
| P1 · Prototype | Direct Hardware Kernels (CubeCL) | Experimental research kernel definition; slice dispatcher runs on CPU fallback. Not integrated into production Burn autodiff graph. |
| P0 · Implemented | Sparse Expert Streaming (MoE) | Zero-copy memory-mapped parameter streaming for MoE experts via memmap2, decoupling parameter banks from VRAM limits. Numerically verified against dense reference. |
| P0 · Implemented | Single-File Native Artifacts | Single-command model compilation into standalone .sea files and standalone runner CLI (`sail export` / `sail run`) cross-compiling to Linux, macOS, Windows, iOS, and browser WebAssembly. |
| P0 · Implemented | Dynamic Data Streaming & Continual Learning | Thread-safe append-only buffer (`StreamDataset`) with directory watcher (`data/incoming/`), Experience Replay anti-forgetting sampling, transactional SQLite deduplication, and atomic live updates to `model.sea`. CLI command: `sail stream-train`. |
| P0 · Implemented | Resumable bulk dataset prefetch | HF shard iterators, independent pinned-revision cursors, bounded text spool, checkpoint-backed cleanup, transactional SQLite deduplication across restarts and sources, split contamination guards, retry/backoff. See docs/HF_STREAMING.md. |
| P0 · Implemented | Concurrent High-Integrity Inference Engine | Snapshot-isolated parallel inference engine (`sail::engine::Engine`) serving multiple threads simultaneously with `RwLock`-synchronized generational pointer swapping and hot-reloading from disk. CLI command: `sail engine`. |
| P0 · Implemented | Omnimodal Unified Backbone (Audio & Video) | Direct 1D causal conv projection for streaming audio and 2D patch projection for multi-frame video, mapping raw perceptual data into the shared $d_{\text{model}}$ latent space without separate pipeline models. Tested end-to-end. |
| P0 · Prototype | Redox OS Microkernel Scheme Simulation (sail:) | Userspace simulation of Redox microkernel C packet protocol (`RedoxPacket` for `SYS_OPEN`, `SYS_READ`, `SYS_WRITE`, `SYS_CLOSE`) exposing `sail:` IPC scheme channels. CLI command: `sail daemon`. |
| P1 | Context-Directed Dynamic Curriculum Feeding | Steer online streaming data according to active developer workflow (e.g. inject high-priority code/Rust corpora when coding, or science/arXiv preprints when researching), balanced against stratified experience replay to preserve general capabilities. |
| P2 | Scheduled replay and consolidation | Human-configured training schedules and replay of externally sourced data; evaluate retention on held-out tasks. The model must not change its architecture, objectives, training policy, or deployment. |
| P0 · Implemented | Live nodes and weighted connections | Separate dashboard view reads real snapshots, exposes all captured tensors and scalar indices, supports channel paging, and shows exact weights and changes. Missing or sampled data is explicit. Browser appearance still needs visual verification. |
| P1 | Training reliability | Validate configuration before allocation; report failures clearly; detect non-finite losses and gradients with actionable diagnostics. |
| P1 | Checkpoint resume | Restore model, optimizer, schedule, and random state; distinguish resume from a fresh run; verify continuity against uninterrupted training. |
| P1 | Efficient telemetry | Measure capture overhead and browser memory on production configurations; support bounded history and efficient delivery without silently omitting weights. |
| P2 | Deeper activity inspection | Capture attention, SSM state, and expert routing observations with clear token/batch coordinates and configurable overhead. |
| P2 | Experiment comparison | Compare validation loss, throughput, parameter counts, seeds, and configurations using reproducible evaluation. |


## Moving into an OS
We should eventually port this text into a fork of the Rust Unix-like microkernel-based operating system: Redox OS

Integration must use human-defined permissions and preserve the foundational alignment rule. The model must not control its own training, modify itself, or deploy replacements.

The goal is to reduce human physical input, so understanding linux commands or having a way to retrace/find approppriate commands


## Deferred research: brain-inspired circuits

Status: proposed, unscheduled. Biological region or transmitter behavior requires
defined computation and learning rules; names and display labels do not implement it.

### R1 — Hippocampus-inspired episodic memory

- Hypothesis: bounded, rapidly writable memory improves delayed recall.
- Prototype: fixed-capacity key/value memory with learned read/write gates and
  explicit replacement/reset policies. Initially separate memory contents from
  optimizer-trained weights.
- Acceptance: compare with the existing model and a parameter/compute-matched
  baseline on held-out delayed recall and language-model validation loss;
  report latency, memory use, and multiple seeds.
- Required checks: prevent future-token access, validation leakage, and memory
  mixing across unrelated sequences; bound storage and computation.

### R2 — PFC-inspired context controller

- Hypothesis: persistent context and learned gates improve memory access and
  expert selection.
- Prototype: small recurrent controller with bounded gates, explicit resets, and
  defined interfaces to episodic memory and the existing SSM/MoE backbone.
- Acceptance: ablate controller and memory independently; demonstrate benefits
  over existing routing and recurrent state at comparable compute.

### R3 — Excitatory, inhibitory, and modulatory populations

- Define population roles separately from receptor-like response dynamics and
  circuit connectivity. Nicotinic/muscarinic response settings are not mutually
  exclusive transmitter-defined neuron types.
- Explore signed outgoing connections, fast/slow gain controls, and bounded
  modulation of memory retention or plasticity one mechanism at a time.
- Define the modulation objective explicitly; next-token loss alone is not a
  biological dopamine reward signal.
- Acceptance: finite activations and gradients, bounded state, tested interaction
  with AdamW, and reproducible benefits over ablated baselines.

### R4 — Circuit observability

- Visualize measured memory writes, retrieval strengths, controller gates,
  population activity, and modulation signals.
- Distinguish observed values from configuration and biological analogy.
- Depends on selecting and validating R1–R3.

Research promotion gate: a reproducible prototype, matched baselines, held-out
evaluation, and documented stability/compute costs before default integration.
