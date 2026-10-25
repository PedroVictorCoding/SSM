# Product backlog

Foundational constraint: [ALIGNMENT.md](ALIGNMENT.md). **No autonomous recursive self-improvement, ever; no exceptions.** All learning features must use context-dependent external data within a human-defined training pipeline.

Current direction: prioritize measured training/inference performance and algorithmic
experiments in the Rust/Burn SSM/MoE model. GUI expansion is secondary. Brain-inspired
architectures remain deferred research proposals.

## Current product

| Priority | Item | Acceptance criteria |
| --- | --- | --- |
| P0 · Opt-in experiment | Work-efficient selective scan | Pair-reduction scan with linear tensor work; compare forward outputs, both input gradients, recurrent decoding, and complete training-step timings against the corrected chunked baseline. CPU results documented in docs/PERFORMANCE.md; GPU measurements pending. |
| P1 | Actual sparse MoE execution | Dispatch only routed tokens to each expert; compare outputs and gradients with the current all-experts reference. Include dispatch overhead, imbalance, dropped-token behavior, CPU/GPU timings, and validation quality. |
| P1 | Fused scan and recomputation | Explore backend-specific fused forward/backward and recomputation to reduce materialized state traffic. Benchmark memory and end-to-end step time; require gradient checks before enabling. |
| P0 · Implemented | Tool-Calling Tokenizer & Engine | Native tool control tokens (<tool_call>...</tool_call>) in BPE engine with stateful recurrence pause/resume and live web search verification. |
| P0 · Implemented | Calibrated Predictive Entropy Trigger | Per-token predictive Shannon entropy calculation in [0, 1] to detect hallucination risks and automatically trigger internet verification when confidence dips. |
| P0 · Implemented | Live Streaming Audio Loop | Real-time microphone capture and speaker playback with cpal, energy-based VAD turn detection, and streaming neural TTS. CLI command: `SSM voice`. |
| P0 · Implemented | Native BPE Subword Engine | Pure Rust byte-level fallback BPE tokenizer with merge ranking tables, greedy encode/decode, and zero <unk> out-of-vocabulary tokens. Tested end-to-end. |
| P1 · Prototype | Direct Hardware Kernels (CubeCL) | Kernel definition exists; the slice dispatcher still uses its CPU fallback. Integrate a verified GPU launch and autodiff backward before claiming accelerated end-to-end training from this kernel. |
| P0 · Implemented | Sparse Expert Streaming (MoE) | Zero-copy memory-mapped parameter streaming for MoE experts via memmap2, decoupling parameter banks from VRAM limits. Numerically verified against dense reference. |
| P0 · Implemented | Single-File Native Artifacts | Single-command model compilation into standalone .sea files and standalone runner CLI (`sail export` / `sail run`) cross-compiling to Linux, macOS, Windows, iOS, and browser WebAssembly. |
| P0 · Implemented | Dynamic Data Streaming & Continual Learning | Thread-safe append-only buffer (`StreamDataset`) with directory watcher (`data/incoming/`), Experience Replay anti-forgetting sampling, and atomic live updates to `model.sea`. CLI command: `sail stream-train`. |
| P0 · Implemented | Resumable bulk dataset prefetch | HF shard iterators, independent pinned-revision cursors, bounded text spool, checkpoint-backed cleanup, retry/backoff and explicit failure without fallback samples. See docs/HF_STREAMING.md. |
| P1 | Context-Directed Dynamic Curriculum Feeding | Steer online streaming data according to active developer workflow (e.g. inject high-priority code/Rust corpora when coding, or science/arXiv preprints when researching), balanced against stratified experience replay to preserve general capabilities. |
| P1 | Omnimodal Unified Backbone (Audio & Video) | Direct 1D causal conv projection for streaming audio and 2D patch projection for multi-frame video, mapping raw perceptual data into the shared $d_{\text{model}}$ latent space without separate pipeline models. |
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
