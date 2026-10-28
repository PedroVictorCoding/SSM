# SAIL Framework Capability Matrix

This table distinguishes the exact engineering state of every subsystem in SAIL across four rigorous tiers:
1. **Implemented**: Code exists and compiles into the binary.
2. **Tested**: Verified with automated unit or integration tests in the test suite.
3. **Benchmarked**: Measured empirically with quantitative timing, memory, or quality metrics.
4. **Planned / Deferred**: Proposed architectural research or future backlog goals.

---

| Subsystem / Feature | Implemented | Tested | Benchmarked | Status & Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **Selective State Space Model (Mamba)** | Yes | Yes | Yes | Bit-exact match between sequential and parallel prefill (`tests/test_mamba_step.rs`). Constant recurrent decode latency measured (~4.8 ms/tok CPU). |
| **Work-Efficient Associative Tree Scan** | Yes | Yes | Yes | Verified in `tests/test_scan_tree.rs`. Benchmarked at **1.76× throughput** over chunked baseline on CPU (107.8 ms vs 189.9 ms). |
| **Sliding-Window Multi-Head Attention** | Yes | Yes | Yes | Strictly $O(1)$ constant memory during generation when `attn_window > 0`. Causal $O(L)$ KV cache growth when `attn_window = 0`. |
| **SwiGLU Mixture-of-Experts (MoE)** | Yes | Yes | Yes | Auxiliary load balancing loss $L_{\text{aux}}$ prevents expert collapse. Active routing parameter accounting. |
| **Sparse Expert Dispatch** | Yes | Yes | Yes | Skips unrouted experts (`sparse_dispatch: true`). Verified bit-exact match on forward output (<1e-6) and backward gradients (<1e-6) in `tests/test_moe_sparse.rs`. Measured **1.47× forward speedup** on 16 experts / Top-1. |
| **Mmap Expert Decoupling** | Yes | Yes | No | Zero-copy `memmap2` weight streaming in `src/model/mmap_moe.rs`. Verified numerically against memory weights; disk I/O latency unmeasured on GPU. |
| **Native Byte-Fallback BPE Tokenizer** | Yes | Yes | Yes | Reversible byte-level fallback with zero OOV tokens (`tests/test_bpe.rs`). Compression ratio ~3.2× on Shakespeare corpus. |
| **Standalone .sea Binary Export & Runner** | Yes | Yes | Yes | Single-file serialization of weights, config, tokenizer (`sail export`, `sail run`). Cross-platform standalone inference verified. |
| **Native Model Inference Adapter** | Yes | Yes | Yes | `NativeSailAdapter`: session state continuity, bit-exact parity with reference scan, $O(1)$ recurrent decoding without prompt re-prefill (`tests/test_backend_adapters.rs`). |
| **Model Container Format Validator** | Yes | Yes | No | `validate_model_format`: strict header validation for `SEAPACK1`, `SSMPACK1`, and GGUF v2/v3. Explicitly rejects ZIP, raw JSON, and Python pickles. |
| **Pretrained Local Model Adapter** | Yes | Yes | No | `PretrainedLocalAdapter`: validates GGUF container headers, reads metadata, and executes `llama-cli` in supervised process groups with token streaming. Fails closed with actionable error if runner is absent; never fabricates fake answers. Deterministic protocol tests use explicit `MockModelAdapter`. |
| **Headless Camera Test Double** | Test Double | Yes | No | `HeadlessCameraProvider`: explicit test double generating controlled synthetic pixel buffers for deterministic lifecycle, circular buffer bound, and motion math verification (`tests/test_camera_capture.rs`). |
| **Native OS Camera Provider** | Implemented | Yes | No | `OsCameraProvider`: probes Linux V4L2 device nodes (`/sys/class/video4linux`, `/dev/video*`) and Windows interfaces. Strict fail-closed contract returning typed `CameraError` on missing hardware or permission denial; never fabricates fake frames. Physical streaming pending attached hardware. |
| **Camera Motion Heuristic** | Yes | Yes | Yes | Subsampled pixel-difference ratio (`analyze_temporal_change`) for ROI burst triggering and keyframe downsampling (<15ms latency, <0.02ms overhead). Fast frame-filtering heuristic, strictly distinguished from semantic VQA. |
| **Streaming Text Trainer & Replay** | Yes | Yes | Yes | Non-blocking directory watcher (`data/incoming/`), Experience Replay sampling, frozen validation, deterministic resume (`tests/test_continual_learning.rs`). |
| **Transactional SQLite Deduplication** | Yes | Yes | Yes | Persists SHA-256 hashes across restarts in `dedup.db`. Prevents duplicate chunk publication. Cross-split contamination guards reject validation leaks. |
| **Omnimodal Any-to-Any Backbone** | Yes | Yes | No | Unified latent space for 1D audio conv, 2D image patch projection, and text (`sail stream-train --multimodal`). Tensor plumbing tested end-to-end; downstream semantic evaluation (WER, VQA) pending large pretraining. |
| **Unified Checkpoint Receipts & Cleanup** | Yes | Yes | Yes | Emits `ingestion.json` committing spool IDs. `Spool.reclaim()` successfully purges consumed chunks across text and multimodal feeds (`test_multimodal_trainer_roundtrip_with_compiled_binary`). |
| **Concurrent Engine (`sail engine`)** | Yes | Yes | Yes | Multi-threaded inference using `RwLock` generational snapshot isolation (`Arc<ModelGeneration>`) with automatic file reload (`tests/test_concurrent_engine.rs`). |
| **Telemetry & Live Observatory** | Yes | Yes | Yes | Zero-allocation single-pass stats and buffered JSON streaming (`neural.json`). Browser GUI on port 8787. Overhead measured. |
| **Predictive Output Entropy Check** | Yes | Yes | No | Per-token Shannon entropy in $[0, 1]$. Prototype heuristic for local tool dispatch; not a calibrated factual hallucination detector. |
| **Voice Interaction Prototype (`sail voice`)** | Yes | Yes | No | Hardware audio capture via `cpal`, energy-based VAD turn detection, and harmonic tone synthesizer. Offline prototype; not neural TTS or cloud ASR. |
| **Tool Calling Knowledge Base (`sail tool`)** | Yes | Yes | No | Local verified knowledge-base keyword prototype demonstrating control token pauses (`<tool_call>`). Not live Internet browsing. |
| **CubeCL Associative Scan Kernel** | Prototype | Yes | No | Experimental CubeCL research kernel in `src/kernel/scan.rs`. Slice dispatcher runs on CPU fallback; not integrated into Burn autodiff graph. |
| **Redox OS Microkernel Daemon (`sail daemon`)** | Prototype | Yes | No | Userspace simulation of Redox microkernel C packet protocol (`RedoxPacket` for `SYS_OPEN`, `SYS_READ`, etc.) and IPC scheme channels (`sail:`). |
| **Decentralized P2P Training (SEQ/ACK)** | Planned | No | No | Peer-to-peer dataset streaming and segment gradient consolidation. Backlog research proposal. |
| **Context-Directed Dynamic Curriculum** | Planned | No | No | Workflow context sensing to steer topic feeds. Backlog research proposal. |

---

## Strict Alignment Boundaries (Preserved Permanently)

- **No autonomous recursive self-improvement, ever, with no exceptions.**
- The model must never autonomously modify its own code, architecture, optimizer, training objectives, permissions, or alignment rules.
- Model-generated answers and self-assessments must never automatically become training data or rewards for its own improvement.
- Continuous learning remains strictly within human-defined pipelines on externally sourced, human-approved data.

