# Hardware, Model, and Format Support Matrix

Target Environment: Native Windows 11 / WSL2 with AMD Radeon RX 7800 XT (RDNA 3, Navi 32, 16 GB GDDR6) and AMD Ryzen 5 7600X host CPU (6 cores / 12 logical processors).

---

## 1. Model Container Format Support and Implementation Status

Format identification and container header validation are enforced by `validate_model_format` in `src/runtime/adapter.rs`. Container header acceptance does not imply full weight/tensor importation.

| Format | Magic Bytes / Header | Implementation Status | Native Runtime Execution Path | Validation & Safety Guarantees |
| :--- | :--- | :--- | :--- | :--- |
| **SAIL Encapsulated Artifact (`.sea`)** | `SEAPACK1` (or legacy `SSMPACK1`) | **Fully Implemented & Benchmarked** | Native Burn CPU (`ndarray`) & GPU (`wgpu` / Vulkan / Direct3D 12) via `NativeSailAdapter` | Full 24-byte `SeaHeader` validation (`config_len`, `tokenizer_len`, `weights_len`), payload length verification against file size, complete tensor deserialization. |
| **GGUF v2/v3 (`.gguf`)** | `0x46554747` (`GGUF`) | **Container Validated; Test Double in Runtime** | Header inspection implemented; backend runtime currently exercises adapter protocol via test double (`PretrainedLocalAdapter`) | Validates 4-byte magic, version (v2 or v3 only), non-zero 64-bit tensor count and metadata KV count. Rejects truncated/empty files. Real inference engine tensor loading remains an open requirement (QA-02). |
| **SafeTensors (`.safetensors`)** | JSON metadata header | **Planned Format** | Not currently implemented in runtime adapter pipeline | Deserialization pipeline not yet integrated into `ModelBackendAdapter`. Awaiting direct tensor importer. |
| **Raw PyTorch Pickles (`.pt`, `.bin`)** | `0x80` pickle opcodes (`\x80\x02`..`\x80\x05`) | **PROHIBITED** | N/A | **Explicitly rejected** at file inspection without loading to prevent arbitrary code execution vulnerabilities. |
| **ZIP Archives (`.zip`, `.docx`, `.jar`)** | `PK\x03\x04` / `PK` | **PROHIBITED** | N/A | **Explicitly rejected**; arbitrary ZIP archives are prohibited from masquerading as binary models. |
| **Raw JSON Configs (`.json`)** | `{` text | **REJECTED AS MODEL BINARY** | N/A | **Explicitly rejected** from model loading pathways; configs must be packed in `.sea` artifacts. |
| **TensorFlow GraphDef (`.pb`)** | Protocol Buffers | **UNSUPPORTED** | N/A | Not supported. |

---

## 2. Hardware Execution & Benchmark Reality on AMD RX 7800 XT

All figures below are direct measurements from the archived benchmark suite in [`docs/benchmarks/rx7800xt_training_and_inference.json`](benchmarks/rx7800xt_training_and_inference.json), executed on Windows NT 10.0.26200 with an AMD Ryzen 5 7600X (12 logical processors) and AMD Radeon RX 7800 XT (driver: AMD proprietary 26.3.1 LLPC).

### A. Autoregressive Single-Token Decode (Batch Size = 1)
Evaluated on a small hybrid SSM model (380,033 total parameters, 158,849 active parameters, layers `[M A M]`, state dimension 8, prompt: `"ROMEO:"`, generating 128 new tokens with greedy temperature 0):

| Subsystem | Backend | Target Hardware | Prefill Latency | Measured Decode Throughput | Dedicated GPU Memory | Output Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mamba / Hybrid SSM (380k)** | Burn `ndarray` (CPU) | AMD Ryzen 5 7600X (6C/12T) | **3.3 ms** | **851.0 tok/s** | 0 B (CPU RAM) | Bit-exact text output match. |
| **Mamba / Hybrid SSM (380k)** | Burn `wgpu` (Vulkan) | AMD Radeon RX 7800 XT (16 GB) | **444.5 ms** | **98.5 tok/s** | 781.5 MiB (peak sampled allocation) | Identical text output. Kernel launch and dispatch overhead dominates per-token step. |
| **Transformer (Full KV)** | Burn `ndarray` (CPU) | AMD Ryzen 5 7600X (6C/12T) | **~3.5 ms** | **485 to 820 tok/s** | 0 B (CPU RAM) | Latency scales quadratically with sequence length $O(L)$ without windowing. |
| **Local Pretrained GGUF** | `PretrainedLocalAdapter` | CPU / GPU | *Unmeasured (Test Double)* | *Unmeasured (Protocol Only)* | *Unmeasured* | Requires real external runner integration (QA-02). |

### B. Training Throughput on AMD RX 7800 XT (B8, L64, SeqLen 64)

| Configuration | Backend | Warm Mean Throughput | End-to-End Throughput | Peak Dedicated Memory | Speedup Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU Baseline (380k)** | `ndarray` (CPU) | **3,827 tok/s** (median) | **2,110 tok/s** | 0 B | Reference baseline training throughput. |
| **GPU Baseline (380k)** | `wgpu` (Vulkan) | **3,792 tok/s** | **2,110 tok/s** | 629.8 MiB | Host-synchronized clipping. |
| **GPU On-Device Fused Norm (380k)** | `wgpu` (Vulkan) | **13,298 tok/s** | **8,408 tok/s** | 647.8 MiB | **3.47× speedup** over baseline host-synchronized norm. |
| **GPU Streaming Trainer (380k)** | `wgpu` (Vulkan) | **12,631 tok/s** | **7,834 tok/s** | 647.8 MiB | Live directory watcher + continual experience replay. |
| **GPU Larger Workload (517k, B32/L128)** | `wgpu` (Vulkan) | **26,646 tok/s** | **23,754 tok/s** | 3,794.0 MiB | Saturated compute on 4-layer `[M A M M]` architecture. |

### Empirical Takeaway and Workload Boundaries
- **Workload-Specific Speedup**: On tiny models ($\sim 380\text{k}$ parameters) during sequential single-token autoregressive decoding, CPU decode is **8.64× faster** (851 tok/s vs 98.5 tok/s) and prefill is **134× faster** (3.3 ms vs 444.5 ms) than Vulkan GPU. This is due to driver submission latency, shader pipeline binding, and PCIe roundtrips on small single-token tensors.
- **Do Not Extrapolate Universally**: This inversion applies specifically to small hybrid models during single-sequence autoregressive decoding. In batched training, the RX 7800 XT delivers up to **26,646 tok/s** (**7.8× faster** than CPU baseline), where large tensor operations fully saturate GPU compute units.
- **Separation of Test Doubles from Benchmarks**: Throughput numbers for GGUF architectures are not published because local pretrained execution currently operates through protocol test doubles (`PretrainedLocalAdapter`). Performance numbers will be benchmarked only after native tensor runtime integration (QA-02) is verified.

---

## 3. Interchangeable Adapter Interface

Both native Sail models and local pretrained models implement the common `ModelBackendAdapter` trait (`src/runtime/adapter.rs`):
- **Recurrent State Continuity**: Preserves `HybridState<B>`, prompt history, and RNG seeds across conversational turns.
- **Session Isolation**: Session states are keyed by `session_id`. Calling `reset_session()` completely purges session memory without leaking into concurrent sessions.
- **Cooperative Cancellation**: Every autoregressive generation step inspects `cancellation_token.is_cancelled()`, promptly committing the generated prefix and halting.
- **Streaming UTF-8 Handling**: `StreamingUtf8Decoder` buffers partial multibyte code points across chunk boundaries, eliminating Unicode replacement character corruption.
- **Typed Tool Proposals**: Tool call syntax `<tool_call>name(args)</tool_call>` is intercepted and converted into typed `ToolProposal` structs.
