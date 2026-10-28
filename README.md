# SAIL: State Space, Mixture of Experts & Active Parameters Hybrid LM in Rust (Burn)

Model File: .sea (State space, mixture of Experts & Active parameters)

A pure Rust implementation of Selective State Space Models (Mamba), Transformer Causal Attention, and Mixture of Experts (MoE) with dynamic active parameters per token, built on the [Burn](https://burn.dev) deep learning framework.

## Foundational alignment rule

**No autonomous recursive self-improvement, ever. This rule is permanent and has no exceptions.** SAIL learns through continuous, context-dependent training on data from Hugging Face and other human-approved external sources, within a human-defined training pipeline. The model must never control its own code, architecture, training policy, permissions, or deployment. See [ALIGNMENT.md](ALIGNMENT.md) for the binding project boundaries.


---

Ideas:
- [ ] Decentralized training, SEQ/ACK p2p infrastructure, where an user trains a segment, and then that segment is merged into the model, then everyone has access to a model train in a greater scale. (user downloads some serve established dataset -> user train segment -> user uploads trained segment -> server merges <- seamlessly)

---

## Key Architecture & Features

### 1. Constant-Memory $O(1)$ Stateful Inference
- **`step(token, state)` API**: During autoregressive text generation, the recurrent hidden state $h_t$ is preserved across steps.
- **Constant decode latency**: Per-token decode time is identical at $L=64$, $L=128$, and $L=256$ (~4.8 ms/token on CPU), avoiding the $O(L^2)$ quadratic cost of re-processing context.
- **Memory Boundedness**: In pure Mamba SSM layers, memory is strictly $O(1)$ constant ($\text{d\_state} \times \text{d\_inner}$). When attention layers are enabled, memory remains strictly $O(1)$ bounded *if and only if* sliding window attention is configured (`--attn-window W > 0`); full causal attention (`--attn-window 0`) grows the KV cache linearly $O(L)$.
- Verified bit-exact match against full-sequence parallel prefill (`outputs identical: true`).

### 2. Full Mamba Selective SSM Layer ([`mamba.rs`](src/model/mamba.rs))
- **Input-dependent discretization**: $\Delta_t$, $B_t$, and $C_t$ are dynamically projected from inputs.
- **Causal Depthwise Convolution**: Short causal 1D convolution window with SiLU non-linearity.
- **S4D-Real Initialization**: Diagonal matrix $A_{i,j} = -(j + 1)$ initialized via log-parameterization with negative exponentiation to guarantee strictly contractive dynamics.
- **Low-Rank $\Delta$ Projection**: Configurable `dt_rank` (defaults to $\lceil d_{\text{model}} / 16 \rceil$) with inverse-softplus initialization.
- **Skip Connection**: Learnable parameter $D$.

### 3. Jamba Hybrid LM Stack ([`hybrid.rs`](src/model/hybrid.rs))
- Interleaved Mamba blocks with occasional Multi-Head Causal Attention layers (e.g., `[M M A M]`), combining linear recurrent inference with needle-in-a-haystack retrieval.
- Attention layers support optional sliding window attention with bounded Key-Value (KV) cache.
- RMSNorm pre-normalization throughout.

### 4. Mixture of Experts (MoE) & Dynamic Active Parameters ([`moe.rs`](src/model/moe.rs))
- Sparse expert routing replacing standard dense MLPs with a bank of $E$ SwiGLU networks.
- **Top-K & Dynamic Thresholding**:
  - Fixed Top-K routing (e.g. 1 out of 4, or 2 out of 8 active experts).
  - Dynamic threshold routing ($\tau > 0$), where tokens dynamically activate experts exceeding confidence threshold $\tau$, giving variable compute per token.
- **Auxiliary Load-Balancing Loss**:
  $$L_{\text{aux}} = E \sum_{e=1}^E f_e \cdot P_e$$
  Prevents expert collapse during training.
- **Parameter Capacity vs Compute**:
  - `active_params_per_token()` accurately tracks the standard theoretical routing parameter footprint (router + $K$ experts).
  - Batched training evaluates all experts in parallel with sparse gating masks for tensor vectorization; physical on-demand memory streaming is implemented in the experimental [`mmap_moe.rs`](src/model/mmap_moe.rs) path.

### 5. Associative Parallel Scan ([`scan.rs`](src/model/scan.rs))
- Replaces sequential loops with a chunked parallel prefix scan using the associative operator:
  $$(A_1, B_1) \circ (A_2, B_2) = (A_1 \cdot A_2, A_2 \cdot B_1 + B_2)$$
- Evaluates in $O(\log L)$ sequential depth across parallel threads or GPU warps.
- Autodiff differentiable with reduced activation memory.

### 6. Hardware Backends ([`backend.rs`](src/backend.rs))
- `ndarray` (CPU, default)
- `wgpu` (Vulkan / Metal / DX12)
- `cuda` (NVIDIA CUDA via `burn-cuda`)

### 7. Native BPE Subword Engine ([`src/data/bpe.rs`](src/data/bpe.rs))
- Pure Rust Byte-Pair Encoding subword tokenizer with byte-level fallback.
- Maps raw bytes $0..255$ directly to dedicated base token IDs ($2..257$), guaranteeing 100% vocabulary coverage with zero `<unk>` out-of-vocabulary tokens.
- Self-contained BPE trainer and fast greedy pair-merge encoder/decoder with ranking index tables.
- Train models with `--bpe --vocab-size <N>`.

### 8. Direct Hardware Kernels in CubeCL ([`src/kernel/scan.rs`](src/kernel/scan.rs))
- Experimental associative scan kernel definition using CubeCL shared memory.
- The public slice dispatcher currently uses its CPU fallback. This kernel is not
  integrated into the autodiff training path; current GPU training uses Burn tensor operations.

### 9. Sparse Expert Streaming ([`src/model/mmap_moe.rs`](src/model/mmap_moe.rs))
- Decouples parameter capacity from VRAM / system RAM via memory-mapped weight streaming (`memmap2`).
- Stores expert networks contiguously in an aligned binary file (`experts.bin`).
- Router dynamically selects active experts for tokens in each batch, streaming *only* the top-$K$ active experts on-demand into compute tensors while keeping inactive experts unallocated.

### 10. Single-File Native Artifacts & Standalone Runner ([`src/export.rs`](src/export.rs))
- Single-command model compilation into standalone `.sea` binary artifacts containing model metadata, embedded tokenizer, and full-precision weights.
- Run native standalone inference directly without external config files, Python, or PyTorch:
  ```bash
  sail export --run runs/shakespeare --out model.sea
  sail run --model model.sea --prompt "ROMEO:" --tokens 300
  ```
- Ready for cross-compilation to Linux, macOS, Windows, iOS, and WebAssembly in browsers.

### 11. Dynamic Data Streaming & Continual Online Learning ([`src/data/stream.rs`](src/data/stream.rs))
- Ingests incoming datasets on the fly without stopping training.
- **Experience Replay Buffer**: Mixes new tokens with historical tokens (e.g., 30% replay ratio) to prevent catastrophic forgetting.
- **Live Atomic Updates**: Regularly and atomically refreshes `model.sea`, enabling concurrent model querying and testing (`sail run --model runs/live_growing/model.sea`) while training continues uninterrupted in the background.

---

## Empirical Benchmarks

For native AMD device selection, reproducible profiling, checkpoint recovery, and
PowerShell commands, see [GPU training and recovery](docs/GPU_TRAINING.md).
[Measured RX 7800 XT results](docs/GPU_RESULTS.md): on-device gradient clipping
improved warm training throughput 3.47×; a larger batch reached 26,646 tok/s.
Full snapshots every update have substantial overhead, and CPU is faster for the
tested small-model single-sequence inference workload.

### Associative Parallel Scan vs. Sequential Scan
Tested on `[Batch=8, D_inner=256, D_state=16]`, chunk size 64:

| Sequence Length $L$ | Sequential Scan | Chunked Parallel Scan | Speed-up |
| :---: | :---: | :---: | :---: |
| **64** | 1,014.9 ms | 259.0 ms | **3.9x** |
| **128** | 4,287.9 ms | 594.8 ms | **7.2x** |
| **256** | 37,531.8 ms | 3,135.4 ms | **12.0x** |

### Stateful $O(1)$ vs. Naive $O(L^2)$ Autoregressive Decoding
Tested on trained hybrid model (`[M A M]`, 386k parameters):

| Context Length $L$ | Stateful ms/tok | Naive ms/tok | Speed-up | Memory / State Size |
| :---: | :---: | :---: | :---: | :---: |
| **64** | **4.84 ms** | 20.05 ms | **4.1x** | 14,976 floats |
| **128** | **4.88 ms** | 30.53 ms | **6.3x** | 23,168 floats |
| **256** | **4.86 ms** | 51.66 ms | **10.6x** | 39,552 floats |

Notice that stateful decoding maintains constant ~4.8 ms latency per token regardless of context length.

---

## Usage

### Install the `sail` system command

SAIL is a normal Rust binary crate named `sail`. After cloning the repository,
install it once and use `sail` from any directory.

Linux, WSL, or macOS:

```bash
./scripts/install.sh
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install.ps1
```

Both installers ask which compute backend to build: portable CPU, WGPU for common
GPUs (including AMD), or CUDA for NVIDIA. They use `cargo install --path` and place
the executable in Cargo's normal binary directory (`~/.cargo/bin` on Linux/macOS,
`%USERPROFILE%\.cargo\bin` on Windows). Rust installers normally add that directory
to `PATH`; otherwise add it once and open a new terminal.

The equivalent noninteractive installation is:

```bash
cargo install --path . --locked --force --no-default-features --features wgpu
```

After installation, run `sail` with no arguments. It opens a terminal wizard for
training, continual stream training, inference, export, and generation. The wizard
validates entries, shows the equivalent reproducible command, and asks before it
starts anything. `sail setup` or `sail configure` opens the same wizard explicitly.
Existing command-line flags remain available for scripts and repeatable benchmarks.

If input is redirected or the command runs unattended, bare `sail` prints help
instead of waiting for answers. The wizard never turns model output into training
data and cannot authorize model-controlled training.

Training performance experiments and reproducible CPU results are documented in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md). The opt-in `--work-efficient-scan`
uses a pair-reduction scan with linear elementwise work. On the measured
4-layer hybrid/MoE configuration, median complete training-step time fell from
187.3 ms to 117.1 ms (**1.60×**); GPU performance is unmeasured.

Add `--work-efficient-scan` to `train` (including `--resume`) to try it. It
overrides `--scan-chunk`. Existing configurations keep the chunked scan by default.

The trainer stops on non-finite loss or gradients and rejects non-finite weights
at startup and before checkpoint saves. Numerical failures write `failure.json`
and a failed dashboard status. Recover from verified finite weights, rather than
resuming a checkpoint saved after NaNs. Missing optimizer files mean fresh AdamW
moments, not an exact continuation.

### 1. Run Tests
```bash
cargo test --release
```

### 2. Train a Hybrid MoE Language Model on TinyShakespeare
```bash
cargo run --release -- train \
  --data data/tinyshakespeare.txt \
  --out runs/moe_hybrid \
  --steps 1000 \
  --batch-size 16 \
  --seq-len 64 \
  --d-model 64 \
  --n-layers 4 \
  --attn-layers 2 \
  --mlp-ratio 2 \
  --moe-experts 4 \
  --moe-top-k 1
```

### 3. Continue Training Where You Left Off (`--resume`)
Full training state (model weights, AdamW optimizer velocity/moments, step count, and cumulative training time) is automatically saved on every evaluation and exit. To continue training seamlessly:
```bash
# Continue from step 1000 to step 2500 without resetting optimizer momentum:
cargo run --release -- train \
  --out runs/moe_hybrid \
  --steps 2500 \
  --resume
```

### 4. Fast $O(1)$ Text Generation
```bash
cargo run --release -- generate \
  --run runs/moe_hybrid \
  --prompt "ROMEO:" \
  --tokens 300 \
  --temperature 0.8
```

### 4. Verify Equivalence (Stateful vs Naive)
```bash
cargo run --release -- generate \
  --run runs/moe_hybrid \
  --prompt "ROMEO:" \
  --tokens 200 \
  --compare
```

### 5. Run Scalability Benchmarks
```bash
cargo run --release -- bench --seq-lens 64,128,256 --run runs/moe_hybrid
```

### 6. Continual Online Streaming Training
Train progressively on incoming data streams with Experience Replay:

For Hugging Face downloads, use the [bulk prefetch workflow](docs/HF_STREAMING.md):
larger local batches, resumable per-dataset cursors, bounded disk prefetch and
checkpoint-backed cleanup. The example feeder below only emits bundled examples.
```bash
# Start background online streaming trainer
cargo run --release -- stream-train \
  --incoming data/incoming \
  --out runs/live_growing \
  --replay-ratio 0.30 \
  --save-every 100 \
  --bpe --vocab-size 512

# Feed continuous knowledge chunks
python3 scripts/stream_data.py --incoming data/incoming --interval 10

# In another terminal, query the latest living model artifact at any time:
cargo run --release -- run --model runs/live_growing/model.sea --prompt "The theory" --tokens 100
```

### 7. Standalone Native Model Execution (.sea)
```bash
# Export trained run into a single self-contained binary artifact
cargo run --release -- export --run runs/moe_hybrid --out model.sea

# Run standalone inference directly (zero dependencies)
cargo run --release -- run --model model.sea --prompt "ROMEO:" --tokens 300
```

### 8. Prototype Voice Assistant & Tool-Calling Demonstration (`sail voice`)
Demonstration of hardware audio streaming (`cpal`), energy-based Voice Activity Detection, and local keyword lookup with normalized output entropy checks:
```bash
cargo run --release -- voice --model model.sea --prompt "What is the capital of France?" --entropy-threshold 0.70
```
*Note*: This demonstrates turn-taking and local knowledge retrieval; it is an offline prototype, not a replacement for high-parameter neural TTS or live Internet crawling.

### 9. Live training observatory

Start the local API and GUI (Python 3.10+, no additional packages):

```bash
python3 dashboard/server.py
```

### 10. Omnimodal Any-to-Any Continual Training (`sail stream-train --multimodal`)

Train hybrid SSM + Attention + MoE models simultaneously on text, audio (WAV / Mel filterbank), and vision (images / video patches) on any OS (Linux, macOS, Windows):

```bash
# Launch continual multimodal streaming trainer
cargo run --release -- stream-train \
  --multimodal \
  --incoming data/incoming \
  --out runs/omnimodal \
  --audio-channels 80 \
  --patch-size 16 \
  --steps 2000 \
  --batch-size 8 \
  --save-every 100 \
  --keep-checkpoints 3 \
  --resume
```

As audio, images, and text chunks drop into `data/incoming/` (e.g. from `hf_streamer.py` or user uploads), the trainer automatically:
1. Ingests and extracts acoustic filterbanks and visual patch projections with explicit schema validation.
2. Interleaves text, audio, and vision embeddings into a shared sequence.
3. Performs experience replay sampling to prevent catastrophic forgetting.
4. Atomically exports live, standalone `.sea` artifacts (`runs/omnimodal/model.sea`) for concurrent inference.
5. Emits durable ingestion receipts (`ingestion.json`) ensuring feeder cleanup occurs only after replay commitment.

*Note on Multimodal Capabilities*: The Omnimodal pipeline establishes the tensor plumbing (strided causal 1D audio conv, 2D image patch projection, interleaved recurrence). Small-scale training proves functional execution; high-level speech comprehension or vision-language reasoning requires large-scale pretraining.

### 11. Multimodal Streaming from Hugging Face (`scripts/hf_streamer.py`)

Continuously fetch and stream human-configured public datasets into `data/incoming/` with transactional SQLite deduplication and split-contamination guards:

```bash
# Stream text datasets (code, science, conversational):
python3 scripts/hf_streamer.py --incoming data/incoming --topic code

# Stream audio speech recognition datasets (e.g. speech_commands / librispeech):
python3 scripts/hf_streamer.py --incoming data/incoming --modality audio

# Stream vision datasets (e.g. cifar10 / coco_captions):
python3 scripts/hf_streamer.py --incoming data/incoming --modality vision

# Stream arbitrary custom multimodal datasets with bounded prefetch and directory locking:
python3 scripts/hf_streamer.py --incoming data/incoming \
  --dataset HuggingFaceM4/DocVQA --modality multimodal \
  --target-field answers --prompt-field question
```

### 12. Concurrent Inference Engine (`sail engine`)

Runs snapshot-isolated multi-threaded parallel inference:
- Uses `RwLock`-synchronized generational pointer swapping with immutable request snapshots (`Arc<ModelGeneration>`).
- Automatically reloads `.sea` artifacts when updated by background training.

```bash
cargo run --release -- engine --model model.sea --prompts "The theory,Explain rust,What is sound" --tokens 50
```

### 13. Redox OS Scheme Protocol Simulator (`sail daemon`)

Simulates userspace Redox microkernel packet handling (`RedoxPacket` for `SYS_OPEN`, `SYS_READ`, `SYS_WRITE`, `SYS_CLOSE`) and exposes the `sail:command`, `sail:voice`, and `sail:vision` IPC scheme channels.

```bash
cargo run --release -- daemon --model model.sea --command "help"
```

### 14. Framework Capabilities & Benchmarks

For the complete capability status matrix and empirical benchmark data:
- [Capability Matrix](docs/CAPABILITIES.md): Detailed verification status distinguishing implemented, tested, benchmarked, and planned subsystems.
- [Benchmark Guide](docs/BENCHMARKS.md): Direct timing, prefill, decode, memory, PyTorch comparison, and modality-ablation benchmarks across CPU and GPU hardware.
- [Overlap Audit](scripts/audit_overlap.py): Automated tool for detecting and rejecting train/validation contamination.

Open **http://127.0.0.1:8787**. The dashboard discovers runs in `runs/` and refreshes
every two seconds. Use `--runs /path/to/runs --port 9000` to choose another directory
or port. It serves a read-only API and binds to loopback by default.

The GUI includes loss and learning rate charts, throughput, progress, generated
validation samples, model topology, and a live neural inspector:

**Live connections** adds a separate node-and-edge view. Select any tensor, browse
row/column ranges, and hover or click a connection for its exact weight and change
since the previous observed capture. Click a channel node to isolate its visible
connections. Changed weights briefly flash; reduced-motion preferences disable
the flash. Vectors display individual parameter nodes. All captured weights are
accessible through tensor selection and range paging, with up to 16 × 16 weights
visible at once. Nodes represent parameter axes, including convolution taps and
SSM dimensions; their colors do not claim to measure neuron activations. Sampled
captures omit unavailable edges explicitly.

- Every parameter tensor: embeddings, normalization, Mamba projections and state
  parameters, attention projections, MoE routers, each expert, and the LM head.
- Weight heatmaps with pages covering every scalar, exact values and tensor
  coordinates on hover, and changes between consecutive captures.
- Per-tensor weight statistics and backward-pass gradient RMS before clipping.
- Actual output activations at the embedding, each block, and logits for the
  **last token of the first sequence in the current training batch**. These are
  probes, not all intermediate activations or attention maps. Different batches
  also change these values.

For capture at every optimizer step, add `--viz-every 1` to a new training command:

```bash
cargo run --release -- train \
  --data data/tinyshakespeare.txt --out runs/live_demo \
  --steps 1000 --batch-size 8 --seq-len 32 \
  --d-model 64 --n-layers 4 --attn-layers 2 --mlp-ratio 2 \
  --moe-experts 4 --moe-top-k 2 --log-every 1 --viz-every 1
```

Neural snapshots default to every **25 steps**, plus initialization, step 1, and
the final configured step. `--viz-max-values 0` (default) includes **all weights**.
Capturing synchronizes tensors to the CPU and writes JSON, which adds time and
memory costs. Use `--viz-max-values 1024` for sampled weights per tensor, increase
`--viz-every`, or set `--viz-every 0` to disable neural capture. Tensor statistics
still cover the whole tensor when weight values are sampled.

Weights are captured **before** the optimizer update at the labeled gradient
step: a snapshot at step N contains weights after N−1 updates. The GUI labels
both. The latest snapshot replaces `neural.json` atomically; the browser keeps
the previous capture it observed for comparison. It does not store full history.

Existing training processes expose their CSV metrics immediately. Neural values,
lifecycle status, and sample history require launching the updated trainer;
instrumentation cannot be added to an already-running process. A run without
lifecycle telemetry displays an unknown state. A stale report can indicate either
a slow step or a stopped process. Use a **new output directory** for a new run:
the trainer overwrites files and does not resume an existing checkpoint.

API endpoints:

| Endpoint | Response |
| --- | --- |
| `GET /api/runs` | Available runs, newest metrics first |
| `GET /api/runs/{id}` | Configs, metrics, lifecycle, samples, neural version |
| `GET /api/runs/{id}/neural` | Latest parameter values, gradient statistics, activation probes |

Verify with `python3 -m unittest discover -s dashboard -v`,
`node --test dashboard/test_graph.cjs`, and `cargo test --release`.

See [the product backlog](PRODUCT_BACKLOG.md) for current improvement priorities
and deferred hippocampus/PFC, episodic memory, and neuromodulation experiments.
