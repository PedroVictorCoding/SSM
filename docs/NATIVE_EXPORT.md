# Native Artifacts & Standalone Deployment Guide

SSM is engineered in 100% pure Rust without external runtime dependencies (Python, PyTorch, C++ compilers, or CUDA toolchains).

---

## 1. Export to Single-File Native Artifacts (`.ssm`)

Once training or fine-tuning completes, you can export the entire model into a single portable binary artifact (`.ssm`):

```bash
cargo run --release -- export --run runs/shakespeare --out model.ssm
```

### Binary Layout (`.ssm`)
| Offset | Field | Format / Description |
|---|---|---|
| `0..8` | Magic | `SSMPACK1` (8 bytes) |
| `8..12` | Config Length | `u32` (Little Endian) |
| `12..16` | Tokenizer Length | `u32` (Little Endian) |
| `16..24` | Weights Length | `u64` (Little Endian) |
| `24..` | Model Config | Architecture hyperparameters (`HybridLmConfig` JSON) |
| Next | Tokenizer | Subword BPE merges & rank table or Char vocabulary (`Tokenizer` JSON) |
| Next | Model Parameters | Raw full-precision tensor weights (MessagePack / Burn record) |

---

## 2. Zero-Dependency Standalone Inference

Run generation directly from the single `.ssm` file with zero configuration:

```bash
cargo run --release -- run --model model.ssm --prompt "ROMEO:" --tokens 300 --temperature 0.8
```

Features:
- **$O(1)$ recurrent step decoding**: Prefill encodes context, then emits tokens at constant memory per layer.
- **Self-contained**: No separate `vocab.json`, `model_config.json`, or Python scripts required.

---

## 3. Sparse Expert Streaming (Memory-Mapped MoE)

For massive parameter banks with limited VRAM:
- Experts are persisted in a contiguous binary file (`experts.bin`) and memory-mapped via `memmap2`.
- The router dynamically evaluates token activations and streams *only* the top-$K$ selected experts into compute device memory.
- Reduces active VRAM footprint by up to $16\times$ on large expert banks.

---

## 4. Hardware Kernels (CubeCL On-Chip SRAM Scan)

- Pure Rust compute kernel compiling to PTX (CUDA) and SPIR-V (Vulkan/Metal/WGPU).
- Fuses associative parallel scan into on-chip SRAM (`SharedMemory`), eliminating global DRAM roundtrips and accelerating throughput.

---

## 5. Cross-Compilation & Target Support

### Standalone Static Binary (Linux musl)
```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
# Creates a fully static binary with zero glibc dependencies
```

### Windows & macOS
```bash
# macOS ARM64 (Apple Silicon)
cargo build --release --target aarch64-apple-darwin

# Windows x86_64
cargo build --release --target x86_64-pc-windows-msvc
```

### WebAssembly in Browser (`wasm32-unknown-unknown`)
With `load_artifact_from_bytes`:
```rust
use ssm::export::load_artifact_from_bytes;
use ssm::backend::InferenceBackend;

const MODEL_BYTES: &[u8] = include_bytes!("../model.ssm");

pub fn init_web_model() {
    let device = Default::default();
    let (model, cfg, tokenizer) = load_artifact_from_bytes::<InferenceBackend>(MODEL_BYTES, &device).unwrap();
    // Ready for interactive in-browser generation!
}
```
Compile with:
```bash
wasm-pack build --target web
```

