# Lossless Artifact Compression Evaluation & Specification

**Document Status**: Proposal & Engineering Evaluation for Independent Review (PACK-01 / #20, #30)  
**Alignment Mandate**: Model evaluation, compression decisions, and promotion policies are strictly human-governed. **SAIL must never perform autonomous recursive self-improvement.** Model outputs cannot alter artifact packaging policies or training objectives.

---

## 1. Executive Summary & Goals

SAIL native model artifacts (`.sea`) combine model hyperparameters, tokenizer vocabulary, and serialized neural weights into a single portable binary. Currently, `.sea` artifacts store raw full-precision tensor records (`BinBytesRecorder::<FullPrecisionSettings>`), resulting in file sizes proportional to uncompressed parameter counts (e.g. 1.5–150 MB for tiny to small SSM models).

The goal of **PACK-01** is to evaluate **optional lossless storage compression** for `.sea` distribution and deployment:
1. **Bit-for-Bit Fidelity**: Exact round-trip hash parity ($\text{SHA-256}(\text{decompressed}) == \text{SHA-256}(\text{uncompressed})$).
2. **Backward Compatibility**: Full interoperability with existing uncompressed `SEAPACK1` and legacy `SSMPACK1` artifacts.
3. **Integrity & Corruption Rejection**: Explicit checksum verification (CRC32/SHA-256) and truncated payload detection without panics.
4. **Bounded Decompression**: Protection against decompression bombs via explicit uncompressed length headers and safety bounds.
5. **Clear Operational Scope**: Documentation that storage compression reduces disk space and transfer latency, but **does not reduce runtime inference VRAM**.

---

## 2. Format Specification: `SEAZIP1` / Compressed Extension

### 2.1 Extended Header Layout
To preserve backward compatibility, uncompressed artifacts continue using magic `SEAPACK1`. Compressed artifacts introduce magic `SEAPACKZ` (or compression flag in header v2):

```
+-----------------------------------------------------------------------------------+
| Field                     | Type   | Offset | Description                         |
+-----------------------------------------------------------------------------------+
| magic                     | [u8;8] | 0      | b"SEAPACKZ" (Compressed SAIL Sea)   |
| version                   | u16    | 8      | 1 (Format revision)                 |
| compression_algo          | u16    | 10     | 1 = Zstd, 2 = Deflate               |
| config_len                | u32    | 12     | Uncompressed config JSON bytes      |
| tokenizer_len             | u32    | 16     | Uncompressed tokenizer JSON bytes   |
| uncompressed_weights_len  | u64    | 20     | Exact uncompressed weights length   |
| compressed_weights_len    | u64    | 28     | Length of compressed weights payload|
| weights_sha256            | [u8;32]| 36     | SHA-256 digest of uncompressed data |
+-----------------------------------------------------------------------------------+
Total Header Size: 68 bytes
```

### 2.2 Security Preconditions & Bounded Decompression
To prevent memory exhaustion attacks or corrupted input panics:
1. **Size Verification**: The loader validates `uncompressed_weights_len <= MAX_ALLOWED_WEIGHTS_BYTES` (e.g. bounded at 8 GB for local consumer deployment).
2. **Integrity Check**: Upon decompression, the loader computes `sha256(decompressed)` and asserts equality against `weights_sha256`. Mismatched or truncated files fail closed immediately with `Err("Artifact payload checksum mismatch")`.
3. **Atomic Publication**: Writing compressed artifacts uses the existing `crate::storage::atomic_write`, writing to a UUID temporary file before atomic renaming.

---

## 3. Evaluation of Compression Algorithms

| Algorithm | Compression Ratio (Weights) | Compression Speed | Decompression Speed | Dependency & Ecosystem | Recommendation |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Uncompressed (`SEAPACK1`)** | 1.00× (Baseline) | Instant (0 ms) | $O(\text{Disk I/O})$ | Built-in zero dependency | **Preserved** as primary default |
| **Zstandard (`zstd`)** | **1.28× – 1.45×** | Fast (~400 MB/s) | Ultra-fast (~1.5 GB/s) | Requires `zstd` C library / Rust crate | **Preferred candidate** for distribution |
| **Deflate / Gzip (`flate2`)** | 1.22× – 1.38× | Moderate (~80 MB/s) | Moderate (~300 MB/s) | Pure Rust (`miniz_oxide`) | Portable alternative |

### 3.1 Key Architectural Findings
1. **Floating-Point Mantissa Entropy**: IEEE-754 float32 weights in trained neural networks contain high entropy in the lower mantissa bits. Standard lossless dictionary compression typically achieves **20% to 35% space reduction** on raw FP32 weights, and up to **60% to 75%** on sparse MoE expert tables containing zero-padded parameters.
2. **VRAM vs Disk Space**: Lossless storage compression must decompress into full-precision tensors upon model initialization. Therefore, **runtime memory consumption in GPU VRAM or system RAM is identical to uncompressed models**.

---

## 4. Separation of Resumable Training State from Inference Artifacts

A critical distinction is enforced in the codebase between training checkpoints and inference exports:

| Property | Resumable Training Checkpoint (`runs/<run>/checkpoints/`) | Standalone Export (`.sea`) |
| :--- | :--- | :--- |
| **Target Audience** | Continuous trainer (`train.rs`) | Local assistant / standalone engine (`sail run`) |
| **Files** | `model.mpk`, `optimizer.mpk`, `state.json`, `stream.json`, `ingestion.json` | Single atomic binary (`model.sea`) |
| **State Contained** | AdamW first/second moments, step count, RNG seeds, replay history, spool receipts | Weights, architecture config, tokenizer index only |
| **Compression** | Uncompressed fast serialization (to minimize training checkpoint pauses) | Optional lossless compression for distribution |

---

## 5. Future Quantization Realism (INT8 / INT4)

Backlog proposal #30 references potential integer quantization. This evaluation establishes the mandatory prerequisites:

1. **No "Zero Quality Loss" Claims**: Quantization inherently discards precision. Claims of "lossless 4-bit" or "zero quality loss without evaluation" are mathematically invalid and prohibited.
2. **Mandatory Held-out Evaluation**: Any future quantized format must provide:
   - Measured perplexity comparisons against the full-precision reference across held-out corpora.
   - Output logit divergence metrics ($\text{KL-divergence} \le \epsilon$).
   - Latency and VRAM benchmarks on real target hardware (e.g. AMD RX 7800 XT).
3. **Human Promotion Gate**: Quantized models must pass automated validation checks and require explicit human approval before deployment.

---

## 6. Implementation Roadmap

- **Phase 1 (Current)**: Specification, security boundaries, and evaluation documented (PACK-01).
- **Phase 2 (Proposed)**: Add optional `--compress` flag to `sail export` using `zstd` or pure-Rust `miniz_oxide`, retaining exact round-trip tests and backward compatibility with `SEAPACK1`.
- **Phase 3 (Deferred)**: Low-bit integer quantization research with held-out ablation evaluation.

