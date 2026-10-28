# Issue #32 Specification: State Space Duality (SSD) Chunkwise GEMM Scan & Selective State Recomputation

**Issue Link:** [#32: [P2] Validate fused scan/recomputation only against measured assistant bottlenecks](https://github.com/PedroVictorCoding/sail-framework/issues/32)  
**Parent Epic:** [#1 — Live voice, screen/camera context and verified tools](https://github.com/PedroVictorCoding/sail-framework/issues/1)  
**Foundational Invariant:** [ALIGNMENT.md](../../ALIGNMENT.md) — No autonomous recursive self-improvement. All architecture updates, training schedules, and evaluation gates are human-defined.

---

## 1. Problem Statement & Bottleneck Analysis

In SAIL's current SSM implementation (`src/kernel/scan.rs` and `src/model/hybrid.rs`):
1. **Memory-Bandwidth Bound Scan:** The selective 1D recurrence:
   $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$
   is computed using an associative scan (`chunked_parallel_scan`). While $O(T)$ in algorithmic steps, associative scans execute elementwise vector-ALU arithmetic. Modern GPUs (including the AMD Radeon RX 7800 XT) achieve their peak TFLOPS exclusively through dense Matrix Cores (GEMM). Non-GEMM operations leave up to $90\%$ of theoretical compute capacity idle.
2. **VRAM Traffic Saturation in Backward Pass:** During the forward pass, materializing intermediate states $h \in \mathbb{R}^{B \times T \times D \times N}$ across all layers for autograd consumes gigabytes of VRAM and saturates PCIe/memory buses during backpropagation.

---

## 2. Mathematical Formulation

### 2.1 State Space Duality (SSD) Formulation (Dao & Gu, ICML 2024)
Under scalar/diagonal structured $\bar{A}_t$, the 1D selective state space recurrence is mathematically equivalent to a structured semi-separable matrix transformation:
$$Y = \left( L \circ (C B^\top) \right) X$$
where:
* $X \in \mathbb{R}^{T \times D}$ is the input token sequence.
* $B, C \in \mathbb{R}^{T \times N}$ are the input/output state projections.
* $L \in \mathbb{R}^{T \times T}$ is the causal lower-triangular decay matrix:
  $$L_{ij} = \begin{cases} \prod_{k=j+1}^i a_k & \text{if } i \ge j \\ 0 & \text{if } i < j \end{cases}$$

### 2.2 Chunkwise Block-Decomposition
Instead of scanning across all $T$ tokens sequentially, divide the sequence into $M = T / Q$ chunks of length $Q$ (e.g. $Q = 64$ or $128$):
1. **Intra-Chunk Computation (Tensor Cores / GEMM):**
   For each chunk $m \in \{0, \dots, M-1\}$:
   $$Y_m^{\text{intra}} = \left( L_m \circ (C_m B_m^\top) \right) X_m$$
   This is computed as two batched matrix multiplications (`matmul`), running directly on hardware Matrix Cores.
2. **Inter-Chunk State Propagation:**
   Boundary states are accumulated across the $M$ chunk boundaries:
   $$h_m = \alpha_m h_{m-1} + B_m^\top X_m, \quad Y_m^{\text{inter}} = C_m h_{m-1}$$
   where $\alpha_m = \prod_{t \in \text{chunk } m} a_t$.
3. **Total Output:**
   $$Y_m = Y_m^{\text{intra}} + Y_m^{\text{inter}}$$

### 2.3 Selective Activation Recomputation (Gradient Checkpointing)
During training:
* **Forward Pass:** Discard the full 4D state tensor $h \in \mathbb{R}^{B \times T \times D \times N}$. Store only inputs $X$, boundary states $h_{m \cdot Q}$, and parameter tensors.
* **Backward Pass:** Inside the backward step, recompute intra-chunk states $h_t$ locally in high-speed GPU cache immediately prior to accumulating $\nabla A, \nabla B, \nabla C$.
* **Impact:** Reduces VRAM read/write traffic by $75\%\text{--}80\%$.

---

## 3. Concrete Implementation Plan in Rust/Burn

1. **Target Module:** `src/kernel/scan.rs`
   - Implement `ssd_chunkwise_gemm_scan<B: Backend>(x, a, b, c, chunk_size: usize)`:
     - Formulate local decay mask $L_m$ via cumulative log-space sum: $L_{ij} = \exp(\text{cumsum}(a)_i - \text{cumsum}(a)_j)$.
     - Execute intra-chunk contractions via `B::matmul`.
     - Execute inter-chunk boundary propagation loop over $M$ chunks.
2. **Autodiff Recomputation in `src/model/hybrid.rs`**:
   - Wrap SSM layer in a custom backward graph node that re-executes forward chunk scan during gradient evaluation rather than holding intermediate activations.
3. **Equivalence & Gradient Verification**:
   - Assert $|Y_{\text{ssd}} - Y_{\text{sequential}}| < 10^{-4}$ across single and double precision.
   - Assert identical gradients $\nabla X, \nabla A, \nabla B, \nabla C$ within float tolerance.

---

## 4. Acceptance Criteria

- [ ] Numerical equivalence test comparing SSD chunkwise GEMM output with existing sequential scan across random inputs.
- [ ] Gradient check verifying identical backward passes between SSD and sequential reference.
- [ ] Measured training throughput increase ($\ge +50\%$ tokens/sec) on AMD Radeon RX 7800 XT with zero loss divergence.

