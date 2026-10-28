# Issue #35 Specification: Decay-Aware Chunkwise Gated Linear Attention (GLA)

**Issue Link:** [#35: [P2] Implement decay-aware chunkwise gated linear attention](https://github.com/PedroVictorCoding/sail-framework/issues/35)  
**Parent Epic:** [#1 — Live voice, screen/camera context and verified tools](https://github.com/PedroVictorCoding/sail-framework/issues/1)  
**Foundational Invariant:** [ALIGNMENT.md](../../ALIGNMENT.md) — Permanent prohibition on autonomous recursive self-improvement.

---

## 1. Problem Statement & Bottleneck Analysis

In SAIL's hybrid architecture (`src/model/hybrid.rs`):
1. **Quadratic Attention Bottlenecks:** Layers designated as attention layers (`attn_layers = [1, 3]`) currently employ standard softmax multi-head attention ($O(L^2)$ memory and compute) or sliding window attention.
2. **Context Scaling Degradation:** As sequence context scales beyond $L=512$ or $1024$ tokens, the standard attention layers consume an increasingly disproportionate share of GPU memory and step execution latency.
3. **Loss of Recurrent State Continuity:** Full softmax attention requires materializing the entire key-value history during inference, disrupting the $O(1)$ constant-memory recurrent decoding guarantee of pure SSM layers.

---

## 2. Mathematical Formulation (Yang et al., GLA / FlashLinearAttention 2023)

### 2.1 Formulation
Gated Linear Attention replaces standard softmax attention with a 2D data-dependent forget gate $G$ and linear projection:
$$O_t = \sum_{s=1}^t \left( Q_t K_s^\top \odot \exp(\Gamma_t - \Gamma_s) \right) V_s$$
where $\Gamma_t = \sum_{i=1}^t \log(\sigma(g_i))$ is the cumulative log-decay gate.

### 2.2 Chunkwise Hardware-Aligned Decomposition
Divide sequence into chunks of length $C$ (e.g. $C = 64$):
1. **Intra-Chunk Parallel Computation (Tiled GEMM):**
   $$O_k^{\text{intra}} = \left( (Q_k K_k^\top) \odot D_k \right) V_k$$
   where $D_k$ is the local intra-chunk decay mask. Runs entirely in hardware Matrix Cores without materializing full $L \times L$ attention matrices.
2. **Inter-Chunk Recurrent State:**
   $$S_k = \exp(\Gamma_k) \odot S_{k-1} + K_k^\top V_k$$
   $$O_k^{\text{inter}} = Q_k S_{k-1}$$
3. **Total Chunk Output:**
   $$O_k = O_k^{\text{intra}} + O_k^{\text{inter}}$$

---

## 3. Concrete Implementation Plan in Rust/Burn

1. **Target Module:** `src/model/hybrid.rs`
   - Implement `GatedLinearAttentionBlock<B: Backend>`:
     - Linear projections for $Q, K, V \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ and gate $g \in \mathbb{R}^{B \times L \times n_{\text{heads}}}$.
     - Chunk partition of size $C = 64$.
     - Intra-chunk tiled matrix multiplications using Burn's `B::matmul`.
     - Inter-chunk recurrent state propagation $S_k \in \mathbb{R}^{B \times n_{\text{heads}} \times d_k \times d_v}$.
2. **Recurrent Decoding Support**:
   - In inference mode, $S_t$ updates at $O(1)$ cost per token: $S_t = \exp(g_t) S_{t-1} + K_t^\top V_t, \quad O_t = Q_t S_t$.
   - Preserves universal $O(1)$ constant-memory decoding across both SSM and Attention layers!

---

## 4. Acceptance Criteria

- [ ] Equivalence test asserting numerical consistency between chunkwise GLA and sequential linear recurrent attention.
- [ ] O(1) constant-memory recurrent inference integration in `sail run`.
- [ ] Measured reduction in memory usage at $L \ge 512$ ($\ge 50\%$ lower activation memory than standard softmax attention).

