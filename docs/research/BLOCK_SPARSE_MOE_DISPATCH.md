# Issue #31 Specification: Dropless Block-Sparse MoE Dispatch (MegaBlocks)

**Issue Link:** [#31: [P2] Validate production sparse MoE dispatch with measured quality and speed](https://github.com/PedroVictorCoding/sail-framework/issues/31)  
**Parent Epic:** [#1 — Live voice, screen/camera context and verified tools](https://github.com/PedroVictorCoding/sail-framework/issues/1)  
**Foundational Invariant:** [ALIGNMENT.md](../../ALIGNMENT.md) — Permanent prohibition on autonomous recursive self-improvement. Model routing policies, capacity factors, and dispatch kernels remain human-defined.

---

## 1. Problem Statement & Bottleneck Analysis

In SAIL's current Mixture-of-Experts implementation (`src/model/moe.rs`):
1. **Full-Matrix Redundant Execution:** The current `sparse_dispatch` mode skips an expert only if its routing weight is identically zero across **every single token in the entire batch**. When all $E$ experts receive at least one token (which is virtually guaranteed when batch size $B \times L \ge 512$), every expert computes the dense forward pass over **all $N$ tokens**.
2. **Compute Inefficiency:** For $N$ tokens with Top-$K$ routing ($K=1$ or $K=2$) across $E=4$ experts:
   - Expected row operations: $N \times K = 1 \cdot N$.
   - Current executed row operations: $N \times E = 4 \cdot N$.
   - $75\%$ of all expert FLOPs are wasted calculating outputs that are subsequently multiplied by zero!
3. **Padded Buffer Traps:** Traditional static-capacity MoE implementations pad expert batches to fixed capacities, dropping overflow tokens and corrupting language modeling accuracy.

---

## 2. Mathematical Formulation (MegaBlocks & Dropless CSR Dispatch)

### 2.1 Router Gating
For input representation $X \in \mathbb{R}^{N \times d_{\text{model}}}$ ($N = B \times L$):
$$G = \text{TopK}(\text{Softmax}(X W_g + \epsilon), K)$$
where $G \in \mathbb{R}^{N \times E}$ is the sparse gate matrix with at most $K$ non-zero entries per row.

### 2.2 Compressed Sparse Token Gather & Scatter
Let $\mathcal{I}_e = \{ i \in \{0, \dots, N-1\} \mid G_{i, e} > 0 \}$ be the set of token indices routed to expert $e \in \{0, \dots, E-1\}$.
1. **Token Gather (No Padding, No Dropping):**
   Extract only the assigned rows for expert $e$:
   $$X^{(e)} = X[\mathcal{I}_e, :] \in \mathbb{R}^{N_e \times d_{\text{model}}}$$
   where $N_e = |\mathcal{I}_e|$ is the exact dynamic number of tokens routed to expert $e$, satisfying:
   $$\sum_{e=0}^{E-1} N_e = K \cdot N$$
2. **Dense Expert MLP Execution:**
   Compute standard dense feed-forward network on the compact token matrix:
   $$H^{(e)} = \text{GELU}(X^{(e)} W_1^{(e)}) W_2^{(e)} \in \mathbb{R}^{N_e \times d_{\text{model}}}$$
   Because $N_e \ll N$, this matrix multiplication is $E / K$ times smaller than the full dense batch.
3. **Weighted Token Scatter:**
   Scatter the expert outputs back to their original sequence positions weighted by gating coefficients:
   $$Y[i, :] = \sum_{e \text{ s.t. } i \in \mathcal{I}_e} G_{i, e} \cdot H^{(e)}[\text{index of } i \text{ in } \mathcal{I}_e, :]$$

---

## 3. Concrete Implementation Plan in Rust/Burn

1. **Target Module:** `src/model/moe.rs`
   - Implement `dropless_block_sparse_dispatch<B: Backend>(x: Tensor<B, 2>, gates: Tensor<B, 2>, experts: &[Mlp<B>])`:
     - Sort indices $(i, e)$ by expert ID $e$ using a single GPU index argsort or CPU coordinate map.
     - Extract row slices via `tensor.select(0, indices)`.
     - Execute each expert's linear layers on its gathered contiguous slice $X^{(e)}$.
     - Scatter results back into output accumulator $Y \in \mathbb{R}^{N \times d_{\text{model}}}$ using `tensor.scatter` or index add.
2. **Gradient & Backward Pass Parity**:
   - Validate that backward autograd gradients $\nabla W_g, \nabla W_1^{(e)}, \nabla W_2^{(e)}, \nabla X$ are numerically identical to the dense reference implementation.

---

## 4. Acceptance Criteria

- [ ] Equivalence test asserting $|Y_{\text{sparse}} - Y_{\text{dense}}| < 10^{-4}$ on arbitrary batch sizes.
- [ ] Gradient check verifying identical weight updates and auxiliary router loss.
- [ ] Confirmed reduction in executed expert FLOPs: exactly $K \cdot N$ rows computed instead of $E \cdot N$.
- [ ] Measured wall-clock speedup on AMD RX 7800 XT with $E=4, K=1$ ($\ge 1.5\times\text{--}2\times$ MoE layer speedup).

