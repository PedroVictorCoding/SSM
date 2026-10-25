//! Linear recurrence scans: `h_t = a_t ⊙ h_{t-1} + b_t`, with `h_{-1} = 0`.
//!
//! The recurrence is associative under the operator
//! `(a₁, b₁) ∘ (a₂, b₂) = (a₁·a₂, a₂·b₁ + b₂)`, so it can be evaluated with a parallel prefix
//! scan in `O(log L)` sequential depth instead of `L` dependent steps. Everything here is built
//! from ordinary tensor ops, so it runs on any Burn backend and is differentiable by autodiff.
//!
//! [`chunked_parallel_scan`] is a two-level scheme:
//! 1. split time into chunks of size `C` and run a Hillis–Steele scan inside every chunk
//!    (all chunks in parallel, `log₂ C` steps);
//! 2. run a second Hillis–Steele scan over the per-chunk end states (`log₂(L/C)` steps) to get
//!    the carry entering each chunk, then apply it: `h = h_local + A_local ⊙ carry`.
//!
//! Compared to a single Hillis–Steele scan over the full sequence this keeps the depth at
//! `log₂ L` while reducing the autodiff memory from `O(L log L)` to `O(L log C)` activations.

use burn::prelude::*;

/// Shift `x` towards higher indices by `offset` along `dim`, filling the vacated slots with `fill`.
fn shift_right<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dim: usize,
    offset: usize,
    fill: f32,
) -> Tensor<B, D> {
    let dims = x.dims();
    let len = dims[dim];
    let device = x.device();
    if offset >= len {
        return Tensor::full(dims, fill, &device);
    }
    // Burn 0.16's cat backward advances slice offsets only for tracked inputs.
    // A constant prefix before a tracked suffix therefore shifts its gradient.
    // Keep the prefix tracked with an identically zero derivative. Recurrence
    // coefficients/states must be finite (as required for meaningful training).
    let pad = x
        .clone()
        .narrow(dim, 0, offset)
        .mul_scalar(0.0)
        .add_scalar(fill);
    let kept = x.narrow(dim, 0, len - offset);
    super::cat_dense(vec![pad, kept], dim)
}

/// Inclusive Hillis–Steele scan of `h_t = a_t ⊙ h_{t-1} + b_t` along `dim`.
///
/// Returns `(A, H)` where `A_t = ∏_{s≤t} a_s` and `H_t = h_t` (with zero initial state).
pub fn hillis_steele_scan<B: Backend, const D: usize>(
    mut a: Tensor<B, D>,
    mut b: Tensor<B, D>,
    dim: usize,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let len = a.dims()[dim];
    let mut offset = 1;
    while offset < len {
        let a_prev = shift_right(a.clone(), dim, offset, 1.0);
        let b_prev = shift_right(b.clone(), dim, offset, 0.0);
        // Compose the earlier segment (a_prev, b_prev) with the current one (a, b).
        b = a.clone() * b_prev + b;
        a = a * a_prev;
        offset *= 2;
    }
    (a, b)
}

/// Reference sequential scan over dim 1 of `[batch, len, d, n]` tensors.
pub fn sequential_scan<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, len, d, n] = a.dims();
    let mut h = Tensor::<B, 3>::zeros([batch, d, n], &a.device());
    let mut states = Vec::with_capacity(len);
    for t in 0..len {
        let a_t: Tensor<B, 3> = a.clone().narrow(1, t, 1).squeeze(1);
        let b_t: Tensor<B, 3> = b.clone().narrow(1, t, 1).squeeze(1);
        h = a_t * h + b_t;
        states.push(h.clone().unsqueeze_dim::<4>(1));
    }
    super::cat_dense(states, 1)
}

/// Work-efficient inclusive scan: pair reduction followed by reconstruction.
///
/// Each level composes adjacent affine recurrences, recursively scans half as
/// many elements, then reconstructs the even states from the preceding pair's
/// end state. T(L) = T(floor(L/2)) + O(L): O(L) elementwise work and saved
/// tensor elements, with O(log L) dependent levels. No division, cumulative
/// product inversion, host transfers or padding to a power of two are needed.
/// This is a portable tensor implementation, not a fused hardware kernel.
pub fn work_efficient_scan<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, len, d, n] = a.dims();
    assert_eq!(a.dims(), b.dims(), "scan input shapes must match");
    assert!(len > 0, "scan sequence must be nonempty");
    if len == 1 {
        return b;
    }
    let pairs = len / 2;
    let ap = a
        .clone()
        .narrow(1, 0, pairs * 2)
        .reshape([batch, pairs, 2, d, n]);
    let bp = b
        .clone()
        .narrow(1, 0, pairs * 2)
        .reshape([batch, pairs, 2, d, n]);
    let a_even = ap.clone().narrow(2, 0, 1).squeeze::<4>(2);
    let a_odd = ap.narrow(2, 1, 1).squeeze::<4>(2);
    let b_even = bp.clone().narrow(2, 0, 1).squeeze::<4>(2);
    let b_odd = bp.narrow(2, 1, 1).squeeze::<4>(2);
    let pair_ends = work_efficient_scan(
        a_odd.clone() * a_even.clone(),
        a_odd * b_even.clone() + b_odd,
    );
    let even_states = a_even * shift_right(pair_ends.clone(), 1, 1, 0.0) + b_even;
    let interleaved = super::cat_dense(
        vec![
            even_states.unsqueeze_dim::<5>(2),
            pair_ends.clone().unsqueeze_dim::<5>(2),
        ],
        2,
    )
    .reshape([batch, pairs * 2, d, n]);
    if len % 2 == 0 {
        interleaved
    } else {
        let tail =
            a.narrow(1, len - 1, 1) * pair_ends.narrow(1, pairs - 1, 1) + b.narrow(1, len - 1, 1);
        super::cat_dense(vec![interleaved, tail], 1)
    }
}

/// Two-level chunked parallel scan over dim 1 of `[batch, len, d, n]` tensors.
pub fn chunked_parallel_scan<B: Backend>(
    a: Tensor<B, 4>,
    b: Tensor<B, 4>,
    chunk: usize,
) -> Tensor<B, 4> {
    let [batch, len, d, n] = a.dims();
    let device = a.device();
    let chunk = chunk.clamp(1, len.max(1));
    let n_chunks = len.div_ceil(chunk);
    let padded = n_chunks * chunk;

    // Pad the tail with the identity element (a = 1, b = 0); it cannot affect earlier outputs.
    let (a, b) = if padded > len {
        let pad = padded - len;
        (
            super::cat_dense(vec![a, Tensor::ones([batch, pad, d, n], &device)], 1),
            super::cat_dense(vec![b, Tensor::zeros([batch, pad, d, n], &device)], 1),
        )
    } else {
        (a, b)
    };

    let a5: Tensor<B, 5> = a.reshape([batch, n_chunks, chunk, d, n]);
    let b5: Tensor<B, 5> = b.reshape([batch, n_chunks, chunk, d, n]);

    // Level 1: independent scans inside every chunk.
    let (a_local, h_local) = hillis_steele_scan(a5, b5, 2);

    // Level 2: propagate chunk-end states across chunks.
    let h = if n_chunks > 1 {
        let a_end = a_local.clone().narrow(2, chunk - 1, 1); // [B, nC, 1, D, N]
        let h_end = h_local.clone().narrow(2, chunk - 1, 1);
        let (_, carry_inclusive) = hillis_steele_scan(a_end, h_end, 1);
        let carry_in = shift_right(carry_inclusive, 1, 1, 0.0); // state entering each chunk
        h_local + a_local * carry_in
    } else {
        h_local
    };

    h.reshape([batch, padded, d, n]).narrow(1, 0, len)
}

/// Dispatch: `chunk == 0` → sequential reference scan, otherwise the chunked parallel scan.
pub fn linear_scan<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>, chunk: usize) -> Tensor<B, 4> {
    if chunk == 0 {
        sequential_scan(a, b)
    } else {
        chunked_parallel_scan(a, b, chunk)
    }
}
