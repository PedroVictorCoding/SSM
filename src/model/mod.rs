//! Model components.

/// Burn 0.16's JIT concatenation uses slice-assign, whose kernel iterates using
/// the source strides. Sliced/noncontiguous sources can leave output elements
/// unwritten. Flattening and restoring the shape materializes a dense source
/// on JIT backends while preserving autodiff and staying on-device.
pub(crate) fn cat_dense<B: burn::tensor::backend::Backend, const D: usize>(
    tensors: Vec<burn::tensor::Tensor<B, D>>,
    dim: usize,
) -> burn::tensor::Tensor<B, D> {
    use burn::tensor::Tensor;
    Tensor::cat(
        tensors
            .into_iter()
            .map(|tensor| {
                let shape = tensor.dims();
                let count: usize = shape.iter().product();
                tensor.reshape([count]).reshape(shape)
            })
            .collect(),
        dim,
    )
}

pub mod attention;
pub mod hybrid;
pub mod mamba;
pub mod mlp;
pub mod mmap_moe;
pub mod moe;
pub mod norm;
pub mod scan;

pub use attention::{AttentionConfig, CausalSelfAttention, KvCache};
pub use hybrid::{Block, HybridLm, HybridLmConfig, HybridState, LayerState};
pub use mamba::{Mamba, MambaConfig, MambaState};
pub use mlp::{SwiGlu, SwiGluConfig};
pub use mmap_moe::{MmapExpertStorage, StreamingMoe};
pub use moe::{Moe, MoeConfig};
pub use norm::{RmsNorm, RmsNormConfig};
