//! A from-scratch selective state space (Mamba-style) / attention hybrid language model in Burn.
//!
//! Module layout:
//! - [`model`]: layers (RMSNorm, Mamba mixer, causal attention, SwiGLU), the parallel scan,
//!   and the [`model::HybridLm`] stack with O(1)-per-token recurrent decoding.
//! - [`data`]: character tokenizer and batching for plain-text corpora (e.g. TinyShakespeare).
//! - [`train`]: training loop with AdamW, warmup + cosine LR, gradient clipping, validation and
//!   checkpointing.
//! - [`generate`]: sampling, both stateful (constant memory per SSM layer) and naive recompute.
//! - [`backend`]: compile-time backend selection (ndarray / wgpu / cuda).

pub mod backend;
pub mod capability;
pub mod data;
pub mod engine;
pub mod export;
pub mod generate;
pub mod kernel;
pub mod model;
pub mod optim;
pub mod os;
mod profile;
pub mod runtime;
pub mod screen;
mod storage;
pub mod telemetry;
pub mod tool;
pub mod train;
pub mod voice;
