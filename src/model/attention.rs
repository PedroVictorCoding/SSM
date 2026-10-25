//! Causal multi-head self-attention with an optional sliding window and a KV cache.
//!
//! Following Jamba, no positional encoding is used: the Mamba layers placed before the attention
//! layers already inject order information into the residual stream (NoPE).
//!
//! With `window = 0` attention is fully causal and the KV cache grows linearly with context.
//! With `window = W > 0` each query sees only the last `W` keys (itself included); the cache is
//! capped at `W` entries, so the whole hybrid model decodes with bounded memory.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub d_model: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    /// Sliding window size; 0 = full causal attention.
    #[config(default = 0)]
    pub window: usize,
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub qkv: Linear<B>,
    pub out: Linear<B>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub window: usize,
}

/// Key/value cache, each `[batch, heads, cached_len, head_dim]`. Empty before the first token.
#[derive(Clone, Debug)]
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    pub fn len(&self) -> usize {
        self.k.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_elements(&self) -> usize {
        let count = |t: &Option<Tensor<B, 4>>| t.as_ref().map(|t| t.shape().num_elements());
        count(&self.k).unwrap_or(0) + count(&self.v).unwrap_or(0)
    }
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model,
            self.n_heads
        );
        CausalSelfAttention {
            qkv: LinearConfig::new(self.d_model, 3 * self.d_model)
                .with_bias(false)
                .init(device),
            out: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .init(device),
            n_heads: self.n_heads,
            head_dim: self.d_model / self.n_heads,
            window: self.window,
        }
    }
}

/// `[len, len]` mask, `true` = blocked. Query `i` may attend to key `j` iff `j <= i` and,
/// with a window, `i - j < window`.
fn attention_mask<B: Backend>(len: usize, window: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    let data: Vec<bool> = (0..len)
        .flat_map(|i| (0..len).map(move |j| j > i || (window > 0 && i - j >= window)))
        .collect();
    Tensor::from_data(TensorData::new(data, [len, len]), device)
}

impl<B: Backend> CausalSelfAttention<B> {
    fn scale(&self) -> f64 {
        1.0 / (self.head_dim as f64).sqrt()
    }

    /// Splits a `[B, L, 3D]` projection into `[B, H, L, Dh]` q, k, v.
    fn split_heads(&self, qkv: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, len, three_d] = qkv.dims();
        let d = three_d / 3;
        let shape = [batch, len, self.n_heads, self.head_dim];
        let head = |start: usize| {
            qkv.clone()
                .narrow(2, start, d)
                .reshape(shape)
                .swap_dims(1, 2)
        };
        (head(0), head(d), head(2 * d))
    }

    /// Keep only the most recent `window` cache entries (no-op for full attention).
    fn truncate(&self, t: Tensor<B, 4>) -> Tensor<B, 4> {
        let len = t.dims()[2];
        if self.window > 0 && len > self.window {
            t.narrow(2, len - self.window, self.window)
        } else {
            t
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_state(x).0
    }

    /// Full-sequence causal attention; also returns the KV cache for continued decoding.
    pub fn forward_with_state(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, KvCache<B>) {
        let [batch, len, d_model] = x.dims();
        let device = x.device();
        let (q, k, v) = self.split_heads(self.qkv.forward(x));

        let scores = q.matmul(k.clone().swap_dims(2, 3)).mul_scalar(self.scale()); // [B,H,L,L]
        let mask = attention_mask::<B>(len, self.window, &device)
            .unsqueeze::<4>()
            .expand([batch, self.n_heads, len, len]);
        let probs = softmax(scores.mask_fill(mask, -1.0e9), 3);
        let ctx = probs
            .matmul(v.clone())
            .swap_dims(1, 2)
            .reshape([batch, len, d_model]);

        let cache = KvCache {
            k: Some(self.truncate(k)),
            v: Some(self.truncate(v)),
        };
        (self.out.forward(ctx), cache)
    }

    /// Single-token step. `x`: `[batch, d_model]`.
    pub fn step(&self, x: Tensor<B, 2>, cache: KvCache<B>) -> (Tensor<B, 2>, KvCache<B>) {
        let [batch, d_model] = x.dims();
        let (q, k, v) = self.split_heads(self.qkv.forward(x).unsqueeze_dim::<3>(1)); // [B,H,1,Dh]

        let append = |old: Option<Tensor<B, 4>>, new: Tensor<B, 4>| match old {
            Some(old) => self.truncate(super::cat_dense(vec![old, new], 2)),
            None => new,
        };
        let k_all = append(cache.k, k);
        let v_all = append(cache.v, v);

        let scores = q
            .matmul(k_all.clone().swap_dims(2, 3))
            .mul_scalar(self.scale()); // [B,H,1,T]
        let ctx = softmax(scores, 3)
            .matmul(v_all.clone())
            .reshape([batch, d_model]);

        (
            self.out.forward(ctx),
            KvCache {
                k: Some(k_all),
                v: Some(v_all),
            },
        )
    }
}
