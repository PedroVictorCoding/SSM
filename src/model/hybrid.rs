//! Jamba-style hybrid language model: a stack of pre-norm residual blocks whose sequence mixer
//! is either a Mamba layer (most layers) or causal self-attention (a few chosen layers),
//! optionally followed by a SwiGLU MLP.
//!
//! ```text
//! tokens → Embedding → [Block × n_layers] → RMSNorm → LM head → logits
//! Block:  x = x + Mixer(RMSNorm(x));   x = x + MLP(RMSNorm(x))   (MLP optional)
//! ```
//!
//! Inference API:
//! - [`HybridLm::forward`]: logits for a whole sequence (training / scoring).
//! - [`HybridLm::prefill`]: same, plus the per-layer recurrent state after the last token.
//! - [`HybridLm::step`]: one token in, next-token logits out, state carried forward.
//!   Mamba layers cost O(1) time/memory per token; attention layers use a KV cache
//!   (bounded if `attn_window > 0`).

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use super::attention::{AttentionConfig, CausalSelfAttention, KvCache};
use super::mamba::{Mamba, MambaConfig, MambaState};
use super::mlp::{SwiGlu, SwiGluConfig};
use super::moe::{Moe, MoeConfig};
use super::norm::{RmsNorm, RmsNormConfig};

#[derive(Config, Debug)]
pub struct HybridLmConfig {
    pub vocab_size: usize,
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    /// SSM state size per channel.
    #[config(default = 16)]
    pub d_state: usize,
    #[config(default = 4)]
    pub d_conv: usize,
    #[config(default = 2)]
    pub expand: usize,
    /// Δ projection rank (0 = ceil(d_model / 16)).
    #[config(default = 0)]
    pub dt_rank: usize,
    /// Indices of layers that use attention instead of Mamba (e.g. `[3]` for a 1:7-ish mix).
    #[config(default = "Vec::new()")]
    pub attn_layers: Vec<usize>,
    #[config(default = 4)]
    pub n_heads: usize,
    /// Attention sliding window (0 = full causal).
    #[config(default = 0)]
    pub attn_window: usize,
    /// SwiGLU hidden size as a multiple of d_model (0 = no MLP sublayer).
    #[config(default = 0)]
    pub mlp_ratio: usize,
    /// Number of MoE experts per layer (0 = standard dense SwiGLU if mlp_ratio > 0; >0 = MoE).
    #[config(default = 0)]
    pub moe_experts: usize,
    /// Active experts per token in MoE layers.
    #[config(default = 1)]
    pub moe_top_k: usize,
    /// Dynamic routing probability threshold (0.0 = fixed top-k; >0.0 = dynamic threshold).
    #[config(default = 0.0)]
    pub moe_threshold: f64,
    /// Parallel scan chunk (0 = sequential reference scan).
    #[config(default = 64)]
    pub scan_chunk: usize,
    /// Opt-in work-efficient pair-reduction scan (overrides scan_chunk).
    // Optional for compatibility with configs saved before this experiment.
    pub work_efficient_scan: Option<bool>,
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub norm: RmsNorm<B>,
    pub mamba: Option<Mamba<B>>,
    pub attn: Option<CausalSelfAttention<B>>,
    pub mlp_norm: Option<RmsNorm<B>>,
    pub mlp: Option<SwiGlu<B>>,
    pub moe: Option<Moe<B>>,
}

/// Recurrent state of one block.
#[derive(Clone, Debug)]
pub enum LayerState<B: Backend> {
    Mamba(MambaState<B>),
    Attention(KvCache<B>),
}

/// Recurrent state of the whole model.
#[derive(Clone, Debug)]
pub struct HybridState<B: Backend> {
    pub layers: Vec<LayerState<B>>,
    /// Number of tokens consumed so far.
    pub position: usize,
}

impl<B: Backend> HybridState<B> {
    /// Total number of f32 values held in the state (SSM states, conv windows, KV caches).
    pub fn num_elements(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Mamba(s) => s.num_elements(),
                LayerState::Attention(c) => c.num_elements(),
            })
            .sum()
    }
}

#[derive(Module, Debug)]
pub struct HybridLm<B: Backend> {
    pub embedding: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    pub norm_f: RmsNorm<B>,
    pub lm_head: Linear<B>,
    pub vocab_size: usize,
}

impl HybridLmConfig {
    pub fn mamba_config(&self) -> MambaConfig {
        MambaConfig::new(self.d_model)
            .with_d_state(self.d_state)
            .with_d_conv(self.d_conv)
            .with_expand(self.expand)
            .with_dt_rank(self.dt_rank)
            .with_scan_chunk(self.scan_chunk)
            .with_work_efficient_scan(self.work_efficient_scan)
    }

    pub fn is_attention_layer(&self, index: usize) -> bool {
        self.attn_layers.contains(&index)
    }

    /// Short human readable layer layout, e.g. `M M M A M M`.
    pub fn layout(&self) -> String {
        (0..self.n_layers)
            .map(|i| if self.is_attention_layer(i) { "A" } else { "M" })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> HybridLm<B> {
        for &i in &self.attn_layers {
            assert!(
                i < self.n_layers,
                "attention layer index {i} >= n_layers {}",
                self.n_layers
            );
        }
        let blocks = (0..self.n_layers)
            .map(|i| {
                let (mamba, attn) = if self.is_attention_layer(i) {
                    let cfg = AttentionConfig::new(self.d_model)
                        .with_n_heads(self.n_heads)
                        .with_window(self.attn_window);
                    (None, Some(cfg.init(device)))
                } else {
                    (Some(self.mamba_config().init(device)), None)
                };
                let (mlp_norm, mlp, moe) = if self.mlp_ratio > 0 {
                    let norm = Some(RmsNormConfig::new(self.d_model).init(device));
                    let d_hidden = self.mlp_ratio * self.d_model;
                    if self.moe_experts > 0 {
                        let moe_cfg = MoeConfig::new(self.d_model, d_hidden)
                            .with_num_experts(self.moe_experts)
                            .with_top_k(self.moe_top_k)
                            .with_dynamic_threshold(self.moe_threshold);
                        (norm, None, Some(moe_cfg.init(device)))
                    } else {
                        let mlp_cfg = SwiGluConfig::new(self.d_model, d_hidden);
                        (norm, Some(mlp_cfg.init(device)), None)
                    }
                } else {
                    (None, None, None)
                };
                Block {
                    norm: RmsNormConfig::new(self.d_model).init(device),
                    mamba,
                    attn,
                    mlp_norm,
                    mlp,
                    moe,
                }
            })
            .collect();

        HybridLm {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            blocks,
            norm_f: RmsNormConfig::new(self.d_model).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            vocab_size: self.vocab_size,
        }
    }
}

impl<B: Backend> Block<B> {
    pub(crate) fn forward_with_state(
        &self,
        x: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, LayerState<B>, Option<Tensor<B, 1>>) {
        let normed = self.norm.forward(x.clone());
        let (mixed, state) = match (&self.mamba, &self.attn) {
            (Some(m), _) => {
                let (y, s) = m.forward_with_state(normed);
                (y, LayerState::Mamba(s))
            }
            (None, Some(a)) => {
                let (y, c) = a.forward_with_state(normed);
                (y, LayerState::Attention(c))
            }
            (None, None) => unreachable!("block without a mixer"),
        };
        let x = x + mixed;
        let (out, aux) = self.apply_mlp(x);
        (out, state, aux)
    }

    pub(crate) fn step(
        &self,
        x: Tensor<B, 2>,
        state: LayerState<B>,
    ) -> (Tensor<B, 2>, LayerState<B>) {
        let normed = self.norm.forward(x.clone());
        let (mixed, state) = match (&self.mamba, &self.attn, state) {
            (Some(m), _, LayerState::Mamba(s)) => {
                let (y, s) = m.step(normed, s);
                (y, LayerState::Mamba(s))
            }
            (None, Some(a), LayerState::Attention(c)) => {
                let (y, c) = a.step(normed, c);
                (y, LayerState::Attention(c))
            }
            _ => panic!("layer state does not match layer type"),
        };
        let x = x + mixed;
        let (out, _) = self.apply_mlp(x);
        (out, state)
    }

    fn apply_mlp<const D: usize>(&self, x: Tensor<B, D>) -> (Tensor<B, D>, Option<Tensor<B, 1>>) {
        if let Some(norm) = &self.mlp_norm {
            let normed = norm.forward(x.clone());
            if let Some(mlp) = &self.mlp {
                return (x + mlp.forward(normed), None);
            } else if let Some(moe) = &self.moe {
                let (moe_out, aux) = moe.forward_with_aux(normed);
                return (x + moe_out, Some(aux));
            }
        }
        (x, None)
    }

    fn init_state(&self, batch: usize, device: &B::Device) -> LayerState<B> {
        match &self.mamba {
            Some(m) => LayerState::Mamba(m.init_state(batch, device)),
            None => LayerState::Attention(KvCache::empty()),
        }
    }
}

impl<B: Backend> HybridLm<B> {
    /// Total parameters in the model.
    pub fn total_params(&self) -> usize {
        self.num_params()
    }

    /// Active parameters per token during inference.
    /// In dense models, active params = total params.
    /// In MoE models, active params only counts the `top_k` active experts per MoE layer!
    pub fn active_params_per_token(&self) -> usize {
        let mut total = self.embedding.weight.shape().num_elements()
            + self.norm_f.gamma.shape().num_elements()
            + self.lm_head.weight.shape().num_elements();
        if let Some(b) = &self.lm_head.bias {
            total += b.shape().num_elements();
        }
        for block in &self.blocks {
            total += block.norm.gamma.shape().num_elements();
            if let Some(m) = &block.mamba {
                total += m.num_params();
            }
            if let Some(a) = &block.attn {
                total += a.num_params();
            }
            if let Some(norm) = &block.mlp_norm {
                total += norm.gamma.shape().num_elements();
            }
            if let Some(mlp) = &block.mlp {
                total += mlp.num_params();
            }
            if let Some(moe) = &block.moe {
                total += moe.active_params_per_token();
            }
        }
        total
    }

    /// Logits for every position: `[batch, len] -> [batch, len, vocab]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.prefill(tokens).0
    }

    /// Forward pass with auxiliary router load-balancing loss accumulated across all MoE layers.
    pub fn forward_with_aux(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        let (logits, aux, _) = self.forward_observed(tokens, false);
        (logits, aux)
    }

    /// Real activation probes: last token of the first sequence in this batch.
    pub fn forward_observed(
        &self,
        tokens: Tensor<B, 2, Int>,
        observe: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>, Vec<serde_json::Value>) {
        fn probe<B: Backend>(name: &str, x: &Tensor<B, 3>) -> serde_json::Value {
            let [_, len, width] = x.dims();
            let values = x
                .clone()
                .slice([0..1, len - 1..len, 0..width])
                .detach()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .expect("f32 activation probe");
            serde_json::json!({"name": name, "values": values, "token_position": len-1,
                "sequence_index": 0})
        }
        let mut probes = Vec::new();
        let mut x = self.embedding.forward(tokens);
        if observe {
            probes.push(probe("embedding", &x));
        }
        let mut total_aux: Option<Tensor<B, 1>> = None;
        for (index, block) in self.blocks.iter().enumerate() {
            let (y, _, aux) = block.forward_with_state(x);
            x = y;
            if observe {
                probes.push(probe(&format!("blocks.{index}"), &x));
            }
            if let Some(a) = aux {
                total_aux = Some(match total_aux {
                    Some(prev) => prev + a,
                    None => a,
                });
            }
        }
        let logits = self.lm_head.forward(self.norm_f.forward(x));
        if observe {
            probes.push(probe("logits", &logits));
        }
        (logits, total_aux, probes)
    }

    /// Processes a prompt in parallel and returns `(logits, state after the last token)`.
    pub fn prefill(&self, tokens: Tensor<B, 2, Int>) -> (Tensor<B, 3>, HybridState<B>) {
        let len = tokens.dims()[1];
        let mut x = self.embedding.forward(tokens);
        let mut layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (y, s, _) = block.forward_with_state(x);
            x = y;
            layers.push(s);
        }
        let logits = self.lm_head.forward(self.norm_f.forward(x));
        (
            logits,
            HybridState {
                layers,
                position: len,
            },
        )
    }

    /// Empty state (no tokens consumed) for `batch` sequences.
    pub fn init_state(&self, batch: usize, device: &B::Device) -> HybridState<B> {
        HybridState {
            layers: self
                .blocks
                .iter()
                .map(|b| b.init_state(batch, device))
                .collect(),
            position: 0,
        }
    }

    /// Consumes one token per sequence (`tokens`: `[batch]`) and returns next-token logits
    /// `[batch, vocab]` together with the updated state.
    pub fn step(
        &self,
        tokens: Tensor<B, 1, Int>,
        state: HybridState<B>,
    ) -> (Tensor<B, 2>, HybridState<B>) {
        let batch = tokens.dims()[0];
        let x: Tensor<B, 2> = self
            .embedding
            .forward(tokens.reshape([batch, 1]))
            .reshape([batch, self.embedding.weight.dims()[1]]);
        self.step_embedding(x, state)
    }

    /// Evaluates pre-embedded sequence `[batch, len, d_model]` directly through the hybrid backbone.
    pub fn forward_embeddings(&self, embeddings: Tensor<B, 3>) -> Tensor<B, 3> {
        self.prefill_embeddings(embeddings).0
    }

    /// Processes pre-embedded sequence in parallel and returns `(logits, state after the last token)`.
    pub fn prefill_embeddings(&self, embeddings: Tensor<B, 3>) -> (Tensor<B, 3>, HybridState<B>) {
        let len = embeddings.dims()[1];
        let mut x = embeddings;
        let mut layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (y, s, _) = block.forward_with_state(x);
            x = y;
            layers.push(s);
        }
        let logits = self.lm_head.forward(self.norm_f.forward(x));
        (
            logits,
            HybridState {
                layers,
                position: len,
            },
        )
    }

    /// Consumes one embedded vector per sequence (`x`: `[batch, d_model]`) and returns next-token logits
    /// `[batch, vocab]` together with the updated state.
    pub fn step_embedding(
        &self,
        x: Tensor<B, 2>,
        state: HybridState<B>,
    ) -> (Tensor<B, 2>, HybridState<B>) {
        let mut cur = x;
        let mut layers = Vec::with_capacity(self.blocks.len());
        for (block, s) in self.blocks.iter().zip(state.layers) {
            let (y, s) = block.step(cur, s);
            cur = y;
            layers.push(s);
        }
        let logits = self.lm_head.forward(self.norm_f.forward(cur));
        (
            logits,
            HybridState {
                layers,
                position: state.position + 1,
            },
        )
    }

    /// Changes the scan implementation of every Mamba layer (0 = sequential).
    pub fn with_scan_chunk(mut self, chunk: usize) -> Self {
        for block in self.blocks.iter_mut() {
            if let Some(m) = block.mamba.as_mut() {
                m.scan_chunk = chunk;
                m.work_efficient_scan = false;
            }
        }
        self
    }
}
