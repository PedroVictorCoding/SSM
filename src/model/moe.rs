//! Mixture of Experts (MoE) with Top-K and Dynamic Threshold Routing (Jamba-style).
//!
//! Replaces dense MLPs with a sparsely-gated bank of `num_experts` SwiGLU networks.
//! For each token, a router selects either:
//! - Fixed Top-K: exactly `K` active experts per token (e.g. 1 out of 4, or 2 out of 8).
//! - Dynamic Threshold: tokens dynamically activate 1..K experts depending on router confidence,
//!   achieving variable active parameter capacity per token.
//!
//! Includes an auxiliary load-balancing loss (Switch Transformer / GShard) to prevent expert collapse:
//! `L_aux = E * sum_{e=1}^E (f_e * P_e)`

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::mlp::{SwiGlu, SwiGluConfig};

#[derive(Config, Debug)]
pub struct MoeConfig {
    pub d_model: usize,
    pub d_hidden: usize,
    /// Total number of expert networks.
    #[config(default = 4)]
    pub num_experts: usize,
    /// Maximum number of active experts routed per token.
    #[config(default = 1)]
    pub top_k: usize,
    /// Dynamic threshold: if > 0.0, only experts with router probability >= threshold
    /// (up to top_k) will be activated for that token, allowing dynamic active parameter count.
    #[config(default = 0.0)]
    pub dynamic_threshold: f64,
}

#[derive(Module, Debug)]
pub struct Moe<B: Backend> {
    pub router: Linear<B>,
    pub experts: Vec<SwiGlu<B>>,
    pub num_experts: usize,
    pub top_k: usize,
    pub dynamic_threshold: f64,
    pub d_model: usize,
    pub d_hidden: usize,
}

impl MoeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Moe<B> {
        assert!(self.num_experts >= 1, "num_experts must be >= 1");
        let top_k = self.top_k.clamp(1, self.num_experts);

        let router = LinearConfig::new(self.d_model, self.num_experts)
            .with_bias(false)
            .init(device);

        let expert_cfg = SwiGluConfig::new(self.d_model, self.d_hidden);
        let experts = (0..self.num_experts)
            .map(|_| expert_cfg.init(device))
            .collect();

        Moe {
            router,
            experts,
            num_experts: self.num_experts,
            top_k,
            dynamic_threshold: self.dynamic_threshold,
            d_model: self.d_model,
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> Moe<B> {
    /// Number of total parameters across all experts and router.
    pub fn total_params(&self) -> usize {
        let router_p = self.d_model * self.num_experts;
        let per_expert_p = 3 * self.d_model * self.d_hidden; // gate, up, down
        router_p + self.num_experts * per_expert_p
    }

    /// Average active parameters per token (router + top_k * expert size).
    pub fn active_params_per_token(&self) -> usize {
        let router_p = self.d_model * self.num_experts;
        let per_expert_p = 3 * self.d_model * self.d_hidden;
        router_p + self.top_k * per_expert_p
    }

    /// Forward pass for arbitrary dimension tensors, returning `(output, aux_loss)`.
    /// `x`: shape `[..., d_model]`.
    pub fn forward_with_aux<const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> (Tensor<B, D>, Tensor<B, 1>) {
        let dims = x.dims();
        let d_model = dims[D - 1];
        let num_tokens: usize = dims[..D - 1].iter().product();
        let device = x.device();

        // Flatten to 2D: [num_tokens, d_model]
        let x_2d: Tensor<B, 2> = x.reshape([num_tokens, d_model]);

        // Router logits and probabilities: [num_tokens, num_experts]
        let router_logits = self.router.forward(x_2d.clone());
        let router_probs = softmax(router_logits.clone(), 1);

        // Auxiliary load balancing loss:
        // L_aux = num_experts * sum_e (f_e * P_e)
        // P_e = mean(router_probs over tokens)
        let p_e = router_probs.clone().mean_dim(0).reshape([self.num_experts]); // [E]

        // Find top-1 / top-k expert indices
        let (weights, f_e) = if self.top_k == 1 {
            // Fast top-1 path
            let max_indices = router_logits.clone().argmax(1); // [num_tokens, 1]
            let mut one_hot_f = Vec::with_capacity(self.num_experts);
            let mut w_cols = Vec::with_capacity(self.num_experts);

            for e in 0..self.num_experts {
                // mask where token chooses expert e: [num_tokens, 1]
                let e_tensor = Tensor::<B, 2, Int>::full([num_tokens, 1], e as i64, &device);
                let is_e = max_indices.clone().equal(e_tensor).float();
                let p_col = router_probs.clone().narrow(1, e, 1);
                let w = if self.dynamic_threshold > 0.0 {
                    // Dynamic threshold: suppress if router confidence < threshold
                    let thresh = Tensor::<B, 2>::full(
                        [num_tokens, 1],
                        self.dynamic_threshold as f32,
                        &device,
                    );
                    let above_thresh = p_col.clone().greater_equal(thresh).float();
                    p_col * is_e.clone() * above_thresh
                } else {
                    p_col * is_e.clone()
                };
                one_hot_f.push(is_e.mean());
                w_cols.push(w);
            }
            let f_e = Tensor::cat(one_hot_f, 0); // [E]
            let weights = Tensor::cat(w_cols, 1); // [num_tokens, E]
            (weights, f_e)
        } else {
            // Top-K path: use topk_with_indices
            let (top_val, top_idx) = router_logits.clone().topk_with_indices(self.top_k, 1);
            let top_probs = softmax(top_val, 1); // re-normalize over top-k: [num_tokens, top_k]

            let mut w_cols = Vec::with_capacity(self.num_experts);
            let mut f_cols = Vec::with_capacity(self.num_experts);

            for e in 0..self.num_experts {
                let e_idx = Tensor::<B, 2, Int>::full([num_tokens, self.top_k], e as i64, &device);
                let is_e = top_idx.clone().equal(e_idx).float(); // [num_tokens, top_k]
                let e_weight = (top_probs.clone() * is_e.clone()).sum_dim(1); // [num_tokens, 1]

                let active_w = if self.dynamic_threshold > 0.0 {
                    let thresh = Tensor::<B, 2>::full(
                        [num_tokens, 1],
                        self.dynamic_threshold as f32,
                        &device,
                    );
                    let above = e_weight.clone().greater_equal(thresh).float();
                    e_weight * above
                } else {
                    e_weight
                };

                let token_used_e = is_e.sum_dim(1).clamp_max(1.0); // 1 if expert e is among top-k
                f_cols.push(token_used_e.mean());
                w_cols.push(active_w);
            }

            let f_e = Tensor::cat(f_cols, 0); // [E]
            let weights = Tensor::cat(w_cols, 1); // [num_tokens, E]
            (weights, f_e)
        };

        let aux_loss = (f_e * p_e).sum().mul_scalar(self.num_experts as f32);

        // Compute expert outputs and accumulate with routing weights:
        // out = sum_e weights[:, e] * Expert_e(x)
        let mut out_accum = Tensor::<B, 2>::zeros([num_tokens, d_model], &device);
        for (e, expert) in self.experts.iter().enumerate() {
            let w_e: Tensor<B, 2> = weights.clone().narrow(1, e, 1); // [num_tokens, 1]
            let expert_out = expert.forward(x_2d.clone()); // [num_tokens, d_model]
            out_accum = out_accum + expert_out * w_e;
        }

        let out = out_accum.reshape(dims);
        (out, aux_loss)
    }

    /// Forward pass (without returning aux loss directly).
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.forward_with_aux(x).0
    }
}
