//! SwiGLU feed-forward network (optional per-layer channel mixer, as in Jamba).

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

#[derive(Config, Debug)]
pub struct SwiGluConfig {
    pub d_model: usize,
    pub d_hidden: usize,
}

#[derive(Module, Debug)]
pub struct SwiGlu<B: Backend> {
    pub gate: Linear<B>,
    pub up: Linear<B>,
    pub down: Linear<B>,
}

impl SwiGluConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGlu<B> {
        let linear = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        SwiGlu {
            gate: linear(self.d_model, self.d_hidden),
            up: linear(self.d_model, self.d_hidden),
            down: linear(self.d_hidden, self.d_model),
        }
    }
}

impl<B: Backend> SwiGlu<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.down
            .forward(silu(self.gate.forward(x.clone())) * self.up.forward(x))
    }
}
