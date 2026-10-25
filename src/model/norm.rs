//! Root Mean Square layer normalisation.

use burn::module::Param;
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct RmsNormConfig {
    pub d_model: usize,
    #[config(default = 1e-5)]
    pub eps: f32,
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub eps: f32,
}

impl RmsNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        RmsNorm {
            gamma: Param::from_tensor(Tensor::ones([self.d_model], device)),
            eps: self.eps,
        }
    }
}

impl<B: Backend> RmsNorm<B> {
    /// Normalises over the last dimension; works for `[B, L, D]` sequences and `[B, D]` steps.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let rms = (x.clone().powi_scalar(2).mean_dim(D - 1) + self.eps).sqrt();
        (x / rms) * self.gamma.val().unsqueeze::<D>()
    }
}
