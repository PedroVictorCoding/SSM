//! Mamba (selective SSM) mixer block — Gu & Dao, 2023.
//!
//! ```text
//!   x ──in_proj──┬── x_in ── causal depthwise conv ── SiLU ── selective SSM ──┐
//!                │                                                            ⊙ ── out_proj ── y
//!                └── z ──────────────────────────────────────────── SiLU ─────┘
//! ```
//!
//! The SSM is input dependent: `Δ`, `B` and `C` are projected from the conv output,
//! discretised as `Ā = exp(Δ·A)`, `B̄ = Δ·B`, and scanned as `h_t = Ā_t h_{t-1} + B̄_t x_t`,
//! `y_t = C_t·h_t + D ⊙ x_t`.
//!
//! Two execution paths share the same parameters:
//! - [`Mamba::forward_with_state`]: whole sequence at once with the parallel scan (training,
//!   prompt prefill). Also returns the final recurrent state.
//! - [`Mamba::step`]: one token, carrying a fixed-size [`MambaState`] — O(1) time and memory
//!   per generated token regardless of context length.

use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::{silu, softplus};

use super::scan::{linear_scan, work_efficient_scan};

#[derive(Config, Debug)]
pub struct MambaConfig {
    pub d_model: usize,
    /// SSM state size N per channel.
    #[config(default = 16)]
    pub d_state: usize,
    /// Causal depthwise convolution width.
    #[config(default = 4)]
    pub d_conv: usize,
    /// Inner expansion factor: `d_inner = expand * d_model`.
    #[config(default = 2)]
    pub expand: usize,
    /// Rank of the Δ projection; 0 = `ceil(d_model / 16)` as in the paper.
    #[config(default = 0)]
    pub dt_rank: usize,
    #[config(default = 0.001)]
    pub dt_min: f64,
    #[config(default = 0.1)]
    pub dt_max: f64,
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,
    /// Parallel scan chunk size; 0 = sequential reference scan.
    #[config(default = 64)]
    pub scan_chunk: usize,
    /// Opt-in pair-reduction scan with linear elementwise work.
    // Optional for compatibility with configs saved before this experiment.
    pub work_efficient_scan: Option<bool>,
}

impl MambaConfig {
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    pub fn resolved_dt_rank(&self) -> usize {
        if self.dt_rank == 0 {
            self.d_model.div_ceil(16)
        } else {
            self.dt_rank
        }
    }
}

#[derive(Module, Debug)]
pub struct Mamba<B: Backend> {
    pub in_proj: Linear<B>,
    /// Depthwise causal conv weights `[d_inner, d_conv]`; tap `d_conv - 1` sees the current token.
    pub conv_weight: Param<Tensor<B, 2>>,
    pub conv_bias: Param<Tensor<B, 1>>,
    pub x_proj: Linear<B>,
    pub dt_proj: Linear<B>,
    /// `A = -exp(a_log)`, shape `[d_inner, d_state]`.
    pub a_log: Param<Tensor<B, 2>>,
    pub d: Param<Tensor<B, 1>>,
    pub out_proj: Linear<B>,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub scan_chunk: usize,
    pub work_efficient_scan: bool,
}

/// Fixed-size recurrent state of one Mamba layer.
#[derive(Clone, Debug)]
pub struct MambaState<B: Backend> {
    /// Last `d_conv - 1` conv inputs, `[batch, d_conv - 1, d_inner]`.
    pub conv: Tensor<B, 3>,
    /// SSM hidden state, `[batch, d_inner, d_state]`.
    pub ssm: Tensor<B, 3>,
}

impl<B: Backend> MambaState<B> {
    pub fn num_elements(&self) -> usize {
        self.conv.shape().num_elements() + self.ssm.shape().num_elements()
    }
}

impl MambaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba<B> {
        assert!(self.d_conv >= 2, "d_conv must be >= 2");
        let d_inner = self.d_inner();
        let dt_rank = self.resolved_dt_rank();
        let n = self.d_state;

        let in_proj = LinearConfig::new(self.d_model, 2 * d_inner)
            .with_bias(false)
            .init(device);

        // PyTorch's default depthwise Conv1d init: U(-1/sqrt(k), 1/sqrt(k)).
        let bound = 1.0 / (self.d_conv as f64).sqrt();
        let conv_weight = Param::from_tensor(Tensor::random(
            [d_inner, self.d_conv],
            Distribution::Uniform(-bound, bound),
            device,
        ));
        let conv_bias = Param::from_tensor(Tensor::random(
            [d_inner],
            Distribution::Uniform(-bound, bound),
            device,
        ));

        let x_proj = LinearConfig::new(d_inner, dt_rank + 2 * n)
            .with_bias(false)
            .init(device);

        // Δ projection: weights U(-r^-0.5, r^-0.5); bias = softplus⁻¹(dt) with dt log-uniform in
        // [dt_min, dt_max], so the initial step sizes span several time scales.
        let std = (dt_rank as f64).powf(-0.5);
        let mut dt_proj = LinearConfig::new(dt_rank, d_inner)
            .with_bias(true)
            .with_initializer(Initializer::Uniform {
                min: -std,
                max: std,
            })
            .init(device);
        let (log_min, log_max) = (self.dt_min.ln(), self.dt_max.ln());
        let dt = Tensor::<B, 1>::random([d_inner], Distribution::Uniform(0.0, 1.0), device)
            .mul_scalar(log_max - log_min)
            .add_scalar(log_min)
            .exp()
            .clamp_min(self.dt_init_floor);
        // softplus⁻¹(x) = x + log(1 - exp(-x))
        let inv_softplus_dt = dt.clone() + dt.neg().exp().neg().add_scalar(1.0).log();
        dt_proj.bias = Some(Param::from_tensor(inv_softplus_dt));

        // S4D-real initialisation: A[:, j] = -(j + 1).
        let a_values: Vec<f32> = (0..d_inner)
            .flat_map(|_| (1..=n).map(|j| (j as f32).ln()))
            .collect();
        let a_log = Param::from_tensor(Tensor::from_data(
            TensorData::new(a_values, [d_inner, n]),
            device,
        ));

        let d = Param::from_tensor(Tensor::ones([d_inner], device));

        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(false)
            .init(device);

        Mamba {
            in_proj,
            conv_weight,
            conv_bias,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            d_inner,
            d_state: n,
            d_conv: self.d_conv,
            dt_rank,
            scan_chunk: self.scan_chunk,
            work_efficient_scan: self.work_efficient_scan.unwrap_or(false),
        }
    }
}

impl<B: Backend> Mamba<B> {
    /// Zero state for `batch` independent sequences.
    pub fn init_state(&self, batch: usize, device: &B::Device) -> MambaState<B> {
        MambaState {
            conv: Tensor::zeros([batch, self.d_conv - 1, self.d_inner], device),
            ssm: Tensor::zeros([batch, self.d_inner, self.d_state], device),
        }
    }

    /// Continuous-time state matrix `A = -exp(a_log)` (strictly negative ⇒ stable).
    fn a(&self) -> Tensor<B, 2> {
        self.a_log.val().exp().neg()
    }

    /// Splits the x_proj output into `(Δ, B, C)` along the last dimension.
    fn split_x_dbl<const D: usize>(
        &self,
        x_dbl: Tensor<B, D>,
    ) -> (Tensor<B, D>, Tensor<B, D>, Tensor<B, D>) {
        let n = self.d_state;
        let r = self.dt_rank;
        let mut delta_shape = x_dbl.dims();
        let dt = x_dbl.clone().narrow(D - 1, 0, r);
        let b = x_dbl.clone().narrow(D - 1, r, n);
        let c = x_dbl.narrow(D - 1, r + n, n);
        // dt is a narrow view into the joint [dt, B, C] projection. Materialize
        // its rows before the small-rank matmul; older JIT kernels otherwise
        // misread the row stride on GPU. Keep this reshape in the autodiff graph.
        let tokens: usize = delta_shape[..D - 1].iter().product();
        delta_shape[D - 1] = self.d_inner;
        let delta = softplus(
            self.dt_proj
                .forward(dt.reshape([tokens, r]))
                .reshape(delta_shape),
            1.0,
        );
        (delta, b, c)
    }

    /// Full-sequence forward. `x`: `[batch, len, d_model]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_state(x).0
    }

    /// Full-sequence forward that also returns the recurrent state after the last token, so that
    /// generation can continue with [`Mamba::step`].
    pub fn forward_with_state(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, MambaState<B>) {
        let [batch, len, _] = x.dims();
        let di = self.d_inner;
        let k = self.d_conv;

        let xz = self.in_proj.forward(x);
        let x_in = xz.clone().narrow(2, 0, di);
        let z = xz.narrow(2, di, di);

        // Causal depthwise conv via left zero-padding and k shifted multiply-adds.
        // Keep the zero prefix tracked: Burn 0.16 cat backward otherwise loses
        // its offset before the tracked suffix. Also materialize strided x_in.
        let padding = x_in
            .clone()
            .narrow(1, 0, 1)
            .mul_scalar(0.0)
            .repeat_dim(1, k - 1);
        let padded = super::cat_dense(vec![padding, x_in], 1); // [B, L + k - 1, Di]
        let w = self.conv_weight.val(); // [Di, k]
        let mut conv = self.conv_bias.val().unsqueeze::<3>(); // [1, 1, Di]
        for j in 0..k {
            let w_j = w.clone().narrow(1, j, 1).reshape([1, 1, di]);
            conv = conv + padded.clone().narrow(1, j, len) * w_j;
        }
        let conv_state = padded.narrow(1, len, k - 1); // last k-1 inputs
        let xc = silu(conv); // [B, L, Di]

        // Input-dependent SSM parameters.
        let (delta, b, c) = self.split_x_dbl(self.x_proj.forward(xc.clone()));

        // Discretisation: Ā = exp(Δ ⊗ A), B̄x = (Δ ⊙ x) ⊗ B — all [B, L, Di, N].
        let a = self.a().unsqueeze::<4>(); // [1, 1, Di, N]
        let da = (delta.clone().unsqueeze_dim::<4>(3) * a).exp();
        let dbx = (delta * xc.clone()).unsqueeze_dim::<4>(3) * b.unsqueeze_dim::<4>(2);

        let h = if self.work_efficient_scan {
            work_efficient_scan(da, dbx)
        } else {
            linear_scan(da, dbx, self.scan_chunk)
        }; // [B, L, Di, N]
        let ssm_state: Tensor<B, 3> = h.clone().narrow(1, len - 1, 1).squeeze(1);

        // y_t = C_t · h_t (contract over N) + D ⊙ x_t
        let y: Tensor<B, 3> = h.matmul(c.unsqueeze_dim::<4>(3)).reshape([batch, len, di]);
        let y = y + xc * self.d.val().unsqueeze::<3>();

        let out = self.out_proj.forward(y * silu(z));
        (
            out,
            MambaState {
                conv: conv_state,
                ssm: ssm_state,
            },
        )
    }

    /// Single-token recurrent step. `x`: `[batch, d_model]`.
    pub fn step(&self, x: Tensor<B, 2>, state: MambaState<B>) -> (Tensor<B, 2>, MambaState<B>) {
        let [batch, _] = x.dims();
        let di = self.d_inner;
        let k = self.d_conv;

        let xz = self.in_proj.forward(x);
        let x_in = xz.clone().narrow(1, 0, di);
        let z = xz.narrow(1, di, di);

        // Sliding conv window: [B, k, Di] = previous k-1 inputs + current input.
        let window = super::cat_dense(vec![state.conv, x_in.unsqueeze_dim::<3>(1)], 1);
        let w = self.conv_weight.val().transpose().unsqueeze::<3>(); // [1, k, Di]
        let conv: Tensor<B, 2> =
            (window.clone() * w).sum_dim(1).reshape([batch, di]) + self.conv_bias.val().unsqueeze();
        let new_conv = window.narrow(1, 1, k - 1);
        let xc = silu(conv); // [B, Di]

        let (delta, b, c) = self.split_x_dbl(self.x_proj.forward(xc.clone()));

        let da = (delta.clone().unsqueeze_dim::<3>(2) * self.a().unsqueeze::<3>()).exp(); // [B, Di, N]
        let dbx = (delta * xc.clone()).unsqueeze_dim::<3>(2) * b.unsqueeze_dim::<3>(1);
        let h = da * state.ssm + dbx;

        let y: Tensor<B, 2> = h
            .clone()
            .matmul(c.unsqueeze_dim::<3>(2))
            .reshape([batch, di]);
        let y = y + xc * self.d.val().unsqueeze();

        let out = self.out_proj.forward(y * silu(z));
        (
            out,
            MambaState {
                conv: new_conv,
                ssm: h,
            },
        )
    }
}
