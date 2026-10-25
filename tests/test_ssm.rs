use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::*;
use burn::tensor::Distribution;

use sail::model::hybrid::HybridLmConfig;
use sail::model::mamba::MambaConfig;
use sail::model::moe::MoeConfig;
use sail::model::scan::{chunked_parallel_scan, sequential_scan};

type TestBackend = NdArray<f32>;

#[test]
fn test_parallel_scan_matches_sequential() {
    let device = NdArrayDevice::Cpu;
    let [b, l, d, n] = [2, 32, 8, 4];

    let a =
        Tensor::<TestBackend, 4>::random([b, l, d, n], Distribution::Uniform(0.5, 0.95), &device);
    let u = Tensor::<TestBackend, 4>::random([b, l, d, n], Distribution::Normal(0.0, 1.0), &device);

    let h_seq = sequential_scan(a.clone(), u.clone());
    let h_par = chunked_parallel_scan(a, u, 8);

    let diff = (h_seq - h_par).abs().max().into_scalar();
    assert!(diff < 1e-4, "parallel scan diff {diff} exceeds tolerance");
}

#[test]
fn test_mamba_recurrent_step_matches_sequence() {
    let device = NdArrayDevice::Cpu;
    let d_model = 16;
    let seq_len = 8;

    let cfg = MambaConfig::new(d_model)
        .with_d_state(4)
        .with_d_conv(4)
        .with_expand(2)
        .with_dt_rank(2);
    let mamba = cfg.init::<TestBackend>(&device);

    let x = Tensor::<TestBackend, 3>::random(
        [1, seq_len, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (y_seq, _) = mamba.forward_with_state(x.clone());

    // Step token by token
    let mut state = mamba.init_state(1, &device);
    let mut step_outputs = Vec::new();
    for t in 0..seq_len {
        let x_t: Tensor<TestBackend, 2> = x.clone().narrow(1, t, 1).squeeze(1);
        let (y_t, next_state) = mamba.step(x_t, state);
        state = next_state;
        step_outputs.push(y_t.unsqueeze_dim::<3>(1));
    }
    let y_stepped = Tensor::cat(step_outputs, 1);

    let max_diff = (y_seq - y_stepped).abs().max().into_scalar();
    assert!(
        max_diff < 1e-4,
        "mamba step diff {max_diff} exceeds tolerance"
    );
}

#[test]
fn test_hybrid_jamba_step_matches_prefill() {
    let device = NdArrayDevice::Cpu;
    let vocab_size = 32;
    let seq_len = 6;

    let cfg = HybridLmConfig::new(vocab_size)
        .with_d_model(16)
        .with_n_layers(3)
        .with_d_state(4)
        .with_d_conv(3)
        .with_attn_layers(vec![1]) // Layer 0: Mamba, Layer 1: Attn, Layer 2: Mamba
        .with_n_heads(2)
        .with_mlp_ratio(2);

    let model = cfg.init::<TestBackend>(&device);

    let tokens = Tensor::<TestBackend, 2, Int>::from_data([[3, 7, 12, 1, 9, 15]], &device);
    let (logits_prefill, _) = model.prefill(tokens.clone());

    // Step autoregressively
    let mut state = model.init_state(1, &device);
    let mut step_logits = Vec::new();
    for t in 0..seq_len {
        let tok: Tensor<TestBackend, 1, Int> = tokens.clone().narrow(1, t, 1).reshape([1]);
        let (out, next_state) = model.step(tok, state);
        state = next_state;
        step_logits.push(out.unsqueeze_dim::<3>(1));
    }
    let logits_stepped = Tensor::cat(step_logits, 1);

    let max_diff = (logits_prefill - logits_stepped).abs().max().into_scalar();
    assert!(
        max_diff < 1e-4,
        "hybrid step diff {max_diff} exceeds tolerance"
    );
}

#[test]
fn test_moe_dynamic_active_parameters() {
    let device = NdArrayDevice::Cpu;
    let d_model = 16;
    let d_hidden = 32;
    let num_experts = 4;
    let top_k = 1;

    let cfg = MoeConfig::new(d_model, d_hidden)
        .with_num_experts(num_experts)
        .with_top_k(top_k);
    let moe = cfg.init::<TestBackend>(&device);

    let total_params = moe.total_params();
    let active_params = moe.active_params_per_token();

    assert!(
        active_params < total_params,
        "active params ({active_params}) must be less than total ({total_params})"
    );
    // With 4 experts and top_k=1, active expert params are ~1/4 of total expert params
    let expert_params = 3 * d_model * d_hidden;
    assert_eq!(
        total_params,
        d_model * num_experts + num_experts * expert_params
    );
    assert_eq!(active_params, d_model * num_experts + top_k * expert_params);

    // Forward pass with aux load balancing loss
    let x =
        Tensor::<TestBackend, 3>::random([2, 8, d_model], Distribution::Normal(0.0, 1.0), &device);
    let (out, aux_loss) = moe.forward_with_aux(x);
    assert_eq!(out.dims(), [2, 8, d_model]);
    assert!(aux_loss.into_scalar() >= 0.0);
}

#[test]
fn test_moe_hybrid_active_params_reporting() {
    let device = NdArrayDevice::Cpu;
    let cfg_dense = HybridLmConfig::new(65)
        .with_d_model(32)
        .with_n_layers(2)
        .with_mlp_ratio(2)
        .with_moe_experts(0);
    let dense_model = cfg_dense.init::<TestBackend>(&device);
    assert_eq!(
        dense_model.total_params(),
        dense_model.active_params_per_token()
    );

    let cfg_moe = HybridLmConfig::new(65)
        .with_d_model(32)
        .with_n_layers(2)
        .with_mlp_ratio(2)
        .with_moe_experts(4)
        .with_moe_top_k(1);
    let moe_model = cfg_moe.init::<TestBackend>(&device);
    assert!(moe_model.active_params_per_token() < moe_model.total_params());
    println!(
        "MoE model: {} total params, {} active params/token (ratio: {:.1}%)",
        moe_model.total_params(),
        moe_model.active_params_per_token(),
        100.0 * moe_model.active_params_per_token() as f64 / moe_model.total_params() as f64
    );
}
