use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::Distribution;

use sail::generate::{SamplingConfig, generate_naive, generate_stateful};
use sail::model::HybridLmConfig;
use sail::model::mamba::MambaConfig;
use sail::model::moe::MoeConfig;
use sail::model::scan::{chunked_parallel_scan, sequential_scan};

type Backend = NdArray<f32>;
type AdBackend = Autodiff<Backend>;

#[test]
fn test_scan_equivalence() {
    let device = NdArrayDevice::Cpu;
    let [b, l, d, n] = [2, 37, 8, 4];
    let shape = [b, l, d, n];

    let a = Tensor::<Backend, 4>::random(shape, Distribution::Uniform(0.1, 0.9), &device);
    let u = Tensor::<Backend, 4>::random(shape, Distribution::Normal(0.0, 1.0), &device);

    let h_seq = sequential_scan(a.clone(), u.clone());
    let h_par = chunked_parallel_scan(a, u, 8);

    let diff = (h_seq - h_par).abs().max().into_scalar();
    assert!(
        diff < 1e-4,
        "Parallel scan and sequential scan deviated by {diff}"
    );
}

#[test]
fn test_mamba_step_equals_forward() {
    let device = NdArrayDevice::Cpu;
    let cfg = MambaConfig::new(16)
        .with_d_state(8)
        .with_d_conv(4)
        .with_expand(2)
        .with_scan_chunk(8);
    let mamba = cfg.init::<Backend>(&device);

    let seq_len = 12;
    let x = Tensor::<Backend, 3>::random([1, seq_len, 16], Distribution::Normal(0.0, 1.0), &device);

    // Full sequence forward
    let (y_seq, _) = mamba.forward_with_state(x.clone());

    // Step-by-step decoding
    let mut state = mamba.init_state(1, &device);
    let mut y_steps = Vec::new();
    for t in 0..seq_len {
        let x_t: Tensor<Backend, 2> = x.clone().narrow(1, t, 1).squeeze(1);
        let (y_t, next_state) = mamba.step(x_t, state);
        state = next_state;
        y_steps.push(y_t.unsqueeze_dim::<3>(1));
    }
    let y_step_all = Tensor::cat(y_steps, 1);

    let diff = (y_seq - y_step_all).abs().max().into_scalar();
    assert!(
        diff < 1e-4,
        "Mamba step-by-step and full forward deviated by {diff}"
    );
}

#[test]
fn test_hybrid_stateful_decoding_equals_naive() {
    let device = NdArrayDevice::Cpu;
    // Hybrid with 1 Attention layer and 2 Mamba layers + MoE
    let cfg = HybridLmConfig::new(32)
        .with_d_model(16)
        .with_n_layers(3)
        .with_attn_layers(vec![1])
        .with_n_heads(2)
        .with_mlp_ratio(2)
        .with_moe_experts(4)
        .with_moe_top_k(1);
    let model = cfg.init::<Backend>(&device);

    let prompt = vec![5u32, 12, 28, 3];
    let new_tokens = 8;
    let greedy = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        seed: 42,
        ..Default::default()
    };

    let (stateful_out, _) = generate_stateful(&model, &prompt, new_tokens, &greedy, &device);
    let (naive_out, _) = generate_naive(&model, &prompt, new_tokens, &greedy, &device);

    assert_eq!(
        stateful_out, naive_out,
        "Stateful O(1) decoding produced different tokens from naive O(L^2) decoding!"
    );
}

#[test]
fn test_moe_active_parameters_and_aux_loss() {
    let device = NdArrayDevice::Cpu;
    let cfg = MoeConfig::new(16, 32)
        .with_num_experts(4)
        .with_top_k(2)
        .with_dynamic_threshold(0.1);
    let moe = cfg.init::<AdBackend>(&device);

    assert_eq!(moe.num_experts, 4);
    assert_eq!(moe.top_k, 2);
    // Router: 16*4 = 64. Each expert: 3 * 16 * 32 = 1536. Total: 64 + 4*1536 = 6208.
    assert_eq!(moe.total_params(), 64 + 4 * 1536);
    // Active per token: 64 + 2 * 1536 = 3136
    assert_eq!(moe.active_params_per_token(), 64 + 2 * 1536);

    let x = Tensor::<AdBackend, 2>::random([10, 16], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();
    let (out, aux_loss) = moe.forward_with_aux(x.clone());

    assert_eq!(out.dims(), [10, 16]);
    let aux_val: f32 = aux_loss.clone().into_scalar();
    assert!(aux_val >= 0.0, "Aux loss must be non-negative");

    // Check backprop works through MoE
    let loss = out.sum() + aux_loss;
    let grads = loss.backward();
    assert!(x.grad(&grads).is_some());
}
