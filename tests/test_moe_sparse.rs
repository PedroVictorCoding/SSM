use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::Distribution;

use sail::model::moe::MoeConfig;

type NdBackend = NdArray<f32>;
type AdBackend = Autodiff<NdBackend>;

#[test]
fn test_sparse_expert_dispatch_matches_dense_reference_and_gradients() {
    let device = NdArrayDevice::Cpu;
    let [batch_size, d_model, d_hidden, num_experts, top_k] = [8, 16, 32, 8, 1];

    let dense_cfg = MoeConfig::new(d_model, d_hidden)
        .with_num_experts(num_experts)
        .with_top_k(top_k);

    // Initialize dense model and clone exact weights for sparse dispatch
    <NdBackend as burn::prelude::Backend>::seed(12345);
    let dense_moe = dense_cfg.init::<AdBackend>(&device);
    let mut sparse_moe = dense_moe.clone();
    sparse_moe.sparse_dispatch = true;

    let x = Tensor::<AdBackend, 2>::random(
        [batch_size, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Forward pass
    let (out_dense, aux_dense) = dense_moe.forward_with_aux(x.clone());
    let (out_sparse, aux_sparse) = sparse_moe.forward_with_aux(x.clone());

    // Compare forward outputs
    let out_diff: f32 = (out_dense.clone() - out_sparse.clone())
        .abs()
        .max()
        .into_scalar();
    let aux_diff: f32 = (aux_dense.clone() - aux_sparse.clone()).abs().into_scalar();

    assert!(
        out_diff < 1e-6,
        "Sparse dispatch forward output deviated from dense reference: {out_diff}"
    );
    assert!(
        aux_diff < 1e-6,
        "Auxiliary loss deviated between sparse and dense: {aux_diff}"
    );

    // Compare backward gradients
    let loss_dense = out_dense.sum() + aux_dense;
    let loss_sparse = out_sparse.sum() + aux_sparse;

    let grads_dense = loss_dense.backward();
    let grads_sparse = loss_sparse.backward();

    let router_grad_dense = dense_moe
        .router
        .weight
        .grad(&grads_dense)
        .expect("dense router grad");
    let router_grad_sparse = sparse_moe
        .router
        .weight
        .grad(&grads_sparse)
        .expect("sparse router grad");

    let grad_diff: f32 = (router_grad_dense - router_grad_sparse)
        .abs()
        .max()
        .into_scalar();
    assert!(
        grad_diff < 1e-6,
        "Router gradient deviated between sparse and dense dispatch: {grad_diff}"
    );
}

#[test]
fn test_sparse_vs_dense_dispatch_timing_and_overhead() {
    let device = NdArrayDevice::Cpu;
    let [batch_size, d_model, d_hidden, num_experts, top_k] = [16, 32, 64, 16, 1];

    let dense_cfg = MoeConfig::new(d_model, d_hidden)
        .with_num_experts(num_experts)
        .with_top_k(top_k);

    <NdBackend as burn::prelude::Backend>::seed(42);
    let dense_moe = dense_cfg.init::<NdBackend>(&device);
    let mut sparse_moe = dense_moe.clone();
    sparse_moe.sparse_dispatch = true;

    let x = Tensor::<NdBackend, 2>::random(
        [batch_size, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Warmup
    for _ in 0..10 {
        let _ = dense_moe.forward(x.clone());
        let _ = sparse_moe.forward(x.clone());
    }

    let iterations = 50;
    let t_dense_start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = dense_moe.forward(x.clone());
    }
    let dense_duration = t_dense_start.elapsed();

    let t_sparse_start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = sparse_moe.forward(x.clone());
    }
    let sparse_duration = t_sparse_start.elapsed();

    println!(
        "\n[MoE Dispatch Benchmark] 16 Experts / Top-1 | Batch 16 | Iterations: {}\n  Dense Forward:  {:?} ({:.3} ms/iter)\n  Sparse Forward: {:?} ({:.3} ms/iter)\n  Ratio: {:.2}x",
        iterations,
        dense_duration,
        dense_duration.as_secs_f64() * 1000.0 / iterations as f64,
        sparse_duration,
        sparse_duration.as_secs_f64() * 1000.0 / iterations as f64,
        dense_duration.as_secs_f64() / sparse_duration.as_secs_f64(),
    );
}
