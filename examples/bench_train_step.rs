//! Full synthetic training-step comparison: forward + loss + backward + AdamW.
//! cargo run --release --example bench_train_step -- 4 64 64 9
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use sail::backend::{BACKEND_NAME, TrainBackend as B, default_device};
use sail::model::{HybridLm, HybridLmConfig};
use std::time::Instant;

fn step(
    model: HybridLm<B>,
    inputs: &Tensor<B, 2, Int>,
    targets: &Tensor<B, 1, Int>,
    optim: &mut impl Optimizer<HybridLm<B>, B>,
) -> (HybridLm<B>, f64, f32) {
    let start = Instant::now();
    let (logits, aux): (Tensor<B, 3>, Option<Tensor<B, 1>>) =
        model.forward_with_aux(inputs.clone());
    let [batch, len, vocab] = logits.dims();
    let mut loss = CrossEntropyLossConfig::new()
        .init(&inputs.device())
        .forward(logits.reshape([batch * len, vocab]), targets.clone());
    if let Some(aux) = aux {
        loss = loss + aux.mul_scalar(0.01);
    }
    let loss_value: f32 = loss.clone().into_scalar();
    assert!(
        loss_value.is_finite(),
        "Non-finite loss in benchmark; timings are invalid"
    );
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let model = optim.step(1e-3, model, grads);
    // Explicitly read updated parameters so asynchronous backends finish the step.
    let _ = model.embedding.weight.val().sum().into_scalar();
    (model, start.elapsed().as_secs_f64() * 1000.0, loss_value)
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|v| v.parse().expect("positive integer"))
        .collect();
    let batch = args.first().copied().unwrap_or(4);
    let len = args.get(1).copied().unwrap_or(64);
    let width = args.get(2).copied().unwrap_or(64);
    let repeats = args.get(3).copied().unwrap_or(9);
    assert!(batch > 0 && len > 0 && width >= 4 && width % 4 == 0 && repeats > 0);
    let device = default_device();
    B::seed(42);
    let cfg = HybridLmConfig::new(65)
        .with_d_model(width)
        .with_d_state(8)
        .with_n_layers(4)
        .with_attn_layers(vec![2])
        .with_n_heads(4)
        .with_mlp_ratio(2)
        .with_moe_experts(4)
        .with_moe_top_k(2);
    let mut chunked = cfg.init::<B>(&device);
    let mut tree = chunked.clone();
    for block in &mut tree.blocks {
        if let Some(mamba) = &mut block.mamba {
            mamba.work_efficient_scan = true;
        }
    }
    let inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            (0..batch * len).map(|i| (i % 65) as i64).collect(),
            [batch, len],
        ),
        &device,
    );
    let targets = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            (0..batch * len).map(|i| ((i + 1) % 65) as i64).collect(),
            [batch * len],
        ),
        &device,
    );
    let mut old_optim = AdamWConfig::new().init();
    let mut new_optim = AdamWConfig::new().init();
    println!(
        "Backend: {BACKEND_NAME}; B={batch}, L={len}, width={width}; 4 layers [M M A M], 4 experts/top-2; {repeats} repeats"
    );
    let mut old_times = Vec::new();
    let mut new_times = Vec::new();
    let mut max_loss_diff = 0.0_f32;
    for i in 0..repeats + 1 {
        let (old_result, new_result) = if i % 2 == 0 {
            let old = step(chunked, &inputs, &targets, &mut old_optim);
            let new = step(tree, &inputs, &targets, &mut new_optim);
            (old, new)
        } else {
            let new = step(tree, &inputs, &targets, &mut new_optim);
            let old = step(chunked, &inputs, &targets, &mut old_optim);
            (old, new)
        };
        chunked = old_result.0;
        tree = new_result.0;
        max_loss_diff = max_loss_diff.max((old_result.2 - new_result.2).abs());
        if i > 0 {
            old_times.push(old_result.1);
            new_times.push(new_result.1);
        }
    }
    old_times.sort_by(f64::total_cmp);
    new_times.sort_by(f64::total_cmp);
    let old = old_times[repeats / 2];
    let new = new_times[repeats / 2];
    println!("chunked_ms,tree_ms,speedup,max_loss_difference");
    println!("{old:.3},{new:.3},{:.3},{max_loss_diff:.8}", old / new);
}
