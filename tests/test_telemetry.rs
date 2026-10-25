use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::GradientsParams;
use burn::prelude::*;
use sail::model::HybridLmConfig;

#[test]
fn neural_snapshot_covers_every_parameter_and_preserves_gradients() {
    type B = Autodiff<NdArray<f32>>;
    let device = NdArrayDevice::Cpu;
    let model = HybridLmConfig::new(5)
        .with_d_model(8)
        .with_n_layers(2)
        .with_d_state(2)
        .with_d_conv(2)
        .with_attn_layers(vec![1])
        .with_n_heads(2)
        .with_mlp_ratio(1)
        .with_moe_experts(2)
        .with_moe_top_k(1)
        .init::<B>(&device);
    let tokens = Tensor::<B, 2, Int>::from_data([[0, 1, 2, 3]], &device);
    let (baseline, baseline_aux) = model.forward_with_aux(tokens.clone());
    let (observed, aux, probes) = model.forward_observed(tokens, true);
    assert!(
        (baseline.clone() - observed.clone())
            .abs()
            .max()
            .into_scalar()
            < 1e-6
    );
    assert!(
        (baseline_aux.unwrap() - aux.unwrap())
            .abs()
            .max()
            .into_scalar()
            < 1e-6
    );
    assert_eq!(probes.len(), 4); // embedding, both blocks, logits
    assert_eq!(probes[1]["values"].as_array().unwrap().len(), 8);
    let loss = CrossEntropyLossConfig::new().init(&device).forward(
        observed.reshape([4, 5]),
        Tensor::<B, 1, Int>::from_data([1, 2, 3, 4], &device),
    );
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let gradient_count = grads.len();
    let dir = std::env::temp_dir().join(format!("ssm-telemetry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    sail::telemetry::publish(&model, Some(&grads), &dir, 1, 0, probes).unwrap();
    assert_eq!(
        grads.len(),
        gradient_count,
        "capture must not consume optimizer gradients"
    );
    let snapshot: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("neural.json")).unwrap()).unwrap();
    let tensors = snapshot["tensors"].as_array().unwrap();
    let count: usize = tensors
        .iter()
        .map(|t| t["count"].as_u64().unwrap() as usize)
        .sum();
    assert_eq!(
        count,
        model.total_params(),
        "every trainable scalar must be included"
    );
    for t in tensors {
        assert_eq!(
            t["values"].as_array().unwrap().len(),
            t["count"].as_u64().unwrap() as usize
        );
        assert_eq!(t["stride"], 1);
        assert!(
            t["gradient"]["rms"].is_number(),
            "missing gradient for {}",
            t["name"]
        );
    }
    sail::telemetry::publish(&model, None, &dir, 2, 7, vec![]).unwrap();
    let sampled: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("neural.json")).unwrap()).unwrap();
    assert!(
        sampled["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .all(|t| t["values"].as_array().unwrap().len() <= 7)
    );
    std::fs::remove_dir_all(dir).unwrap();
}
