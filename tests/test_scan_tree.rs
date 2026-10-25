use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::Distribution;
use sail::model::scan::{chunked_parallel_scan, sequential_scan, work_efficient_scan};
use sail::model::{HybridLmConfig, MambaConfig};
type B = Autodiff<NdArray<f32>>;

#[test]
fn parallel_scans_match_reference_outputs_and_gradients() {
    let device = NdArrayDevice::Cpu;
    B::seed(123);
    for len in [1, 2, 3, 7, 16, 31, 64, 127] {
        for mode in 0..3 {
            // Signed transitions rule out unstable division/cumulative-product shortcuts.
            let a =
                Tensor::<B, 4>::random([2, len, 3, 2], Distribution::Uniform(-0.9, 0.9), &device)
                    .require_grad();
            let b = Tensor::<B, 4>::random([2, len, 3, 2], Distribution::Normal(0.0, 1.0), &device)
                .require_grad();
            let test_a = a.clone().detach().require_grad();
            let test_b = b.clone().detach().require_grad();
            let reference = sequential_scan(a.clone(), b.clone());
            let candidate = match mode {
                0 => work_efficient_scan(test_a.clone(), test_b.clone()),
                1 => chunked_parallel_scan(test_a.clone(), test_b.clone(), 4),
                _ => chunked_parallel_scan(test_a.clone(), test_b.clone(), 64),
            };
            let diff = (reference.clone() - candidate.clone())
                .abs()
                .max()
                .into_scalar();
            assert!(diff < 2e-5, "output mode={mode} L={len} diff={diff}");
            // Non-uniform upstream gradients exercise all output positions.
            let ref_grads = reference.powf_scalar(2.0).sum().backward();
            let test_grads = candidate.powf_scalar(2.0).sum().backward();
            for (name, input, test_input) in [("a", &a, &test_a), ("b", &b, &test_b)] {
                let expected = input
                    .grad(&ref_grads)
                    .unwrap_or_else(|| Tensor::zeros([2, len, 3, 2], &device));
                let actual = test_input
                    .grad(&test_grads)
                    .unwrap_or_else(|| Tensor::zeros([2, len, 3, 2], &device));
                let diff = (expected - actual).abs().max().into_scalar();
                assert!(
                    diff < 5e-5,
                    "gradient {name} mode={mode} L={len} diff={diff}"
                );
            }
        }
    }
}

#[test]
fn tree_handles_zero_and_identity_transitions() {
    let device = NdArrayDevice::Cpu;
    let b = Tensor::<B, 4>::ones([1, 17, 2, 1], &device);
    let reset = work_efficient_scan(Tensor::zeros([1, 17, 2, 1], &device), b.clone());
    assert_eq!((reset - b.clone()).abs().max().into_scalar(), 0.0);
    let prefix = work_efficient_scan(Tensor::ones([1, 17, 2, 1], &device), b);
    assert_eq!(prefix.narrow(1, 16, 1).sum().into_scalar(), 34.0);
}

#[test]
fn tree_mamba_matches_recurrent_decode_and_old_configs_load() {
    type Inner = NdArray<f32>;
    let device = NdArrayDevice::Cpu;
    let model = MambaConfig::new(8)
        .with_d_state(3)
        .with_d_conv(2)
        .with_work_efficient_scan(Some(true))
        .init::<Inner>(&device);
    let x = Tensor::<Inner, 3>::random([2, 17, 8], Distribution::Normal(0.0, 1.0), &device);
    let full = model.forward_with_state(x.clone()).0;
    let mut state = model.init_state(2, &device);
    let mut outputs = Vec::new();
    for i in 0..17 {
        let (output, next) = model.step(x.clone().narrow(1, i, 1).squeeze(1), state);
        state = next;
        outputs.push(output.unsqueeze_dim::<3>(1));
    }
    let diff = (full - Tensor::cat(outputs, 1)).abs().max().into_scalar();
    assert!(
        diff < 2e-5,
        "tree prefill differs from recurrent decoding: {diff}"
    );
    let mut legacy = serde_json::to_value(HybridLmConfig::new(65)).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("work_efficient_scan");
    let old: HybridLmConfig = serde_json::from_value(legacy).unwrap();
    assert!(!old.work_efficient_scan.unwrap_or(false));
    let new = old.with_work_efficient_scan(Some(true));
    assert_eq!(new.mamba_config().work_efficient_scan, Some(true));
}
