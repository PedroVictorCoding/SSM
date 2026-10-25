//! Run on a GPU host with --no-default-features --features wgpu,ndarray.
//! The wgpu-unfused,ndarray variant is also available for diagnosis.
#![cfg(all(feature = "wgpu", feature = "ndarray", not(feature = "cuda")))]
use burn::backend::{NdArray, ndarray::NdArrayDevice};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use sail::backend::{InferenceBackend as Gpu, default_device};
use sail::model::HybridLmConfig;

type Cpu = NdArray<f32>;

#[test]
fn gpu_logits_match_cpu_with_and_without_observation() {
    let cpu_device = NdArrayDevice::Cpu;
    let gpu_device = default_device();
    Cpu::seed(71);
    let cfg = HybridLmConfig::new(8)
        .with_d_model(8)
        .with_n_layers(3)
        .with_d_state(3)
        .with_d_conv(2)
        .with_n_heads(2)
        .with_attn_layers(vec![1])
        .with_mlp_ratio(2)
        .with_moe_experts(4)
        .with_moe_top_k(1);
    let cpu = cfg.init::<Cpu>(&cpu_device);
    let dir = std::env::temp_dir().join(format!("ssm-gpu-reference-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    cpu.clone().save_file(dir.join("model"), &recorder).unwrap();
    let gpu = cfg
        .init::<Gpu>(&gpu_device)
        .load_file(dir.join("model"), &recorder, &gpu_device)
        .unwrap();
    for len in [7, 32, 128] {
        let ids: Vec<i64> = (0..len).map(|i| (i % 8) as i64).collect();
        let expected = cpu
            .forward(Tensor::from_data(
                TensorData::new(ids.clone(), [1, len]),
                &cpu_device,
            ))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let input = Tensor::<Gpu, 2, Int>::from_data(TensorData::new(ids, [1, len]), &gpu_device);
        for mode in ["prefill", "unobserved", "observed"] {
            let logits = match mode {
                "observed" => gpu.forward_observed(input.clone(), true).0,
                "unobserved" => gpu.forward_with_aux(input.clone()).0,
                _ => gpu.forward(input.clone()),
            };
            let actual = logits.into_data().convert::<f32>().to_vec::<f32>().unwrap();
            for (i, (a, b)) in expected.iter().zip(&actual).enumerate() {
                assert!(
                    b.is_finite() && (a - b).abs() <= 1e-3 + 1e-4 * a.abs(),
                    "GPU mismatch mode={mode} len={len} scalar={i}: CPU={a}, GPU={b}"
                );
            }
        }
        let mut state = gpu.init_state(1, &gpu_device);
        for position in 0..len {
            let token = Tensor::<Gpu, 1, Int>::from_data(
                TensorData::new(vec![(position % 8) as i64], [1]),
                &gpu_device,
            );
            let (logits, next_state) = gpu.step(token, state);
            state = next_state;
            let actual = logits.into_data().to_vec::<f32>().unwrap();
            for (index, value) in actual.into_iter().enumerate() {
                let reference = expected[position * 8 + index];
                assert!(
                    value.is_finite() && (value - reference).abs() <= 1e-3 + reference.abs() * 1e-4,
                    "GPU recurrent mismatch len={len} position={position}: CPU={reference}, GPU={value}"
                );
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

use burn::backend::Autodiff;
use burn::module::{AutodiffModule, ModuleVisitor, ParamId};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use sail::model::HybridLm;

struct GradientDump<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    tensors: Vec<(Vec<usize>, Option<Vec<f32>>)>,
    marker: std::marker::PhantomData<B>,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for GradientDump<'_, B> {
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
        let data = self
            .grads
            .get::<B::InnerBackend, D>(id)
            .map(|g| g.into_data().convert::<f32>().to_vec::<f32>().unwrap());
        self.tensors.push((tensor.dims().to_vec(), data));
    }
}

fn backward<B: AutodiffBackend>(
    model: &HybridLm<B>,
    device: &B::Device,
    observe: bool,
) -> (f32, Vec<(Vec<usize>, Option<Vec<f32>>)>) {
    let batch = 2;
    let len = 17;
    let ids: Vec<i64> = (0..batch * len).map(|i| (i % 8) as i64).collect();
    let targets: Vec<i64> = (0..batch * len).map(|i| ((i + 1) % 8) as i64).collect();
    let (logits, aux, _) = model.forward_observed(
        Tensor::from_data(TensorData::new(ids, [batch, len]), device),
        observe,
    );
    let ce = CrossEntropyLossConfig::new().init(device).forward(
        logits.reshape([batch * len, 8]),
        Tensor::from_data(TensorData::new(targets, [batch * len]), device),
    );
    let loss = if let Some(aux) = aux {
        ce + aux.mul_scalar(0.01)
    } else {
        ce
    };
    let scalar = loss.clone().inner().into_scalar().elem::<f32>();
    let grads = GradientsParams::from_grads(loss.backward(), model);
    let mut dump = GradientDump::<B> {
        grads: &grads,
        tensors: Vec::new(),
        marker: std::marker::PhantomData,
    };
    model.visit(&mut dump);
    (scalar, dump.tensors)
}

#[test]
fn gpu_parameter_gradients_and_validation_match_cpu() {
    type CpuAd = Autodiff<Cpu>;
    type GpuAd = Autodiff<Gpu>;
    let cd = NdArrayDevice::Cpu;
    let gd = default_device();
    let dir = std::env::temp_dir().join(format!("sail-gpu-gradients-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    for (top_k, tree, observe) in [(1, false, false), (2, true, true)] {
        CpuAd::seed(43);
        let cfg = HybridLmConfig::new(8)
            .with_d_model(16)
            .with_d_state(4)
            .with_d_conv(3)
            .with_n_layers(3)
            .with_attn_layers(vec![1])
            .with_n_heads(4)
            .with_mlp_ratio(2)
            .with_moe_experts(4)
            .with_moe_top_k(top_k)
            .with_work_efficient_scan(Some(tree));
        let cpu = cfg.init::<CpuAd>(&cd);
        cpu.clone().save_file(dir.join("model"), &recorder).unwrap();
        let gpu = cfg
            .init::<GpuAd>(&gd)
            .load_file(dir.join("model"), &recorder, &gd)
            .unwrap();
        let (cl, cg) = backward(&cpu, &cd, observe);
        let (gl, gg) = backward(&gpu, &gd, observe);
        assert!((cl - gl).abs() < 1e-4, "training loss CPU={cl} GPU={gl}");
        assert_eq!(cg.len(), gg.len());
        let mut scalars = 0;
        for (index, ((cs, c), (gs, g))) in cg.iter().zip(&gg).enumerate() {
            assert_eq!(cs, gs);
            assert_eq!(c.is_some(), g.is_some(), "gradient presence tensor {index}");
            if let (Some(c), Some(g)) = (c, g) {
                assert_eq!(c.len(), g.len());
                scalars += c.len();
                for (i, (a, b)) in c.iter().zip(g).enumerate() {
                    assert!(
                        b.is_finite() && (a - b).abs() <= 2e-4 + 2e-3 * a.abs(),
                        "gradient tensor={index} scalar={i} CPU={a} GPU={b} top_k={top_k} tree={tree}"
                    );
                }
            }
        }
        assert_eq!(
            scalars,
            cpu.total_params(),
            "every trainable parameter must have a gradient"
        );
        let val_tokens: Vec<u32> = (0..96).map(|i| (i % 8) as u32).collect();
        let cv = sail::train::evaluate(&cpu.valid(), &val_tokens, &[0, 7, 23, 53], 17, 2, &cd);
        let gv = sail::train::evaluate(&gpu.valid(), &val_tokens, &[0, 7, 23, 53], 17, 2, &gd);
        assert!(
            cv.is_finite() && gv.is_finite() && (cv - gv).abs() < 1e-4,
            "validation CPU={cv} GPU={gv}"
        );
        println!(
            "top_k={top_k} tree={tree} observe={observe}: {scalars} parameter gradients checked; loss diff={}, validation diff={}",
            (cl - gl).abs(),
            (cv - gv).abs()
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}
