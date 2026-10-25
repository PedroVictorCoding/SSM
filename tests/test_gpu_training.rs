//! Compare the actual regular and streaming loops from identical CPU weights.
#![cfg(all(feature = "wgpu", feature = "ndarray", not(feature = "cuda")))]
use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use sail::{
    backend::{InferenceBackend as GpuInner, TrainBackend as Gpu, default_device},
    data::{CharTokenizer, StreamConfig, StreamDataset, Tokenizer},
    model::HybridLmConfig,
    train::{CheckpointState, TrainConfig, stream_train_with_validation, train},
};
use std::{fs, path::Path};
type Cpu = NdArray<f32>;

fn setup(path: &Path, model: &HybridLmConfig, tokenizer: &Tokenizer) {
    fs::create_dir_all(path).unwrap();
    Cpu::seed(97);
    model
        .init::<Cpu>(&NdArrayDevice::Cpu)
        .save_file(
            path.join("model"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    model.save(path.join("model_config.json")).unwrap();
    tokenizer.save(path.join("vocab.json")).unwrap();
    fs::write(
        path.join("checkpoint.json"),
        r#"{"step":0,"best_val_loss":10000000000.0,"elapsed_secs":0.0,"lr":0.001}"#,
    )
    .unwrap();
}

fn logits<B: Backend>(path: &Path, cfg: &HybridLmConfig, device: &B::Device) -> Vec<f32> {
    let state: CheckpointState =
        serde_json::from_slice(&fs::read(path.join("checkpoint.json")).unwrap()).unwrap();
    assert_eq!(state.step, 4);
    let weights = path.join(state.model_file.unwrap());
    let model = cfg
        .init::<B>(device)
        .load_file(
            weights,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .unwrap();
    model
        .forward(Tensor::from_data(
            TensorData::new((0..16).map(|i| (i % 8) as i64).collect::<Vec<_>>(), [2, 8]),
            device,
        ))
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap()
}

fn validation(path: &Path) -> Vec<f64> {
    fs::read_to_string(path.join("metrics.csv"))
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|line| line.split(',').nth(3).and_then(|v| v.parse().ok()))
        .collect()
}

fn compare_loops(streaming: bool) {
    let root = std::env::temp_dir().join(format!(
        "sail-loop-parity-{streaming}-{}",
        std::process::id()
    ));
    assert!(
        !root.exists(),
        "test directory already exists: {}",
        root.display()
    );
    let cpu_dir = root.join("cpu");
    let gpu_dir = root.join("gpu");
    let baseline_dir = root.join("cpu-burn-clip");
    let text = "abcdefgh\nhgfedcba\n".repeat(64);
    let tokenizer = Tokenizer::Char(CharTokenizer::from_text(&text));
    let cfg = HybridLmConfig::new(tokenizer.vocab_size())
        .with_d_model(16)
        .with_n_layers(3)
        .with_d_state(3)
        .with_d_conv(3)
        .with_attn_layers(vec![1])
        .with_n_heads(2)
        .with_mlp_ratio(2)
        .with_moe_experts(4)
        .with_moe_top_k(2);
    setup(&cpu_dir, &cfg, &tokenizer);
    for dir in [&gpu_dir, &baseline_dir] {
        fs::create_dir_all(dir).unwrap();
        for file in [
            "model.mpk",
            "model_config.json",
            "vocab.json",
            "checkpoint.json",
        ] {
            fs::copy(cpu_dir.join(file), dir.join(file)).unwrap();
        }
    }
    let run = TrainConfig::new(cfg.clone())
        .with_device_clip(Some(true))
        .with_resume(true)
        .with_steps(4)
        .with_batch_size(2)
        .with_seq_len(8)
        .with_lr(0.001)
        .with_min_lr(0.001)
        .with_warmup_steps(0)
        .with_grad_clip(0.1)
        .with_eval_every(2)
        .with_eval_windows(4)
        .with_save_every(2)
        .with_log_every(2)
        .with_viz_every(1)
        .with_viz_max_values(16)
        .with_sample_chars(0);
    let device = default_device();
    let baseline = run.clone().with_device_clip(Some(false));
    if streaming {
        stream_train_with_validation::<Autodiff<Cpu>>(
            baseline,
            StreamDataset::new(tokenizer.encode(&text).unwrap(), StreamConfig::default()),
            None,
            tokenizer.clone(),
            &baseline_dir,
            &NdArrayDevice::Cpu,
            Some(tokenizer.encode(&"hgfedcba\n".repeat(8)).unwrap()),
        )
        .unwrap();
        let cpu_data =
            StreamDataset::new(tokenizer.encode(&text).unwrap(), StreamConfig::default());
        let gpu_data =
            StreamDataset::new(tokenizer.encode(&text).unwrap(), StreamConfig::default());
        let validation = tokenizer.encode(&"hgfedcba\n".repeat(8)).unwrap();
        stream_train_with_validation::<Autodiff<Cpu>>(
            run.clone(),
            cpu_data,
            None,
            tokenizer.clone(),
            &cpu_dir,
            &NdArrayDevice::Cpu,
            Some(validation.clone()),
        )
        .unwrap();
        stream_train_with_validation::<Gpu>(
            run,
            gpu_data,
            None,
            tokenizer,
            &gpu_dir,
            &device,
            Some(validation),
        )
        .unwrap();
    } else {
        train::<Autodiff<Cpu>>(baseline, &text, &baseline_dir, &NdArrayDevice::Cpu).unwrap();
        train::<Autodiff<Cpu>>(run.clone(), &text, &cpu_dir, &NdArrayDevice::Cpu).unwrap();
        train::<Gpu>(run, &text, &gpu_dir, &device).unwrap();
    }
    let cpu = logits::<Cpu>(&cpu_dir, &cfg, &NdArrayDevice::Cpu);
    let gpu = logits::<GpuInner>(&gpu_dir, &cfg, &device);
    let burn_clip = logits::<Cpu>(&baseline_dir, &cfg, &NdArrayDevice::Cpu);
    let clipping_error = cpu
        .iter()
        .zip(&burn_clip)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        clipping_error < 2e-5,
        "device clipping changed updates: {clipping_error}"
    );
    let max_error = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error < 2e-3,
        "streaming={streaming}: updated logits diverged {max_error}"
    );
    let cpu_val = validation(&cpu_dir);
    let gpu_val = validation(&gpu_dir);
    assert_eq!(cpu_val.len(), 2);
    assert_eq!(cpu_val.len(), gpu_val.len());
    for (a, b) in cpu_val.iter().zip(&gpu_val) {
        assert!((a - b).abs() < 2e-4, "CPU validation={a}, GPU={b}");
    }
    println!(
        "streaming={streaming}: max updated logit difference={max_error}, CPU validation={cpu_val:?}, GPU validation={gpu_val:?}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn regular_training_matches_cpu() {
    compare_loops(false);
}
#[test]
fn streaming_training_matches_cpu() {
    compare_loops(true);
}
