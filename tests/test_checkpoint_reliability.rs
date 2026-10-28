use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use sail::backend::{InferenceBackend as Inner, TrainBackend as B, default_device};
use sail::data::{CharTokenizer, StreamConfig, StreamDataset, Tokenizer};
use sail::export::{load_artifact, pack_to_artifact};
use sail::generate::{SamplingConfig, generate_stateful};
use sail::model::HybridLmConfig;
use sail::train::{CheckpointState, TrainConfig, stream_train_with_validation, train};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

fn root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("sail-{name}-{}", std::process::id()));
    if path.exists() {
        fs::remove_dir_all(&path).unwrap();
    }
    fs::create_dir_all(&path).unwrap();
    path
}
fn fixture(path: &Path) -> (HybridLmConfig, Tokenizer, String) {
    let device = default_device();
    let text = "abcdefgh\nhgfedcba\n".repeat(64);
    let tok = Tokenizer::Char(CharTokenizer::from_text(&text));
    let cfg = HybridLmConfig::new(tok.vocab_size())
        .with_d_model(8)
        .with_n_layers(2)
        .with_d_state(3)
        .with_d_conv(2)
        .with_attn_layers(vec![1])
        .with_n_heads(2)
        .with_mlp_ratio(1)
        .with_moe_experts(2);
    let model = cfg.init::<Inner>(&device);
    model
        .save_file(
            path.join("model"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .unwrap();
    cfg.save(path.join("model_config.json")).unwrap();
    tok.save(path.join("vocab.json")).unwrap();
    fs::write(
        path.join("checkpoint.json"),
        serde_json::json!({"step":0,"best_val_loss":1e10,"elapsed_secs":0.0,"lr":0.001})
            .to_string(),
    )
    .unwrap();
    (cfg, tok, text)
}
fn config(cfg: HybridLmConfig) -> TrainConfig {
    TrainConfig::new(cfg)
        .with_resume(true)
        .with_steps(6)
        .with_batch_size(2)
        .with_seq_len(8)
        .with_eval_every(2)
        .with_eval_windows(3)
        .with_save_every(2)
        .with_log_every(2)
        .with_viz_every(1)
        .with_viz_max_values(16)
        .with_sample_chars(0)
        .with_warmup_steps(0)
}
fn copy_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let to = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &to)
        } else {
            fs::copy(entry.path(), to).unwrap();
        }
    }
}
fn restore_step_two(full: &Path, resumed: &Path) {
    let generation = fs::read_dir(full.join("checkpoints"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("step-00000002-")
        })
        .unwrap();
    fs::create_dir_all(resumed).unwrap();
    for file in ["vocab.json", "model_config.json"] {
        fs::copy(full.join(file), resumed.join(file)).unwrap();
    }
    copy_tree(
        &generation,
        &resumed
            .join("checkpoints")
            .join(generation.file_name().unwrap()),
    );
    fs::copy(
        generation.join("state.json"),
        resumed.join("checkpoint.json"),
    )
    .unwrap();
    // An interrupted, unpublished checkpoint must never be chosen by restore.
    let unfinished = resumed.join("checkpoints/step-99999999-uncommitted");
    fs::create_dir_all(&unfinished).unwrap();
    fs::write(unfinished.join("model.mpk"), b"incomplete checkpoint").unwrap();
}
fn final_logits(path: &Path, cfg: &HybridLmConfig) -> Vec<f32> {
    let state: CheckpointState =
        serde_json::from_slice(&fs::read(path.join("checkpoint.json")).unwrap()).unwrap();
    assert_eq!(state.step, 6);
    assert!(state.rng.is_some());
    let device = default_device();
    let model = cfg
        .init::<Inner>(&device)
        .load_file(
            path.join(state.model_file.unwrap()),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )
        .unwrap();
    model
        .forward(Tensor::from_data([[0, 1, 2, 3, 4, 5, 6, 7]], &device))
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap()
}

#[test]
fn regular_resume_matches_uninterrupted_with_optimizer_and_sampler_state() {
    let full = root("regular-full");
    let resumed = root("regular-resumed");
    let (cfg, _, text) = fixture(&full);
    let settings = config(cfg.clone());
    let device = default_device();
    train::<B>(settings.clone(), &text, &full, &device).unwrap();
    restore_step_two(&full, &resumed);
    train::<B>(settings, &text, &resumed, &device).unwrap();
    let a = final_logits(&full, &cfg);
    let b = final_logits(&resumed, &cfg);
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        assert!(
            (a - b).abs() < 2e-3,
            "resume mismatch scalar {i}: {a} vs {b}"
        );
    }
    fs::remove_dir_all(full).unwrap();
    fs::remove_dir_all(resumed).unwrap();
}

#[test]
fn streaming_resume_restores_replay_rng_validation_and_visualization() {
    let full = root("stream-full");
    let resumed = root("stream-resumed");
    let (cfg, tok, text) = fixture(&full);
    let dataset = StreamDataset::new(
        tok.encode(&text).unwrap(),
        StreamConfig {
            min_tokens_to_train: 16,
            ..Default::default()
        },
    );
    dataset.append_text("abcdefgh\nabcdefgh\nabcdefgh\n", &tok);
    let settings = config(cfg.clone());
    let validation = tok.encode(&"hgfedcba\n".repeat(16)).unwrap();
    let device = default_device();
    stream_train_with_validation::<B>(
        settings.clone(),
        dataset,
        None,
        tok.clone(),
        &full,
        &device,
        Some(validation.clone()),
    )
    .unwrap();
    restore_step_two(&full, &resumed);
    let empty = StreamDataset::new(
        Vec::new(),
        StreamConfig {
            min_tokens_to_train: 16,
            ..Default::default()
        },
    );
    stream_train_with_validation::<B>(
        settings,
        empty.clone(),
        None,
        tok,
        &resumed,
        &device,
        Some(validation),
    )
    .unwrap();
    assert!(
        empty.total_tokens() > 16,
        "replay must be restored from the committed generation"
    );
    let a = final_logits(&full, &cfg);
    let b = final_logits(&resumed, &cfg);
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        assert!(
            (a - b).abs() < 5e-4,
            "stream resume mismatch scalar {i}: {a} vs {b}"
        );
    }
    assert!(resumed.join("neural.json").is_file());
    let metrics = fs::read_to_string(resumed.join("metrics.csv")).unwrap();
    assert!(metrics.lines().next().unwrap().contains("tokens_per_sec"));
    assert!(metrics.lines().any(|line| {
        line.split(',')
            .nth(3)
            .is_some_and(|v| !v.is_empty() && v.parse::<f64>().is_ok())
    }));
    load_artifact::<Inner>(resumed.join("model.sea"), &device).unwrap();
    fs::remove_dir_all(full).unwrap();
    fs::remove_dir_all(resumed).unwrap();
}

#[test]
fn live_inference_reads_complete_artifacts_while_training_publishes() {
    let dir = root("live-reader");
    let (cfg, tok, text) = fixture(&dir);
    let device = default_device();
    let model = cfg
        .init::<Inner>(&device)
        .load_file(
            dir.join("model"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )
        .unwrap();
    pack_to_artifact(&model, &cfg, &tok, dir.join("model.sea")).unwrap();
    let initial = fs::read(dir.join("model.sea")).unwrap();
    let dataset = StreamDataset::new(
        tok.encode(&text).unwrap(),
        StreamConfig {
            min_tokens_to_train: 16,
            ..Default::default()
        },
    );
    let validation = tok.encode(&"hgfedcba\n".repeat(16)).unwrap();
    let settings = config(cfg)
        .with_steps(20)
        .with_save_every(1)
        .with_eval_every(10)
        .with_viz_every(0);
    let finished = Arc::new(AtomicBool::new(false));
    let flag = finished.clone();
    let train_dir = dir.clone();
    let worker = std::thread::spawn(move || {
        let result = stream_train_with_validation::<B>(
            settings,
            dataset,
            None,
            tok,
            &train_dir,
            &default_device(),
            Some(validation),
        );
        flag.store(true, Ordering::Release);
        result
    });
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut reads_during_updates = 0;
    let mut saw_new_version = false;
    while !finished.load(Ordering::Acquire) && Instant::now() < deadline {
        let bytes = fs::read(dir.join("model.sea")).unwrap();
        saw_new_version |= bytes != initial;
        let (live, _, tok) = load_artifact::<Inner>(dir.join("model.sea"), &device)
            .expect("reader must never see a partial publication");
        let prompt = tok.encode("abc").unwrap();
        let (ids, _) = generate_stateful(
            &live,
            &prompt,
            2,
            &SamplingConfig {
                temperature: 0.0,
                ..Default::default()
            },
            &device,
        );
        assert_eq!(ids.len(), prompt.len() + 2);
        if let Ok(status) = fs::read(dir.join("status.json")) {
            let state: serde_json::Value = serde_json::from_slice(&status).unwrap();
            if state["step"].as_u64().unwrap_or(0) > 0 && !finished.load(Ordering::Acquire) {
                reads_during_updates += 1;
            }
        }
    }
    let summary = worker.join().unwrap().unwrap();
    assert_eq!(summary.steps_run, 20);
    assert!(
        saw_new_version && reads_during_updates > 0,
        "must observe new versions while updates are still running"
    );
    fs::remove_dir_all(dir).unwrap();
}
