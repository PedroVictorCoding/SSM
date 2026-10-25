use std::fs;
use std::path::Path;

use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;

use sail::model::hybrid::HybridLmConfig;
use sail::train::{CheckpointState, TrainConfig, train};

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_training_checkpoint_and_resume() {
    let device = NdArrayDevice::Cpu;
    let run_dir = Path::new("target/test_resume_checkpoint");
    let _ = fs::remove_dir_all(run_dir);

    let corpus = "ROMEO:\nO, she doth teach the torches to burn bright!\nIt seems she hangs upon the cheek of night.\n";

    let model_cfg = HybridLmConfig::new(32)
        .with_d_model(16)
        .with_n_layers(2)
        .with_d_state(4)
        .with_d_conv(3)
        .with_mlp_ratio(2);

    // Phase 1: Train for 10 steps
    let cfg1 = TrainConfig::new(model_cfg.clone())
        .with_steps(10)
        .with_batch_size(2)
        .with_seq_len(16)
        .with_eval_every(5)
        .with_log_every(5)
        .with_sample_chars(0);

    let summary1 = train::<B>(cfg1, corpus, run_dir, &device).expect("Phase 1 training failed");
    assert_eq!(summary1.steps_run, 10);

    // Verify checkpoint files exist
    let cp_json_path = run_dir.join("checkpoint.json");
    let cp_model_path = run_dir.join("checkpoint_model.mpk");
    let cp_optim_path = run_dir.join("optimizer.mpk");

    assert!(cp_json_path.exists(), "checkpoint.json must exist");
    assert!(cp_model_path.exists(), "checkpoint_model.mpk must exist");
    assert!(cp_optim_path.exists(), "optimizer.mpk must exist");

    let cp_str = fs::read_to_string(&cp_json_path).expect("read checkpoint.json");
    let cp_state: CheckpointState = serde_json::from_str(&cp_str).expect("parse checkpoint.json");
    assert_eq!(cp_state.step, 10);

    // Phase 2: Resume and train up to 20 steps
    let cfg2 = TrainConfig::new(model_cfg.with_work_efficient_scan(Some(true)))
        .with_steps(20)
        .with_batch_size(2)
        .with_seq_len(16)
        .with_eval_every(5)
        .with_log_every(5)
        .with_resume(true)
        .with_sample_chars(0);

    let summary2 =
        train::<B>(cfg2, corpus, run_dir, &device).expect("Phase 2 resume training failed");
    assert_eq!(summary2.steps_run, 20);
    let resumed_model_cfg: HybridLmConfig =
        serde_json::from_str(&fs::read_to_string(run_dir.join("model_config.json")).unwrap())
            .unwrap();
    assert_eq!(resumed_model_cfg.work_efficient_scan, Some(true));
    assert!(summary2.final_train_loss.is_finite());

    // Verify checkpoint updated to step 20
    let cp_str2 = fs::read_to_string(&cp_json_path).expect("read updated checkpoint.json");
    let cp_state2: CheckpointState =
        serde_json::from_str(&cp_str2).expect("parse updated checkpoint.json");
    assert_eq!(cp_state2.step, 20);
    assert!(cp_state2.elapsed_secs > cp_state.elapsed_secs);

    // Verify metrics.csv preserved lines from both phases
    let metrics = fs::read_to_string(run_dir.join("metrics.csv")).expect("read metrics.csv");
    let lines: Vec<&str> = metrics.lines().collect();
    // Header + phase 1 (steps 5, 10) + phase 2 (steps 15, 20)
    assert!(
        lines.len() >= 5,
        "metrics.csv should contain logged steps from both phases"
    );

    let _ = fs::remove_dir_all(run_dir);
}
