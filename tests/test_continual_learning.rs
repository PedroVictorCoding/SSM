use std::fs;
use std::path::PathBuf;

use sail::backend::{InferenceBackend, TrainBackend, default_device};
use sail::data::{
    IncomingDirectoryWatcher, StreamConfig, StreamDataset, Tokenizer, bpe::BpeTokenizer,
};
use sail::export::load_artifact;
use sail::generate::{SamplingConfig, generate_stateful};
use sail::model::HybridLmConfig;
use sail::train::{TrainConfig, stream_train};

#[test]
fn test_continual_stream_training_and_live_sea_artifact() {
    let base_dir = PathBuf::from("target/test_continual_learning");
    let incoming_dir = base_dir.join("incoming");
    let run_dir = base_dir.join("run");
    let _ = fs::remove_dir_all(&base_dir);
    fs::create_dir_all(&incoming_dir).unwrap();
    fs::create_dir_all(&run_dir).unwrap();

    let initial_text = "The theory of general relativity and quantum mechanics explore the universe. \
Continual streaming models ingest knowledge progressively without catastrophic forgetting. ";

    let tokenizer = Tokenizer::Bpe(BpeTokenizer::train(initial_text, 280));
    let initial_tokens = tokenizer.encode(initial_text).unwrap();

    let stream_cfg = StreamConfig {
        replay_ratio: 0.3,
        min_tokens_to_train: 32,
        scan_interval_ms: 50,
    };
    let dataset = StreamDataset::new(initial_tokens, stream_cfg);
    let mut watcher = IncomingDirectoryWatcher::new(&incoming_dir).unwrap();

    // Drop an incoming file before starting
    let chunk1 = incoming_dir.join("chunk1.txt");
    fs::write(
        &chunk1,
        "Deep neural networks and state space models scale linearly with context length. ",
    )
    .unwrap();
    let initial_ingested = watcher.scan_and_ingest(&dataset, &tokenizer);
    assert!(initial_ingested > 0);

    let model_cfg = HybridLmConfig::new(tokenizer.vocab_size())
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(8)
        .with_d_conv(2)
        .with_expand(2);

    let train_cfg = TrainConfig::new(model_cfg)
        .with_steps(6)
        .with_batch_size(2)
        .with_seq_len(16)
        .with_log_every(2)
        .with_save_every(3)
        .with_eval_every(6);

    let device = default_device();
    let summary = stream_train::<TrainBackend>(
        train_cfg,
        dataset.clone(),
        Some(watcher),
        tokenizer.clone(),
        &run_dir,
        &device,
    )
    .expect("stream_train should succeed");

    assert_eq!(summary.steps_run, 6);

    // Verify live atomic .sea model artifact was created
    let live_sea_file = run_dir.join("model.sea");
    assert!(live_sea_file.exists(), "live model.sea artifact must exist");

    // Test loading and sampling directly from the .sea artifact
    let (loaded_model, loaded_cfg, loaded_tok) =
        load_artifact::<InferenceBackend>(&live_sea_file, &device)
            .expect("must load valid .sea artifact");

    assert_eq!(loaded_cfg.d_model, 32);
    assert_eq!(loaded_cfg.n_layers, 2);

    let prompt_ids = loaded_tok.encode("The").unwrap();
    let sampling = SamplingConfig {
        temperature: 0.8,
        top_k: 5,
        seed: 42,
        ..Default::default()
    };

    let (generated_ids, _stats) =
        generate_stateful(&loaded_model, &prompt_ids, 10, &sampling, &device);
    let generated_text = loaded_tok.decode(&generated_ids);

    assert!(generated_text.starts_with("The"));
    assert_eq!(generated_ids.len(), prompt_ids.len() + 10);

    let _ = fs::remove_dir_all(&base_dir);
}
