use sail::backend::TrainBackend;
use sail::export::{
    SEA_ARTIFACT_MAGIC, export_to_artifact, load_artifact, load_artifact_from_bytes,
};
use sail::generate::{SamplingConfig, generate_stateful};
use sail::model::HybridLmConfig;
use sail::train::{TrainConfig, load_run, train};

#[test]
fn test_export_and_native_standalone_runner() {
    let tmp_dir = std::env::temp_dir().join("sail_test_export");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    let run_dir = tmp_dir.join("run");
    let artifact_path = tmp_dir.join("model.sea");

    let text =
        "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.";
    let device = Default::default();

    let model_cfg = HybridLmConfig::new(65)
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(8)
        .with_expand(2);

    let train_cfg = TrainConfig::new(model_cfg)
        .with_steps(5)
        .with_batch_size(2)
        .with_seq_len(16)
        .with_eval_every(5);

    let _ = train::<TrainBackend>(train_cfg, text, &run_dir, &device).expect("training failed");

    // Export run directory into a single standalone artifact
    let exported_path = export_to_artifact::<TrainBackend>(&run_dir, &artifact_path, &device)
        .expect("export failed");
    assert!(exported_path.exists());

    // Verify magic bytes
    let raw_bytes = std::fs::read(&artifact_path).expect("read artifact");
    assert_eq!(&raw_bytes[0..8], SEA_ARTIFACT_MAGIC);

    // Load artifact from disk
    let (loaded_model, loaded_cfg, loaded_tok) =
        load_artifact::<TrainBackend>(&artifact_path, &device).expect("load_artifact failed");

    // Also verify load directly from in-memory byte slice (WASM / static binary mode)
    let (mem_model, _, _) = load_artifact_from_bytes::<TrainBackend>(&raw_bytes, &device)
        .expect("load_artifact_from_bytes failed");

    assert_eq!(loaded_cfg.d_model, 32);
    assert_eq!(loaded_cfg.n_layers, 2);

    // Verify generation matches between original run and exported artifact
    let (orig_model, _, orig_tok) =
        load_run::<TrainBackend>(&run_dir, &device).expect("load original");
    let prompt_ids = orig_tok.encode("First").expect("encode");

    let greedy = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        seed: 42,
        ..Default::default()
    };

    let (orig_gen, _) = generate_stateful(&orig_model, &prompt_ids, 8, &greedy, &device);
    let (loaded_gen, _) = generate_stateful(&loaded_model, &prompt_ids, 8, &greedy, &device);
    let (mem_gen, _) = generate_stateful(&mem_model, &prompt_ids, 8, &greedy, &device);

    assert_eq!(orig_gen, loaded_gen);
    assert_eq!(orig_gen, mem_gen);

    let decoded = loaded_tok.decode(&loaded_gen);
    assert!(!decoded.is_empty());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
