use std::fs;
use std::path::PathBuf;

use sail::backend::{InferenceBackend, TrainBackend, default_device};
use sail::data::multimodal::{
    AudioData, IncomingMultimodalWatcher, OmnimodalDataset, OmnimodalRecord, VisionData,
};
use sail::data::{CharTokenizer, Tokenizer};
use sail::export::load_omnimodal_artifact;
use sail::model::HybridLmConfig;
use sail::model::multimodal::{AudioProjectorConfig, OmnimodalConfig, VisionProjectorConfig};
use sail::train::{TrainConfig, stream_train_omnimodal};

#[test]
fn test_omnimodal_stream_train_end_to_end_and_resume() {
    let test_dir = PathBuf::from("target/test_omni_stream_train");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();

    let incoming_dir = test_dir.join("incoming");
    fs::create_dir_all(&incoming_dir).unwrap();
    let run_dir = test_dir.join("run");
    fs::create_dir_all(&run_dir).unwrap();

    let device = default_device();
    let d_model = 32;
    let text_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:!?-_\n";
    let tokenizer = Tokenizer::Char(CharTokenizer::from_text(text_chars));
    let vocab_size = tokenizer.vocab_size();

    let omni_cfg = OmnimodalConfig {
        hybrid: HybridLmConfig::new(vocab_size)
            .with_d_model(d_model)
            .with_n_layers(2)
            .with_d_state(8)
            .with_d_conv(2)
            .with_expand(2),
        audio: AudioProjectorConfig::new(d_model)
            .with_in_channels(20)
            .with_kernel_size(3)
            .with_stride(2),
        vision: VisionProjectorConfig::new(d_model)
            .with_in_channels(3)
            .with_patch_size(16)
            .with_max_frames(4)
            .with_max_patches_per_frame(16),
    };

    // 1. Prepare initial dataset records
    let seed_record = OmnimodalRecord {
        text: Some("Prompt 1".into()),
        target: "Target 1 response".into(),
        audio: Some(AudioData {
            channels: 20,
            frames: 16,
            data: vec![0.1; 20 * 16],
        }),
        vision: Some(VisionData {
            channels: 3,
            height: 32,
            width: 32,
            data: vec![0.2; 3 * 32 * 32],
        }),
        audio_file: None,
        image_file: None,
        provenance: None,
    };

    let dataset = OmnimodalDataset::new(vec![seed_record], 0.3);
    let watcher = IncomingMultimodalWatcher::new(&incoming_dir).unwrap();

    // 2. Drop a new chunk into incoming before starting
    let chunk_path = incoming_dir.join("chunk_01.omni.jsonl");
    let chunk_record = OmnimodalRecord {
        text: Some("Audio question".into()),
        target: "Audio answer".into(),
        audio: Some(AudioData {
            channels: 20,
            frames: 12,
            data: vec![0.3; 20 * 12],
        }),
        vision: None,
        audio_file: None,
        image_file: None,
        provenance: None,
    };
    fs::write(
        &chunk_path,
        format!("{}\n", serde_json::to_string(&chunk_record).unwrap()),
    )
    .unwrap();

    // 3. Configure training
    let train_cfg = TrainConfig::new(omni_cfg.hybrid.clone())
        .with_steps(6)
        .with_batch_size(2)
        .with_seq_len(16)
        .with_lr(1e-3)
        .with_save_every(3)
        .with_log_every(2)
        .with_seed(42);

    let summary = stream_train_omnimodal::<TrainBackend>(
        train_cfg,
        omni_cfg.clone(),
        dataset.clone(),
        Some(watcher),
        tokenizer.clone(),
        &run_dir,
        &device,
    )
    .expect("Omnimodal stream training should succeed");

    assert_eq!(summary.steps_run, 6);
    assert!(summary.final_train_loss > 0.0);

    // 4. Verify live .sea artifact was generated and is valid
    let live_sea = run_dir.join("model.sea");
    assert!(live_sea.exists(), "Live model.sea artifact must exist");

    let (loaded_model, loaded_cfg, _loaded_tok) =
        load_omnimodal_artifact::<InferenceBackend>(&live_sea, &device)
            .expect("Must load published live omnimodal .sea artifact");
    assert_eq!(loaded_cfg.hybrid.d_model, d_model);

    // Test inference through loaded artifact
    let test_batch = dataset
        .sample_batch::<InferenceBackend, _>(1, &tokenizer, &mut rand::thread_rng(), &device)
        .expect("Must sample inference batch");
    let logits = loaded_model.forward_batch(&test_batch);
    let [b, _l, v] = logits.dims();
    assert_eq!(b, 1);
    assert_eq!(v, vocab_size);

    // 5. Test deterministic resume from step 6 to step 10
    let watcher2 = IncomingMultimodalWatcher::new(&incoming_dir).unwrap();
    let resume_cfg = TrainConfig::new(omni_cfg.hybrid.clone())
        .with_steps(10)
        .with_batch_size(2)
        .with_seq_len(16)
        .with_lr(1e-3)
        .with_save_every(2)
        .with_resume(true)
        .with_seed(42);

    let resume_summary = stream_train_omnimodal::<TrainBackend>(
        resume_cfg,
        omni_cfg,
        dataset,
        Some(watcher2),
        tokenizer,
        &run_dir,
        &device,
    )
    .expect("Omnimodal stream resume should succeed");

    assert_eq!(resume_summary.steps_run, 10);

    let _ = fs::remove_dir_all(&test_dir);
}
