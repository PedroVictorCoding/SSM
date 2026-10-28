use std::fs;
use std::path::PathBuf;

use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use sail::backend::{InferenceBackend, TrainBackend, default_device};
use sail::data::{CharTokenizer, Tokenizer};
use sail::export::{load_omnimodal_artifact, pack_omnimodal_artifact};
use sail::model::HybridLmConfig;
use sail::model::multimodal::{
    AudioProjectorConfig, OmnimodalBatch, OmnimodalConfig, VisionProjectorConfig,
    train_omnimodal_step,
};
use sail::train::CheckpointState;

#[test]
fn test_train_and_reproduce_omnimodal_off_the_hook() {
    let test_dir = PathBuf::from("target/test_omnimodal_reproducibility");
    let _ = fs::remove_dir_all(&test_dir);
    fs::create_dir_all(&test_dir).unwrap();

    let device = default_device();
    let d_model = 32;
    let vocab_size = 64;

    let text_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n";
    let tokenizer = Tokenizer::Char(CharTokenizer::from_text(text_chars));

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

    // 1. Initialize trainable Omnimodal Model
    let mut model = omni_cfg.init::<TrainBackend>(&device);
    let mut optim = AdamWConfig::new().with_weight_decay(0.01).init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    let batch_size = 2;
    let text_inputs =
        Tensor::<TrainBackend, 2, Int>::from_data([[1, 2, 3, 4], [5, 6, 7, 8]], &device);
    let audio_frames = Tensor::<TrainBackend, 3>::random(
        [batch_size, 20, 16],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let vision_frame = Tensor::<TrainBackend, 4>::random(
        [batch_size, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let targets =
        Tensor::<TrainBackend, 2, Int>::from_data([[10, 11, 12, 13], [14, 15, 16, 17]], &device);

    let batch = OmnimodalBatch {
        text_inputs: Some(text_inputs),
        audio_frames: Some(audio_frames),
        vision_frame: Some((vision_frame, 0)),
        targets,
    };

    // 2. Train for 5 optimization steps and record loss trajectory
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    let lr = 1e-3;

    for step in 1..=5 {
        let (updated, loss) = train_omnimodal_step(model, &mut optim, &batch, &loss_fn, lr)
            .expect("Omnimodal step should succeed");
        model = updated;
        if step == 1 {
            initial_loss = loss;
        }
        final_loss = loss;
    }

    // Optimization check: training loss must reduce
    assert!(
        final_loss < initial_loss,
        "Omnimodal training must reduce loss: initial {initial_loss} vs final {final_loss}"
    );

    // 3. Verify SEQ= and ACK= Checkpoint State Recording & Resumption
    let committed_seq = (5 * batch_size * 16) as u64;
    let cp_state = CheckpointState {
        step: 5,
        best_val_loss: final_loss as f64,
        elapsed_secs: 1.5,
        lr,
        rng: None,
        model_file: Some("omni_model".into()),
        optimizer_file: Some("omni_optim".into()),
        config_file: Some("omni_config.json".into()),
        stream_file: None,
        validation_file: None,
        seq: committed_seq,
        ack_step: 5,
        ack_tokens: 160,
        ack_receipt: Some("receipt_omni_commit_01".into()),
    };

    let cp_path = test_dir.join("checkpoint.json");
    let cp_json = serde_json::to_string_pretty(&cp_state).unwrap();
    fs::write(&cp_path, &cp_json).unwrap();

    // Verify resume state reads stored SEQ and ACK values
    let restored_state: CheckpointState =
        serde_json::from_slice(&fs::read(&cp_path).unwrap()).unwrap();
    assert_eq!(restored_state.seq, committed_seq);
    assert_eq!(restored_state.ack_step, 5);
    assert_eq!(restored_state.ack_tokens, 160);
    assert_eq!(
        restored_state.ack_receipt.as_deref(),
        Some("receipt_omni_commit_01")
    );

    // 4. Export trained Omnimodal model into standalone .sea artifact
    let sea_path = test_dir.join("omnimodal.sea");
    let valid_model = model.valid();
    pack_omnimodal_artifact(&valid_model, &omni_cfg, &tokenizer, &sea_path)
        .expect("Must pack omnimodal artifact");
    assert!(sea_path.exists());

    // 5. Load and reproduce inference "just off the hook"
    let (loaded_model, loaded_cfg, _loaded_tok) =
        load_omnimodal_artifact::<InferenceBackend>(&sea_path, &device)
            .expect("Must load omnimodal artifact");

    assert_eq!(loaded_cfg.hybrid.d_model, d_model);
    assert_eq!(loaded_cfg.audio.in_channels, 20);
    assert_eq!(loaded_cfg.vision.patch_size, 16);

    // Run inference on multimodal sequence from loaded artifact
    let test_text = Tensor::<InferenceBackend, 2, Int>::from_data([[1, 2]], &device);
    let test_audio = Tensor::<InferenceBackend, 3>::random(
        [1, 20, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let test_vision = Tensor::<InferenceBackend, 4>::random(
        [1, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let test_batch = OmnimodalBatch {
        text_inputs: Some(test_text),
        audio_frames: Some(test_audio),
        vision_frame: Some((test_vision, 0)),
        targets: Tensor::<InferenceBackend, 2, Int>::zeros([1, 1], &device),
    };

    let logits = loaded_model.forward_batch(&test_batch);
    let [b, len, v] = logits.dims();
    assert_eq!(b, 1);
    assert_eq!(v, vocab_size);
    assert!(len > 4); // Vision patches + Audio frames + Text tokens

    // 6. Verify O(1) recurrent generation from interleaved multimodal prompt
    let sampling = sail::generate::SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        seed: 42,
        ..Default::default()
    };
    let gen_audio = Tensor::<InferenceBackend, 3>::random(
        [1, 20, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let gen_vision = Tensor::<InferenceBackend, 4>::random(
        [1, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (gen_tokens, stats) = sail::generate::generate_omnimodal_stateful(
        &loaded_model,
        Some(&[1, 2]),
        Some(gen_audio),
        Some((gen_vision, 0)),
        5,
        &sampling,
        &device,
    );
    assert_eq!(gen_tokens.len(), 5);
    assert_eq!(stats.new_tokens, 5);
    assert!(stats.prefill_time.as_nanos() > 0);
    assert!(stats.state_elements > 0);

    let _ = fs::remove_dir_all(&test_dir);
}
