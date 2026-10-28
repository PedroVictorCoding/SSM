use burn::prelude::*;
use sail::backend::{InferenceBackend, default_device};
use sail::model::HybridLmConfig;
use sail::model::multimodal::{AudioProjectorConfig, OmnimodalLm, VisionProjectorConfig};

#[test]
fn test_audio_projector_causal_striding() {
    let device = default_device();
    let d_model = 64;
    let in_channels = 80; // 80 Mel bins
    let cfg = AudioProjectorConfig::new(d_model)
        .with_in_channels(in_channels)
        .with_kernel_size(3)
        .with_stride(2);
    let projector = cfg.init::<InferenceBackend>(&device);

    let batch = 2;
    let time_frames = 32;
    let audio_input = Tensor::<InferenceBackend, 3>::random(
        [batch, in_channels, time_frames],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let embeddings = projector.forward(audio_input);
    let [b, t, d] = embeddings.dims();

    assert_eq!(b, batch);
    assert_eq!(d, d_model);
    // Two layers of stride 2: 32 -> 16 -> 8 tokens (with causal padding)
    assert!(t > 0);
}

#[test]
fn test_vision_projector_patch_embeddings() {
    let device = default_device();
    let d_model = 64;
    let patch_size = 16;
    let cfg = VisionProjectorConfig::new(d_model)
        .with_in_channels(3)
        .with_patch_size(patch_size)
        .with_max_frames(8)
        .with_max_patches_per_frame(64);
    let projector = cfg.init::<InferenceBackend>(&device);

    let batch = 2;
    // 32x32 image with 16x16 patches = 2x2 = 4 patches
    let image_input = Tensor::<InferenceBackend, 4>::random(
        [batch, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let patches = projector.forward_frame(image_input, 0);
    let [b, num_patches, d] = patches.dims();

    assert_eq!(b, batch);
    assert_eq!(num_patches, 4);
    assert_eq!(d, d_model);
}

#[test]
fn test_omnimodal_hybrid_interleaved_forward_and_recurrent_step() {
    let device = default_device();
    let d_model = 64;
    let vocab_size = 128;

    let hybrid_cfg = HybridLmConfig::new(vocab_size)
        .with_d_model(d_model)
        .with_n_layers(2)
        .with_d_state(8)
        .with_d_conv(2);

    let audio_cfg = AudioProjectorConfig::new(d_model)
        .with_in_channels(40)
        .with_kernel_size(3)
        .with_stride(2);

    let vision_cfg = VisionProjectorConfig::new(d_model)
        .with_in_channels(3)
        .with_patch_size(16)
        .with_max_frames(4)
        .with_max_patches_per_frame(16);

    let model = OmnimodalLm::<InferenceBackend>::new(&hybrid_cfg, &audio_cfg, &vision_cfg, &device);

    let batch = 1;
    // 1. Text input: 4 tokens
    let text_tokens = Tensor::<InferenceBackend, 2, Int>::from_data([[10, 20, 30, 40]], &device);

    // 2. Audio input: 40 channels x 16 frames
    let audio_frames = Tensor::<InferenceBackend, 3>::random(
        [batch, 40, 16],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    // 3. Vision input: 32x32 frame = 4 patches
    let vision_frame = Tensor::<InferenceBackend, 4>::random(
        [batch, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Interleave all three modalities into a single continuous representation
    let sequence = model.interleave_modalities(
        Some(text_tokens),
        Some(audio_frames),
        Some((vision_frame, 0)),
    );

    let [b, total_len, d] = sequence.dims();
    assert_eq!(b, batch);
    assert_eq!(d, d_model);
    assert!(total_len > 4); // Vision (4) + Audio (>0) + Text (4)

    // Full sequence forward through Mamba SSM + Attention
    let logits = model.forward_embeddings(sequence.clone());
    assert_eq!(logits.dims(), [b, total_len, vocab_size]);

    // Recurrent O(1) step decoding over one embedding vector
    let (prefill_logits, state) = model.prefill_embeddings(sequence);
    assert_eq!(prefill_logits.dims()[1], total_len);

    let next_step_vector = Tensor::<InferenceBackend, 2>::random(
        [batch, d_model],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (step_logits, next_state) = model.step_embedding(next_step_vector, state);

    assert_eq!(step_logits.dims(), [batch, vocab_size]);
    assert_eq!(next_state.position, total_len + 1);
}
