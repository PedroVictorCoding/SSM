use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::nn::loss::CrossEntropyLossConfig;

use sail::data::multimodal::OmnimodalRecord;
use sail::data::{CharTokenizer, Tokenizer};
use sail::model::HybridLmConfig;
use sail::model::multimodal::{
    AudioProjectorConfig, OmnimodalConfig, OmnimodalLm, VisionProjectorConfig,
};

type Backend = NdArray<f32>;

#[test]
fn test_multimodal_modality_ablation_verifies_active_representation() {
    let device = NdArrayDevice::Cpu;
    let d_model = 16;
    let audio_channels = 80;
    let patch_size = 16;
    let vocab_size = 64;

    let text_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :.\n";
    let tokenizer = Tokenizer::Char(CharTokenizer::from_text(text_chars));

    let omni_cfg = OmnimodalConfig {
        hybrid: HybridLmConfig::new(vocab_size)
            .with_d_model(d_model)
            .with_n_layers(2)
            .with_d_state(4)
            .with_d_conv(2)
            .with_expand(2),
        audio: AudioProjectorConfig::new(d_model)
            .with_in_channels(audio_channels)
            .with_kernel_size(3)
            .with_stride(2),
        vision: VisionProjectorConfig::new(d_model)
            .with_in_channels(3)
            .with_patch_size(patch_size)
            .with_max_frames(1)
            .with_max_patches_per_frame(4),
    };

    let model: OmnimodalLm<Backend> = omni_cfg.init(&device);
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    // Create a multimodal record with active audio and vision
    let audio_feat = sail::data::multimodal::AudioData {
        channels: audio_channels,
        frames: 8,
        data: vec![0.5f32; audio_channels * 8],
    };
    let vision_feat = sail::data::multimodal::VisionData {
        channels: 3,
        height: 32,
        width: 32,
        data: vec![0.8f32; 3 * 32 * 32],
    };

    let record_full = OmnimodalRecord {
        text: None,
        target: "audio vision target".to_string(),
        audio: Some(audio_feat.clone()),
        vision: Some(vision_feat.clone()),
        audio_file: None,
        image_file: None,
        provenance: None,
    };

    // Ablated records
    let record_no_vision = OmnimodalRecord {
        text: None,
        target: "audio vision target".to_string(),
        audio: Some(audio_feat.clone()),
        vision: None,
        audio_file: None,
        image_file: None,
        provenance: None,
    };
    let record_no_audio = OmnimodalRecord {
        text: None,
        target: "audio vision target".to_string(),
        audio: None,
        vision: Some(vision_feat.clone()),
        audio_file: None,
        image_file: None,
        provenance: None,
    };
    let record_text_only = OmnimodalRecord {
        text: None,
        target: "audio vision target".to_string(),
        audio: None,
        vision: None,
        audio_file: None,
        image_file: None,
        provenance: None,
    };

    let batch_full =
        sail::data::multimodal::batch_from_records::<Backend>(&[record_full], &tokenizer, &device)
            .unwrap();
    let batch_no_vision = sail::data::multimodal::batch_from_records::<Backend>(
        &[record_no_vision],
        &tokenizer,
        &device,
    )
    .unwrap();
    let batch_no_audio = sail::data::multimodal::batch_from_records::<Backend>(
        &[record_no_audio],
        &tokenizer,
        &device,
    )
    .unwrap();
    let batch_text_only = sail::data::multimodal::batch_from_records::<Backend>(
        &[record_text_only],
        &tokenizer,
        &device,
    )
    .unwrap();

    println!(
        "batch_full vision is_some={:?}, audio is_some={:?}",
        batch_full.vision_frame.is_some(),
        batch_full.audio_frames.is_some()
    );
    println!(
        "batch_no_vis vision is_some={:?}, audio is_some={:?}",
        batch_no_vision.vision_frame.is_some(),
        batch_no_vision.audio_frames.is_some()
    );
    println!(
        "batch_no_aud vision is_some={:?}, audio is_some={:?}",
        batch_no_audio.vision_frame.is_some(),
        batch_no_audio.audio_frames.is_some()
    );
    println!(
        "batch_text vision is_some={:?}, audio is_some={:?}",
        batch_text_only.vision_frame.is_some(),
        batch_text_only.audio_frames.is_some()
    );

    let logits_full = model.forward_batch(&batch_full);
    let logits_text = model.forward_batch(&batch_text_only);
    println!("Logits full: {:?}", logits_full.dims());
    println!("Logits text: {:?}", logits_text.dims());

    let start_f = 46 - 26;
    let start_t = 40 - 26;
    let s_f = logits_full.clone().slice([0..1, start_f..46, 0..64]);
    let s_t = logits_text.clone().slice([0..1, start_t..40, 0..64]);
    let diff_logits: f32 = (s_f - s_t).abs().max().into_scalar();
    println!("Max logit diff at target positions between full and text: {diff_logits}");

    let loss_full: f32 = model.compute_loss(&batch_full, &loss_fn).into_scalar();
    let loss_no_vis: f32 = model.compute_loss(&batch_no_vision, &loss_fn).into_scalar();
    let loss_no_aud: f32 = model.compute_loss(&batch_no_audio, &loss_fn).into_scalar();
    let loss_text_only: f32 = model.compute_loss(&batch_text_only, &loss_fn).into_scalar();

    println!("\n[Modality Ablation Results]");
    println!("  Full Multimodal Loss:     {loss_full:.4}");
    println!(
        "  Vision-Ablated Loss:      {loss_no_vis:.4} (diff: {:.4})",
        (loss_full - loss_no_vis).abs()
    );
    println!(
        "  Audio-Ablated Loss:       {loss_no_aud:.4} (diff: {:.4})",
        (loss_full - loss_no_aud).abs()
    );
    println!(
        "  Text-Only Loss:           {loss_text_only:.4} (diff: {:.4})",
        (loss_full - loss_text_only).abs()
    );

    // Verify each modality contributes to representations (ablation alters loss)
    assert!(
        (loss_full - loss_no_vis).abs() > 1e-5,
        "Vision ablation should alter task loss"
    );
    assert!(
        (loss_full - loss_no_aud).abs() > 1e-4,
        "Audio ablation should alter task loss"
    );
    assert!(
        (loss_full - loss_text_only).abs() > 1e-5,
        "Full ablation should alter task loss"
    );
}
