//! Omnimodal Any-to-Any Projection Stems (Audio, Vision, and Text) for SAIL.
//!
//! Maps continuous acoustic streams and multi-frame visual patches directly into the
//! shared latent dimension `d_model` of the HybridLm (Mamba + Attention + MoE) backbone.
//!
//! Architecture:
//! - Audio: 1D Causal Convolution stem with temporal striding (e.g. 80 Mel bins -> d_model).
//! - Vision: 2D Spatial Patch projection (e.g. 16x16 patches) + sinusoidal spatial & learned temporal embeddings.
//! - Shared Sequence: Arbitrary interleaved sequences of Text, Audio, and Video tokens processed
//!   with O(1) recurrent step latency through Mamba SSM layers.

use std::fs;
use std::path::Path;

use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};

use super::hybrid::{HybridLm, HybridLmConfig, HybridState};

/// Configuration for 1D Causal Audio Projection Stem.
#[derive(Config, Debug)]
pub struct AudioProjectorConfig {
    /// Input feature dimension per frame (e.g. 80 Mel-filterbank channels or 1 raw PCM channel).
    #[config(default = 80)]
    pub in_channels: usize,
    /// Target model latent dimension.
    pub d_model: usize,
    /// Convolutional kernel size for causal audio patch grouping.
    #[config(default = 5)]
    pub kernel_size: usize,
    /// Temporal striding to reduce acoustic frame rate to semantic token rate.
    #[config(default = 2)]
    pub stride: usize,
}

impl AudioProjectorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioProjector<B> {
        let mid_channels = self.d_model / 2;
        let conv1 = Conv1dConfig::new(self.in_channels, mid_channels, self.kernel_size)
            .with_stride(self.stride)
            .init(device);
        let conv2 = Conv1dConfig::new(mid_channels, self.d_model, self.kernel_size)
            .with_stride(self.stride)
            .init(device);
        let proj = LinearConfig::new(self.d_model, self.d_model).init(device);

        AudioProjector {
            conv1,
            conv2,
            proj,
            kernel_size: self.kernel_size,
        }
    }
}

/// 1D Causal Audio Projector converting acoustic spectrograms/waveforms into `d_model` embeddings.
#[derive(Module, Debug)]
pub struct AudioProjector<B: Backend> {
    pub conv1: Conv1d<B>,
    pub conv2: Conv1d<B>,
    pub proj: Linear<B>,
    pub kernel_size: usize,
}

impl<B: Backend> AudioProjector<B> {
    /// Projects an audio feature tensor `[batch, in_channels, time_frames]`
    /// into sequence embeddings `[batch, time_steps, d_model]`.
    pub fn forward(&self, audio_frames: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, _t] = audio_frames.dims();
        let pad = self.kernel_size - 1;
        let device = audio_frames.device();

        // Causal zero padding along time dimension (dim 2)
        let zeros1 = Tensor::<B, 3>::zeros([b, c, pad], &device);
        let padded1 = Tensor::cat(vec![zeros1, audio_frames], 2);
        let x = burn::tensor::activation::silu(self.conv1.forward(padded1));

        let [xb, xc, _xt] = x.dims();
        let zeros2 = Tensor::<B, 3>::zeros([xb, xc, pad], &device);
        let padded2 = Tensor::cat(vec![zeros2, x], 2);
        let x2 = burn::tensor::activation::silu(self.conv2.forward(padded2));

        // Transpose from [batch, d_model, time] to [batch, time, d_model]
        let transposed = x2.swap_dims(1, 2);
        self.proj.forward(transposed)
    }
}

/// Configuration for 2D Spatial Patch Vision Projector.
#[derive(Config, Debug)]
pub struct VisionProjectorConfig {
    /// Input image channels (3 for RGB, 1 for grayscale).
    #[config(default = 3)]
    pub in_channels: usize,
    /// Spatial patch dimension (e.g. 16 for 16x16 patches).
    #[config(default = 16)]
    pub patch_size: usize,
    /// Target model latent dimension.
    pub d_model: usize,
    /// Maximum temporal video frames in context window.
    #[config(default = 64)]
    pub max_frames: usize,
    /// Maximum spatial patches per frame (e.g. 14x14 = 196 for 224x224 image).
    #[config(default = 256)]
    pub max_patches_per_frame: usize,
}

impl VisionProjectorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionProjector<B> {
        let patch_conv = Conv2dConfig::new(
            [self.in_channels, self.d_model],
            [self.patch_size, self.patch_size],
        )
        .with_stride([self.patch_size, self.patch_size])
        .init(device);

        let temporal_embed = EmbeddingConfig::new(self.max_frames, self.d_model).init(device);
        let spatial_embed =
            EmbeddingConfig::new(self.max_patches_per_frame, self.d_model).init(device);
        let proj = LinearConfig::new(self.d_model, self.d_model).init(device);

        VisionProjector {
            patch_conv,
            temporal_embed,
            spatial_embed,
            proj,
            patch_size: self.patch_size,
        }
    }
}

/// 2D Vision Patch Projector converting images or video frames into `d_model` patch sequence embeddings.
#[derive(Module, Debug)]
pub struct VisionProjector<B: Backend> {
    pub patch_conv: Conv2d<B>,
    pub temporal_embed: Embedding<B>,
    pub spatial_embed: Embedding<B>,
    pub proj: Linear<B>,
    pub patch_size: usize,
}

impl<B: Backend> VisionProjector<B> {
    /// Projects a batch of video frames `[batch, channels, height, width]` at a given `frame_idx`
    /// into patch embeddings `[batch, num_patches, d_model]`.
    pub fn forward_frame(&self, frame: Tensor<B, 4>, frame_idx: usize) -> Tensor<B, 3> {
        let [batch, _c, h, w] = frame.dims();
        let patches_h = h / self.patch_size;
        let patches_w = w / self.patch_size;
        let num_patches = patches_h * patches_w;

        // Conv2d with stride=patch_size maps each patch to d_model: [batch, d_model, patches_h, patches_w]
        let patch_features = self.patch_conv.forward(frame);
        let d_out = patch_features.dims()[1];

        // Reshape to [batch, d_model, num_patches] then swap to [batch, num_patches, d_model]
        let patches = patch_features
            .reshape([batch, d_out, num_patches])
            .swap_dims(1, 2);

        // Add spatial patch position embeddings
        let device = &patches.device();
        let patch_indices =
            Tensor::<B, 1, Int>::arange(0..num_patches as i64, device).reshape([1, num_patches]);
        let spatial = self.spatial_embed.forward(patch_indices);

        // Add temporal frame position embeddings
        let frame_index =
            Tensor::<B, 1, Int>::from_data([frame_idx as i32], device).reshape([1, 1]);
        let temporal = self.temporal_embed.forward(frame_index);

        let combined = patches + spatial + temporal;
        self.proj.forward(combined)
    }
}

/// Combined Omnimodal Configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OmnimodalConfig {
    pub hybrid: HybridLmConfig,
    pub audio: AudioProjectorConfig,
    pub vision: VisionProjectorConfig,
}

impl OmnimodalConfig {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {
            hybrid: HybridLmConfig::new(vocab_size).with_d_model(d_model),
            audio: AudioProjectorConfig::new(d_model),
            vision: VisionProjectorConfig::new(d_model),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> OmnimodalLm<B> {
        OmnimodalLm::new(&self.hybrid, &self.audio, &self.vision, device)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?;
        serde_json::from_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// A multimodal batch containing any combination of text, audio, and visual data.
#[derive(Clone, Debug)]
pub struct OmnimodalBatch<B: Backend> {
    pub text_inputs: Option<Tensor<B, 2, Int>>,
    pub audio_frames: Option<Tensor<B, 3>>,
    pub vision_frame: Option<(Tensor<B, 4>, usize)>,
    pub targets: Tensor<B, 2, Int>,
}

/// Unified Omnimodal Hybrid Model: Combines Text BPE, Audio Projector, and Vision Projector
/// into a single seamless Mamba SSM + Attention + MoE architecture.
#[derive(Module, Debug)]
pub struct OmnimodalLm<B: Backend> {
    pub hybrid: HybridLm<B>,
    pub audio_projector: AudioProjector<B>,
    pub vision_projector: VisionProjector<B>,
    pub d_model: usize,
}

impl<B: Backend> OmnimodalLm<B> {
    pub fn new(
        hybrid_cfg: &HybridLmConfig,
        audio_cfg: &AudioProjectorConfig,
        vision_cfg: &VisionProjectorConfig,
        device: &B::Device,
    ) -> Self {
        let hybrid = hybrid_cfg.init(device);
        let audio_projector = audio_cfg.init(device);
        let vision_projector = vision_cfg.init(device);

        Self {
            hybrid,
            audio_projector,
            vision_projector,
            d_model: hybrid_cfg.d_model,
        }
    }

    /// Evaluates pre-embedded sequence `[batch, seq_len, d_model]` directly through the hybrid backbone.
    pub fn forward_embeddings(&self, embeddings: Tensor<B, 3>) -> Tensor<B, 3> {
        self.hybrid.forward_embeddings(embeddings)
    }

    /// Processes pre-embedded sequence and initializes recurrent state.
    pub fn prefill_embeddings(&self, embeddings: Tensor<B, 3>) -> (Tensor<B, 3>, HybridState<B>) {
        self.hybrid.prefill_embeddings(embeddings)
    }

    /// O(1) recurrent step decoding for any modality embedding vector `[batch, d_model]`.
    pub fn step_embedding(
        &self,
        embedding_step: Tensor<B, 2>,
        state: HybridState<B>,
    ) -> (Tensor<B, 2>, HybridState<B>) {
        self.hybrid.step_embedding(embedding_step, state)
    }

    /// Interleaves text tokens, audio frames, and video patches into a single continuous sequence.
    pub fn interleave_modalities(
        &self,
        text_tokens: Option<Tensor<B, 2, Int>>,
        audio_frames: Option<Tensor<B, 3>>,
        vision_frame: Option<(Tensor<B, 4>, usize)>,
    ) -> Tensor<B, 3> {
        let mut components: Vec<Tensor<B, 3>> = Vec::new();

        if let Some((frame, frame_idx)) = vision_frame {
            components.push(self.vision_projector.forward_frame(frame, frame_idx));
        }

        if let Some(audio) = audio_frames {
            components.push(self.audio_projector.forward(audio));
        }

        if let Some(tokens) = text_tokens {
            components.push(self.hybrid.embedding.forward(tokens));
        }

        if components.is_empty() {
            panic!("Cannot interleave empty modality inputs");
        }

        if components.len() == 1 {
            components.pop().unwrap()
        } else {
            crate::model::cat_dense(components, 1)
        }
    }

    /// Forwards an OmnimodalBatch through all modality stems and the hybrid backbone.
    pub fn forward_batch(&self, batch: &OmnimodalBatch<B>) -> Tensor<B, 3> {
        let sequence = self.interleave_modalities(
            batch.text_inputs.clone(),
            batch.audio_frames.clone(),
            batch.vision_frame.clone(),
        );
        self.forward_embeddings(sequence)
    }

    /// Computes cross-entropy next-token loss on the target tokens of a multimodal batch.
    pub fn compute_loss(
        &self,
        batch: &OmnimodalBatch<B>,
        loss_fn: &CrossEntropyLoss<B>,
    ) -> Tensor<B, 1> {
        let logits = self.forward_batch(batch);
        let [b, l, v] = logits.dims();
        let targets = batch.targets.clone();
        let [tb, tl] = targets.dims();

        // Slice the logits corresponding to target positions: last tl positions
        let start = l.saturating_sub(tl);
        let actual_tl = l - start;
        let sliced = logits.slice([0..b, start..l, 0..v]);
        let flat_logits = sliced.reshape([b * actual_tl, v]);
        let sliced_targets = if tl > actual_tl {
            targets.slice([0..tb, (tl - actual_tl)..tl])
        } else {
            targets
        };
        let flat_targets = sliced_targets.reshape([b * actual_tl]);
        loss_fn.forward(flat_logits, flat_targets)
    }

    /// Total parameters across all modality stems and the hybrid LM stack.
    pub fn total_params(&self) -> usize {
        self.num_params()
    }
}

/// Executes a single optimization training step on an Omnimodal model.
/// Updates weights across both the hybrid backbone, audio projector, and vision projector.
pub fn train_omnimodal_step<B: AutodiffBackend, O: Optimizer<OmnimodalLm<B>, B>>(
    model: OmnimodalLm<B>,
    optim: &mut O,
    batch: &OmnimodalBatch<B>,
    loss_fn: &CrossEntropyLoss<B>,
    lr: f64,
) -> Result<(OmnimodalLm<B>, f32), String> {
    let loss = model.compute_loss(batch, loss_fn);
    let loss_val: f32 = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    if !loss_val.is_finite() {
        return Err(format!("non-finite omnimodal loss: {loss_val}"));
    }

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    crate::train::check_finite(&model, Some(&grads))?;

    let updated_model = optim.step(lr, model, grads);
    crate::train::check_finite(&updated_model, None)?;

    Ok((updated_model, loss_val))
}
