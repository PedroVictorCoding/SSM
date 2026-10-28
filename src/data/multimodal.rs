//! Multimodal Data Structures, Feature Extractors, and Streaming Ingestion for SAIL.
//!
//! Provides:
//! - Pure Rust Mel-spectrogram audio feature extraction (STFT + Mel filterbank) and WAV loading.
//! - Vision pixel normalization and PPM/RGB image parsing.
//! - Structured `OmnimodalRecord` schema (text prompt, audio, vision, target).
//! - `OmnimodalDataset` with Experience Replay sampling and batch collation into `OmnimodalBatch`.
//! - `IncomingMultimodalWatcher` for live monitoring and ingestion of `.omni.jsonl` files in `data/incoming/`.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use burn::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::Tokenizer;
use crate::model::multimodal::OmnimodalBatch;

/// Raw or pre-extracted acoustic features for an audio sequence.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AudioData {
    /// Number of feature channels per time frame (e.g. 80 Mel bins).
    pub channels: usize,
    /// Number of temporal frames.
    pub frames: usize,
    /// Flat feature buffer of length `channels * frames` in channel-major or frame-major order.
    pub data: Vec<f32>,
}

/// Raw or pre-extracted spatial pixel features for a vision frame.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VisionData {
    /// Number of color channels (e.g. 3 for RGB, 1 for grayscale).
    pub channels: usize,
    /// Height in pixels.
    pub height: usize,
    /// Width in pixels.
    pub width: usize,
    /// Flat pixel buffer of length `channels * height * width` (values in [-1.0, 1.0]).
    pub data: Vec<f32>,
}

/// A multimodal training sample containing any combination of modalities and target text.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct OmnimodalRecord {
    /// Optional prefix text prompt (e.g. "Transcribe audio:" or "Describe image:").
    #[serde(default)]
    pub text: Option<String>,
    /// Target text sequence for next-token cross-entropy supervision.
    pub target: String,
    /// Optional acoustic features.
    #[serde(default)]
    pub audio: Option<AudioData>,
    /// Optional visual frame.
    #[serde(default)]
    pub vision: Option<VisionData>,
    /// Optional relative path to a WAV audio file on disk.
    #[serde(default)]
    pub audio_file: Option<String>,
    /// Optional relative path to a PPM image file on disk.
    #[serde(default)]
    pub image_file: Option<String>,
    /// Optional provenance tracking origin of the record.
    #[serde(default)]
    pub provenance: Option<String>,
}

impl OmnimodalRecord {
    /// Validates the record fields, rejecting empty targets, mismatched shapes, or non-finite values.
    pub fn validate(&self, expected_audio_channels: Option<usize>) -> Result<(), String> {
        if self.target.trim().is_empty() {
            return Err("target sequence cannot be empty".to_string());
        }
        if let Some(prov) = &self.provenance {
            let lower = prov.to_lowercase();
            if lower.contains("runtime_synthetic")
                || lower.contains("agent_trace")
                || lower.contains("session_capture")
                || lower.contains("model_generated")
            {
                return Err(format!(
                    "prohibited provenance '{prov}': recursive self-improvement is strictly forbidden by ALIGNMENT.md"
                ));
            }
        }
        if let Some(audio) = &self.audio {
            if audio.channels == 0 || audio.frames == 0 {
                return Err(format!(
                    "invalid audio shape: channels={}, frames={}",
                    audio.channels, audio.frames
                ));
            }
            if let Some(exp_ch) = expected_audio_channels {
                if audio.channels != exp_ch {
                    return Err(format!(
                        "mismatched audio channels: expected {}, got {}",
                        exp_ch, audio.channels
                    ));
                }
            }
            if audio.data.len() != audio.channels * audio.frames {
                return Err(format!(
                    "audio data length {} != channels {} * frames {}",
                    audio.data.len(),
                    audio.channels,
                    audio.frames
                ));
            }
            if audio.data.iter().any(|v| !v.is_finite()) {
                return Err("non-finite value in audio data".to_string());
            }
        }
        if let Some(vision) = &self.vision {
            if vision.channels != 3 || vision.height == 0 || vision.width == 0 {
                return Err(format!(
                    "invalid vision shape: channels={}, height={}, width={}",
                    vision.channels, vision.height, vision.width
                ));
            }
            if vision.data.len() != vision.channels * vision.height * vision.width {
                return Err(format!(
                    "vision data length {} != 3 * {} * {}",
                    vision.data.len(),
                    vision.height,
                    vision.width
                ));
            }
            if vision.data.iter().any(|v| !v.is_finite()) {
                return Err("non-finite pixel value in vision data".to_string());
            }
        }
        Ok(())
    }
}

/// Computes Mel-filterbank energies from raw PCM audio samples using discrete STFT.
///
/// Parameters:
/// - `samples`: mono f32 PCM samples normalized in `[-1.0, 1.0]`.
/// - `sample_rate`: audio sampling rate in Hz (e.g. 16000).
/// - `n_mels`: number of Mel filterbank channels (default 80).
/// - `n_fft`: FFT window length (default 400 = 25ms at 16kHz).
/// - `hop_length`: frame shift (default 160 = 10ms at 16kHz).
pub fn extract_mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
) -> AudioData {
    if samples.len() < n_fft {
        // Zero pad if shorter than one window
        let mut padded = samples.to_vec();
        padded.resize(n_fft, 0.0);
        return extract_mel_spectrogram(&padded, sample_rate, n_mels, n_fft, hop_length);
    }

    let num_frames = (samples.len() - n_fft) / hop_length + 1;
    let n_freq_bins = n_fft / 2 + 1;

    // Precompute Hann window
    let mut hann = Vec::with_capacity(n_fft);
    for i in 0..n_fft {
        let val = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32).cos());
        hann.push(val);
    }

    // Precompute Mel filterbank matrix: [n_mels, n_freq_bins]
    let mel_filters = create_mel_filterbank(sample_rate, n_fft, n_mels);

    // Flat buffer for output: [n_mels, num_frames]
    let mut out_data = vec![0.0f32; n_mels * num_frames];

    // Compute power spectrum per frame and apply filterbank
    for t in 0..num_frames {
        let frame_start = t * hop_length;
        let frame_slice = &samples[frame_start..frame_start + n_fft];

        // Discrete Fourier Transform magnitude squared
        let mut power_spec = vec![0.0f32; n_freq_bins];
        for k in 0..n_freq_bins {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for n in 0..n_fft {
                let windowed = frame_slice[n] * hann[n];
                let angle = 2.0 * std::f32::consts::PI * (k * n) as f32 / n_fft as f32;
                real += windowed * angle.cos();
                imag -= windowed * angle.sin();
            }
            power_spec[k] = (real * real + imag * imag) / n_fft as f32;
        }

        // Apply Mel filters
        for m in 0..n_mels {
            let mut mel_energy = 0.0f32;
            let filter_row = &mel_filters[m];
            for k in 0..n_freq_bins {
                mel_energy += filter_row[k] * power_spec[k];
            }
            // Log Mel compression
            let log_mel = (mel_energy.max(1e-5)).ln();
            // Store in [channel, frame] layout
            out_data[m * num_frames + t] = log_mel;
        }
    }

    AudioData {
        channels: n_mels,
        frames: num_frames,
        data: out_data,
    }
}

/// Creates triangular Mel filterbank weights of shape `[n_mels, n_freq_bins]`.
fn create_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let n_freq_bins = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;

    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(nyquist);

    // Uniformly spaced points in Mel scale
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=(n_mels + 1) {
        let mel = min_mel + (max_mel - min_mel) * (i as f32 / (n_mels + 1) as f32);
        mel_points.push(mel_to_hz(mel));
    }

    // Convert Hz to FFT bin indices
    let mut bin_indices = Vec::with_capacity(n_mels + 2);
    for hz in mel_points {
        let bin = ((n_fft + 1) as f32 * hz / sample_rate as f32).floor() as usize;
        bin_indices.push(bin.min(n_freq_bins - 1));
    }

    let mut filterbank = vec![vec![0.0f32; n_freq_bins]; n_mels];
    for m in 0..n_mels {
        let left = bin_indices[m];
        let center = bin_indices[m + 1];
        let right = bin_indices[m + 2];

        for k in left..center {
            if center > left {
                filterbank[m][k] = (k - left) as f32 / (center - left) as f32;
            }
        }
        for k in center..=right {
            if right > center {
                filterbank[m][k] = (right - k) as f32 / (right - center) as f32;
            }
        }
    }

    filterbank
}

/// Reads a standard 16-bit PCM WAV file via `hound` and extracts Mel spectrogram features.
pub fn load_wav_as_mel(path: impl AsRef<Path>, n_mels: usize) -> std::io::Result<AudioData> {
    if n_mels == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "n_mels must be > 0",
        ));
    }
    let mut reader = hound::WavReader::open(path.as_ref())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let spec = reader.spec();
    if spec.sample_rate < 4000 || spec.sample_rate > 192000 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported audio sample rate: {}", spec.sample_rate),
        ));
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
    };

    if samples.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "audio file contains zero samples",
        ));
    }
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "non-finite audio samples in WAV",
        ));
    }

    // If stereo or multi-channel, average to mono
    let mono_samples: Vec<f32> = if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples
            .chunks_exact(ch)
            .map(|chunk| chunk.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        samples
    };

    Ok(extract_mel_spectrogram(
        &mono_samples,
        spec.sample_rate,
        n_mels,
        400,
        160,
    ))
}

/// Reads a binary Netpbm PPM (P6) or plain PPM (P3) image and normalizes pixels to `[-1.0, 1.0]`.
pub fn load_ppm_image(path: impl AsRef<Path>) -> std::io::Result<VisionData> {
    let content = fs::read(path.as_ref())?;
    if content.len() < 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "PPM file too short",
        ));
    }

    let is_binary = content.starts_with(b"P6");
    let is_plain = content.starts_with(b"P3");
    if !is_binary && !is_plain {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Unsupported image format: must be P3 or P6 PPM",
        ));
    }

    let mut cursor = 2;
    // Helper to read next non-whitespace non-comment token
    let mut next_token = || -> Option<String> {
        let mut token = String::new();
        while cursor < content.len() {
            let b = content[cursor];
            cursor += 1;
            if b == b'#' {
                // skip comment to end of line
                while cursor < content.len() && content[cursor] != b'\n' {
                    cursor += 1;
                }
            } else if b.is_ascii_whitespace() {
                if !token.is_empty() {
                    return Some(token);
                }
            } else {
                token.push(b as char);
            }
        }
        if !token.is_empty() { Some(token) } else { None }
    };

    let width: usize = next_token()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing PPM width"))?;
    let height: usize = next_token().and_then(|s| s.parse().ok()).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing PPM height")
    })?;
    let max_val: f32 = next_token().and_then(|s| s.parse().ok()).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing PPM maxval")
    })?;

    if width == 0 || height == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid PPM dimensions: {width}x{height}"),
        ));
    }
    if max_val <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "max color value must be > 0",
        ));
    }

    let total_pixels = width * height;
    let mut r = Vec::with_capacity(total_pixels);
    let mut g = Vec::with_capacity(total_pixels);
    let mut b = Vec::with_capacity(total_pixels);

    if is_binary {
        // Binary P6: raw bytes follow header
        let raw = &content[cursor..];
        if raw.len() < total_pixels * 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "truncated binary PPM data",
            ));
        }
        for chunk in raw[..total_pixels * 3].chunks_exact(3) {
            r.push((chunk[0] as f32 / max_val) * 2.0 - 1.0);
            g.push((chunk[1] as f32 / max_val) * 2.0 - 1.0);
            b.push((chunk[2] as f32 / max_val) * 2.0 - 1.0);
        }
    } else {
        // Plain text P3
        for _ in 0..total_pixels {
            let rv: f32 = next_token().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let gv: f32 = next_token().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let bv: f32 = next_token().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            r.push((rv / max_val) * 2.0 - 1.0);
            g.push((gv / max_val) * 2.0 - 1.0);
            b.push((bv / max_val) * 2.0 - 1.0);
        }
    }

    if r.iter()
        .chain(g.iter())
        .chain(b.iter())
        .any(|p| !p.is_finite())
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "non-finite pixel values in image",
        ));
    }

    // Channel-first layout [3, H, W]
    let mut data = Vec::with_capacity(total_pixels * 3);
    data.extend(r);
    data.extend(g);
    data.extend(b);

    Ok(VisionData {
        channels: 3,
        height,
        width,
        data,
    })
}

/// Dynamic, thread-safe multimodal dataset supporting online ingestion and Experience Replay.
#[derive(Clone)]
pub struct OmnimodalDataset {
    records: Arc<RwLock<Vec<OmnimodalRecord>>>,
    new_data_watermark: Arc<RwLock<usize>>,
    pub replay_ratio: f64,
    pub min_samples_to_train: usize,
    pub max_replay_samples: Option<usize>,
}

/// Serializable snapshot of the multimodal streaming buffer for checkpointing and deterministic resumption.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OmnimodalSnapshot {
    pub records: Vec<OmnimodalRecord>,
    pub new_data_watermark: usize,
    pub processed_files: Vec<PathBuf>,
}

impl OmnimodalDataset {
    pub fn new(initial_records: Vec<OmnimodalRecord>, replay_ratio: f64) -> Self {
        let watermark = initial_records.len();
        Self {
            records: Arc::new(RwLock::new(initial_records)),
            new_data_watermark: Arc::new(RwLock::new(watermark)),
            replay_ratio: replay_ratio.clamp(0.0, 1.0),
            min_samples_to_train: 1,
            max_replay_samples: None,
        }
    }

    pub fn with_max_replay_samples(mut self, max: usize) -> Self {
        self.max_replay_samples = Some(max);
        self
    }

    pub fn snapshot(&self) -> OmnimodalSnapshot {
        let recs = self.records.read().unwrap();
        let watermark = *self.new_data_watermark.read().unwrap();
        OmnimodalSnapshot {
            records: recs.clone(),
            new_data_watermark: watermark,
            processed_files: Vec::new(),
        }
    }

    pub fn restore(&self, snapshot: &OmnimodalSnapshot) -> Result<(), String> {
        if snapshot.new_data_watermark > snapshot.records.len() {
            return Err("watermark exceeds record count".into());
        }
        let mut recs = self.records.write().unwrap();
        let mut watermark = self.new_data_watermark.write().unwrap();
        *recs = snapshot.records.clone();
        *watermark = snapshot.new_data_watermark;
        Ok(())
    }

    /// Reserves a held-out fraction of records from the initial buffer for frozen validation.
    /// Those records are removed from the training buffer to guarantee no train/val leakage.
    pub fn reserve_validation(&self, fraction: f64) -> Vec<OmnimodalRecord> {
        let mut recs = self.records.write().unwrap();
        let mut watermark = self.new_data_watermark.write().unwrap();
        let count = (recs.len() as f64 * fraction).round() as usize;
        if count == 0 || recs.len().saturating_sub(count) < self.min_samples_to_train {
            return Vec::new();
        }
        let split = recs.len() - count;
        let held_out = recs.split_off(split);
        *watermark = (*watermark).min(split);
        held_out
    }

    pub fn len(&self) -> usize {
        self.records.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn append_record(&self, record: OmnimodalRecord) {
        let mut recs = self.records.write().unwrap();
        recs.push(record);
    }

    pub fn append_records(&self, new_records: Vec<OmnimodalRecord>) {
        if new_records.is_empty() {
            return;
        }
        let mut recs = self.records.write().unwrap();
        let current_len = recs.len();
        recs.extend(new_records);
        let mut watermark = self.new_data_watermark.write().unwrap();
        *watermark = current_len;

        if let Some(max) = self.max_replay_samples {
            if max > 0 && recs.len() > max {
                let anchor_count = (max as f64 * 0.20) as usize;
                let recent_count = max.saturating_sub(anchor_count);
                let total = recs.len();
                let mut pruned = Vec::with_capacity(max);
                pruned.extend_from_slice(&recs[..anchor_count]);
                pruned.extend_from_slice(&recs[total - recent_count..]);
                *recs = pruned;
                *watermark = (*watermark).min(recs.len());
            }
        }
    }

    /// Samples indices using experience replay: mixes historical records with freshly arrived records.
    pub fn sample_indices<R: Rng>(&self, batch_size: usize, rng: &mut R) -> Vec<usize> {
        let recs = self.records.read().unwrap();
        let total = recs.len();
        if total == 0 {
            return Vec::new();
        }

        let watermark = *self.new_data_watermark.read().unwrap();
        let num_replay = ((batch_size as f64) * self.replay_ratio).round() as usize;

        let mut indices = Vec::with_capacity(batch_size);

        // Historical sample
        if watermark > 0 && num_replay > 0 {
            for _ in 0..num_replay {
                indices.push(rng.gen_range(0..watermark));
            }
        }

        // Fresh sample (from watermark to end)
        if watermark < total {
            for _ in indices.len()..batch_size {
                indices.push(rng.gen_range(watermark..total));
            }
        } else {
            // Uniform fallback
            for _ in indices.len()..batch_size {
                indices.push(rng.gen_range(0..total));
            }
        }

        indices
    }

    /// Samples a collated `OmnimodalBatch` ready for the model's forward step.
    pub fn sample_batch<B: Backend, R: Rng>(
        &self,
        batch_size: usize,
        tokenizer: &Tokenizer,
        rng: &mut R,
        device: &B::Device,
    ) -> Option<OmnimodalBatch<B>> {
        let recs = self.records.read().unwrap();
        if recs.len() < self.min_samples_to_train {
            return None;
        }

        let indices = self.sample_indices(batch_size, rng);
        if indices.is_empty() {
            return None;
        }

        let selected: Vec<&OmnimodalRecord> = indices.iter().map(|&i| &recs[i]).collect();
        collate_records(&selected, tokenizer, device)
    }
}

/// Collates an arbitrary slice of OmnimodalRecord references into an OmnimodalBatch.
pub fn collate_records<B: Backend>(
    selected: &[&OmnimodalRecord],
    tokenizer: &Tokenizer,
    device: &B::Device,
) -> Option<OmnimodalBatch<B>> {
    if selected.is_empty() {
        return None;
    }
    let batch_size = selected.len();

    // 1. Process Targets and Text Inputs
    let mut target_token_seqs = Vec::with_capacity(batch_size);
    let mut text_token_seqs = Vec::with_capacity(batch_size);
    let mut max_target_len = 1;
    let mut max_text_len = 0;

    for r in selected {
        let tokens = tokenizer.encode(&r.target).unwrap_or_else(|_| vec![1]);
        let t_len = tokens.len().max(1);
        if t_len > max_target_len {
            max_target_len = t_len;
        }

        let mut inp = if let Some(t) = &r.text {
            tokenizer.encode(t).unwrap_or_default()
        } else {
            Vec::new()
        };

        // Teacher forcing: the sequence includes target prefix [..t_len - 1]
        if tokens.len() > 1 {
            inp.extend_from_slice(&tokens[..tokens.len() - 1]);
        }

        if inp.len() > max_text_len {
            max_text_len = inp.len();
        }

        target_token_seqs.push(tokens);
        text_token_seqs.push(inp);
    }

    // Collate targets with pad 0
    let mut flat_targets = Vec::with_capacity(batch_size * max_target_len);
    for tokens in &target_token_seqs {
        flat_targets.extend(tokens);
        for _ in tokens.len()..max_target_len {
            flat_targets.push(0);
        }
    }
    let target_tensor = Tensor::<B, 1, Int>::from_ints(
        &flat_targets.iter().map(|&v| v as i32).collect::<Vec<_>>()[..],
        device,
    )
    .reshape([batch_size, max_target_len]);

    // 2. Collate Text Inputs (ensuring input tokens exist if no vision/audio)
    let has_audio = selected.iter().any(|r| r.audio.is_some());
    let has_vision = selected.iter().any(|r| r.vision.is_some());
    let need_dummy = !has_audio && !has_vision && max_text_len == 0;
    let effective_max_text = if need_dummy { 1 } else { max_text_len };

    let text_inputs = if effective_max_text > 0 {
        let mut flat_text = Vec::with_capacity(batch_size * effective_max_text);
        for tokens in &text_token_seqs {
            flat_text.extend(tokens);
            for _ in tokens.len()..effective_max_text {
                flat_text.push(0);
            }
        }
        let tensor = Tensor::<B, 1, Int>::from_ints(
            &flat_text.iter().map(|&v| v as i32).collect::<Vec<_>>()[..],
            device,
        )
        .reshape([batch_size, effective_max_text]);
        Some(tensor)
    } else {
        None
    };

    // 3. Process Audio Features (if any present)
    let audio_frames = if has_audio {
        let mut audio_channels = 80;
        let mut max_frames = 1;

        for r in selected {
            if let Some(a) = &r.audio {
                audio_channels = a.channels;
                if a.frames > max_frames {
                    max_frames = a.frames;
                }
            }
        }

        let mut flat_audio = vec![0.0f32; batch_size * audio_channels * max_frames];
        for (b_idx, r) in selected.iter().enumerate() {
            if let Some(a) = &r.audio {
                let channels = a.channels;
                let frames = a.frames;
                for c in 0..channels {
                    for t in 0..frames {
                        let src_idx = c * frames + t;
                        let dst_idx = (b_idx * audio_channels + c) * max_frames + t;
                        if src_idx < a.data.len() {
                            flat_audio[dst_idx] = a.data[src_idx];
                        }
                    }
                }
            }
        }

        let tensor = Tensor::<B, 1>::from_floats(&flat_audio[..], device).reshape([
            batch_size,
            audio_channels,
            max_frames,
        ]);
        Some(tensor)
    } else {
        None
    };

    // 4. Process Vision Frame (if any present)
    let vision_frame = if has_vision {
        let mut channels = 3;
        let mut height = 32;
        let mut width = 32;

        for r in selected {
            if let Some(v) = &r.vision {
                channels = v.channels;
                height = v.height;
                width = v.width;
                break;
            }
        }

        let frame_size = channels * height * width;
        let mut flat_vision = vec![0.0f32; batch_size * frame_size];
        for (b_idx, r) in selected.iter().enumerate() {
            if let Some(v) = &r.vision {
                let take = v.data.len().min(frame_size);
                let dst_start = b_idx * frame_size;
                flat_vision[dst_start..dst_start + take].copy_from_slice(&v.data[..take]);
            }
        }

        let tensor = Tensor::<B, 1>::from_floats(&flat_vision[..], device)
            .reshape([batch_size, channels, height, width]);
        Some((tensor, 0))
    } else {
        None
    };

    Some(OmnimodalBatch {
        text_inputs,
        audio_frames,
        vision_frame,
        targets: target_tensor,
    })
}

/// Collates an owned slice of OmnimodalRecord into an OmnimodalBatch.
pub fn batch_from_records<B: Backend>(
    records: &[OmnimodalRecord],
    tokenizer: &Tokenizer,
    device: &B::Device,
) -> Option<OmnimodalBatch<B>> {
    let refs: Vec<&OmnimodalRecord> = records.iter().collect();
    collate_records(&refs, tokenizer, device)
}

/// Watches an incoming directory (e.g. `data/incoming/`), ingesting any new `.omni.jsonl` files.
pub struct IncomingMultimodalWatcher {
    incoming_dir: PathBuf,
    processed_files: HashSet<PathBuf>,
}

impl IncomingMultimodalWatcher {
    pub fn new(incoming_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let incoming_dir = incoming_dir.as_ref().to_path_buf();
        fs::create_dir_all(&incoming_dir)?;
        Ok(Self {
            incoming_dir,
            processed_files: HashSet::new(),
        })
    }

    pub fn processed_files(&self) -> Vec<PathBuf> {
        let mut names: Vec<_> = self
            .processed_files
            .iter()
            .filter_map(|p| p.file_name().map(PathBuf::from))
            .collect();
        names.sort();
        names
    }

    pub fn restore_processed_files(&mut self, names: &[PathBuf]) {
        self.processed_files = names.iter().map(|p| self.incoming_dir.join(p)).collect();
    }

    /// Bound the producer's cleanup receipt to a spool identity, portable between
    /// a WSL downloader and a native Windows trainer viewing the same directory.
    pub fn ingestion_receipt(&self) -> serde_json::Value {
        let spool_id = fs::read(self.incoming_dir.join(".hf_streamer/state.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|state| {
                state
                    .get("id")
                    .and_then(|id| id.as_str())
                    .map(str::to_owned)
            });
        serde_json::json!({"spool_id": spool_id, "processed_files": self.processed_files()})
    }

    /// Scans the incoming folder for `.omni.jsonl` or `.omni.json` files and parses them into records.
    /// Resolves any relative `audio_file` or `image_file` paths relative to `incoming_dir`.
    pub fn scan_and_ingest(&mut self, dataset: &OmnimodalDataset, n_mels: usize) -> usize {
        let entries = match fs::read_dir(&self.incoming_dir) {
            Ok(e) => e,
            Err(_) => return 0,
        };

        let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();

        let mut total_ingested = 0;

        for path in paths {
            if !path.is_file() || self.processed_files.contains(&path) {
                continue;
            }

            let file_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if file_name.ends_with(".omni.jsonl") || file_name.ends_with(".omni.json") {
                if let Ok(file) = File::open(&path) {
                    let reader = BufReader::new(file);
                    let mut file_records = Vec::new();

                    for line in reader.lines().flatten() {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<OmnimodalRecord>(trimmed) {
                            Ok(mut record) => {
                                // Resolve audio file if audio tensor not already present
                                if record.audio.is_none() {
                                    if let Some(audio_rel) = &record.audio_file {
                                        let audio_path = self.incoming_dir.join(audio_rel);
                                        match load_wav_as_mel(&audio_path, n_mels) {
                                            Ok(mel) => record.audio = Some(mel),
                                            Err(e) => {
                                                eprintln!(
                                                    "[Multimodal Ingest] Rejecting audio {}: {e}",
                                                    audio_path.display()
                                                );
                                                continue;
                                            }
                                        }
                                    }
                                }

                                // Resolve image file if vision tensor not already present
                                if record.vision.is_none() {
                                    if let Some(img_rel) = &record.image_file {
                                        let img_path = self.incoming_dir.join(img_rel);
                                        match load_ppm_image(&img_path) {
                                            Ok(vis) => record.vision = Some(vis),
                                            Err(e) => {
                                                eprintln!(
                                                    "[Multimodal Ingest] Rejecting image {}: {e}",
                                                    img_path.display()
                                                );
                                                continue;
                                            }
                                        }
                                    }
                                }

                                if let Err(msg) = record.validate(Some(n_mels)) {
                                    eprintln!(
                                        "[Multimodal Ingest] Rejecting malformed record: {msg}"
                                    );
                                    continue;
                                }

                                file_records.push(record);
                            }
                            Err(e) => {
                                eprintln!(
                                    "[Multimodal Ingest] JSON syntax error in {file_name}: {e}"
                                );
                            }
                        }
                    }

                    if !file_records.is_empty() {
                        let count = file_records.len();
                        dataset.append_records(file_records);
                        total_ingested += count;
                        println!("[Multimodal Ingest] Ingested {count} records from {file_name}");
                        self.processed_files.insert(path);
                    }
                }
            }
        }

        total_ingested
    }
}
