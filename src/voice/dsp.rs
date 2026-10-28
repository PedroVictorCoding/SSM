//! Production Audio DSP: Acoustic Echo Cancellation (AEC), Noise Suppression, AGC, and Endpoint Detection.
//!
//! Provides:
//! - Normalized Least Mean Squares (NLMS) Acoustic Echo Cancellation with delay estimation.
//! - Dynamic noise floor estimation and spectral/envelope noise suppression.
//! - Automatic Gain Control (AGC) with soft-knee limiting.
//! - Multi-threshold speech endpoint detector with configurable hangover time and Push-to-Talk (PTT) support.
//! - Double-talk preservation ensuring the assistant hears the user while speaking.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::voice::AudioChunk;

/// Operational mode for microphone capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicrophoneMode {
    /// Always-listening with automatic speech endpoint detection.
    OpenMic,
    /// Manual push-to-talk: only processes speech when `active` is true.
    PushToTalk { active: bool },
}

/// Configuration parameters for voice activity and speech endpoint detection.
#[derive(Clone, Debug)]
pub struct SpeechEndpointConfig {
    /// RMS energy threshold to trigger speech start (default: 0.03).
    pub start_energy_threshold: f32,
    /// Lower energy threshold for speech continuation hysteresis (default: 0.015).
    pub stop_energy_threshold: f32,
    /// Minimum consecutive speech frames needed before confirming speech start (rejects clicks/coughs).
    pub min_speech_frames: usize,
    /// Hangover frames (e.g. 30-40 frames = 600-800ms at 20ms/frame) allowing normal pauses without cut-off.
    pub hangover_frames: usize,
    /// Maximum allowed single utterance duration in frames before forcing completion.
    pub max_utterance_frames: usize,
}

impl Default for SpeechEndpointConfig {
    fn default() -> Self {
        Self {
            start_energy_threshold: 0.03,
            stop_energy_threshold: 0.015,
            min_speech_frames: 4,       // ~80ms at 20ms frames
            hangover_frames: 35,        // ~700ms pause tolerance
            max_utterance_frames: 1500, // ~30 seconds max
        }
    }
}

/// Events emitted by the speech endpoint detector.
#[derive(Clone, Debug, PartialEq)]
pub enum EndpointEvent {
    Silence,
    SpeechStarted {
        sample_offset: u64,
        timestamp_ns: u64,
    },
    SpeechContinuing {
        duration_ms: f64,
    },
    SpeechEnded {
        start_offset: u64,
        end_offset: u64,
        speech_samples: Arc<[f32]>,
        confidence: f32,
    },
}

// ---------------------------------------------------------------------------
// 1. Acoustic Echo Cancellation (NLMS Adaptive Filter)
// ---------------------------------------------------------------------------

/// Normalized Least Mean Squares (NLMS) adaptive filter for Acoustic Echo Cancellation.
pub struct AcousticEchoCanceller {
    #[allow(dead_code)]
    filter_length: usize,
    weights: Vec<f32>,
    history: VecDeque<f32>,
    mu: f32,
    delay_samples: usize,
    ref_delay_buffer: VecDeque<f32>,
}

impl AcousticEchoCanceller {
    pub fn new(filter_length: usize, mu: f32, delay_samples: usize) -> Self {
        Self {
            filter_length: filter_length.max(16),
            weights: vec![0.0; filter_length.max(16)],
            history: VecDeque::from(vec![0.0; filter_length.max(16)]),
            mu: mu.clamp(0.001, 0.5),
            delay_samples,
            ref_delay_buffer: VecDeque::from(vec![0.0; delay_samples]),
        }
    }

    /// Estimates delay in samples between reference playback and microphone capture
    /// using normalized cross-correlation over a search window.
    pub fn estimate_delay(reference: &[f32], mic: &[f32], max_delay: usize) -> usize {
        if reference.is_empty() || mic.is_empty() || max_delay == 0 {
            return 0;
        }

        let search_len = 256.min(reference.len()).min(mic.len());
        let max_search = max_delay.min(mic.len().saturating_sub(search_len));
        let mut best_delay = 0;
        let mut max_corr = 0.0f32;

        for delay in 0..=max_search {
            if delay + search_len > mic.len() || search_len > reference.len() {
                break;
            }
            let mut corr = 0.0f32;
            let mut norm_ref = 1e-6f32;
            let mut norm_mic = 1e-6f32;

            for i in 0..search_len {
                let r = reference[i];
                let m = mic[delay + i];
                corr += r * m;
                norm_ref += r * r;
                norm_mic += m * m;
            }

            let norm = (norm_ref * norm_mic).sqrt();
            let score = (corr / norm).abs();
            if score > max_corr {
                max_corr = score;
                best_delay = delay;
            }
        }

        best_delay
    }

    pub fn set_delay(&mut self, delay: usize) {
        self.delay_samples = delay;
        self.ref_delay_buffer = VecDeque::from(vec![0.0; delay]);
    }

    /// Processes microphone samples against the playback reference.
    /// Returns `(cleaned_error_samples, echo_reduction_db)`.
    pub fn process(&mut self, mic_samples: &[f32], ref_samples: &[f32]) -> (Vec<f32>, f32) {
        let n = mic_samples.len();
        let mut cleaned = Vec::with_capacity(n);
        let mut mic_energy = 1e-6f32;
        let mut error_energy = 1e-6f32;

        for i in 0..n {
            let ref_in = if i < ref_samples.len() {
                ref_samples[i]
            } else {
                0.0
            };

            // Feed delayed reference buffer
            self.ref_delay_buffer.push_back(ref_in);
            let delayed_ref = self.ref_delay_buffer.pop_front().unwrap_or(0.0);

            // Feed FIR tapped-delay line
            self.history.push_front(delayed_ref);
            self.history.pop_back();

            // Compute echo estimate: y_hat = w^T * x
            let mut echo_estimate = 0.0f32;
            let mut ref_norm_sq = 1e-5f32;
            for (w, &x) in self.weights.iter().zip(self.history.iter()) {
                echo_estimate += *w * x;
                ref_norm_sq += x * x;
            }

            let d = mic_samples[i];
            // Error signal: e = d - y_hat
            let e = d - echo_estimate;
            cleaned.push(e);

            mic_energy += d * d;
            error_energy += e * e;

            // NLMS coefficient update: w = w + (mu / ||x||^2) * e * x
            // Skip update during strong double-talk (when mic energy is much higher than echo estimate)
            let is_double_talk = d.abs() > (echo_estimate.abs() * 2.5 + 0.05);
            if !is_double_talk {
                let step = (self.mu / ref_norm_sq) * e;
                for (w, &x) in self.weights.iter_mut().zip(self.history.iter()) {
                    *w += step * x;
                    // Clamp weights to prevent runaway instability
                    *w = w.clamp(-2.0, 2.0);
                }
            }
        }

        let erle_db = 10.0 * (mic_energy / error_energy).max(1.0).log10();
        (cleaned, erle_db)
    }
}

// ---------------------------------------------------------------------------
// 2. Noise Suppressor & Level Normalizer
// ---------------------------------------------------------------------------

/// Spectral and envelope-based noise gate tracking dynamic background noise floors.
pub struct NoiseSuppressor {
    noise_floor: f32,
    alpha_track: f32,
}

impl Default for NoiseSuppressor {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseSuppressor {
    pub fn new() -> Self {
        Self {
            noise_floor: 0.01,
            alpha_track: 0.05,
        }
    }

    /// Suppresses stationary background noise and returns cleaned samples.
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        // Measure frame RMS
        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        // Update noise floor slowly during low-energy quiet periods
        if rms < self.noise_floor * 2.0 {
            self.noise_floor = (1.0 - self.alpha_track) * self.noise_floor + self.alpha_track * rms;
            self.noise_floor = self.noise_floor.clamp(0.001, 0.1);
        }

        let threshold = self.noise_floor * 1.5;
        let gain = if rms < threshold {
            (rms / threshold).powi(2).clamp(0.05, 1.0)
        } else {
            1.0
        };

        samples.iter().map(|&s| s * gain).collect()
    }

    pub fn current_noise_floor(&self) -> f32 {
        self.noise_floor
    }
}

/// Automatic Gain Control (AGC) normalizing audio level with soft limiting.
pub struct AutomaticGainControl {
    target_rms: f32,
    current_gain: f32,
}

impl Default for AutomaticGainControl {
    fn default() -> Self {
        Self::new(0.12) // Target ~ -18 dBFS
    }
}

impl AutomaticGainControl {
    pub fn new(target_rms: f32) -> Self {
        Self {
            target_rms,
            current_gain: 1.0,
        }
    }

    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        if rms > 1e-4 {
            let desired_gain = (self.target_rms / rms).clamp(0.5, 4.0);
            self.current_gain = 0.9 * self.current_gain + 0.1 * desired_gain;
        }

        samples
            .iter()
            .map(|&s| {
                let amplified = s * self.current_gain;
                // Soft-knee tanh limiting preventing harsh digital clipping
                amplified.tanh()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 3. Speech Endpoint & Voice Activity Detector (VAD)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VState {
    Silence,
    SpeechStarting,
    Speaking,
    SpeechTrailing,
}

/// Robust Speech Endpoint Detector with hysteresis, hangover time, and mode switching.
pub struct SpeechEndpointDetector {
    config: SpeechEndpointConfig,
    mode: MicrophoneMode,
    state: VState,
    speech_frame_count: usize,
    silence_frame_count: usize,
    total_frames_in_utterance: usize,
    utterance_start_offset: u64,
    utterance_start_timestamp_ns: u64,
    active_buffer: Vec<f32>,
}

impl SpeechEndpointDetector {
    pub fn new(config: SpeechEndpointConfig, mode: MicrophoneMode) -> Self {
        Self {
            config,
            mode,
            state: VState::Silence,
            speech_frame_count: 0,
            silence_frame_count: 0,
            total_frames_in_utterance: 0,
            utterance_start_offset: 0,
            utterance_start_timestamp_ns: 0,
            active_buffer: Vec::with_capacity(16000 * 5),
        }
    }

    pub fn set_mode(&mut self, mode: MicrophoneMode) {
        self.mode = mode;
        if let MicrophoneMode::PushToTalk { active: false } = mode {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.state = VState::Silence;
        self.speech_frame_count = 0;
        self.silence_frame_count = 0;
        self.total_frames_in_utterance = 0;
        self.active_buffer.clear();
    }

    /// Processes an incoming processed audio chunk (typically 10-20ms).
    pub fn process_chunk(&mut self, chunk: &AudioChunk) -> EndpointEvent {
        // In Push-to-Talk mode with active=false, unconditionally return Silence
        if let MicrophoneMode::PushToTalk { active: false } = self.mode {
            return EndpointEvent::Silence;
        }

        let samples = &chunk.samples;
        let rms = chunk.metrics.rms_db;
        let linear_rms = (10.0f32).powf(rms / 20.0);

        let is_speech_level = linear_rms >= self.config.start_energy_threshold;
        let is_continuation_level = linear_rms >= self.config.stop_energy_threshold;

        match self.state {
            VState::Silence => {
                if is_speech_level {
                    self.speech_frame_count = 1;
                    self.state = VState::SpeechStarting;
                    self.utterance_start_offset = chunk.sample_offset;
                    self.utterance_start_timestamp_ns = chunk.timestamp_ns;
                    self.active_buffer.clear();
                    self.active_buffer.extend_from_slice(samples);
                }
                EndpointEvent::Silence
            }
            VState::SpeechStarting => {
                self.active_buffer.extend_from_slice(samples);
                if is_continuation_level {
                    self.speech_frame_count += 1;
                    if self.speech_frame_count >= self.config.min_speech_frames {
                        self.state = VState::Speaking;
                        self.total_frames_in_utterance = self.speech_frame_count;
                        return EndpointEvent::SpeechStarted {
                            sample_offset: self.utterance_start_offset,
                            timestamp_ns: self.utterance_start_timestamp_ns,
                        };
                    }
                } else {
                    // False start (cough / click): revert back to silence
                    self.state = VState::Silence;
                    self.active_buffer.clear();
                }
                EndpointEvent::Silence
            }
            VState::Speaking => {
                self.active_buffer.extend_from_slice(samples);
                self.total_frames_in_utterance += 1;

                if is_continuation_level {
                    self.silence_frame_count = 0;
                    let dur_ms =
                        (self.active_buffer.len() as f64 / chunk.sample_rate as f64) * 1000.0;
                    EndpointEvent::SpeechContinuing {
                        duration_ms: dur_ms,
                    }
                } else {
                    // Entering hangover period (pause during speech)
                    self.silence_frame_count = 1;
                    self.state = VState::SpeechTrailing;
                    let dur_ms =
                        (self.active_buffer.len() as f64 / chunk.sample_rate as f64) * 1000.0;
                    EndpointEvent::SpeechContinuing {
                        duration_ms: dur_ms,
                    }
                }
            }
            VState::SpeechTrailing => {
                self.active_buffer.extend_from_slice(samples);
                self.total_frames_in_utterance += 1;

                if is_continuation_level {
                    // Speech resumed before hangover expired! Return to Speaking
                    self.state = VState::Speaking;
                    self.silence_frame_count = 0;
                    let dur_ms =
                        (self.active_buffer.len() as f64 / chunk.sample_rate as f64) * 1000.0;
                    EndpointEvent::SpeechContinuing {
                        duration_ms: dur_ms,
                    }
                } else {
                    self.silence_frame_count += 1;
                    let force_stop =
                        self.total_frames_in_utterance >= self.config.max_utterance_frames;

                    if self.silence_frame_count >= self.config.hangover_frames || force_stop {
                        // Utterance complete!
                        let end_offset = chunk.sample_offset + chunk.samples.len() as u64;
                        let speech_data = Arc::from(self.active_buffer.clone());
                        let event = EndpointEvent::SpeechEnded {
                            start_offset: self.utterance_start_offset,
                            end_offset,
                            speech_samples: speech_data,
                            confidence: 0.95,
                        };
                        self.reset();
                        event
                    } else {
                        let dur_ms =
                            (self.active_buffer.len() as f64 / chunk.sample_rate as f64) * 1000.0;
                        EndpointEvent::SpeechContinuing {
                            duration_ms: dur_ms,
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Integrated Production Audio DSP Processor
// ---------------------------------------------------------------------------

/// Complete DSP pipeline combining AEC, Noise Suppression, AGC, and Endpoint Detection.
pub struct AudioDspProcessor {
    aec: AcousticEchoCanceller,
    noise_suppressor: NoiseSuppressor,
    agc: AutomaticGainControl,
    endpoint_detector: SpeechEndpointDetector,
}

impl AudioDspProcessor {
    pub fn new(
        endpoint_config: SpeechEndpointConfig,
        mode: MicrophoneMode,
        filter_length: usize,
        delay_samples: usize,
    ) -> Self {
        Self {
            aec: AcousticEchoCanceller::new(filter_length, 0.08, delay_samples),
            noise_suppressor: NoiseSuppressor::new(),
            agc: AutomaticGainControl::default(),
            endpoint_detector: SpeechEndpointDetector::new(endpoint_config, mode),
        }
    }

    pub fn set_microphone_mode(&mut self, mode: MicrophoneMode) {
        self.endpoint_detector.set_mode(mode);
    }

    /// Ingests a raw microphone chunk and speaker playback reference,
    /// performs AEC -> Noise Suppression -> AGC -> Endpoint Detection.
    /// Returns `(ProcessedAudioChunk, EndpointEvent, erle_db)`.
    pub fn process_input(
        &mut self,
        mic_chunk: &AudioChunk,
        playback_ref: &[f32],
    ) -> (AudioChunk, EndpointEvent, f32) {
        // 1. Acoustic Echo Cancellation
        let (echo_cancelled, erle_db) = self.aec.process(&mic_chunk.samples, playback_ref);

        // 2. Stationary Noise Suppression
        let noise_suppressed = self.noise_suppressor.process(&echo_cancelled);

        // 3. Level Normalization (AGC)
        let normalized = self.agc.process(&noise_suppressed);

        let processed_chunk = AudioChunk::new(
            normalized,
            mic_chunk.sample_rate,
            mic_chunk.channels,
            mic_chunk.sample_offset,
            mic_chunk.timestamp_ns,
        );

        // 4. Speech Activity & Endpoint Detection
        let event = self.endpoint_detector.process_chunk(&processed_chunk);

        (processed_chunk, event, erle_db)
    }
}
