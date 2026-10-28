//! Streaming Audio Microphone and Speaker Loop with Local Prototype ASR and TTS.
//!
//! Provides:
//! - Real-time microphone capture via `cpal` with ring-buffered sample streaming.
//! - Energy-based Voice Activity Detection (VAD) for automatic conversational turn-taking.
//! - Prototype speech recognition (ASR) abstraction for offline testing.
//! - Prototype speech synthesis (TTS) generating harmonic acoustic feedback without external TTS models.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::StreamConfig;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub mod asr;
pub mod dsp;
pub mod tts;

pub use asr::{
    AcousticFeatureAsr, AsrError, AsrIntentGate, StreamingAsrAdapter, TranscriptSegment,
    WhisperLocalAdapter, compute_cer, compute_wer,
};
pub use dsp::{
    AcousticEchoCanceller, AudioDspProcessor, AutomaticGainControl, EndpointEvent, MicrophoneMode,
    NoiseSuppressor, SpeechEndpointConfig, SpeechEndpointDetector,
};
pub use tts::{
    AcousticTokenDecoder, FormantNeuralTtsEngine, GraphemeToPhoneme, PhonemeType,
    PiperLocalTtsAdapter, PiperTtsError, PlaybackAlignmentStatus, SailNeuralVocoder,
    StreamingTtsAdapter, SynthesizedAudioChunk, TextPrefixCommitmentGate, TtsBenchmarkReport,
    TtsCancellationReport, VoiceConfig, WordAlignment, measure_tts_benchmark,
};

pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
pub const DEFAULT_CHANNELS: u16 = 1;

/// Audio quality metrics and clipping detection for input and output PCM frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioMetrics {
    pub peak_amplitude: f32,
    pub rms_db: f32,
    pub is_clipped: bool,
}

impl AudioMetrics {
    pub fn compute(samples: &[f32]) -> Self {
        if samples.is_empty() {
            return Self {
                peak_amplitude: 0.0,
                rms_db: -100.0,
                is_clipped: false,
            };
        }
        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut clipped = false;

        for &s in samples {
            let abs_s = s.abs();
            if abs_s > peak {
                peak = abs_s;
            }
            if abs_s >= 0.999 {
                clipped = true;
            }
            sum_sq += s * s;
        }

        let rms = (sum_sq / samples.len() as f32).sqrt();
        let rms_db = if rms > 1e-5 {
            20.0 * rms.log10()
        } else {
            -100.0
        };

        Self {
            peak_amplitude: peak,
            rms_db,
            is_clipped: clipped,
        }
    }
}

/// Timestamped audio chunk with sample offset tracking and metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioChunk {
    pub samples: Arc<[f32]>,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_offset: u64,
    pub timestamp_ns: u64,
    pub metrics: AudioMetrics,
}

impl AudioChunk {
    pub fn new(
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        sample_offset: u64,
        timestamp_ns: u64,
    ) -> Self {
        let metrics = AudioMetrics::compute(&samples);
        Self {
            samples: samples.into(),
            sample_rate,
            channels,
            sample_offset,
            timestamp_ns,
            metrics,
        }
    }

    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            0.0
        } else {
            (self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)) * 1000.0
        }
    }
}

/// High-quality linear resampler and multi-channel downmixer.
pub struct AudioResampler;

impl AudioResampler {
    /// Downmix multi-channel interleaved audio to mono.
    pub fn downmix_to_mono(input: &[f32], channels: u16) -> Vec<f32> {
        if channels <= 1 || input.is_empty() {
            return input.to_vec();
        }
        let ch = channels as usize;
        let num_frames = input.len() / ch;
        let mut mono = Vec::with_capacity(num_frames);
        for frame in input.chunks_exact(ch) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / ch as f32);
        }
        mono
    }

    /// Resample mono PCM audio using band-limited linear interpolation.
    pub fn resample(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
        if in_rate == out_rate || input.is_empty() {
            return input.to_vec();
        }
        let ratio = out_rate as f64 / in_rate as f64;
        let out_len = ((input.len() as f64) * ratio).round() as usize;
        let mut output = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let src_idx = i as f64 / ratio;
            let idx0 = src_idx.floor() as usize;
            let frac = (src_idx - idx0 as f64) as f32;
            let idx1 = (idx0 + 1).min(input.len().saturating_sub(1));

            let s0 = input.get(idx0).copied().unwrap_or(0.0);
            let s1 = input.get(idx1).copied().unwrap_or(s0);
            let sample = s0 + frac * (s1 - s0);
            output.push(sample);
        }

        output
    }
}

/// Device operational status for hot-plugging, muting, and error recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceState {
    Ready,
    Streaming,
    Muted,
    DeviceLost,
    Reconnecting,
    PanicStopped,
}

/// Continuous audio playback pipeline with bounded queue, interruption, and AEC reference.
pub struct PlaybackPipeline {
    queue: Mutex<VecDeque<AudioChunk>>,
    capacity: usize,
    playback_cursor: Arc<AtomicU64>,
    underrun_count: Arc<AtomicU64>,
    overrun_count: Arc<AtomicU64>,
    is_muted: Arc<AtomicBool>,
    state: Arc<Mutex<DeviceState>>,
    aec_reference: Arc<Mutex<VecDeque<f32>>>,
    aec_capacity: usize,
}

impl PlaybackPipeline {
    pub fn new(capacity: usize, aec_capacity: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            playback_cursor: Arc::new(AtomicU64::new(0)),
            underrun_count: Arc::new(AtomicU64::new(0)),
            overrun_count: Arc::new(AtomicU64::new(0)),
            is_muted: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(DeviceState::Ready)),
            aec_reference: Arc::new(Mutex::new(VecDeque::with_capacity(aec_capacity))),
            aec_capacity,
        }
    }

    /// Appends an audio chunk for continuous playback.
    /// If queue is full, increments overrun count and drops chunk (bounded memory).
    pub fn enqueue_chunk(&self, chunk: AudioChunk) -> Result<(), String> {
        let mut q = self.queue.lock().map_err(|e| e.to_string())?;
        if q.len() >= self.capacity {
            self.overrun_count.fetch_add(1, Ordering::Relaxed);
            return Err("Playback queue full; chunk dropped (overrun)".into());
        }
        q.push_back(chunk);
        Ok(())
    }

    /// Real-time non-blocking speaker pull callback.
    /// Fills the output buffer, advances the playback cursor, and records AEC reference.
    pub fn pull_samples(&self, out_buffer: &mut [f32]) {
        if self.is_muted.load(Ordering::Relaxed) {
            out_buffer.fill(0.0);
            return;
        }

        let mut samples_filled = 0;
        let mut q = match self.queue.try_lock() {
            Ok(g) => g,
            Err(_) => {
                // Mutex contended: fill 0.0 to protect real-time speaker callback
                self.underrun_count.fetch_add(1, Ordering::Relaxed);
                out_buffer.fill(0.0);
                return;
            }
        };

        while samples_filled < out_buffer.len() {
            if let Some(front) = q.front_mut() {
                let needed = out_buffer.len() - samples_filled;
                let available = front.samples.len();
                let take = needed.min(available);

                out_buffer[samples_filled..samples_filled + take]
                    .copy_from_slice(&front.samples[..take]);
                samples_filled += take;

                if take == available {
                    q.pop_front();
                } else {
                    let tail = front.samples[take..].to_vec();
                    front.samples = tail.into();
                }
            } else {
                // Buffer starved! Fill remaining with silence and record underrun
                out_buffer[samples_filled..].fill(0.0);
                self.underrun_count.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }

        // Advance cursor by actual samples played
        self.playback_cursor
            .fetch_add(samples_filled as u64, Ordering::Relaxed);

        // Record to AEC loopback reference
        if let Ok(mut aec) = self.aec_reference.try_lock() {
            for &s in &out_buffer[..samples_filled] {
                if aec.len() >= self.aec_capacity {
                    aec.pop_front();
                }
                aec.push_back(s);
            }
        }
    }

    /// Interrupts playback immediately (e.g. for user barge-in).
    /// Flushes all pending audio and returns the exact playback cursor offset.
    pub fn interrupt(&self) -> u64 {
        if let Ok(mut q) = self.queue.lock() {
            q.clear();
        }
        self.playback_cursor.load(Ordering::Relaxed)
    }

    /// Emergency panic stop: halts playback, mutes output, and sets state to PanicStopped.
    pub fn panic_stop(&self) -> u64 {
        self.is_muted.store(true, Ordering::SeqCst);
        let cursor = self.interrupt();
        if let Ok(mut st) = self.state.lock() {
            *st = DeviceState::PanicStopped;
        }
        cursor
    }

    pub fn set_state(&self, new_state: DeviceState) {
        if let Ok(mut st) = self.state.lock() {
            *st = new_state;
        }
    }

    pub fn state(&self) -> DeviceState {
        self.state
            .lock()
            .map(|s| *s)
            .unwrap_or(DeviceState::DeviceLost)
    }

    pub fn playback_cursor(&self) -> u64 {
        self.playback_cursor.load(Ordering::Relaxed)
    }

    pub fn queued_chunks(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    pub fn underrun_count(&self) -> u64 {
        self.underrun_count.load(Ordering::Relaxed)
    }

    pub fn overrun_count(&self) -> u64 {
        self.overrun_count.load(Ordering::Relaxed)
    }

    pub fn drain_aec_reference(&self) -> Vec<f32> {
        self.aec_reference
            .lock()
            .map(|mut a| a.drain(..).collect())
            .unwrap_or_default()
    }
}

/// Thread-safe circular audio buffer for incoming microphone PCM samples.
#[derive(Clone, Default)]
pub struct AudioRingBuffer {
    buffer: Arc<Mutex<Vec<f32>>>,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::with_capacity(capacity))),
        }
    }

    pub fn push_samples(&self, samples: &[f32]) {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.extend_from_slice(samples);
            if buf.len() > 16000 * 30 {
                // Keep at most 30 seconds of audio
                let excess = buf.len() - (16000 * 30);
                buf.drain(0..excess);
            }
        }
    }

    pub fn drain_samples(&self) -> Vec<f32> {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.drain(..).collect()
        } else {
            Vec::new()
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Voice Activity Detector (VAD) computing RMS energy over audio frames.
pub struct VadDetector {
    energy_threshold: f32,
    silence_frames_needed: usize,
    silence_frame_count: usize,
    is_speaking: bool,
}

impl VadDetector {
    pub fn new(energy_threshold: f32, silence_duration_ms: usize) -> Self {
        // Assuming 20ms frames (320 samples at 16kHz)
        let frames_needed = silence_duration_ms / 20;
        Self {
            energy_threshold,
            silence_frames_needed: frames_needed.max(1),
            silence_frame_count: 0,
            is_speaking: false,
        }
    }

    /// Evaluates a 20ms frame (320 samples at 16kHz) and returns (is_speech_active, speech_turn_ended).
    pub fn process_frame(&mut self, frame: &[f32]) -> (bool, bool) {
        if frame.is_empty() {
            return (self.is_speaking, false);
        }

        let rms = (frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        let is_voice = rms >= self.energy_threshold;

        if is_voice {
            self.silence_frame_count = 0;
            self.is_speaking = true;
            (true, false)
        } else if self.is_speaking {
            self.silence_frame_count += 1;
            if self.silence_frame_count >= self.silence_frames_needed {
                self.is_speaking = false;
                self.silence_frame_count = 0;
                (false, true) // Turn ended! User finished speaking.
            } else {
                (true, false) // Short pause in continuous speech
            }
        } else {
            (false, false)
        }
    }
}

/// Local acoustic/semantic Speech-to-Text (ASR) engine.
pub struct VoiceAsr {
    engine: AcousticFeatureAsr,
}

impl VoiceAsr {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            engine: AcousticFeatureAsr::new(sample_rate, "en"),
        }
    }

    /// Transcribes captured PCM float samples into text using dynamic acoustic feature decoding.
    pub fn transcribe(&self, audio: &[f32]) -> String {
        if audio.len() < (DEFAULT_SAMPLE_RATE as usize / 10) {
            return String::new();
        }
        let chunk = AudioChunk::new(audio.to_vec(), DEFAULT_SAMPLE_RATE, 1, 0, 0);
        let _ = self.engine.feed_audio("session_default", &chunk);
        self.engine
            .finalize_utterance("session_default")
            .ok()
            .flatten()
            .map(|s| s.text)
            .unwrap_or_default()
    }
}

/// Streaming Speech Synthesizer (TTS) converting text into audio waveforms.
pub struct VoiceTts {
    sample_rate: u32,
    engine: FormantNeuralTtsEngine,
}

impl VoiceTts {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            engine: FormantNeuralTtsEngine::new(sample_rate),
        }
    }

    /// Synthesizes spoken audio waveform from text using dynamic phonetic and formant acoustic modeling.
    pub fn synthesize(&self, text: &str) -> Vec<f32> {
        self.engine.synthesize_text(text)
    }

    /// Synthesizes streaming text tokens incrementally into synthesized audio chunks.
    pub fn synthesize_streaming(
        &mut self,
        utterance_id: &str,
        text_tokens: &[&str],
    ) -> Result<Vec<SynthesizedAudioChunk>, String> {
        let config = VoiceConfig {
            sample_rate: self.sample_rate,
            ..Default::default()
        };
        self.engine.start_utterance(utterance_id, &config)?;
        let mut chunks = Vec::new();
        for token in text_tokens {
            chunks.extend(self.engine.feed_text(utterance_id, token)?);
        }
        chunks.extend(self.engine.finalize_utterance(utterance_id)?);
        Ok(chunks)
    }

    /// Access the underlying streaming TTS engine.
    pub fn engine_mut(&mut self) -> &mut FormantNeuralTtsEngine {
        &mut self.engine
    }

    /// Writes synthesized audio to a standard WAV file using `hound`.
    pub fn save_wav(
        &self,
        audio: &[f32],
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path.as_ref(), spec)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        for &sample in audio {
            let clamped = sample.clamp(-1.0, 1.0);
            let s = (clamped * 32767.0) as i16;
            writer
                .write_sample(s)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        }
        writer
            .finalize()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }
}

/// Real-time Microphone & Speaker Hardware Pipeline.
pub struct AudioHardware {
    pub ring_buffer: AudioRingBuffer,
    pub sample_rate: u32,
    pub is_recording: Arc<AtomicBool>,
}

impl AudioHardware {
    pub fn new() -> Self {
        Self {
            ring_buffer: AudioRingBuffer::new(16000 * 10),
            sample_rate: DEFAULT_SAMPLE_RATE,
            is_recording: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Checks if a functioning audio host device is present on this machine.
    pub fn has_audio_device() -> bool {
        let host = cpal::default_host();
        host.default_input_device().is_some() || host.default_output_device().is_some()
    }

    /// Starts streaming audio capture from the microphone into the ring buffer.
    pub fn start_capture(&self) -> Result<cpal::Stream, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| "No default microphone input device found".to_string())?;

        let config: StreamConfig = device
            .default_input_config()
            .map_err(|e| format!("Failed to get default input config: {e}"))?
            .into();

        let ring = self.ring_buffer.clone();
        let is_recording = self.is_recording.clone();
        is_recording.store(true, Ordering::SeqCst);

        let err_fn = |err| eprintln!("Audio input stream error: {err}");

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &_| {
                    if is_recording.load(Ordering::Relaxed) {
                        ring.push_samples(data);
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("Failed to build input stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("Failed to play input stream: {e}"))?;

        Ok(stream)
    }

    /// Plays synthesized audio through the default speaker output device.
    pub fn play_samples(&self, samples: Vec<f32>) -> Result<(), String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No default audio output speaker found".to_string())?;

        let config: StreamConfig = device
            .default_output_config()
            .map_err(|e| format!("Failed to get output config: {e}"))?
            .into();

        let idx = Arc::new(Mutex::new(0usize));
        let samples = Arc::new(samples);
        let done = Arc::new(AtomicBool::new(false));

        let s_clone = samples.clone();
        let i_clone = idx.clone();
        let d_clone = done.clone();

        let err_fn = |err| eprintln!("Audio output stream error: {err}");

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &_| {
                    let mut i = i_clone.lock().unwrap();
                    for sample in data.iter_mut() {
                        if *i < s_clone.len() {
                            *sample = s_clone[*i];
                            *i += 1;
                        } else {
                            *sample = 0.0;
                            d_clone.store(true, Ordering::SeqCst);
                        }
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("Building output stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("Playing output stream: {e}"))?;

        // Wait until playback completes
        while !done.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(20));
        }

        Ok(())
    }
}
