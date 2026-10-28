//! Integration and Verification Tests for Production Audio Capture and Continuous Playback.
//!
//! Verifies acceptance criteria for GitHub Issue #5:
//! 1. Multi-rate audio resampling, stereo-to-mono downmixing, and duration/timestamp accuracy.
//! 2. Bounded playback buffer under overload, zero memory explosion, and overrun tracking.
//! 3. Continuous playback callback with accurate sample cursor tracking and underrun recording.
//! 4. User interruption (barge-in) and panic stop immediately flushing pending playback and reporting exact sample offset.
//! 5. AEC loopback reference ring buffer capture.
//! 6. Device state management and recovery.

use sail::voice::{AudioChunk, AudioMetrics, AudioResampler, DeviceState, PlaybackPipeline};

/// Test 1: Stereo-to-mono downmixing, multi-rate resampling, and timestamp calculation.
#[test]
fn test_audio_resampler_and_downmixing() {
    // 1. Stereo to mono downmixing
    let stereo_samples = vec![
        1.0, 0.0, // frame 0: avg 0.5
        0.5, 0.5, // frame 1: avg 0.5
        -0.8, 0.2, // frame 2: avg -0.3
    ];
    let mono = AudioResampler::downmix_to_mono(&stereo_samples, 2);
    assert_eq!(mono.len(), 3);
    assert!((mono[0] - 0.5).abs() < 1e-5);
    assert!((mono[1] - 0.5).abs() < 1e-5);
    assert!((mono[2] - (-0.3)).abs() < 1e-5);

    // 2. Resampling from 48kHz (native mic/soundcard) to 16kHz (encoder/model)
    // 480 samples at 48kHz = 10ms
    let input_48k: Vec<f32> = (0..480).map(|i| i as f32 / 480.0).collect();
    let resampled_16k = AudioResampler::resample(&input_48k, 48000, 16000);
    // At 16kHz, 10ms = 160 samples
    assert_eq!(resampled_16k.len(), 160);
    // Boundary fidelity check
    assert!((resampled_16k[0] - input_48k[0]).abs() < 1e-4);
    assert!((resampled_16k.last().unwrap() - input_48k.last().unwrap()).abs() < 0.02);

    // 3. Resampling up from 16kHz to 48kHz
    let upsampled_48k = AudioResampler::resample(&resampled_16k, 16000, 48000);
    assert_eq!(upsampled_48k.len(), 480);

    // 4. Empty and identity resampling
    assert!(AudioResampler::resample(&[], 16000, 48000).is_empty());
    let identity = AudioResampler::resample(&input_48k, 48000, 48000);
    assert_eq!(identity.len(), input_48k.len());
}

/// Test 2: Audio metrics, peak amplitude, RMS dB, and clipping detection.
#[test]
fn test_audio_metrics_and_clipping() {
    let silent = vec![0.0f32; 100];
    let m_silent = AudioMetrics::compute(&silent);
    assert_eq!(m_silent.peak_amplitude, 0.0);
    assert!(m_silent.rms_db <= -99.0);
    assert!(!m_silent.is_clipped);

    let normal_speech = vec![0.3f32; 100];
    let m_speech = AudioMetrics::compute(&normal_speech);
    assert!((m_speech.peak_amplitude - 0.3).abs() < 1e-4);
    assert!(!m_speech.is_clipped);

    // Severely clipped audio reaching 1.0
    let clipped = vec![0.2f32, 0.8, 1.0, 0.5];
    let m_clipped = AudioMetrics::compute(&clipped);
    assert!(m_clipped.is_clipped);
    assert_eq!(m_clipped.peak_amplitude, 1.0);
}

/// Test 3: Continuous playback pipeline, cursor advancement, and underrun tracking.
#[test]
fn test_continuous_playback_pipeline_and_cursor() {
    // 5 chunks capacity, 1024 samples AEC buffer
    let pipeline = PlaybackPipeline::new(5, 1024);
    assert_eq!(pipeline.playback_cursor(), 0);
    assert_eq!(pipeline.underrun_count(), 0);

    // Enqueue 2 chunks: 200 samples each (total 400 samples)
    let chunk1 = AudioChunk::new(vec![0.1; 200], 16000, 1, 0, 0);
    let chunk2 = AudioChunk::new(vec![0.2; 200], 16000, 1, 200, 12_500_000);
    pipeline.enqueue_chunk(chunk1).unwrap();
    pipeline.enqueue_chunk(chunk2).unwrap();

    // Pull 150 samples (from chunk1)
    let mut out_buf = vec![0.0f32; 150];
    pipeline.pull_samples(&mut out_buf);
    assert_eq!(pipeline.playback_cursor(), 150);
    assert_eq!(out_buf[0], 0.1);
    assert_eq!(out_buf[149], 0.1);
    assert_eq!(pipeline.underrun_count(), 0);

    // Pull next 200 samples (50 from chunk1, 150 from chunk2)
    let mut out_buf2 = vec![0.0f32; 200];
    pipeline.pull_samples(&mut out_buf2);
    assert_eq!(pipeline.playback_cursor(), 350);
    assert_eq!(out_buf2[0], 0.1);
    assert_eq!(out_buf2[50], 0.2);
    assert_eq!(pipeline.underrun_count(), 0);

    // Pull 100 samples (50 from chunk2, 50 starved/underrun)
    let mut out_buf3 = vec![0.0f32; 100];
    pipeline.pull_samples(&mut out_buf3);
    assert_eq!(pipeline.playback_cursor(), 400); // Only 50 real samples played
    assert_eq!(out_buf3[0], 0.2);
    assert_eq!(out_buf3[49], 0.2);
    assert_eq!(out_buf3[50], 0.0); // silence filled
    assert_eq!(pipeline.underrun_count(), 1); // 1 underrun recorded

    // Check AEC loopback reference
    let aec_ref = pipeline.drain_aec_reference();
    assert_eq!(aec_ref.len(), 400);
}

/// Test 4: Simulated overload and bounded buffer overflow (overrun tracking).
#[test]
fn test_simulated_overload_and_overrun_tracking() {
    let pipeline = PlaybackPipeline::new(3, 512);

    // Enqueue 3 chunks to fill capacity
    for i in 1..=3 {
        let chunk = AudioChunk::new(vec![0.1; 160], 16000, 1, (i - 1) * 160, i * 10_000_000);
        assert!(pipeline.enqueue_chunk(chunk).is_ok());
    }

    // 4th and 5th chunks must be dropped to protect memory bounds
    let overflow1 = AudioChunk::new(vec![0.1; 160], 16000, 1, 480, 40_000_000);
    let overflow2 = AudioChunk::new(vec![0.1; 160], 16000, 1, 640, 50_000_000);

    assert!(pipeline.enqueue_chunk(overflow1).is_err());
    assert!(pipeline.enqueue_chunk(overflow2).is_err());

    assert_eq!(pipeline.overrun_count(), 2);
}

/// Test 5: User interruption (barge-in) and panic stop immediately flushes audio and returns exact sample cursor.
#[test]
fn test_interruption_and_panic_stop() {
    let pipeline = PlaybackPipeline::new(10, 512);

    for i in 1..=5 {
        let chunk = AudioChunk::new(vec![0.05; 160], 16000, 1, (i - 1) * 160, 0);
        pipeline.enqueue_chunk(chunk).unwrap();
    }

    // Play 160 samples (chunk 1)
    let mut speaker_buf = vec![0.0f32; 160];
    pipeline.pull_samples(&mut speaker_buf);
    assert_eq!(pipeline.playback_cursor(), 160);

    // User speaks / barge-in happens!
    let stopped_offset = pipeline.interrupt();
    assert_eq!(stopped_offset, 160);

    // Verify all remaining queued audio was discarded
    let mut next_buf = vec![99.0f32; 160];
    pipeline.pull_samples(&mut next_buf);
    assert_eq!(next_buf[0], 0.0); // silence, no old audio played
    assert_eq!(pipeline.playback_cursor(), 160); // cursor stayed at 160

    // Panic stop sets state to PanicStopped
    pipeline.panic_stop();
    assert_eq!(pipeline.state(), DeviceState::PanicStopped);
}
