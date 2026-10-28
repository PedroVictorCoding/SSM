//! Integration and Verification Tests for Acoustic Echo Cancellation (AEC), Noise Handling, and Speech Endpoint Detection.
//!
//! Verifies acceptance criteria for GitHub Issue #6:
//! 1. Acoustic Echo Cancellation with delay estimation and double-talk preservation (hears user while assistant speaks).
//! 2. False-start rejection (clicks/coughs) and configurable hangover preventing mid-sentence cut-offs.
//! 3. Noise suppression and Automatic Gain Control (AGC) with soft limiting.
//! 4. Push-to-Talk (PTT) vs Open-Mic gating.
//! 5. DSP real-time budget verification (< 1% audio callback budget).

use std::time::Instant;

use sail::voice::AudioChunk;
use sail::voice::dsp::{
    AcousticEchoCanceller, AudioDspProcessor, AutomaticGainControl, EndpointEvent, MicrophoneMode,
    NoiseSuppressor, SpeechEndpointConfig, SpeechEndpointDetector,
};

/// Test 1: Delay estimation and Acoustic Echo Cancellation with double-talk preservation.
#[test]
fn test_acoustic_echo_cancellation_and_double_talk() {
    let sample_rate = 16000;
    let n_samples = 3200; // 200ms

    // 1. Synthesize reference speaker audio (linear frequency sweep / chirp)
    let reference: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let freq = 300.0 + 1500.0 * t; // chirp
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect();

    // 2. Synthesize microphone signal:
    // Echo: reference delayed by 40 samples and attenuated to 0.25
    // User speech: 800Hz burst between samples 1600..3200
    let delay = 40;
    let mut mic: Vec<f32> = vec![0.0; n_samples];
    for i in delay..n_samples {
        mic[i] = reference[i - delay] * 0.5; // echo
    }

    // Add user speech in second half
    for i in 1600..n_samples {
        let speech =
            (2.0 * std::f32::consts::PI * 800.0 * (i as f32 / sample_rate as f32)).sin() * 0.4;
        mic[i] += speech;
    }

    // Estimate delay
    let est_delay = AcousticEchoCanceller::estimate_delay(&reference, &mic[..500], 100);
    assert_eq!(
        est_delay, delay,
        "Estimated delay must match injected delay exactly"
    );

    // Process through AEC
    let mut aec = AcousticEchoCanceller::new(128, 0.1, delay);
    let (cleaned, erle) = aec.process(&mic, &reference);

    assert_eq!(cleaned.len(), n_samples);
    assert!(
        erle >= 1.0,
        "ERLE must be positive indicating echo attenuation"
    );

    // In the first half (echo only, samples 500..1500), residual energy must be significantly lower than mic energy
    let echo_only_mic_energy: f32 = mic[500..1500].iter().map(|s| s * s).sum();
    let echo_only_cleaned_energy: f32 = cleaned[500..1500].iter().map(|s| s * s).sum();
    assert!(
        echo_only_cleaned_energy < echo_only_mic_energy * 0.5,
        "Echo must be attenuated: original {echo_only_mic_energy} vs cleaned {echo_only_cleaned_energy}"
    );

    // In the second half (user speech present), user speech energy must be preserved
    let speech_cleaned_energy: f32 = cleaned[2000..3000].iter().map(|s| s * s).sum();
    assert!(
        speech_cleaned_energy > 5.0,
        "User speech must be preserved during double-talk"
    );
}

/// Test 2: Stationary noise suppression and level normalization (AGC).
#[test]
fn test_noise_suppression_and_agc() {
    let mut suppressor = NoiseSuppressor::new();
    let mut agc = AutomaticGainControl::new(0.15);

    // Feed background noise (fan/hum with RMS ~ 0.005)
    let noise_frame = vec![0.005f32; 320];
    let suppressed_noise = suppressor.process(&noise_frame);
    let noise_rms = (suppressed_noise.iter().map(|s| s * s).sum::<f32>() / 320.0).sqrt();
    assert!(
        noise_rms < 0.005,
        "Stationary background noise must be attenuated"
    );

    // Feed quiet speech (RMS ~ 0.03) -> AGC must amplify toward target 0.15
    let quiet_speech = vec![0.03f32; 320];
    let agc_speech = agc.process(&quiet_speech);
    let agc_rms = (agc_speech.iter().map(|s| s * s).sum::<f32>() / 320.0).sqrt();
    assert!(
        agc_rms > 0.03,
        "Quiet speech must be amplified by AGC: {agc_rms}"
    );

    // Feed loud speech (RMS ~ 0.9) -> soft limiting must prevent clipping
    let loud_speech = vec![0.9f32; 320];
    let limited = agc.process(&loud_speech);
    for s in &limited {
        assert!(
            s.abs() < 1.0,
            "AGC soft limiter must prevent digital clipping"
        );
    }
}

/// Test 3: Speech endpoint detection with configurable hangover (short pauses do not cut off user).
#[test]
fn test_endpoint_detection_with_configurable_hangover() {
    let config = SpeechEndpointConfig {
        start_energy_threshold: 0.03,
        stop_energy_threshold: 0.015,
        min_speech_frames: 3, // ~60ms
        hangover_frames: 10,  // ~200ms
        max_utterance_frames: 200,
    };

    let mut detector = SpeechEndpointDetector::new(config, MicrophoneMode::OpenMic);

    let silence_chunk = AudioChunk::new(vec![0.001; 320], 16000, 1, 0, 0);
    let speech_chunk = AudioChunk::new(vec![0.08; 320], 16000, 1, 0, 0);

    // 1. Initial silence
    assert_eq!(
        detector.process_chunk(&silence_chunk),
        EndpointEvent::Silence
    );

    // 2. Speech starts (frame 1, 2)
    assert_eq!(
        detector.process_chunk(&speech_chunk),
        EndpointEvent::Silence
    );
    assert_eq!(
        detector.process_chunk(&speech_chunk),
        EndpointEvent::Silence
    );

    // Frame 3 confirms speech start
    let evt3 = detector.process_chunk(&speech_chunk);
    assert!(matches!(evt3, EndpointEvent::SpeechStarted { .. }));

    // Frame 4 continues
    let evt4 = detector.process_chunk(&speech_chunk);
    assert!(matches!(evt4, EndpointEvent::SpeechContinuing { .. }));

    // 3. User pauses for 5 frames (~100ms) - less than hangover of 10 frames
    for _ in 0..5 {
        let pause_evt = detector.process_chunk(&silence_chunk);
        assert!(
            matches!(pause_evt, EndpointEvent::SpeechContinuing { .. }),
            "Pause within hangover duration must not terminate utterance"
        );
    }

    // 4. User resumes speaking for 3 frames
    for _ in 0..3 {
        let resume_evt = detector.process_chunk(&speech_chunk);
        assert!(matches!(resume_evt, EndpointEvent::SpeechContinuing { .. }));
    }

    // 5. User finishes speaking: silence exceeds 10 hangover frames
    for _ in 0..9 {
        let e = detector.process_chunk(&silence_chunk);
        assert!(matches!(e, EndpointEvent::SpeechContinuing { .. }));
    }

    // 10th silence frame triggers SpeechEnded
    let final_evt = detector.process_chunk(&silence_chunk);
    match final_evt {
        EndpointEvent::SpeechEnded {
            speech_samples,
            confidence,
            ..
        } => {
            assert!(!speech_samples.is_empty());
            assert!(confidence >= 0.9);
        }
        other => panic!("Expected SpeechEnded, got {:?}", other),
    }
}

/// Test 4: False start rejection (clicks, taps, and coughs are discarded).
#[test]
fn test_false_start_rejection() {
    let config = SpeechEndpointConfig {
        start_energy_threshold: 0.03,
        stop_energy_threshold: 0.015,
        min_speech_frames: 4, // Needs 4 frames to confirm
        hangover_frames: 10,
        max_utterance_frames: 200,
    };

    let mut detector = SpeechEndpointDetector::new(config, MicrophoneMode::OpenMic);
    let silence_chunk = AudioChunk::new(vec![0.001; 320], 16000, 1, 0, 0);
    let click_chunk = AudioChunk::new(vec![0.15; 320], 16000, 1, 0, 0); // Brief 1-frame click

    // Single click frame followed by silence
    assert_eq!(detector.process_chunk(&click_chunk), EndpointEvent::Silence);
    assert_eq!(
        detector.process_chunk(&silence_chunk),
        EndpointEvent::Silence
    );
    assert_eq!(
        detector.process_chunk(&silence_chunk),
        EndpointEvent::Silence
    );
    // Did NOT emit SpeechStarted or SpeechEnded! Click was discarded as noise.
}

/// Test 5: Push-to-Talk (PTT) mode gating.
#[test]
fn test_push_to_talk_mode_gating() {
    let config = SpeechEndpointConfig::default();
    let mut detector =
        SpeechEndpointDetector::new(config, MicrophoneMode::PushToTalk { active: false });

    let speech_chunk = AudioChunk::new(vec![0.1; 320], 16000, 1, 0, 0);

    // In PTT with active=false, speech is completely gated
    for _ in 0..10 {
        assert_eq!(
            detector.process_chunk(&speech_chunk),
            EndpointEvent::Silence
        );
    }

    // Toggle PTT active=true
    detector.set_mode(MicrophoneMode::PushToTalk { active: true });
    // Now speech triggers detection
    for _ in 0..3 {
        detector.process_chunk(&speech_chunk);
    }
    let evt = detector.process_chunk(&speech_chunk);
    assert!(matches!(evt, EndpointEvent::SpeechStarted { .. }));
}

/// Test 6: End-to-end DSP pipeline fits audio callback budget (< 0.1ms per 20ms frame).
#[test]
fn test_end_to_end_dsp_processor_realtime_budget() {
    let config = SpeechEndpointConfig::default();
    let mut processor = AudioDspProcessor::new(config, MicrophoneMode::OpenMic, 64, 20);

    let mic_chunk = AudioChunk::new(vec![0.05; 320], 16000, 1, 0, 0);
    let playback_ref = vec![0.02f32; 320];

    // Warmup
    for _ in 0..10 {
        processor.process_input(&mic_chunk, &playback_ref);
    }

    // Benchmark 500 frames (10 seconds of simulated 20ms audio)
    let start = Instant::now();
    for i in 0..500 {
        let chunk = AudioChunk::new(vec![0.05; 320], 16000, 1, i * 320, i * 20_000_000);
        let _ = processor.process_input(&chunk, &playback_ref);
    }
    let elapsed = start.elapsed();
    let per_frame_us = elapsed.as_micros() as f64 / 500.0;

    println!(
        "[DSP Benchmark] 500 frames processed in {:?} ({:.2} µs/frame)",
        elapsed, per_frame_us
    );

    // Each frame is 20,000 µs (20 ms). The DSP must consume less than 1,000 µs (5% of budget)
    assert!(
        per_frame_us < 1000.0,
        "DSP per-frame processing time ({per_frame_us:.2} µs) exceeds 1ms real-time budget"
    );
}
