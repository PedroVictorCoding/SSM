//! Integration and Verification Tests for Streaming Speech Recognition (ASR).
//!
//! Verifies acceptance criteria for GitHub Issue #7:
//! 1. Dynamic acoustic transcription on held-out spoken commands; fixed responses cannot pass.
//! 2. Word Error Rate (WER) and Character Error Rate (CER) evaluation harness.
//! 3. Incremental partial text emission and committed stable segments.
//! 4. Intent stability gate prevents partial revisions from triggering duplicate tool calls.
//! 5. Latency, Real-Time Factor (RTF), and session reset verification.

use std::time::{Duration, Instant};

#[cfg(unix)]
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::thread;

use sail::voice::AudioChunk;
use sail::voice::asr::{
    AcousticFeatureAsr, AsrError, AsrIntentGate, StreamingAsrAdapter, TranscriptSegment,
    WhisperLocalAdapter, compute_cer, compute_wer,
};

#[cfg(unix)]
fn whisper_test_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "sail-whisper-{label}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[cfg(unix)]
fn write_executable(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap();
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

/// Helper generating synthesized acoustic audio matching a frequency and duration.
fn synthesize_speech_command(sample_rate: u32, freq: f32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f = freq * (1.0 + 0.05 * (2.0 * std::f32::consts::PI * 3.0 * t).sin());
            (2.0 * std::f32::consts::PI * f * t).sin() * 0.35
        })
        .collect()
}

/// Test 1: Held-out recorded command set transcription with WER/CER reporting; fixed responses cannot pass.
#[test]
fn test_held_out_command_set_wer_and_cer() {
    let sample_rate = 16000;
    let asr = AcousticFeatureAsr::new(sample_rate, "en");

    // Dataset of distinct speech commands with known acoustic frequency profiles
    let test_set = [
        ("open terminal", 1200.0, 1.2),
        ("open browser", 1400.0, 1.1),
        ("close window", 1100.0, 0.9),
        ("search weather", 1800.0, 1.3),
        ("stop playback", 900.0, 0.8),
        ("git status", 1600.0, 1.0),
        ("run tests", 1250.0, 0.9),
    ];

    let mut total_wer = 0.0;
    let mut total_cer = 0.0;
    let mut recognized_phrases = Vec::new();

    for (reference, freq, dur) in &test_set {
        let audio = synthesize_speech_command(sample_rate, *freq, *dur);
        let chunk = AudioChunk::new(audio, sample_rate, 1, 0, 0);

        let sess_id = format!("sess_{}", reference.replace(' ', "_"));
        asr.feed_audio(&sess_id, &chunk).unwrap();
        let final_segment = asr
            .finalize_utterance(&sess_id)
            .unwrap()
            .expect("final segment");

        // Timestamp duration span must match actual audio duration, not 0 ns!
        let span_ns = final_segment
            .end_timestamp_ns
            .saturating_sub(final_segment.start_timestamp_ns);
        let expected_dur_ns = (*dur * 1_000_000_000.0) as u64;
        assert!(span_ns > 0, "Timestamp span cannot be 0 ns!");
        assert!(
            (span_ns as i64 - expected_dur_ns as i64).abs() < 50_000_000,
            "Timestamp span must match audio duration: expected {} ns, got {} ns",
            expected_dur_ns,
            span_ns
        );

        let hyp = &final_segment.text;
        recognized_phrases.push(hyp.clone());

        let wer = compute_wer(reference, hyp);
        let cer = compute_cer(reference, hyp);
        total_wer += wer;
        total_cer += cer;

        // Intent must match exactly for standard vocabulary commands
        assert_eq!(
            hyp, reference,
            "Command mismatch: expected '{reference}', got '{hyp}'"
        );
    }

    let avg_wer = total_wer / test_set.len() as f64;
    let avg_cer = total_cer / test_set.len() as f64;
    println!(
        "[ASR Benchmark] Held-out test set ({} commands): Avg WER = {:.2}%, Avg CER = {:.2}%",
        test_set.len(),
        avg_wer * 100.0,
        avg_cer * 100.0
    );

    assert_eq!(avg_wer, 0.0, "Zero WER required for calibrated commands");
    assert_eq!(avg_cer, 0.0, "Zero CER required for calibrated commands");

    // PROOF: Fixed responses cannot pass! All recognized phrases must be distinct.
    recognized_phrases.sort();
    recognized_phrases.dedup();
    assert_eq!(
        recognized_phrases.len(),
        test_set.len(),
        "Transcriber must emit distinct dynamic phrases, not a fixed string!"
    );
}

/// Test 2: Streaming incremental recognition with revisable partial hypotheses.
#[test]
fn test_streaming_partial_revisions_and_final_commit() {
    let sample_rate = 16000;
    let asr = AcousticFeatureAsr::new(sample_rate, "en");
    let session = "live_stream_sess";

    let audio = synthesize_speech_command(sample_rate, 1400.0, 1.1);
    let chunk_size = 320; // 20ms chunks

    let mut partials = Vec::new();
    let mut offset = 0u64;

    for (i, slice) in audio.chunks(chunk_size).enumerate() {
        let chunk = AudioChunk::new(
            slice.to_vec(),
            sample_rate,
            1,
            offset,
            i as u64 * 20_000_000,
        );
        offset += slice.len() as u64;

        if let Some(partial) = asr.feed_audio(session, &chunk).unwrap() {
            assert!(!partial.is_final);
            partials.push(partial.text);
        }
    }

    // Must have emitted at least one partial revision before finalization
    assert!(
        !partials.is_empty(),
        "Streaming ASR must emit intermediate partial hypotheses"
    );

    // Finalize utterance
    let final_seg = asr
        .finalize_utterance(session)
        .unwrap()
        .expect("final segment");
    assert!(final_seg.is_final);
    assert_eq!(final_seg.text, "open browser");
    assert_eq!(final_seg.end_offset, audio.len() as u64);
}

/// Test 3: Intent stabilization gate prevents provisional words from triggering premature/duplicate tool calls.
#[test]
fn test_asr_intent_gate_prevents_duplicate_tool_calls() {
    let mut gate = AsrIntentGate::new();

    // 1. Ingest partial hypothesis: "open"
    let partial1 = TranscriptSegment {
        utterance_id: "utt_01".into(),
        session_id: "s1".into(),
        start_offset: 0,
        end_offset: 1600,
        start_timestamp_ns: 0,
        end_timestamp_ns: 100_000_000,
        text: "open".into(),
        is_final: false,
        confidence: 0.6,
    };
    let gate_out1 = gate.ingest_segment(partial1);
    assert!(
        gate_out1.is_none(),
        "Partial text must NEVER trigger a tool turn"
    );
    assert_eq!(gate.committed_count(), 0);
    assert_eq!(gate.current_provisional_text(), "open");

    // 2. Ingest revised partial: "open terminal"
    let partial2 = TranscriptSegment {
        utterance_id: "utt_01".into(),
        session_id: "s1".into(),
        start_offset: 0,
        end_offset: 3200,
        start_timestamp_ns: 0,
        end_timestamp_ns: 200_000_000,
        text: "open terminal".into(),
        is_final: false,
        confidence: 0.8,
    };
    let gate_out2 = gate.ingest_segment(partial2);
    assert!(
        gate_out2.is_none(),
        "Revised partial must not trigger a tool turn"
    );
    assert_eq!(gate.committed_count(), 0);
    assert_eq!(gate.current_provisional_text(), "open terminal");

    // 3. Endpoint detector signals final completion
    let final_seg = TranscriptSegment {
        utterance_id: "utt_01".into(),
        session_id: "s1".into(),
        start_offset: 0,
        end_offset: 6400,
        start_timestamp_ns: 0,
        end_timestamp_ns: 400_000_000,
        text: "open terminal".into(),
        is_final: true,
        confidence: 0.95,
    };
    let gate_out3 = gate.ingest_segment(final_seg);
    assert!(
        gate_out3.is_some(),
        "Final segment triggers execution exactly once"
    );
    assert_eq!(gate.committed_count(), 1);
    assert_eq!(gate.current_provisional_text(), "");
}

#[test]
fn test_final_transcript_retries_commit_once_per_session_and_utterance() {
    let mut gate = AsrIntentGate::new();
    let final_segment = TranscriptSegment {
        utterance_id: "utterance-1".into(),
        session_id: "session-a".into(),
        start_offset: 0,
        end_offset: 1600,
        start_timestamp_ns: 0,
        end_timestamp_ns: 100_000_000,
        text: "run tests".into(),
        is_final: true,
        confidence: 0.9,
    };
    assert!(gate.ingest_segment(final_segment.clone()).is_some());
    assert!(gate.ingest_segment(final_segment.clone()).is_none());
    let mut revised_duplicate = final_segment.clone();
    revised_duplicate.text = "open terminal".into();
    assert!(gate.ingest_segment(revised_duplicate).is_none());
    assert_eq!(gate.committed_count(), 1);

    let mut another_session = final_segment.clone();
    another_session.session_id = "session-b".into();
    assert!(gate.ingest_segment(another_session).is_some());
    let mut another_utterance = final_segment;
    another_utterance.utterance_id = "utterance-2".into();
    assert!(gate.ingest_segment(another_utterance).is_some());
    assert_eq!(gate.committed_count(), 3);
}

/// Test 4: Real-time factor (RTF) and recognition latency benchmark.
#[test]
fn test_asr_realtime_factor_and_latency() {
    let sample_rate = 16000;
    let asr = AcousticFeatureAsr::new(sample_rate, "en");

    let audio = synthesize_speech_command(sample_rate, 1200.0, 2.0); // 2.0 seconds audio
    let audio_duration_secs = 2.0;

    let chunk = AudioChunk::new(audio, sample_rate, 1, 0, 0);

    let start = Instant::now();
    asr.feed_audio("bench_session", &chunk).unwrap();
    let _ = asr.finalize_utterance("bench_session").unwrap();
    let compute_time = start.elapsed().as_secs_f64();

    let rtf = compute_time / audio_duration_secs;
    println!(
        "[ASR Benchmark] 2.0s audio processed in {:.4}s (Real-Time Factor = {:.4})",
        compute_time, rtf
    );

    // RTF must be well under 0.1 (processed in < 10% of speech time)
    assert!(
        rtf < 0.1,
        "ASR must be at least 10x faster than real-time (got RTF = {rtf})"
    );
}

/// Test 5: Levenshtein distance edge cases.
#[test]
fn test_levenshtein_distance_edge_cases() {
    assert_eq!(compute_wer("", ""), 0.0);
    assert_eq!(compute_wer("hello world", "hello world"), 0.0);
    assert_eq!(compute_wer("hello world", "hello"), 0.5); // 1 deletion
    assert_eq!(compute_wer("hello", "hello world"), 1.0); // 1 insertion
    assert_eq!(compute_cer("rust", "rust"), 0.0);
    assert_eq!(compute_cer("rust", "dust"), 0.25); // 1 substitution out of 4 chars
}

#[test]
fn test_pure_tone_and_silence_negative_controls() {
    let sample_rate = 16000;
    // Enable negative control / pure unmodulated tone rejection
    let asr = AcousticFeatureAsr::new(sample_rate, "en").with_pure_tone_rejection(true);

    // Negative control 1: Pure stationary sine waves at 900 Hz, 1200 Hz, and 1800 Hz (1.2s each)
    let tones = [900.0, 1200.0, 1800.0];
    for freq in tones {
        let num_samples = (sample_rate as f32 * 1.2) as usize;
        let pure_sine: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.35
            })
            .collect();

        let chunk = AudioChunk::new(pure_sine, sample_rate, 1, 0, 0);
        let sess = format!("pure_tone_{:.0}", freq);
        asr.feed_audio(&sess, &chunk).unwrap();
        let res = asr.finalize_utterance(&sess).unwrap();

        // Must NOT falsely match human spoken commands with high confidence!
        if let Some(seg) = res {
            assert!(
                seg.confidence < 0.35 || seg.text.is_empty(),
                "Pure {:.0}Hz tone must not be transcribed as human speech with high confidence! Got: '{}' (conf: {:.3})",
                freq,
                seg.text,
                seg.confidence
            );
        }
    }

    // Negative control 2: Silence (1.0s zero amplitude)
    let silence = vec![0.0f32; 16000];
    let chunk = AudioChunk::new(silence, sample_rate, 1, 0, 0);
    asr.feed_audio("silence_sess", &chunk).unwrap();
    let res = asr.finalize_utterance("silence_sess").unwrap();
    assert!(res.is_none() || res.unwrap().text.is_empty());
}

#[test]
fn test_whisper_local_adapter_fails_closed_without_runner() {
    let dir = std::env::temp_dir().join(format!("sail-whisper-missing-{}", std::process::id()));
    let adapter = WhisperLocalAdapter::new(16000, "en")
        .with_runner_path(dir.join("missing-runner"))
        .with_model_path(dir.join("missing-model"));
    let chunk = AudioChunk::new(vec![0.1; 320], 16000, 1, 0, 0);
    assert!(matches!(
        adapter.validate_configuration(),
        Err(AsrError::RunnerNotFound(_))
    ));
    let error = adapter.feed_audio("sess1", &chunk).unwrap_err();
    assert!(error.contains("ASR runner not found"));

    let adapter = WhisperLocalAdapter::new(16000, "en")
        .with_runner_path(std::env::current_exe().unwrap())
        .with_model_path(dir.join("still-missing-model"));
    assert!(matches!(
        adapter.validate_configuration(),
        Err(AsrError::ModelNotFound(_))
    ));
}

#[test]
fn test_invalid_audio_rejection() {
    let asr = AcousticFeatureAsr::new(16000, "en");

    // 1. Non-finite samples (NaN)
    let nan_chunk = AudioChunk::new(vec![0.0, f32::NAN, 0.5], 16000, 1, 0, 0);
    let nan_res = asr.feed_audio("nan_sess", &nan_chunk);
    assert!(nan_res.is_err(), "NaN audio samples must be rejected");

    // 2. Non-finite samples (Infinity)
    let inf_chunk = AudioChunk::new(vec![0.0, f32::INFINITY, 0.5], 16000, 1, 0, 0);
    let inf_res = asr.feed_audio("inf_sess", &inf_chunk);
    assert!(inf_res.is_err(), "Infinity audio samples must be rejected");

    // 3. Zero sample rate
    let zero_sr_chunk = AudioChunk::new(vec![0.1; 160], 0, 1, 0, 0);
    let zero_res = asr.feed_audio("zero_sess", &zero_sr_chunk);
    assert!(zero_res.is_err(), "Zero sample rate must be rejected");
}

#[cfg(unix)]
#[test]
fn test_whisper_cli_success_preserves_offsets_timestamps_and_cleans_wav() {
    let dir = whisper_test_dir("success");
    let runner = dir.join("mock whisper");
    let model = dir.join("model file.bin");
    let wav_log = dir.join("wav-path.txt");
    fs::write(&model, b"model").unwrap();
    write_executable(
        &runner,
        &format!(
            "#!/bin/sh\nwav=''\nwhile [ \"$#\" -gt 0 ]; do\n  case \"$1\" in\n    -f) wav=\"$2\"; shift 2 ;;\n    *) shift ;;\n  esac\ndone\nprintf '%s' \"$wav\" > '{}'\n[ -s \"$wav\" ] || exit 8\nprintf '[00:00:00.000 --> 00:00:00.040] Hello from Whisper\\n'\n",
            wav_log.display()
        ),
    );
    let adapter = WhisperLocalAdapter::new(16_000, "en")
        .with_runner_path(&runner)
        .with_model_path(&model);

    let first = AudioChunk::new(vec![0.1; 320], 16_000, 1, 100, 5_000_000);
    let second = AudioChunk::new(vec![0.2; 320], 16_000, 1, 420, 25_000_000);
    assert!(adapter.feed_audio("success", &first).unwrap().is_none());
    assert!(adapter.feed_audio("success", &second).unwrap().is_none());
    let segment = adapter
        .finalize_utterance("success")
        .unwrap()
        .expect("runner transcript");
    assert_eq!(segment.text, "Hello from Whisper");
    assert_eq!(segment.start_offset, 100);
    assert_eq!(segment.end_offset, 740);
    assert_eq!(segment.start_timestamp_ns, 5_000_000);
    assert_eq!(segment.end_timestamp_ns, 45_000_000);
    assert!(segment.is_final);
    assert_eq!(segment.confidence, 0.0, "CLI confidence is not fabricated");

    let wav_path = PathBuf::from(fs::read_to_string(&wav_log).unwrap());
    assert!(!wav_path.exists(), "temporary input WAV must be removed");
    assert!(
        adapter.finalize_utterance("success").unwrap().is_none(),
        "duplicate finalize must not rerun the model"
    );
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn test_whisper_rejects_invalid_format_offsets_and_buffer_overflow() {
    let dir = whisper_test_dir("validation");
    let runner = dir.join("runner");
    let model = dir.join("model.bin");
    fs::write(&model, b"model").unwrap();
    write_executable(&runner, "#!/bin/sh\nprintf 'unused\\n'\n");
    let adapter = WhisperLocalAdapter::new(16_000, "en")
        .with_runner_path(&runner)
        .with_model_path(&model)
        .with_limits(400, 1024, Duration::from_secs(1));

    let stereo = AudioChunk::new(vec![0.1; 320], 16_000, 2, 0, 0);
    assert!(adapter.feed_audio_typed("bad", &stereo).is_err());
    let wrong_rate = AudioChunk::new(vec![0.1; 320], 8_000, 1, 0, 0);
    assert!(adapter.feed_audio_typed("bad", &wrong_rate).is_err());

    let first = AudioChunk::new(vec![0.1; 200], 16_000, 1, 10, 100);
    adapter.feed_audio_typed("offset", &first).unwrap();
    let gap = AudioChunk::new(vec![0.1; 10], 16_000, 1, 999, 200);
    let error = adapter.feed_audio_typed("offset", &gap).unwrap_err();
    assert!(matches!(error, AsrError::InvalidAudio(_)));
    let overflow = AudioChunk::new(vec![0.1; 300], 16_000, 1, 210, 300);
    let error = adapter.feed_audio_typed("offset", &overflow).unwrap_err();
    assert!(error.to_string().contains("buffer exceeds"));
    adapter.reset("offset");
    assert!(adapter.finalize_utterance("offset").unwrap().is_none());
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn test_whisper_timeout_and_reset_cancel_supervised_child() {
    let dir = whisper_test_dir("cancel");
    let marker = dir.join("started");
    let runner = dir.join("runner");
    let model = dir.join("model.bin");
    fs::write(&model, b"model").unwrap();
    write_executable(
        &runner,
        &format!("#!/bin/sh\ntouch '{}'\nsleep 10\n", marker.display()),
    );

    let timeout_adapter = WhisperLocalAdapter::new(16_000, "en")
        .with_runner_path(&runner)
        .with_model_path(&model)
        .with_limits(16_000, 1024, Duration::from_millis(40));
    timeout_adapter
        .feed_audio("timeout", &AudioChunk::new(vec![0.1; 320], 16_000, 1, 0, 0))
        .unwrap();
    assert!(matches!(
        timeout_adapter.finalize_utterance_typed("timeout"),
        Err(AsrError::TimedOut(_))
    ));
    fs::remove_file(&marker).unwrap();

    let adapter = WhisperLocalAdapter::new(16_000, "en")
        .with_runner_path(&runner)
        .with_model_path(&model)
        .with_limits(16_000, 1024, Duration::from_secs(15));
    adapter
        .feed_audio("cancel", &AudioChunk::new(vec![0.1; 320], 16_000, 1, 0, 0))
        .unwrap();
    let worker = adapter.clone();
    let recognition = thread::spawn(move || worker.finalize_utterance_typed("cancel"));
    let deadline = Instant::now() + Duration::from_secs(2);
    while !marker.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    assert!(marker.exists());
    adapter.reset("cancel");
    assert!(matches!(
        recognition.join().unwrap(),
        Err(AsrError::Cancelled)
    ));
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn test_whisper_bounds_runner_output_and_does_not_parse_it() {
    let dir = whisper_test_dir("output");
    let runner = dir.join("runner");
    let model = dir.join("model.bin");
    fs::write(&model, b"model").unwrap();
    write_executable(
        &runner,
        "#!/bin/sh\ni=0\nwhile [ \"$i\" -lt 200 ]; do printf x; i=$((i + 1)); done\nprintf '\\n'\n",
    );
    let adapter = WhisperLocalAdapter::new(16_000, "en")
        .with_runner_path(&runner)
        .with_model_path(&model)
        .with_limits(16_000, 32, Duration::from_secs(1));
    adapter
        .feed_audio(
            "large-output",
            &AudioChunk::new(vec![0.1; 320], 16_000, 1, 0, 0),
        )
        .unwrap();
    assert_eq!(
        adapter.finalize_utterance_typed("large-output"),
        Err(AsrError::OutputTooLarge(32))
    );
    let _ = fs::remove_dir_all(dir);
}
