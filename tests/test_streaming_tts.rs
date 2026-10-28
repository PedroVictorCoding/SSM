//! Comprehensive test suite for streaming speech synthesis, prefix commitment,
//! word-alignment tracking, cancellation semantics, and real-time performance.

use sail::voice::{
    AcousticTokenDecoder, FormantNeuralTtsEngine, PiperLocalTtsAdapter, PiperTtsError,
    PlaybackPipeline, SailNeuralVocoder, StreamingTtsAdapter, TextPrefixCommitmentGate,
    VoiceConfig, measure_tts_benchmark,
};

#[cfg(unix)]
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::thread;
#[cfg(unix)]
use std::time::{Duration, Instant};

#[cfg(unix)]
fn piper_test_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "sail-piper-{label}-{}-{}",
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

#[cfg(unix)]
fn write_wav_fixture(path: &Path) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 22_050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for index in 0..2205 {
        let sample = ((index as f32 * 0.05).sin() * i16::MAX as f32 * 0.25) as i16;
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn test_text_prefix_commitment_gate_prevents_partial_fragments() {
    let mut gate = TextPrefixCommitmentGate::new();

    // Push partial fragment of "Hello"
    let c1 = gate.push_token("Hel");
    assert!(c1.is_empty(), "Incomplete token fragment should not commit");

    // Complete the word with trailing space
    let c2 = gate.push_token("lo ");
    assert_eq!(c2, vec!["Hello"]);

    // Push multiple words and punctuation
    let c3 = gate.push_token("world! How are");
    assert_eq!(c3, vec!["world", "How"]);

    // Complete "are" and add "you?"
    let c4 = gate.push_token(" you?");
    assert_eq!(c4, vec!["are", "you"]);

    // Flush any trailing buffer
    let c5 = gate.push_token("final");
    assert!(c5.is_empty());
    let flushed = gate.flush();
    assert_eq!(flushed, vec!["final"]);

    assert_eq!(
        gate.committed_words(),
        &["Hello", "world", "How", "are", "you", "final"]
    );
}

#[test]
fn test_streaming_tts_synthesis_and_word_alignment() {
    let mut engine = FormantNeuralTtsEngine::new(16000);
    let config = VoiceConfig::default();
    let utterance_id = "test_utt_alignment";

    engine
        .start_utterance(utterance_id, &config)
        .expect("start utterance");

    // Feed streaming tokens
    let chunks1 = engine
        .feed_text(utterance_id, "The system ")
        .expect("feed text 1");
    assert_eq!(
        chunks1.len(),
        2,
        "Should emit chunks for 'The' and 'system'"
    );

    let chunks2 = engine
        .feed_text(utterance_id, "is online.")
        .expect("feed text 2");
    assert_eq!(chunks2.len(), 2, "Should emit chunks for 'is' and 'online'");

    let final_chunks = engine
        .finalize_utterance(utterance_id)
        .expect("finalize utterance");
    assert_eq!(final_chunks.len(), 1);
    assert!(final_chunks[0].is_final_chunk);

    // Verify word alignments and timestamps
    let mut all_chunks = Vec::new();
    all_chunks.extend(chunks1);
    all_chunks.extend(chunks2);

    let mut prev_end_sample = 0;
    for chunk in &all_chunks {
        assert!(!chunk.samples.is_empty());
        assert_eq!(chunk.spoken_words.len(), 1);
        let align = &chunk.spoken_words[0];
        assert_eq!(align.start_sample, prev_end_sample);
        assert_eq!(align.end_sample, prev_end_sample + chunk.samples.len());
        assert!(align.end_ms > align.start_ms);
        prev_end_sample = align.end_sample;
    }
}

#[test]
fn test_mid_utterance_cancellation_and_playback_progress() {
    let mut engine = FormantNeuralTtsEngine::new(16000);
    let config = VoiceConfig::default();
    let utterance_id = "test_utt_cancellation";

    engine
        .start_utterance(utterance_id, &config)
        .expect("start utterance");

    let chunks = engine
        .feed_text(utterance_id, "One two three four five six.")
        .expect("feed text");
    assert!(!chunks.is_empty());

    // Calculate sample offset when word 2 ("two") finishes
    let word2_end_sample = chunks[1].spoken_words[0].end_sample as u64;

    // Check playback status when cursor is at word 2
    let status = engine
        .get_playback_status(utterance_id, word2_end_sample)
        .expect("playback status");
    assert_eq!(status.played_words, vec!["One", "two"]);
    assert_eq!(status.pending_words, vec!["three", "four", "five", "six"]);
    assert!(!status.is_completed);

    // Cancel utterance at this exact cursor offset
    let cancel_report = engine
        .cancel_utterance(utterance_id, word2_end_sample)
        .expect("cancel utterance");

    assert_eq!(cancel_report.utterance_id, utterance_id);
    assert_eq!(cancel_report.played_words, vec!["One", "two"]);
    assert_eq!(
        cancel_report.committed_words,
        vec!["One", "two", "three", "four", "five", "six"]
    );
    assert_eq!(cancel_report.played_samples, word2_end_sample);
    assert!(cancel_report.discarded_samples > 0);
    assert_eq!(
        cancel_report.total_synthesized_samples,
        cancel_report.played_samples + cancel_report.discarded_samples
    );

    // After cancellation, feeding further text emits nothing
    let post_chunks = engine
        .feed_text(utterance_id, "seven eight")
        .expect("post feed");
    assert!(post_chunks.is_empty());
}

#[test]
fn test_phonetic_intelligibility_and_distinctiveness() {
    let engine = FormantNeuralTtsEngine::new(16000);

    // Three phonetically very different phrases
    let p1 = engine.synthesize_text("cat sat on the mat");
    let p2 = engine.synthesize_text("ship sailing across the blue ocean");
    let p3 = engine.synthesize_text("buzzing bees in sunny breeze");

    assert!(!p1.is_empty());
    assert!(!p2.is_empty());
    assert!(!p3.is_empty());

    // Calculate zero-crossing rate (ZCR) representing high vs low frequency phonetic balance
    let calc_zcr = |samples: &[f32]| -> f32 {
        let mut zc = 0;
        for i in 1..samples.len() {
            if (samples[i - 1] >= 0.0 && samples[i] < 0.0)
                || (samples[i - 1] < 0.0 && samples[i] >= 0.0)
            {
                zc += 1;
            }
        }
        zc as f32 / samples.len() as f32
    };

    let zcr1 = calc_zcr(&p1);
    let zcr2 = calc_zcr(&p2);
    let zcr3 = calc_zcr(&p3);

    // Verify phrases have distinct acoustic signatures, not uniform tones
    assert!(
        (zcr1 - zcr2).abs() > 0.005 || (zcr2 - zcr3).abs() > 0.005,
        "Acoustic features should distinctly reflect phoneme phonetic composition (zcr1={zcr1}, zcr2={zcr2}, zcr3={zcr3})"
    );

    // Verify audio amplitude is well-scaled without hard clipping
    let max_amp1 = p1.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    let max_amp2 = p2.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    assert!(max_amp1 > 0.05 && max_amp1 <= 0.99);
    assert!(max_amp2 > 0.05 && max_amp2 <= 0.99);
}

#[test]
fn test_tts_benchmark_latency_and_real_time_factor() {
    let mut engine = FormantNeuralTtsEngine::new(16000);
    let phrases = [
        "Welcome to the Sail multimodal framework.",
        "Executing local neural inference on available hardware.",
        "Speech synthesis delivers real-time acoustic waveforms.",
    ];

    let report = measure_tts_benchmark(&mut engine, &phrases);

    assert_eq!(report.num_phrases, 3);
    assert!(report.total_audio_duration_sec > 3.0);
    assert!(report.words_synthesized > 10);

    // First audio chunk latency must be low for conversational responsiveness (< 50ms target)
    println!(
        "TTS First-Audio Latency: {:.2} ms (Target: < 50 ms)",
        report.first_chunk_latency_ms
    );
    assert!(
        report.first_chunk_latency_ms < 50.0,
        "First-audio latency must be < 50ms, got {:.2}ms",
        report.first_chunk_latency_ms
    );

    // Real-Time Factor (RTF) must be substantially below 1.0 (target < 0.5, actual < 0.05)
    println!(
        "TTS Real-Time Factor: {:.4} (Target: < 0.50, Headroom: {:.1}x)",
        report.real_time_factor,
        1.0 / report.real_time_factor.max(1e-6)
    );
    assert!(
        report.real_time_factor < 0.50,
        "RTF must be < 0.50 with headroom, got {:.4}",
        report.real_time_factor
    );
}

#[test]
fn test_voice_config_pitch_and_timbre() {
    let female = VoiceConfig::female();
    let male = VoiceConfig::male();

    assert!(female.base_f0 > male.base_f0);
    assert!(female.formant_scale > male.formant_scale);

    let mut engine = FormantNeuralTtsEngine::new(16000);

    engine.start_utterance("utt_female", &female).unwrap();
    let c_female = engine.feed_text("utt_female", "Hello world").unwrap();
    engine.finalize_utterance("utt_female").unwrap();

    engine.start_utterance("utt_male", &male).unwrap();
    let c_male = engine.feed_text("utt_male", "Hello world").unwrap();
    engine.finalize_utterance("utt_male").unwrap();

    assert!(!c_female.is_empty());
    assert!(!c_male.is_empty());

    // Both outputs should be non-empty and well-formed
    assert!(c_female[0].samples.len() > 0);
    assert!(c_male[0].samples.len() > 0);
}

#[test]
fn test_playback_pipeline_direct_feed_and_interruption() {
    let pipeline = PlaybackPipeline::new(16000, 32);
    let mut engine = FormantNeuralTtsEngine::new(16000);
    let config = VoiceConfig::default();

    engine.start_utterance("pipe_test", &config).unwrap();
    let chunks = engine
        .feed_text("pipe_test", "Realtime playback pipeline test.")
        .unwrap();
    engine.finalize_utterance("pipe_test").unwrap();

    // Push chunks directly into the PlaybackPipeline
    for chunk in chunks {
        let sail_chunk = sail::voice::AudioChunk::new(
            chunk.samples.to_vec(),
            chunk.sample_rate,
            chunk.channels,
            chunk.sample_offset,
            0,
        );
        let res = pipeline.enqueue_chunk(sail_chunk);
        assert!(res.is_ok(), "Playback pipeline should accept chunks");
    }

    assert!(pipeline.queued_chunks() > 0);

    // Pull a frame through audio speaker callback
    let mut out_buffer = vec![0.0f32; 320];
    pipeline.pull_samples(&mut out_buffer);
    assert!(out_buffer.iter().any(|&s| s != 0.0));
    assert_eq!(pipeline.playback_cursor(), 320);

    // Mid-stream interruption immediately halts playback
    let cursor_at_stop = pipeline.interrupt();
    assert_eq!(pipeline.queued_chunks(), 0, "Interrupt must flush queue");
    assert!(cursor_at_stop >= 320);
}

#[test]
fn test_neural_vocoder_token_decoding() {
    let vocoder = SailNeuralVocoder::new(256);
    assert_eq!(vocoder.codebook_size(), 256);

    let tokens = vec![12u32, 45, 99, 150, 210];
    let audio = vocoder
        .decode_tokens(&tokens, 16000)
        .expect("decode tokens");

    // 5 tokens * 320 samples (20ms each) = 1600 samples = 100ms
    assert_eq!(audio.len(), 5 * 320);
    let max_amp = audio.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    assert!(max_amp > 0.1 && max_amp <= 0.5);
}

#[cfg(unix)]
#[test]
fn test_piper_adapter_synthesizes_committed_text_without_shell_interpolation() {
    let dir = piper_test_dir("success");
    let fixture = dir.join("fixture.wav");
    let input_log = dir.join("stdin.txt");
    let model = dir.join("voice model.onnx");
    let runner = dir.join("mock piper");
    write_wav_fixture(&fixture);
    fs::write(&model, b"test voice model").unwrap();
    write_executable(
        &runner,
        &format!(
            "#!/bin/sh\nout=''\nwhile [ \"$#\" -gt 0 ]; do\n  case \"$1\" in\n    --output_file) out=\"$2\"; shift 2 ;;\n    *) shift ;;\n  esac\ndone\ncat > '{}'\ncp '{}' \"$out\"\n",
            input_log.display(),
            fixture.display()
        ),
    );

    let mut adapter = PiperLocalTtsAdapter::new(&runner, &model);
    assert_eq!(adapter.name(), "piper-local");
    assert_eq!(adapter.runner_path(), runner.as_path());
    assert_eq!(adapter.voice_model_path(), model.as_path());
    adapter
        .start_utterance("utt;not-a-shell-command", &VoiceConfig::default())
        .unwrap();

    let chunks = adapter
        .feed_text("utt;not-a-shell-command", "Hello world ")
        .unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].sample_rate, 22_050);
    assert_eq!(chunks[0].channels, 1);
    assert_eq!(chunks[0].samples.len(), 2205);
    assert_eq!(chunks[0].spoken_words.len(), 2);
    assert!((chunks[0].duration_ms - 100.0).abs() < 0.1);
    assert_eq!(fs::read_to_string(&input_log).unwrap(), "Hello world\n");

    let final_chunks = adapter
        .finalize_utterance("utt;not-a-shell-command")
        .unwrap();
    assert_eq!(final_chunks.len(), 1);
    assert!(final_chunks[0].is_final_chunk);
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn test_piper_adapter_missing_dependencies_fail_closed_without_audio() {
    let dir = piper_test_dir("missing");
    let missing_runner = dir.join("missing-piper");
    let missing_model = dir.join("missing.onnx");
    let mut adapter = PiperLocalTtsAdapter::new(&missing_runner, &missing_model);
    assert_eq!(
        adapter.validate_configuration(),
        Err(PiperTtsError::RunnerNotFound(missing_runner.clone()))
    );
    let error = adapter
        .start_utterance("missing", &VoiceConfig::default())
        .expect_err("missing runner must fail closed");
    assert!(error.contains("PiperRunnerNotFound"));

    write_executable(&missing_runner, "#!/bin/sh\nexit 0\n");
    let mut adapter = PiperLocalTtsAdapter::new(&missing_runner, &missing_model);
    let error = adapter
        .start_utterance("missing-model", &VoiceConfig::default())
        .expect_err("missing voice must fail closed");
    assert!(error.contains("PiperVoiceModelNotFound"));
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn test_piper_adapter_cancellation_terminates_child_and_clears_state() {
    let dir = piper_test_dir("cancel");
    let marker = dir.join("started");
    let model = dir.join("voice.onnx");
    let runner = dir.join("slow-piper");
    fs::write(&model, b"test voice model").unwrap();
    write_executable(
        &runner,
        &format!(
            "#!/bin/sh\nout=''\nwhile [ \"$#\" -gt 0 ]; do\n  case \"$1\" in\n    --output_file) out=\"$2\"; shift 2 ;;\n    *) shift ;;\n  esac\ndone\ncat >/dev/null\ntouch '{}'\nsleep 10\n",
            marker.display()
        ),
    );

    let mut adapter =
        PiperLocalTtsAdapter::new(&runner, &model).with_timeout(Duration::from_secs(15));
    adapter
        .start_utterance("cancel-me", &VoiceConfig::default())
        .unwrap();
    let mut worker = adapter.clone();
    let synthesis = thread::spawn(move || worker.feed_text("cancel-me", "Please stop now "));

    let deadline = Instant::now() + Duration::from_secs(2);
    while !marker.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    assert!(marker.exists(), "mock Piper child did not start");
    let report = adapter.cancel_utterance("cancel-me", 0).unwrap();
    assert_eq!(report.total_synthesized_samples, 0);
    assert_eq!(report.committed_words, vec!["Please", "stop", "now"]);

    let result = synthesis.join().unwrap();
    let error = result.expect_err("cancelled synthesis must emit no fabricated audio");
    assert!(
        error.contains("PiperCancelled"),
        "unexpected error: {error}"
    );
    let status = adapter.get_playback_status("cancel-me", 0).unwrap();
    assert!(status.pending_words.is_empty());
    let _ = fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn test_piper_adapter_bounds_child_runtime_and_reports_process_failure() {
    let dir = piper_test_dir("bounded");
    let model = dir.join("voice.onnx");
    let slow_runner = dir.join("slow-piper");
    fs::write(&model, b"test voice model").unwrap();
    write_executable(&slow_runner, "#!/bin/sh\ncat >/dev/null\nsleep 10\n");

    let mut adapter =
        PiperLocalTtsAdapter::new(&slow_runner, &model).with_timeout(Duration::from_millis(40));
    adapter
        .start_utterance("timeout", &VoiceConfig::default())
        .unwrap();
    let error = adapter
        .feed_text("timeout", "Bound this child ")
        .expect_err("slow child must time out");
    assert!(error.contains("PiperTimedOut"), "unexpected error: {error}");

    let failing_runner = dir.join("failing-piper");
    write_executable(&failing_runner, "#!/bin/sh\ncat >/dev/null\nexit 7\n");
    let mut adapter = PiperLocalTtsAdapter::new(&failing_runner, &model);
    adapter
        .start_utterance("failure", &VoiceConfig::default())
        .unwrap();
    let error = adapter
        .feed_text("failure", "Fail explicitly ")
        .expect_err("non-zero child must fail");
    assert!(error.contains("PiperProcessFailed"));
    assert!(error.contains("status"));
    let _ = fs::remove_dir_all(dir);
}
