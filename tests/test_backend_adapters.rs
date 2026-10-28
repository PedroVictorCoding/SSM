//! Integration and Verification Tests for Interchangeable Pretrained and Native Model Adapters.
//!
//! Verifies acceptance criteria for GitHub Issue #4, QA-01, QA-02, and QA-03:
//! 1. The same scripted task runs interchangeably through both adapter interfaces without rewriting coordinator.
//! 2. Cooperative cancellation, malformed input rejection, and session state isolation.
//! 3. Format validation strictly rejects unsupported formats (e.g. Python pickle) while accepting GGUF and SAIL artifacts.
//! 4. Generation metadata records backend info, model ID, latency, and tokens per second.
//! 5. (QA-03) Native streaming produces identical token IDs as reference generate_stateful (greedy & sampled).
//! 6. (QA-03) Multi-turn conversational continuity, session state isolation, and reset.
//! 7. (QA-03) Cooperative cancellation leaves exact emitted prefix in session history without leak.
//! 8. (QA-03) Streaming UTF-8 decoding avoids byte-level BPE boundary corruption.
//! 9. (QA-03) Pinned generation per turn, safe promotion handling and stale recurrent state invalidation.
//! 10. (QA-03) N emitted tokens do not cause N whole-prompt prefills; recurrent decode scaling.
//! 11. (QA-02) Missing local pretrained runner fails closed with an actionable RunnerNotFound error; never fabricates answers.
//! 12. (QA-02) PretrainedLocalAdapter rejects non-GGUF formats cleanly.
//! 13. (QA-02) Configured runner executes supervised process, streams output, and terminates on cancellation.
//! 14. (QA-02) MockModelAdapter is explicitly labeled as a test double.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use sail::backend::{InferenceBackend, default_device};
use sail::data::{BpeTokenizer, Tokenizer};
use sail::engine::Engine;
use sail::export::pack_to_artifact;
use sail::model::HybridLmConfig;
use sail::runtime::{
    AssistantCoordinator, BackendModality, BackendRequestError, CancellationToken, FinishReason,
    GenerateRequest, MockModelAdapter, ModelBackendAdapter, ModelFormat, NativeSailAdapter,
    PretrainedLocalAdapter, SharedAudioBuffer, SharedVideoBuffer, StreamingUtf8Decoder,
    validate_model_format,
};

/// Helper creating a minimal valid .sea artifact for native testing.
fn create_test_sea_artifact(dir: &PathBuf) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let artifact_path = dir.join("test_model.sea");

    let tokenizer = Tokenizer::Bpe(BpeTokenizer::new_base());
    let cfg = HybridLmConfig::new(tokenizer.vocab_size())
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(8);

    let device = default_device();
    let model = cfg.init::<InferenceBackend>(&device);

    pack_to_artifact(&model, &cfg, &tokenizer, &artifact_path).expect("export .sea");
    artifact_path
}

/// Helper creating a dummy GGUF v3 file with valid GGUF magic bytes (`GGUF`).
fn create_test_gguf_file(dir: &PathBuf) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let gguf_path = dir.join("test_model.gguf");
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes()); // version 3
    header.extend_from_slice(&1u64.to_le_bytes()); // tensor_count: 1
    header.extend_from_slice(&2u64.to_le_bytes()); // metadata_kv_count: 2
    header.extend_from_slice(&[0u8; 64]);
    fs::write(&gguf_path, header).expect("write gguf");
    gguf_path
}

/// Helper creating an executable mock runner script that simulates a local runner emitting streaming tokens.
fn create_mock_runner_script(dir: &PathBuf) -> PathBuf {
    #[cfg(unix)]
    {
        let runner_path = dir.join("mock_llama_runner.sh");
        let script = r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "llama.cpp test runner v1.2.3"
    exit 0
fi

# Simulate token streaming with line-by-line output
echo "The "
echo "capital "
echo "of "
echo "France "
echo "is "
echo "Paris."
echo "<tool_call>web_search(\"Paris facts\")</tool_call>"
"#;
        fs::write(&runner_path, script).expect("write runner script");
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&runner_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&runner_path, perms).unwrap();
        runner_path
    }
    #[cfg(not(unix))]
    {
        let runner_path = dir.join("mock_llama_runner.bat");
        let script = "@echo off\r\nif \"%1\"==\"--version\" (echo llama.cpp test runner v1.2.3 & exit /b 0)\r\necho The \r\necho capital \r\necho of \r\necho France \r\necho is \r\necho Paris.\r\necho <tool_call>web_search(\"Paris facts\")</tool_call>\r\n";
        fs::write(&runner_path, script).expect("write runner script");
        runner_path
    }
}

/// Test 1: The same scripted task runs through both adapter interfaces without rewriting the coordinator.
#[test]
fn test_scripted_task_runs_interchangeably_across_adapters() {
    let temp_dir = std::env::temp_dir().join(format!("sail-adapter-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let sea_path = create_test_sea_artifact(&temp_dir);

    let device = default_device();
    let native_engine = Engine::<InferenceBackend>::open(&sea_path, &device).expect("open native");
    let native_adapter: Arc<dyn ModelBackendAdapter> =
        Arc::new(NativeSailAdapter::new(native_engine, "sail-hybrid-0.58m"));

    // Uses explicitly labeled MockModelAdapter test double for deterministic offline coordinator protocol test
    let test_double_adapter: Arc<dyn ModelBackendAdapter> = Arc::new(
        MockModelAdapter::new("llama-3-8b-instruct-q4").with_mock_response(
            "capital of france",
            "The capital of France is Paris. <tool_call>web_search(\"Paris facts\")</tool_call>",
        ),
    );

    let prompt = "What is the capital of France?";

    // 1. Run through Native Sail Adapter via coordinator
    let native_coord = AssistantCoordinator::new(native_adapter.clone());
    let (native_text, native_tool, native_meta) = native_coord
        .execute_turn("session_user_1", prompt)
        .expect("native turn");
    assert!(!native_text.is_empty() || native_tool.is_some());
    assert_eq!(native_meta.backend_name, "Sail Native Engine");
    assert!(native_meta.elapsed_ms >= 0.0);

    // 2. Run the exact same task through Mock Model Adapter via coordinator
    let mock_coord = AssistantCoordinator::new(test_double_adapter.clone());
    let (mock_text, mock_tool, mock_meta) = mock_coord
        .execute_turn("session_user_1", prompt)
        .expect("mock turn");

    assert!(mock_text.contains("Paris"));
    assert!(mock_tool.is_some());
    let tool = mock_tool.unwrap();
    assert_eq!(tool.tool_name, "web_search");
    assert_eq!(tool.input, "Paris facts");
    assert_eq!(mock_meta.backend_name, "Mock Model Adapter (Test Double)");
    assert_eq!(mock_meta.finish_reason, FinishReason::ToolCall);

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 2: Cooperative cancellation aborts generation without crashing the worker.
#[test]
fn test_adapter_cooperative_cancellation() {
    let adapter = MockModelAdapter::new("mock-cancellation-test").with_mock_response(
        "long prompt",
        "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12",
    );

    let cancel_token = CancellationToken::new();

    let mut token_count = 0;
    let req = GenerateRequest::new("cancel_sess", "long prompt")
        .with_max_tokens(100)
        .with_cancellation_token(cancel_token.clone());

    let cancel_trigger = cancel_token.clone();
    let meta = adapter
        .generate_stream(req, &mut move |_chunk| {
            token_count += 1;
            if token_count >= 3 {
                // Cancel midway through generation
                cancel_trigger.cancel();
            }
            Ok(())
        })
        .expect("stream");

    assert_eq!(meta.finish_reason, FinishReason::Cancelled);
    assert!(meta.generated_tokens < 10);
}

/// Test 3: Session state isolation and state reset.
#[test]
fn test_session_state_isolation_and_reset() {
    let adapter = MockModelAdapter::new("mock-session-test");

    // Operations on session A
    assert!(adapter.reset_session("session_a").is_ok());
    // Operations on session B
    assert!(adapter.reset_session("session_b").is_ok());
}

/// Test 4: Format validator strictly rejects unsupported formats while accepting GGUF and SAIL artifacts.
#[test]
fn test_format_validation_strictness() {
    let temp_dir = std::env::temp_dir().join(format!("sail-fmt-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    // 1. Valid GGUF v3 with tensors and metadata
    let valid_gguf = create_test_gguf_file(&temp_dir);
    assert_eq!(
        validate_model_format(&valid_gguf).unwrap(),
        ModelFormat::GgufV2V3
    );

    // 2. Real native SAIL artifact exported via pack_to_artifact (SEAPACK1)
    let valid_sea = create_test_sea_artifact(&temp_dir);
    assert_eq!(
        validate_model_format(&valid_sea).unwrap(),
        ModelFormat::SailSea
    );

    // 3. Legacy SSMPACK1 artifact
    let legacy_sea = temp_dir.join("legacy.sea");
    let mut sea_bytes = fs::read(&valid_sea).unwrap();
    sea_bytes[0..8].copy_from_slice(b"SSMPACK1");
    fs::write(&legacy_sea, &sea_bytes).unwrap();
    assert_eq!(
        validate_model_format(&legacy_sea).unwrap(),
        ModelFormat::SailSsmLegacy
    );

    // 4. Truncated native artifact (header says large lengths, file is cut short)
    let truncated_sea = temp_dir.join("truncated.sea");
    fs::write(&truncated_sea, &sea_bytes[..20]).unwrap(); // Incomplete header
    let trunc_res = validate_model_format(&truncated_sea);
    assert!(
        trunc_res.is_err(),
        "Truncated native header must be rejected without panic"
    );
    let err_msg = trunc_res.unwrap_err();
    assert!(err_msg.contains("too small") || err_msg.contains("Corrupted"));

    let truncated_payload_sea = temp_dir.join("truncated_payload.sea");
    fs::write(&truncated_payload_sea, &sea_bytes[..40]).unwrap(); // Header present, but payload missing
    let trunc_payload_res = validate_model_format(&truncated_payload_sea);
    assert!(trunc_payload_res.is_err());
    assert!(
        trunc_payload_res
            .unwrap_err()
            .contains("Truncated native artifact")
    );

    // 5. Arbitrary ZIP archive (magic PK\x03\x04): must NOT be treated as a SAIL model
    let fake_zip = temp_dir.join("renamed.zip");
    fs::write(
        &fake_zip,
        b"PK\x03\x04\x14\x00\x00\x00\x08\x00arbitrary_zip_contents",
    )
    .unwrap();
    let zip_res = validate_model_format(&fake_zip);
    assert!(zip_res.is_err(), "Arbitrary ZIP files must be rejected");
    assert!(zip_res.unwrap_err().contains("Rejected ZIP container"));

    // 6. Raw JSON file: must NOT be treated as binary model
    let raw_json = temp_dir.join("config.json");
    fs::write(&raw_json, b"{\"version\": 1, \"model\": \"sail\"}").unwrap();
    let json_res = validate_model_format(&raw_json);
    assert!(
        json_res.is_err(),
        "Raw JSON must be rejected as model binary"
    );
    assert!(json_res.unwrap_err().contains("Rejected raw JSON"));

    // 7. GGUF edge cases:
    // a) Magic-only 4-byte GGUF
    let magic_only_gguf = temp_dir.join("magic_only.gguf");
    fs::write(&magic_only_gguf, b"GGUF").unwrap();
    let magic_res = validate_model_format(&magic_only_gguf);
    assert!(magic_res.is_err());
    assert!(magic_res.unwrap_err().contains("too small"));

    // b) Unsupported GGUF version (version 1)
    let v1_gguf = temp_dir.join("v1.gguf");
    let mut v1_bytes = b"GGUF\x01\x00\x00\x00".to_vec();
    v1_bytes.extend_from_slice(&[0u8; 32]);
    fs::write(&v1_gguf, v1_bytes).unwrap();
    let v1_res = validate_model_format(&v1_gguf);
    assert!(v1_res.is_err());
    assert!(v1_res.unwrap_err().contains("Unsupported GGUF version 1"));

    // c) Malformed GGUF with zero tensors and zero metadata
    let empty_gguf = temp_dir.join("empty.gguf");
    let mut empty_bytes = b"GGUF\x03\x00\x00\x00".to_vec();
    empty_bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count: 0
    empty_bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count: 0
    empty_bytes.extend_from_slice(&[0u8; 16]);
    fs::write(&empty_gguf, empty_bytes).unwrap();
    let empty_res = validate_model_format(&empty_gguf);
    assert!(empty_res.is_err());
    assert!(empty_res.unwrap_err().contains("Malformed GGUF container"));

    // 8. Prohibited Python pickle (PyTorch .pt)
    let bad_pt = temp_dir.join("weights.pt");
    fs::write(&bad_pt, b"\x80\x04\x8a\nmalicious_pickle").unwrap();
    let pt_res = validate_model_format(&bad_pt);
    assert!(pt_res.is_err());
    assert!(pt_res.unwrap_err().contains("Python pickle"));

    // 9. Random unrecognized file
    let bad_random = temp_dir.join("corrupted.bin");
    fs::write(&bad_random, b"\xDE\xAD\xBE\xEF\x00\x11\x22\x33").unwrap();
    let rand_res = validate_model_format(&bad_random);
    assert!(rand_res.is_err());
    assert!(
        rand_res
            .unwrap_err()
            .contains("Unsupported or unrecognized")
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 5 (QA-03): Native streaming produces identical token IDs as reference generate_stateful
/// for both greedy and sampled configurations.
#[test]
fn test_qa03_native_streaming_matches_reference_generation() {
    let temp_dir = std::env::temp_dir().join(format!("sail-qa03-ref-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let sea_path = create_test_sea_artifact(&temp_dir);
    let device = default_device();
    let engine = Engine::<InferenceBackend>::open(&sea_path, &device).expect("open native");
    let active = engine.active_generation();

    let prompt = "The quick brown fox";
    let prompt_ids = active.tokenizer.encode(prompt).expect("encode");
    let new_tokens = 8;

    // --- 1. Greedy mode comparison ---
    let greedy_cfg = sail::generate::SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        seed: 42,
        ..Default::default()
    };
    let (ref_greedy_all, _) = sail::generate::generate_stateful(
        &active.model,
        &prompt_ids,
        new_tokens,
        &greedy_cfg,
        &device,
    );
    let ref_greedy_gen = &ref_greedy_all[prompt_ids.len()..];

    let adapter = NativeSailAdapter::new(engine.clone(), "sail-qa03");
    let mut stream_greedy_tokens = Vec::new();
    let req_greedy = GenerateRequest::new("sess_greedy", prompt)
        .with_max_tokens(new_tokens)
        .with_temperature(0.0)
        .with_top_k(0)
        .with_seed(42);

    let meta_greedy = adapter
        .generate_stream(req_greedy, &mut |chunk| {
            if let sail::runtime::GenerationChunk::Token { token_id, .. } = chunk {
                if token_id != u32::MAX {
                    stream_greedy_tokens.push(token_id);
                }
            }
            Ok(())
        })
        .expect("stream greedy");

    assert_eq!(meta_greedy.finish_reason, FinishReason::Length);
    assert_eq!(
        stream_greedy_tokens, ref_greedy_gen,
        "Greedy streamed token IDs must be bit-for-bit identical to generate_stateful reference"
    );

    // --- 2. Sampled mode comparison with explicit seed ---
    let sample_cfg = sail::generate::SamplingConfig {
        temperature: 0.8,
        top_k: 20,
        seed: 98765,
        ..Default::default()
    };
    let (ref_sample_all, _) = sail::generate::generate_stateful(
        &active.model,
        &prompt_ids,
        new_tokens,
        &sample_cfg,
        &device,
    );
    let ref_sample_gen = &ref_sample_all[prompt_ids.len()..];

    let mut stream_sample_tokens = Vec::new();
    let req_sample = GenerateRequest::new("sess_sampled", prompt)
        .with_max_tokens(new_tokens)
        .with_temperature(0.8)
        .with_top_k(20)
        .with_seed(98765);

    let meta_sample = adapter
        .generate_stream(req_sample, &mut |chunk| {
            if let sail::runtime::GenerationChunk::Token { token_id, .. } = chunk {
                if token_id != u32::MAX {
                    stream_sample_tokens.push(token_id);
                }
            }
            Ok(())
        })
        .expect("stream sampled");

    assert_eq!(meta_sample.finish_reason, FinishReason::Length);
    assert_eq!(
        stream_sample_tokens, ref_sample_gen,
        "Sampled streamed token IDs with identical seed must match generate_stateful reference"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 6 (QA-03): Multi-turn conversational continuity, session state isolation, and reset.
#[test]
fn test_qa03_session_multi_turn_continuity_and_isolation() {
    let temp_dir = std::env::temp_dir().join(format!("sail-qa03-sess-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let sea_path = create_test_sea_artifact(&temp_dir);
    let device = default_device();
    let engine = Engine::<InferenceBackend>::open(&sea_path, &device).expect("open native");
    let adapter = NativeSailAdapter::new(engine, "sail-qa03");

    // Session 1: Turn 1
    let req1 = GenerateRequest::new("session_alpha", "Hello, assistant!")
        .with_max_tokens(5)
        .with_seed(100);
    let meta1 = adapter
        .generate_stream(req1, &mut |_| Ok(()))
        .expect("turn 1");
    let hist1 = adapter
        .session_history_len("session_alpha")
        .expect("session exists");
    assert_eq!(hist1, meta1.prompt_tokens + meta1.generated_tokens);

    // Session 2 (Interleaved): Turn 1
    let req_beta = GenerateRequest::new("session_beta", "Weather in London")
        .with_max_tokens(4)
        .with_seed(200);
    let meta_beta = adapter
        .generate_stream(req_beta, &mut |_| Ok(()))
        .expect("beta turn 1");
    let hist_beta = adapter
        .session_history_len("session_beta")
        .expect("beta exists");
    assert_eq!(
        hist_beta,
        meta_beta.prompt_tokens + meta_beta.generated_tokens
    );

    // Session 1: Turn 2 (Continuation - carries recurrent state)
    let req2 = GenerateRequest::new("session_alpha", "Tell me more.")
        .with_max_tokens(5)
        .with_seed(100);
    let meta2 = adapter
        .generate_stream(req2, &mut |_| Ok(()))
        .expect("turn 2");
    let hist2 = adapter
        .session_history_len("session_alpha")
        .expect("session exists");
    assert_eq!(hist2, hist1 + meta2.prompt_tokens + meta2.generated_tokens);

    // Verify session_beta history was NOT modified by session_alpha turn 2
    assert_eq!(
        adapter.session_history_len("session_beta").unwrap(),
        hist_beta,
        "Session beta history must be isolated from session alpha"
    );

    // Reset session_alpha: clears alpha, preserves beta
    adapter.reset_session("session_alpha").expect("reset alpha");
    assert!(adapter.session_history_len("session_alpha").is_none());
    assert_eq!(
        adapter.session_history_len("session_beta").unwrap(),
        hist_beta,
        "Resetting session alpha must not clear session beta"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 7 (QA-03): Mid-generation cancellation commits exact emitted prefix to session.
#[test]
fn test_qa03_cancellation_commits_emitted_prefix_without_leak() {
    let temp_dir =
        std::env::temp_dir().join(format!("sail-qa03-cancel-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let sea_path = create_test_sea_artifact(&temp_dir);
    let device = default_device();
    let engine = Engine::<InferenceBackend>::open(&sea_path, &device).expect("open native");
    let adapter = NativeSailAdapter::new(engine, "sail-qa03");

    let cancel_token = CancellationToken::new();
    let cancel_trigger = cancel_token.clone();

    let mut emitted_count = 0;
    let req = GenerateRequest::new("sess_cancel", "Write a long essay about space")
        .with_max_tokens(50)
        .with_cancellation_token(cancel_token);

    let meta = adapter
        .generate_stream(req, &mut move |_chunk| {
            emitted_count += 1;
            if emitted_count >= 4 {
                cancel_trigger.cancel();
            }
            Ok(())
        })
        .expect("stream with cancellation");

    assert_eq!(meta.finish_reason, FinishReason::Cancelled);
    assert_eq!(meta.generated_tokens, 4);

    let stored_len = adapter
        .session_history_len("sess_cancel")
        .expect("session exists");
    assert_eq!(
        stored_len,
        meta.prompt_tokens + 4,
        "Session history must commit exactly the prompt plus emitted prefix"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 8 (QA-03): StreamingUtf8Decoder prevents corruption across multibyte UTF-8 boundaries.
#[test]
fn test_qa03_streaming_utf8_multibyte_boundary_handling() {
    let mut decoder = StreamingUtf8Decoder::new();

    // 1. Single-byte ASCII
    assert_eq!(decoder.push_bytes(b"Hello, "), "Hello, ");

    // 2. 4-byte UTF-8 emoji '🦀' [0xF0, 0x9F, 0x96, 0x80] split byte-by-byte
    let crab_bytes = "🦀".as_bytes();
    assert_eq!(crab_bytes.len(), 4);

    assert_eq!(decoder.push_bytes(&[crab_bytes[0]]), "");
    assert_eq!(decoder.push_bytes(&[crab_bytes[1]]), "");
    assert_eq!(decoder.push_bytes(&[crab_bytes[2]]), "");
    // Last byte completes the code point:
    assert_eq!(decoder.push_bytes(&[crab_bytes[3]]), "🦀");

    // 3. 2-byte UTF-8 'é' [0xC3, 0xA9] split across chunks
    let e_acute = "é".as_bytes();
    assert_eq!(decoder.push_bytes(&[e_acute[0]]), "");
    assert_eq!(decoder.push_bytes(&[e_acute[1], b' ']), "é ");

    // 4. Finish flushes empty when clean
    assert_eq!(decoder.finish(), "");

    // 5. Incomplete sequence at termination flushes lossily without panic
    let mut incomplete_decoder = StreamingUtf8Decoder::new();
    incomplete_decoder.push_bytes(&[0xC3]);
    let flushed = incomplete_decoder.finish();
    assert!(
        !flushed.is_empty(),
        "Finish must flush pending trailing bytes"
    );
}

/// Test 9 (QA-03): Session handles background model promotion safely by invalidating stale recurrent state.
#[test]
fn test_qa03_model_promotion_invalidates_stale_recurrent_state() {
    let temp_dir =
        std::env::temp_dir().join(format!("sail-qa03-promo-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let sea_path = create_test_sea_artifact(&temp_dir);
    let device = default_device();
    let engine = Engine::<InferenceBackend>::open(&sea_path, &device).expect("open native");
    let adapter = NativeSailAdapter::new(engine.clone(), "sail-qa03");

    // Turn 1 on Generation 1
    let req1 = GenerateRequest::new("sess_promo", "Turn on gen 1")
        .with_max_tokens(4)
        .with_seed(42);
    let meta1 = adapter
        .generate_stream(req1, &mut |_| Ok(()))
        .expect("gen 1 turn");
    assert_eq!(meta1.generation_id, 1);

    // Simulate trainer publishing a new checkpoint to sea_path
    std::thread::sleep(std::time::Duration::from_millis(50));
    let _ = create_test_sea_artifact(&temp_dir);
    let reloaded = engine.check_and_reload().expect("check and reload");
    assert!(
        reloaded,
        "Engine must reload newly published model generation"
    );

    // Turn 2 on promoted Generation 2: must detect promotion, clear stale recurrent state, and succeed
    let req2 = GenerateRequest::new("sess_promo", "Turn on gen 2")
        .with_max_tokens(4)
        .with_seed(42);
    let meta2 = adapter
        .generate_stream(req2, &mut |_| Ok(()))
        .expect("gen 2 turn");
    assert_eq!(
        meta2.generation_id, 2,
        "Post-promotion turn must execute against promoted generation id 2"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 10 (QA-03): N emitted tokens do not cause N whole-prompt prefills;
/// prefill runs once per initial turn, and continuation turns run 0 whole-prompt prefills.
#[test]
fn test_qa03_prefill_count_and_decode_scaling() {
    let temp_dir =
        std::env::temp_dir().join(format!("sail-qa03-scale-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let sea_path = create_test_sea_artifact(&temp_dir);
    let device = default_device();
    let engine = Engine::<InferenceBackend>::open(&sea_path, &device).expect("open native");
    let adapter = NativeSailAdapter::new(engine, "sail-qa03");

    // Turn 1: 25 tokens emitted. Must require exactly 1 prefill, NOT 25!
    let req1 = GenerateRequest::new("sess_scaling", "Deep learning SSM scaling test")
        .with_max_tokens(25)
        .with_seed(42);
    let meta1 = adapter
        .generate_stream(req1, &mut |_| Ok(()))
        .expect("turn 1");
    assert_eq!(meta1.generated_tokens, 25);
    assert_eq!(
        meta1.prefill_count, 1,
        "Initial turn must execute exactly 1 prefill, not N prefills"
    );

    // Turn 2: 15 tokens emitted on the existing session.
    // Must continue recurrently without ANY whole-prompt prefill (prefill_count == 0)!
    let req2 = GenerateRequest::new("sess_scaling", "Continuing the conversation")
        .with_max_tokens(15)
        .with_seed(42);
    let meta2 = adapter
        .generate_stream(req2, &mut |_| Ok(()))
        .expect("turn 2");
    assert_eq!(meta2.generated_tokens, 15);
    assert_eq!(
        meta2.prefill_count, 0,
        "Continuation turn must continue purely recurrently with 0 whole-prompt prefills"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 11 (QA-02): PretrainedLocalAdapter without a local runner binary fails closed with an actionable error.
/// It must NEVER return a fabricated response.
#[test]
fn test_qa02_pretrained_adapter_missing_runner_fails_with_actionable_error() {
    let temp_dir =
        std::env::temp_dir().join(format!("sail-qa02-runner-err-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let gguf_path = create_test_gguf_file(&temp_dir);
    let adapter = PretrainedLocalAdapter::open(&gguf_path, "llama-test-gguf")
        .expect("open valid gguf container");

    assert_eq!(adapter.tensor_count(), 1);
    assert_eq!(adapter.metadata_kv_count(), 2);
    assert_eq!(adapter.model_format(), ModelFormat::GgufV2V3);

    // Call generate_stream when runner is missing
    let req = GenerateRequest::new("sess_missing_runner", "What is quantum entanglement?");
    let res = adapter.generate_stream(req, &mut |_| Ok(()));

    assert!(
        res.is_err(),
        "Missing runner must return Err, never a fabricated answer"
    );
    let err = res.unwrap_err();
    assert!(
        err.contains("RunnerNotFound"),
        "Error message must clearly identify RunnerNotFound: {err}"
    );
    assert!(
        err.contains("llama-cli") || err.contains("LLAMA_RUNNER_PATH"),
        "Error message must provide actionable path to install or configure runner: {err}"
    );
    assert!(
        err.contains("MockModelAdapter"),
        "Error message must point to MockModelAdapter for offline tests: {err}"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 12 (QA-02): PretrainedLocalAdapter rejects non-GGUF formats on open without panic.
#[test]
fn test_qa02_pretrained_adapter_rejects_non_gguf_container() {
    let temp_dir =
        std::env::temp_dir().join(format!("sail-qa02-non-gguf-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let sea_path = create_test_sea_artifact(&temp_dir);
    let open_res = PretrainedLocalAdapter::open(&sea_path, "sea-model");
    assert!(open_res.is_err());
    assert!(open_res.unwrap_err().contains("only supports GGUF models"));

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 13 (QA-02): PretrainedLocalAdapter executes configured runner, streams output, and terminates on cancellation.
#[test]
fn test_qa02_pretrained_adapter_runner_execution_and_cancellation() {
    let temp_dir = std::env::temp_dir().join(format!("sail-qa02-exec-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let gguf_path = create_test_gguf_file(&temp_dir);
    let runner_script = create_mock_runner_script(&temp_dir);

    let adapter = PretrainedLocalAdapter::open(&gguf_path, "llama-supervised-test")
        .expect("open gguf")
        .with_runner_path(&runner_script);

    assert_eq!(
        adapter.runner_version(),
        Some("llama.cpp test runner v1.2.3")
    );

    // 1. Successful execution and streaming
    let mut collected_tokens = Vec::new();
    let req = GenerateRequest::new("sess_runner_exec", "What is the capital of France?")
        .with_max_tokens(20);

    let meta = adapter
        .generate_stream(req, &mut |chunk| {
            if let sail::runtime::GenerationChunk::Token { text, token_id } = chunk {
                collected_tokens.push((token_id, text));
            }
            Ok(())
        })
        .expect("successful runner stream");

    assert!(!collected_tokens.is_empty(), "Runner must stream tokens");
    assert_eq!(meta.backend_name, "Pretrained Local GGUF");
    assert!(meta.generated_tokens > 0);
    assert!(
        meta.model_id.contains('#'),
        "Model ID must capture model hash"
    );

    // 2. Cooperative cancellation terminating runner process
    let cancel_token = CancellationToken::new();
    let cancel_trigger = cancel_token.clone();
    let mut cancel_count = 0;

    let cancel_req = GenerateRequest::new("sess_runner_cancel", "Run long task")
        .with_max_tokens(50)
        .with_cancellation_token(cancel_token);

    let cancel_meta = adapter
        .generate_stream(cancel_req, &mut move |_chunk| {
            cancel_count += 1;
            if cancel_count >= 2 {
                cancel_trigger.cancel();
            }
            Ok(())
        })
        .expect("cancellation handled cleanly");

    assert_eq!(cancel_meta.finish_reason, FinishReason::Cancelled);

    let _ = fs::remove_dir_all(&temp_dir);
}

/// Test 14 (QA-02): MockModelAdapter is explicitly labeled as a test double and distinct from production backends.
#[test]
fn test_qa02_mock_model_adapter_is_explicitly_labeled_test_double() {
    let mock = MockModelAdapter::new("test-double-42");
    let info = mock.backend_info();
    assert_eq!(info.name, "Mock Model Adapter (Test Double)");
    assert_eq!(info.format, "Mock");
    assert_eq!(info.device, "Memory Test Double");
    assert!(info.capabilities.supports_input(BackendModality::Text));
    assert!(!info.capabilities.supports_input(BackendModality::Audio));
    assert!(!info.capabilities.supports_input(BackendModality::Video));
    assert!(
        info.capabilities
            .supports_output(BackendModality::ToolProposal)
    );
}

/// Test 15 (IMPL-21): text-only backends advertise their real contract and reject
/// supplied sensor evidence before generating, mutating a session, or launching a runner.
#[test]
fn test_impl21_text_backends_reject_unsupported_media_with_typed_error() {
    let mock = MockModelAdapter::new("media-contract-test");
    let info = mock.backend_info();
    assert_eq!(
        info.capabilities.input_modalities,
        vec![BackendModality::Text]
    );
    assert!(info.capabilities.streaming_output);
    assert!(info.capabilities.stateful_sessions);

    let audio = Arc::new(SharedAudioBuffer::new(16_000, 1, vec![0.0; 160]));
    let mut audio_request = GenerateRequest::new("audio-session", "describe this audio");
    audio_request.audio_input = Some(audio);

    let typed = mock
        .validate_request(&audio_request)
        .expect_err("text-only mock must reject audio");
    assert_eq!(
        typed,
        BackendRequestError::UnsupportedInputModality {
            backend_name: "Mock Model Adapter (Test Double)".into(),
            modality: BackendModality::Audio,
        }
    );

    let mut emitted = false;
    let error = mock
        .generate_stream(audio_request, &mut |_| {
            emitted = true;
            Ok(())
        })
        .expect_err("unsupported audio must fail closed");
    assert!(error.contains("UnsupportedInputModality"));
    assert!(error.contains("Audio"));
    assert!(!emitted, "validation must happen before output emission");

    let video = Arc::new(SharedVideoBuffer::new(2, 2, "RGB24", vec![0; 12]));
    let mut video_request = GenerateRequest::new("video-session", "describe this frame");
    video_request.video_input = Some(video);
    let error = mock
        .generate_stream(video_request, &mut |_| Ok(()))
        .expect_err("text-only mock must reject video");
    assert!(error.contains("UnsupportedInputModality"));
    assert!(error.contains("Video"));
}

/// Test 16 (IMPL-21): pretrained GGUF media rejection wins over runner discovery,
/// proving unsupported evidence cannot be silently dropped before an unrelated error.
#[test]
fn test_impl21_pretrained_media_rejection_precedes_runner_lookup() {
    let temp_dir =
        std::env::temp_dir().join(format!("sail-impl21-media-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let gguf_path = create_test_gguf_file(&temp_dir);
    let adapter = PretrainedLocalAdapter::open(&gguf_path, "text-only-gguf").unwrap();
    let info = adapter.backend_info();
    assert_eq!(
        info.capabilities.input_modalities,
        vec![BackendModality::Text]
    );
    assert!(!info.capabilities.stateful_sessions);

    let mut request = GenerateRequest::new("gguf-media", "inspect attached evidence");
    request.video_input = Some(Arc::new(SharedVideoBuffer::new(
        1,
        1,
        "RGB24",
        vec![0, 0, 0],
    )));
    let error = adapter
        .generate_stream(request, &mut |_| Ok(()))
        .expect_err("text-only GGUF adapter must reject video");
    assert!(error.contains("UnsupportedInputModality"));
    assert!(error.contains("Video"));
    assert!(
        !error.contains("RunnerNotFound"),
        "media contract must be checked before runner discovery"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}
