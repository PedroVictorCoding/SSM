use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use sail::runtime::adapter::{
    BackendCapabilities, BackendInfo, FinishReason, GenerateRequest, GenerationChunk,
    GenerationMetadata, MockModelAdapter, ModelBackendAdapter,
};
use sail::runtime::multimodal_context::{
    ContextProvenanceTrace, MultimodalContextAssembler, SynchronizedContextPacket,
};
use sail::runtime::segmented::{
    SegmentedAssistantCoordinator, SegmentedAssistantError, SegmentedDegradation,
    SegmentedTurnDisposition,
};
use sail::voice::TranscriptSegment;
use sail::voice::tts::{
    PlaybackAlignmentStatus, StreamingTtsAdapter, SynthesizedAudioChunk, TtsCancellationReport,
    VoiceConfig,
};

fn transcript(id: &str, text: &str, is_final: bool) -> TranscriptSegment {
    TranscriptSegment {
        utterance_id: id.into(),
        session_id: "session-1".into(),
        start_offset: 0,
        end_offset: 160,
        start_timestamp_ns: 1_000_000,
        end_timestamp_ns: 11_000_000,
        text: text.into(),
        is_final,
        confidence: 0.9,
    }
}

fn context(text: &str) -> SynchronizedContextPacket {
    SynchronizedContextPacket {
        context_id: "ctx-1".into(),
        prompt_text: text.into(),
        resolved_reference: None,
        formatted_prompt: format!(
            "trusted user transcript: {text}\nuntrusted screen evidence: compiler error"
        ),
        visual_tokens: Vec::new(),
        provenance_trace: ContextProvenanceTrace {
            session_id: "session-1".into(),
            utterance_timestamp_ms: 11,
            active_window_id: None,
            selected_evidence_ids: vec!["frame-1".into()],
            evidence_freshness_ms: Default::default(),
            screen_age_ms: 5.0,
            target_screen_age_target_ms: 1000.0,
            causal_mask_applied: true,
        },
        assembly_latency_ms: 0.1,
    }
}

#[derive(Default)]
struct RecordingTts {
    active: Option<String>,
    text: String,
}

impl StreamingTtsAdapter for RecordingTts {
    fn name(&self) -> &'static str {
        "recording-tts"
    }

    fn start_utterance(&mut self, id: &str, _: &VoiceConfig) -> Result<(), String> {
        self.active = Some(id.into());
        Ok(())
    }

    fn feed_text(&mut self, id: &str, text: &str) -> Result<Vec<SynthesizedAudioChunk>, String> {
        if self.active.as_deref() != Some(id) {
            return Err("wrong utterance".into());
        }
        self.text.push_str(text);
        Ok(Vec::new())
    }

    fn finalize_utterance(&mut self, _: &str) -> Result<Vec<SynthesizedAudioChunk>, String> {
        Ok(Vec::new())
    }

    fn cancel_utterance(&mut self, id: &str, cursor: u64) -> Result<TtsCancellationReport, String> {
        Ok(TtsCancellationReport {
            utterance_id: id.into(),
            total_synthesized_samples: 100,
            played_samples: cursor.min(100),
            discarded_samples: 100 - cursor.min(100),
            committed_words: Vec::new(),
            played_words: Vec::new(),
            interrupted_at_ms: 0.0,
        })
    }

    fn get_playback_status(&self, id: &str, _: u64) -> Result<PlaybackAlignmentStatus, String> {
        Ok(PlaybackAlignmentStatus {
            utterance_id: id.into(),
            played_words: Vec::new(),
            current_word: None,
            pending_words: Vec::new(),
            played_duration_ms: 0.0,
            total_duration_ms: 0.0,
            is_completed: false,
        })
    }
}

struct CountingBackend {
    calls: AtomicUsize,
}

impl ModelBackendAdapter for CountingBackend {
    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "counting".into(),
            model_id: "test".into(),
            format: "mock".into(),
            context_limit: 1,
            precision: "none".into(),
            device: "cpu".into(),
            capabilities: BackendCapabilities::text_backend(false),
        }
    }
    fn reset_session(&self, _: &str) -> Result<(), String> {
        Ok(())
    }
    fn generate_stream(
        &self,
        _: GenerateRequest,
        _: &mut dyn FnMut(GenerationChunk) -> Result<(), String>,
    ) -> Result<GenerationMetadata, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        unreachable!("partial transcripts must not reach the backend")
    }
}

#[test]
fn partial_asr_never_reaches_semantic_backend() {
    let backend = Arc::new(CountingBackend {
        calls: AtomicUsize::new(0),
    });
    let coordinator = SegmentedAssistantCoordinator::new("session-1", backend.clone(), None);
    let outcome = coordinator
        .process_transcript(transcript("u1", "open", false), None)
        .unwrap();
    assert_eq!(
        outcome.disposition,
        SegmentedTurnDisposition::IgnoredPartialTranscript
    );
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn stable_transcript_context_model_and_tts_form_one_turn_without_executing_tools() {
    let backend = Arc::new(MockModelAdapter::new("mock").with_mock_response(
        "compiler error",
        "I found it <tool_call>run_test(\"one\")</tool_call>",
    ));
    let coordinator = SegmentedAssistantCoordinator::new(
        "session-1",
        backend,
        Some(Box::new(RecordingTts::default())),
    );
    let outcome = coordinator
        .process_transcript(
            transcript("u2", "what is this error?", true),
            Some(&context("what is this error?")),
        )
        .unwrap();

    assert_eq!(outcome.disposition, SegmentedTurnDisposition::Completed);
    assert_eq!(outcome.context_id.as_deref(), Some("ctx-1"));
    assert_eq!(outcome.tool_proposals.len(), 1);
    assert_eq!(outcome.tool_proposals[0].tool_name, "run_test");
    assert!(outcome.generation.is_some());
}

#[test]
fn absent_screen_worker_is_an_explicit_degraded_mode() {
    let backend = Arc::new(MockModelAdapter::new("mock").with_mock_response("hello", "hello back"));
    let coordinator = SegmentedAssistantCoordinator::new("session-1", backend, None);
    let outcome = coordinator
        .process_transcript(transcript("u3", "hello", true), None)
        .unwrap();
    assert!(
        outcome
            .degradations
            .contains(&SegmentedDegradation::NoVisualContext)
    );
    assert!(
        outcome
            .degradations
            .contains(&SegmentedDegradation::SpeechOutputUnavailable)
    );
    assert_eq!(outcome.disposition, SegmentedTurnDisposition::Completed);
}

#[test]
fn real_context_assembler_connects_stable_asr_to_segmented_turn() {
    let assembler = MultimodalContextAssembler::new("session-1");
    let packet = assembler
        .assemble_synchronized_context("summarize the screen", 11, false)
        .expect("assemble context");
    let backend = Arc::new(
        MockModelAdapter::new("mock")
            .with_mock_response("summarize the screen", "No screen target is available."),
    );
    let coordinator = SegmentedAssistantCoordinator::new(
        "session-1",
        backend,
        Some(Box::new(RecordingTts::default())),
    );

    let outcome = coordinator
        .process_transcript(
            transcript("u-context", "summarize the screen", true),
            Some(&packet),
        )
        .expect("segmented turn");
    assert_eq!(
        outcome.context_id.as_deref(),
        Some(packet.context_id.as_str())
    );
    assert_eq!(outcome.disposition, SegmentedTurnDisposition::Completed);
}

#[test]
fn mismatched_context_is_rejected_instead_of_becoming_evidence() {
    let backend = Arc::new(MockModelAdapter::new("mock"));
    let coordinator = SegmentedAssistantCoordinator::new("session-1", backend, None);
    let error = coordinator
        .process_transcript(
            transcript("u-mismatch", "actual words", true),
            Some(&context("different words")),
        )
        .unwrap_err();
    assert_eq!(error, SegmentedAssistantError::ContextPromptMismatch);

    let corrected = coordinator
        .process_transcript(
            transcript("u-mismatch", "actual words", true),
            Some(&context("actual words")),
        )
        .expect("a rejected context must not consume the finalized utterance");
    assert_eq!(corrected.disposition, SegmentedTurnDisposition::Completed);
}

struct SlowBackend {
    started: Arc<AtomicUsize>,
}

impl ModelBackendAdapter for SlowBackend {
    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "slow".into(),
            model_id: "test".into(),
            format: "mock".into(),
            context_limit: 100,
            precision: "none".into(),
            device: "cpu".into(),
            capabilities: BackendCapabilities::text_backend(false),
        }
    }
    fn reset_session(&self, _: &str) -> Result<(), String> {
        Ok(())
    }
    fn generate_stream(
        &self,
        request: GenerateRequest,
        sender: &mut dyn FnMut(GenerationChunk) -> Result<(), String>,
    ) -> Result<GenerationMetadata, String> {
        self.started.store(1, Ordering::SeqCst);
        let start = Instant::now();
        let mut emitted = 0;
        for id in 0..200 {
            if request.cancellation_token.is_cancelled() {
                break;
            }
            sender(GenerationChunk::Token {
                text: " word".into(),
                token_id: id,
            })?;
            emitted += 1;
            thread::sleep(Duration::from_millis(2));
        }
        let reason = if request.cancellation_token.is_cancelled() {
            FinishReason::Cancelled
        } else {
            FinishReason::Stop
        };
        let meta = GenerationMetadata {
            backend_name: "slow".into(),
            model_id: "test".into(),
            generation_id: 1,
            prompt_tokens: 1,
            generated_tokens: emitted,
            prefill_count: 1,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            tokens_per_sec: 0.0,
            finish_reason: reason,
        };
        sender(GenerationChunk::Complete(meta.clone()))?;
        Ok(meta)
    }
}

#[test]
fn barge_in_cancels_generation_and_pending_speech() {
    let started = Arc::new(AtomicUsize::new(0));
    let coordinator = Arc::new(SegmentedAssistantCoordinator::new(
        "session-1",
        Arc::new(SlowBackend {
            started: started.clone(),
        }),
        Some(Box::new(RecordingTts::default())),
    ));
    let worker = {
        let coordinator = coordinator.clone();
        thread::spawn(move || {
            coordinator
                .process_transcript(transcript("u4", "stop", true), None)
                .unwrap()
        })
    };
    while started.load(Ordering::SeqCst) == 0 {
        thread::yield_now();
    }
    let cancellation = coordinator.cancel_active(25).expect("active turn");
    assert!(cancellation.model_cancelled);
    assert!(cancellation.speech_cancelled);
    let outcome = worker.join().unwrap();
    assert_eq!(outcome.disposition, SegmentedTurnDisposition::Cancelled);
    assert!(!coordinator.has_active_turn());
}
