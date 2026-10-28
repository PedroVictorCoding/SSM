use std::sync::{Arc, Mutex};

use sail::runtime::{
    MockModelAdapter, SegmentedAssistantCoordinator, SegmentedAssistantHost, SegmentedDegradation,
    SegmentedHostEvent, TimelineEntry,
};
use sail::voice::asr::{StreamingAsrAdapter, TranscriptSegment};
use sail::voice::{AudioChunk, FormantNeuralTtsEngine, PlaybackPipeline};

struct ScriptedAsr {
    final_text: Option<String>,
    partial_sent: Mutex<bool>,
}

impl StreamingAsrAdapter for ScriptedAsr {
    fn reset(&self, _: &str) {
        *self.partial_sent.lock().unwrap() = false;
    }

    fn feed_audio(
        &self,
        session_id: &str,
        chunk: &AudioChunk,
    ) -> Result<Option<TranscriptSegment>, String> {
        let mut sent = self.partial_sent.lock().unwrap();
        if *sent {
            return Ok(None);
        }
        *sent = true;
        Ok(Some(TranscriptSegment {
            utterance_id: "utt-host".into(),
            session_id: session_id.into(),
            start_offset: chunk.sample_offset,
            end_offset: chunk.sample_offset + chunk.samples.len() as u64,
            start_timestamp_ns: chunk.timestamp_ns,
            end_timestamp_ns: chunk.timestamp_ns + 10_000_000,
            text: "what is".into(),
            is_final: false,
            confidence: 0.5,
        }))
    }

    fn finalize_utterance(&self, session_id: &str) -> Result<Option<TranscriptSegment>, String> {
        Ok(self.final_text.as_ref().map(|text| TranscriptSegment {
            utterance_id: "utt-host".into(),
            session_id: session_id.into(),
            start_offset: 0,
            end_offset: 320,
            start_timestamp_ns: 1_000_000,
            end_timestamp_ns: 21_000_000,
            text: text.clone(),
            is_final: true,
            confidence: 0.9,
        }))
    }
}

fn host(final_text: Option<&str>) -> (SegmentedAssistantHost, Arc<PlaybackPipeline>) {
    let backend = Arc::new(
        MockModelAdapter::new("host-mock")
            .with_mock_response("what is this error", "It is a compiler error."),
    );
    let coordinator = Arc::new(SegmentedAssistantCoordinator::new(
        "session-host",
        backend,
        Some(Box::new(FormantNeuralTtsEngine::new(16_000))),
    ));
    let playback = Arc::new(PlaybackPipeline::new(100_000, 100_000));
    let asr = Box::new(ScriptedAsr {
        final_text: final_text.map(str::to_string),
        partial_sent: Mutex::new(false),
    });
    (
        SegmentedAssistantHost::new("session-host", asr, coordinator, Some(playback.clone())),
        playback,
    )
}

#[test]
fn partial_audio_is_observable_but_only_final_asr_creates_turn_and_playback() {
    let (host, playback) = host(Some("what is this error"));
    let audio = AudioChunk::new(vec![0.1; 320], 16_000, 1, 0, 1_000_000);

    let partial = host.feed_audio(&audio, false).unwrap().unwrap();
    assert!(matches!(partial, SegmentedHostEvent::TranscriptObserved(ref s) if !s.is_final));
    assert_eq!(playback.queued_chunks(), 0);

    let final_event = host.finalize_utterance(false).unwrap();
    let SegmentedHostEvent::TurnCompleted(outcome) = final_event else {
        panic!("expected completed turn");
    };
    assert!(
        outcome
            .degradations
            .contains(&SegmentedDegradation::NoVisualContext)
    );
    assert!(!outcome.synthesized_audio.is_empty());
    assert!(playback.queued_chunks() > 0);
    assert!(outcome.tool_proposals.is_empty());
}

#[test]
fn missing_asr_result_is_explicit_and_never_fabricated() {
    let (host, playback) = host(None);
    assert!(matches!(
        host.finalize_utterance(false).unwrap(),
        SegmentedHostEvent::NoSpeechRecognized
    ));
    assert_eq!(playback.queued_chunks(), 0);
}

#[test]
fn non_visual_timeline_entries_do_not_claim_screen_context() {
    let (host, _) = host(Some("what is this error"));
    host.push_timeline(TimelineEntry::PointerLocation {
        timestamp_ms: 20,
        x: 10,
        y: 20,
    });
    let event = host.finalize_utterance(false).unwrap();
    let SegmentedHostEvent::TurnCompleted(outcome) = event else {
        panic!("expected completed turn");
    };
    assert!(
        outcome
            .degradations
            .contains(&SegmentedDegradation::NoVisualContext)
    );
}
