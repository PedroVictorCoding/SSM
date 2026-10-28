//! Host-facing bridge for the stage-one segmented assistant.
//!
//! Audio recognition, screen/UI evidence, semantic inference, and playback remain
//! replaceable components. Only finalized ASR text can initiate a model turn.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::runtime::multimodal_context::{MultimodalContextAssembler, TimelineEntry};
use crate::runtime::segmented::{
    SegmentedAssistantCoordinator, SegmentedAssistantError, SegmentedCancellationOutcome,
    SegmentedDegradation, SegmentedTurnOutcome,
};
use crate::voice::asr::{StreamingAsrAdapter, TranscriptSegment};
use crate::voice::{AudioChunk, PlaybackPipeline};

#[derive(Clone, Debug)]
pub enum SegmentedHostEvent {
    TranscriptObserved(TranscriptSegment),
    TurnCompleted(SegmentedTurnOutcome),
    NoSpeechRecognized,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentedHostError {
    Asr(String),
    Context(String),
    Assistant(SegmentedAssistantError),
}

impl fmt::Display for SegmentedHostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asr(message) => write!(f, "ASR failed: {message}"),
            Self::Context(message) => write!(f, "context assembly failed: {message}"),
            Self::Assistant(error) => write!(f, "assistant turn failed: {error}"),
        }
    }
}

impl std::error::Error for SegmentedHostError {}

/// Connects a real ASR adapter and timestamped evidence timeline to the segmented
/// assistant coordinator and optional playback queue.
pub struct SegmentedAssistantHost {
    session_id: String,
    asr: Box<dyn StreamingAsrAdapter>,
    coordinator: Arc<SegmentedAssistantCoordinator>,
    context: Mutex<MultimodalContextAssembler>,
    playback: Option<Arc<PlaybackPipeline>>,
    has_visual_evidence: AtomicBool,
}

impl SegmentedAssistantHost {
    pub fn new(
        session_id: impl Into<String>,
        asr: Box<dyn StreamingAsrAdapter>,
        coordinator: Arc<SegmentedAssistantCoordinator>,
        playback: Option<Arc<PlaybackPipeline>>,
    ) -> Self {
        let session_id = session_id.into();
        Self {
            context: Mutex::new(MultimodalContextAssembler::new(&session_id)),
            session_id,
            asr,
            coordinator,
            playback,
            has_visual_evidence: AtomicBool::new(false),
        }
    }

    pub fn push_timeline(&self, entry: TimelineEntry) {
        if matches!(
            &entry,
            TimelineEntry::ScreenFrame { .. } | TimelineEntry::UiTreeSnapshot { .. }
        ) {
            self.has_visual_evidence.store(true, Ordering::Release);
        }
        self.context
            .lock()
            .expect("context mutex poisoned")
            .push_entry(entry);
    }

    /// Feeds one timestamped audio chunk. Partial hypotheses are observable but
    /// cannot reach the semantic backend.
    pub fn feed_audio(
        &self,
        chunk: &AudioChunk,
        consequential: bool,
    ) -> Result<Option<SegmentedHostEvent>, SegmentedHostError> {
        let segment = self
            .asr
            .feed_audio(&self.session_id, chunk)
            .map_err(SegmentedHostError::Asr)?;
        match segment {
            Some(segment) if segment.is_final => {
                self.route_final_segment(segment, consequential).map(Some)
            }
            Some(segment) => Ok(Some(SegmentedHostEvent::TranscriptObserved(segment))),
            None => Ok(None),
        }
    }

    /// Finalizes the recognizer's current utterance. An empty recognizer result is
    /// explicit and never replaced with guessed text.
    pub fn finalize_utterance(
        &self,
        consequential: bool,
    ) -> Result<SegmentedHostEvent, SegmentedHostError> {
        match self
            .asr
            .finalize_utterance(&self.session_id)
            .map_err(SegmentedHostError::Asr)?
        {
            Some(segment) if segment.is_final => self.route_final_segment(segment, consequential),
            Some(segment) => Ok(SegmentedHostEvent::TranscriptObserved(segment)),
            None => Ok(SegmentedHostEvent::NoSpeechRecognized),
        }
    }

    /// Stops queued playback first, then propagates the exact playback cursor to
    /// model/TTS cancellation.
    pub fn barge_in(&self) -> Option<SegmentedCancellationOutcome> {
        let cursor = self
            .playback
            .as_ref()
            .map(|pipeline| pipeline.interrupt())
            .unwrap_or(0);
        self.coordinator.cancel_active(cursor)
    }

    pub fn reset_asr(&self) {
        self.asr.reset(&self.session_id);
    }

    fn route_final_segment(
        &self,
        segment: TranscriptSegment,
        consequential: bool,
    ) -> Result<SegmentedHostEvent, SegmentedHostError> {
        let context = if self.has_visual_evidence.load(Ordering::Acquire) {
            Some(
                self.context
                    .lock()
                    .expect("context mutex poisoned")
                    .assemble_synchronized_context(
                        &segment.text,
                        segment.end_timestamp_ns / 1_000_000,
                        consequential,
                    )
                    .map_err(SegmentedHostError::Context)?,
            )
        } else {
            None
        };

        let mut outcome = self
            .coordinator
            .process_transcript(segment, context.as_ref())
            .map_err(SegmentedHostError::Assistant)?;

        if !outcome.synthesized_audio.is_empty() {
            match &self.playback {
                Some(playback) => {
                    for audio in &outcome.synthesized_audio {
                        if audio.samples.is_empty() {
                            continue;
                        }
                        let chunk = AudioChunk::new(
                            audio.samples.to_vec(),
                            audio.sample_rate,
                            audio.channels,
                            audio.sample_offset,
                            0,
                        );
                        if let Err(error) = playback.enqueue_chunk(chunk) {
                            outcome
                                .degradations
                                .push(SegmentedDegradation::SpeechOutputFailed(format!(
                                    "playback queue rejected synthesized audio: {error}"
                                )));
                            break;
                        }
                    }
                }
                None => outcome
                    .degradations
                    .push(SegmentedDegradation::SpeechOutputFailed(
                        "playback pipeline unavailable; synthesized audio was not played".into(),
                    )),
            }
        }

        Ok(SegmentedHostEvent::TurnCompleted(outcome))
    }
}
