//! Stage-one segmented assistant orchestration for the staged-hybrid architecture.
//!
//! This module joins stable ASR, synchronized screen/UI context, an interchangeable
//! semantic backend, and interruptible TTS. It intentionally does not execute tool
//! proposals and has no dependency on training or replay code.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::runtime::adapter::{
    FinishReason, GenerateRequest, GenerationChunk, GenerationMetadata, ModelBackendAdapter,
    ToolProposal,
};
use crate::runtime::bus::CancellationToken;
use crate::runtime::multimodal_context::SynchronizedContextPacket;
use crate::voice::asr::{AsrIntentGate, TranscriptSegment};
use crate::voice::tts::{StreamingTtsAdapter, SynthesizedAudioChunk, VoiceConfig};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentedDegradation {
    NoVisualContext,
    SpeechOutputUnavailable,
    SpeechOutputFailed(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentedTurnDisposition {
    IgnoredPartialTranscript,
    Completed,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct SegmentedTurnOutcome {
    pub disposition: SegmentedTurnDisposition,
    pub utterance_id: String,
    pub context_id: Option<String>,
    pub response_text: String,
    /// Proposals are returned to the caller for policy review. This coordinator
    /// never dispatches them.
    pub tool_proposals: Vec<ToolProposal>,
    pub synthesized_chunks: usize,
    pub synthesized_samples: u64,
    /// Bounded-by-turn audio ready for the host playback pipeline. The coordinator
    /// never silently discards successfully synthesized speech.
    pub synthesized_audio: Vec<SynthesizedAudioChunk>,
    pub degradations: Vec<SegmentedDegradation>,
    pub generation: Option<GenerationMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentedAssistantError {
    SessionMismatch { transcript: String, context: String },
    ContextPromptMismatch,
    Backend(String),
    CoordinatorBusy,
}

impl fmt::Display for SegmentedAssistantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionMismatch {
                transcript,
                context,
            } => write!(
                f,
                "transcript session '{transcript}' does not match context session '{context}'"
            ),
            Self::ContextPromptMismatch => {
                write!(f, "synchronized context does not belong to this transcript")
            }
            Self::Backend(message) => write!(f, "semantic backend failed: {message}"),
            Self::CoordinatorBusy => write!(f, "another segmented assistant turn is active"),
        }
    }
}

impl std::error::Error for SegmentedAssistantError {}

#[derive(Clone)]
struct ActiveTurn {
    utterance_id: String,
    cancellation: CancellationToken,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentedCancellationOutcome {
    pub utterance_id: String,
    pub model_cancelled: bool,
    pub speech_cancelled: bool,
    pub discarded_samples: u64,
}

/// Coordinates the first, segmented product stage. Perception remains specialized;
/// the semantic backend receives a provenance-bearing textual context packet.
pub struct SegmentedAssistantCoordinator {
    session_id: String,
    backend: Arc<dyn ModelBackendAdapter>,
    tts: Mutex<Option<Box<dyn StreamingTtsAdapter>>>,
    asr_gate: Mutex<AsrIntentGate>,
    active_turn: Mutex<Option<ActiveTurn>>,
    turn_counter: AtomicU64,
    voice: VoiceConfig,
}

impl SegmentedAssistantCoordinator {
    pub fn new(
        session_id: impl Into<String>,
        backend: Arc<dyn ModelBackendAdapter>,
        tts: Option<Box<dyn StreamingTtsAdapter>>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            backend,
            tts: Mutex::new(tts),
            asr_gate: Mutex::new(AsrIntentGate::new()),
            active_turn: Mutex::new(None),
            turn_counter: AtomicU64::new(1),
            voice: VoiceConfig::default(),
        }
    }

    pub fn with_voice_config(mut self, voice: VoiceConfig) -> Self {
        self.voice = voice;
        self
    }

    /// Accepts only finalized ASR segments. Optional context is an explicit degraded
    /// mode, not an invented visual observation.
    pub fn process_transcript(
        &self,
        segment: TranscriptSegment,
        context: Option<&SynchronizedContextPacket>,
    ) -> Result<SegmentedTurnOutcome, SegmentedAssistantError> {
        let utterance_id = segment.utterance_id.clone();
        if !segment.is_final {
            self.asr_gate
                .lock()
                .expect("ASR gate mutex poisoned")
                .ingest_segment(segment);
            return Ok(SegmentedTurnOutcome {
                disposition: SegmentedTurnDisposition::IgnoredPartialTranscript,
                utterance_id,
                context_id: None,
                response_text: String::new(),
                tool_proposals: Vec::new(),
                synthesized_chunks: 0,
                synthesized_samples: 0,
                synthesized_audio: Vec::new(),
                degradations: Vec::new(),
                generation: None,
            });
        }

        let mut degradations = Vec::new();
        let (prompt, context_id) = match context {
            Some(packet) => {
                if packet.provenance_trace.session_id != segment.session_id {
                    return Err(SegmentedAssistantError::SessionMismatch {
                        transcript: segment.session_id,
                        context: packet.provenance_trace.session_id.clone(),
                    });
                }
                if packet.prompt_text.trim() != segment.text.trim() {
                    return Err(SegmentedAssistantError::ContextPromptMismatch);
                }
                (
                    packet.formatted_prompt.clone(),
                    Some(packet.context_id.clone()),
                )
            }
            None => {
                degradations.push(SegmentedDegradation::NoVisualContext);
                (
                    format!(
                        "--- SEGMENTED INPUT ---\nFinal user transcript: {}\nVisual context: unavailable\n--- END INPUT ---",
                        segment.text
                    ),
                    None,
                )
            }
        };

        let turn_number = self.turn_counter.fetch_add(1, Ordering::Relaxed);
        let response_id = format!("seg_{}_{}", turn_number, segment.utterance_id);
        let cancellation = CancellationToken::new();
        {
            let mut active = self.active_turn.lock().expect("active turn mutex poisoned");
            if active.is_some() {
                return Err(SegmentedAssistantError::CoordinatorBusy);
            }
            let committed = self
                .asr_gate
                .lock()
                .expect("ASR gate mutex poisoned")
                .ingest_segment(segment.clone());
            if committed.is_none() {
                return Ok(SegmentedTurnOutcome {
                    disposition: SegmentedTurnDisposition::IgnoredPartialTranscript,
                    utterance_id,
                    context_id: None,
                    response_text: String::new(),
                    tool_proposals: Vec::new(),
                    synthesized_chunks: 0,
                    synthesized_samples: 0,
                    synthesized_audio: Vec::new(),
                    degradations: Vec::new(),
                    generation: None,
                });
            }
            *active = Some(ActiveTurn {
                utterance_id: response_id.clone(),
                cancellation: cancellation.clone(),
            });
        }

        let mut speech_enabled = false;
        match self.tts.lock().expect("TTS mutex poisoned").as_mut() {
            Some(tts) => match tts.start_utterance(&response_id, &self.voice) {
                Ok(()) => speech_enabled = true,
                Err(error) => degradations.push(SegmentedDegradation::SpeechOutputFailed(error)),
            },
            None => degradations.push(SegmentedDegradation::SpeechOutputUnavailable),
        }

        let mut response_text = String::new();
        let mut proposals = Vec::new();
        let mut synthesized_chunks = 0usize;
        let mut synthesized_samples = 0u64;
        let mut synthesized_audio = Vec::new();
        let mut clause_buffer = String::new();
        let request = GenerateRequest::new(&self.session_id, prompt)
            .with_cancellation_token(cancellation.clone());

        let generation_result = self.backend.generate_stream(request, &mut |chunk| {
            match chunk {
                GenerationChunk::Token { text, .. } => {
                    response_text.push_str(&text);
                    if speech_enabled {
                        clause_buffer.push_str(&text);
                        while let Some(clause) = crate::voice::tts::extract_speech_clause(&mut clause_buffer) {
                            if cancellation.is_cancelled() {
                                break;
                            }
                            let result = self
                                .tts
                                .lock()
                                .map_err(|_| "TTS mutex poisoned".to_string())?
                                .as_mut()
                                .ok_or_else(|| "TTS backend disappeared".to_string())?
                                .feed_text(&response_id, &format!("{clause} "));
                            match result {
                                Ok(chunks) => {
                                    synthesized_chunks += chunks.len();
                                    synthesized_samples += chunks
                                        .iter()
                                        .map(|audio| audio.samples.len() as u64)
                                        .sum::<u64>();
                                    synthesized_audio.extend(chunks);
                                }
                                Err(error) => {
                                    degradations.push(SegmentedDegradation::SpeechOutputFailed(error));
                                    speech_enabled = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                GenerationChunk::ToolCall(proposal) => proposals.push(proposal),
                GenerationChunk::Complete(_) => {}
            }
            Ok(())
        });

        let generation = match generation_result {
            Ok(metadata) => metadata,
            Err(error) => {
                self.clear_active(&response_id);
                return Err(SegmentedAssistantError::Backend(error));
            }
        };

        if speech_enabled && !cancellation.is_cancelled() {
            let mut tts_guard = self.tts.lock().expect("TTS mutex poisoned");
            if let Some(tts) = tts_guard.as_mut() {
                let remaining = clause_buffer.trim();
                if !remaining.is_empty() {
                    match tts.feed_text(&response_id, &format!("{remaining} ")) {
                        Ok(chunks) => {
                            synthesized_chunks += chunks.len();
                            synthesized_samples += chunks
                                .iter()
                                .map(|audio| audio.samples.len() as u64)
                                .sum::<u64>();
                            synthesized_audio.extend(chunks);
                        }
                        Err(error) => {
                            degradations.push(SegmentedDegradation::SpeechOutputFailed(error));
                        }
                    }
                }
                match tts.finalize_utterance(&response_id) {
                    Ok(chunks) => {
                        synthesized_chunks += chunks.len();
                        synthesized_samples += chunks
                            .iter()
                            .map(|audio| audio.samples.len() as u64)
                            .sum::<u64>();
                        synthesized_audio.extend(chunks);
                    }
                    Err(error) => degradations.push(SegmentedDegradation::SpeechOutputFailed(error)),
                }
            }
        }

        let disposition =
            if cancellation.is_cancelled() || generation.finish_reason == FinishReason::Cancelled {
                SegmentedTurnDisposition::Cancelled
            } else {
                SegmentedTurnDisposition::Completed
            };
        self.clear_active(&response_id);

        Ok(SegmentedTurnOutcome {
            disposition,
            utterance_id: segment.utterance_id,
            context_id,
            response_text,
            tool_proposals: proposals,
            synthesized_chunks,
            synthesized_samples,
            synthesized_audio,
            degradations,
            generation: Some(generation),
        })
    }

    /// Cancels active semantic generation and pending speech. Tool proposals are
    /// never executed here, so there are no hidden side effects to roll back.
    pub fn cancel_active(&self, playback_cursor: u64) -> Option<SegmentedCancellationOutcome> {
        let active = self
            .active_turn
            .lock()
            .expect("active turn mutex poisoned")
            .clone()?;
        active.cancellation.cancel();

        let (speech_cancelled, discarded_samples) =
            match self.tts.lock().expect("TTS mutex poisoned").as_mut() {
                Some(tts) => match tts.cancel_utterance(&active.utterance_id, playback_cursor) {
                    Ok(report) => (true, report.discarded_samples),
                    Err(_) => (false, 0),
                },
                None => (false, 0),
            };

        Some(SegmentedCancellationOutcome {
            utterance_id: active.utterance_id,
            model_cancelled: true,
            speech_cancelled,
            discarded_samples,
        })
    }

    pub fn has_active_turn(&self) -> bool {
        self.active_turn
            .lock()
            .expect("active turn mutex poisoned")
            .is_some()
    }

    fn clear_active(&self, response_id: &str) {
        let mut active = self.active_turn.lock().expect("active turn mutex poisoned");
        if active
            .as_ref()
            .is_some_and(|turn| turn.utterance_id == response_id)
        {
            *active = None;
        }
    }
}
