//! Full-duplex conversational dialogue coordinator, barge-in, and cancellation semantics.
//!
//! Provides:
//! - Explicit lifecycle states: `Idle`, `Listening`, `Interpreting`, `Speaking`, `AwaitingTool`, `Interrupted`.
//! - Continuous microphone capture during speech playback to detect immediate user interruptions.
//! - Causal response tracking (`CausalResponseId`): late or stale model generation chunks are rejected by ID.
//! - Atomic mid-speech barge-in: flushes audio buffers, stops pending TTS, cancels generation tokens,
//!   and preserves completed tool side effects without rollbacks.
//! - Accurate tool cancellation tracking (`NotStarted`, `Running`, `Completed`, `Cancelled`, `UncertainOutcome`).
//! - Committed dialogue history containing ONLY words confirmed played through speaker hardware.
//! - Interrupt-to-playback-stop latency tracking with p95 benchmark reporting.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::capability::CapabilityPolicy;
use crate::runtime::adapter::{
    FinishReason, GenerateRequest, GenerationChunk, ModelBackendAdapter, ToolProposal,
};
use crate::runtime::bus::CancellationToken;
use crate::runtime::resource::ResourceScheduler;
use crate::tool::ToolRegistry;
use crate::voice::dsp::{
    EndpointEvent, MicrophoneMode, SpeechEndpointConfig, SpeechEndpointDetector,
};
use crate::voice::tts::{FormantNeuralTtsEngine, StreamingTtsAdapter, VoiceConfig};
use crate::voice::{AudioChunk, PlaybackPipeline};

// ---------------------------------------------------------------------------
// Dialogue States & Transition Events
// ---------------------------------------------------------------------------

/// Formal conversation state machine states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DialogueState {
    Idle,
    Listening,
    Interpreting,
    Speaking,
    AwaitingTool,
    Interrupted,
}

/// Lifecycle status for tool invocations across interruption boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolExecutionStatus {
    NotStarted,
    Running,
    Completed,
    Cancelled,
    UncertainOutcome,
}

/// Detailed record of a tracked tool call during a turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackedToolExecution {
    pub call_id: String,
    pub tool_name: String,
    pub input: String,
    pub status: ToolExecutionStatus,
    pub response_id: String,
    pub output: Option<String>,
    pub side_effects_committed: bool,
    pub start_ms: u64,
    pub end_ms: Option<u64>,
}

/// Transition event emitted during dialogue state changes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateTransitionEvent {
    pub transition_id: u64,
    pub from_state: DialogueState,
    pub to_state: DialogueState,
    pub causal_response_id: Option<String>,
    pub trigger_reason: String,
    pub timestamp_ms: u64,
}

/// Event emitted by the full-duplex dialogue coordinator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DialogueEvent {
    StateChanged(StateTransitionEvent),
    UserSpeechDetected {
        timestamp_ms: u64,
    },
    UserSpeechFinalized {
        text: String,
        timestamp_ms: u64,
    },
    GenerationStarted {
        response_id: String,
        prompt: String,
        timestamp_ms: u64,
    },
    TokenEmitted {
        response_id: String,
        text: String,
        timestamp_ms: u64,
    },
    ToolProposed {
        response_id: String,
        proposal: ToolProposal,
        timestamp_ms: u64,
    },
    ToolExecutionStarted {
        response_id: String,
        call_id: String,
        tool_name: String,
        timestamp_ms: u64,
    },
    ToolExecutionCompleted {
        response_id: String,
        call_id: String,
        status: ToolExecutionStatus,
        output: String,
        timestamp_ms: u64,
    },
    SpeechChunkEmitted {
        response_id: String,
        chunk_index: usize,
        duration_ms: f64,
        sample_offset: u64,
        timestamp_ms: u64,
    },
    InterruptionTriggered {
        response_id: String,
        interrupt_latency_ms: f64,
        played_words: Vec<String>,
        discarded_samples: u64,
        cancelled_tools: Vec<String>,
        timestamp_ms: u64,
    },
    TurnCompleted {
        response_id: String,
        played_text: String,
        was_interrupted: bool,
        timestamp_ms: u64,
    },
}

/// Immutable record of a completed or interrupted conversational turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnRecord {
    pub turn_id: u64,
    pub response_id: String,
    pub user_text: String,
    pub assistant_played_text: String,
    pub assistant_unspoken_text: Option<String>,
    pub was_interrupted: bool,
    pub executed_tools: Vec<TrackedToolExecution>,
    pub timestamp_ms: u64,
}

// ---------------------------------------------------------------------------
// Full-Duplex Dialogue Coordinator
// ---------------------------------------------------------------------------

/// Coordinates live full-duplex conversations with immediate barge-in,
/// stateful cancellation, and causal response validation.
pub struct FullDuplexCoordinator {
    state: DialogueState,
    session_id: String,
    turn_counter: u64,
    transition_counter: u64,
    active_response_id: Option<String>,
    active_cancellation_token: Option<CancellationToken>,
    backend_adapter: Arc<dyn ModelBackendAdapter>,
    tts_engine: Arc<Mutex<FormantNeuralTtsEngine>>,
    playback_pipeline: Arc<PlaybackPipeline>,
    endpoint_detector: SpeechEndpointDetector,
    capability_policy: CapabilityPolicy,
    conversation_history: Vec<TurnRecord>,
    active_tools: Vec<TrackedToolExecution>,
    active_user_prompt: String,
    active_generated_text: String,
    interrupt_latencies_ms: Vec<f64>,
    stale_rejection_count: u64,
    scheduler: Option<Arc<ResourceScheduler>>,
}

impl FullDuplexCoordinator {
    pub fn new(
        session_id: impl Into<String>,
        backend_adapter: Arc<dyn ModelBackendAdapter>,
        tts_engine: Arc<Mutex<FormantNeuralTtsEngine>>,
        playback_pipeline: Arc<PlaybackPipeline>,
        capability_policy: CapabilityPolicy,
    ) -> Self {
        let ep_config = SpeechEndpointConfig::default();

        Self {
            state: DialogueState::Idle,
            session_id: session_id.into(),
            turn_counter: 0,
            transition_counter: 0,
            active_response_id: None,
            active_cancellation_token: None,
            backend_adapter,
            tts_engine,
            playback_pipeline,
            endpoint_detector: SpeechEndpointDetector::new(ep_config, MicrophoneMode::OpenMic),
            capability_policy,
            conversation_history: Vec::new(),
            active_tools: Vec::new(),
            active_user_prompt: String::new(),
            active_generated_text: String::new(),
            interrupt_latencies_ms: Vec::new(),
            stale_rejection_count: 0,
            scheduler: None,
        }
    }

    /// Attaches an optional resource scheduler to notify on dialogue activity and trigger background training preemption.
    pub fn with_resource_scheduler(mut self, scheduler: Arc<ResourceScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Current formal dialogue state.
    pub fn state(&self) -> DialogueState {
        self.state
    }

    /// Currently active causal response ID (if any).
    pub fn active_response_id(&self) -> Option<&str> {
        self.active_response_id.as_deref()
    }

    /// Full committed conversational history.
    pub fn conversation_history(&self) -> &[TurnRecord] {
        &self.conversation_history
    }

    /// Count of stale or orphaned generation chunks rejected by ID.
    pub fn stale_rejection_count(&self) -> u64 {
        self.stale_rejection_count
    }

    /// Transition to a new dialogue state with causal auditing.
    fn transition_to(
        &mut self,
        new_state: DialogueState,
        trigger_reason: impl Into<String>,
        timestamp_ms: u64,
    ) -> DialogueEvent {
        self.transition_counter += 1;
        let event = StateTransitionEvent {
            transition_id: self.transition_counter,
            from_state: self.state,
            to_state: new_state,
            causal_response_id: self.active_response_id.clone(),
            trigger_reason: trigger_reason.into(),
            timestamp_ms,
        };
        self.state = new_state;

        if let Some(scheduler) = &self.scheduler {
            match new_state {
                DialogueState::Listening
                | DialogueState::Interpreting
                | DialogueState::Speaking
                | DialogueState::AwaitingTool
                | DialogueState::Interrupted => {
                    scheduler.on_dialogue_started();
                }
                DialogueState::Idle => {
                    scheduler.on_dialogue_ended();
                }
            }
        }

        DialogueEvent::StateChanged(event)
    }

    /// Ingests a 20ms audio frame continuously.
    ///
    /// Even while the assistant is speaking, input capture continues uninterrupted
    /// so the user can barge in and correct or redirect at any time.
    pub fn process_audio_frame(
        &mut self,
        frame: &[f32],
        timestamp_ms: u64,
    ) -> Result<Vec<DialogueEvent>, String> {
        let mut events = Vec::new();
        let chunk = AudioChunk::new(frame.to_vec(), 16000, 1, 0, timestamp_ms * 1_000_000);
        let ep_event = self.endpoint_detector.process_chunk(&chunk);

        match ep_event {
            EndpointEvent::SpeechStarted { .. } => {
                events.push(DialogueEvent::UserSpeechDetected { timestamp_ms });

                // If user starts speaking while assistant is speaking or interpreting, trigger barge-in!
                if self.state == DialogueState::Speaking
                    || self.state == DialogueState::Interpreting
                {
                    let barge_events = self.trigger_barge_in(timestamp_ms)?;
                    events.extend(barge_events);
                } else if self.state == DialogueState::Idle {
                    events.push(self.transition_to(
                        DialogueState::Listening,
                        "User started speaking",
                        timestamp_ms,
                    ));
                }
            }
            EndpointEvent::SpeechEnded { .. } => {
                // Speech ended. In a full system, ASR produces final text; here we notify finalize.
                if self.state == DialogueState::Listening {
                    events.push(self.transition_to(
                        DialogueState::Interpreting,
                        "Speech turn finalized",
                        timestamp_ms,
                    ));
                }
            }
            EndpointEvent::SpeechContinuing { .. } => {
                if self.state == DialogueState::Speaking {
                    // Continuing speech during Speaking also triggers barge-in
                    let barge_events = self.trigger_barge_in(timestamp_ms)?;
                    events.extend(barge_events);
                }
            }
            EndpointEvent::Silence => {}
        }

        Ok(events)
    }

    /// Submits a recognized or typed user utterance and executes the turn.
    pub fn submit_user_turn(
        &mut self,
        user_prompt: &str,
        tool_registry: &ToolRegistry,
        timestamp_ms: u64,
    ) -> Result<Vec<DialogueEvent>, String> {
        let mut events = Vec::new();

        // If previously speaking, interrupt before starting new turn
        if self.state == DialogueState::Speaking || self.state == DialogueState::Interpreting {
            let interrupt_events = self.trigger_barge_in(timestamp_ms)?;
            events.extend(interrupt_events);
        }

        self.turn_counter += 1;
        let response_id = format!("resp_t{}_{}", self.turn_counter, rand::random::<u16>());
        self.active_response_id = Some(response_id.clone());
        self.active_user_prompt = user_prompt.to_string();
        self.active_generated_text.clear();
        self.active_tools.clear();

        let cancel_token = CancellationToken::new();
        self.active_cancellation_token = Some(cancel_token.clone());

        events.push(DialogueEvent::UserSpeechFinalized {
            text: user_prompt.to_string(),
            timestamp_ms,
        });

        events.push(self.transition_to(
            DialogueState::Interpreting,
            format!("Starting turn for prompt: \"{user_prompt}\""),
            timestamp_ms,
        ));

        events.push(DialogueEvent::GenerationStarted {
            response_id: response_id.clone(),
            prompt: user_prompt.to_string(),
            timestamp_ms,
        });

        // Initialize TTS utterance
        {
            let mut engine = self.tts_engine.lock().map_err(|e| e.to_string())?;
            let voice_cfg = VoiceConfig::default();
            engine.start_utterance(&response_id, &voice_cfg)?;
        }

        let mut proposed_tools = Vec::new();
        let mut text_chunks = Vec::new();

        let req = GenerateRequest::new(&self.session_id, user_prompt)
            .with_cancellation_token(cancel_token.clone());

        // Stream tokens from backend adapter
        let gen_meta = self.backend_adapter.generate_stream(req, &mut |chunk| {
            if cancel_token.is_cancelled() {
                return Ok(());
            }

            match chunk {
                GenerationChunk::Token { text, .. } => {
                    text_chunks.push(text);
                    Ok(())
                }
                GenerationChunk::ToolCall(prop) => {
                    proposed_tools.push(prop);
                    Ok(())
                }
                GenerationChunk::Complete(_) => Ok(()),
            }
        })?;

        if cancel_token.is_cancelled() || gen_meta.finish_reason == FinishReason::Cancelled {
            // Cancelled mid-generation
            return Ok(events);
        }

        // Synthesize and stream text chunks
        if !text_chunks.is_empty() {
            events.push(self.transition_to(
                DialogueState::Speaking,
                "Synthesizing and speaking response",
                timestamp_ms,
            ));

            let mut all_audio_chunks = Vec::new();
            {
                let mut engine = self.tts_engine.lock().map_err(|e| e.to_string())?;
                let mut clause_buffer = String::new();
                for token in &text_chunks {
                    events.push(DialogueEvent::TokenEmitted {
                        response_id: response_id.clone(),
                        text: token.clone(),
                        timestamp_ms,
                    });
                    self.active_generated_text.push_str(token);
                    clause_buffer.push_str(token);

                    while let Some(clause) = crate::voice::tts::extract_speech_clause(&mut clause_buffer) {
                        let synth_chunks = engine.feed_text(&response_id, &format!("{clause} "))?;
                        all_audio_chunks.extend(synth_chunks);
                    }
                }
                let remaining = clause_buffer.trim();
                if !remaining.is_empty() {
                    let synth_chunks = engine.feed_text(&response_id, &format!("{remaining} "))?;
                    all_audio_chunks.extend(synth_chunks);
                }
                let final_chunks = engine.finalize_utterance(&response_id)?;
                all_audio_chunks.extend(final_chunks);
            }

            // Enqueue chunks into PlaybackPipeline
            for chunk in all_audio_chunks {
                if chunk.is_final_chunk || chunk.samples.is_empty() {
                    continue;
                }
                let audio_chunk = AudioChunk::new(
                    chunk.samples.to_vec(),
                    chunk.sample_rate,
                    chunk.channels,
                    chunk.sample_offset,
                    timestamp_ms * 1_000_000,
                );
                let _ = self.playback_pipeline.enqueue_chunk(audio_chunk);

                events.push(DialogueEvent::SpeechChunkEmitted {
                    response_id: response_id.clone(),
                    chunk_index: chunk.chunk_index,
                    duration_ms: chunk.duration_ms,
                    sample_offset: chunk.sample_offset,
                    timestamp_ms,
                });
            }
        }

        // Handle proposed tools
        if !proposed_tools.is_empty() {
            events.push(self.transition_to(
                DialogueState::AwaitingTool,
                "Awaiting tool verification and execution",
                timestamp_ms,
            ));

            for prop in proposed_tools {
                events.push(DialogueEvent::ToolProposed {
                    response_id: response_id.clone(),
                    proposal: prop.clone(),
                    timestamp_ms,
                });

                let mut tracked = TrackedToolExecution {
                    call_id: prop.call_id.clone(),
                    tool_name: prop.tool_name.clone(),
                    input: prop.input.clone(),
                    status: ToolExecutionStatus::NotStarted,
                    response_id: response_id.clone(),
                    output: None,
                    side_effects_committed: false,
                    start_ms: timestamp_ms,
                    end_ms: None,
                };

                // Check capability policy
                events.push(DialogueEvent::ToolExecutionStarted {
                    response_id: response_id.clone(),
                    call_id: prop.call_id.clone(),
                    tool_name: prop.tool_name.clone(),
                    timestamp_ms,
                });
                tracked.status = ToolExecutionStatus::Running;

                let call_str = if prop.input.is_empty() {
                    format!("{}()", prop.tool_name)
                } else {
                    format!("{}(\"{}\")", prop.tool_name, prop.input)
                };

                let exec_result =
                    tool_registry.dispatch_with_policy(&call_str, &self.capability_policy);

                match exec_result {
                    Ok(out) => {
                        tracked.status = ToolExecutionStatus::Completed;
                        tracked.side_effects_committed = true;
                        tracked.output = Some(out.clone());
                        tracked.end_ms = Some(timestamp_ms);
                        events.push(DialogueEvent::ToolExecutionCompleted {
                            response_id: response_id.clone(),
                            call_id: prop.call_id.clone(),
                            status: ToolExecutionStatus::Completed,
                            output: out,
                            timestamp_ms,
                        });
                    }
                    Err(e) => {
                        let err_msg = e.to_string();
                        tracked.status = ToolExecutionStatus::UncertainOutcome;
                        tracked.output = Some(err_msg.clone());
                        tracked.end_ms = Some(timestamp_ms);
                        events.push(DialogueEvent::ToolExecutionCompleted {
                            response_id: response_id.clone(),
                            call_id: prop.call_id.clone(),
                            status: ToolExecutionStatus::UncertainOutcome,
                            output: err_msg,
                            timestamp_ms,
                        });
                    }
                }

                self.active_tools.push(tracked);
            }
        }

        // If no speech was queued, or if playback has already drained, complete turn
        if self.playback_pipeline.queued_chunks() == 0 {
            let record = TurnRecord {
                turn_id: self.turn_counter,
                response_id: response_id.clone(),
                user_text: user_prompt.to_string(),
                assistant_played_text: self.active_generated_text.clone(),
                assistant_unspoken_text: None,
                was_interrupted: false,
                executed_tools: self.active_tools.clone(),
                timestamp_ms,
            };
            self.conversation_history.push(record);

            events.push(DialogueEvent::TurnCompleted {
                response_id: response_id.clone(),
                played_text: self.active_generated_text.clone(),
                was_interrupted: false,
                timestamp_ms,
            });

            events.push(self.transition_to(
                DialogueState::Idle,
                "Turn completed normally",
                timestamp_ms,
            ));
            self.active_response_id = None;
            self.active_cancellation_token = None;
        }

        Ok(events)
    }

    /// Checks if active speaking turn has finished draining through speaker playback pipeline.
    pub fn poll_playback_completion(&mut self, timestamp_ms: u64) -> Vec<DialogueEvent> {
        let mut events = Vec::new();
        if (self.state == DialogueState::Speaking || self.state == DialogueState::AwaitingTool)
            && self.playback_pipeline.queued_chunks() == 0
        {
            if let Some(resp_id) = self.active_response_id.take() {
                let record = TurnRecord {
                    turn_id: self.turn_counter,
                    response_id: resp_id.clone(),
                    user_text: self.active_user_prompt.clone(),
                    assistant_played_text: self.active_generated_text.clone(),
                    assistant_unspoken_text: None,
                    was_interrupted: false,
                    executed_tools: self.active_tools.clone(),
                    timestamp_ms,
                };
                self.conversation_history.push(record);

                events.push(DialogueEvent::TurnCompleted {
                    response_id: resp_id,
                    played_text: self.active_generated_text.clone(),
                    was_interrupted: false,
                    timestamp_ms,
                });

                events.push(self.transition_to(
                    DialogueState::Idle,
                    "Playback drained; turn completed",
                    timestamp_ms,
                ));
                self.active_cancellation_token = None;
            }
        }
        events
    }

    /// Drains all remaining samples from the playback pipeline and completes the active speaking turn.
    pub fn drain_playback(&mut self, timestamp_ms: u64) -> Vec<DialogueEvent> {
        let mut dummy = vec![0.0f32; 16000];
        while self.playback_pipeline.queued_chunks() > 0 {
            self.playback_pipeline.pull_samples(&mut dummy);
        }
        self.poll_playback_completion(timestamp_ms)
    }

    /// Triggers immediate barge-in interruption.
    ///
    /// Stops playback, flushes pending TTS audio, cancels generation token,
    /// cancels queued tools, commits only heard speech, and preserves completed tools.
    pub fn trigger_barge_in(&mut self, timestamp_ms: u64) -> Result<Vec<DialogueEvent>, String> {
        let t_start = Instant::now();
        let mut events = Vec::new();

        let response_id = match self.active_response_id.take() {
            Some(id) => id,
            None => return Ok(events),
        };

        // 1. Cancel in-flight model generation
        if let Some(token) = self.active_cancellation_token.take() {
            token.cancel();
        }

        // 2. Halt speaker playback immediately and retrieve cursor
        let playback_cursor = self.playback_pipeline.interrupt();

        // 3. Cancel TTS utterance and compute exact played vs discarded words
        let cancel_report = {
            let mut engine = self.tts_engine.lock().map_err(|e| e.to_string())?;
            engine.cancel_utterance(&response_id, playback_cursor)?
        };

        let latency_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        self.interrupt_latencies_ms.push(latency_ms);

        // 4. Cancel any tools that were queued but not started
        let mut cancelled_tool_names = Vec::new();
        for tool in &mut self.active_tools {
            if tool.status == ToolExecutionStatus::NotStarted {
                tool.status = ToolExecutionStatus::Cancelled;
                cancelled_tool_names.push(tool.tool_name.clone());
            }
        }

        // 5. Commit ONLY the words that were actually played through speaker hardware
        let played_text = cancel_report.played_words.join(" ");
        let unspoken_text = if !self.active_generated_text.is_empty() {
            Some(self.active_generated_text.clone())
        } else {
            None
        };

        let turn_record = TurnRecord {
            turn_id: self.turn_counter,
            response_id: response_id.clone(),
            user_text: self.active_user_prompt.clone(),
            assistant_played_text: played_text.clone(),
            assistant_unspoken_text: unspoken_text,
            was_interrupted: true,
            executed_tools: self.active_tools.clone(),
            timestamp_ms,
        };
        self.conversation_history.push(turn_record);

        events.push(self.transition_to(
            DialogueState::Interrupted,
            format!("Barge-in: playback stopped at sample {playback_cursor} in {latency_ms:.2}ms"),
            timestamp_ms,
        ));

        events.push(DialogueEvent::InterruptionTriggered {
            response_id: response_id.clone(),
            interrupt_latency_ms: latency_ms,
            played_words: cancel_report.played_words,
            discarded_samples: cancel_report.discarded_samples,
            cancelled_tools: cancelled_tool_names,
            timestamp_ms,
        });

        // Transition back to Listening for the user's correction
        events.push(self.transition_to(
            DialogueState::Listening,
            "Listening to user correction / follow-up",
            timestamp_ms,
        ));

        Ok(events)
    }

    /// Verifies if a generation chunk belongs to the current causal response,
    /// rejecting stale or orphaned chunks from previous turns.
    pub fn accept_chunk_for_response(&mut self, chunk_response_id: &str) -> bool {
        match &self.active_response_id {
            Some(cur_id) if cur_id == chunk_response_id => true,
            _ => {
                self.stale_rejection_count += 1;
                false
            }
        }
    }

    /// Calculates p50 and p95 interrupt-to-playback-stop latency in milliseconds.
    pub fn interrupt_latency_stats(&self) -> (f64, f64, usize) {
        if self.interrupt_latencies_ms.is_empty() {
            return (0.0, 0.0, 0);
        }
        let mut sorted = self.interrupt_latencies_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let count = sorted.len();
        let p50_idx = ((count as f64 * 0.50).round() as usize).min(count - 1);
        let p95_idx = ((count as f64 * 0.95).round() as usize).min(count - 1);
        (sorted[p50_idx], sorted[p95_idx], count)
    }
}
