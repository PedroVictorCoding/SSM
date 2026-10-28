//! Versioned Structured Events and Reference-Counted Media Buffers for SAIL Native Host Runtime.
//!
//! Provides attributable, causal, and timestamped event definitions:
//! - Strict monotonic timestamps and validity intervals.
//! - Causal parent tracking and per-source sequence numbering.
//! - Zero-copy reference-counted shared media buffers (`Arc<SharedAudioBuffer>`, `Arc<SharedVideoBuffer>`).
//! - Model output events enriched with `generation_id` and `response_id`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Immutable, reference-counted PCM audio buffer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SharedAudioBuffer {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Arc<[f32]>,
}

impl SharedAudioBuffer {
    pub fn new(sample_rate: u32, channels: u16, samples: Vec<f32>) -> Self {
        Self {
            sample_rate,
            channels,
            samples: samples.into(),
        }
    }

    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            0.0
        } else {
            self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
        }
    }
}

/// Immutable, reference-counted video/screen frame buffer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SharedVideoBuffer {
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub pixels: Arc<[u8]>,
}

impl SharedVideoBuffer {
    pub fn new(width: u32, height: u32, format: impl Into<String>, pixels: Vec<u8>) -> Self {
        Self {
            width,
            height,
            format: format.into(),
            pixels: pixels.into(),
        }
    }

    pub fn byte_size(&self) -> usize {
        self.pixels.len()
    }
}

/// Typed payloads carried by `RuntimeEvent`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EventPayload {
    /// Ingested microphone audio chunk
    AudioInput(Arc<SharedAudioBuffer>),
    /// Synthesized speech playback chunk
    AudioOutput(Arc<SharedAudioBuffer>),
    /// Captured screen or camera frame
    VideoFrame(Arc<SharedVideoBuffer>),
    /// User speech-to-text transcript or typed text
    UserInputText { text: String, is_final: bool },
    /// Autoregressively generated model token / chunk
    ModelOutputChunk {
        generation_id: u64,
        response_id: String,
        token_id: u32,
        text: String,
        is_final: bool,
    },
    /// Request to execute an external tool
    ToolRequest {
        call_id: String,
        tool_name: String,
        input: String,
    },
    /// Result returned from a tool execution
    ToolResponse {
        call_id: String,
        tool_name: String,
        output: String,
        is_error: bool,
    },
    /// Assistant state or pipeline control command
    ControlCommand { command: String },
    /// Heartbeat or state update from a worker
    WorkerStatusNotification { worker_id: String, state: String },
}

/// A versioned, attributable runtime event in the SAIL host bus.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RuntimeEvent {
    /// Event schema version (currently 1)
    pub version: u32,
    /// Globally unique event identifier
    pub event_id: u64,
    /// Interactive session identifier
    pub session_id: String,
    /// Originating source component (e.g. "mic", "screen", "model", "tool_runner")
    pub source_id: String,
    /// Optional logical task identifier
    pub task_id: Option<String>,
    /// Per-source strictly monotonic sequence counter
    pub sequence_number: u64,
    /// Host monotonic timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Causal parent event identifier (if triggered by a prior event)
    pub parent_event_id: Option<u64>,
    /// Optional validity deadline in nanoseconds (events expired before processing are dropped)
    pub valid_until_ns: Option<u64>,
    /// Specific typed payload
    pub payload: EventPayload,
}

impl RuntimeEvent {
    pub fn new(
        event_id: u64,
        session_id: impl Into<String>,
        source_id: impl Into<String>,
        sequence_number: u64,
        timestamp_ns: u64,
        payload: EventPayload,
    ) -> Self {
        Self {
            version: 1,
            event_id,
            session_id: session_id.into(),
            source_id: source_id.into(),
            task_id: None,
            sequence_number,
            timestamp_ns,
            parent_event_id: None,
            valid_until_ns: None,
            payload,
        }
    }

    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    pub fn with_parent_event_id(mut self, parent_id: u64) -> Self {
        self.parent_event_id = Some(parent_id);
        self
    }

    pub fn with_ttl_ns(mut self, ttl_ns: u64) -> Self {
        self.valid_until_ns = Some(self.timestamp_ns + ttl_ns);
        self
    }

    pub fn is_expired(&self, current_time_ns: u64) -> bool {
        match self.valid_until_ns {
            Some(deadline) => current_time_ns > deadline,
            None => false,
        }
    }
}

impl fmt::Display for RuntimeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Event#{} [{}] src={} seq={} t={}ns",
            self.event_id, self.session_id, self.source_id, self.sequence_number, self.timestamp_ns
        )
    }
}
