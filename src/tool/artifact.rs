//! Structured Tool Artifacts and Speech Summaries.
//!
//! Provides clean, structured result representations suitable for both
//! deep model verification and real-time TTS speech summarization.

use serde::{Deserialize, Serialize};

/// Structured outcome of a verified tool execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuredToolArtifact {
    pub tool_name: String,
    pub success: bool,
    pub speech_summary: String,
    pub detailed_payload: serde_json::Value,
    pub warnings: Vec<String>,
}

impl StructuredToolArtifact {
    pub fn new(
        tool_name: impl Into<String>,
        success: bool,
        speech_summary: impl Into<String>,
        detailed_payload: serde_json::Value,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            success,
            speech_summary: speech_summary.into(),
            detailed_payload,
            warnings: Vec::new(),
        }
    }

    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }
}
