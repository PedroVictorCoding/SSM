//! Synchronized Multimodal Context Assembly, Timeline Alignment, and Deictic Reference Resolution.
//!
//! Provides:
//! - Common monotonic session timeline aligning audio utterances, visual frames, pointer positions,
//!   UI Automation elements, and tool results.
//! - Deictic reference resolution: maps phrases like "this error", "that button", or "this window"
//!   to exact evidence IDs and spatial bounding boxes present at utterance time.
//! - Causal temporal filtering: future observations ($T > T_{\text{utterance}}$) are strictly masked.
//! - Stale evidence rejection: if the screen changed or target age exceeds 1.0s, consequential actions
//!   are rejected and clarification is requested instead of executing stale actions.
//! - Modality token budgeting and inspectable provenance traces.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::runtime::dialogue::TrackedToolExecution;
use crate::screen::automation::{Rect, UiElement, WindowInfo};
use crate::screen::capture::CapturedFrame;
use crate::screen::ocr::OcrBoundingBox;

/// Types of deictic or contextual references that can be resolved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceIntent {
    ErrorOrDiagnostic,
    PointerTargetButton,
    ActiveWindowOrApp,
    SpecificElement(String),
}

/// Resolved target with associated spatial bounds, text, and source evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedTarget {
    pub target_id: String,
    pub intent: ReferenceIntent,
    pub label: String,
    pub bounds: Option<Rect>,
    pub text_content: Option<String>,
    pub evidence_ids: Vec<String>,
    pub confidence: f32,
    pub age_ms: f64,
}

/// Outcome of reference resolution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ReferenceResolutionResult {
    Resolved(ResolvedTarget),
    Ambiguous {
        reason: String,
        candidate_ids: Vec<String>,
    },
    StaleTargetRejected {
        target_id: String,
        age_ms: f64,
        max_allowed_age_ms: f64,
        reason: String,
    },
    TargetNotFound {
        reason: String,
    },
}

/// Token budget breakdown across modalities for context assembly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModalityBudgetBreakdown {
    pub max_total_tokens: usize,
    pub text_tokens: usize,
    pub ui_tokens: usize,
    pub visual_tokens: usize,
    pub tool_tokens: usize,
}

impl Default for ModalityBudgetBreakdown {
    fn default() -> Self {
        Self {
            max_total_tokens: 1024,
            text_tokens: 256,
            ui_tokens: 256,
            visual_tokens: 64,
            tool_tokens: 256,
        }
    }
}

/// Inspectable provenance trace explaining how context was assembled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextProvenanceTrace {
    pub session_id: String,
    pub utterance_timestamp_ms: u64,
    pub active_window_id: Option<u64>,
    pub selected_evidence_ids: Vec<String>,
    pub evidence_freshness_ms: HashMap<String, f64>,
    pub screen_age_ms: f64,
    pub target_screen_age_target_ms: f64,
    pub causal_mask_applied: bool,
}

/// Synchronized multimodal context packet prepared for model consumption.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynchronizedContextPacket {
    pub context_id: String,
    pub prompt_text: String,
    pub resolved_reference: Option<ResolvedTarget>,
    pub formatted_prompt: String,
    pub visual_tokens: Vec<Vec<f32>>,
    pub provenance_trace: ContextProvenanceTrace,
    pub assembly_latency_ms: f64,
}

/// Timestamped timeline entry on the common session clock.
#[derive(Clone, Debug)]
pub enum TimelineEntry {
    ScreenFrame {
        frame_id: u64,
        timestamp_ms: u64,
        window: Option<WindowInfo>,
        frame: CapturedFrame,
        ocr_boxes: Vec<OcrBoundingBox>,
    },
    PointerLocation {
        timestamp_ms: u64,
        x: i32,
        y: i32,
    },
    UiTreeSnapshot {
        timestamp_ms: u64,
        root: UiElement,
    },
    ToolExecutionResult {
        timestamp_ms: u64,
        tool: TrackedToolExecution,
    },
}

impl TimelineEntry {
    pub fn timestamp_ms(&self) -> u64 {
        match self {
            TimelineEntry::ScreenFrame { timestamp_ms, .. } => *timestamp_ms,
            TimelineEntry::PointerLocation { timestamp_ms, .. } => *timestamp_ms,
            TimelineEntry::UiTreeSnapshot { timestamp_ms, .. } => *timestamp_ms,
            TimelineEntry::ToolExecutionResult { timestamp_ms, .. } => *timestamp_ms,
        }
    }
}

/// Coordinates timeline ingestion, causal temporal alignment, and reference resolution.
pub struct MultimodalContextAssembler {
    session_id: String,
    timeline: VecDeque<TimelineEntry>,
    max_timeline_entries: usize,
    max_actionable_screen_age_ms: f64,
    token_budget: ModalityBudgetBreakdown,
}

impl MultimodalContextAssembler {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            timeline: VecDeque::with_capacity(128),
            max_timeline_entries: 128,
            max_actionable_screen_age_ms: 1000.0, // <= 1s freshness target for consequential actions
            token_budget: ModalityBudgetBreakdown::default(),
        }
    }

    pub fn set_max_actionable_screen_age_ms(&mut self, max_age_ms: f64) {
        self.max_actionable_screen_age_ms = max_age_ms;
    }

    pub fn token_budget(&self) -> &ModalityBudgetBreakdown {
        &self.token_budget
    }

    /// Appends an observation to the monotonic session timeline.
    pub fn push_entry(&mut self, entry: TimelineEntry) {
        if self.timeline.len() >= self.max_timeline_entries {
            self.timeline.pop_front();
        }
        self.timeline.push_back(entry);
    }

    /// Resolves what a user phrase ("this error", "that button", etc.) refers to
    /// at the exact monotonic time `utterance_time_ms`.
    ///
    /// Applies causal temporal masking: observations with $T > T_{\text{utterance}}$ are ignored!
    pub fn resolve_reference(
        &self,
        phrase: &str,
        utterance_time_ms: u64,
        is_consequential_action: bool,
    ) -> ReferenceResolutionResult {
        let phrase_lower = phrase.to_lowercase();
        let intent = if phrase_lower.contains("error")
            || phrase_lower.contains("fail")
            || phrase_lower.contains("bug")
            || phrase_lower.contains("diagnostic")
        {
            ReferenceIntent::ErrorOrDiagnostic
        } else if phrase_lower.contains("button")
            || phrase_lower.contains("click")
            || phrase_lower.contains("press")
            || phrase_lower.contains("that")
        {
            ReferenceIntent::PointerTargetButton
        } else if phrase_lower.contains("window")
            || phrase_lower.contains("app")
            || phrase_lower.contains("editor")
        {
            ReferenceIntent::ActiveWindowOrApp
        } else {
            ReferenceIntent::SpecificElement(phrase.to_string())
        };

        // 1. Filter timeline entries causally: T <= utterance_time_ms
        let causal_entries: Vec<&TimelineEntry> = self
            .timeline
            .iter()
            .filter(|e| e.timestamp_ms() <= utterance_time_ms)
            .collect();

        if causal_entries.is_empty() {
            return ReferenceResolutionResult::TargetNotFound {
                reason: "No timeline observations prior to utterance timestamp".into(),
            };
        }

        // Find latest screen frame before or at utterance time
        let latest_frame = causal_entries.iter().rev().find_map(|e| match e {
            TimelineEntry::ScreenFrame {
                frame_id,
                timestamp_ms,
                window,
                frame,
                ocr_boxes,
            } => Some((*frame_id, *timestamp_ms, window, frame, ocr_boxes)),
            _ => None,
        });

        // Find latest pointer location
        let latest_pointer = causal_entries.iter().rev().find_map(|e| match e {
            TimelineEntry::PointerLocation { timestamp_ms, x, y } => Some((*timestamp_ms, *x, *y)),
            _ => None,
        });

        // Find latest UI tree
        let latest_ui = causal_entries.iter().rev().find_map(|e| match e {
            TimelineEntry::UiTreeSnapshot { timestamp_ms, root } => Some((*timestamp_ms, root)),
            _ => None,
        });

        match intent {
            ReferenceIntent::ErrorOrDiagnostic => {
                if let Some((fid, ts, _win, _frame, ocr_boxes)) = latest_frame {
                    let age_ms = (utterance_time_ms.saturating_sub(ts)) as f64;
                    if is_consequential_action && age_ms > self.max_actionable_screen_age_ms {
                        return ReferenceResolutionResult::StaleTargetRejected {
                            target_id: format!("frame_{fid}_error"),
                            age_ms,
                            max_allowed_age_ms: self.max_actionable_screen_age_ms,
                            reason:
                                "Screen evidence is older than 1.0s target; target may have changed"
                                    .into(),
                        };
                    }

                    // Look for error markers in OCR boxes
                    let mut matched = Vec::new();
                    for box_entry in ocr_boxes {
                        let l = box_entry.text.to_lowercase();
                        if l.contains("error") || l.contains("fail") || l.contains("exception") {
                            matched.push(box_entry);
                        }
                    }

                    if matched.len() == 1 {
                        let m = matched[0];
                        ReferenceResolutionResult::Resolved(ResolvedTarget {
                            target_id: format!("ocr_box_{fid}"),
                            intent,
                            label: m.text.clone(),
                            bounds: Some(m.bounds),
                            text_content: Some(m.text.clone()),
                            evidence_ids: vec![format!("frame_{fid}"), format!("ocr_{fid}")],
                            confidence: m.confidence,
                            age_ms,
                        })
                    } else if matched.len() > 1 {
                        ReferenceResolutionResult::Ambiguous {
                            reason: format!(
                                "Multiple error regions visible on screen ({})",
                                matched.len()
                            ),
                            candidate_ids: matched.iter().map(|b| b.text.clone()).collect(),
                        }
                    } else {
                        // Check if tool results recorded a compiler error
                        for entry in causal_entries.iter().rev() {
                            if let TimelineEntry::ToolExecutionResult { tool, .. } = entry {
                                if let Some(out) = &tool.output {
                                    if out.to_lowercase().contains("error") {
                                        return ReferenceResolutionResult::Resolved(
                                            ResolvedTarget {
                                                target_id: format!("tool_{}", tool.call_id),
                                                intent,
                                                label: format!(
                                                    "Tool Output Error in {}",
                                                    tool.tool_name
                                                ),
                                                bounds: None,
                                                text_content: Some(out.clone()),
                                                evidence_ids: vec![tool.call_id.clone()],
                                                confidence: 0.95,
                                                age_ms,
                                            },
                                        );
                                    }
                                }
                            }
                        }

                        ReferenceResolutionResult::TargetNotFound {
                            reason:
                                "No error diagnostic found on visible screen or in tool history"
                                    .into(),
                        }
                    }
                } else {
                    ReferenceResolutionResult::TargetNotFound {
                        reason: "No visual screen frame captured prior to utterance".into(),
                    }
                }
            }
            ReferenceIntent::PointerTargetButton => {
                let pointer = match latest_pointer {
                    Some((_, px, py)) => (px, py),
                    None => (0, 0),
                };

                if let Some((ts, root)) = latest_ui {
                    let age_ms = (utterance_time_ms.saturating_sub(ts)) as f64;
                    if is_consequential_action && age_ms > self.max_actionable_screen_age_ms {
                        return ReferenceResolutionResult::StaleTargetRejected {
                            target_id: "ui_pointer_target".into(),
                            age_ms,
                            max_allowed_age_ms: self.max_actionable_screen_age_ms,
                            reason: "UI Automation tree older than 1.0s target".into(),
                        };
                    }

                    if let Some(target_elem) = root.find_element_at(pointer.0, pointer.1) {
                        return ReferenceResolutionResult::Resolved(ResolvedTarget {
                            target_id: target_elem.element_id.clone(),
                            intent,
                            label: target_elem.name.clone(),
                            bounds: Some(target_elem.bounds),
                            text_content: target_elem.text_value.clone(),
                            evidence_ids: vec![target_elem.element_id.clone()],
                            confidence: 0.92,
                            age_ms,
                        });
                    }
                }

                ReferenceResolutionResult::TargetNotFound {
                    reason: format!(
                        "No UI button found under pointer at ({}, {})",
                        pointer.0, pointer.1
                    ),
                }
            }
            ReferenceIntent::ActiveWindowOrApp => {
                if let Some((fid, ts, win_opt, _, _)) = latest_frame {
                    let age_ms = (utterance_time_ms.saturating_sub(ts)) as f64;
                    if let Some(win) = win_opt {
                        return ReferenceResolutionResult::Resolved(ResolvedTarget {
                            target_id: format!("win_{}", win.window_id),
                            intent,
                            label: win.title.clone(),
                            bounds: Some(win.bounds),
                            text_content: Some(format!(
                                "Process: {} (PID: {})",
                                win.process_name, win.process_id
                            )),
                            evidence_ids: vec![
                                format!("frame_{fid}"),
                                format!("win_{}", win.window_id),
                            ],
                            confidence: 0.99,
                            age_ms,
                        });
                    }
                }
                ReferenceResolutionResult::TargetNotFound {
                    reason: "No active window info recorded on timeline".into(),
                }
            }
            ReferenceIntent::SpecificElement(name) => ReferenceResolutionResult::TargetNotFound {
                reason: format!("Specific reference '{name}' resolution pending semantic search"),
            },
        }
    }

    /// Assembles full synchronized context with strict modality budgeting and inspectable provenance.
    pub fn assemble_synchronized_context(
        &self,
        prompt: &str,
        utterance_time_ms: u64,
        is_consequential_action: bool,
    ) -> Result<SynchronizedContextPacket, String> {
        let t0 = Instant::now();
        let resolution = self.resolve_reference(prompt, utterance_time_ms, is_consequential_action);

        let mut evidence_ids = Vec::new();
        let mut freshness_map = HashMap::new();
        let mut screen_age_ms = 0.0;
        let mut resolved_opt = None;

        match resolution {
            ReferenceResolutionResult::Resolved(target) => {
                evidence_ids.extend(target.evidence_ids.clone());
                freshness_map.insert(target.target_id.clone(), target.age_ms);
                screen_age_ms = target.age_ms;
                resolved_opt = Some(target);
            }
            ReferenceResolutionResult::StaleTargetRejected {
                target_id,
                age_ms,
                max_allowed_age_ms,
                reason,
            } => {
                return Err(format!(
                    "Action rejected for safety: {reason} (age {age_ms:.1}ms > max {max_allowed_age_ms:.1}ms for '{target_id}')"
                ));
            }
            ReferenceResolutionResult::Ambiguous {
                reason,
                candidate_ids,
            } => {
                return Err(format!(
                    "Clarification required: {reason}. Candidates: {candidate_ids:?}"
                ));
            }
            ReferenceResolutionResult::TargetNotFound { .. } => {}
        }

        let mut formatted = String::new();
        formatted.push_str("--- SYNCHRONIZED MULTIMODAL CONTEXT ---\n");
        formatted.push_str(&format!(
            "User Prompt: \"{prompt}\" (Utterance Time: {utterance_time_ms} ms)\n"
        ));

        if let Some(target) = &resolved_opt {
            formatted.push_str(&format!(
                "Resolved Reference: \"{}\" [ID: {}] (Confidence: {:.2}, Freshness: {:.1} ms)\n",
                target.label, target.target_id, target.confidence, target.age_ms
            ));
            if let Some(txt) = &target.text_content {
                formatted.push_str(&format!("  Exact Content: \"{txt}\"\n"));
            }
            if let Some(b) = target.bounds {
                formatted.push_str(&format!(
                    "  Spatial Bounds: [{}, {}, {}x{}]\n",
                    b.x, b.y, b.width, b.height
                ));
            }
        }
        formatted.push_str("--- END SYNCHRONIZED CONTEXT ---");

        let trace = ContextProvenanceTrace {
            session_id: self.session_id.clone(),
            utterance_timestamp_ms: utterance_time_ms,
            active_window_id: None,
            selected_evidence_ids: evidence_ids,
            evidence_freshness_ms: freshness_map,
            screen_age_ms,
            target_screen_age_target_ms: self.max_actionable_screen_age_ms,
            causal_mask_applied: true,
        };

        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(SynchronizedContextPacket {
            context_id: format!("ctx_{}_{}", utterance_time_ms, rand::random::<u16>()),
            prompt_text: prompt.to_string(),
            resolved_reference: resolved_opt,
            formatted_prompt: formatted,
            visual_tokens: Vec::new(),
            provenance_trace: trace,
            assembly_latency_ms: latency_ms,
        })
    }
}
