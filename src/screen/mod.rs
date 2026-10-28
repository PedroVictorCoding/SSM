//! Screen-Aware Assistant Subsystem: Window Capture, UI Automation, and OCR Context.
//!
//! Provides:
//! - Selected-window and display capture with pause/resume and active capture indicator.
//! - UI Automation accessibility hierarchy, window management, and multi-monitor DPI scaling.
//! - Local Optical Character Recognition (OCR) for rasterized/canvas text.
//! - Change detection, bounded frame sampling, and coordinate transformations.
//! - Unified screen context snapshots formatted for LLM reasoning.

pub mod automation;
pub mod camera;
pub mod capture;
pub mod grounding;
pub mod ocr;

use std::time::Instant;

use serde::{Deserialize, Serialize};

pub use automation::{CoordinateTransformer, MonitorInfo, Rect, UiElement, WindowInfo};
pub use camera::{
    CameraDeviceConfig, CameraError, CameraFrame, CameraProvider, CameraService, CameraState,
    HeadlessCameraProvider, OsCameraProvider, TemporalChangeSummary,
};
pub use capture::{
    CaptureSource, CapturedFrame, HeadlessCaptureProvider, ScreenCaptureProvider,
    ScreenCaptureService,
};
pub use grounding::{
    EncodedVisualContext, GroundedObservation, LocalVisualGroundingEngine, VisualEncoderAdapter,
    VisualTokenBudget, verify_modality_ablation,
};
pub use ocr::{
    LocalOpticalCharacterRecognizer, OcrBoundingBox, OcrEngine, OcrResult, UiOcrDiscrepancy,
    reconcile_ui_and_ocr,
};

use crate::capability::CapabilityPolicy;

/// Unified multimodal screen context snapshot assembled for assistant reasoning.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScreenContextSnapshot {
    pub frame_id: u64,
    pub timestamp_ns: u64,
    pub frame_age_ms: f64,
    pub active_window: Option<WindowInfo>,
    pub cursor_position: (i32, i32),
    pub ui_text_blocks: Vec<String>,
    pub ocr_text: String,
    pub unified_text_summary: String,
    pub discrepancies: Vec<UiOcrDiscrepancy>,
    pub has_changed: bool,
    pub capture_overhead_ms: f64,
    pub ocr_latency_ms: f64,
}

impl ScreenContextSnapshot {
    /// Formats the screen snapshot into a structured prompt context block for the model.
    pub fn to_context_prompt(&self) -> String {
        let mut prompt = String::new();
        prompt.push_str("--- SCREEN CONTEXT ---\n");

        if let Some(win) = &self.active_window {
            prompt.push_str(&format!(
                "Active Window: \"{}\" (PID: {}, Process: \"{}\", Bounds: [{}, {}, {}x{}], Scale: {:.2})\n",
                win.title, win.process_id, win.process_name, win.bounds.x, win.bounds.y, win.bounds.width, win.bounds.height, win.display_scale
            ));
        }

        prompt.push_str(&format!(
            "Cursor Position: ({}, {})\n",
            self.cursor_position.0, self.cursor_position.1
        ));

        if !self.ui_text_blocks.is_empty() {
            prompt.push_str("UI Automation Elements:\n");
            for t in &self.ui_text_blocks {
                prompt.push_str(&format!("  - {t}\n"));
            }
        }

        if !self.ocr_text.is_empty() {
            prompt.push_str("OCR Detected Text:\n");
            for line in self.ocr_text.lines() {
                prompt.push_str(&format!("  | {line}\n"));
            }
        }

        if !self.discrepancies.is_empty() {
            prompt.push_str("Inaccessible/Canvas Regions:\n");
            for d in &self.discrepancies {
                prompt.push_str(&format!(
                    "  * [Region {}x{}]: {}\n",
                    d.region.width, d.region.height, d.explanation
                ));
            }
        }

        prompt.push_str("--- END SCREEN CONTEXT ---");
        prompt
    }
}

/// Unified coordinator collecting window pixels, UI Automation data, and OCR context.
pub struct ScreenContextCoordinator {
    capture_service: ScreenCaptureService,
    ocr_engine: Box<dyn OcrEngine>,
    active_ui_tree: Option<UiElement>,
    cursor_position: (i32, i32),
}

impl ScreenContextCoordinator {
    pub fn new(capture_service: ScreenCaptureService, ocr_engine: Box<dyn OcrEngine>) -> Self {
        Self {
            capture_service,
            ocr_engine,
            active_ui_tree: None,
            cursor_position: (0, 0),
        }
    }

    pub fn capture_service_mut(&mut self) -> &mut ScreenCaptureService {
        &mut self.capture_service
    }

    pub fn set_cursor_position(&mut self, x: i32, y: i32) {
        self.cursor_position = (x, y);
    }

    pub fn set_active_ui_tree(&mut self, tree: UiElement) {
        self.active_ui_tree = Some(tree);
    }

    /// Assembles a fresh screen context snapshot combining video frame, UI tree, and OCR scan.
    pub fn assemble_context(
        &mut self,
        timestamp_ns: u64,
        policy: &CapabilityPolicy,
    ) -> Result<ScreenContextSnapshot, String> {
        let frame_opt = self.capture_service.sample_frame(timestamp_ns, policy)?;
        let frame = match frame_opt {
            Some(f) => f,
            None => {
                return Err("Screen capture is paused, stopped, or not permitted".into());
            }
        };

        let frame_age_ms = frame.age_ms(timestamp_ns);

        // Run OCR scan over frame
        let t_ocr0 = Instant::now();
        let ocr_res = self.ocr_engine.recognize_text(&frame.buffer, None)?;
        let ocr_latency_ms = t_ocr0.elapsed().as_secs_f64() * 1000.0;

        // Collect UI Automation text
        let ui_texts = if let Some(tree) = &self.active_ui_tree {
            tree.collect_all_text()
        } else {
            Vec::new()
        };

        // Reconcile UI Automation text vs OCR text
        let mut discrepancies = Vec::new();
        let combined_ui_str = ui_texts.join(" ");
        if let Some(disc) = reconcile_ui_and_ocr(
            if combined_ui_str.is_empty() {
                None
            } else {
                Some(&combined_ui_str)
            },
            &ocr_res,
            frame.bounds,
        ) {
            discrepancies.push(disc);
        }

        let unified_summary = if !ui_texts.is_empty() && !ocr_res.full_text.is_empty() {
            format!("{}\n{}", ui_texts.join("\n"), ocr_res.full_text)
        } else if !ui_texts.is_empty() {
            ui_texts.join("\n")
        } else {
            ocr_res.full_text.clone()
        };

        Ok(ScreenContextSnapshot {
            frame_id: frame.frame_id,
            timestamp_ns,
            frame_age_ms,
            active_window: None, // Can be populated from provider
            cursor_position: self.cursor_position,
            ui_text_blocks: ui_texts,
            ocr_text: ocr_res.full_text,
            unified_text_summary: unified_summary,
            discrepancies,
            has_changed: frame.has_changed,
            capture_overhead_ms: self.capture_service.mean_capture_overhead_ms(),
            ocr_latency_ms,
        })
    }
}
