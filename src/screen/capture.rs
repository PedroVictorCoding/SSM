//! Native and Cross-Platform Screen Capture Service with Change Detection and Bounded Buffering.
//!
//! Provides:
//! - Human-controlled window/display capture with active capture indicator and pause control.
//! - Strict capability policy verification (`Capability::ScreenCapture`).
//! - Change detection, perceptual diffing, and bounded frame sampling.
//! - Capture loss detection (window closed, minimized, or display disconnected).
//! - Telemetry metrics: frame age, capture overhead, and buffer occupancy.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::capability::{Capability, CapabilityPolicy, SecurityError};
use crate::runtime::event::SharedVideoBuffer;
use crate::screen::automation::{MonitorInfo, Rect, WindowInfo};

/// Configured capture target.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptureSource {
    SelectedWindow(u64),
    SelectedDisplay(String),
}

/// Metadata and pixel payload for a captured video frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapturedFrame {
    pub frame_id: u64,
    pub source: CaptureSource,
    pub timestamp_ns: u64,
    pub buffer: Arc<SharedVideoBuffer>,
    pub bounds: Rect,
    pub scale_factor: f32,
    pub has_changed: bool,
    pub change_ratio: f32,
}

impl CapturedFrame {
    pub fn age_ms(&self, current_timestamp_ns: u64) -> f64 {
        if current_timestamp_ns >= self.timestamp_ns {
            (current_timestamp_ns - self.timestamp_ns) as f64 / 1_000_000.0
        } else {
            0.0
        }
    }
}

/// Abstraction for OS-native capture backends (Windows.Graphics.Capture, X11, Wayland, Headless).
pub trait ScreenCaptureProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn capture_frame(&mut self, source: &CaptureSource) -> Result<Arc<SharedVideoBuffer>, String>;
    fn query_windows(&self) -> Vec<WindowInfo>;
    fn query_monitors(&self) -> Vec<MonitorInfo>;
}

/// Headless / cross-platform capture provider for automated testing, WSL, and headless CI.
pub struct HeadlessCaptureProvider {
    windows: Vec<WindowInfo>,
    monitors: Vec<MonitorInfo>,
    frame_counter: u64,
    mock_pixels: Option<Vec<u8>>,
    simulate_loss_on_frame: Option<u64>,
}

impl HeadlessCaptureProvider {
    pub fn new() -> Self {
        let default_monitor = MonitorInfo {
            monitor_id: "DISPLAY_1".into(),
            name: "Primary Monitor 1440p".into(),
            bounds: Rect::new(0, 0, 2560, 1440),
            scale_factor: 1.0,
            is_primary: true,
        };

        let default_window = WindowInfo {
            window_id: 101,
            title: "Rust Visual Studio Code".into(),
            process_id: 4200,
            process_name: "Code.exe".into(),
            bounds: Rect::new(100, 100, 1920, 1080),
            monitor_id: "DISPLAY_1".into(),
            display_scale: 1.0,
            is_active: true,
            is_minimized: false,
        };

        Self {
            windows: vec![default_window],
            monitors: vec![default_monitor],
            frame_counter: 0,
            mock_pixels: None,
            simulate_loss_on_frame: None,
        }
    }

    pub fn with_mock_pixels(mut self, pixels: Vec<u8>) -> Self {
        self.mock_pixels = Some(pixels);
        self
    }

    pub fn with_windows(mut self, windows: Vec<WindowInfo>) -> Self {
        self.windows = windows;
        self
    }

    pub fn simulate_loss_at(mut self, frame: u64) -> Self {
        self.simulate_loss_on_frame = Some(frame);
        self
    }
}

impl Default for HeadlessCaptureProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ScreenCaptureProvider for HeadlessCaptureProvider {
    fn name(&self) -> &'static str {
        "Headless / Windows.Graphics.Capture Abstraction"
    }

    fn capture_frame(&mut self, source: &CaptureSource) -> Result<Arc<SharedVideoBuffer>, String> {
        self.frame_counter += 1;

        if let Some(loss_frame) = self.simulate_loss_on_frame {
            if self.frame_counter >= loss_frame {
                return Err("Capture target lost: window was closed or minimized".into());
            }
        }

        match source {
            CaptureSource::SelectedWindow(id) => {
                let win = self
                    .windows
                    .iter()
                    .find(|w| w.window_id == *id)
                    .ok_or_else(|| format!("Window {id} not found or closed"))?;

                if win.is_minimized {
                    return Err(format!("Window {id} is minimized; capture unavailable"));
                }

                let pixels = if let Some(p) = &self.mock_pixels {
                    p.clone()
                } else {
                    vec![240u8; (win.bounds.width * win.bounds.height * 4) as usize]
                };

                Ok(Arc::new(SharedVideoBuffer::new(
                    win.bounds.width,
                    win.bounds.height,
                    "RGBA8",
                    pixels,
                )))
            }
            CaptureSource::SelectedDisplay(mon_id) => {
                let mon = self
                    .monitors
                    .iter()
                    .find(|m| m.monitor_id == *mon_id)
                    .ok_or_else(|| format!("Display {mon_id} not found"))?;

                let pixels = if let Some(p) = &self.mock_pixels {
                    p.clone()
                } else {
                    vec![220u8; (mon.bounds.width * mon.bounds.height * 4) as usize]
                };

                Ok(Arc::new(SharedVideoBuffer::new(
                    mon.bounds.width,
                    mon.bounds.height,
                    "RGBA8",
                    pixels,
                )))
            }
        }
    }

    fn query_windows(&self) -> Vec<WindowInfo> {
        self.windows.clone()
    }

    fn query_monitors(&self) -> Vec<MonitorInfo> {
        self.monitors.clone()
    }
}

/// Service managing human-controlled window capture, policy gating, and bounded buffering.
pub struct ScreenCaptureService {
    provider: Box<dyn ScreenCaptureProvider>,
    source: Option<CaptureSource>,
    is_capturing: bool,
    is_paused: bool,
    active_indicator: bool,
    frame_counter: u64,
    last_frame_buffer: Option<Arc<SharedVideoBuffer>>,
    frame_ring_buffer: VecDeque<CapturedFrame>,
    max_buffer_capacity: usize,
    capture_overheads_ms: Vec<f64>,
    capture_loss_count: u64,
}

impl ScreenCaptureService {
    pub fn new(provider: Box<dyn ScreenCaptureProvider>, capacity: usize) -> Self {
        Self {
            provider,
            source: None,
            is_capturing: false,
            is_paused: false,
            active_indicator: false,
            frame_counter: 0,
            last_frame_buffer: None,
            frame_ring_buffer: VecDeque::with_capacity(capacity),
            max_buffer_capacity: capacity.max(2),
            capture_overheads_ms: Vec::new(),
            capture_loss_count: 0,
        }
    }

    /// Visible UI capture indicator (true whenever capture hardware/API is active).
    pub fn active_indicator(&self) -> bool {
        self.active_indicator
    }

    pub fn is_capturing(&self) -> bool {
        self.is_capturing
    }

    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    /// Starts capturing a configured window or display after capability policy verification.
    pub fn start_capture(
        &mut self,
        source: CaptureSource,
        policy: &CapabilityPolicy,
    ) -> Result<(), SecurityError> {
        // Enforce alignment & capability boundary: ScreenCapture permission must be human-granted
        policy.check_capability(&Capability::ScreenCapture)?;

        self.source = Some(source);
        self.is_capturing = true;
        self.is_paused = false;
        self.active_indicator = true;
        self.frame_ring_buffer.clear();
        self.last_frame_buffer = None;

        Ok(())
    }

    /// Temporarily pauses capture without unhooking the configured window.
    pub fn pause_capture(&mut self) {
        self.is_paused = true;
        self.active_indicator = false;
    }

    /// Resumes paused capture after policy re-verification.
    pub fn resume_capture(&mut self, policy: &CapabilityPolicy) -> Result<(), SecurityError> {
        policy.check_capability(&Capability::ScreenCapture)?;
        if self.is_capturing {
            self.is_paused = false;
            self.active_indicator = true;
        }
        Ok(())
    }

    /// Completely halts capture, drops frame buffers, and clears the active indicator.
    pub fn stop_capture(&mut self) {
        self.is_capturing = false;
        self.is_paused = false;
        self.active_indicator = false;
        self.source = None;
        self.frame_ring_buffer.clear();
        self.last_frame_buffer = None;
    }

    /// Captures a new frame with change detection and bounded buffer sampling.
    pub fn sample_frame(
        &mut self,
        timestamp_ns: u64,
        policy: &CapabilityPolicy,
    ) -> Result<Option<CapturedFrame>, String> {
        if !self.is_capturing || self.is_paused {
            return Ok(None);
        }

        // Verify policy is still valid
        policy
            .check_capability(&Capability::ScreenCapture)
            .map_err(|e| e.to_string())?;

        let source = self
            .source
            .as_ref()
            .ok_or_else(|| "No capture source configured".to_string())?;

        let t0 = Instant::now();
        let raw_buffer = match self.provider.capture_frame(source) {
            Ok(buf) => buf,
            Err(e) => {
                self.capture_loss_count += 1;
                self.is_capturing = false;
                self.active_indicator = false;
                return Err(format!("Capture target lost: {e}"));
            }
        };

        let overhead_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.capture_overheads_ms.push(overhead_ms);

        self.frame_counter += 1;

        // Change detection against previous frame
        let (has_changed, change_ratio) = if let Some(prev) = &self.last_frame_buffer {
            compute_pixel_diff(prev, &raw_buffer)
        } else {
            (true, 1.0)
        };

        self.last_frame_buffer = Some(raw_buffer.clone());

        let bounds = Rect::new(0, 0, raw_buffer.width, raw_buffer.height);
        let frame = CapturedFrame {
            frame_id: self.frame_counter,
            source: source.clone(),
            timestamp_ns,
            buffer: raw_buffer,
            bounds,
            scale_factor: 1.0,
            has_changed,
            change_ratio,
        };

        // Bounded queue management: enforce memory limits
        if self.frame_ring_buffer.len() >= self.max_buffer_capacity {
            self.frame_ring_buffer.pop_front();
        }
        self.frame_ring_buffer.push_back(frame.clone());

        Ok(Some(frame))
    }

    /// Retrieves the most recent frame in the bounded buffer.
    pub fn latest_frame(&self) -> Option<&CapturedFrame> {
        self.frame_ring_buffer.back()
    }

    /// Current number of buffered frames.
    pub fn buffered_frames_count(&self) -> usize {
        self.frame_ring_buffer.len()
    }

    /// Mean capture overhead latency in milliseconds.
    pub fn mean_capture_overhead_ms(&self) -> f64 {
        if self.capture_overheads_ms.is_empty() {
            0.0
        } else {
            self.capture_overheads_ms.iter().sum::<f64>() / self.capture_overheads_ms.len() as f64
        }
    }
}

/// Fast subsampled pixel diff computing change ratio between consecutive frames.
fn compute_pixel_diff(a: &SharedVideoBuffer, b: &SharedVideoBuffer) -> (bool, f32) {
    if a.width != b.width || a.height != b.height || a.pixels.len() != b.pixels.len() {
        return (true, 1.0);
    }
    if a.pixels.is_empty() {
        return (false, 0.0);
    }

    let p_a = &a.pixels;
    let p_b = &b.pixels;

    // Subsample every 16th pixel for sub-millisecond change evaluation
    let step = 16 * 4; // 16 pixels * 4 bytes/pixel
    let mut total_samples = 0;
    let mut diff_samples = 0;

    let mut i = 0;
    while i + 4 <= p_a.len() {
        total_samples += 1;
        if p_a[i] != p_b[i] || p_a[i + 1] != p_b[i + 1] || p_a[i + 2] != p_b[i + 2] {
            diff_samples += 1;
        }
        i += step;
    }

    let ratio = if total_samples > 0 {
        diff_samples as f32 / total_samples as f32
    } else {
        0.0
    };

    let has_changed = ratio > 0.001; // > 0.1% pixel change threshold
    (has_changed, ratio)
}
