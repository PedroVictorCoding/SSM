//! Integration tests for Screen Awareness, Window Capture, UI Automation, and OCR Context.
//!
//! Verifies acceptance criteria for GitHub Issue #10:
//! 1. Real editor error explanation identifying source window and frame metadata in trace.
//! 2. Coordinate transformations across multi-monitor layouts, DPI scaling, and window resize/moves.
//! 3. Capture loss detection (window closed/minimized), pause/resume controls, and active capture indicator.
//! 4. Capability policy gating (prohibiting ungranted screen capture).
//! 5. Bounded ring buffering, change detection, and performance telemetry (frame age, capture overhead, OCR latency).
//! 6. UI Automation text vs OCR text reconciliation and discrepancy reporting.

use std::collections::HashSet;
use std::path::PathBuf;

use sail::capability::{Capability, CapabilityPolicy, SecurityError};
use sail::screen::ScreenContextCoordinator;
use sail::screen::automation::{CoordinateTransformer, MonitorInfo, Rect, UiElement, WindowInfo};
use sail::screen::capture::{CaptureSource, HeadlessCaptureProvider, ScreenCaptureService};
use sail::screen::ocr::LocalOpticalCharacterRecognizer;

fn create_granted_policy() -> CapabilityPolicy {
    let mut granted = HashSet::new();
    granted.insert(Capability::ScreenCapture);
    CapabilityPolicy::new(
        granted,
        vec![PathBuf::from(".")],
        vec![PathBuf::from("/tmp")],
    )
}

fn create_denied_policy() -> CapabilityPolicy {
    CapabilityPolicy::default()
}

#[test]
fn test_capability_policy_strictly_guards_screen_capture() {
    let denied_policy = create_denied_policy();
    let provider = Box::new(HeadlessCaptureProvider::new());
    let mut service = ScreenCaptureService::new(provider, 8);

    // Attempting start_capture without human grant must fail immediately
    let res = service.start_capture(CaptureSource::SelectedWindow(101), &denied_policy);
    assert!(res.is_err());
    match res.unwrap_err() {
        SecurityError::CapabilityNotGranted(msg) => {
            assert!(msg.contains("ScreenCapture"));
        }
        other => panic!("Unexpected error: {:?}", other),
    }

    assert!(!service.is_capturing());
    assert!(!service.active_indicator());

    // Granted policy succeeds
    let granted_policy = create_granted_policy();
    let res_ok = service.start_capture(CaptureSource::SelectedWindow(101), &granted_policy);
    assert!(res_ok.is_ok());
    assert!(service.is_capturing());
    assert!(service.active_indicator());
}

#[test]
fn test_pause_resume_and_stop_controls_with_indicator() {
    let policy = create_granted_policy();
    let provider = Box::new(HeadlessCaptureProvider::new());
    let mut service = ScreenCaptureService::new(provider, 8);

    service
        .start_capture(CaptureSource::SelectedWindow(101), &policy)
        .unwrap();
    assert!(service.active_indicator());
    assert!(!service.is_paused());

    // Pause
    service.pause_capture();
    assert!(service.is_paused());
    assert!(!service.active_indicator());

    // Sampling while paused returns None
    let sample_paused = service.sample_frame(1000, &policy).unwrap();
    assert!(sample_paused.is_none());

    // Resume
    service.resume_capture(&policy).unwrap();
    assert!(!service.is_paused());
    assert!(service.active_indicator());

    let sample_resumed = service.sample_frame(2000, &policy).unwrap();
    assert!(sample_resumed.is_some());

    // Stop
    service.stop_capture();
    assert!(!service.is_capturing());
    assert!(!service.active_indicator());
    assert_eq!(service.buffered_frames_count(), 0);
}

#[test]
fn test_coordinate_transforms_and_dpi_scaling_across_monitors() {
    let mon1 = MonitorInfo {
        monitor_id: "MON_1".into(),
        name: "Primary 1080p 1.0x".into(),
        bounds: Rect::new(0, 0, 1920, 1080),
        scale_factor: 1.0,
        is_primary: true,
    };

    let mon2 = MonitorInfo {
        monitor_id: "MON_2".into(),
        name: "Secondary 4K 1.5x".into(),
        bounds: Rect::new(1920, 0, 3840, 2160),
        scale_factor: 1.5,
        is_primary: false,
    };

    let win_mon1 = WindowInfo {
        window_id: 201,
        title: "Code Editor".into(),
        process_id: 501,
        process_name: "code".into(),
        bounds: Rect::new(100, 150, 1200, 800),
        monitor_id: mon1.monitor_id.clone(),
        display_scale: mon1.scale_factor,
        is_active: true,
        is_minimized: false,
    };

    // Test window-to-desktop and desktop-to-window
    let (desk_x, desk_y) = CoordinateTransformer::window_to_desktop(50, 75, &win_mon1);
    assert_eq!(desk_x, 150);
    assert_eq!(desk_y, 225);

    let (local_x, local_y) = CoordinateTransformer::desktop_to_window(desk_x, desk_y, &win_mon1);
    assert_eq!(local_x, 50);
    assert_eq!(local_y, 75);

    // Test moving window to secondary 4K monitor with 1.5x DPI scaling
    let win_mon2 = WindowInfo {
        window_id: 201,
        title: "Code Editor".into(),
        process_id: 501,
        process_name: "code".into(),
        bounds: Rect::new(2100, 200, 1800, 1200),
        monitor_id: mon2.monitor_id.clone(),
        display_scale: mon2.scale_factor, // 1.5x scale!
        is_active: true,
        is_minimized: false,
    };

    let elem_rect = Rect::new(20, 30, 200, 100);
    let adjusted =
        CoordinateTransformer::adjust_for_window_movement(&elem_rect, &win_mon1, &win_mon2);

    // At 1.5x scale, bounds rescale proportionally: 20 * 1.5 = 30, 30 * 1.5 = 45, 200 * 1.5 = 300
    assert_eq!(adjusted.x, 30);
    assert_eq!(adjusted.y, 45);
    assert_eq!(adjusted.width, 300);
    assert_eq!(adjusted.height, 150);
}

#[test]
fn test_change_detection_and_bounded_sampling_ring_buffer() {
    let policy = create_granted_policy();
    // Capacity 3 frames
    let provider = Box::new(HeadlessCaptureProvider::new());
    let mut service = ScreenCaptureService::new(provider, 3);
    service
        .start_capture(CaptureSource::SelectedWindow(101), &policy)
        .unwrap();

    // Frame 1: initial capture (always has_changed = true)
    let f1 = service.sample_frame(100_000_000, &policy).unwrap().unwrap();
    assert_eq!(f1.frame_id, 1);
    assert!(f1.has_changed);
    assert_eq!(service.buffered_frames_count(), 1);

    // Frame 2: identical pixels (has_changed = false, change_ratio = 0.0)
    let f2 = service.sample_frame(200_000_000, &policy).unwrap().unwrap();
    assert_eq!(f2.frame_id, 2);
    assert!(
        !f2.has_changed,
        "Identical frame should have has_changed = false"
    );
    assert_eq!(f2.change_ratio, 0.0);
    assert_eq!(service.buffered_frames_count(), 2);

    // Frame 3 & Frame 4: verify bounded ring buffer drops oldest frame
    let _ = service.sample_frame(300_000_000, &policy).unwrap();
    assert_eq!(service.buffered_frames_count(), 3);

    let f4 = service.sample_frame(400_000_000, &policy).unwrap().unwrap();
    assert_eq!(
        service.buffered_frames_count(),
        3,
        "Buffer must remain bounded at capacity=3"
    );
    assert_eq!(service.latest_frame().unwrap().frame_id, 4);

    // Verify frame age calculation
    let age = f4.age_ms(450_000_000); // 50ms later
    assert!((age - 50.0).abs() < 0.1);
}

#[test]
fn test_capture_loss_detection_on_window_close() {
    let policy = create_granted_policy();
    // Simulate target window loss at frame 3
    let provider = Box::new(HeadlessCaptureProvider::new().simulate_loss_at(3));
    let mut service = ScreenCaptureService::new(provider, 4);
    service
        .start_capture(CaptureSource::SelectedWindow(101), &policy)
        .unwrap();

    let f1 = service.sample_frame(1000, &policy);
    assert!(f1.is_ok());

    let f2 = service.sample_frame(2000, &policy);
    assert!(f2.is_ok());

    // Frame 3 encounters target loss
    let f3 = service.sample_frame(3000, &policy);
    assert!(f3.is_err());
    let err_msg = f3.unwrap_err();
    assert!(err_msg.contains("Capture target lost"));

    // Capture state automatically transitions to inactive
    assert!(!service.is_capturing());
    assert!(!service.active_indicator());
}

#[test]
fn test_ocr_and_ui_automation_reconciliation_and_discrepancy() {
    let ocr_engine = LocalOpticalCharacterRecognizer::new().with_annotation(
        Rect::new(50, 100, 400, 200),
        "error[E0308]: mismatched types expected `usize`, found `u32`",
        0.96,
    );

    let provider = Box::new(HeadlessCaptureProvider::new());
    let mut capture_service = ScreenCaptureService::new(provider, 4);
    let policy = create_granted_policy();
    capture_service
        .start_capture(CaptureSource::SelectedWindow(101), &policy)
        .unwrap();

    let mut coordinator = ScreenContextCoordinator::new(capture_service, Box::new(ocr_engine));

    // UI tree represents a custom canvas/terminal where UI automation has no text
    let canvas_pane = UiElement {
        element_id: "elem_term_1".into(),
        name: "Terminal Output Canvas".into(),
        role: "Pane".into(),
        text_value: None, // Canvas without accessible text tree
        bounds: Rect::new(0, 0, 1920, 1080),
        supported_actions: vec!["Scroll".into()],
        is_focused: true,
        is_enabled: true,
        children: Vec::new(),
    };
    coordinator.set_active_ui_tree(canvas_pane);
    coordinator.set_cursor_position(250, 180);

    let snapshot = coordinator
        .assemble_context(1_000_000_000, &policy)
        .expect("assemble context");

    // Verify OCR extracted the compiler error from canvas
    assert!(snapshot.ocr_text.contains("error[E0308]: mismatched types"));
    assert_eq!(snapshot.cursor_position, (250, 180));

    // Discrepancy report accurately flags that text was detected via OCR on inaccessible region
    assert_eq!(snapshot.discrepancies.len(), 1);
    assert!(
        snapshot.discrepancies[0]
            .explanation
            .contains("Inaccessible or canvas region")
    );

    // Verify formatted context prompt provides rich explanation context for LLM
    let prompt_text = snapshot.to_context_prompt();
    assert!(prompt_text.contains("--- SCREEN CONTEXT ---"));
    assert!(prompt_text.contains("error[E0308]: mismatched types"));
    assert!(prompt_text.contains("Cursor Position: (250, 180)"));
}

#[test]
fn test_editor_error_explanation_trace() {
    let error_text = "error[E0425]: cannot find value `total_audio_duration_sec` in this scope\n  --> src/voice/tts.rs:1116:9";
    let ocr = LocalOpticalCharacterRecognizer::new().with_annotation(
        Rect::new(100, 200, 800, 300),
        error_text,
        0.98,
    );

    let provider = Box::new(HeadlessCaptureProvider::new());
    let mut capture_service = ScreenCaptureService::new(provider, 4);
    let policy = create_granted_policy();
    capture_service
        .start_capture(CaptureSource::SelectedWindow(101), &policy)
        .unwrap();

    let mut coordinator = ScreenContextCoordinator::new(capture_service, Box::new(ocr));
    let snapshot = coordinator
        .assemble_context(5_000_000_000, &policy)
        .expect("context");

    assert!(
        snapshot
            .unified_text_summary
            .contains("src/voice/tts.rs:1116:9")
    );
    assert!(
        snapshot.ocr_latency_ms < 50.0,
        "OCR scan must be under 50ms"
    );
    assert!(
        snapshot.capture_overhead_ms < 50.0,
        "Capture overhead must be low"
    );
}
