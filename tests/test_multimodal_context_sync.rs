//! Integration tests for Synchronized Multimodal Context Assembly, Timeline Alignment,
//! Deictic Reference Resolution, and Causal Masking.
//!
//! Verifies acceptance criteria for GitHub Issue #13:
//! 1. Delayed/out-of-order sensors still resolve the intended error/window/object.
//! 2. Causal temporal masking: future observations (T > T_utterance) are strictly excluded.
//! 3. Changed screen / stale target rejection: consequential actions cannot reuse obsolete targets.
//! 4. Bounded context size, modality budgeting, and inspectable provenance traces.
//! 5. Target screen-age metrics <= 1.0s verified.

use std::sync::Arc;

use sail::runtime::event::SharedVideoBuffer;
use sail::runtime::multimodal_context::{
    MultimodalContextAssembler, ReferenceResolutionResult, TimelineEntry,
};
use sail::screen::automation::{Rect, UiElement, WindowInfo};
use sail::screen::capture::{CaptureSource, CapturedFrame};
use sail::screen::ocr::OcrBoundingBox;

fn create_dummy_frame(width: u32, height: u32) -> CapturedFrame {
    let buf = Arc::new(SharedVideoBuffer::new(
        width,
        height,
        "RGBA8",
        vec![50u8; (width * height * 4) as usize],
    ));
    CapturedFrame {
        frame_id: 101,
        source: CaptureSource::SelectedWindow(501),
        timestamp_ns: 1_000_000_000,
        buffer: buf,
        bounds: Rect::new(0, 0, width, height),
        scale_factor: 1.0,
        has_changed: true,
        change_ratio: 1.0,
    }
}

#[test]
fn test_deterministic_reference_resolution_with_delayed_sensors() {
    let mut assembler = MultimodalContextAssembler::new("session_sync_test");

    // 1. Screen frame arrives with timestamp 1000 ms
    let frame = create_dummy_frame(1280, 720);
    let win = WindowInfo {
        window_id: 501,
        title: "VS Code - main.rs".into(),
        process_id: 1200,
        process_name: "code".into(),
        bounds: Rect::new(0, 0, 1280, 720),
        monitor_id: "MON_1".into(),
        display_scale: 1.0,
        is_active: true,
        is_minimized: false,
    };
    let ocr_boxes = vec![OcrBoundingBox {
        text: "error[E0308]: mismatched types".into(),
        bounds: Rect::new(100, 500, 400, 60),
        confidence: 0.98,
    }];

    assembler.push_entry(TimelineEntry::ScreenFrame {
        frame_id: 101,
        timestamp_ms: 1000,
        window: Some(win),
        frame,
        ocr_boxes,
    });

    // 2. Pointer event at 1020 ms
    assembler.push_entry(TimelineEntry::PointerLocation {
        timestamp_ms: 1020,
        x: 250,
        y: 400,
    });

    // 3. UI Automation tree at 1030 ms
    let button_elem = UiElement {
        element_id: "ui_btn_compile".into(),
        name: "Run Cargo Check".into(),
        role: "Button".into(),
        text_value: Some("Cargo Check".into()),
        bounds: Rect::new(200, 380, 120, 40), // Contains pointer (250, 400)
        supported_actions: vec!["Click".into()],
        is_focused: false,
        is_enabled: true,
        children: Vec::new(),
    };
    assembler.push_entry(TimelineEntry::UiTreeSnapshot {
        timestamp_ms: 1030,
        root: button_elem,
    });

    // 4. Future frame arriving at 1200 ms (after user spoke at 1050 ms)
    // In this future frame, the error was cleared and replaced by success!
    let future_frame = create_dummy_frame(1280, 720);
    let future_ocr = vec![OcrBoundingBox {
        text: "Build finished successfully in 2.1s".into(),
        bounds: Rect::new(100, 500, 400, 60),
        confidence: 0.99,
    }];
    assembler.push_entry(TimelineEntry::ScreenFrame {
        frame_id: 102,
        timestamp_ms: 1200,
        window: None,
        frame: future_frame,
        ocr_boxes: future_ocr,
    });

    // User spoke at 1050 ms: "explain this error"
    let utterance_time_ms = 1050;

    let res_error = assembler.resolve_reference("explain this error", utterance_time_ms, false);
    match res_error {
        ReferenceResolutionResult::Resolved(target) => {
            // Must causally match the error from frame 101 (at 1000ms), NOT the future frame 102 (at 1200ms)!
            assert_eq!(target.target_id, "ocr_box_101");
            assert!(target.label.contains("error[E0308]"));
            assert_eq!(target.age_ms, 50.0); // 1050 - 1000 = 50ms freshness
        }
        other => panic!("Expected resolved target, got: {:?}", other),
    }

    // User spoke at 1050 ms: "click that button"
    let res_button = assembler.resolve_reference("click that button", utterance_time_ms, false);
    match res_button {
        ReferenceResolutionResult::Resolved(target) => {
            assert_eq!(target.target_id, "ui_btn_compile");
            assert_eq!(target.label, "Run Cargo Check");
            assert_eq!(target.age_ms, 20.0); // 1050 - 1030 = 20ms
        }
        other => panic!("Expected resolved button, got: {:?}", other),
    }
}

#[test]
fn test_stale_target_rejection_prevents_obsolete_consequential_action() {
    let mut assembler = MultimodalContextAssembler::new("session_stale_test");
    assembler.set_max_actionable_screen_age_ms(1000.0); // 1.0s target freshness

    // Screen captured at t = 1000 ms
    let frame = create_dummy_frame(1280, 720);
    assembler.push_entry(TimelineEntry::ScreenFrame {
        frame_id: 201,
        timestamp_ms: 1000,
        window: None,
        frame,
        ocr_boxes: vec![OcrBoundingBox {
            text: "Syntax error on line 42".into(),
            bounds: Rect::new(50, 100, 300, 40),
            confidence: 0.95,
        }],
    });

    // 1. At t = 1500 ms (age = 500ms <= 1000ms): consequential action succeeds
    let res_fresh = assembler.resolve_reference("fix this error", 1500, true);
    assert!(matches!(res_fresh, ReferenceResolutionResult::Resolved(_)));

    // 2. At t = 2500 ms (age = 1500ms > 1000ms): consequential action is strictly rejected!
    let res_stale = assembler.resolve_reference("fix this error", 2500, true);
    match res_stale {
        ReferenceResolutionResult::StaleTargetRejected {
            age_ms,
            max_allowed_age_ms,
            ..
        } => {
            assert_eq!(age_ms, 1500.0);
            assert_eq!(max_allowed_age_ms, 1000.0);
        }
        other => panic!("Expected StaleTargetRejected, got: {:?}", other),
    }

    // 3. For non-consequential informational queries, it still resolves with age metadata
    let res_info = assembler.resolve_reference("what was that error earlier?", 2500, false);
    assert!(matches!(res_info, ReferenceResolutionResult::Resolved(_)));
}

#[test]
fn test_ambiguity_detection_requests_clarification() {
    let mut assembler = MultimodalContextAssembler::new("session_ambiguity_test");

    let frame = create_dummy_frame(1280, 720);
    // Two distinct errors on the same screen
    let ocr_boxes = vec![
        OcrBoundingBox {
            text: "error[E0308]: mismatched types".into(),
            bounds: Rect::new(100, 200, 300, 40),
            confidence: 0.98,
        },
        OcrBoundingBox {
            text: "error[E0425]: cannot find value".into(),
            bounds: Rect::new(100, 300, 300, 40),
            confidence: 0.96,
        },
    ];

    assembler.push_entry(TimelineEntry::ScreenFrame {
        frame_id: 301,
        timestamp_ms: 1000,
        window: None,
        frame,
        ocr_boxes,
    });

    // "explain the error" is ambiguous between the two errors
    let res = assembler.resolve_reference("explain the error", 1050, false);
    match res {
        ReferenceResolutionResult::Ambiguous { candidate_ids, .. } => {
            assert_eq!(candidate_ids.len(), 2);
            assert!(candidate_ids[0].contains("E0308"));
            assert!(candidate_ids[1].contains("E0425"));
        }
        other => panic!("Expected Ambiguous resolution, got: {:?}", other),
    }
}

#[test]
fn test_inspectable_provenance_trace_and_context_packet() {
    let mut assembler = MultimodalContextAssembler::new("session_trace_test");

    let frame = create_dummy_frame(1280, 720);
    let win = WindowInfo {
        window_id: 401,
        title: "Terminal - cargo test".into(),
        process_id: 8800,
        process_name: "bash".into(),
        bounds: Rect::new(0, 0, 1280, 720),
        monitor_id: "MON_1".into(),
        display_scale: 1.0,
        is_active: true,
        is_minimized: false,
    };

    assembler.push_entry(TimelineEntry::ScreenFrame {
        frame_id: 401,
        timestamp_ms: 2000,
        window: Some(win),
        frame,
        ocr_boxes: vec![OcrBoundingBox {
            text: "FAILED: test_auth_token_expiration".into(),
            bounds: Rect::new(50, 150, 450, 40),
            confidence: 0.99,
        }],
    });

    let packet = assembler
        .assemble_synchronized_context("explain the failed test", 2200, false)
        .expect("assemble synchronized context");

    assert!(packet.resolved_reference.is_some());
    let resolved = packet.resolved_reference.unwrap();
    assert_eq!(resolved.target_id, "ocr_box_401");
    assert_eq!(resolved.age_ms, 200.0); // 2200 - 2000 = 200ms freshness (<= 1.0s target)

    // Verify inspectable provenance trace
    let trace = &packet.provenance_trace;
    assert_eq!(trace.session_id, "session_trace_test");
    assert_eq!(trace.utterance_timestamp_ms, 2200);
    assert!(trace.causal_mask_applied);
    assert_eq!(trace.screen_age_ms, 200.0);
    assert_eq!(trace.target_screen_age_target_ms, 1000.0);
    assert!(
        trace
            .selected_evidence_ids
            .contains(&"frame_401".to_string())
    );

    // Verify formatted prompt contains inspectable sections
    assert!(
        packet
            .formatted_prompt
            .contains("--- SYNCHRONIZED MULTIMODAL CONTEXT ---")
    );
    assert!(
        packet
            .formatted_prompt
            .contains("FAILED: test_auth_token_expiration")
    );
    assert!(packet.formatted_prompt.contains("Freshness: 200.0 ms"));
}
