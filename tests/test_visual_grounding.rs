//! Integration tests for Visual Understanding, Region of Interest (ROI) Selection,
//! and Multimodal Grounding.
//!
//! Verifies acceptance criteria for GitHub Issue #12:
//! 1. Grounded questions and element/region localization on unseen screens.
//! 2. Modality ablation: removing or shuffling visual input degrades visually dependent accuracy.
//! 3. Coordinate round trips identify the correct region after resize/DPI changes.
//! 4. Bounded visual token resampling under realistic workloads with latency/memory tracking.
//! 5. Explicit distinction between verified observations and speculative hypotheses.

use sail::runtime::event::SharedVideoBuffer;
use sail::screen::automation::{CoordinateTransformer, Rect, WindowInfo};
use sail::screen::grounding::{
    LocalVisualGroundingEngine, VisualEncoderAdapter, VisualTokenBudget, verify_modality_ablation,
};
use sail::screen::ocr::OcrBoundingBox;

fn create_synthetic_editor_frame(width: u32, height: u32) -> SharedVideoBuffer {
    let mut pixels = vec![30u8; (width * height * 4) as usize]; // Dark editor theme background

    // Region 1: Terminal pane at bottom (100, 600, 800, 200) with bright red error banner
    for y in 600..800 {
        for x in 100..900 {
            let idx = (y * width + x) as usize * 4;
            if idx + 3 < pixels.len() {
                pixels[idx] = 220; // Red
                pixels[idx + 1] = 40; // Green
                pixels[idx + 2] = 40; // Blue
                pixels[idx + 3] = 255;
            }
        }
    }

    // Region 2: Active editor code tab at top (100, 50, 400, 50) with blue tab header
    for y in 50..100 {
        for x in 100..500 {
            let idx = (y * width + x) as usize * 4;
            if idx + 3 < pixels.len() {
                pixels[idx] = 40; // Red
                pixels[idx + 1] = 80; // Green
                pixels[idx + 2] = 200; // Blue
                pixels[idx + 3] = 255;
            }
        }
    }

    SharedVideoBuffer::new(width, height, "RGBA8", pixels)
}

#[test]
fn test_grounded_question_and_region_localization() {
    let engine = LocalVisualGroundingEngine::new(64);
    let frame = create_synthetic_editor_frame(1280, 720);

    let candidates = [
        Rect::new(100, 50, 400, 50),   // Code tab
        Rect::new(100, 600, 800, 200), // Error pane
        Rect::new(950, 50, 300, 500),  // File explorer
    ];

    let ocr_boxes = [
        OcrBoundingBox {
            text: "compiler error[E0308]: mismatched types".into(),
            bounds: Rect::new(120, 620, 350, 40),
            confidence: 0.98,
        },
        OcrBoundingBox {
            text: "main.rs".into(),
            bounds: Rect::new(120, 60, 80, 30),
            confidence: 0.95,
        },
    ];

    // Query 1: Locate compiler error
    let obs_error = engine
        .localize_query(
            &frame,
            "where is the compiler error?",
            &candidates,
            &ocr_boxes,
        )
        .expect("localize error");

    assert_eq!(obs_error.region, Rect::new(120, 620, 350, 40));
    assert!(
        obs_error
            .extracted_text
            .as_ref()
            .unwrap()
            .contains("error[E0308]")
    );
    assert!(obs_error.confidence > 0.80);
    assert!(
        !obs_error.is_hypothesis,
        "Direct OCR match must be an observation, not hypothesis"
    );

    // Query 2: Locate source code tab
    let obs_tab = engine
        .localize_query(&frame, "find main.rs file tab", &candidates, &ocr_boxes)
        .expect("localize tab");

    assert_eq!(obs_tab.region, Rect::new(120, 60, 80, 30));
    assert_eq!(obs_tab.label, "main.rs");
    assert!(!obs_tab.is_hypothesis);
}

#[test]
fn test_modality_ablation_verifies_visual_dependency() {
    let engine = LocalVisualGroundingEngine::new(64);
    let orig_frame = create_synthetic_editor_frame(1280, 720);

    // Create a shuffled/randomized frame of identical dimensions
    let mut shuffled_pixels = orig_frame.pixels.to_vec();
    // Invert and shuffle color channels to destroy spatial feature saliency
    for p in shuffled_pixels.chunks_exact_mut(4) {
        let tmp = p[0];
        p[0] = 255 - p[2];
        p[1] = 255 - p[1];
        p[2] = 255 - tmp;
    }
    let shuffled_frame = SharedVideoBuffer::new(1280, 720, "RGBA8", shuffled_pixels);

    let candidates = [
        Rect::new(100, 50, 400, 50),
        Rect::new(100, 600, 800, 200),
        Rect::new(950, 50, 300, 500),
    ];

    let (orig_conf, shuffled_conf, degrades) = verify_modality_ablation(
        &engine,
        &orig_frame,
        &shuffled_frame,
        &candidates,
        "locate salient red highlight",
    );

    println!(
        "Modality Ablation: Original Conf: {:.3} | Shuffled Conf: {:.3} | Degrades: {}",
        orig_conf, shuffled_conf, degrades
    );

    assert!(
        degrades,
        "Removing/shuffling visual input must degrade localization or confidence (orig: {orig_conf:.3}, shuffled: {shuffled_conf:.3})"
    );
}

#[test]
fn test_coordinate_round_trip_across_dpi_and_resize() {
    let win_1080p = WindowInfo {
        window_id: 301,
        title: "Development IDE".into(),
        process_id: 1234,
        process_name: "ide".into(),
        bounds: Rect::new(200, 150, 1600, 900),
        monitor_id: "MON_1".into(),
        display_scale: 1.0,
        is_active: true,
        is_minimized: false,
    };

    let win_4k = WindowInfo {
        window_id: 301,
        title: "Development IDE".into(),
        process_id: 1234,
        process_name: "ide".into(),
        bounds: Rect::new(400, 300, 2400, 1350),
        monitor_id: "MON_2".into(),
        display_scale: 1.5, // 1.5x DPI
        is_active: true,
        is_minimized: false,
    };

    let local_error_rect = Rect::new(120, 620, 350, 40);

    // 1. Transform to virtual desktop on 1080p
    let (desk_x, desk_y) = CoordinateTransformer::window_to_desktop(
        local_error_rect.x,
        local_error_rect.y,
        &win_1080p,
    );
    assert_eq!(desk_x, 320);
    assert_eq!(desk_y, 770);

    // 2. Adjust local rect when moved to 1.5x 4K monitor
    let adjusted_rect =
        CoordinateTransformer::adjust_for_window_movement(&local_error_rect, &win_1080p, &win_4k);
    assert_eq!(adjusted_rect.x, 180); // 120 * 1.5
    assert_eq!(adjusted_rect.y, 930); // 620 * 1.5
    assert_eq!(adjusted_rect.width, 525); // 350 * 1.5
    assert_eq!(adjusted_rect.height, 60); // 40 * 1.5

    // 3. Round-trip back to desktop on the new monitor
    let (desk_x2, desk_y2) =
        CoordinateTransformer::window_to_desktop(adjusted_rect.x, adjusted_rect.y, &win_4k);
    assert_eq!(desk_x2, 580); // 400 + 180
    assert_eq!(desk_y2, 1230); // 300 + 930
}

#[test]
fn test_bounded_visual_token_budget_and_resampling() {
    let engine = LocalVisualGroundingEngine::new(64);
    let frame = create_synthetic_editor_frame(1920, 1080);

    let budget = VisualTokenBudget {
        max_tokens_per_frame: 16,
        target_d_model: 64,
    };

    let encoded = engine
        .encode_frame(&frame, None, &budget)
        .expect("encode frame");

    assert_eq!(encoded.token_count, 16);
    assert_eq!(encoded.tokens.len(), 16);
    assert_eq!(encoded.tokens[0].len(), 64);
    println!(
        "Visual encoding latency for 1080p: {:.2} ms (Target: < 50 ms)",
        encoded.latency_ms
    );
    assert!(
        encoded.latency_ms < 50.0,
        "Encoding 1080p into 16 visual tokens must be fast, got {:.2}ms",
        encoded.latency_ms
    );
}

#[test]
fn test_observation_vs_hypothesis_distinction() {
    let engine = LocalVisualGroundingEngine::new(64);
    let frame = create_synthetic_editor_frame(640, 480);

    // Unmatched arbitrary query with no candidates and no OCR text
    let obs_speculative = engine
        .localize_query(&frame, "find nonexistent coffee cup", &[], &[])
        .expect("fallback observation");

    // Speculative ungrounded fallback MUST have is_hypothesis = true and low confidence
    assert!(
        obs_speculative.is_hypothesis,
        "Fallback ungrounded guess must be marked as a hypothesis"
    );
    assert!(
        obs_speculative.confidence <= 0.25,
        "Speculative hypothesis must carry low confidence"
    );
}
