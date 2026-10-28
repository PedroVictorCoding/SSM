//! Visual Understanding, Region of Interest (ROI) Selection, and Multimodal Grounding.
//!
//! Provides:
//! - Visual adapter trait (`VisualEncoderAdapter`) usable interchangeably by pretrained workers
//!   (e.g. CLIP / SigLIP) and native Sail patch projectors.
//! - Grounded observations (`GroundedObservation`) linking semantic queries to spatial coordinates (`Rect`),
//!   timestamps, and explicit confidence scores, distinguishing observations from speculative hypotheses.
//! - Bounded visual token resampling (`VisualTokenResampler`) compressing large resolution frames
//!   into configurable token budgets (e.g. 16-64 tokens) without context explosion.
//! - Fusion of visual embeddings with structured UI Automation and OCR text data.
//! - Modality ablation verification ensuring visual decisions degrade when visual inputs are removed or shuffled.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::runtime::event::SharedVideoBuffer;
use crate::screen::automation::Rect;
use crate::screen::ocr::OcrBoundingBox;

/// Grounded visual observation linking perceived content to source coordinates and time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroundedObservation {
    pub observation_id: String,
    pub source: String,
    pub timestamp_ns: u64,
    pub freshness_ms: f64,
    pub region: Rect,
    pub label: String,
    pub confidence: f32,
    pub extracted_text: Option<String>,
    pub visual_features: Option<Vec<f32>>,
    pub is_hypothesis: bool,
}

/// Token budget and resampling strategy for visual embeddings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualTokenBudget {
    pub max_tokens_per_frame: usize,
    pub target_d_model: usize,
}

impl Default for VisualTokenBudget {
    fn default() -> Self {
        Self {
            max_tokens_per_frame: 32,
            target_d_model: 64,
        }
    }
}

/// Bounded visual context emitted by visual encoders.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncodedVisualContext {
    pub tokens: Vec<Vec<f32>>,
    pub token_count: usize,
    pub grounded_regions: Vec<GroundedObservation>,
    pub latency_ms: f64,
}

/// Abstract visual adapter interface for both pretrained and native Sail backends.
pub trait VisualEncoderAdapter: Send + Sync {
    fn name(&self) -> &'static str;

    /// Encodes a full frame or subregion into bounded visual embeddings.
    fn encode_frame(
        &self,
        frame: &SharedVideoBuffer,
        roi: Option<Rect>,
        budget: &VisualTokenBudget,
    ) -> Result<EncodedVisualContext, String>;

    /// Localizes a query (e.g. "error message", "submit button") to the highest-confidence bounding box.
    fn localize_query(
        &self,
        frame: &SharedVideoBuffer,
        query: &str,
        candidates: &[Rect],
        ocr_boxes: &[OcrBoundingBox],
    ) -> Result<GroundedObservation, String>;
}

/// Lightweight, deterministic local visual grounding engine.
///
/// Combines spatial color/luminance patch pooling, OCR text matching,
/// and geometric priors to ground queries to exact coordinates.
pub struct LocalVisualGroundingEngine {
    d_model: usize,
}

impl LocalVisualGroundingEngine {
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }
}

impl Default for LocalVisualGroundingEngine {
    fn default() -> Self {
        Self::new(64)
    }
}

impl VisualEncoderAdapter for LocalVisualGroundingEngine {
    fn name(&self) -> &'static str {
        "Local Grounding Engine (Patch-Pool + OCR Fusion)"
    }

    fn encode_frame(
        &self,
        frame: &SharedVideoBuffer,
        roi: Option<Rect>,
        budget: &VisualTokenBudget,
    ) -> Result<EncodedVisualContext, String> {
        let t0 = Instant::now();
        let roi_rect = roi.unwrap_or_else(|| Rect::new(0, 0, frame.width, frame.height));

        let num_tokens = budget.max_tokens_per_frame.min(64).max(1);
        let d = budget.target_d_model.min(self.d_model);

        let mut tokens = Vec::with_capacity(num_tokens);

        // Spatial grid resampling over ROI
        let grid_side = (num_tokens as f32).sqrt().ceil() as usize;
        let cell_w = (roi_rect.width as usize / grid_side).max(1);
        let cell_h = (roi_rect.height as usize / grid_side).max(1);

        for gy in 0..grid_side {
            for gx in 0..grid_side {
                if tokens.len() >= num_tokens {
                    break;
                }

                let x0 = (roi_rect.x as usize + gx * cell_w).min(frame.width as usize);
                let y0 = (roi_rect.y as usize + gy * cell_h).min(frame.height as usize);
                let x1 = (x0 + cell_w).min(frame.width as usize);
                let y1 = (y0 + cell_h).min(frame.height as usize);

                // Subsampled fast pooling (step by 4) for low-overhead inference
                let mut sum_r = 0u64;
                let mut sum_g = 0u64;
                let mut sum_b = 0u64;
                let mut count = 0u64;

                let mut y = y0;
                while y < y1 {
                    let mut x = x0;
                    while x < x1 {
                        let idx = (y * frame.width as usize + x) * 4;
                        if idx + 3 < frame.pixels.len() {
                            sum_r += frame.pixels[idx] as u64;
                            sum_g += frame.pixels[idx + 1] as u64;
                            sum_b += frame.pixels[idx + 2] as u64;
                            count += 1;
                        }
                        x += 4;
                    }
                    y += 4;
                }

                let avg_r = if count > 0 {
                    (sum_r / count) as f32 / 255.0
                } else {
                    0.0
                };
                let avg_g = if count > 0 {
                    (sum_g / count) as f32 / 255.0
                } else {
                    0.0
                };
                let avg_b = if count > 0 {
                    (sum_b / count) as f32 / 255.0
                } else {
                    0.0
                };

                // Build d-dimensional feature embedding
                let mut feat = Vec::with_capacity(d);
                for i in 0..d {
                    let harmonic = ((i as f32 * 0.25).sin() + 1.0) * 0.5;
                    let v = match i % 3 {
                        0 => avg_r * harmonic,
                        1 => avg_g * harmonic,
                        _ => avg_b * harmonic,
                    };
                    feat.push(v);
                }
                tokens.push(feat);
            }
        }

        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(EncodedVisualContext {
            token_count: tokens.len(),
            tokens,
            grounded_regions: Vec::new(),
            latency_ms,
        })
    }

    fn localize_query(
        &self,
        frame: &SharedVideoBuffer,
        query: &str,
        candidates: &[Rect],
        ocr_boxes: &[OcrBoundingBox],
    ) -> Result<GroundedObservation, String> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let is_stop = |w: &str| {
            matches!(
                w,
                "where"
                    | "is"
                    | "the"
                    | "a"
                    | "an"
                    | "in"
                    | "at"
                    | "find"
                    | "locate"
                    | "on"
                    | "of"
                    | "to"
                    | "for"
                    | "what"
                    | "which"
                    | "please"
            )
        };
        let keywords: Vec<&str> = query_words
            .iter()
            .copied()
            .filter(|&w| !is_stop(w.trim_matches(|c: char| !c.is_alphanumeric())))
            .collect();

        // 1. Text-Visual Match against OCR bounding boxes
        let mut best_ocr_match = None;
        let mut highest_ocr_score = 0.0f32;

        for obox in ocr_boxes {
            let box_text_lower = obox.text.to_lowercase();
            let mut matches = 0;
            let check_list = if !keywords.is_empty() {
                &keywords
            } else {
                &query_words
            };

            for qw in check_list {
                let clean_kw = qw.trim_matches(|c: char| !c.is_alphanumeric());
                if !clean_kw.is_empty() && box_text_lower.contains(clean_kw) {
                    matches += 1;
                }
            }
            if matches > 0 {
                let match_ratio = matches as f32 / check_list.len().max(1) as f32;
                // Boost direct match
                let score = (0.5 + 0.5 * match_ratio) * obox.confidence;
                if score > highest_ocr_score {
                    highest_ocr_score = score;
                    best_ocr_match = Some((obox.clone(), score));
                }
            }
        }

        if let Some((obox, score)) = best_ocr_match {
            return Ok(GroundedObservation {
                observation_id: format!("obs_ocr_{}", rand::random::<u16>()),
                source: "screen://frame/ocr_match".into(),
                timestamp_ns: 0,
                freshness_ms: 0.0,
                region: obox.bounds,
                label: obox.text.clone(),
                confidence: score.min(0.99),
                extracted_text: Some(obox.text),
                visual_features: None,
                is_hypothesis: false,
            });
        }

        // 2. Visual Saliency and Color Match across candidate regions
        if !candidates.is_empty() {
            let mut best_cand = candidates[0];
            let mut highest_score = -1.0f32;

            let wants_red = query_lower.contains("red");
            let wants_blue = query_lower.contains("blue");
            let wants_green = query_lower.contains("green");

            for &cand in candidates {
                let x0 = (cand.x.max(0) as usize).min(frame.width as usize);
                let y0 = (cand.y.max(0) as usize).min(frame.height as usize);
                let x1 = (x0 + cand.width as usize).min(frame.width as usize);
                let y1 = (y0 + cand.height as usize).min(frame.height as usize);

                let mut sum_r = 0u64;
                let mut sum_g = 0u64;
                let mut sum_b = 0u64;
                let mut count = 0u64;

                let mut y = y0;
                while y < y1 {
                    let mut x = x0;
                    while x < x1 {
                        let idx = (y * frame.width as usize + x) * 4;
                        if idx + 2 < frame.pixels.len() {
                            sum_r += frame.pixels[idx] as u64;
                            sum_g += frame.pixels[idx + 1] as u64;
                            sum_b += frame.pixels[idx + 2] as u64;
                            count += 1;
                        }
                        x += 4;
                    }
                    y += 4;
                }

                if count > 0 {
                    let r = (sum_r / count) as f32 / 255.0;
                    let g = (sum_g / count) as f32 / 255.0;
                    let b = (sum_b / count) as f32 / 255.0;
                    let lum = r * 0.299 + g * 0.587 + b * 0.114;

                    let mut score = lum * 0.5;
                    if wants_red {
                        let red_dominance = (r - g.max(b)).max(0.0);
                        score += red_dominance * 1.5;
                    } else if wants_blue {
                        let blue_dominance = (b - r.max(g)).max(0.0);
                        score += blue_dominance * 1.5;
                    } else if wants_green {
                        let green_dominance = (g - r.max(b)).max(0.0);
                        score += green_dominance * 1.5;
                    }

                    if score > highest_score {
                        highest_score = score;
                        best_cand = cand;
                    }
                }
            }

            return Ok(GroundedObservation {
                observation_id: format!("obs_cand_{}", rand::random::<u16>()),
                source: "screen://frame/candidate_salience".into(),
                timestamp_ns: 0,
                freshness_ms: 0.0,
                region: best_cand,
                label: format!("Candidate region for query: \"{query}\""),
                confidence: highest_score.clamp(0.35, 0.95),
                extracted_text: None,
                visual_features: None,
                is_hypothesis: false,
            });
        }

        // 3. Fallback: Full frame center region
        Ok(GroundedObservation {
            observation_id: format!("obs_fallback_{}", rand::random::<u16>()),
            source: "screen://frame/center_fallback".into(),
            timestamp_ns: 0,
            freshness_ms: 0.0,
            region: Rect::new(
                (frame.width / 4) as i32,
                (frame.height / 4) as i32,
                frame.width / 2,
                frame.height / 2,
            ),
            label: "Unlocalized full-frame observation".into(),
            confidence: 0.20,
            extracted_text: None,
            visual_features: None,
            is_hypothesis: true,
        })
    }
}

/// Verifies modality ablation: removing or shuffling pixels must degrade visual query confidence.
pub fn verify_modality_ablation(
    engine: &dyn VisualEncoderAdapter,
    original_frame: &SharedVideoBuffer,
    shuffled_frame: &SharedVideoBuffer,
    candidates: &[Rect],
    query: &str,
) -> (f32, f32, bool) {
    let obs_orig = engine
        .localize_query(original_frame, query, candidates, &[])
        .unwrap();
    let obs_shuffled = engine
        .localize_query(shuffled_frame, query, candidates, &[])
        .unwrap();

    let orig_conf = obs_orig.confidence;
    let shuffled_conf = obs_shuffled.confidence;

    // Modality ablation is confirmed if degraded/randomized pixels change the grounded region or reduce confidence
    let degrades_without_visual =
        obs_orig.region != obs_shuffled.region || (orig_conf - shuffled_conf) > 0.05;

    (orig_conf, shuffled_conf, degrades_without_visual)
}
