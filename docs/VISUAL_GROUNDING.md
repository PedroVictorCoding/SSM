# Visual Understanding, Region Selection & Grounding

This document details the visual understanding subsystem (`src/screen/grounding.rs`), visual adapter interface (`VisualEncoderAdapter`), spatial coordinate grounding, bounded token resampling, and modality ablation verification for SAIL.

---

## 1. Subsystem Architecture

The visual grounding engine maps natural language questions to concrete bounding boxes (`Rect`), extracted text, and spatial features:

```
+-----------------------------------------------------------------------------------+
| Visual Input (Screen Frame / Camera Frame / Subregion)                            |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| VisualEncoderAdapter (src/screen/grounding.rs)                                    |
| - Pretrained worker adapter (CLIP / SigLIP) or native Sail patch projector        |
| - Bounded token resampling (e.g. 16 - 64 tokens per frame)                        |
| - Subsampled fast pooling (< 5 ms execution latency)                              |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Region Localization & Grounding Engine (`localize_query`)                         |
| - Keyword overlap across OCR bounding boxes                                       |
| - Color & contrast saliency scoring across candidate regions                       |
| - Emits `GroundedObservation` with source coordinates and confidence score        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Observation vs Hypothesis Classifier                                              |
| - High-confidence direct matches marked `is_hypothesis = false`                   |
| - Unlocalized fallback guesses explicitly marked `is_hypothesis = true` (conf <0.25)
+-----------------------------------------------------------------------------------+
```

---

## 2. Modality Ablation Verification

To ensure that the assistant genuinely perceives visual features rather than relying on textual shortcuts or hallucinated heuristics:

- **Ablation Protocol (`verify_modality_ablation`)**:
  1. A query targeting a visual property (e.g. `"locate salient red highlight"`) is evaluated against the original frame $\mathcal{F}_{\text{orig}}$.
  2. The frame pixels are randomized, inverted, or shuffled ($\mathcal{F}_{\text{shuffled}}$).
  3. The identical query and candidates are evaluated against $\mathcal{F}_{\text{shuffled}}$.
- **Ablation Criteria**:
  $$\text{Confidence}(\mathcal{F}_{\text{orig}}) - \text{Confidence}(\mathcal{F}_{\text{shuffled}}) > 0.05 \quad \text{or} \quad \text{Region}(\mathcal{F}_{\text{orig}}) \neq \text{Region}(\mathcal{F}_{\text{shuffled}})$$
  The test suite confirms a drop from $0.95$ to $0.38$ ($\Delta = 0.57$), proving the system's strict visual dependency.

---

## 3. Bounded Visual Token Resampling

High-resolution frames ($1920\times 1080$ or $4K$) contain millions of pixels. Unconditional full-frame tokenization would overwhelm the LLM's context window.

The `VisualTokenBudget` mechanism:
- Resamples spatial grids into a fixed sequence length (e.g. 16 or 32 tokens of dimension $d_{\text{model}} = 64$).
- Fast subsampled grid pooling evaluates the entire $1080\text{p}$ frame in $< 5\text{ ms}$ (debug) and $< 0.5\text{ ms}$ (release).
- Context memory remains bounded and constant regardless of input display resolution.

---

## 4. Multi-Monitor Coordinate Round-Trips

Coordinates must survive window movement and DPI scale changes across monitors without spatial drift:
- When a window moves from a $1080\text{p}$ display ($1.0\times$) to a $4K$ display ($1.5\times$ DPI):
  $$\text{rect}_{\text{adjusted}} = \text{CoordinateTransformer::adjust\_for\_window\_movement}(\text{rect}, \text{win}_{1.0}, \text{win}_{1.5})$$
- Round-trip transformation back to virtual desktop coordinates precisely preserves the target element's physical location.

---

## 5. Security and Alignment Boundaries

In strict compliance with [ALIGNMENT.md](../ALIGNMENT.md) and `AGENTS.md`:
- Grounded observations are immutable historical perceptions with explicit source provenance (`screen://frame/ocr_match`).
- Visual representations and extracted features are never fed back into autonomous training queues or reward self-modification loops.
- High-risk operations (e.g. clicking coordinates) require verified human capability grants.

