# Speech Synthesis (TTS) Architecture & Backend Support Matrix

This document describes the streaming synthesis prototypes in `src/voice/tts.rs`,
prefix commitment, alignment metadata and the remaining product verification gates.
The implemented formant engine uses hand-coded phonetic rules and filters; the
acoustic-token decoder is a tone-generating stub. Neither loads trained neural
speech weights. Public type names are retained for API compatibility.

---

## 1. Architecture Overview

The streaming interface can deliver PCM from committed language-model text while
generation continues. Naturalness and intelligibility require separate evaluation.

```
+-----------------------------------------------------------------------------+
| LLM Token Stream (e.g., "The", " system", " is", " online.")                |
+-----------------------------------------------------------------------------+
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
| TextPrefixCommitmentGate (src/voice/tts.rs)                                 |
| - Buffers token fragments ("Hel" + "lo" -> "Hello")                         |
| - Commits whole words at whitespace and punctuation boundaries              |
| - Prevents speculative or revised tokens from being spoken twice            |
+-----------------------------------------------------------------------------+
                                       │ Committed Word Sequence
                                       ▼
+-----------------------------------------------------------------------------+
| StreamingTtsAdapter (formant prototype / acoustic-token decoder trait)     |
| - G2P Conversion: Grapheme-to-Phoneme mapping                               |
| - Prosodic F0 Contour: Pitch declination, phrase stress, question rise      |
| - Resonant IIR Formant Synthesis: F1, F2, F3 bandpass cascade               |
| - Unvoiced Excitation: Colored pseudo-random noise for fricatives/plosives  |
| - Coarticulation: Continuous formant smoothing across 20-30ms transitions   |
+-----------------------------------------------------------------------------+
                                       │ SynthesizedAudioChunk (Arc<[f32]>)
                                       ▼
+-----------------------------------------------------------------------------+
| PlaybackPipeline (src/voice.rs)                                            |
| - Non-blocking speaker callback (`pull_samples`)                             |
| - Playback cursor tracking & AEC loopback reference                         |
| - Real-time interruption & barge-in support (`interrupt`)                   |
| - Bounded queue preventing memory exhaustion on long responses              |
+-----------------------------------------------------------------------------+
```

---

## 2. Backend Support Matrix

| Backend Adapter | Type | Dependencies | Latency (First Chunk) | Real-Time Factor (RTF) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`FormantNeuralTtsEngine`** | Hand-coded formant / phonetic synthesizer | Pure Rust | No archived target-device measurement | No archived target-device measurement | **Prototype; intelligibility unverified** |
| **`SailNeuralVocoder`** | Acoustic-code-to-tone stub | Pure Rust | Unmeasured | Unmeasured | **Test prototype; not a learned codec decoder** |
| **`NeuralAcousticDecoder`** | Proposed learned decoder | Dependency/model to be selected | Unmeasured | Unmeasured | **Roadmap; no implementation established here** |

---

## 3. Targets and available evidence

The current tests exercise generated PCM, prefix handling, word/sample bookkeeping
and headless timing. They do not establish target-device speaker latency, natural
speech quality or a learned vocoder. Earlier numerical measurements in this table
had no linked raw benchmark artifact and are not retained as verified results.

| Metric | Proposed target / check | Current evidence and remaining verification |
| :--- | :--- | :--- |
| **First-audio latency** | <50 ms synthesis-side target | Archive raw timing on specified backend/hardware; measure device playback separately. |
| **Sustained real-time factor** | <0.50 | Measure a representative corpus with the actual selected backend/model. |
| **Buffer memory growth** | Explicit byte/sample bounds | Chunk-count tests alone need payload-size and sustained-load checks. |
| **Cancellation latency** | <5 ms control-side target | Queue purge timing does not include already submitted device audio. |
| **Intelligibility** | Understandable speech on held-out text | Waveform zero-crossing differences demonstrate different signals, not intelligibility. Listening/independent transcription evaluation remains open. |

Related implementation and acceptance requests are recorded under QA-11 in
[the shared review ledger](../LLM_HANDOFF.md).

---

## 4. Key Mechanisms

### 4.1. Text Prefix Commitment Gate
Prevents speech stuttering and duplicate speech turns:
- Token streams emitted by LLMs often arrive in partial word fragments (e.g. `"Hel"`, `"lo "`).
- The `TextPrefixCommitmentGate` accumulates fragments until a word or punctuation boundary (`' '`, `'.' `, `','`, `'!' `, `'?' `, `';'`) is encountered.
- Only committed, immutable word prefixes are dispatched to the acoustic synthesizer.
- Trailing text is cleanly flushed at turn end via `flush()`.

### 4.2. Word-Level Timing Alignment & Mid-Utterance Interruption
For conversational assistants, user barge-in requires knowing exactly which words were spoken:
- Each `SynthesizedAudioChunk` carries `WordAlignment` metadata detailing `word`, `start_sample`, `end_sample`, `start_ms`, and `end_ms`.
- When the user interrupts (detected by VAD or endpoint detector):
  1. The host runtime calls `playback_pipeline.interrupt()`, which instantly flushes speaker buffers and returns the exact playback sample cursor.
  2. The assistant coordinator calls `engine.cancel_utterance(utterance_id, playback_cursor)`.
  3. The engine generates a `TtsCancellationReport`:
     - `played_words`: Words preceding the reported software playback cursor.
     - `discarded_samples`: Audio frames that were synthesized but never heard.
     - `interrupted_at_ms`: Monotonic millisecond timestamp of the barge-in.
  4. The coordinator can retain the reported played prefix. Relating that software
     cursor to audio physically rendered by the device requires accounting for
     device buffering and playback latency; actual heard-prefix behavior remains
     a separate end-to-end acceptance check.

---

## 5. Alignment & Governance Invariant

In strict adherence to [ALIGNMENT.md](../ALIGNMENT.md) and `AGENTS.md`:
- Speech synthesis is purely an inference output modality.
- Voice configurations and speech policies are strictly human-controlled.
- Synthesized speech outputs are never automatically fed back into model training queues or self-reinforcement loops.

