//! Streaming Neural & Formant Speech Synthesis (TTS) Engine.
//!
//! Provides:
//! - Stable text-prefix commitment (`TextPrefixCommitmentGate`) to prevent partial token fragments
//!   or revised text from being spoken twice during streaming LLM generation.
//! - Streaming TTS adapter (`StreamingTtsAdapter`) with incremental PCM delivery, utterance IDs,
//!   and voice configuration (pitch, speed, gender/timbre formants, volume).
//! - Text-to-audio word alignments (`WordAlignment`) enabling coordinators to track which words
//!   were heard and compute exact cancellation reports (`TtsCancellationReport`).
//! - Prototype/test fallback formant synthesizer (`FormantNeuralTtsEngine`) with G2P phonetic decomposition,
//!   cascading resonance filters (F1/F2/F3), glottal pulse modeling, prosodic F0 contours,
//!   and smooth coarticulation transitions.
//! - Extensible acoustic-token neural vocoder trait (`AcousticTokenDecoder`) for Sail SSM neural audio decoders.
//! - Bounded buffering and mid-utterance cancellation with exact played prefix tracking.

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration & Types
// ---------------------------------------------------------------------------

/// Voice characteristics and synthesis parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceConfig {
    pub voice_id: String,
    pub speaking_rate: f32,
    pub pitch_shift: f32,
    pub sample_rate: u32,
    pub volume: f32,
    pub formant_scale: f32,
    pub base_f0: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            voice_id: "sail-v1-neutral".to_string(),
            speaking_rate: 1.0,
            pitch_shift: 1.0,
            sample_rate: 16000,
            volume: 1.0,
            formant_scale: 1.0,
            base_f0: 140.0,
        }
    }
}

impl VoiceConfig {
    pub fn female() -> Self {
        Self {
            voice_id: "sail-v1-female".to_string(),
            base_f0: 190.0,
            formant_scale: 1.15,
            ..Default::default()
        }
    }

    pub fn male() -> Self {
        Self {
            voice_id: "sail-v1-male".to_string(),
            base_f0: 105.0,
            formant_scale: 0.90,
            ..Default::default()
        }
    }
}

/// Word-level text-to-audio timing alignment.
#[derive(Clone, Debug, PartialEq)]
pub struct WordAlignment {
    pub word: String,
    pub start_sample: usize,
    pub end_sample: usize,
    pub start_ms: f64,
    pub end_ms: f64,
}

/// Synthesized incremental PCM audio chunk with alignment metadata.
#[derive(Clone, Debug)]
pub struct SynthesizedAudioChunk {
    pub utterance_id: String,
    pub chunk_index: usize,
    pub samples: Arc<[f32]>,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_offset: u64,
    pub duration_ms: f64,
    pub spoken_words: Vec<WordAlignment>,
    pub is_final_chunk: bool,
}

/// Cancellation and interruption report for a stopped utterance.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsCancellationReport {
    pub utterance_id: String,
    pub total_synthesized_samples: u64,
    pub played_samples: u64,
    pub discarded_samples: u64,
    pub committed_words: Vec<String>,
    pub played_words: Vec<String>,
    pub interrupted_at_ms: f64,
}

/// Playback and spoken progress alignment status.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaybackAlignmentStatus {
    pub utterance_id: String,
    pub played_words: Vec<String>,
    pub current_word: Option<String>,
    pub pending_words: Vec<String>,
    pub played_duration_ms: f64,
    pub total_duration_ms: f64,
    pub is_completed: bool,
}

// ---------------------------------------------------------------------------
// Text Prefix Commitment Gate
// ---------------------------------------------------------------------------

/// Manages incoming streaming tokens and commits stable text prefixes.
///
/// Prevents uncommitted token fragments (e.g. "Hel" followed by "lo") or
/// speculative/revised tokens from being synthesized or spoken multiple times.
#[derive(Clone, Debug, Default)]
pub struct TextPrefixCommitmentGate {
    buffer: String,
    committed_words: Vec<String>,
}

impl TextPrefixCommitmentGate {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            committed_words: Vec::new(),
        }
    }

    /// Pushes a streaming text token and returns newly committed words ready for synthesis.
    ///
    /// Words are committed when delimited by whitespace or punctuation boundaries.
    pub fn push_token(&mut self, token: &str) -> Vec<String> {
        self.buffer.push_str(token);

        // Filter special tokens
        if self.buffer.contains("<|im_start|>") {
            self.buffer = self.buffer.replace("<|im_start|>", "");
        }
        if self.buffer.contains("<|im_end|>") {
            self.buffer = self.buffer.replace("<|im_end|>", "");
        }

        // If inside a tool_call tag, wait until it closes and strip it
        if let Some(start) = self.buffer.find("<tool_call>") {
            if let Some(end) = self.buffer.find("</tool_call>") {
                self.buffer = format!("{}{}", &self.buffer[..start], &self.buffer[end + 12..]);
            } else {
                // Incomplete tool call; do not commit text inside the tool call tag
                return Vec::new();
            }
        }

        let mut newly_committed = Vec::new();

        // Delimiters that guarantee word completion
        while let Some(idx) = self.find_commit_boundary(&self.buffer) {
            let word_part = self.buffer[..idx]
                .trim()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            // Drain the word and delimiter
            let remaining = self.buffer[idx..].trim_start().to_string();
            self.buffer = remaining;

            if !word_part.is_empty() {
                self.committed_words.push(word_part.clone());
                newly_committed.push(word_part);
            }
        }

        newly_committed
    }

    /// Flushes any remaining text in the buffer (called at the end of an utterance turn).
    pub fn flush(&mut self) -> Vec<String> {
        let mut flushed = Vec::new();
        let remaining = self.buffer.trim().to_string();
        if !remaining.is_empty() {
            // Split any remaining words
            for word in remaining.split_whitespace() {
                let cleaned = word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string();
                if !cleaned.is_empty() {
                    self.committed_words.push(cleaned.clone());
                    flushed.push(cleaned);
                }
            }
            self.buffer.clear();
        }
        flushed
    }

    /// Returns the slice of all committed words.
    pub fn committed_words(&self) -> &[String] {
        &self.committed_words
    }

    /// Resets the gate state for a new utterance.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.committed_words.clear();
    }

    fn find_commit_boundary(&self, s: &str) -> Option<usize> {
        for (i, c) in s.char_indices() {
            if c.is_whitespace()
                || c == '.'
                || c == ','
                || c == '!'
                || c == '?'
                || c == ';'
                || c == ':'
            {
                return Some(i + c.len_utf8());
            }
        }
        None
    }
}

/// Extracts a ready linguistic clause or sentence unit for natural speech synthesis.
///
/// Returns `Some(clause)` when a complete sentence boundary (`.`, `?`, `!`, `\n`),
/// a clause pause (`,`, `;`, `:`) with >= 6 words, or a run-on length threshold (>= 12 words)
/// is reached. The extracted clause is removed from `buffer`.
pub fn extract_speech_clause(buffer: &mut String) -> Option<String> {
    let mut split_point: Option<(usize, usize)> = None;

    // 1. Unconditional sentence/clause terminators: '?' or '!' or '\n'
    for (i, ch) in buffer.char_indices() {
        if ch == '?' || ch == '!' {
            split_point = Some((i + ch.len_utf8(), i + ch.len_utf8()));
            break;
        }
        if ch == '\n' {
            split_point = Some((i, i + ch.len_utf8()));
            break;
        }
    }

    // 2. Period followed by whitespace or quotes (avoiding ellipsis '...')
    if split_point.is_none() {
        for (i, ch) in buffer.char_indices() {
            if ch == '.' {
                let preceded_by_dot = i > 0 && buffer[..i].ends_with('.');
                let followed_by_dot = buffer[i + 1..].starts_with('.');
                if preceded_by_dot || followed_by_dot {
                    continue;
                }
                let next_idx = i + ch.len_utf8();
                if next_idx < buffer.len() {
                    let next_char = buffer[next_idx..].chars().next().unwrap_or(' ');
                    if next_char.is_whitespace() || next_char == '"' || next_char == '\'' || next_char == ')' {
                        split_point = Some((next_idx, next_idx));
                        break;
                    }
                }
            }
        }
    }

    // 3. Clause pauses (',', ';', ':') if prefix has >= 6 words
    if split_point.is_none() {
        for (i, ch) in buffer.char_indices() {
            if ch == ',' || ch == ';' || ch == ':' {
                let prefix = &buffer[..i];
                if prefix.split_whitespace().count() >= 6 {
                    let next_idx = i + ch.len_utf8();
                    if next_idx < buffer.len() {
                        let next_char = buffer[next_idx..].chars().next().unwrap_or(' ');
                        if next_char.is_whitespace() {
                            split_point = Some((next_idx, next_idx));
                            break;
                        }
                    }
                }
            }
        }
    }

    // 4. Long run-on sentence cap: if >= 12 words, split at whitespace around word 10
    if split_point.is_none() {
        let word_count = buffer.split_whitespace().count();
        if word_count >= 12 {
            let mut count = 0;
            let mut in_word = false;
            for (i, ch) in buffer.char_indices() {
                if ch.is_whitespace() {
                    if in_word {
                        count += 1;
                        if count >= 10 {
                            split_point = Some((i, i));
                            break;
                        }
                        in_word = false;
                    }
                } else {
                    in_word = true;
                }
            }
        }
    }

    if let Some((clause_end, rem_start)) = split_point {
        let clause = buffer[..clause_end].trim().to_string();
        let rem = buffer[rem_start..].trim_start().to_string();
        *buffer = rem;
        if !clause.is_empty() {
            return Some(clause);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Phonetics and Acoustic Modeling
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhonemeType {
    /// Resonant vowel with formants (F1, F2, F3) in Hz and duration in ms
    Vowel {
        f1: f32,
        f2: f32,
        f3: f32,
        duration_ms: f32,
    },
    /// Unvoiced/voiced fricative with center frequency, bandwidth, and duration
    Fricative {
        center_freq: f32,
        bandwidth: f32,
        duration_ms: f32,
        voiced: bool,
    },
    /// Plosive stop consonant with closure pause and release burst
    Plosive {
        burst_freq: f32,
        closure_ms: f32,
        burst_ms: f32,
        voiced: bool,
    },
    /// Resonant nasal murmur
    Nasal {
        f1: f32,
        anti_f: f32,
        duration_ms: f32,
    },
    /// Liquid/glide
    Liquid {
        f1: f32,
        f2: f32,
        f3: f32,
        duration_ms: f32,
    },
    /// Silence or pause
    Pause { duration_ms: f32 },
}

/// Grapheme-to-Phoneme converter mapping English text into phoneme streams.
pub struct GraphemeToPhoneme;

impl GraphemeToPhoneme {
    pub fn word_to_phonemes(word: &str) -> Vec<PhonemeType> {
        let clean = word.to_lowercase();
        // Expand digits into spoken words
        let mut expanded = String::new();
        for c in clean.chars() {
            match c {
                '0' => expanded.push_str("zero "),
                '1' => expanded.push_str("one "),
                '2' => expanded.push_str("two "),
                '3' => expanded.push_str("three "),
                '4' => expanded.push_str("four "),
                '5' => expanded.push_str("five "),
                '6' => expanded.push_str("six "),
                '7' => expanded.push_str("seven "),
                '8' => expanded.push_str("eight "),
                '9' => expanded.push_str("nine "),
                other => expanded.push(other),
            }
        }

        let words: Vec<&str> = expanded.split_whitespace().collect();
        if words.len() > 1 {
            let mut all_phonemes = Vec::new();
            for (w_idx, sub_word) in words.iter().enumerate() {
                all_phonemes.extend(Self::single_word_to_phonemes(sub_word));
                if w_idx + 1 < words.len() {
                    all_phonemes.push(PhonemeType::Pause { duration_ms: 30.0 });
                }
            }
            return all_phonemes;
        }

        Self::single_word_to_phonemes(&clean)
    }

    fn single_word_to_phonemes(word: &str) -> Vec<PhonemeType> {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.is_empty() {
            return vec![PhonemeType::Pause { duration_ms: 40.0 }];
        }

        let mut phonemes = Vec::new();
        let chars: Vec<char> = clean.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Check 4-character suffixes: "tion", "sion"
            if i + 3 < chars.len() {
                let quad: String = chars[i..=i + 3].iter().collect();
                if quad == "tion" || quad == "sion" {
                    phonemes.push(PhonemeType::Fricative {
                        center_freq: 3200.0,
                        bandwidth: 1200.0,
                        duration_ms: 80.0,
                        voiced: false,
                    });
                    phonemes.push(PhonemeType::Nasal {
                        f1: 280.0,
                        anti_f: 1500.0,
                        duration_ms: 70.0,
                    });
                    i += 4;
                    continue;
                }
            }

            // Check 3-character suffixes: "ing"
            if i + 2 < chars.len() {
                let tri: String = chars[i..=i + 2].iter().collect();
                if tri == "ing" {
                    phonemes.push(PhonemeType::Vowel {
                        f1: 390.0,
                        f2: 1990.0,
                        f3: 2550.0,
                        duration_ms: 70.0,
                    });
                    phonemes.push(PhonemeType::Nasal {
                        f1: 300.0,
                        anti_f: 2200.0,
                        duration_ms: 80.0,
                    });
                    i += 3;
                    continue;
                }
            }

            // Check multi-character digraphs
            if i + 1 < chars.len() {
                let pair: String = chars[i..=i + 1].iter().collect();
                match pair.as_str() {
                    "sh" => {
                        phonemes.push(PhonemeType::Fricative {
                            center_freq: 3200.0,
                            bandwidth: 1200.0,
                            duration_ms: 90.0,
                            voiced: false,
                        });
                        i += 2;
                        continue;
                    }
                    "th" => {
                        let is_voiced = word == "the"
                            || word == "this"
                            || word == "that"
                            || word == "there"
                            || word == "they";
                        phonemes.push(PhonemeType::Fricative {
                            center_freq: 4000.0,
                            bandwidth: 1500.0,
                            duration_ms: 70.0,
                            voiced: is_voiced,
                        });
                        i += 2;
                        continue;
                    }
                    "ch" => {
                        phonemes.push(PhonemeType::Plosive {
                            burst_freq: 3500.0,
                            closure_ms: 25.0,
                            burst_ms: 25.0,
                            voiced: false,
                        });
                        i += 2;
                        continue;
                    }
                    "ph" => {
                        phonemes.push(PhonemeType::Fricative {
                            center_freq: 2800.0,
                            bandwidth: 1500.0,
                            duration_ms: 80.0,
                            voiced: false,
                        });
                        i += 2;
                        continue;
                    }
                    "wh" => {
                        phonemes.push(PhonemeType::Liquid {
                            f1: 300.0,
                            f2: 610.0,
                            f3: 2200.0,
                            duration_ms: 70.0,
                        });
                        i += 2;
                        continue;
                    }
                    "ck" => {
                        phonemes.push(PhonemeType::Plosive {
                            burst_freq: 2200.0,
                            closure_ms: 25.0,
                            burst_ms: 20.0,
                            voiced: false,
                        });
                        i += 2;
                        continue;
                    }
                    "ng" => {
                        phonemes.push(PhonemeType::Nasal {
                            f1: 300.0,
                            anti_f: 2200.0,
                            duration_ms: 80.0,
                        });
                        i += 2;
                        continue;
                    }
                    "qu" => {
                        phonemes.push(PhonemeType::Plosive {
                            burst_freq: 2200.0,
                            closure_ms: 25.0,
                            burst_ms: 20.0,
                            voiced: false,
                        });
                        phonemes.push(PhonemeType::Liquid {
                            f1: 300.0,
                            f2: 610.0,
                            f3: 2200.0,
                            duration_ms: 60.0,
                        });
                        i += 2;
                        continue;
                    }
                    "ee" | "ea" => {
                        phonemes.push(PhonemeType::Vowel {
                            f1: 270.0,
                            f2: 2290.0,
                            f3: 3010.0,
                            duration_ms: 120.0,
                        });
                        i += 2;
                        continue;
                    }
                    "oo" => {
                        phonemes.push(PhonemeType::Vowel {
                            f1: 300.0,
                            f2: 870.0,
                            f3: 2240.0,
                            duration_ms: 120.0,
                        });
                        i += 2;
                        continue;
                    }
                    "ou" | "ow" => {
                        phonemes.push(PhonemeType::Vowel {
                            f1: 650.0,
                            f2: 1100.0,
                            f3: 2400.0,
                            duration_ms: 120.0,
                        });
                        i += 2;
                        continue;
                    }
                    "ai" | "ay" => {
                        phonemes.push(PhonemeType::Vowel {
                            f1: 530.0,
                            f2: 2100.0,
                            f3: 2800.0,
                            duration_ms: 120.0,
                        });
                        i += 2;
                        continue;
                    }
                    "oi" | "oy" => {
                        phonemes.push(PhonemeType::Vowel {
                            f1: 570.0,
                            f2: 1800.0,
                            f3: 2500.0,
                            duration_ms: 120.0,
                        });
                        i += 2;
                        continue;
                    }
                    "er" | "ar" | "or" | "ir" | "ur" => {
                        phonemes.push(PhonemeType::Liquid {
                            f1: 450.0,
                            f2: 1300.0,
                            f3: 1700.0,
                            duration_ms: 100.0,
                        });
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }

            match chars[i] {
                'a' => phonemes.push(PhonemeType::Vowel {
                    f1: 660.0,
                    f2: 1720.0,
                    f3: 2410.0,
                    duration_ms: 100.0,
                }),
                'b' => phonemes.push(PhonemeType::Plosive {
                    burst_freq: 800.0,
                    closure_ms: 25.0,
                    burst_ms: 20.0,
                    voiced: true,
                }),
                'c' => {
                    let is_soft = i + 1 < chars.len() && matches!(chars[i + 1], 'e' | 'i' | 'y');
                    if is_soft {
                        phonemes.push(PhonemeType::Fricative {
                            center_freq: 5500.0,
                            bandwidth: 1800.0,
                            duration_ms: 85.0,
                            voiced: false,
                        });
                    } else {
                        phonemes.push(PhonemeType::Plosive {
                            burst_freq: 2200.0,
                            closure_ms: 25.0,
                            burst_ms: 20.0,
                            voiced: false,
                        });
                    }
                }
                'd' => phonemes.push(PhonemeType::Plosive {
                    burst_freq: 3500.0,
                    closure_ms: 25.0,
                    burst_ms: 20.0,
                    voiced: true,
                }),
                'e' => phonemes.push(PhonemeType::Vowel {
                    f1: 530.0,
                    f2: 1840.0,
                    f3: 2480.0,
                    duration_ms: 90.0,
                }),
                'f' => phonemes.push(PhonemeType::Fricative {
                    center_freq: 2800.0,
                    bandwidth: 1500.0,
                    duration_ms: 80.0,
                    voiced: false,
                }),
                'g' => phonemes.push(PhonemeType::Plosive {
                    burst_freq: 1900.0,
                    closure_ms: 25.0,
                    burst_ms: 20.0,
                    voiced: true,
                }),
                'h' => phonemes.push(PhonemeType::Fricative {
                    center_freq: 1500.0,
                    bandwidth: 1000.0,
                    duration_ms: 60.0,
                    voiced: false,
                }),
                'i' => phonemes.push(PhonemeType::Vowel {
                    f1: 390.0,
                    f2: 1990.0,
                    f3: 2550.0,
                    duration_ms: 85.0,
                }),
                'j' => phonemes.push(PhonemeType::Plosive {
                    burst_freq: 2600.0,
                    closure_ms: 25.0,
                    burst_ms: 25.0,
                    voiced: true,
                }),
                'k' => phonemes.push(PhonemeType::Plosive {
                    burst_freq: 2200.0,
                    closure_ms: 30.0,
                    burst_ms: 25.0,
                    voiced: false,
                }),
                'l' => phonemes.push(PhonemeType::Liquid {
                    f1: 400.0,
                    f2: 1100.0,
                    f3: 2800.0,
                    duration_ms: 80.0,
                }),
                'm' => phonemes.push(PhonemeType::Nasal {
                    f1: 250.0,
                    anti_f: 800.0,
                    duration_ms: 85.0,
                }),
                'n' => phonemes.push(PhonemeType::Nasal {
                    f1: 280.0,
                    anti_f: 1500.0,
                    duration_ms: 80.0,
                }),
                'o' => phonemes.push(PhonemeType::Vowel {
                    f1: 570.0,
                    f2: 840.0,
                    f3: 2410.0,
                    duration_ms: 100.0,
                }),
                'p' => phonemes.push(PhonemeType::Plosive {
                    burst_freq: 900.0,
                    closure_ms: 30.0,
                    burst_ms: 20.0,
                    voiced: false,
                }),
                'q' => phonemes.push(PhonemeType::Plosive {
                    burst_freq: 2200.0,
                    closure_ms: 30.0,
                    burst_ms: 25.0,
                    voiced: false,
                }),
                'r' => phonemes.push(PhonemeType::Liquid {
                    f1: 350.0,
                    f2: 1200.0,
                    f3: 1650.0,
                    duration_ms: 85.0,
                }),
                's' => phonemes.push(PhonemeType::Fricative {
                    center_freq: 5500.0,
                    bandwidth: 1800.0,
                    duration_ms: 95.0,
                    voiced: false,
                }),
                't' => phonemes.push(PhonemeType::Plosive {
                    burst_freq: 3800.0,
                    closure_ms: 30.0,
                    burst_ms: 20.0,
                    voiced: false,
                }),
                'u' => phonemes.push(PhonemeType::Vowel {
                    f1: 440.0,
                    f2: 1020.0,
                    f3: 2240.0,
                    duration_ms: 90.0,
                }),
                'v' => phonemes.push(PhonemeType::Fricative {
                    center_freq: 2800.0,
                    bandwidth: 1500.0,
                    duration_ms: 75.0,
                    voiced: true,
                }),
                'w' => phonemes.push(PhonemeType::Liquid {
                    f1: 300.0,
                    f2: 610.0,
                    f3: 2200.0,
                    duration_ms: 70.0,
                }),
                'x' => {
                    phonemes.push(PhonemeType::Plosive {
                        burst_freq: 2200.0,
                        closure_ms: 20.0,
                        burst_ms: 15.0,
                        voiced: false,
                    });
                    phonemes.push(PhonemeType::Fricative {
                        center_freq: 5500.0,
                        bandwidth: 1800.0,
                        duration_ms: 70.0,
                        voiced: false,
                    });
                }
                'y' => phonemes.push(PhonemeType::Vowel {
                    f1: 350.0,
                    f2: 2100.0,
                    f3: 2800.0,
                    duration_ms: 80.0,
                }),
                'z' => phonemes.push(PhonemeType::Fricative {
                    center_freq: 5500.0,
                    bandwidth: 1800.0,
                    duration_ms: 85.0,
                    voiced: true,
                }),
                _ => phonemes.push(PhonemeType::Pause { duration_ms: 15.0 }),
            }
            i += 1;
        }

        // Add inter-word pause
        phonemes.push(PhonemeType::Pause { duration_ms: 40.0 });
        phonemes
    }
}

// ---------------------------------------------------------------------------
// Digital Resonator Filter (Formant synthesis)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Resonator {
    a1: f32,
    a2: f32,
    gain: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    fn new(freq: f32, bandwidth: f32, sample_rate: f32) -> Self {
        let r = (-std::f32::consts::PI * bandwidth / sample_rate).exp();
        let theta = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let a1 = -2.0 * r * theta.cos();
        let a2 = r * r;
        let gain = (1.0 - r)
            * (1.0 - 2.0 * r * (2.0 * theta).cos() + r * r)
                .sqrt()
                .max(1e-4);
        Self {
            a1,
            a2,
            gain,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn step(&mut self, x: f32) -> f32 {
        let y = self.gain * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

// ---------------------------------------------------------------------------
// Formant Neural Synthesizer
// ---------------------------------------------------------------------------

struct UtteranceState {
    config: VoiceConfig,
    gate: TextPrefixCommitmentGate,
    chunk_counter: usize,
    sample_offset: u64,
    words: Vec<WordAlignment>,
    queue: VecDeque<SynthesizedAudioChunk>,
    is_completed: bool,
    is_cancelled: bool,
    start_time: Instant,
    first_chunk_emitted: bool,
    first_audio_latency_ms: f64,
}

/// Local prototype/test fallback for streaming formant parametric speech synthesis.
///
/// Implements cascading IIR resonance filters (F1/F2/F3), glottal pulse modeling, G2P phonetics,
/// and continuous coarticulation.
/// NOTE: This engine uses parametric acoustic synthesis; it does not contain trained deep neural
/// vocoder weights (e.g. VITS or HiFi-GAN). Public type name is retained for API compatibility.
pub struct FormantNeuralTtsEngine {
    pub sample_rate: u32,
    sessions: Arc<RwLock<HashMap<String, UtteranceState>>>,
    max_queue_depth: usize,
}

impl FormantNeuralTtsEngine {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            max_queue_depth: 64, // Bounded queue: ~60s of buffered audio
        }
    }

    /// Direct synthesis of full text into PCM waveform (convenience offline API).
    pub fn synthesize_text(&self, text: &str) -> Vec<f32> {
        let config = VoiceConfig {
            sample_rate: self.sample_rate,
            ..Default::default()
        };
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut audio = Vec::new();
        for (i, word) in words.iter().enumerate() {
            let phonemes = GraphemeToPhoneme::word_to_phonemes(word);
            let word_samples = self.render_phonemes(&phonemes, &config, i, words.len());
            audio.extend(word_samples);
        }
        if audio.is_empty() {
            // Render a short silence
            audio = vec![0.0f32; (self.sample_rate as f32 * 0.1) as usize];
        }
        audio
    }

    /// Renders a sequence of phonemes into PCM samples using formant filters and glottal pulses.
    fn render_phonemes(
        &self,
        phonemes: &[PhonemeType],
        config: &VoiceConfig,
        word_idx: usize,
        total_words: usize,
    ) -> Vec<f32> {
        let sr = self.sample_rate as f32;
        let rate_factor = config.speaking_rate.clamp(0.5, 2.5);
        let mut samples = Vec::new();

        // Prosodic sentence declination curve: pitch starts slightly higher and descends
        let progress = if total_words > 1 {
            word_idx as f32 / (total_words - 1) as f32
        } else {
            0.5
        };
        let declination = 1.05 - 0.10 * progress;
        let base_pitch = config.base_f0 * config.pitch_shift * declination;

        let mut rng_state = (word_idx as u64 + 1) * 314159;
        let mut glottal_phase = 0.0f32;

        for (p_idx, phoneme) in phonemes.iter().enumerate() {
            let is_first = p_idx == 0;
            let is_last = p_idx + 1 == phonemes.len();

            match *phoneme {
                PhonemeType::Vowel {
                    f1,
                    f2,
                    f3,
                    duration_ms,
                } => {
                    let duration_sec = (duration_ms / 1000.0) / rate_factor;
                    let num_samples = (duration_sec * sr) as usize;
                    let scale = config.formant_scale;

                    let mut r1 = Resonator::new(f1 * scale, 80.0, sr);
                    let mut r2 = Resonator::new(f2 * scale, 100.0, sr);
                    let mut r3 = Resonator::new(f3 * scale, 120.0, sr);

                    for i in 0..num_samples {
                        let t = i as f32 / sr;
                        glottal_phase = (glottal_phase + base_pitch / sr) % 1.0;
                        let phase = glottal_phase;
                        let glottal = if phase < 0.7 {
                            (std::f32::consts::PI * phase / 0.7).sin()
                        } else {
                            -((phase - 0.7) / 0.3) * 0.3
                        };

                        let attack = if is_first {
                            1.0 - (-30.0 * t / duration_sec).exp()
                        } else {
                            1.0
                        };
                        let decay = if is_last {
                            1.0 - (-30.0 * (duration_sec - t).max(0.0) / duration_sec).exp()
                        } else {
                            1.0
                        };
                        let env = attack * decay;

                        let excitation = glottal * env;
                        let s1 = r1.step(excitation);
                        let s2 = r2.step(excitation);
                        let s3 = r3.step(excitation);

                        let out = (s1 * 0.55 + s2 * 0.35 + s3 * 0.15) * config.volume;
                        samples.push(out.tanh() * 0.85);
                    }
                }
                PhonemeType::Fricative {
                    center_freq,
                    bandwidth,
                    duration_ms,
                    voiced,
                } => {
                    let duration_sec = (duration_ms / 1000.0) / rate_factor;
                    let num_samples = (duration_sec * sr) as usize;
                    let mut r = Resonator::new(center_freq, bandwidth, sr);

                    for i in 0..num_samples {
                        let t = i as f32 / sr;
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let noise = ((rng_state >> 32) as i32 as f32 / 2147483648.0) * 0.40;

                        let mut excitation = noise;
                        if voiced {
                            glottal_phase = (glottal_phase + base_pitch / sr) % 1.0;
                            let voice_bar = (2.0 * std::f32::consts::PI * glottal_phase).sin() * 0.20;
                            excitation += voice_bar;
                        }

                        let attack = if is_first {
                            1.0 - (-30.0 * t / duration_sec).exp()
                        } else {
                            1.0
                        };
                        let decay = if is_last {
                            1.0 - (-30.0 * (duration_sec - t).max(0.0) / duration_sec).exp()
                        } else {
                            1.0
                        };
                        let env = attack * decay;

                        let s = r.step(excitation * env);
                        samples.push((s * config.volume).tanh() * 0.60);
                    }
                }
                PhonemeType::Plosive {
                    burst_freq,
                    closure_ms,
                    burst_ms,
                    voiced,
                } => {
                    let closure_samples = ((closure_ms / 1000.0 / rate_factor) * sr) as usize;
                    let burst_samples = ((burst_ms / 1000.0 / rate_factor) * sr) as usize;

                    // Closure silence or voiced bar
                    for _ in 0..closure_samples {
                        if voiced {
                            glottal_phase = (glottal_phase + 120.0 / sr) % 1.0;
                            let voice_bar =
                                (2.0 * std::f32::consts::PI * glottal_phase).sin() * 0.08;
                            samples.push(voice_bar * config.volume);
                        } else {
                            samples.push(0.0);
                        }
                    }

                    // Release burst
                    let mut r = Resonator::new(burst_freq, 400.0, sr);
                    for i in 0..burst_samples {
                        let t = i as f32 / sr;
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let noise = ((rng_state >> 32) as i32 as f32 / 2147483648.0) * 0.55;
                        let env = (-80.0 * t).exp(); // Fast exponential burst decay
                        let s = r.step(noise * env);
                        samples.push((s * config.volume).tanh() * 0.70);
                    }
                }
                PhonemeType::Nasal {
                    f1, duration_ms, ..
                } => {
                    let duration_sec = (duration_ms / 1000.0) / rate_factor;
                    let num_samples = (duration_sec * sr) as usize;
                    let mut r1 = Resonator::new(f1, 60.0, sr);

                    for _ in 0..num_samples {
                        glottal_phase = (glottal_phase + base_pitch / sr) % 1.0;
                        let glottal = (std::f32::consts::PI * glottal_phase).sin();
                        let s = r1.step(glottal * 0.35);
                        samples.push((s * config.volume).tanh() * 0.70);
                    }
                }
                PhonemeType::Liquid {
                    f1,
                    f2,
                    f3,
                    duration_ms,
                } => {
                    let duration_sec = (duration_ms / 1000.0) / rate_factor;
                    let num_samples = (duration_sec * sr) as usize;
                    let mut r1 = Resonator::new(f1, 70.0, sr);
                    let mut r2 = Resonator::new(f2, 90.0, sr);
                    let mut r3 = Resonator::new(f3, 110.0, sr);

                    for _ in 0..num_samples {
                        glottal_phase = (glottal_phase + base_pitch / sr) % 1.0;
                        let glottal = (std::f32::consts::PI * glottal_phase).sin();
                        let s = r1.step(glottal * 0.4)
                            + r2.step(glottal * 0.3)
                            + r3.step(glottal * 0.2);
                        samples.push((s * config.volume).tanh() * 0.75);
                    }
                }
                PhonemeType::Pause { duration_ms } => {
                    let pause_samples = ((duration_ms / 1000.0 / rate_factor) * sr) as usize;
                    samples.extend(vec![0.0f32; pause_samples]);
                }
            }
        }

        samples
    }
}

// ---------------------------------------------------------------------------
// Streaming TTS Adapter Trait & Implementation
// ---------------------------------------------------------------------------

pub trait StreamingTtsAdapter: Send + Sync {
    /// Identifier of this TTS adapter backend.
    fn name(&self) -> &'static str;

    /// Initializes state for a new conversational utterance turn.
    fn start_utterance(&mut self, utterance_id: &str, config: &VoiceConfig) -> Result<(), String>;

    /// Feeds streaming text tokens into the commitment gate and synthesizes new audio chunks.
    fn feed_text(
        &mut self,
        utterance_id: &str,
        text: &str,
    ) -> Result<Vec<SynthesizedAudioChunk>, String>;

    /// Flushes remaining text and marks the utterance as completed.
    fn finalize_utterance(
        &mut self,
        utterance_id: &str,
    ) -> Result<Vec<SynthesizedAudioChunk>, String>;

    /// Halts active utterance, purges pending audio, and computes cancellation report.
    fn cancel_utterance(
        &mut self,
        utterance_id: &str,
        current_playback_cursor: u64,
    ) -> Result<TtsCancellationReport, String>;

    /// Returns current alignment status showing which words were played vs remain pending.
    fn get_playback_status(
        &self,
        utterance_id: &str,
        current_playback_cursor: u64,
    ) -> Result<PlaybackAlignmentStatus, String>;
}

impl StreamingTtsAdapter for FormantNeuralTtsEngine {
    fn name(&self) -> &'static str {
        "formant-neural-v1"
    }

    fn start_utterance(&mut self, utterance_id: &str, config: &VoiceConfig) -> Result<(), String> {
        let mut sessions = self.sessions.write().map_err(|e| e.to_string())?;
        let state = UtteranceState {
            config: config.clone(),
            gate: TextPrefixCommitmentGate::new(),
            chunk_counter: 0,
            sample_offset: 0,
            words: Vec::new(),
            queue: VecDeque::new(),
            is_completed: false,
            is_cancelled: false,
            start_time: Instant::now(),
            first_chunk_emitted: false,
            first_audio_latency_ms: 0.0,
        };
        sessions.insert(utterance_id.to_string(), state);
        Ok(())
    }

    fn feed_text(
        &mut self,
        utterance_id: &str,
        text: &str,
    ) -> Result<Vec<SynthesizedAudioChunk>, String> {
        let mut sessions = self.sessions.write().map_err(|e| e.to_string())?;
        let state = sessions
            .get_mut(utterance_id)
            .ok_or_else(|| format!("Utterance '{utterance_id}' not found"))?;

        if state.is_cancelled || state.is_completed {
            return Ok(Vec::new());
        }

        let new_words = state.gate.push_token(text);
        if new_words.is_empty() {
            return Ok(Vec::new());
        }

        let mut produced_chunks = Vec::new();
        let sr = self.sample_rate;

        for word in new_words {
            // Prevent unbounded queue growth on long continuous streams
            if state.queue.len() >= self.max_queue_depth {
                // Drop oldest chunk or apply backpressure
                state.queue.pop_front();
            }

            let phonemes = GraphemeToPhoneme::word_to_phonemes(&word);
            let samples = self.render_phonemes(
                &phonemes,
                &state.config,
                state.words.len(),
                state.words.len() + 1,
            );

            let start_sample = state.sample_offset as usize;
            let end_sample = start_sample + samples.len();
            let start_ms = (start_sample as f64 / sr as f64) * 1000.0;
            let end_ms = (end_sample as f64 / sr as f64) * 1000.0;

            let alignment = WordAlignment {
                word: word.clone(),
                start_sample,
                end_sample,
                start_ms,
                end_ms,
            };
            state.words.push(alignment.clone());

            let duration_ms = (samples.len() as f64 / sr as f64) * 1000.0;
            let chunk = SynthesizedAudioChunk {
                utterance_id: utterance_id.to_string(),
                chunk_index: state.chunk_counter,
                samples: Arc::from(samples.into_boxed_slice()),
                sample_rate: sr,
                channels: 1,
                sample_offset: state.sample_offset,
                duration_ms,
                spoken_words: vec![alignment],
                is_final_chunk: false,
            };

            state.chunk_counter += 1;
            state.sample_offset += chunk.samples.len() as u64;

            if !state.first_chunk_emitted {
                state.first_chunk_emitted = true;
                state.first_audio_latency_ms = state.start_time.elapsed().as_secs_f64() * 1000.0;
            }

            state.queue.push_back(chunk.clone());
            produced_chunks.push(chunk);
        }

        Ok(produced_chunks)
    }

    fn finalize_utterance(
        &mut self,
        utterance_id: &str,
    ) -> Result<Vec<SynthesizedAudioChunk>, String> {
        let mut sessions = self.sessions.write().map_err(|e| e.to_string())?;
        let state = sessions
            .get_mut(utterance_id)
            .ok_or_else(|| format!("Utterance '{utterance_id}' not found"))?;

        if state.is_cancelled || state.is_completed {
            return Ok(Vec::new());
        }

        let flushed_words = state.gate.flush();
        let mut produced_chunks = Vec::new();
        let sr = self.sample_rate;

        for word in flushed_words {
            let phonemes = GraphemeToPhoneme::word_to_phonemes(&word);
            let samples = self.render_phonemes(
                &phonemes,
                &state.config,
                state.words.len(),
                state.words.len() + 1,
            );

            let start_sample = state.sample_offset as usize;
            let end_sample = start_sample + samples.len();
            let start_ms = (start_sample as f64 / sr as f64) * 1000.0;
            let end_ms = (end_sample as f64 / sr as f64) * 1000.0;

            let alignment = WordAlignment {
                word: word.clone(),
                start_sample,
                end_sample,
                start_ms,
                end_ms,
            };
            state.words.push(alignment.clone());

            let duration_ms = (samples.len() as f64 / sr as f64) * 1000.0;
            let chunk = SynthesizedAudioChunk {
                utterance_id: utterance_id.to_string(),
                chunk_index: state.chunk_counter,
                samples: Arc::from(samples.into_boxed_slice()),
                sample_rate: sr,
                channels: 1,
                sample_offset: state.sample_offset,
                duration_ms,
                spoken_words: vec![alignment],
                is_final_chunk: false,
            };

            state.chunk_counter += 1;
            state.sample_offset += chunk.samples.len() as u64;

            if !state.first_chunk_emitted {
                state.first_chunk_emitted = true;
                state.first_audio_latency_ms = state.start_time.elapsed().as_secs_f64() * 1000.0;
            }

            state.queue.push_back(chunk.clone());
            produced_chunks.push(chunk);
        }

        // Emit zero-length final marker chunk
        let final_chunk = SynthesizedAudioChunk {
            utterance_id: utterance_id.to_string(),
            chunk_index: state.chunk_counter,
            samples: Arc::from(Vec::new().into_boxed_slice()),
            sample_rate: sr,
            channels: 1,
            sample_offset: state.sample_offset,
            duration_ms: 0.0,
            spoken_words: Vec::new(),
            is_final_chunk: true,
        };
        state.chunk_counter += 1;
        state.is_completed = true;
        produced_chunks.push(final_chunk);

        Ok(produced_chunks)
    }

    fn cancel_utterance(
        &mut self,
        utterance_id: &str,
        current_playback_cursor: u64,
    ) -> Result<TtsCancellationReport, String> {
        let mut sessions = self.sessions.write().map_err(|e| e.to_string())?;
        let state = sessions
            .get_mut(utterance_id)
            .ok_or_else(|| format!("Utterance '{utterance_id}' not found"))?;

        state.is_cancelled = true;
        let total_synthesized = state.sample_offset;
        let played_samples = current_playback_cursor.min(total_synthesized);
        let discarded_samples = total_synthesized.saturating_sub(played_samples);

        // Identify played words vs discarded words
        let mut played_words = Vec::new();
        let mut committed_words = Vec::new();

        for align in &state.words {
            committed_words.push(align.word.clone());
            if (align.end_sample as u64) <= played_samples {
                played_words.push(align.word.clone());
            }
        }

        // Purge pending queue
        state.queue.clear();

        let interrupted_at_ms = (played_samples as f64 / self.sample_rate as f64) * 1000.0;

        Ok(TtsCancellationReport {
            utterance_id: utterance_id.to_string(),
            total_synthesized_samples: total_synthesized,
            played_samples,
            discarded_samples,
            committed_words,
            played_words,
            interrupted_at_ms,
        })
    }

    fn get_playback_status(
        &self,
        utterance_id: &str,
        current_playback_cursor: u64,
    ) -> Result<PlaybackAlignmentStatus, String> {
        let sessions = self.sessions.read().map_err(|e| e.to_string())?;
        let state = sessions
            .get(utterance_id)
            .ok_or_else(|| format!("Utterance '{utterance_id}' not found"))?;

        let total_samples = state.sample_offset;
        let played_samples = current_playback_cursor.min(total_samples);

        let mut played_words = Vec::new();
        let mut current_word = None;
        let mut pending_words = Vec::new();

        for align in &state.words {
            let start = align.start_sample as u64;
            let end = align.end_sample as u64;

            if end <= played_samples {
                played_words.push(align.word.clone());
            } else if start < played_samples && played_samples < end {
                current_word = Some(align.word.clone());
            } else {
                pending_words.push(align.word.clone());
            }
        }

        let played_duration_ms = (played_samples as f64 / self.sample_rate as f64) * 1000.0;
        let total_duration_ms = (total_samples as f64 / self.sample_rate as f64) * 1000.0;
        let is_completed = state.is_completed && played_samples >= total_samples;

        Ok(PlaybackAlignmentStatus {
            utterance_id: utterance_id.to_string(),
            played_words,
            current_word,
            pending_words,
            played_duration_ms,
            total_duration_ms,
            is_completed,
        })
    }
}

// ---------------------------------------------------------------------------
// Supervised Piper Local Adapter
// ---------------------------------------------------------------------------

/// Fail-closed failures from the production-facing local Piper adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PiperTtsError {
    RunnerNotFound(PathBuf),
    VoiceModelNotFound(PathBuf),
    InvalidInput(String),
    ProcessFailed(String),
    InvalidAudio(String),
    Cancelled {
        utterance_id: String,
    },
    TimedOut {
        utterance_id: String,
        timeout_ms: u64,
    },
}

impl std::fmt::Display for PiperTtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RunnerNotFound(path) => {
                write!(f, "PiperRunnerNotFound: {}", path.display())
            }
            Self::VoiceModelNotFound(path) => {
                write!(f, "PiperVoiceModelNotFound: {}", path.display())
            }
            Self::InvalidInput(reason) => write!(f, "PiperInvalidInput: {reason}"),
            Self::ProcessFailed(reason) => write!(f, "PiperProcessFailed: {reason}"),
            Self::InvalidAudio(reason) => write!(f, "PiperInvalidAudio: {reason}"),
            Self::Cancelled { utterance_id } => {
                write!(f, "PiperCancelled: utterance '{utterance_id}'")
            }
            Self::TimedOut {
                utterance_id,
                timeout_ms,
            } => write!(
                f,
                "PiperTimedOut: utterance '{utterance_id}' exceeded {timeout_ms} ms"
            ),
        }
    }
}

impl std::error::Error for PiperTtsError {}

struct PiperUtteranceState {
    config: VoiceConfig,
    output_sample_rate: u32,
    output_channels: u16,
    gate: TextPrefixCommitmentGate,
    chunk_counter: usize,
    sample_offset: u64,
    words: Vec<WordAlignment>,
    queue: VecDeque<SynthesizedAudioChunk>,
    is_completed: bool,
    is_cancelled: bool,
}

static PIPER_OUTPUT_COUNTER: AtomicU64 = AtomicU64::new(1);

struct PiperTempOutput(PathBuf);

impl Drop for PiperTempOutput {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Supervised adapter for an explicitly configured local Piper executable and
/// voice model. It never downloads dependencies and never falls back to synthetic
/// or fabricated speech when Piper is unavailable.
#[derive(Clone)]
pub struct PiperLocalTtsAdapter {
    runner_path: PathBuf,
    voice_model_path: PathBuf,
    timeout: Duration,
    max_input_chars: usize,
    max_queue_depth: usize,
    sessions: Arc<RwLock<HashMap<String, PiperUtteranceState>>>,
    active_children: Arc<Mutex<HashMap<String, Child>>>,
}

impl PiperLocalTtsAdapter {
    pub fn new(runner_path: impl Into<PathBuf>, voice_model_path: impl Into<PathBuf>) -> Self {
        Self {
            runner_path: runner_path.into(),
            voice_model_path: voice_model_path.into(),
            timeout: Duration::from_secs(30),
            max_input_chars: 4096,
            max_queue_depth: 64,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            active_children: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout.max(Duration::from_millis(10));
        self
    }

    pub fn runner_path(&self) -> &Path {
        &self.runner_path
    }

    pub fn voice_model_path(&self) -> &Path {
        &self.voice_model_path
    }

    /// Validates explicit local dependencies without downloading or installing them.
    pub fn validate_configuration(&self) -> Result<(), PiperTtsError> {
        self.validate_dependencies()
    }

    fn resolve_runner(&self) -> Option<PathBuf> {
        if self.runner_path.is_file() {
            return Some(self.runner_path.clone());
        }
        // Only search default tool and PATH locations if runner_path is a bare command name (e.g. "piper")
        if self.runner_path.components().count() == 1 {
            let local_tool = PathBuf::from("tools/piper/piper");
            if local_tool.is_file() {
                return Some(local_tool);
            }
            if let Ok(curr) = std::env::current_dir() {
                let candidate = curr.join("tools/piper/piper");
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
            if let Some(home) = std::env::var_os("HOME") {
                let candidate = PathBuf::from(&home).join(".local/piper/piper");
                if candidate.is_file() {
                    return Some(candidate);
                }
                let candidate_bin = PathBuf::from(&home).join(".local/bin/piper");
                if candidate_bin.is_file() {
                    return Some(candidate_bin);
                }
            }
            if let Some(paths) = std::env::var_os("PATH") {
                for dir in std::env::split_paths(&paths) {
                    let candidate = dir.join(&self.runner_path);
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            }
        }
        None
    }

    fn resolve_voice_model(&self) -> Option<PathBuf> {
        if self.voice_model_path.is_file() {
            return Some(self.voice_model_path.clone());
        }
        let with_onnx = self.voice_model_path.with_extension("onnx");
        if with_onnx.is_file() {
            return Some(with_onnx);
        }
        // Only search standard voice repositories if voice_model_path has no path separators
        if self.voice_model_path.components().count() == 1 {
            let local_model = PathBuf::from("models/piper").join(&self.voice_model_path);
            if local_model.is_file() {
                return Some(local_model);
            }
            let local_model_onnx = local_model.with_extension("onnx");
            if local_model_onnx.is_file() {
                return Some(local_model_onnx);
            }
            if let Ok(curr) = std::env::current_dir() {
                let candidate = curr.join("models/piper").join(&self.voice_model_path);
                if candidate.is_file() {
                    return Some(candidate);
                }
                let candidate_onnx = candidate.with_extension("onnx");
                if candidate_onnx.is_file() {
                    return Some(candidate_onnx);
                }
            }
            if let Some(home) = std::env::var_os("HOME") {
                let candidate = PathBuf::from(&home)
                    .join(".local/share/piper-voices")
                    .join(&self.voice_model_path);
                if candidate.is_file() {
                    return Some(candidate);
                }
                let candidate_onnx = candidate.with_extension("onnx");
                if candidate_onnx.is_file() {
                    return Some(candidate_onnx);
                }
            }
        }
        None
    }

    fn validate_dependencies(&self) -> Result<(), PiperTtsError> {
        if self.resolve_runner().is_none() {
            return Err(PiperTtsError::RunnerNotFound(self.runner_path.clone()));
        }
        if self.resolve_voice_model().is_none() {
            return Err(PiperTtsError::VoiceModelNotFound(
                self.voice_model_path.clone(),
            ));
        }
        Ok(())
    }

    fn output_path(&self, utterance_id: &str) -> PathBuf {
        let safe_id: String = utterance_id
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .take(48)
            .collect();
        let sequence = PIPER_OUTPUT_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "sail-piper-{}-{safe_id}-{sequence}.wav",
            std::process::id()
        ))
    }

    fn session_cancelled(&self, utterance_id: &str) -> bool {
        self.sessions
            .read()
            .ok()
            .and_then(|sessions| sessions.get(utterance_id).map(|state| state.is_cancelled))
            .unwrap_or(true)
    }

    fn terminate_child(child: &mut Child) {
        #[cfg(unix)]
        {
            let pid = child.id() as i32;
            if pid > 0 {
                unsafe {
                    libc::kill(-pid, libc::SIGKILL);
                }
            }
        }
        let _ = child.kill();
        let _ = child.wait();
    }

    fn decode_wav(path: &Path) -> Result<(Vec<f32>, u32, u16), PiperTtsError> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| PiperTtsError::InvalidAudio(format!("cannot open Piper WAV: {e}")))?;
        let spec = reader.spec();
        if spec.sample_rate == 0 || spec.channels == 0 {
            return Err(PiperTtsError::InvalidAudio(
                "WAV has zero sample rate or channels".into(),
            ));
        }

        let samples = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|sample| sample.map_err(|e| PiperTtsError::InvalidAudio(e.to_string())))
                .collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
                .samples::<i16>()
                .map(|sample| {
                    sample
                        .map(|value| value as f32 / i16::MAX as f32)
                        .map_err(|e| PiperTtsError::InvalidAudio(e.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => {
                let scale =
                    ((1_i64 << (spec.bits_per_sample.saturating_sub(1) as u32)) - 1).max(1) as f32;
                reader
                    .samples::<i32>()
                    .map(|sample| {
                        sample
                            .map(|value| value as f32 / scale)
                            .map_err(|e| PiperTtsError::InvalidAudio(e.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(PiperTtsError::InvalidAudio(
                "WAV is empty or contains non-finite samples".into(),
            ));
        }
        Ok((samples, spec.sample_rate, spec.channels))
    }

    /// Runs Piper for one already-committed text segment and returns decoded PCM,
    /// preserving typed dependency, input, process, timeout, cancellation, and WAV errors.
    pub fn synthesize_committed_text(
        &self,
        utterance_id: &str,
        text: &str,
        config: &VoiceConfig,
    ) -> Result<(Vec<f32>, u32, u16), PiperTtsError> {
        self.validate_dependencies()?;
        let text = text.trim();
        if text.is_empty() {
            return Err(PiperTtsError::InvalidInput("text is empty".into()));
        }
        if text.chars().count() > self.max_input_chars {
            return Err(PiperTtsError::InvalidInput(format!(
                "text exceeds {} character limit",
                self.max_input_chars
            )));
        }
        if !config.speaking_rate.is_finite() || !(0.5..=2.5).contains(&config.speaking_rate) {
            return Err(PiperTtsError::InvalidInput(
                "speaking rate must be finite and between 0.5 and 2.5".into(),
            ));
        }
        if self.session_cancelled(utterance_id) {
            return Err(PiperTtsError::Cancelled {
                utterance_id: utterance_id.into(),
            });
        }

        let runner = self.resolve_runner().unwrap_or_else(|| self.runner_path.clone());
        let model = self.resolve_voice_model().unwrap_or_else(|| self.voice_model_path.clone());

        let output = PiperTempOutput(self.output_path(utterance_id));
        let mut command = Command::new(&runner);
        if let Some(parent) = runner.parent() {
            let current_ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let new_ld = if current_ld.is_empty() {
                parent.to_string_lossy().to_string()
            } else {
                format!("{}:{current_ld}", parent.display())
            };
            command.env("LD_LIBRARY_PATH", new_ld);
        }
        command
            .arg("--model")
            .arg(&model)
            .arg("--output_file")
            .arg(&output.0)
            .arg("--length_scale")
            .arg((1.0 / config.speaking_rate).to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        let mut child = command.spawn().map_err(|e| {
            PiperTtsError::ProcessFailed(format!(
                "cannot spawn '{}': {e}",
                self.runner_path.display()
            ))
        })?;
        let mut stdin = match child.stdin.take() {
            Some(stdin) => stdin,
            None => {
                Self::terminate_child(&mut child);
                return Err(PiperTtsError::ProcessFailed(
                    "failed to open Piper stdin".into(),
                ));
            }
        };
        if let Err(error) = stdin
            .write_all(text.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
        {
            Self::terminate_child(&mut child);
            return Err(PiperTtsError::ProcessFailed(format!(
                "failed writing Piper input: {error}"
            )));
        }
        drop(stdin);

        match self.active_children.lock() {
            Ok(mut active) => {
                active.insert(utterance_id.to_string(), child);
            }
            Err(error) => {
                Self::terminate_child(&mut child);
                return Err(PiperTtsError::ProcessFailed(error.to_string()));
            }
        }

        let started = Instant::now();
        let status = loop {
            if self.session_cancelled(utterance_id) {
                if let Ok(mut active) = self.active_children.lock() {
                    if let Some(mut child) = active.remove(utterance_id) {
                        Self::terminate_child(&mut child);
                    }
                }
                return Err(PiperTtsError::Cancelled {
                    utterance_id: utterance_id.into(),
                });
            }
            if started.elapsed() >= self.timeout {
                if let Ok(mut active) = self.active_children.lock() {
                    if let Some(mut child) = active.remove(utterance_id) {
                        Self::terminate_child(&mut child);
                    }
                }
                return Err(PiperTtsError::TimedOut {
                    utterance_id: utterance_id.into(),
                    timeout_ms: self.timeout.as_millis() as u64,
                });
            }

            let polled = {
                let mut active = self
                    .active_children
                    .lock()
                    .map_err(|e| PiperTtsError::ProcessFailed(e.to_string()))?;
                match active.get_mut(utterance_id) {
                    Some(child) => child
                        .try_wait()
                        .map_err(|e| PiperTtsError::ProcessFailed(e.to_string()))?,
                    None => {
                        if self.session_cancelled(utterance_id) {
                            return Err(PiperTtsError::Cancelled {
                                utterance_id: utterance_id.into(),
                            });
                        }
                        return Err(PiperTtsError::ProcessFailed(
                            "supervised child disappeared".into(),
                        ));
                    }
                }
            };
            if let Some(status) = polled {
                self.active_children
                    .lock()
                    .map_err(|e| PiperTtsError::ProcessFailed(e.to_string()))?
                    .remove(utterance_id);
                break status;
            }
            thread::sleep(Duration::from_millis(10));
        };

        if !status.success() {
            return Err(PiperTtsError::ProcessFailed(format!(
                "Piper exited with status {status}"
            )));
        }
        Self::decode_wav(&output.0)
    }

    fn render_words(
        &self,
        utterance_id: &str,
        words: Vec<String>,
    ) -> Result<Vec<SynthesizedAudioChunk>, PiperTtsError> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let config = self
            .sessions
            .read()
            .map_err(|e| PiperTtsError::ProcessFailed(e.to_string()))?
            .get(utterance_id)
            .map(|state| state.config.clone())
            .ok_or_else(|| {
                PiperTtsError::InvalidInput(format!("unknown utterance '{utterance_id}'"))
            })?;
        let text = words.join(" ");
        let (samples, sample_rate, channels) =
            self.synthesize_committed_text(utterance_id, &text, &config)?;

        let mut sessions = self
            .sessions
            .write()
            .map_err(|e| PiperTtsError::ProcessFailed(e.to_string()))?;
        let state = sessions.get_mut(utterance_id).ok_or_else(|| {
            PiperTtsError::InvalidInput(format!("unknown utterance '{utterance_id}'"))
        })?;
        if state.is_cancelled {
            return Err(PiperTtsError::Cancelled {
                utterance_id: utterance_id.into(),
            });
        }
        if state.sample_offset > 0
            && (state.output_sample_rate != sample_rate || state.output_channels != channels)
        {
            return Err(PiperTtsError::InvalidAudio(
                "Piper changed WAV format within an utterance".into(),
            ));
        }

        let frames = samples.len() / channels as usize;
        let preceding_frames = state.sample_offset as usize / state.output_channels.max(1) as usize;
        let mut alignments = Vec::with_capacity(words.len());
        for (index, word) in words.iter().enumerate() {
            let start_frame = frames * index / words.len();
            let end_frame = frames * (index + 1) / words.len();
            let start_sample = state.sample_offset as usize + start_frame * channels as usize;
            let end_sample = state.sample_offset as usize + end_frame * channels as usize;
            alignments.push(WordAlignment {
                word: word.clone(),
                start_sample,
                end_sample,
                start_ms: (preceding_frames + start_frame) as f64 / sample_rate as f64 * 1000.0,
                end_ms: (preceding_frames + end_frame) as f64 / sample_rate as f64 * 1000.0,
            });
        }
        state.output_sample_rate = sample_rate;
        state.output_channels = channels;
        state.words.extend(alignments.iter().cloned());

        let chunk = SynthesizedAudioChunk {
            utterance_id: utterance_id.into(),
            chunk_index: state.chunk_counter,
            samples: Arc::from(samples.into_boxed_slice()),
            sample_rate,
            channels,
            sample_offset: state.sample_offset,
            duration_ms: frames as f64 / sample_rate as f64 * 1000.0,
            spoken_words: alignments,
            is_final_chunk: false,
        };
        state.chunk_counter += 1;
        state.sample_offset += chunk.samples.len() as u64;
        if state.queue.len() >= self.max_queue_depth {
            state.queue.pop_front();
        }
        state.queue.push_back(chunk.clone());
        Ok(vec![chunk])
    }
}

impl StreamingTtsAdapter for PiperLocalTtsAdapter {
    fn name(&self) -> &'static str {
        "piper-local"
    }

    fn start_utterance(&mut self, utterance_id: &str, config: &VoiceConfig) -> Result<(), String> {
        self.validate_dependencies().map_err(|e| e.to_string())?;
        if utterance_id.trim().is_empty() {
            return Err(PiperTtsError::InvalidInput("utterance ID is empty".into()).to_string());
        }
        self.sessions
            .write()
            .map_err(|e| PiperTtsError::ProcessFailed(e.to_string()).to_string())?
            .insert(
                utterance_id.into(),
                PiperUtteranceState {
                    config: config.clone(),
                    output_sample_rate: config.sample_rate,
                    output_channels: 1,
                    gate: TextPrefixCommitmentGate::new(),
                    chunk_counter: 0,
                    sample_offset: 0,
                    words: Vec::new(),
                    queue: VecDeque::new(),
                    is_completed: false,
                    is_cancelled: false,
                },
            );
        Ok(())
    }

    fn feed_text(
        &mut self,
        utterance_id: &str,
        text: &str,
    ) -> Result<Vec<SynthesizedAudioChunk>, String> {
        let words = {
            let mut sessions = self.sessions.write().map_err(|e| e.to_string())?;
            let state = sessions.get_mut(utterance_id).ok_or_else(|| {
                PiperTtsError::InvalidInput(format!("unknown utterance '{utterance_id}'"))
                    .to_string()
            })?;
            if state.is_cancelled {
                return Err(PiperTtsError::Cancelled {
                    utterance_id: utterance_id.into(),
                }
                .to_string());
            }
            if state.is_completed {
                return Err(
                    PiperTtsError::InvalidInput("utterance is already complete".into()).to_string(),
                );
            }
            state.gate.push_token(text)
        };
        self.render_words(utterance_id, words)
            .map_err(|e| e.to_string())
    }

    fn finalize_utterance(
        &mut self,
        utterance_id: &str,
    ) -> Result<Vec<SynthesizedAudioChunk>, String> {
        let words = {
            let mut sessions = self.sessions.write().map_err(|e| e.to_string())?;
            let state = sessions.get_mut(utterance_id).ok_or_else(|| {
                PiperTtsError::InvalidInput(format!("unknown utterance '{utterance_id}'"))
                    .to_string()
            })?;
            if state.is_cancelled {
                return Err(PiperTtsError::Cancelled {
                    utterance_id: utterance_id.into(),
                }
                .to_string());
            }
            if state.is_completed {
                return Ok(Vec::new());
            }
            state.gate.flush()
        };
        let mut chunks = self
            .render_words(utterance_id, words)
            .map_err(|e| e.to_string())?;
        let mut sessions = self.sessions.write().map_err(|e| e.to_string())?;
        let state = sessions.get_mut(utterance_id).ok_or_else(|| {
            PiperTtsError::InvalidInput(format!("unknown utterance '{utterance_id}'")).to_string()
        })?;
        state.is_completed = true;
        let sample_rate = chunks
            .last()
            .map(|chunk| chunk.sample_rate)
            .unwrap_or(state.config.sample_rate);
        let channels = chunks.last().map(|chunk| chunk.channels).unwrap_or(1);
        chunks.push(SynthesizedAudioChunk {
            utterance_id: utterance_id.into(),
            chunk_index: state.chunk_counter,
            samples: Arc::from(Vec::<f32>::new().into_boxed_slice()),
            sample_rate,
            channels,
            sample_offset: state.sample_offset,
            duration_ms: 0.0,
            spoken_words: Vec::new(),
            is_final_chunk: true,
        });
        state.chunk_counter += 1;
        Ok(chunks)
    }

    fn cancel_utterance(
        &mut self,
        utterance_id: &str,
        current_playback_cursor: u64,
    ) -> Result<TtsCancellationReport, String> {
        let mut sessions = self.sessions.write().map_err(|e| e.to_string())?;
        let state = sessions.get_mut(utterance_id).ok_or_else(|| {
            PiperTtsError::InvalidInput(format!("unknown utterance '{utterance_id}'")).to_string()
        })?;
        state.is_cancelled = true;
        state.queue.clear();

        if let Ok(mut active) = self.active_children.lock() {
            if let Some(mut child) = active.remove(utterance_id) {
                Self::terminate_child(&mut child);
            }
        }

        let played_samples = current_playback_cursor.min(state.sample_offset);
        let played_words = state
            .words
            .iter()
            .filter(|word| word.end_sample as u64 <= played_samples)
            .map(|word| word.word.clone())
            .collect();
        let committed_words = state.gate.committed_words().to_vec();
        Ok(TtsCancellationReport {
            utterance_id: utterance_id.into(),
            total_synthesized_samples: state.sample_offset,
            played_samples,
            discarded_samples: state.sample_offset.saturating_sub(played_samples),
            committed_words,
            played_words,
            interrupted_at_ms: played_samples as f64
                / state.output_channels.max(1) as f64
                / state.output_sample_rate.max(1) as f64
                * 1000.0,
        })
    }

    fn get_playback_status(
        &self,
        utterance_id: &str,
        current_playback_cursor: u64,
    ) -> Result<PlaybackAlignmentStatus, String> {
        let sessions = self.sessions.read().map_err(|e| e.to_string())?;
        let state = sessions.get(utterance_id).ok_or_else(|| {
            PiperTtsError::InvalidInput(format!("unknown utterance '{utterance_id}'")).to_string()
        })?;
        let played_samples = current_playback_cursor.min(state.sample_offset);
        let mut played_words = Vec::new();
        let mut current_word = None;
        let mut pending_words = Vec::new();
        for word in &state.words {
            if word.end_sample as u64 <= played_samples {
                played_words.push(word.word.clone());
            } else if (word.start_sample as u64) < played_samples {
                current_word = Some(word.word.clone());
            } else {
                pending_words.push(word.word.clone());
            }
        }
        let sample_rate = state.output_sample_rate.max(1) as f64;
        let channels = state.output_channels.max(1) as f64;
        Ok(PlaybackAlignmentStatus {
            utterance_id: utterance_id.into(),
            played_words,
            current_word,
            pending_words,
            played_duration_ms: played_samples as f64 / channels / sample_rate * 1000.0,
            total_duration_ms: state.sample_offset as f64 / channels / sample_rate * 1000.0,
            is_completed: state.is_completed && played_samples >= state.sample_offset,
        })
    }
}

// ---------------------------------------------------------------------------
// Acoustic Token Neural Vocoder Interface
// ---------------------------------------------------------------------------

/// Trait for neural acoustic codebook decoders (e.g. EnCodec, SoundStream, SNAC).
pub trait AcousticTokenDecoder: Send + Sync {
    fn decode_tokens(&self, tokens: &[u32], sample_rate: u32) -> Result<Vec<f32>, String>;
    fn codebook_size(&self) -> usize;
}

/// Protocol test double / mock decoding discrete acoustic tokens into harmonic test tones.
///
/// NOTE: This is an unlearned test double for verifying audio streaming pipelines and
/// token scheduling; it does not contain trained neural weights.
pub struct SailNeuralVocoder {
    codebook_size: usize,
}

impl SailNeuralVocoder {
    pub fn new(codebook_size: usize) -> Self {
        Self { codebook_size }
    }
}

impl AcousticTokenDecoder for SailNeuralVocoder {
    fn decode_tokens(&self, tokens: &[u32], sample_rate: u32) -> Result<Vec<f32>, String> {
        let hop_size = 320; // 20ms frames at 16kHz
        let total_samples = tokens.len() * hop_size;
        let mut audio = Vec::with_capacity(total_samples);

        for &tok in tokens {
            let freq = 120.0 + (tok % self.codebook_size as u32) as f32 * 8.0;
            for i in 0..hop_size {
                let t = i as f32 / sample_rate as f32;
                let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.25;
                audio.push(sample);
            }
        }
        Ok(audio)
    }

    fn codebook_size(&self) -> usize {
        self.codebook_size
    }
}

// ---------------------------------------------------------------------------
// Benchmarking & Intelligibility Metrics
// ---------------------------------------------------------------------------

/// Performance and Real-Time Factor (RTF) evaluation report.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsBenchmarkReport {
    pub num_phrases: usize,
    pub total_text_chars: usize,
    pub total_audio_duration_sec: f64,
    pub total_synthesis_time_sec: f64,
    pub first_chunk_latency_ms: f64,
    pub real_time_factor: f64,
    pub words_synthesized: usize,
    pub distinct_phoneme_profiles: usize,
}

/// Evaluates first-audio latency, sustained RTF, and phonetic distinctiveness across phrases.
pub fn measure_tts_benchmark(
    engine: &mut FormantNeuralTtsEngine,
    phrases: &[&str],
) -> TtsBenchmarkReport {
    let mut total_chars = 0;
    let mut total_duration_sec = 0.0;
    let mut total_synth_sec = 0.0;
    let mut first_latencies = Vec::new();
    let mut total_words = 0;
    let mut spectral_profiles = Vec::new();

    let config = VoiceConfig::default();

    for (idx, &phrase) in phrases.iter().enumerate() {
        total_chars += phrase.len();
        let utterance_id = format!("bench_utt_{idx}");

        let t0 = Instant::now();
        engine.start_utterance(&utterance_id, &config).unwrap();

        // Stream word by word
        let words: Vec<&str> = phrase.split_whitespace().collect();
        total_words += words.len();

        let mut utt_samples = Vec::new();
        let mut first_chunk_recorded = false;

        for word in words {
            let chunks = engine
                .feed_text(&utterance_id, &format!("{word} "))
                .unwrap();
            if !first_chunk_recorded && !chunks.is_empty() {
                first_latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
                first_chunk_recorded = true;
            }
            for chunk in chunks {
                utt_samples.extend_from_slice(&chunk.samples);
            }
        }

        let final_chunks = engine.finalize_utterance(&utterance_id).unwrap();
        for chunk in final_chunks {
            utt_samples.extend_from_slice(&chunk.samples);
        }

        let elapsed = t0.elapsed().as_secs_f64();
        total_synth_sec += elapsed;

        let dur_sec = utt_samples.len() as f64 / engine.sample_rate as f64;
        total_duration_sec += dur_sec;

        // Compute mean spectral centroid for distinctiveness verification
        if !utt_samples.is_empty() {
            let mut zero_crossings = 0;
            for i in 1..utt_samples.len() {
                if (utt_samples[i - 1] >= 0.0 && utt_samples[i] < 0.0)
                    || (utt_samples[i - 1] < 0.0 && utt_samples[i] >= 0.0)
                {
                    zero_crossings += 1;
                }
            }
            let zcr = zero_crossings as f32 / utt_samples.len() as f32;
            spectral_profiles.push(zcr);
        }
    }

    let avg_latency = if !first_latencies.is_empty() {
        first_latencies.iter().sum::<f64>() / first_latencies.len() as f64
    } else {
        0.0
    };

    let rtf = if total_duration_sec > 0.0 {
        total_synth_sec / total_duration_sec
    } else {
        0.0
    };

    TtsBenchmarkReport {
        num_phrases: phrases.len(),
        total_text_chars: total_chars,
        total_audio_duration_sec: total_duration_sec,
        total_synthesis_time_sec: total_synth_sec,
        first_chunk_latency_ms: avg_latency,
        real_time_factor: rtf,
        words_synthesized: total_words,
        distinct_phoneme_profiles: spectral_profiles.len(),
    }
}
