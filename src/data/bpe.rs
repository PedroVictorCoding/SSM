//! Native Byte-Pair Encoding (BPE) subword tokenizer with byte-level fallback
//! and native tool-calling control tokens.
//!
//! Features:
//! - Byte-level fallback (every UTF-8 byte 0..=255 has a dedicated base token).
//!   Guarantees 100% vocabulary coverage with zero `<unk>` out-of-vocabulary tokens.
//! - Tool-calling control tokens: `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`.
//! - Self-contained BPE training from any text corpus to a target vocabulary size.
//! - Fast greedy pair-merge encoding with ranking tables.
//! - Clean JSON persistence format for merges and vocabulary.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

pub const SPECIAL_PAD: &str = "<|pad|>";
pub const SPECIAL_EOS: &str = "<|endoftext|>";
pub const SPECIAL_TOOL_CALL: &str = "<tool_call>";
pub const SPECIAL_TOOL_CALL_END: &str = "</tool_call>";
pub const SPECIAL_TOOL_RESP: &str = "<tool_response>";
pub const SPECIAL_TOOL_RESP_END: &str = "</tool_response>";

fn default_pad_id() -> u32 {
    0
}
fn default_eos_id() -> u32 {
    1
}
fn default_tool_call_id() -> u32 {
    2
}
fn default_tool_call_end_id() -> u32 {
    3
}
fn default_tool_resp_id() -> u32 {
    4
}
fn default_tool_resp_end_id() -> u32 {
    5
}
fn default_byte_offset() -> u32 {
    6
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeRule {
    pub a: u32,
    pub b: u32,
    pub rank: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// Token ID -> raw byte sequence.
    pub id_to_bytes: Vec<Vec<u8>>,
    /// Raw byte sequence -> Token ID.
    #[serde(skip)]
    pub bytes_to_id: HashMap<Vec<u8>, u32>,
    /// Serializable merge rules.
    pub merge_rules: Vec<MergeRule>,
    /// Merge lookup table in memory: `(token_a, token_b) -> rank`.
    #[serde(skip)]
    pub merges: HashMap<(u32, u32), u32>,
    /// Special token IDs.
    #[serde(default = "default_pad_id")]
    pub pad_id: u32,
    #[serde(default = "default_eos_id")]
    pub eos_id: u32,
    #[serde(default = "default_tool_call_id")]
    pub tool_call_id: u32,
    #[serde(default = "default_tool_call_end_id")]
    pub tool_call_end_id: u32,
    #[serde(default = "default_tool_resp_id")]
    pub tool_resp_id: u32,
    #[serde(default = "default_tool_resp_end_id")]
    pub tool_resp_end_id: u32,
    #[serde(default = "default_byte_offset")]
    pub byte_offset: u32,
}

impl BpeTokenizer {
    /// Creates a base tokenizer with 256 byte tokens plus special control tokens.
    pub fn new_base() -> Self {
        let mut id_to_bytes = Vec::with_capacity(300);
        let mut bytes_to_id = HashMap::with_capacity(300);

        // Special tokens
        let pad_id = 0u32;
        let pad_bytes = SPECIAL_PAD.as_bytes().to_vec();
        bytes_to_id.insert(pad_bytes.clone(), pad_id);
        id_to_bytes.push(pad_bytes);

        let eos_id = 1u32;
        let eos_bytes = SPECIAL_EOS.as_bytes().to_vec();
        bytes_to_id.insert(eos_bytes.clone(), eos_id);
        id_to_bytes.push(eos_bytes);

        let tool_call_id = 2u32;
        let tool_call_bytes = SPECIAL_TOOL_CALL.as_bytes().to_vec();
        bytes_to_id.insert(tool_call_bytes.clone(), tool_call_id);
        id_to_bytes.push(tool_call_bytes);

        let tool_call_end_id = 3u32;
        let tool_call_end_bytes = SPECIAL_TOOL_CALL_END.as_bytes().to_vec();
        bytes_to_id.insert(tool_call_end_bytes.clone(), tool_call_end_id);
        id_to_bytes.push(tool_call_end_bytes);

        let tool_resp_id = 4u32;
        let tool_resp_bytes = SPECIAL_TOOL_RESP.as_bytes().to_vec();
        bytes_to_id.insert(tool_resp_bytes.clone(), tool_resp_id);
        id_to_bytes.push(tool_resp_bytes);

        let tool_resp_end_id = 5u32;
        let tool_resp_end_bytes = SPECIAL_TOOL_RESP_END.as_bytes().to_vec();
        bytes_to_id.insert(tool_resp_end_bytes.clone(), tool_resp_end_id);
        id_to_bytes.push(tool_resp_end_bytes);

        let byte_offset = 6u32;

        // 256 individual byte tokens (IDs 6..=261)
        for b in 0..=255u8 {
            let b_vec = vec![b];
            let id = id_to_bytes.len() as u32;
            bytes_to_id.insert(b_vec.clone(), id);
            id_to_bytes.push(b_vec);
        }

        Self {
            id_to_bytes,
            bytes_to_id,
            merge_rules: Vec::new(),
            merges: HashMap::new(),
            pad_id,
            eos_id,
            tool_call_id,
            tool_call_end_id,
            tool_resp_id,
            tool_resp_end_id,
            byte_offset,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Rebuilds reverse lookup tables after deserialization.
    pub fn rebuild_index(&mut self) {
        self.bytes_to_id.clear();
        for (id, bytes) in self.id_to_bytes.iter().enumerate() {
            self.bytes_to_id.insert(bytes.clone(), id as u32);
        }
        self.merges.clear();
        for rule in &self.merge_rules {
            self.merges.insert((rule.a, rule.b), rule.rank);
        }
    }

    /// Trains a BPE tokenizer from raw text up to `target_vocab_size`.
    pub fn train(text: &str, target_vocab_size: usize) -> Self {
        let mut tokenizer = Self::new_base();
        assert!(
            target_vocab_size >= tokenizer.vocab_size(),
            "target_vocab_size must be at least {}",
            tokenizer.vocab_size()
        );

        let offset = tokenizer.byte_offset;
        let words: Vec<Vec<u32>> = text
            .split_inclusive(char::is_whitespace)
            .filter(|w| !w.is_empty())
            .map(|w| {
                // Map each byte to its base byte token ID (byte + offset)
                w.as_bytes().iter().map(|&b| (b as u32) + offset).collect()
            })
            .collect();

        let mut corpus_words = words;
        let merges_needed = target_vocab_size - tokenizer.vocab_size();

        for merge_rank in 0..merges_needed {
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for word in &corpus_words {
                if word.len() < 2 {
                    continue;
                }
                for window in word.windows(2) {
                    *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
                }
            }

            let best_pair = pair_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .filter(|&(_, count)| count > 1);

            let (pair, _) = match best_pair {
                Some(p) => p,
                None => break,
            };

            let new_id = tokenizer.id_to_bytes.len() as u32;
            let mut new_bytes = tokenizer.id_to_bytes[pair.0 as usize].clone();
            new_bytes.extend_from_slice(&tokenizer.id_to_bytes[pair.1 as usize]);

            tokenizer.id_to_bytes.push(new_bytes.clone());
            tokenizer.bytes_to_id.insert(new_bytes, new_id);
            tokenizer.merges.insert(pair, merge_rank as u32);
            tokenizer.merge_rules.push(MergeRule {
                a: pair.0,
                b: pair.1,
                rank: merge_rank as u32,
            });

            for word in &mut corpus_words {
                let mut new_word = Vec::with_capacity(word.len());
                let mut i = 0;
                while i < word.len() {
                    if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                        new_word.push(new_id);
                        i += 2;
                    } else {
                        new_word.push(word[i]);
                        i += 1;
                    }
                }
                *word = new_word;
            }
        }

        tokenizer
    }

    fn encode_word(&self, word: &str, result: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }

        let mut tokens: Vec<u32> = word
            .as_bytes()
            .iter()
            .map(|&b| (b as u32) + self.byte_offset)
            .collect();

        while tokens.len() >= 2 {
            let mut best_pair: Option<((u32, u32), u32, usize)> = None;

            for (idx, window) in tokens.windows(2).enumerate() {
                let pair = (window[0], window[1]);
                if let Some(&rank) = self.merges.get(&pair) {
                    match best_pair {
                        None => best_pair = Some((pair, rank, idx)),
                        Some((_, best_rank, _)) if rank < best_rank => {
                            best_pair = Some((pair, rank, idx));
                        }
                        _ => {}
                    }
                }
            }

            match best_pair {
                None => break,
                Some((pair, _, pos)) => {
                    let mut b = self.id_to_bytes[pair.0 as usize].clone();
                    b.extend_from_slice(&self.id_to_bytes[pair.1 as usize]);
                    if let Some(&merged_id) = self.bytes_to_id.get(&b) {
                        tokens[pos] = merged_id;
                        tokens.remove(pos + 1);
                    } else {
                        break;
                    }
                }
            }
        }

        result.extend(tokens);
    }

    /// Encodes text into BPE token IDs, recognizing special tool tokens natively.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let special_markers = [
            (SPECIAL_TOOL_CALL, self.tool_call_id),
            (SPECIAL_TOOL_CALL_END, self.tool_call_end_id),
            (SPECIAL_TOOL_RESP, self.tool_resp_id),
            (SPECIAL_TOOL_RESP_END, self.tool_resp_end_id),
            (SPECIAL_PAD, self.pad_id),
            (SPECIAL_EOS, self.eos_id),
        ];

        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut earliest: Option<(usize, usize, u32)> = None;
            for &(marker, id) in &special_markers {
                if let Some(idx) = remaining.find(marker) {
                    match earliest {
                        None => earliest = Some((idx, marker.len(), id)),
                        Some((min_idx, _, _)) if idx < min_idx => {
                            earliest = Some((idx, marker.len(), id));
                        }
                        _ => {}
                    }
                }
            }

            match earliest {
                Some((0, len, id)) => {
                    result.push(id);
                    remaining = &remaining[len..];
                }
                Some((idx, len, id)) => {
                    let prefix = &remaining[..idx];
                    for word in prefix.split_inclusive(char::is_whitespace) {
                        self.encode_word(word, &mut result);
                    }
                    result.push(id);
                    remaining = &remaining[idx + len..];
                }
                None => {
                    for word in remaining.split_inclusive(char::is_whitespace) {
                        self.encode_word(word, &mut result);
                    }
                    break;
                }
            }
        }

        result
    }

    /// Decodes BPE token IDs back into a UTF-8 string, rendering tool tokens.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            if id == self.pad_id || id == self.eos_id {
                continue;
            }
            if id == self.tool_call_id {
                bytes.extend_from_slice(SPECIAL_TOOL_CALL.as_bytes());
                continue;
            }
            if id == self.tool_call_end_id {
                bytes.extend_from_slice(SPECIAL_TOOL_CALL_END.as_bytes());
                continue;
            }
            if id == self.tool_resp_id {
                bytes.extend_from_slice(SPECIAL_TOOL_RESP.as_bytes());
                continue;
            }
            if id == self.tool_resp_end_id {
                bytes.extend_from_slice(SPECIAL_TOOL_RESP_END.as_bytes());
                continue;
            }
            if let Some(token_bytes) = self.id_to_bytes.get(id as usize) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Checks whether a token ID corresponds to EOS.
    pub fn is_eos(&self, id: u32) -> bool {
        id == self.eos_id
    }

    /// Returns the raw byte sequence corresponding to a token ID.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        if id == self.pad_id || id == self.eos_id {
            Vec::new()
        } else if id == self.tool_call_id {
            SPECIAL_TOOL_CALL.as_bytes().to_vec()
        } else if id == self.tool_call_end_id {
            SPECIAL_TOOL_CALL_END.as_bytes().to_vec()
        } else if id == self.tool_resp_id {
            SPECIAL_TOOL_RESP.as_bytes().to_vec()
        } else if id == self.tool_resp_end_id {
            SPECIAL_TOOL_RESP_END.as_bytes().to_vec()
        } else if let Some(bytes) = self.id_to_bytes.get(id as usize) {
            bytes.clone()
        } else {
            Vec::new()
        }
    }

    /// Saves the BPE tokenizer to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }

    /// Loads a BPE tokenizer from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut tokenizer: Self = serde_json::from_str(&content)?;
        tokenizer.rebuild_index();
        Ok(tokenizer)
    }
}
