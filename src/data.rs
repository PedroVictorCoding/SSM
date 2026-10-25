//! Tokenization and dataset batching (Character-level and Native BPE subword).

pub mod bpe;
pub mod stream;
pub use bpe::BpeTokenizer;
pub use stream::{IncomingDirectoryWatcher, StreamConfig, StreamDataset};

use std::collections::HashMap;
use std::path::Path;

use burn::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Unified tokenizer abstraction supporting both character and subword BPE.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tokenizer {
    Char(CharTokenizer),
    Bpe(BpeTokenizer),
}

impl Tokenizer {
    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Char(c) => c.vocab_size(),
            Tokenizer::Bpe(b) => b.vocab_size(),
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        match self {
            Tokenizer::Char(c) => c.encode(text),
            Tokenizer::Bpe(b) => Ok(b.encode(text)),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Tokenizer::Char(c) => c.decode(ids),
            Tokenizer::Bpe(b) => b.decode(ids),
        }
    }

    /// Returns `(tool_call, tool_call_end, tool_response, tool_response_end)` token IDs if supported.
    pub fn tool_tokens(&self) -> Option<(u32, u32, u32, u32)> {
        match self {
            Tokenizer::Bpe(b) => Some((
                b.tool_call_id,
                b.tool_call_end_id,
                b.tool_resp_id,
                b.tool_resp_end_id,
            )),
            Tokenizer::Char(_) => None,
        }
    }

    pub fn rebuild_index(&mut self) {
        match self {
            Tokenizer::Char(c) => c.rebuild_index(),
            Tokenizer::Bpe(b) => b.rebuild_index(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        // 1. Try tagged Tokenizer
        if let Ok(mut tok) = serde_json::from_str::<Tokenizer>(&content) {
            tok.rebuild_index();
            return Ok(tok);
        }
        // 2. Try raw BpeTokenizer
        if let Ok(mut bpe) = serde_json::from_str::<BpeTokenizer>(&content) {
            bpe.rebuild_index();
            return Ok(Tokenizer::Bpe(bpe));
        }
        // 3. Fallback to legacy CharTokenizer (json array of chars)
        if let Ok(chars) = serde_json::from_str::<Vec<char>>(&content) {
            return Ok(Tokenizer::Char(CharTokenizer::from_chars(chars)));
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse tokenizer from {}", path.as_ref().display()),
        ))
    }
}

/// Maps each distinct character of the training corpus to an id (sorted by code point).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharTokenizer {
    pub chars: Vec<char>,
    #[serde(skip)]
    index: HashMap<char, u32>,
}

impl CharTokenizer {
    pub fn rebuild_index(&mut self) {
        self.index = self
            .chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
    }

    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        Self::from_chars(chars)
    }

    pub fn from_chars(chars: Vec<char>) -> Self {
        let index = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Self { chars, index }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    /// Encodes `text`; characters missing from the vocabulary are reported as an error.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        text.chars()
            .map(|c| {
                self.index
                    .get(&c)
                    .copied()
                    .ok_or_else(|| format!("character {c:?} is not in the vocabulary"))
            })
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.chars[i as usize]).collect()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, serde_json::to_string(&self.chars)?)
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let chars: Vec<char> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        Ok(Self::from_chars(chars))
    }
}

/// Train / validation token streams.
pub struct TextDataset {
    pub train: Vec<u32>,
    pub val: Vec<u32>,
}

impl TextDataset {
    /// Contiguous split: the first `1 - val_fraction` of the corpus trains, the rest validates.
    pub fn split(tokens: Vec<u32>, val_fraction: f64) -> Self {
        let n_val = ((tokens.len() as f64) * val_fraction).round() as usize;
        let n_train = tokens.len() - n_val;
        let val = tokens[n_train..].to_vec();
        let mut train = tokens;
        train.truncate(n_train);
        Self { train, val }
    }
}

/// A batch of `(inputs, targets)` with targets shifted one position to the left.
pub struct Batch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

/// Builds a batch from window start offsets into `tokens`.
pub fn batch_from_starts<B: Backend>(
    tokens: &[u32],
    starts: &[usize],
    seq_len: usize,
    device: &B::Device,
) -> Batch<B> {
    let mut inputs = Vec::with_capacity(starts.len() * seq_len);
    let mut targets = Vec::with_capacity(starts.len() * seq_len);
    for &s in starts {
        inputs.extend(tokens[s..s + seq_len].iter().map(|&t| t as i64));
        targets.extend(tokens[s + 1..s + seq_len + 1].iter().map(|&t| t as i64));
    }
    let shape = [starts.len(), seq_len];
    Batch {
        inputs: Tensor::from_data(TensorData::new(inputs, shape), device),
        targets: Tensor::from_data(TensorData::new(targets, shape), device),
    }
}

/// Uniformly random training windows.
pub fn random_batch<B: Backend, R: Rng>(
    tokens: &[u32],
    batch_size: usize,
    seq_len: usize,
    rng: &mut R,
    device: &B::Device,
) -> Batch<B> {
    assert!(tokens.len() > seq_len + 1, "corpus shorter than seq_len");
    let max_start = tokens.len() - seq_len - 1;
    let starts: Vec<usize> = (0..batch_size)
        .map(|_| rng.gen_range(0..=max_start))
        .collect();
    batch_from_starts(tokens, &starts, seq_len, device)
}

/// Deterministic, evenly spaced evaluation windows (at most `max_windows`) covering the stream.
pub fn eval_starts(tokens: &[u32], seq_len: usize, max_windows: usize) -> Vec<usize> {
    if tokens.len() <= seq_len + 1 || max_windows == 0 {
        return Vec::new();
    }
    let max_start = tokens.len() - seq_len - 1;
    let n = ((tokens.len() - 1) / seq_len).clamp(1, max_windows);
    let stride = max_start / n.max(2).saturating_sub(1).max(1);
    (0..n).map(|i| (i * stride).min(max_start)).collect()
}
