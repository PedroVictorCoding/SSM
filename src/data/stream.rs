//! Dynamic Streaming Dataset with Experience Replay and Directory Watcher.
//!
//! Enables continual online learning:
//! - Append-only token buffer that can grow dynamically during training.
//! - Experience Replay sampling: mixes newly arrived tokens with historical tokens
//!   to prevent catastrophic forgetting.
//! - Incoming Directory Watcher: monitors a folder (e.g. `data/incoming/`) for new `.txt`
//!   or `.jsonl` files, automatically tokenizes them, and ingests them into the live buffer.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use burn::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::{Batch, Tokenizer, batch_from_starts};

fn default_max_replay_tokens() -> usize {
    1_000_000
}

/// Configuration for streaming continual dataset ingestion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Ratio of training windows sampled from historical data vs newly arrived data (e.g. 0.3 = 30% replay).
    pub replay_ratio: f64,
    /// Minimum tokens required in buffer before training batches can be sampled.
    pub min_tokens_to_train: usize,
    /// Watcher scan interval in milliseconds.
    pub scan_interval_ms: u64,
    /// Maximum experience replay tokens in RAM (bounded replay buffer capacity).
    #[serde(default = "default_max_replay_tokens")]
    pub max_replay_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamSnapshot {
    pub tokens: Vec<u32>,
    pub new_data_watermark: usize,
    pub processed_files: Vec<PathBuf>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            replay_ratio: 0.3,
            min_tokens_to_train: 256,
            scan_interval_ms: 1000,
            max_replay_tokens: default_max_replay_tokens(),
        }
    }
}

/// Dynamic, thread-safe append-only dataset buffer with Experience Replay.
#[derive(Clone)]
pub struct StreamDataset {
    tokens: Arc<RwLock<Vec<u32>>>,
    /// Index marking the boundary between historical data and newly appended data.
    new_data_watermark: Arc<RwLock<usize>>,
    pub config: StreamConfig,
}

impl StreamDataset {
    pub fn snapshot(&self) -> StreamSnapshot {
        let tokens = self.tokens.read().unwrap();
        let watermark = *self.new_data_watermark.read().unwrap();
        StreamSnapshot {
            tokens: tokens.clone(),
            new_data_watermark: watermark,
            processed_files: Vec::new(),
        }
    }

    pub fn restore(&self, snapshot: &StreamSnapshot) -> Result<(), String> {
        if snapshot.new_data_watermark > snapshot.tokens.len() {
            return Err("stream watermark exceeds token count".into());
        }
        let mut tokens = self.tokens.write().unwrap();
        let mut watermark = self.new_data_watermark.write().unwrap();
        *tokens = snapshot.tokens.clone();
        *watermark = snapshot.new_data_watermark;
        Ok(())
    }

    /// Freeze a held-out tail before training; those tokens are removed from replay.
    pub fn reserve_validation(&self, fraction: f64, seq_len: usize) -> Vec<u32> {
        let mut tokens = self.tokens.write().unwrap();
        let mut watermark = self.new_data_watermark.write().unwrap();
        let count = (tokens.len() as f64 * fraction) as usize;
        if count <= seq_len + 1 || tokens.len().saturating_sub(count) <= seq_len + 1 {
            return Vec::new();
        }
        let split = tokens.len() - count;
        let held_out = tokens.split_off(split);
        *watermark = (*watermark).min(split);
        held_out
    }
    /// Creates a new streaming dataset initialized with optional initial tokens.
    pub fn new(initial_tokens: Vec<u32>, config: StreamConfig) -> Self {
        let watermark = initial_tokens.len();
        Self {
            tokens: Arc::new(RwLock::new(initial_tokens)),
            new_data_watermark: Arc::new(RwLock::new(watermark)),
            config,
        }
    }

    /// Appends newly arrived tokens into the active streaming buffer.
    pub fn append_tokens(&self, new_tokens: &[u32]) {
        if new_tokens.is_empty() {
            return;
        }
        let mut tokens = self.tokens.write().unwrap();
        let current_len = tokens.len();
        tokens.extend_from_slice(new_tokens);

        // Update watermark so replay sampler knows what is new
        let mut watermark = self.new_data_watermark.write().unwrap();
        *watermark = current_len;

        // Bounded Experience Replay Buffer: enforce max_replay_tokens capacity
        if self.config.max_replay_tokens > 0 && tokens.len() > self.config.max_replay_tokens {
            let excess = tokens.len() - self.config.max_replay_tokens;
            tokens.drain(0..excess);
            *watermark = watermark.saturating_sub(excess);
        }
    }

    /// Appends raw text, tokenizing it with the provided tokenizer.
    pub fn append_text(&self, text: &str, tokenizer: &Tokenizer) -> usize {
        let tokens = tokenizer.encode(text).unwrap_or_default();
        let count = tokens.len();
        self.append_tokens(&tokens);
        count
    }

    /// Total number of tokens currently in the streaming buffer.
    pub fn total_tokens(&self) -> usize {
        self.tokens.read().unwrap().len()
    }

    /// Number of tokens in historical pool vs newly arrived pool.
    pub fn token_distribution(&self) -> (usize, usize) {
        let tokens = self.tokens.read().unwrap();
        let watermark = *self.new_data_watermark.read().unwrap();
        let total = tokens.len();
        let historical = watermark.min(total);
        let new = total.saturating_sub(historical);
        (historical, new)
    }

    /// Samples a batch using Experience Replay:
    /// Blends windows from historical data and newly streamed data according to `replay_ratio`.
    pub fn sample_batch<B: Backend, R: Rng>(
        &self,
        batch_size: usize,
        seq_len: usize,
        rng: &mut R,
        device: &B::Device,
    ) -> Option<Batch<B>> {
        let tokens_guard = self.tokens.read().unwrap();
        let tokens: &[u32] = &tokens_guard;

        if tokens.len() <= seq_len + 1 || tokens.len() < self.config.min_tokens_to_train {
            return None;
        }

        let max_start = tokens.len() - seq_len - 1;
        let watermark = *self.new_data_watermark.read().unwrap();

        let num_replay = ((batch_size as f64) * self.config.replay_ratio).round() as usize;
        let _num_new = batch_size.saturating_sub(num_replay);

        let mut starts = Vec::with_capacity(batch_size);

        // Sample historical replay windows (if enough historical tokens exist)
        if watermark > seq_len + 1 && num_replay > 0 {
            let hist_max = watermark - seq_len - 1;
            for _ in 0..num_replay {
                starts.push(rng.gen_range(0..=hist_max));
            }
        }

        // Sample newly arrived windows (from watermark to end)
        if watermark + seq_len + 1 < tokens.len() {
            let new_min = watermark;
            for _ in starts.len()..batch_size {
                starts.push(rng.gen_range(new_min..=max_start));
            }
        } else {
            // Fallback to uniform sampling across whole buffer
            for _ in starts.len()..batch_size {
                starts.push(rng.gen_range(0..=max_start));
            }
        }

        Some(batch_from_starts(tokens, &starts, seq_len, device))
    }
}

/// Watches an incoming directory (e.g. `data/incoming/`), ingesting any new text files.
pub struct IncomingDirectoryWatcher {
    incoming_dir: PathBuf,
    processed_files: HashSet<PathBuf>,
}

impl IncomingDirectoryWatcher {
    /// Bound the producer's cleanup receipt to a spool identity, portable between
    /// a WSL downloader and a native Windows trainer viewing the same directory.
    pub fn ingestion_receipt(&self) -> serde_json::Value {
        let spool_id = fs::read(self.incoming_dir.join(".hf_streamer/state.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|state| {
                state
                    .get("id")
                    .and_then(|id| id.as_str())
                    .map(str::to_owned)
            });
        serde_json::json!({"spool_id": spool_id, "processed_files": self.processed_files()})
    }
    pub fn processed_files(&self) -> Vec<PathBuf> {
        let mut names: Vec<_> = self
            .processed_files
            .iter()
            .filter_map(|p| p.file_name().map(PathBuf::from))
            .collect();
        names.sort();
        names
    }
    pub fn restore_processed_files(&mut self, names: &[PathBuf]) -> Result<(), String> {
        if names
            .iter()
            .any(|p| p.components().count() != 1 || p.file_name().is_none())
        {
            return Err("invalid processed stream filename".into());
        }
        self.processed_files = names.iter().map(|p| self.incoming_dir.join(p)).collect();
        Ok(())
    }
    pub fn new(incoming_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let incoming_dir = incoming_dir.as_ref().to_path_buf();
        fs::create_dir_all(&incoming_dir)?;
        Ok(Self {
            incoming_dir,
            processed_files: HashSet::new(),
        })
    }

    /// Scans the directory for newly dropped files, tokenizes them, and appends to `dataset`.
    /// Returns the number of newly ingested tokens.
    pub fn scan_and_ingest(&mut self, dataset: &StreamDataset, tokenizer: &Tokenizer) -> usize {
        let mut new_tokens_count = 0;

        let entries = match fs::read_dir(&self.incoming_dir) {
            Ok(e) => e,
            Err(_) => return 0,
        };

        let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if path.is_file() && !self.processed_files.contains(&path) {
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                if ext == "txt" || ext == "jsonl" || ext == "md" {
                    if let Ok(content) = fs::read_to_string(&path) {
                        if !content.trim().is_empty() {
                            let lower = content.to_lowercase();
                            if lower.contains("\"origin\":\"runtime_synthetic\"")
                                || lower.contains("\"origin\": \"runtime_synthetic\"")
                                || lower.contains("\"origin\":\"agent_trace\"")
                                || lower.contains("\"origin\": \"agent_trace\"")
                                || lower.contains("\"origin\":\"session_capture\"")
                                || lower.contains("\"origin\": \"session_capture\"")
                                || lower.contains("\"provenance\":\"runtime_output\"")
                                || lower.contains("\"provenance\": \"runtime_output\"")
                            {
                                eprintln!(
                                    "[Stream Ingest] Rejected {}: Autonomous recursive self-improvement prohibited. Runtime-generated or session capture data cannot enter training queue.",
                                    path.display()
                                );
                                self.processed_files.insert(path);
                                continue;
                            }

                            let tokens = match tokenizer.encode(&content) {
                                Ok(tokens) if !tokens.is_empty() => tokens,
                                _ => {
                                    eprintln!(
                                        "[Stream Ingest] No tokens from {}; retaining file without acknowledgement (check tokenizer coverage)",
                                        path.display()
                                    );
                                    continue;
                                }
                            };
                            let count = tokens.len();
                            dataset.append_tokens(&tokens);
                            new_tokens_count += count;
                            println!(
                                "[Stream Ingest] Added {} tokens from {}",
                                count,
                                path.file_name().unwrap().to_string_lossy()
                            );
                            self.processed_files.insert(path);
                        }
                    }
                }
            }
        }

        new_tokens_count
    }
}
