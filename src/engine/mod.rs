//! Concurrent High-Integrity Inference Engine for SAIL.
//!
//! Provides snapshot isolation and multi-threaded parallel inference:
//! - Background trainer continuously updates weights and atomically publishes `model.sea`.
//! - Multiple concurrent inference workers run in parallel without blocking or crashes.
//! - Uses `RwLock`-synchronized generational pointer swapping: active workers hold immutable
//!   `Arc<ModelGeneration>` handles while `check_and_reload()` swaps to newly published generations.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use burn::prelude::*;

use crate::data::Tokenizer;
use crate::export::load_artifact;
use crate::generate::{GenerationStats, SamplingConfig, generate_stateful};
use crate::model::{HybridLm, HybridLmConfig};

/// An immutable, snapshot-isolated generation of a loaded SAIL model.
pub struct ModelGeneration<B: Backend> {
    pub model: HybridLm<B>,
    pub config: HybridLmConfig,
    pub tokenizer: Tokenizer,
    pub device: B::Device,
    pub modified_time: SystemTime,
    pub generation_id: u64,
}

unsafe impl<B: Backend> Send for ModelGeneration<B> {}
unsafe impl<B: Backend> Sync for ModelGeneration<B> {}

/// Concurrent High-Integrity Engine supporting parallel execution and live updates.
#[derive(Clone)]
pub struct Engine<B: Backend> {
    model_path: PathBuf,
    current: Arc<RwLock<Arc<ModelGeneration<B>>>>,
    device: B::Device,
}

impl<B: Backend> Engine<B> {
    /// Opens a model artifact and initializes the concurrent engine.
    pub fn open(model_path: impl AsRef<Path>, device: &B::Device) -> Result<Self, String> {
        let path = model_path.as_ref().to_path_buf();
        let metadata = fs::metadata(&path)
            .map_err(|e| format!("reading metadata of {}: {e}", path.display()))?;
        let mtime = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);

        let (model, config, tokenizer) = load_artifact::<B>(&path, device)?;
        let generation = ModelGeneration {
            model,
            config,
            tokenizer,
            device: device.clone(),
            modified_time: mtime,
            generation_id: 1,
        };

        Ok(Self {
            model_path: path,
            current: Arc::new(RwLock::new(Arc::new(generation))),
            device: device.clone(),
        })
    }

    /// Returns a shared reference to the current active generation for snapshot isolation.
    pub fn active_generation(&self) -> Arc<ModelGeneration<B>> {
        self.current.read().unwrap().clone()
    }

    /// Checks if a newer `.sea` artifact has been published by the trainer and atomically reloads it.
    /// In-flight readers on previous generations continue uninterrupted.
    pub fn check_and_reload(&self) -> Result<bool, String> {
        let metadata = match fs::metadata(&self.model_path) {
            Ok(m) => m,
            Err(_) => return Ok(false),
        };
        let file_mtime = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);

        let cur_gen = self.active_generation();
        if file_mtime > cur_gen.modified_time {
            // Attempt loading the freshly published atomic artifact
            match load_artifact::<B>(&self.model_path, &self.device) {
                Ok((model, config, tokenizer)) => {
                    let next_gen = ModelGeneration {
                        model,
                        config,
                        tokenizer,
                        device: self.device.clone(),
                        modified_time: file_mtime,
                        generation_id: cur_gen.generation_id + 1,
                    };
                    let mut lock = self.current.write().unwrap();
                    *lock = Arc::new(next_gen);
                    Ok(true)
                }
                Err(e) => {
                    // If a write is in mid-flush, skip reload safely without crashing active readers
                    eprintln!("[Engine] Notice: skipping partial file during reload check: {e}");
                    Ok(false)
                }
            }
        } else {
            Ok(false)
        }
    }

    /// Executes thread-safe, stateful autoregressive generation from the active generation snapshot.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Result<(String, GenerationStats), String> {
        let generation = self.active_generation();
        let prompt_ids = generation.tokenizer.encode(prompt)?;
        if prompt_ids.is_empty() {
            return Err("Prompt must not be empty".into());
        }

        let (ids, stats) = generate_stateful(
            &generation.model,
            &prompt_ids,
            max_tokens,
            sampling,
            &generation.device,
        );
        let text = generation.tokenizer.decode(&ids);
        Ok((text, stats))
    }

    /// Executes multiple inference queries concurrently in parallel across threads.
    pub fn generate_parallel(
        &self,
        prompts: &[String],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Vec<Result<(String, GenerationStats), String>> {
        use std::thread;

        let generation = self.active_generation();
        let mut handles = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            let model_clone = generation.model.clone();
            let tokenizer_clone = generation.tokenizer.clone();
            let device_clone = generation.device.clone();
            let prompt_str = prompt.clone();
            let sampling_cfg = sampling.clone();

            let handle = thread::spawn(move || {
                let prompt_ids = tokenizer_clone.encode(&prompt_str)?;
                if prompt_ids.is_empty() {
                    return Err("Prompt must not be empty".into());
                }
                let (ids, stats) = generate_stateful(
                    &model_clone,
                    &prompt_ids,
                    max_tokens,
                    &sampling_cfg,
                    &device_clone,
                );
                let text = tokenizer_clone.decode(&ids);
                Ok((text, stats))
            });

            handles.push(handle);
        }

        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("Worker thread panicked".into()))
            })
            .collect()
    }
}
