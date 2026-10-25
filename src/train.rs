//! Training loop: AdamW + linear warmup / cosine decay, gradient clipping, periodic validation,
//! best-checkpoint saving and a CSV metrics log.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, ModuleVisitor, ParamId};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng as StdRng;
use serde::{Deserialize, Serialize};

use crate::data::{
    BpeTokenizer, CharTokenizer, IncomingDirectoryWatcher, StreamDataset, TextDataset, Tokenizer,
    batch_from_starts, eval_starts, random_batch,
};
use crate::generate::{SamplingConfig, generate_stateful};
use crate::model::{HybridLm, HybridLmConfig};

/// Checkpoint state metadata saved alongside model and optimizer weights.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CheckpointState {
    pub step: usize,
    pub best_val_loss: f64,
    pub elapsed_secs: f64,
    pub lr: f64,
    pub rng: Option<StdRng>,
    pub model_file: Option<String>,
    pub optimizer_file: Option<String>,
    pub config_file: Option<String>,
    pub stream_file: Option<String>,
    pub validation_file: Option<String>,
}

#[derive(Config, Debug)]
pub struct TrainConfig {
    pub model: HybridLmConfig,
    /// Synchronized stage timing (None/false disables it, including legacy configs).
    pub profile: Option<bool>,
    /// Equivalent per-parameter norm clipping without host reads (opt-in).
    pub device_clip: Option<bool>,
    #[config(default = 128)]
    pub seq_len: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 2000)]
    pub steps: usize,
    #[config(default = 2e-3)]
    pub lr: f64,
    #[config(default = 1e-4)]
    pub min_lr: f64,
    #[config(default = 100)]
    pub warmup_steps: usize,
    #[config(default = 0.1)]
    pub weight_decay: f32,
    #[config(default = 1.0)]
    pub grad_clip: f32,
    #[config(default = 0.1)]
    pub val_fraction: f64,
    #[config(default = 250)]
    pub eval_every: usize,
    /// Periodic checkpoint saving interval in steps (0 = save on eval).
    #[config(default = 0)]
    pub save_every: usize,
    /// Use Native BPE subword tokenization instead of character-level.
    #[config(default = false)]
    pub use_bpe: bool,
    /// Target vocabulary size for BPE tokenizer training.
    #[config(default = 512)]
    pub bpe_vocab_size: usize,
    /// Resume training from existing checkpoint in out_dir if available.
    #[config(default = false)]
    pub resume: bool,
    /// Maximum number of validation windows (each `seq_len` long).
    #[config(default = 64)]
    pub eval_windows: usize,
    #[config(default = 25)]
    pub log_every: usize,
    /// Parameter/activation snapshot interval (0 disables).
    #[config(default = 25)]
    pub viz_every: usize,
    /// Maximum sampled values per parameter tensor (0 = every value).
    #[config(default = 0)]
    pub viz_max_values: usize,
    /// Characters generated as a qualitative sample at every evaluation (0 = none).
    #[config(default = 200)]
    pub sample_chars: usize,
    /// Stop early after this many minutes (0 = no limit).
    #[config(default = 0.0)]
    pub max_minutes: f64,
    #[config(default = 42)]
    pub seed: u64,
}

/// Learning rate at `step` (1-based): linear warmup then cosine decay to `min_lr`.
pub fn learning_rate(cfg: &TrainConfig, step: usize) -> f64 {
    if cfg.warmup_steps > 0 && step <= cfg.warmup_steps {
        return cfg.lr * step as f64 / cfg.warmup_steps as f64;
    }
    let decay_steps = cfg.steps.saturating_sub(cfg.warmup_steps).max(1);
    let progress = ((step - cfg.warmup_steps) as f64 / decay_steps as f64).min(1.0);
    cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Mean next-token cross-entropy (nats) over the given windows.
pub fn evaluate<B: Backend>(
    model: &HybridLm<B>,
    tokens: &[u32],
    starts: &[usize],
    seq_len: usize,
    batch_size: usize,
    device: &B::Device,
) -> f64 {
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut total = 0.0;
    let mut count = 0usize;
    for chunk in starts.chunks(batch_size.max(1)) {
        let batch = batch_from_starts::<B>(tokens, chunk, seq_len, device);
        let logits = model.forward(batch.inputs);
        let [b, l, v] = logits.dims();
        let ce_loss = loss_fn.forward(logits.reshape([b * l, v]), batch.targets.reshape([b * l]));
        total += ce_loss.into_scalar().elem::<f64>() * chunk.len() as f64;
        count += chunk.len();
    }
    total / count.max(1) as f64
}

/// Files of a training run directory.
pub struct RunPaths {
    pub dir: PathBuf,
}

impl RunPaths {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }
    /// Record path without extension (the recorder appends `.mpk`).
    pub fn model(&self) -> PathBuf {
        self.dir.join("model")
    }
    /// Latest checkpoint model weights (without extension).
    pub fn checkpoint_model(&self) -> PathBuf {
        self.dir.join("checkpoint_model")
    }
    /// Optimizer state (without extension).
    pub fn optimizer(&self) -> PathBuf {
        self.dir.join("optimizer")
    }
    /// Metadata file tracking current step and metrics.
    pub fn checkpoint_state(&self) -> PathBuf {
        self.dir.join("checkpoint.json")
    }
    pub fn model_config(&self) -> PathBuf {
        self.dir.join("model_config.json")
    }
    pub fn train_config(&self) -> PathBuf {
        self.dir.join("train_config.json")
    }
    pub fn vocab(&self) -> PathBuf {
        self.dir.join("vocab.json")
    }
    pub fn metrics(&self) -> PathBuf {
        self.dir.join("metrics.csv")
    }
}

fn validate_config(cfg: &TrainConfig, streaming: bool) -> std::io::Result<()> {
    if cfg.batch_size == 0
        || cfg.seq_len == 0
        || cfg.log_every == 0
        || (!streaming && cfg.eval_every == 0)
    {
        return Err(std::io::Error::other(
            "batch-size, seq-len, log-every (and regular eval-every) must be positive",
        ));
    }
    if !cfg.lr.is_finite()
        || !cfg.min_lr.is_finite()
        || cfg.lr < 0.0
        || cfg.min_lr < 0.0
        || !cfg.grad_clip.is_finite()
        || cfg.grad_clip < 0.0
        || !(0.0..1.0).contains(&cfg.val_fraction)
    {
        return Err(std::io::Error::other(
            "invalid learning rate, gradient clip, or validation fraction",
        ));
    }
    Ok(())
}

fn recorder() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

fn checkpoint_metadata(paths: &RunPaths, resume: bool) -> std::io::Result<Option<CheckpointState>> {
    if !resume {
        return Ok(None);
    }
    let bytes = fs::read(paths.checkpoint_state()).map_err(|e| {
        std::io::Error::other(format!("cannot resume without checkpoint metadata: {e}"))
    })?;
    Ok(Some(serde_json::from_slice(&bytes).map_err(|e| {
        std::io::Error::other(format!("invalid checkpoint metadata: {e}"))
    })?))
}

fn checkpoint_path(
    paths: &RunPaths,
    file: &Option<String>,
    legacy: &str,
) -> std::io::Result<PathBuf> {
    match file {
        None => Ok(paths.dir.join(legacy)),
        Some(file) => {
            let relative = Path::new(file);
            if relative
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(std::io::Error::other(
                    "checkpoint path must stay inside the run directory",
                ));
            }
            Ok(paths.dir.join(relative))
        }
    }
}

/// Commit a complete immutable generation, then switch the metadata pointer.
/// Legacy aliases remain for existing tools, but restore uses generation paths.
fn save_checkpoint<B: AutodiffBackend, O: Optimizer<HybridLm<B>, B>>(
    paths: &RunPaths,
    cfg: &TrainConfig,
    model: &HybridLm<B>,
    optim: &O,
    rng: &StdRng,
    step: usize,
    best_val_loss: f64,
    elapsed_secs: f64,
    lr: f64,
    stream: Option<(&StreamDataset, Option<&IncomingDirectoryWatcher>, &[u32])>,
) -> std::io::Result<()> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let relative = format!("checkpoints/step-{step:08}-{}-{stamp}", std::process::id());
    let dir = paths.dir.join(&relative);
    fs::create_dir_all(paths.dir.join("checkpoints"))?;
    fs::create_dir(&dir)?;
    model
        .clone()
        .save_file(dir.join("model"), &recorder())
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    recorder()
        .record(optim.to_record(), dir.join("optimizer"))
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    cfg.model.save(dir.join("model_config.json"))?;
    let (stream_file, validation_file) = if let Some((dataset, watcher, validation)) = stream {
        let mut snapshot = dataset.snapshot();
        snapshot.processed_files = watcher.map(|w| w.processed_files()).unwrap_or_default();
        fs::write(dir.join("stream.json"), serde_json::to_vec(&snapshot)?)?;
        fs::write(dir.join("validation.json"), serde_json::to_vec(validation)?)?;
        (
            Some(format!("{relative}/stream.json")),
            Some(format!("{relative}/validation.json")),
        )
    } else {
        (None, None)
    };
    let state = CheckpointState {
        step,
        best_val_loss: if best_val_loss.is_finite() {
            best_val_loss
        } else {
            f64::MAX
        },
        elapsed_secs,
        lr,
        rng: Some(rng.clone()),
        model_file: Some(format!("{relative}/model")),
        optimizer_file: Some(format!("{relative}/optimizer")),
        config_file: Some(format!("{relative}/model_config.json")),
        stream_file,
        validation_file,
    };
    fs::write(dir.join("state.json"), serde_json::to_vec_pretty(&state)?)?;
    for file in fs::read_dir(&dir)? {
        let path = file?.path();
        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?
            .sync_all()?;
    }
    for (source, destination) in [
        ("model.mpk", "checkpoint_model.mpk"),
        ("optimizer.mpk", "optimizer.mpk"),
    ] {
        crate::storage::atomic_write(&paths.dir.join(destination), &fs::read(dir.join(source))?)?;
    }
    crate::storage::atomic_write(
        &paths.checkpoint_state(),
        &serde_json::to_vec_pretty(&state)?,
    )
}

/// Atomically publish a small snapshot so readers never see half-written JSON.
fn publish_status(paths: &RunPaths, phase: &str, step: usize, elapsed: f64) -> std::io::Result<()> {
    let status = serde_json::json!({
        "phase": phase, "step": step, "elapsed_secs": elapsed,
        "updated_at": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64(),
    });
    fs::write(
        paths.dir.join("status.json.tmp"),
        serde_json::to_vec(&status)?,
    )?;
    fs::rename(
        paths.dir.join("status.json.tmp"),
        paths.dir.join("status.json"),
    )
}

/// One small host read for all parameter/gradient health flags, rather than
/// copying every tensor or synchronizing after each parameter.
fn check_finite<B: AutodiffBackend>(
    model: &HybridLm<B>,
    grads: Option<&GradientsParams>,
) -> Result<(), String> {
    struct Check<'a, B: AutodiffBackend> {
        grads: Option<&'a GradientsParams>,
        flags: Vec<Tensor<B::InnerBackend, 1>>,
        ids: Vec<ParamId>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Check<'_, B> {
        fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
            let value = match self.grads {
                Some(grads) => match grads.get::<B::InnerBackend, D>(id) {
                    Some(value) => value,
                    None => return,
                },
                None => tensor.clone().inner(),
            };
            // NaN comparisons are false; infinities exceed the largest finite f32.
            self.flags
                .push(value.abs().lower_equal_elem(f32::MAX).all().float());
            self.ids.push(id);
        }
    }
    let mut check = Check::<B> {
        grads,
        flags: Vec::new(),
        ids: Vec::new(),
    };
    model.visit(&mut check);
    if check.flags.is_empty() {
        return Ok(());
    }
    let flags = Tensor::cat(check.flags, 0)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| format!("reading health flags: {e:?}"))?;
    if let Some(index) = flags.iter().position(|v| *v != 1.0) {
        return Err(format!(
            "non-finite {} in parameter {}",
            if grads.is_some() {
                "gradient"
            } else {
                "weight"
            },
            check.ids[index]
        ));
    }
    Ok(())
}

fn numerical_failure(
    paths: &RunPaths,
    step: usize,
    elapsed: f64,
    message: String,
) -> std::io::Error {
    let _ = publish_status(paths, "failed", step, elapsed);
    let _ = fs::write(
        paths.dir.join("failure.json"),
        serde_json::json!({"step": step, "message": message}).to_string(),
    );
    std::io::Error::other(format!(
        "step {step}: {message}. Training stopped; no checkpoint was saved for this step. Recover from a verified finite checkpoint."
    ))
}

/// Loads a trained model, its config and tokenizer from a run directory.
pub fn load_run<B: Backend>(
    dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(HybridLm<B>, HybridLmConfig, Tokenizer), String> {
    let paths = RunPaths::new(dir);
    let cfg = HybridLmConfig::load(paths.model_config())
        .map_err(|e| format!("loading {}: {e}", paths.model_config().display()))?;
    let tokenizer = Tokenizer::load(paths.vocab())
        .map_err(|e| format!("loading {}: {e}", paths.vocab().display()))?;
    let model = cfg
        .init::<B>(device)
        .load_file(paths.model(), &recorder(), device)
        .map_err(|e| format!("loading {}: {e}", paths.model().display()))?;
    Ok((model, cfg, tokenizer))
}

/// Summary returned at the end of training.
#[derive(Debug, Clone)]
pub struct TrainSummary {
    pub steps_run: usize,
    pub best_val_loss: f64,
    pub final_train_loss: f64,
    pub elapsed_secs: f64,
}

/// Trains a model on `text`, writing checkpoints and metrics into `out_dir`.
pub fn train<B: AutodiffBackend>(
    cfg: TrainConfig,
    text: &str,
    out_dir: impl AsRef<Path>,
    device: &B::Device,
) -> std::io::Result<TrainSummary> {
    validate_config(&cfg, false)?;
    let paths = RunPaths::new(out_dir);
    fs::create_dir_all(&paths.dir)?;
    let mut profiler = crate::profile::StepProfiler::new(&paths.dir, cfg.profile.unwrap_or(false))?;

    let resume_state = checkpoint_metadata(&paths, cfg.resume)?;
    let is_resuming = resume_state.is_some();

    let mut cfg = cfg;
    let tokenizer = if is_resuming {
        Tokenizer::load(paths.vocab())?
    } else if cfg.use_bpe {
        Tokenizer::Bpe(BpeTokenizer::train(text, cfg.bpe_vocab_size))
    } else {
        Tokenizer::Char(CharTokenizer::from_text(text))
    };

    let requested_work_efficient_scan = cfg.model.work_efficient_scan;
    if let Some(state) = &resume_state {
        let config_path = checkpoint_path(&paths, &state.config_file, "model_config.json")?;
        cfg.model =
            HybridLmConfig::load(config_path).map_err(|e| std::io::Error::other(e.to_string()))?;
    }

    // Execution strategy may be changed while preserving checkpoint parameters.
    if let Some(enabled) = requested_work_efficient_scan {
        cfg.model.work_efficient_scan = Some(enabled);
    }

    let tokens = tokenizer
        .encode(text)
        .expect("corpus encodes with its own vocab");
    let data = TextDataset::split(tokens, cfg.val_fraction);
    cfg.model.vocab_size = tokenizer.vocab_size();

    if !paths.vocab().exists() || !is_resuming {
        tokenizer.save(paths.vocab())?;
    }
    if !paths.model_config().exists() || !is_resuming || requested_work_efficient_scan.is_some() {
        cfg.model.save(paths.model_config())?;
    }
    cfg.save(paths.train_config())?;

    B::seed(cfg.seed);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut model: HybridLm<B> = cfg.model.init(device);
    fs::write(
        paths.dir.join("telemetry.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "backend": crate::backend::BACKEND_NAME,
            "total_params": model.total_params(),
            "active_params_per_token": model.active_params_per_token(),
        }))?,
    )?;
    let mut samples = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths.dir.join("samples.jsonl"))?;
    if paths.dir.join("neural.json").exists() {
        fs::remove_file(paths.dir.join("neural.json"))?;
    }
    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .with_grad_clipping(if cfg.device_clip.unwrap_or(false) {
            None
        } else {
            Some(GradientClippingConfig::Norm(cfg.grad_clip))
        })
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);

    let mut start_step = 0;
    let mut accumulated_elapsed = 0.0;
    let mut best_val = f64::INFINITY;

    if let Some(state) = resume_state {
        start_step = state.step;
        best_val = state.best_val_loss;
        accumulated_elapsed = state.elapsed_secs;
        let legacy = if paths.dir.join("checkpoint_model.mpk").exists() {
            "checkpoint_model"
        } else {
            "model"
        };
        model = model
            .load_file(
                checkpoint_path(&paths, &state.model_file, legacy)?,
                &recorder(),
                device,
            )
            .map_err(|e| std::io::Error::other(format!("Failed to restore model: {e}")))?;
        let optimizer = checkpoint_path(&paths, &state.optimizer_file, "optimizer")?;
        if optimizer.with_extension("mpk").exists() {
            optim =
                optim.load_record(recorder().load(optimizer, device).map_err(|e| {
                    std::io::Error::other(format!("Failed to restore optimizer: {e}"))
                })?);
        } else if state.optimizer_file.is_some() {
            return Err(std::io::Error::other(
                "committed optimizer checkpoint is missing",
            ));
        } else {
            println!("No optimizer checkpoint found; using fresh AdamW moments.");
        }
        if let Some(saved_rng) = state.rng {
            rng = saved_rng;
        } else {
            println!("Legacy checkpoint: sampler RNG restarts from seed.");
        }
        println!("Resuming run from checkpoint at step {start_step}");
    }

    if let Err(message) = check_finite(&model, None) {
        return Err(numerical_failure(
            &paths,
            start_step,
            accumulated_elapsed,
            message,
        ));
    }

    if cfg.viz_every > 0 {
        crate::telemetry::publish(
            &model,
            None,
            &paths.dir,
            start_step,
            cfg.viz_max_values,
            Vec::new(),
        )?;
    }
    publish_status(&paths, "training", start_step, accumulated_elapsed)?;
    if start_step >= cfg.steps {
        println!(
            "Checkpoint step ({}) is already >= requested total steps ({}). Increase --steps to continue training further.",
            start_step, cfg.steps
        );
        return Ok(TrainSummary {
            steps_run: start_step,
            best_val_loss: best_val,
            final_train_loss: 0.0,
            elapsed_secs: accumulated_elapsed,
        });
    }

    let val_starts = eval_starts(&data.val, cfg.seq_len, cfg.eval_windows);
    let prompt = tokenizer.encode("\n").unwrap_or_else(|_| vec![0]);

    println!(
        "Corpus: {} chars, vocab {} | train {} / val {} tokens | {} val windows",
        text.chars().count(),
        tokenizer.vocab_size(),
        data.train.len(),
        data.val.len(),
        val_starts.len()
    );
    println!(
        "Model: {} total params ({} active params/token) | layers [{}] | d_model {} | d_state {} | expand {} | dt_rank {}",
        model.total_params(),
        model.active_params_per_token(),
        cfg.model.layout(),
        cfg.model.d_model,
        cfg.model.d_state,
        cfg.model.expand,
        cfg.model.mamba_config().resolved_dt_rank(),
    );
    println!(
        "Training: {} steps (starting at {}) × batch {} × seq {} | lr {:.1e} → {:.1e} (warmup {}) | scan chunk {}",
        cfg.steps,
        start_step,
        cfg.batch_size,
        cfg.seq_len,
        cfg.lr,
        cfg.min_lr,
        cfg.warmup_steps,
        cfg.model.scan_chunk
    );
    if cfg.model.work_efficient_scan.unwrap_or(false) {
        println!("Scan strategy: work-efficient pair reduction (scan-chunk overridden)");
    }

    let write_header = !paths.metrics().exists()
        || fs::metadata(paths.metrics())
            .map(|m| m.len() == 0)
            .unwrap_or(true);
    let mut metrics = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths.metrics())?;
    if write_header {
        writeln!(
            metrics,
            "step,lr,train_loss,val_loss,val_bpc,tokens_per_sec,elapsed_secs"
        )?;
    }

    let start = Instant::now();
    let mut window_start = Instant::now();
    let mut running: Option<Tensor<B::InnerBackend, 1>> = None;
    let mut running_steps = 0usize;
    let mut last_train_loss = f64::NAN;
    let mut steps_run = start_step;

    for step in (start_step + 1)..=cfg.steps {
        profiler.begin::<B>(step, device);
        let lr = learning_rate(&cfg, step);
        let batch =
            random_batch::<B, _>(&data.train, cfg.batch_size, cfg.seq_len, &mut rng, device);

        profiler.stage::<B>("batch", device);
        let observe =
            cfg.viz_every > 0 && (step == 1 || step % cfg.viz_every == 0 || step == cfg.steps);
        let (logits, aux_loss, activations) = model.forward_observed(batch.inputs, observe);
        let [b, l, v] = logits.dims();
        let ce_loss = loss_fn.forward(logits.reshape([b * l, v]), batch.targets.reshape([b * l]));
        let loss = match aux_loss {
            Some(aux) => ce_loss + aux.mul_scalar(0.01),
            None => ce_loss,
        };

        profiler.stage::<B>("forward_loss", device);
        let loss_value = loss.clone().inner().into_scalar().elem::<f64>();
        if !loss_value.is_finite() {
            return Err(numerical_failure(
                &paths,
                step,
                accumulated_elapsed + start.elapsed().as_secs_f64(),
                format!("non-finite training loss ({loss_value}) before optimizer update"),
            ));
        }

        // Keep the loss accumulator on-device. The numerical check above
        // deliberately synchronizes each step before weights can be updated.
        let detached = loss.clone().inner();
        running = Some(match running.take() {
            Some(r) => r + detached,
            None => detached,
        });
        running_steps += 1;

        profiler.stage::<B>("loss_guard", device);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        profiler.stage::<B>("backward", device);
        if let Err(message) = check_finite(&model, Some(&grads)) {
            return Err(numerical_failure(
                &paths,
                step,
                accumulated_elapsed + start.elapsed().as_secs_f64(),
                message,
            ));
        }
        profiler.stage::<B>("gradient_guard", device);
        if observe {
            crate::telemetry::publish(
                &model,
                Some(&grads),
                &paths.dir,
                step,
                cfg.viz_max_values,
                activations,
            )?;
        }
        profiler.stage::<B>("visualization", device);
        let grads = if cfg.device_clip.unwrap_or(false) {
            crate::optim::clip_gradients_on_device(&model, grads, cfg.grad_clip)
        } else {
            grads
        };
        model = optim.step(lr, model, grads);
        profiler.stage::<B>("optimizer", device);
        steps_run = step;

        let out_of_time =
            cfg.max_minutes > 0.0 && start.elapsed().as_secs_f64() > cfg.max_minutes * 60.0;
        let is_last = step == cfg.steps || out_of_time;

        let total_elapsed = accumulated_elapsed + start.elapsed().as_secs_f64();
        let should_save_checkpoint = step % cfg.eval_every == 0
            || is_last
            || (cfg.save_every > 0 && step % cfg.save_every == 0);

        if should_save_checkpoint {
            if let Err(message) = check_finite(&model, None) {
                return Err(numerical_failure(&paths, step, total_elapsed, message));
            }
        }

        if step % cfg.log_every == 0 || is_last {
            let sum = running
                .take()
                .map(|r| r.into_scalar().elem::<f64>())
                .unwrap_or(0.0);
            last_train_loss = sum / running_steps.max(1) as f64;
            let tok_s =
                (running_steps * b * l) as f64 / window_start.elapsed().as_secs_f64().max(1e-9);
            println!(
                "step {step:>6} | lr {lr:.2e} | train loss {last_train_loss:.4} | {tok_s:>8.0} tok/s | {:>6.1}s",
                total_elapsed
            );
            writeln!(
                metrics,
                "{step},{lr},{last_train_loss},,,{tok_s},{total_elapsed}",
            )?;
            metrics.flush()?;
            publish_status(&paths, "training", step, total_elapsed)?;
            running_steps = 0;
            window_start = Instant::now();
        }

        if step % cfg.eval_every == 0 || is_last {
            publish_status(&paths, "evaluating", step, total_elapsed)?;
            let eval_start = Instant::now();
            let valid = model.valid();
            let val_loss = evaluate(
                &valid,
                &data.val,
                &val_starts,
                cfg.seq_len,
                cfg.batch_size,
                device,
            );
            let bpc = val_loss / std::f64::consts::LN_2;
            if !val_loss.is_finite() {
                return Err(numerical_failure(
                    &paths,
                    step,
                    total_elapsed,
                    format!("non-finite validation loss ({val_loss})"),
                ));
            }
            let improved = val_loss < best_val;
            if improved {
                best_val = val_loss;
                model
                    .clone()
                    .save_file(paths.model(), &recorder())
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
            }
            println!(
                "  eval  | val loss {val_loss:.4} | {bpc:.3} bits/char | best {best_val:.4}{} | {:.1}s",
                if improved { " (saved)" } else { "" },
                eval_start.elapsed().as_secs_f64()
            );
            writeln!(metrics, "{step},{lr},,{val_loss},{bpc},,{total_elapsed}",)?;
            metrics.flush()?;

            if cfg.sample_chars > 0 {
                let sampling = SamplingConfig {
                    temperature: 0.8,
                    top_k: 0,
                    seed: step as u64,
                    ..Default::default()
                };
                let (ids, _) =
                    generate_stateful(&valid, &prompt, cfg.sample_chars, &sampling, device);
                let text = tokenizer.decode(&ids);
                println!("  ---- sample ----\n{}\n  ----------------", text.trim());
                writeln!(
                    samples,
                    "{}",
                    serde_json::json!({"step": step, "text": text})
                )?;
                samples.flush()?;
            }
            publish_status(&paths, "training", step, total_elapsed)?;
            // Evaluation time should not count against training throughput.
            window_start = Instant::now();
        }

        if should_save_checkpoint {
            save_checkpoint(
                &paths,
                &cfg,
                &model,
                &optim,
                &rng,
                step,
                best_val,
                total_elapsed,
                lr,
                None,
            )?;
        }

        profiler.stage::<B>("report_eval_checkpoint", device);
        profiler.finish()?;
        if out_of_time {
            println!("Time budget of {} min reached.", cfg.max_minutes);
            break;
        }
    }

    let final_elapsed = accumulated_elapsed + start.elapsed().as_secs_f64();
    publish_status(&paths, "finished", steps_run, final_elapsed)?;
    Ok(TrainSummary {
        steps_run,
        best_val_loss: best_val,
        final_train_loss: last_train_loss,
        elapsed_secs: final_elapsed,
    })
}

/// Continuously trains a model on a dynamic, append-only `StreamDataset`.
/// Backwards-compatible streaming API. A frozen validation tail is reserved from
/// the initial buffer when it is large enough; callers may supply a separate set.
pub fn stream_train<B: AutodiffBackend>(
    cfg: TrainConfig,
    dataset: StreamDataset,
    watcher: Option<IncomingDirectoryWatcher>,
    tokenizer: Tokenizer,
    out_dir: impl AsRef<Path>,
    device: &B::Device,
) -> std::io::Result<TrainSummary> {
    stream_train_with_validation::<B>(cfg, dataset, watcher, tokenizer, out_dir, device, None)
}

pub fn stream_train_with_validation<B: AutodiffBackend>(
    cfg: TrainConfig,
    dataset: StreamDataset,
    mut watcher: Option<IncomingDirectoryWatcher>,
    mut tokenizer: Tokenizer,
    out_dir: impl AsRef<Path>,
    device: &B::Device,
    validation_tokens: Option<Vec<u32>>,
) -> std::io::Result<TrainSummary> {
    validate_config(&cfg, true)?;
    let paths = RunPaths::new(out_dir);
    fs::create_dir_all(&paths.dir)?;
    let mut profiler = crate::profile::StepProfiler::new(&paths.dir, cfg.profile.unwrap_or(false))?;
    let resume_state = checkpoint_metadata(&paths, cfg.resume)?;
    let mut cfg = cfg;
    let requested_scan = cfg.model.work_efficient_scan;
    if let Some(state) = &resume_state {
        tokenizer = Tokenizer::load(paths.vocab())?;
        cfg.model = HybridLmConfig::load(checkpoint_path(
            &paths,
            &state.config_file,
            "model_config.json",
        )?)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
        if let Some(enabled) = requested_scan {
            cfg.model.work_efficient_scan = Some(enabled);
        }
        if state.stream_file.is_some() {
            let snapshot: crate::data::stream::StreamSnapshot = serde_json::from_slice(&fs::read(
                checkpoint_path(&paths, &state.stream_file, "stream.json")?,
            )?)?;
            dataset.restore(&snapshot).map_err(std::io::Error::other)?;
            if let Some(watcher) = watcher.as_mut() {
                watcher
                    .restore_processed_files(&snapshot.processed_files)
                    .map_err(std::io::Error::other)?;
            }
        } else {
            println!("Legacy checkpoint has no replay snapshot; using the supplied buffer.");
        }
    }
    cfg.model.vocab_size = tokenizer.vocab_size();
    let validation = if let Some(state) = &resume_state {
        if state.validation_file.is_some() {
            let saved: Vec<u32> = serde_json::from_slice(&fs::read(checkpoint_path(
                &paths,
                &state.validation_file,
                "validation.json",
            )?)?)?;
            if validation_tokens.as_ref().is_some_and(|v| v != &saved) {
                return Err(std::io::Error::other(
                    "resume validation data differs from the checkpoint",
                ));
            }
            saved
        } else {
            validation_tokens
                .unwrap_or_else(|| dataset.reserve_validation(cfg.val_fraction, cfg.seq_len))
        }
    } else {
        validation_tokens
            .unwrap_or_else(|| dataset.reserve_validation(cfg.val_fraction, cfg.seq_len))
    };
    if !validation.is_empty() && validation.len() <= cfg.seq_len + 1 {
        return Err(std::io::Error::other(
            "validation corpus is shorter than sequence length",
        ));
    }
    if dataset
        .snapshot()
        .tokens
        .iter()
        .chain(validation.iter())
        .any(|id| *id as usize >= tokenizer.vocab_size())
    {
        return Err(std::io::Error::other(
            "stream or validation token exceeds the saved vocabulary",
        ));
    }
    let val_starts = eval_starts(&validation, cfg.seq_len, cfg.eval_windows);
    if val_starts.is_empty() {
        println!("No held-out validation windows; validation metrics remain empty.");
    }
    tokenizer.save(paths.vocab())?;
    cfg.model.save(paths.model_config())?;
    cfg.save(paths.train_config())?;
    B::seed(cfg.seed);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut model: HybridLm<B> = cfg.model.init(device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .with_grad_clipping(if cfg.device_clip.unwrap_or(false) {
            None
        } else {
            Some(GradientClippingConfig::Norm(cfg.grad_clip))
        })
        .init();
    let mut step = 0;
    let mut accumulated_elapsed = 0.0;
    let mut best_val = f64::MAX;
    if let Some(state) = resume_state {
        step = state.step;
        accumulated_elapsed = state.elapsed_secs;
        best_val = state.best_val_loss;
        let legacy = if paths.dir.join("checkpoint_model.mpk").exists() {
            "checkpoint_model"
        } else {
            "model"
        };
        model = model
            .load_file(
                checkpoint_path(&paths, &state.model_file, legacy)?,
                &recorder(),
                device,
            )
            .map_err(|e| std::io::Error::other(format!("Failed to restore model: {e}")))?;
        let optimizer = checkpoint_path(&paths, &state.optimizer_file, "optimizer")?;
        if optimizer.with_extension("mpk").exists() {
            optim =
                optim.load_record(recorder().load(optimizer, device).map_err(|e| {
                    std::io::Error::other(format!("Failed to restore optimizer: {e}"))
                })?);
        } else if state.optimizer_file.is_some() {
            return Err(std::io::Error::other(
                "committed optimizer checkpoint is missing",
            ));
        } else {
            println!("No optimizer checkpoint found; using fresh AdamW moments.");
        }
        if let Some(saved_rng) = state.rng {
            rng = saved_rng;
        } else {
            println!("Legacy checkpoint: sampler RNG restarts from seed.");
        }
        println!("Resuming streaming run from checkpoint at step {step}");
    }
    check_finite(&model, None)
        .map_err(|e| numerical_failure(&paths, step, accumulated_elapsed, e))?;
    fs::write(
        paths.dir.join("telemetry.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
        "backend":crate::backend::BACKEND_NAME,"training_mode":"streaming",
        "total_params":model.total_params(),"active_params_per_token":model.active_params_per_token()}))?,
    )?;
    if cfg.viz_every > 0 {
        crate::telemetry::publish(
            &model,
            None,
            &paths.dir,
            step + 1,
            cfg.viz_max_values,
            Vec::new(),
        )?;
    }
    let live_sea = paths.dir.join("model.sea");
    crate::export::pack_to_artifact(&model.valid(), &cfg.model, &tokenizer, &live_sea)
        .map_err(std::io::Error::other)?;
    publish_status(&paths, "training", step, accumulated_elapsed)?;
    let header = "step,lr,train_loss,val_loss,val_bpc,tokens_per_sec,elapsed_secs,total_tokens,hist_tokens,new_tokens";
    if paths.metrics().exists() {
        let old = fs::read_to_string(paths.metrics())?;
        if old.lines().next().is_some_and(|h| h != header) {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            fs::rename(
                paths.metrics(),
                paths.dir.join(format!("metrics.legacy-{stamp}.csv")),
            )?;
        }
    }
    let needs_header = !paths.metrics().exists() || fs::metadata(paths.metrics())?.len() == 0;
    let mut metrics = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths.metrics())?;
    if needs_header {
        writeln!(metrics, "{header}")?;
        metrics.flush()?;
    }
    let mut samples = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths.dir.join("samples.jsonl"))?;
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let start = Instant::now();
    let start_step = step;
    let mut window_start = Instant::now();
    let mut running = 0.0;
    let mut count = 0;
    let mut last_train_loss = 0.0;
    let mut last_scan: Option<Instant> = None;
    while step < cfg.steps {
        if cfg.max_minutes > 0.0 && start.elapsed().as_secs_f64() >= cfg.max_minutes * 60.0 {
            break;
        }
        if last_scan.is_none_or(|scan| {
            scan.elapsed().as_millis() >= dataset.config.scan_interval_ms as u128
        }) {
            if let Some(watcher) = watcher.as_mut() {
                watcher.scan_and_ingest(&dataset, &tokenizer);
            }
            last_scan = Some(Instant::now());
        }
        profiler.begin::<B>(step + 1, device);
        let batch = match dataset.sample_batch(cfg.batch_size, cfg.seq_len, &mut rng, device) {
            Some(batch) => batch,
            None => {
                publish_status(
                    &paths,
                    "waiting_for_data",
                    step,
                    accumulated_elapsed + start.elapsed().as_secs_f64(),
                )?;
                std::thread::sleep(std::time::Duration::from_millis(
                    dataset.config.scan_interval_ms.clamp(1, 500),
                ));
                continue;
            }
        };
        step += 1;
        profiler.stage::<B>("batch", device);
        let lr = learning_rate(&cfg, step);
        let observe = cfg.viz_every > 0
            && (step == start_step + 1 || step % cfg.viz_every == 0 || step == cfg.steps);
        let [b, l] = batch.inputs.dims();
        let (logits, aux, activations) = model.forward_observed(batch.inputs, observe);
        let ce = loss_fn.forward(
            logits.reshape([b * l, cfg.model.vocab_size]),
            batch.targets.reshape([b * l]),
        );
        let loss = if let Some(aux) = aux {
            ce + aux.mul_scalar(0.01)
        } else {
            ce
        };
        profiler.stage::<B>("forward_loss", device);
        let scalar = loss.clone().inner().into_scalar().elem::<f64>();
        let elapsed = accumulated_elapsed + start.elapsed().as_secs_f64();
        if !scalar.is_finite() {
            return Err(numerical_failure(
                &paths,
                step,
                elapsed,
                "non-finite streaming training loss".into(),
            ));
        }
        profiler.stage::<B>("loss_guard", device);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        profiler.stage::<B>("backward", device);
        check_finite(&model, Some(&grads))
            .map_err(|e| numerical_failure(&paths, step, elapsed, e))?;
        profiler.stage::<B>("gradient_guard", device);
        if observe {
            crate::telemetry::publish(
                &model,
                Some(&grads),
                &paths.dir,
                step,
                cfg.viz_max_values,
                activations,
            )?;
        }
        profiler.stage::<B>("visualization", device);
        let grads = if cfg.device_clip.unwrap_or(false) {
            crate::optim::clip_gradients_on_device(&model, grads, cfg.grad_clip)
        } else {
            grads
        };
        model = optim.step(lr, model, grads);
        profiler.stage::<B>("optimizer", device);
        last_train_loss = scalar;
        running += scalar;
        count += 1;
        let elapsed = accumulated_elapsed + start.elapsed().as_secs_f64();
        let is_last = step == cfg.steps
            || (cfg.max_minutes > 0.0 && start.elapsed().as_secs_f64() >= cfg.max_minutes * 60.0);
        let evaluate_now = !val_starts.is_empty()
            && ((cfg.eval_every > 0 && step % cfg.eval_every == 0) || is_last);
        let save_now =
            evaluate_now || (cfg.save_every > 0 && step % cfg.save_every == 0) || is_last;
        if save_now {
            check_finite(&model, None).map_err(|e| numerical_failure(&paths, step, elapsed, e))?;
        }
        let (hist, fresh) = dataset.token_distribution();
        let total = hist + fresh;
        if step % cfg.log_every == 0 || is_last {
            let rate = (count * b * l) as f64 / window_start.elapsed().as_secs_f64().max(1e-9);
            let average = running / count.max(1) as f64;
            println!(
                "step {step:6} | lr {lr:.2e} | train loss {average:.4} | {rate:8.0} tok/s | {total} stream tokens"
            );
            writeln!(
                metrics,
                "{step},{lr},{average},,,{rate},{elapsed},{total},{hist},{fresh}"
            )?;
            metrics.flush()?;
            running = 0.0;
            count = 0;
            window_start = Instant::now();
        }
        if evaluate_now {
            publish_status(&paths, "evaluating", step, elapsed)?;
            let valid = model.valid();
            let val = evaluate(
                &valid,
                &validation,
                &val_starts,
                cfg.seq_len,
                cfg.batch_size,
                device,
            );
            if !val.is_finite() {
                return Err(numerical_failure(
                    &paths,
                    step,
                    elapsed,
                    "non-finite streaming validation loss".into(),
                ));
            }
            if val < best_val {
                best_val = val;
                model
                    .clone()
                    .save_file(paths.model(), &recorder())
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
            }
            let bpc = val / std::f64::consts::LN_2;
            println!("  eval | val loss {val:.4} | best {best_val:.4}");
            writeln!(
                metrics,
                "{step},{lr},,{val},{bpc},,{elapsed},{total},{hist},{fresh}"
            )?;
            metrics.flush()?;
            if cfg.sample_chars > 0 {
                let prompt = tokenizer.encode("\n").unwrap_or_else(|_| vec![0]);
                let sampling = SamplingConfig {
                    temperature: 0.8,
                    top_k: 0,
                    seed: step as u64,
                    ..Default::default()
                };
                let (ids, _) =
                    generate_stateful(&valid, &prompt, cfg.sample_chars, &sampling, device);
                writeln!(
                    samples,
                    "{}",
                    serde_json::json!({"step":step,"text":tokenizer.decode(&ids)})
                )?;
                samples.flush()?;
            }
            window_start = Instant::now();
        }
        if save_now {
            save_checkpoint(
                &paths,
                &cfg,
                &model,
                &optim,
                &rng,
                step,
                best_val,
                elapsed,
                lr,
                Some((&dataset, watcher.as_ref(), &validation)),
            )?;
            crate::export::pack_to_artifact(&model.valid(), &cfg.model, &tokenizer, &live_sea)
                .map_err(std::io::Error::other)?;
            println!(
                "[Live Artifact] Atomically updated {} (step {step})",
                live_sea.display()
            );
        }
        publish_status(
            &paths,
            "training",
            step,
            accumulated_elapsed + start.elapsed().as_secs_f64(),
        )?;
        profiler.stage::<B>("report_eval_checkpoint", device);
        profiler.finish()?;
        if is_last {
            break;
        }
    }
    let elapsed = accumulated_elapsed + start.elapsed().as_secs_f64();
    // If the time limit expired while waiting after updates, publish those updates.
    if step > start_step && step < cfg.steps {
        check_finite(&model, None).map_err(|e| numerical_failure(&paths, step, elapsed, e))?;
        save_checkpoint(
            &paths,
            &cfg,
            &model,
            &optim,
            &rng,
            step,
            best_val,
            elapsed,
            learning_rate(&cfg, step),
            Some((&dataset, watcher.as_ref(), &validation)),
        )?;
        crate::export::pack_to_artifact(&model.valid(), &cfg.model, &tokenizer, &live_sea)
            .map_err(std::io::Error::other)?;
    }
    publish_status(&paths, "finished", step, elapsed)?;
    Ok(TrainSummary {
        steps_run: step,
        best_val_loss: best_val,
        final_train_loss: last_train_loss,
        elapsed_secs: elapsed,
    })
}

#[cfg(test)]
mod numerical_tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn health_checks_reject_nan_weights_and_infinite_gradients() {
        let device = NdArrayDevice::Cpu;
        let mut model = HybridLmConfig::new(8)
            .with_d_model(4)
            .with_n_layers(1)
            .with_d_state(2)
            .with_d_conv(2)
            .init::<B>(&device);
        assert!(check_finite(&model, None).is_ok());
        let mut grads = GradientsParams::new();
        grads.register::<NdArray<f32>, 1>(
            model.norm_f.gamma.id,
            Tensor::from_data([0.0, f32::INFINITY, 0.0, 0.0], &device),
        );
        assert!(
            check_finite(&model, Some(&grads))
                .unwrap_err()
                .contains("gradient")
        );
        model.norm_f.gamma = burn::module::Param::from_tensor(Tensor::<B, 1>::from_data(
            [1.0, f32::NAN, 1.0, 1.0],
            &device,
        ));
        assert!(check_finite(&model, None).unwrap_err().contains("weight"));
    }

    #[test]
    fn numerical_failure_keeps_checkpoint_files_and_reports_failed_state() {
        let dir = std::env::temp_dir().join(format!("ssm-numerical-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let paths = RunPaths::new(&dir);
        fs::write(paths.dir.join("checkpoint.json"), "last healthy checkpoint").unwrap();
        fs::write(paths.dir.join("checkpoint_model.mpk"), "healthy weights").unwrap();
        let error = numerical_failure(&paths, 42, 3.0, "non-finite training loss".into());
        assert!(error.to_string().contains("no checkpoint was saved"));
        assert_eq!(
            fs::read_to_string(paths.dir.join("checkpoint.json")).unwrap(),
            "last healthy checkpoint"
        );
        assert_eq!(
            fs::read_to_string(paths.dir.join("checkpoint_model.mpk")).unwrap(),
            "healthy weights"
        );
        let status: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("status.json")).unwrap()).unwrap();
        assert_eq!(status["phase"], "failed");
        assert_eq!(status["step"], 42);
        fs::remove_dir_all(dir).unwrap();
    }
}
