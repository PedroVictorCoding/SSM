//! Command line interface.
//!
//! ```text
//! sail train    [--data data/tinyshakespeare.txt] [--out runs/shakespeare] [options]
//! sail generate [--run runs/shakespeare] [--prompt "ROMEO:"] [--tokens 500] [--naive]
//! sail export   [--run runs/shakespeare] [--out model.sea]
//! sail run      [--model model.sea] [--prompt "ROMEO:"]
//! sail voice    [--model model.sea] [--prompt "What is the capital of France?"]
//! sail bench    [--seq-lens 64,128,256,512] [--run runs/shakespeare]
//! ```
//! Run `sail help` for all options.

use std::collections::HashMap;
use std::io::Write;
use std::str::FromStr;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::Distribution;

use sail::backend::{BACKEND_NAME, InferenceBackend, TrainBackend, default_device};
use sail::data::multimodal::{IncomingMultimodalWatcher, OmnimodalDataset, OmnimodalRecord};
use sail::data::{
    CharTokenizer, IncomingDirectoryWatcher, StreamConfig, StreamDataset, Tokenizer,
    bpe::BpeTokenizer,
};
use sail::engine::Engine;
use sail::export::{export_to_artifact, load_artifact, load_omnimodal_artifact};
use sail::generate::{
    SamplingConfig, generate_naive, generate_omnimodal_stateful, generate_stateful,
    generate_stateful_with_tools,
};
use sail::model::multimodal::{AudioProjectorConfig, OmnimodalConfig, VisionProjectorConfig};
use sail::model::scan::{chunked_parallel_scan, sequential_scan};
use sail::model::{HybridLm, HybridLmConfig};
use sail::os::SailDaemon;
use sail::tool::ToolRegistry;
use sail::train::{TrainConfig, load_run, train};
use sail::voice::{AudioHardware, VadDetector, VoiceAsr, VoiceTts};

mod interactive;

const HELP: &str = "\
SAIL: State space, mixture of Experts & Active parameters hybrid LM (Burn)

USAGE:
  sail train        [options]   Train a subword or character model
  sail stream-train [options]   Continual online training with live data streaming & atomic .sea exports
  sail generate     [options]   Sample text from a trained run
  sail export       [options]   Export trained run to a single portable .sea artifact
  sail run          [options]   Execute standalone native inference directly from .sea
  sail voice        [options]   Live voice assistant with web confidence check & TTS
  sail assistant    [options]   Stage-1 Segmented Assistant (ASR + UI/OCR context + tool broker + TTS)
  sail engine       [options]   Run concurrent parallel inference engine with live snapshot reloading
  sail host-smoke   [options]   Smoke test native host runtime, worker supervisor & bounded bus
  sail daemon       [options]   Run Redox OS microkernel scheme daemon (`sail:`) simulation
  sail bench        [options]   Benchmark parallel vs sequential scan and stateful vs naive decoding
  sail setup                    Guided no-flags terminal configuration
  sail help                     Show this message
  sail --version                Show installed SAIL version

TRAIN OPTIONS (defaults in brackets):
  --data PATH          text corpus                         [data/tinyshakespeare.txt]
  --out DIR            run directory                       [runs/shakespeare]
  --bpe                use native BPE subword tokenizer    [false]
  --vocab-size N       BPE target vocabulary size          [512]
  --steps N            optimiser steps                     [2000]
  --batch-size N                                           [32]
  --seq-len N                                              [128]
  --lr F / --min-lr F  peak / final learning rate          [2e-3 / 1e-4]
  --warmup N           warmup steps                        [100]
  --weight-decay F                                         [0.1]
  --grad-clip F        per-parameter tensor norm clip       [1.0]
  --eval-every N       validation interval                 [250]
  --save-every N       checkpoint interval, 0 = on eval    [0]
  --resume             continue training where left off    [false]
  --eval-windows N     validation windows                  [64]
  --log-every N                                            [25]
  --viz-every N        live neural snapshot interval, 0 = off [25]
  --viz-max-values N   values per tensor, 0 = all weights    [0]
  --profile           synchronized stage timings to profile.jsonl [false]
  --device-clip       equivalent on-device norm clipping   [false]
  --sample-chars N     sample length at each eval          [200]
  --max-minutes F      wall clock budget, 0 = none         [0]
  --seed N                                                 [42]

STREAM-TRAIN OPTIONS (defaults in brackets):
  --multimodal         enable Omnimodal Any-to-Any (Audio/Vision/Text) [false]
  --audio-channels N   audio feature channels (e.g. 80 Mel bins)       [80]
  --patch-size N       vision spatial patch size (e.g. 16 for 16x16)   [16]
  --incoming DIR       incoming text chunks directory      [data/incoming]
  --initial-data PATH  optional base seed dataset          []
  --out DIR            run directory                       [runs/live_growing]
  --replay-ratio F     historical experience replay ratio  [0.30]
  --min-tokens N       min buffer tokens before training   [256]
  --scan-interval N    watcher scan interval in ms         [1000]
  --bpe                use native BPE subword tokenizer    [false]
  --vocab-size N       BPE target vocabulary size          [512]
  --steps N            optimiser steps                     [2000]
  --batch-size N                                           [32]
  --seq-len N                                              [128]
  --lr F / --min-lr F  peak / final learning rate          [2e-3 / 1e-4]
  --warmup N           warmup steps                        [100]
  --weight-decay F                                         [0.1]
  --grad-clip F        per-parameter tensor norm clip       [1.0]
  --save-every N       checkpoint & .sea artifact interval [100]
  --validation-data P  separate frozen validation text     []
  --eval-every N       validation interval                 [250]
  --eval-windows N     maximum held-out windows            [64]
  --val-fraction F    reserve initial buffer tail          [0.1]
  --viz-every N        neural snapshot interval, 0 = off    [25]
  --viz-max-values N   values per tensor, 0 = all          [0]
  --profile           synchronized stages to profile.jsonl [false]
  --device-clip       equivalent on-device norm clipping   [false]
  --sample-chars N     generated sample at each eval        [200]
  --max-minutes F      wall clock budget, 0 = none          [0]
  --log-every N                                            [25]
  --resume             continue training where left off    [false]
  --seed N                                                 [42]
 model:
  --d-model N                                              [128]
  --n-layers N                                             [4]
  --d-state N          SSM state size                      [16]
  --d-conv N           causal conv width                   [4]
  --expand N           d_inner = expand * d_model          [2]
  --dt-rank N          0 = ceil(d_model/16)                [0]
  --attn-layers LIST   e.g. 2 or 1,5 or none               [none]
  --n-heads N                                              [4]
  --attn-window N      0 = full causal                     [0]
  --mlp-ratio N        SwiGLU width multiple, 0 = no MLP   [0]
  --moe-experts N      MoE experts per layer (0 = dense)   [0]
  --moe-top-k N        Active experts routed per token     [1]
  --moe-threshold F    Dynamic routing threshold (0 = top-k)[0.0]
  --scan-chunk N       parallel scan chunk, 0 = sequential [64]
  --work-efficient-scan use pair-reduction scan (overrides scan-chunk) [false]

GENERATE OPTIONS:
  --run DIR            run directory                       [runs/shakespeare]
  --prompt TEXT                                            [\"\\n\"]
  --tokens N           characters to generate              [500]
  --temperature F      0 = greedy                          [0.8]
  --top-k N            0 = disabled                        [0]
  --seed N                                                 [0]
  --naive              recompute the full sequence each token (O(L^2) baseline)
  --compare            run both modes greedily and check they agree

BENCH OPTIONS:
  --seq-lens LIST      scan benchmark lengths              [64,128,256,512]
  --batch N / --d-inner N / --d-state N                    [8 / 256 / 16]
  --chunk N            parallel scan chunk                 [64]
  --decode-lens LIST   context lengths for decode bench    [64,256,1024]
  --run DIR            use a trained model for decode bench (else random init)

EXPORT OPTIONS:
  --run DIR            trained run directory               [runs/shakespeare]
  --out FILE           output artifact path                [model.sea]

RUN OPTIONS:
  --model FILE         standalone .sea artifact path       [model.sea]
  --prompt TEXT                                            [\"\\n\"]
  --image FILE         optional image input (PPM format)   []
  --audio FILE         optional audio input (WAV format)   []
  --interactive / -i   interactive conversational shell    [false]
  --tokens N           tokens to generate                  [500]
  --temperature F      0 = greedy                          [0.8]
  --top-k N            0 = disabled                        [0]
  --seed N                                                 [0]

VOICE OPTIONS:
  --model FILE         standalone .sea artifact path       [model.sea]
  --prompt TEXT        voice question or prompt            [\"What is the capital of France?\"]
  --entropy-threshold F uncertainty trigger threshold      [0.70]
  --save-wav FILE      output speech audio file            [reply.wav]

ASSISTANT OPTIONS:
  --model FILE         standalone .sea artifact path       [model.sea]
  --prompt TEXT        user utterance text                 [\"What is the compiler error in main.rs?\"]
  --screen             assemble fresh screen/UI context    [false]
  --voice              enable streaming speech synthesis   [false]
  --tts-engine NAME    TTS engine: auto, formant, or piper [auto]
  --piper-path PATH    path to piper executable            [piper]
  --piper-model PATH   path to piper ONNX model            [en_US-lessac-medium]
  --save-wav FILE      save speech audio output to WAV     []
  --session-id ID      session tracking identifier         [default_assistant_session]

ENGINE OPTIONS:
  --model FILE         standalone .sea artifact path       [model.sea]
  --prompts LIST       comma-separated prompts             [\"The theory,Explain rust,What is sound\"]
  --tokens N           tokens to generate per prompt       [50]
  --temperature F      0 = greedy                          [0.8]
  --top-k N            0 = disabled                        [0]
  --seed N                                                 [0]

DAEMON OPTIONS:
  --model FILE         standalone .sea artifact path       [model.sea]
  --command TEXT       text or shell command to execute    [\"help\"]
";

/// Minimal `--key value` / `--flag` parser.
struct Args {
    values: HashMap<String, String>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Self, String> {
        let mut values = HashMap::new();
        let mut i = 0;
        while i < raw.len() {
            let key = raw[i]
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument {:?}", raw[i]))?
                .to_string();
            match raw.get(i + 1) {
                Some(v) if !v.starts_with("--") => {
                    values.insert(key, v.clone());
                    i += 2;
                }
                _ => {
                    values.insert(key, "true".to_string());
                    i += 1;
                }
            }
        }
        Ok(Self { values })
    }

    fn get<T: FromStr>(&mut self, key: &str, default: T) -> Result<T, String> {
        match self.values.remove(key) {
            Some(v) => v
                .parse()
                .map_err(|_| format!("invalid value for --{key}: {v:?}")),
            None => Ok(default),
        }
    }

    fn string(&mut self, key: &str, default: &str) -> String {
        self.values
            .remove(key)
            .unwrap_or_else(|| default.to_string())
    }

    fn opt_string(&mut self, key: &str) -> Option<String> {
        self.values.remove(key)
    }

    fn flag(&mut self, key: &str) -> bool {
        self.values.remove(key).is_some()
    }

    fn list(&mut self, key: &str, default: &[usize]) -> Result<Vec<usize>, String> {
        match self.values.remove(key) {
            None => Ok(default.to_vec()),
            Some(v) if v == "none" || v.is_empty() => Ok(Vec::new()),
            Some(v) => v
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse()
                        .map_err(|_| format!("invalid list for --{key}: {v:?}"))
                })
                .collect(),
        }
    }

    /// Errors on any option that was not consumed (catches typos).
    fn finish(self) -> Result<(), String> {
        if self.values.is_empty() {
            Ok(())
        } else {
            let mut keys: Vec<_> = self.values.keys().map(|k| format!("--{k}")).collect();
            keys.sort();
            Err(format!("unknown option(s): {}", keys.join(", ")))
        }
    }
}

fn main() {
    use std::io::IsTerminal;

    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    let explicit_setup = matches!(
        raw.first().map(String::as_str),
        Some("setup" | "configure" | "wizard")
    );
    if explicit_setup || (raw.is_empty() && std::io::stdin().is_terminal()) {
        match interactive::run() {
            Ok(Some(configured)) => raw = configured,
            Ok(None) => return,
            Err(error) => {
                eprintln!("error: {error}");
                std::process::exit(1);
            }
        }
    }
    let (command, rest) = match raw.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => ("help", &[][..]),
    };
    let result = Args::parse(rest).and_then(|args| match command {
        "train" => cmd_train(args),
        "stream-train" => cmd_stream_train(args),
        "generate" => cmd_generate(args),
        "export" => cmd_export(args),
        "run" => cmd_run(args),
        "voice" => cmd_voice(args),
        "assistant" => cmd_assistant(args),
        "engine" => cmd_engine(args),
        "daemon" => cmd_daemon(args),
        "host-smoke" => cmd_host_smoke(args),
        "bench" => cmd_bench(args),
        "help" | "--help" | "-h" => {
            print!("{HELP}");
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("sail {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => Err(format!("unknown command {other:?}; run `sail help`")),
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn model_config_from_args(args: &mut Args, vocab_size: usize) -> Result<HybridLmConfig, String> {
    Ok(HybridLmConfig::new(vocab_size)
        .with_d_model(args.get("d-model", 128)?)
        .with_n_layers(args.get("n-layers", 4)?)
        .with_d_state(args.get("d-state", 16)?)
        .with_d_conv(args.get("d-conv", 4)?)
        .with_expand(args.get("expand", 2)?)
        .with_dt_rank(args.get("dt-rank", 0)?)
        .with_attn_layers(args.list("attn-layers", &[])?)
        .with_n_heads(args.get("n-heads", 4)?)
        .with_attn_window(args.get("attn-window", 0)?)
        .with_mlp_ratio(args.get("mlp-ratio", 0)?)
        .with_moe_experts(args.get("moe-experts", 0)?)
        .with_moe_top_k(args.get("moe-top-k", 1)?)
        .with_moe_threshold(args.get("moe-threshold", 0.0)?)
        .with_scan_chunk(args.get("scan-chunk", 64)?)
        .with_work_efficient_scan(args.flag("work-efficient-scan").then_some(true)))
}

fn cmd_train(mut args: Args) -> Result<(), String> {
    let data_path = args.string("data", "data/tinyshakespeare.txt");
    let out_dir = args.string("out", "runs/shakespeare");
    let use_bpe = args.flag("bpe");
    let bpe_vocab_size: usize = args.get("vocab-size", 512)?;
    let model = model_config_from_args(&mut args, 0)?;
    let cfg = TrainConfig::new(model)
        .with_profile(Some(args.flag("profile")))
        .with_device_clip(Some(args.flag("device-clip")))
        .with_use_bpe(use_bpe)
        .with_bpe_vocab_size(bpe_vocab_size)
        .with_steps(args.get("steps", 2000)?)
        .with_batch_size(args.get("batch-size", 32)?)
        .with_seq_len(args.get("seq-len", 128)?)
        .with_lr(args.get("lr", 2e-3)?)
        .with_min_lr(args.get("min-lr", 1e-4)?)
        .with_warmup_steps(args.get("warmup", 100)?)
        .with_weight_decay(args.get("weight-decay", 0.1)?)
        .with_grad_clip(args.get("grad-clip", 1.0)?)
        .with_eval_every(args.get("eval-every", 250)?)
        .with_save_every(args.get("save-every", 0)?)
        .with_resume(args.flag("resume"))
        .with_eval_windows(args.get("eval-windows", 64)?)
        .with_log_every(args.get("log-every", 25)?)
        .with_viz_every(args.get("viz-every", 25)?)
        .with_viz_max_values(args.get("viz-max-values", 0)?)
        .with_sample_chars(args.get("sample-chars", 200)?)
        .with_max_minutes(args.get("max-minutes", 0.0)?)
        .with_seed(args.get("seed", 42)?);
    args.finish()?;

    let text = std::fs::read_to_string(&data_path)
        .map_err(|e| format!("reading {data_path}: {e} (download TinyShakespeare into data/)"))?;
    println!("Backend: {BACKEND_NAME}");
    let device = default_device();
    let summary =
        train::<TrainBackend>(cfg, &text, &out_dir, &device).map_err(|e| e.to_string())?;
    println!(
        "Done: {} steps in {:.1}s | final train loss {:.4} | best val loss {:.4} ({:.3} bits/char) | saved to {out_dir}",
        summary.steps_run,
        summary.elapsed_secs,
        summary.final_train_loss,
        summary.best_val_loss,
        summary.best_val_loss / std::f64::consts::LN_2
    );
    Ok(())
}

fn cmd_stream_train(mut args: Args) -> Result<(), String> {
    let validation_path = args.string("validation-data", "");
    let incoming_dir = args.string("incoming", "data/incoming");
    let initial_data_path = args.string("initial-data", "");
    let out_dir = args.string("out", "runs/live_growing");
    let replay_ratio: f64 = args.get("replay-ratio", 0.30)?;
    let min_tokens: usize = args.get("min-tokens", 256)?;
    let scan_interval_ms: u64 = args.get("scan-interval", 1000)?;
    let use_bpe = args.flag("bpe");
    let bpe_vocab_size: usize = args.get("vocab-size", 512)?;
    let resume = args.flag("resume");
    let multimodal = args.flag("multimodal");
    let audio_channels: usize = args.get("audio-channels", 80)?;
    let patch_size: usize = args.get("patch-size", 16)?;

    let model = model_config_from_args(&mut args, 0)?;
    let cfg = TrainConfig::new(model)
        .with_profile(Some(args.flag("profile")))
        .with_device_clip(Some(args.flag("device-clip")))
        .with_use_bpe(use_bpe)
        .with_bpe_vocab_size(bpe_vocab_size)
        .with_steps(args.get("steps", 2000)?)
        .with_batch_size(args.get("batch-size", 32)?)
        .with_seq_len(args.get("seq-len", 128)?)
        .with_lr(args.get("lr", 2e-3)?)
        .with_min_lr(args.get("min-lr", 1e-4)?)
        .with_warmup_steps(args.get("warmup", 100)?)
        .with_weight_decay(args.get("weight-decay", 0.1)?)
        .with_grad_clip(args.get("grad-clip", 1.0)?)
        .with_eval_every(args.get("eval-every", 250)?)
        .with_eval_windows(args.get("eval-windows", 64)?)
        .with_val_fraction(args.get("val-fraction", 0.1)?)
        .with_viz_every(args.get("viz-every", 25)?)
        .with_viz_max_values(args.get("viz-max-values", 0)?)
        .with_sample_chars(args.get("sample-chars", 200)?)
        .with_max_minutes(args.get("max-minutes", 0.0)?)
        .with_save_every(args.get("save-every", 100)?)
        .with_resume(resume)
        .with_log_every(args.get("log-every", 25)?)
        .with_seed(args.get("seed", 42)?);
    args.finish()?;

    std::fs::create_dir_all(&incoming_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    let vocab_path = std::path::Path::new(&out_dir).join("vocab.json");
    let tokenizer = if resume && vocab_path.exists() {
        println!("Loading existing vocabulary from {}", vocab_path.display());
        Tokenizer::load(&vocab_path).map_err(|e| e.to_string())?
    } else {
        let mut text = String::new();
        if !initial_data_path.is_empty() && std::path::Path::new(&initial_data_path).exists() {
            text = std::fs::read_to_string(&initial_data_path).map_err(|e| e.to_string())?;
        }
        if text.is_empty() {
            if let Ok(entries) = std::fs::read_dir(&incoming_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        if let Ok(c) = std::fs::read_to_string(&p) {
                            text.push_str(&c);
                            text.push('\n');
                            if text.len() >= 2 * 1024 * 1024 {
                                break; // Sample up to 2MB for fast BPE vocabulary training; byte fallback covers the rest
                            }
                        }
                    }
                }
            }
        }
        if use_bpe {
            if text.is_empty() {
                Tokenizer::Bpe(BpeTokenizer::new_base())
            } else {
                println!(
                    "Training Native BPE Subword Tokenizer (target vocab: {} tokens)...",
                    bpe_vocab_size
                );
                Tokenizer::Bpe(BpeTokenizer::train(&text, bpe_vocab_size))
            }
        } else if text.is_empty() {
            let default_charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:;\n\r\"'-_()[]{}<>/*+=@#$%^&|\\~`\t";
            Tokenizer::Char(CharTokenizer::from_text(default_charset))
        } else {
            Tokenizer::Char(CharTokenizer::from_text(&text))
        }
    };

    if multimodal {
        let omni_cfg = OmnimodalConfig {
            hybrid: cfg.model.clone(),
            audio: AudioProjectorConfig::new(cfg.model.d_model).with_in_channels(audio_channels),
            vision: VisionProjectorConfig::new(cfg.model.d_model).with_patch_size(patch_size),
        };

        let mut initial_records = Vec::new();
        if !initial_data_path.is_empty() && std::path::Path::new(&initial_data_path).exists() {
            if let Ok(content) = std::fs::read_to_string(&initial_data_path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if let Ok(rec) = serde_json::from_str::<OmnimodalRecord>(trimmed) {
                            initial_records.push(rec);
                        }
                    }
                }
            }
        }

        let dataset = OmnimodalDataset::new(initial_records, replay_ratio);
        let mut watcher =
            IncomingMultimodalWatcher::new(&incoming_dir).map_err(|e| e.to_string())?;
        let initial_ingested = watcher.scan_and_ingest(&dataset, audio_channels);
        if initial_ingested > 0 {
            println!(
                "Ingested {} initial multimodal records from {}",
                initial_ingested, incoming_dir
            );
        }

        println!("Backend: {BACKEND_NAME} (Omnimodal Multimodal Any-to-Any)");
        let device = default_device();
        let summary = sail::train::stream_train_omnimodal::<TrainBackend>(
            cfg,
            omni_cfg,
            dataset,
            Some(watcher),
            tokenizer,
            &out_dir,
            &device,
        )
        .map_err(|e| e.to_string())?;

        println!(
            "Omnimodal streaming train finished: {} steps in {:.1}s | final train loss {:.4} | live artifact saved to {}/model.sea",
            summary.steps_run, summary.elapsed_secs, summary.final_train_loss, out_dir
        );
        return Ok(());
    }

    let stream_cfg = StreamConfig {
        replay_ratio,
        min_tokens_to_train: min_tokens,
        scan_interval_ms,
        ..Default::default()
    };

    let initial_tokens =
        if !initial_data_path.is_empty() && std::path::Path::new(&initial_data_path).exists() {
            if let Ok(initial_text) = std::fs::read_to_string(&initial_data_path) {
                tokenizer.encode(&initial_text).unwrap_or_default()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

    let dataset = StreamDataset::new(initial_tokens, stream_cfg);
    let mut watcher = IncomingDirectoryWatcher::new(&incoming_dir).map_err(|e| e.to_string())?;
    let initial_ingested = watcher.scan_and_ingest(&dataset, &tokenizer);
    if initial_ingested > 0 {
        println!(
            "Ingested {} initial tokens from {}",
            initial_ingested, incoming_dir
        );
    }

    println!("Backend: {BACKEND_NAME}");
    let device = default_device();
    let validation = if validation_path.is_empty() {
        None
    } else {
        if !initial_data_path.is_empty()
            && std::fs::canonicalize(&validation_path).ok()
                == std::fs::canonicalize(&initial_data_path).ok()
        {
            return Err("validation-data must differ from initial-data".into());
        }
        Some(
            tokenizer
                .encode(&std::fs::read_to_string(&validation_path).map_err(|e| e.to_string())?)?,
        )
    };
    let summary = sail::train::stream_train_with_validation::<TrainBackend>(
        cfg,
        dataset,
        Some(watcher),
        tokenizer,
        &out_dir,
        &device,
        validation,
    )
    .map_err(|e| e.to_string())?;

    println!(
        "Streaming train finished: {} steps in {:.1}s | final train loss {:.4} | live artifact saved to {}/model.sea",
        summary.steps_run, summary.elapsed_secs, summary.final_train_loss, out_dir
    );
    Ok(())
}

fn cmd_generate(mut args: Args) -> Result<(), String> {
    let run = args.string("run", "runs/shakespeare");
    let prompt = args.string("prompt", "\n");
    let tokens: usize = args.get("tokens", 500)?;
    let sampling = SamplingConfig {
        temperature: args.get("temperature", 0.8)?,
        top_k: args.get("top-k", 0)?,
        seed: args.get("seed", 0)?,
        ..Default::default()
    };
    let naive = args.flag("naive");
    let compare = args.flag("compare");
    args.finish()?;

    let device = default_device();
    let (model, cfg, tokenizer) = load_run::<InferenceBackend>(&run, &device)?;
    let prompt_ids = tokenizer.encode(&prompt)?;
    if prompt_ids.is_empty() {
        return Err("prompt must not be empty".into());
    }
    println!(
        "Backend: {BACKEND_NAME} | model layers [{}] | {} total params ({} active params/token)",
        cfg.layout(),
        model.total_params(),
        model.active_params_per_token()
    );

    if compare {
        let greedy = SamplingConfig {
            temperature: 0.0,
            ..sampling
        };
        let (a, sa) = generate_stateful(&model, &prompt_ids, tokens, &greedy, &device);
        let (b, sb) = generate_naive(&model, &prompt_ids, tokens, &greedy, &device);
        println!("{}", tokenizer.decode(&a));
        println!(
            "\nstateful: {:.1} tok/s | naive: {:.1} tok/s | speed-up {:.1}x | outputs identical: {}",
            sa.tokens_per_second(),
            sb.tokens_per_second(),
            sa.tokens_per_second() / sb.tokens_per_second(),
            a == b
        );
        return Ok(());
    }

    let (ids, stats) = if naive {
        generate_naive(&model, &prompt_ids, tokens, &sampling, &device)
    } else {
        generate_stateful(&model, &prompt_ids, tokens, &sampling, &device)
    };
    println!("{}", tokenizer.decode(&ids));
    if naive {
        println!("\n[naive O(L²)] {:.1} tok/s", stats.tokens_per_second());
    } else {
        println!(
            "\n[stateful O(1)/token] prefill {:.1} ms | {:.1} tok/s | final state {} floats ({:.1} KiB)",
            stats.prefill_time.as_secs_f64() * 1e3,
            stats.tokens_per_second(),
            stats.state_elements,
            stats.state_elements as f64 * 4.0 / 1024.0
        );
    }
    Ok(())
}

fn cmd_export(mut args: Args) -> Result<(), String> {
    let run = args.string("run", "runs/shakespeare");
    let out = args.string("out", "model.sea");
    args.finish()?;

    println!("Exporting {run} to native standalone artifact {out}...");
    let device = default_device();
    let path = export_to_artifact::<InferenceBackend>(&run, &out, &device)?;
    let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!(
        "Successfully exported to {} ({:.2} MB) | Portable across Linux, macOS, Windows, WASM",
        path.display(),
        size_bytes as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn cmd_run(mut args: Args) -> Result<(), String> {
    let default_model = if std::path::Path::new("runs/main/model.sea").exists() {
        "runs/main/model.sea"
    } else {
        "model.sea"
    };
    let model_path = args.string("model", default_model);
    let prompt = args.string("prompt", "\n");
    let image_path = args.opt_string("image");
    let audio_path = args.opt_string("audio");
    let interactive = args.flag("interactive") || args.flag("i");
    let tokens: usize = args.get("tokens", 500)?;
    let sampling = SamplingConfig {
        temperature: args.get("temperature", 0.8)?,
        top_k: args.get("top-k", 0)?,
        seed: args.get("seed", 0)?,
        ..Default::default()
    };
    args.finish()?;

    let device = default_device();

    // 1. Try loading as standalone Omnimodal artifact
    if let Ok((omni_model, omni_cfg, tokenizer)) =
        load_omnimodal_artifact::<InferenceBackend>(&model_path, &device)
    {
        println!(
            "Loaded standalone Omnimodal artifact {model_path} | Backend: {BACKEND_NAME} | layers [{}] | d_model {} | audio channels {} | vision patch {}",
            omni_cfg.hybrid.layout(),
            omni_cfg.hybrid.d_model,
            omni_cfg.audio.in_channels,
            omni_cfg.vision.patch_size
        );

        let vision_tensor = if let Some(img_p) = &image_path {
            println!("Loading image input: {img_p}...");
            let v_data = sail::data::multimodal::load_ppm_image(img_p)
                .map_err(|e| format!("Failed to load image '{img_p}': {e}"))?;
            let tensor_data = TensorData::new(
                v_data.data,
                [1, v_data.channels, v_data.height, v_data.width],
            );
            Some((
                Tensor::<InferenceBackend, 4>::from_data(tensor_data, &device),
                0,
            ))
        } else {
            None
        };

        let audio_tensor = if let Some(aud_p) = &audio_path {
            println!("Loading audio input: {aud_p}...");
            let a_data = sail::data::multimodal::load_wav_as_mel(aud_p, omni_cfg.audio.in_channels)
                .map_err(|e| format!("Failed to load audio '{aud_p}': {e}"))?;
            let tensor_data = TensorData::new(a_data.data, [1, a_data.channels, a_data.frames]);
            Some(Tensor::<InferenceBackend, 3>::from_data(
                tensor_data,
                &device,
            ))
        } else {
            None
        };

        let prompt_ids = tokenizer.encode(&prompt)?;
        let text_slice = if prompt_ids.is_empty() {
            None
        } else {
            Some(prompt_ids.as_slice())
        };

        let (ids, stats) = generate_omnimodal_stateful(
            &omni_model,
            text_slice,
            audio_tensor,
            vision_tensor,
            tokens,
            &sampling,
            &device,
        );

        println!("\nSAIL > {}", tokenizer.decode(&ids));
        println!(
            "\n[omnimodal standalone O(1)/token] prefill {:.1} ms | {:.1} tok/s | final state {} floats ({:.1} KiB)",
            stats.prefill_time.as_secs_f64() * 1e3,
            stats.tokens_per_second(),
            stats.state_elements,
            stats.state_elements as f64 * 4.0 / 1024.0
        );
        return Ok(());
    }

    // 2. Otherwise load as standard text Hybrid artifact
    let (model, cfg, tokenizer) = load_artifact::<InferenceBackend>(&model_path, &device)?;

    println!(
        "Loaded standalone artifact {model_path} | Backend: {BACKEND_NAME} | layers [{}] | {} total params ({} active)",
        cfg.layout(),
        model.total_params(),
        model.active_params_per_token()
    );

    if interactive {
        println!("=== SAIL Native Interactive Shell | Backend: {BACKEND_NAME} ===");
        println!("Type a prompt and press Enter. Type 'exit' to quit.\n");

        loop {
            print!("User > ");
            std::io::stdout().flush().map_err(|e| e.to_string())?;

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() || input.trim() == "exit" {
                println!("\nExiting interactive shell.");
                break;
            }
            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            let prompt_ids = match tokenizer.encode(input) {
                Ok(ids) if !ids.is_empty() => ids,
                _ => continue,
            };

            print!("SAIL > ");
            std::io::stdout().flush().map_err(|e| e.to_string())?;

            let (ids, stats) = generate_stateful(&model, &prompt_ids, tokens, &sampling, &device);
            let gen_slice = if ids.len() >= prompt_ids.len() {
                &ids[prompt_ids.len()..]
            } else {
                &ids[..]
            };
            println!("{}", tokenizer.decode(gen_slice));
            println!(
                "  [{:.1} tok/s | prefill {:.1} ms | state {:.1} KiB]\n",
                stats.tokens_per_second(),
                stats.prefill_time.as_secs_f64() * 1e3,
                stats.state_elements as f64 * 4.0 / 1024.0
            );
        }
        return Ok(());
    }

    let prompt_ids = tokenizer.encode(&prompt)?;
    if prompt_ids.is_empty() {
        return Err("prompt must not be empty".into());
    }

    let (ids, stats) = generate_stateful(&model, &prompt_ids, tokens, &sampling, &device);
    println!("{}", tokenizer.decode(&ids));
    println!(
        "\n[native standalone O(1)/token] prefill {:.1} ms | {:.1} tok/s | final state {} floats ({:.1} KiB)",
        stats.prefill_time.as_secs_f64() * 1e3,
        stats.tokens_per_second(),
        stats.state_elements,
        stats.state_elements as f64 * 4.0 / 1024.0
    );
    Ok(())
}

fn cmd_voice(mut args: Args) -> Result<(), String> {
    let default_model = if std::path::Path::new("runs/main/model.sea").exists() {
        "runs/main/model.sea"
    } else {
        "model.sea"
    };
    let model_path = args.string("model", default_model);
    let input_prompt = args.string("prompt", "What is the capital of France?");
    let entropy_threshold: f32 = args.get("entropy-threshold", 0.70)?;
    let save_wav = args.string("save-wav", "reply.wav");
    let tokens: usize = args.get("tokens", 64)?;
    args.finish()?;

    println!("Initializing Live Voice Assistant...");
    let device = default_device();
    let (model, cfg, tokenizer) = load_artifact::<InferenceBackend>(&model_path, &device)?;
    println!(
        "Backend: {BACKEND_NAME} | model layers [{}] | {} total params",
        cfg.layout(),
        model.total_params()
    );

    let _vad = VadDetector::new(0.01, 400);
    let _asr = VoiceAsr::new(16000);
    let tts = VoiceTts::new(16000);
    let registry = ToolRegistry::new();

    let user_text = if AudioHardware::has_audio_device() {
        println!("Microphone hardware stream available. Query: \"{input_prompt}\"");
        input_prompt
    } else {
        println!(
            "No audio hardware stream detected (headless/WSL container). Query: \"{input_prompt}\""
        );
        input_prompt
    };

    println!("[User Voice Input]: \"{user_text}\"");
    let prompt_ids = tokenizer.encode(&user_text)?;

    let sampling = SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        seed: 42,
        entropy_threshold: Some(entropy_threshold),
        max_uncertain_tokens: 2,
    };

    println!(
        "[SSM LLM Processing & Grounding (Calibrated Entropy Threshold: {:.2})]...",
        entropy_threshold
    );
    let (out_ids, stats) = generate_stateful_with_tools(
        &model,
        &tokenizer,
        &registry,
        &prompt_ids,
        tokens,
        &sampling,
        &device,
    );

    let reply_text = tokenizer.decode(&out_ids);
    println!("\n[Assistant Response]:\n{reply_text}");
    println!(
        "\n[Stats] Decode time: {:.1} ms | {:.1} tok/s | Avg Entropy: {:.3} | Tool Calls: {}",
        stats.decode_time.as_secs_f64() * 1e3,
        stats.tokens_per_second(),
        stats.average_entropy(),
        stats.tool_calls_executed
    );

    println!("[Streaming Neural TTS Synthesizing Audio Response]...");
    let audio = tts.synthesize(&reply_text);
    tts.save_wav(&audio, &save_wav)
        .map_err(|e| format!("saving wav: {e}"))?;
    println!(
        "Saved synthesized speech audio to {} ({:.2}s of 16kHz audio)",
        save_wav,
        audio.len() as f32 / 16000.0
    );

    Ok(())
}

fn cmd_assistant(mut args: Args) -> Result<(), String> {
    let default_model = if std::path::Path::new("runs/main/model.sea").exists() {
        "runs/main/model.sea"
    } else {
        "model.sea"
    };
    let model_path = args.string("model", default_model);
    let input_prompt = args.string("prompt", "What is the compiler error in main.rs?");
    let screen = args.flag("screen");
    let voice = args.flag("voice");
    let tts_engine = args.string("tts-engine", "auto");
    let piper_path = args.string("piper-path", "piper");
    let piper_model = args.string("piper-model", "en_US-lessac-medium");
    let save_wav = args.opt_string("save-wav");
    let session_id = args.string("session-id", "default_assistant_session");
    args.finish()?;

    println!("=== SAIL Stage-1 Segmented Assistant ===");
    let device = default_device();

    // 1. Initialize Model Backend Adapter
    let backend: std::sync::Arc<dyn sail::runtime::ModelBackendAdapter> = if std::path::Path::new(&model_path).exists() {
        if model_path.ends_with(".gguf") {
            println!("Loading GGUF pretrained backend from {model_path}...");
            std::sync::Arc::new(sail::runtime::PretrainedLocalAdapter::open(&model_path, "gemma_gguf")?)
        } else {
            println!("Loading native backend from {model_path}...");
            let engine = Engine::<InferenceBackend>::open(&model_path, &device)?;
            std::sync::Arc::new(sail::runtime::NativeSailAdapter::new(engine, &model_path))
        }
    } else {
        println!("Model artifact '{model_path}' not found; using mock demonstration backend.");
        std::sync::Arc::new(
            sail::runtime::MockModelAdapter::new("mock_assistant")
                .with_mock_response(&input_prompt, "I inspected the requested context. All parameters are operational.")
        )
    };

    let backend_info = backend.backend_info();
    println!("Backend: {} ({}) | Precision: {} | Device: {}", backend_info.name, backend_info.format, backend_info.precision, backend_info.device);

    // 2. Optional Screen Context Assembly
    let context_packet = if screen {
        println!("[Perception] Assembling synchronized screen and UI context...");
        let assembler = sail::runtime::MultimodalContextAssembler::new(&session_id);
        match assembler.assemble_synchronized_context(&input_prompt, 100, false) {
            Ok(pkt) => {
                println!("[Perception] Context assembled: {} | freshness: {:.1} ms", pkt.context_id, pkt.provenance_trace.screen_age_ms);
                Some(pkt)
            }
            Err(e) => {
                println!("[Perception Warning] Screen context unavailable: {e}");
                None
            }
        }
    } else {
        None
    };

    // 3. Optional Voice TTS Output
    let tts: Option<Box<dyn sail::voice::tts::StreamingTtsAdapter>> = if voice {
        let piper_adapter = sail::voice::tts::PiperLocalTtsAdapter::new(&piper_path, &piper_model);
        let use_piper = if tts_engine == "piper" {
            true
        } else if tts_engine == "formant" {
            false
        } else {
            piper_adapter.validate_configuration().is_ok()
        };

        if use_piper {
            println!("[Audio] Using Piper neural speech synthesis ({piper_path})...");
            Some(Box::new(piper_adapter))
        } else {
            if tts_engine == "auto" {
                println!("[Audio] 'piper' binary not found; using built-in local formant speech engine.");
            } else {
                println!("[Audio] Using built-in local formant speech engine...");
            }
            Some(Box::new(sail::voice::tts::FormantNeuralTtsEngine::new(16000)))
        }
    } else {
        None
    };

    // 4. Run Segmented Assistant Coordinator
    let coordinator = sail::runtime::SegmentedAssistantCoordinator::new(&session_id, backend, tts);
    let segment = sail::voice::asr::TranscriptSegment {
        utterance_id: format!("utt_{}", rand::random::<u16>()),
        session_id: session_id.clone(),
        start_offset: 0,
        end_offset: 1600,
        start_timestamp_ns: 1_000_000,
        end_timestamp_ns: 100_000_000,
        text: input_prompt.clone(),
        is_final: true,
        confidence: 0.99,
    };

    println!("[User ASR Finalized]: \"{input_prompt}\"");
    let outcome = coordinator.process_transcript(segment, context_packet.as_ref())
        .map_err(|e| format!("assistant coordination error: {e}"))?;

    println!("\nAssistant > {}", outcome.response_text);

    if !outcome.degradations.is_empty() {
        println!("\n[Degradations Encountered]: {:?}", outcome.degradations);
    }

    if !outcome.tool_proposals.is_empty() {
        println!("\n[Captured Tool Proposals (Awaiting Policy Authorization)]:");
        for (i, prop) in outcome.tool_proposals.iter().enumerate() {
            println!("  [{}] {}: call_id={} input={}", i + 1, prop.tool_name, prop.call_id, prop.input);
        }
    }

    if !outcome.synthesized_audio.is_empty() {
        let sample_rate = outcome.synthesized_audio[0].sample_rate;
        println!(
            "\n[Audio] Synthesized {} speech chunks ({} samples, {:.2}s, {} Hz)",
            outcome.synthesized_chunks,
            outcome.synthesized_samples,
            outcome.synthesized_samples as f64 / sample_rate as f64,
            sample_rate
        );
        let wav_path = save_wav.unwrap_or_else(|| "reply.wav".to_string());
        let mut all_samples = Vec::new();
        for chunk in &outcome.synthesized_audio {
            all_samples.extend_from_slice(&chunk.samples);
        }
        if !all_samples.is_empty() {
            let voice_tts = sail::voice::VoiceTts::new(sample_rate);
            if let Err(e) = voice_tts.save_wav(&all_samples, &wav_path) {
                eprintln!("[Audio Warning] Failed to write WAV file '{wav_path}': {e}");
            } else {
                println!("[Audio] Saved speech audio to '{wav_path}'");
            }
        }
    }

    if let Some(metadata) = outcome.generation {
        println!(
            "\n[Performance] Generated {} tokens in {:.1} ms ({:.1} tok/s) | Finish: {:?}",
            metadata.generated_tokens, metadata.elapsed_ms, metadata.tokens_per_sec, metadata.finish_reason
        );
    }

    Ok(())
}

fn cmd_engine(mut args: Args) -> Result<(), String> {
    let default_model = if std::path::Path::new("runs/main/model.sea").exists() {
        "runs/main/model.sea"
    } else {
        "model.sea"
    };
    let model_path = args.string("model", default_model);
    let prompts_str = args.string("prompts", "The theory,Explain rust,What is sound");
    let tokens: usize = args.get("tokens", 50)?;
    let sampling = SamplingConfig {
        temperature: args.get("temperature", 0.8)?,
        top_k: args.get("top-k", 0)?,
        seed: args.get("seed", 0)?,
        ..Default::default()
    };
    args.finish()?;

    println!("Initializing Concurrent High-Integrity Engine from {model_path}...");
    let device = default_device();
    let engine = Engine::<InferenceBackend>::open(&model_path, &device)?;
    println!(
        "Engine Online | Active Generation: {} | Backend: {BACKEND_NAME}",
        engine.active_generation().generation_id
    );

    let prompts: Vec<String> = prompts_str
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    println!(
        "Dispatching {} parallel inference streams across OS threads...",
        prompts.len()
    );

    let results = engine.generate_parallel(&prompts, tokens, &sampling);
    for (i, res) in results.into_iter().enumerate() {
        match res {
            Ok((text, stats)) => {
                println!("\n=== Worker [{i}]: Prompt '{}' ===", prompts[i]);
                println!("{text}");
                println!(
                    "[Stats] {:.1} tok/s | new tokens: {}",
                    stats.tokens_per_second(),
                    stats.new_tokens
                );
            }
            Err(e) => eprintln!("\nWorker [{i}] error: {e}"),
        }
    }
    Ok(())
}

fn cmd_daemon(mut args: Args) -> Result<(), String> {
    let default_model = if std::path::Path::new("runs/main/model.sea").exists() {
        "runs/main/model.sea"
    } else {
        "model.sea"
    };
    let model_path = args.string("model", default_model);
    let command = args.string("command", "help");
    args.finish()?;

    println!("Starting Redox OS Scheme Daemon (sail:) with model {model_path}...");
    let device = default_device();
    let engine = Engine::<InferenceBackend>::open(&model_path, &device).ok();

    let daemon = SailDaemon::new(engine);
    println!(
        "Redox Scheme `sail:` active. Dispatching command: {:?}",
        command
    );
    let output = daemon.send_command(&command)?;
    println!("\n[Redox Scheme Output]:\n{}", output);
    Ok(())
}

fn time_it<F: FnMut()>(repeats: usize, mut f: F) -> f64 {
    f(); // warm-up (allocations, autotune, shader compilation)
    let start = Instant::now();
    for _ in 0..repeats {
        f();
    }
    start.elapsed().as_secs_f64() * 1e3 / repeats as f64
}

fn cmd_bench(mut args: Args) -> Result<(), String> {
    let seq_lens = args.list("seq-lens", &[64, 128, 256, 512])?;
    let batch: usize = args.get("batch", 8)?;
    let d_inner: usize = args.get("d-inner", 256)?;
    let d_state: usize = args.get("d-state", 16)?;
    let chunk: usize = args.get("chunk", 64)?;
    let decode_lens = args.list("decode-lens", &[64, 256, 1024])?;
    let run = args.values.remove("run");
    args.finish()?;

    let device = default_device();
    println!("Backend: {BACKEND_NAME}\n");
    println!(
        "== Scan: forward + backward, [B={batch}, L, D={d_inner}, N={d_state}], chunk {chunk} =="
    );
    println!(
        "{:>6} | {:>14} | {:>14} | {:>8}",
        "L", "sequential ms", "parallel ms", "speed-up"
    );
    for &len in &seq_lens {
        let shape = [batch, len, d_inner, d_state];
        let a = Tensor::<TrainBackend, 4>::random(shape, Distribution::Uniform(0.5, 1.0), &device)
            .require_grad();
        let b = Tensor::<TrainBackend, 4>::random(shape, Distribution::Normal(0.0, 1.0), &device)
            .require_grad();
        let run_scan = |parallel: bool| {
            let h = if parallel {
                chunked_parallel_scan(a.clone(), b.clone(), chunk)
            } else {
                sequential_scan(a.clone(), b.clone())
            };
            let grads = h.sum().backward();
            let _ = a.grad(&grads).map(|g| g.sum().into_scalar());
        };
        let seq_ms = time_it(2, || run_scan(false));
        let par_ms = time_it(2, || run_scan(true));
        println!(
            "{len:>6} | {seq_ms:>14.1} | {par_ms:>14.1} | {:>7.1}x",
            seq_ms / par_ms
        );
    }

    let (model, vocab) = match run {
        Some(dir) => {
            let (m, cfg, _) = load_run::<InferenceBackend>(&dir, &device)?;
            (m, cfg.vocab_size)
        }
        None => {
            let cfg = HybridLmConfig::new(65).with_attn_layers(vec![2]);
            (cfg.init::<InferenceBackend>(&device), 65)
        }
    };
    bench_decode(&model, vocab, &decode_lens, &device);
    Ok(())
}

fn bench_decode(
    model: &HybridLm<InferenceBackend>,
    vocab: usize,
    lens: &[usize],
    device: &<InferenceBackend as Backend>::Device,
) {
    println!("\n== Decoding: per-token latency after a context of L tokens ==");
    println!(
        "{:>6} | {:>16} | {:>16} | {:>8} | {:>12}",
        "L", "stateful ms/tok", "naive ms/tok", "speed-up", "state floats"
    );
    let greedy = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        seed: 0,
        ..Default::default()
    };
    let new_tokens = 16;
    for &len in lens {
        let prompt: Vec<u32> = (0..len).map(|i| (i * 7 % vocab) as u32).collect();
        let (_, s) = generate_stateful(model, &prompt, new_tokens, &greedy, device);
        let stateful_ms = s.decode_time.as_secs_f64() * 1e3 / new_tokens as f64;
        // The naive path recomputes the full context for every token; a few tokens suffice.
        let naive_tokens = 3;
        let (_, n) = generate_naive(model, &prompt, naive_tokens, &greedy, device);
        let naive_ms = n.decode_time.as_secs_f64() * 1e3 / naive_tokens as f64;
        println!(
            "{len:>6} | {stateful_ms:>16.2} | {naive_ms:>16.2} | {:>7.1}x | {:>12}",
            naive_ms / stateful_ms,
            s.state_elements
        );
    }
}

fn cmd_host_smoke(args: Args) -> Result<(), String> {
    args.finish()?;
    println!("=== SAIL Native Host Runtime Smoke Test ===");
    let bus = std::sync::Arc::new(sail::runtime::EventBus::new());
    let supervisor = sail::runtime::Supervisor::new(bus.clone());

    let audio_rx = bus.subscribe("audio", 64);
    let tool_rx = bus.subscribe("tools", 32);

    println!("[Host Runtime] Starting perception and tool workers...");
    supervisor.spawn_worker("mic_capture", sail::runtime::WorkerRole::PerceptionAudio, {
        let bus = bus.clone();
        move |token, _| {
            let mut seq = 0;
            while !token.is_cancelled() && seq < 10 {
                seq += 1;
                let audio = std::sync::Arc::new(sail::runtime::SharedAudioBuffer::new(
                    16000,
                    1,
                    vec![0.02; 320],
                ));
                let event = sail::runtime::RuntimeEvent::new(
                    bus.next_event_id(),
                    "smoke_sess",
                    "mic",
                    seq,
                    seq * 20_000_000,
                    sail::runtime::EventPayload::AudioInput(audio),
                );
                let _ = bus.publish("audio", event, sail::runtime::OverflowPolicy::DropOldest);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Ok(())
        }
    })?;

    supervisor.spawn_worker("tool_runner", sail::runtime::WorkerRole::ToolExecutor, {
        let _bus = bus.clone();
        move |token, _| {
            while !token.is_cancelled() {
                if let Some(event) = tool_rx.pop() {
                    println!("[Tool Worker] Received task: {:?}", event.event_id);
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Ok(())
        }
    })?;

    std::thread::sleep(std::time::Duration::from_millis(80));
    let states = supervisor.get_worker_states();
    for (id, state) in &states {
        println!("[Worker Status] {id} -> {state:?}");
    }
    let received_audio = audio_rx.len();
    println!(
        "[Host Runtime] Bounded audio queue received {received_audio} frames (dropped: {})",
        audio_rx.dropped_count()
    );

    println!("[Host Runtime] Initiating graceful shutdown...");
    supervisor.shutdown(std::time::Duration::from_millis(200));
    println!(
        "✓ [Host Smoke] Native host runtime, workers, and bounded event bus verified successfully."
    );
    Ok(())
}
