//! Autoregressive sampling, calibrated entropy metrics, and interactive tool execution.
//!
//! - [`generate_stateful`]: prefill prompt once with the parallel scan, then decode one token
//!   at a time with [`HybridLm::step`], carrying recurrent state with O(1) constant memory.
//! - [`generate_stateful_with_tools`]: pauses recurrent state on tool call tokens, executes
//!   external tool queries (e.g. Web Search), and feeds verified responses back into the state.
//! - [`compute_normalized_entropy`]: measures per-token prediction uncertainty in [0.0, 1.0],
//!   enabling automatic verification when model confidence drops to avoid confident lies.

use std::time::{Duration, Instant};

use burn::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::data::Tokenizer;
use crate::model::HybridLm;
use crate::tool::ToolRegistry;

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// 0 = greedy argmax.
    pub temperature: f32,
    /// 0 = no top-k filtering.
    pub top_k: usize,
    pub seed: u64,
    /// Optional normalized entropy threshold (e.g. 0.70). If consecutive tokens
    /// exceed this uncertainty, an automatic verification check is triggered.
    pub entropy_threshold: Option<f32>,
    /// Number of consecutive high-entropy tokens required to trigger verification.
    pub max_uncertain_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 0,
            seed: 0,
            entropy_threshold: None,
            max_uncertain_tokens: 3,
        }
    }
}

/// Timing, memory, and entropy statistics of a generation run.
#[derive(Clone, Debug, Default)]
pub struct GenerationStats {
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub new_tokens: usize,
    /// Number of f32 values in the recurrent state at the end (stateful mode only).
    pub state_elements: usize,
    /// Normalized Shannon entropy per generated token in [0.0, 1.0].
    pub token_entropies: Vec<f32>,
    /// Number of tool calls executed during generation.
    pub tool_calls_executed: usize,
}

impl GenerationStats {
    pub fn tokens_per_second(&self) -> f64 {
        self.new_tokens as f64 / self.decode_time.as_secs_f64().max(1e-9)
    }

    /// Average normalized predictive entropy across all generated tokens.
    pub fn average_entropy(&self) -> f32 {
        if self.token_entropies.is_empty() {
            0.0
        } else {
            self.token_entropies.iter().sum::<f32>() / self.token_entropies.len() as f32
        }
    }

    /// Maximum predictive entropy observed across any generated token.
    pub fn max_entropy(&self) -> f32 {
        self.token_entropies.iter().copied().fold(0.0f32, f32::max)
    }
}

/// Computes normalized Shannon entropy in [0.0, 1.0] from a slice of unnormalized logits.
pub fn compute_normalized_entropy(logits: &[f32]) -> f32 {
    if logits.len() <= 1 {
        return 0.0;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - max_val).exp();
    }
    if sum_exp <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0f32;
    for &l in logits {
        let p = (l - max_val).exp() / sum_exp;
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }

    let max_entropy = (logits.len() as f32).ln().max(1e-6);
    (entropy / max_entropy).clamp(0.0, 1.0)
}

/// Samples a token id from a single row of logits.
pub fn sample_logits(logits: &[f32], cfg: &SamplingConfig, rng: &mut impl Rng) -> u32 {
    if cfg.temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .map(|&l| l / cfg.temperature)
        .enumerate()
        .collect();
    if cfg.top_k > 0 && cfg.top_k < scaled.len() {
        scaled.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scaled.truncate(cfg.top_k);
    }
    let max = scaled
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scaled.iter().map(|&(_, l)| (l - max).exp()).collect();
    let total: f32 = weights.iter().sum();
    let mut r = rng.r#gen::<f32>() * total;
    for (&(i, _), w) in scaled.iter().zip(&weights) {
        r -= w;
        if r <= 0.0 {
            return i as u32;
        }
    }
    scaled.last().map(|&(i, _)| i as u32).unwrap_or(0)
}

fn row_to_vec<B: Backend>(logits: Tensor<B, 1>) -> Vec<f32> {
    logits
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("logits to vec")
}

fn tokens_tensor<B: Backend>(ids: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let data: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
    Tensor::from_data(TensorData::new(data, [1, ids.len()]), device)
}

/// O(1)-per-token generation using the recurrent state with entropy logging.
pub fn generate_stateful<B: Backend>(
    model: &HybridLm<B>,
    prompt: &[u32],
    new_tokens: usize,
    cfg: &SamplingConfig,
    device: &B::Device,
) -> (Vec<u32>, GenerationStats) {
    assert!(!prompt.is_empty(), "prompt must contain at least one token");
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut stats = GenerationStats::default();
    let mut out = prompt.to_vec();

    let t0 = Instant::now();
    let (logits, mut state) = model.prefill(tokens_tensor(prompt, device));
    let [_, len, vocab] = logits.dims();
    let mut last: Tensor<B, 1> = logits.narrow(1, len - 1, 1).reshape([vocab]);
    let first_row = row_to_vec(last.clone());
    stats.prefill_time = t0.elapsed();

    let t1 = Instant::now();
    let mut current_row = first_row;

    for i in 0..new_tokens {
        let ent = compute_normalized_entropy(&current_row);
        stats.token_entropies.push(ent);

        let next = sample_logits(&current_row, cfg, &mut rng);
        out.push(next);
        if i + 1 == new_tokens {
            break;
        }
        let token = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![next as i64], [1]), device);
        let (logits, new_state) = model.step(token, state);
        state = new_state;
        last = logits.reshape([vocab]);
        current_row = row_to_vec(last.clone());
    }
    stats.decode_time = t1.elapsed();
    stats.new_tokens = new_tokens;
    stats.state_elements = state.num_elements();
    (out, stats)
}

/// Stateful generation with native tool calling and calibrated entropy confidence checking.
///
/// Pauses generation when `<tool_call>` is emitted, invokes external search tools,
/// and feeds verified evidence back into recurrent state to prevent hallucinations.
pub fn generate_stateful_with_tools<B: Backend>(
    model: &HybridLm<B>,
    tokenizer: &Tokenizer,
    tool_registry: &ToolRegistry,
    prompt: &[u32],
    new_tokens: usize,
    cfg: &SamplingConfig,
    device: &B::Device,
) -> (Vec<u32>, GenerationStats) {
    assert!(!prompt.is_empty(), "prompt must contain at least one token");
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut stats = GenerationStats::default();
    let mut out = prompt.to_vec();

    let tool_tokens = tokenizer.tool_tokens();

    let t0 = Instant::now();
    let (logits, mut state) = model.prefill(tokens_tensor(prompt, device));
    let [_, len, vocab] = logits.dims();
    let mut last: Tensor<B, 1> = logits.narrow(1, len - 1, 1).reshape([vocab]);
    let mut current_row = row_to_vec(last.clone());
    stats.prefill_time = t0.elapsed();

    // Check if the prompt already contains a pending/unresolved tool call
    if let Some((tc_start, tc_end, _tr_start, _tr_end)) = tool_tokens {
        if let Some(start_pos) = prompt.iter().rposition(|&t| t == tc_start) {
            if let Some(end_pos) = prompt[start_pos..].iter().position(|&t| t == tc_end) {
                let actual_end = start_pos + end_pos;
                if actual_end == prompt.len() - 1 {
                    let call_payload = &prompt[start_pos + 1..actual_end];
                    let call_text = tokenizer.decode(call_payload);
                    let tool_result = tool_registry.dispatch(&call_text);
                    stats.tool_calls_executed += 1;

                    let resp_text = format!("<tool_response>{}</tool_response>", tool_result);
                    let resp_tokens = tokenizer.encode(&resp_text).unwrap_or_default();

                    for &rt in &resp_tokens {
                        out.push(rt);
                        let token = Tensor::<B, 1, Int>::from_data(
                            TensorData::new(vec![rt as i64], [1]),
                            device,
                        );
                        let (logits, new_state) = model.step(token, state);
                        state = new_state;
                        last = logits.reshape([vocab]);
                    }
                    current_row = row_to_vec(last.clone());
                }
            }
        }
    }

    let t1 = Instant::now();
    let mut in_tool_call = false;
    let mut tool_call_buffer = Vec::new();
    let mut consecutive_high_entropy = 0usize;
    let mut step_idx = 0;

    while step_idx < new_tokens {
        let ent = compute_normalized_entropy(&current_row);
        stats.token_entropies.push(ent);

        // Check calibrated entropy trigger (anti-hallucination)
        if let Some(thresh) = cfg.entropy_threshold {
            if ent >= thresh {
                consecutive_high_entropy += 1;
                if consecutive_high_entropy >= cfg.max_uncertain_tokens {
                    // Model has low confidence: trigger automatic web search verification
                    let recent_tokens = &out[out.len().saturating_sub(12)..];
                    let recent_text = tokenizer.decode(recent_tokens);
                    let query = recent_text.trim();

                    let tool_result = tool_registry.dispatch(query);
                    stats.tool_calls_executed += 1;

                    let verified_text =
                        format!("\n<tool_response>{}</tool_response>\n", tool_result);
                    let verified_tokens = tokenizer.encode(&verified_text).unwrap_or_default();

                    for &vt in &verified_tokens {
                        out.push(vt);
                        let token = Tensor::<B, 1, Int>::from_data(
                            TensorData::new(vec![vt as i64], [1]),
                            device,
                        );
                        let (logits, new_state) = model.step(token, state);
                        state = new_state;
                        last = logits.reshape([vocab]);
                    }
                    current_row = row_to_vec(last.clone());
                    consecutive_high_entropy = 0;
                    step_idx += 1;
                    continue;
                }
            } else {
                consecutive_high_entropy = 0;
            }
        }

        let next = sample_logits(&current_row, cfg, &mut rng);
        out.push(next);
        step_idx += 1;

        // Check if tool call marker tokens were emitted
        if let Some((tc_start, tc_end, _tr_start, _tr_end)) = tool_tokens {
            if next == tc_start {
                in_tool_call = true;
                tool_call_buffer.clear();
            } else if next == tc_end && in_tool_call {
                in_tool_call = false;
                let call_text = tokenizer.decode(&tool_call_buffer);
                let tool_result = tool_registry.dispatch(&call_text);
                stats.tool_calls_executed += 1;

                let resp_text = format!("<tool_response>{}</tool_response>", tool_result);
                let resp_tokens = tokenizer.encode(&resp_text).unwrap_or_default();

                for &rt in &resp_tokens {
                    out.push(rt);
                    let token = Tensor::<B, 1, Int>::from_data(
                        TensorData::new(vec![rt as i64], [1]),
                        device,
                    );
                    let (logits, new_state) = model.step(token, state);
                    state = new_state;
                    last = logits.reshape([vocab]);
                }
                current_row = row_to_vec(last.clone());
                tool_call_buffer.clear();
                continue;
            } else if in_tool_call {
                tool_call_buffer.push(next);
            }
        }

        if step_idx >= new_tokens {
            break;
        }

        let token = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![next as i64], [1]), device);
        let (logits, new_state) = model.step(token, state);
        state = new_state;
        last = logits.reshape([vocab]);
        current_row = row_to_vec(last.clone());
    }

    stats.decode_time = t1.elapsed();
    stats.new_tokens = step_idx;
    stats.state_elements = state.num_elements();
    (out, stats)
}

/// Baseline: recompute the full forward pass over the whole sequence for every new token.
pub fn generate_naive<B: Backend>(
    model: &HybridLm<B>,
    prompt: &[u32],
    new_tokens: usize,
    cfg: &SamplingConfig,
    device: &B::Device,
) -> (Vec<u32>, GenerationStats) {
    assert!(!prompt.is_empty(), "prompt must contain at least one token");
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut stats = GenerationStats::default();
    let mut out = prompt.to_vec();

    let t1 = Instant::now();
    for _ in 0..new_tokens {
        let logits = model.forward(tokens_tensor(&out, device));
        let [_, len, vocab] = logits.dims();
        let last: Tensor<B, 1> = logits.narrow(1, len - 1, 1).reshape([vocab]);
        let row = row_to_vec(last);
        stats.token_entropies.push(compute_normalized_entropy(&row));
        out.push(sample_logits(&row, cfg, &mut rng));
    }
    stats.decode_time = t1.elapsed();
    stats.new_tokens = new_tokens;
    (out, stats)
}
