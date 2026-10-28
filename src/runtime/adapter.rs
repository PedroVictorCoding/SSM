//! Interchangeable Model Inference Backend Adapters for SAIL.
//!
//! Provides a session-oriented inference interface enabling the assistant coordinator
//! to switch seamlessly between native SAIL models and local pretrained models (e.g. GGUF):
//! - Incremental streaming token emission.
//! - Typed tool proposals.
//! - Cooperative per-step cancellation.
//! - Session state isolation and reset.
//! - Strict format validation rejecting unsupported or corrupted formats.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use burn::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::generate::{SamplingConfig, row_to_vec, sample_logits, tokens_tensor};
use crate::model::HybridState;
use crate::runtime::bus::CancellationToken;
use crate::runtime::event::{SharedAudioBuffer, SharedVideoBuffer};

/// Typed tool call proposed by the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolProposal {
    pub call_id: String,
    pub tool_name: String,
    pub input: String,
}

/// Reason why token generation completed or terminated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    Cancelled,
    Error,
}

/// Detailed execution metadata for a generation turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub backend_name: String,
    pub model_id: String,
    pub generation_id: u64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_count: usize,
    pub elapsed_ms: f64,
    pub tokens_per_sec: f64,
    pub finish_reason: FinishReason,
}

/// Input request for an inference turn.
#[derive(Clone, Debug)]
pub struct GenerateRequest {
    pub session_id: String,
    pub prompt_text: String,
    pub audio_input: Option<Arc<SharedAudioBuffer>>,
    pub video_input: Option<Arc<SharedVideoBuffer>>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub cancellation_token: CancellationToken,
    pub seed: Option<u64>,
    pub stop_on_eos: bool,
}

impl GenerateRequest {
    pub fn new(session_id: impl Into<String>, prompt_text: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            prompt_text: prompt_text.into(),
            audio_input: None,
            video_input: None,
            max_new_tokens: 128,
            temperature: 0.0,
            top_k: 0,
            cancellation_token: CancellationToken::new(),
            seed: None,
            stop_on_eos: false,
        }
    }

    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_new_tokens = max;
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_stop_on_eos(mut self, stop: bool) -> Self {
        self.stop_on_eos = stop;
        self
    }

    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = token;
        self
    }
}

/// Buffers incomplete UTF-8 byte sequences across chunk emissions, preventing corruption
/// at byte-level BPE boundaries.
#[derive(Default, Debug)]
pub struct StreamingUtf8Decoder {
    buffer: Vec<u8>,
}

impl StreamingUtf8Decoder {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Appends raw bytes and extracts any newly completed valid UTF-8 text prefix.
    /// Trailing incomplete UTF-8 bytes are retained in the internal buffer until more bytes arrive.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.buffer.extend_from_slice(bytes);

        match std::str::from_utf8(&self.buffer) {
            Ok(_) => String::from_utf8(std::mem::take(&mut self.buffer)).unwrap(),
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if let Some(err_len) = e.error_len() {
                    // Invalid sequence encountered; emit valid prefix plus replacement for invalid chunk
                    let emit_len = valid_up_to + err_len;
                    let valid_chunk = String::from_utf8_lossy(&self.buffer[..emit_len]).to_string();
                    self.buffer.drain(..emit_len);
                    valid_chunk
                } else if valid_up_to > 0 {
                    // Valid prefix followed by incomplete sequence at tail
                    let valid_chunk = std::str::from_utf8(&self.buffer[..valid_up_to])
                        .unwrap()
                        .to_string();
                    self.buffer.drain(..valid_up_to);
                    valid_chunk
                } else {
                    // Incomplete sequence waiting for more bytes
                    String::new()
                }
            }
        }
    }

    /// Flushes any remaining bytes at stream end (lossy if still incomplete).
    pub fn finish(&mut self) -> String {
        if self.buffer.is_empty() {
            String::new()
        } else {
            let s = String::from_utf8_lossy(&self.buffer).to_string();
            self.buffer.clear();
            s
        }
    }
}

/// Incremental chunk emitted by an inference backend.
#[derive(Clone, Debug, PartialEq)]
pub enum GenerationChunk {
    Token { text: String, token_id: u32 },
    ToolCall(ToolProposal),
    Complete(GenerationMetadata),
}

/// A modality that an inference backend can consume or emit semantically.
///
/// Declaring a modality means the backend actually uses it during inference. Merely
/// accepting a field in [`GenerateRequest`] is not support.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendModality {
    Text,
    Audio,
    Video,
    ToolProposal,
}

/// Inspectable input/output contract for an inference backend.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub input_modalities: Vec<BackendModality>,
    pub output_modalities: Vec<BackendModality>,
    pub streaming_output: bool,
    pub stateful_sessions: bool,
}

impl BackendCapabilities {
    pub fn text_backend(stateful_sessions: bool) -> Self {
        Self {
            input_modalities: vec![BackendModality::Text],
            output_modalities: vec![BackendModality::Text, BackendModality::ToolProposal],
            streaming_output: true,
            stateful_sessions,
        }
    }

    pub fn supports_input(&self, modality: BackendModality) -> bool {
        self.input_modalities.contains(&modality)
    }

    pub fn supports_output(&self, modality: BackendModality) -> bool {
        self.output_modalities.contains(&modality)
    }
}

/// Typed request-contract failure returned before inference or runner execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendRequestError {
    UnsupportedInputModality {
        backend_name: String,
        modality: BackendModality,
    },
}

impl std::fmt::Display for BackendRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedInputModality {
                backend_name,
                modality,
            } => write!(
                f,
                "UnsupportedInputModality: backend '{backend_name}' does not consume {modality:?} input"
            ),
        }
    }
}

impl std::error::Error for BackendRequestError {}

/// Metadata describing an inference backend.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendInfo {
    pub name: String,
    pub model_id: String,
    pub format: String,
    pub context_limit: usize,
    pub precision: String,
    pub device: String,
    pub capabilities: BackendCapabilities,
}

/// Validates that every supplied media input will be consumed by `backend`.
///
/// This check is intentionally performed before session mutation, generation
/// counters, model work, or subprocess launch. Unsupported evidence must never be
/// silently discarded.
pub fn validate_request_capabilities(
    backend: &BackendInfo,
    request: &GenerateRequest,
) -> Result<(), BackendRequestError> {
    if request.audio_input.is_some() && !backend.capabilities.supports_input(BackendModality::Audio)
    {
        return Err(BackendRequestError::UnsupportedInputModality {
            backend_name: backend.name.clone(),
            modality: BackendModality::Audio,
        });
    }
    if request.video_input.is_some() && !backend.capabilities.supports_input(BackendModality::Video)
    {
        return Err(BackendRequestError::UnsupportedInputModality {
            backend_name: backend.name.clone(),
            modality: BackendModality::Video,
        });
    }
    Ok(())
}

/// Unified, session-oriented interface for interchangeable model inference.
pub trait ModelBackendAdapter: Send + Sync {
    fn backend_info(&self) -> BackendInfo;
    fn validate_request(&self, request: &GenerateRequest) -> Result<(), BackendRequestError> {
        validate_request_capabilities(&self.backend_info(), request)
    }
    fn reset_session(&self, session_id: &str) -> Result<(), String>;
    fn generate_stream(
        &self,
        request: GenerateRequest,
        chunk_sender: &mut dyn FnMut(GenerationChunk) -> Result<(), String>,
    ) -> Result<GenerationMetadata, String>;
}

// ---------------------------------------------------------------------------
// Native SAIL Adapter
// ---------------------------------------------------------------------------

/// Conversational session state preserving recurrent state, token history, and sampling RNG across turns.
pub struct SessionState<B: Backend> {
    pub history_tokens: Vec<u32>,
    pub recurrent_state: Option<HybridState<B>>,
    pub last_logits_row: Option<Vec<f32>>,
    pub rng: StdRng,
    pub generation_id: u64,
}

unsafe impl<B: Backend> Send for SessionState<B> {}
unsafe impl<B: Backend> Sync for SessionState<B> {}

/// Adapter wrapping the native SAIL `Engine` for hybrid SSM inference.
pub struct NativeSailAdapter<B: Backend> {
    engine: Mutex<Engine<B>>,
    model_name: String,
    sessions: Arc<RwLock<HashMap<String, SessionState<B>>>>,
    generation_counter: Arc<Mutex<u64>>,
}

impl<B: Backend> NativeSailAdapter<B> {
    pub fn new(engine: Engine<B>, model_name: impl Into<String>) -> Self {
        Self {
            engine: Mutex::new(engine),
            model_name: model_name.into(),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            generation_counter: Arc::new(Mutex::new(1)),
        }
    }

    /// Returns session history token count if session exists.
    pub fn session_history_len(&self, session_id: &str) -> Option<usize> {
        let sessions = self.sessions.read().unwrap();
        sessions.get(session_id).map(|s| s.history_tokens.len())
    }
}

impl<B: Backend> ModelBackendAdapter for NativeSailAdapter<B> {
    fn backend_info(&self) -> BackendInfo {
        let engine = self.engine.lock().unwrap();
        let active = engine.active_generation();
        BackendInfo {
            name: "Sail Native Engine".into(),
            model_id: self.model_name.clone(),
            format: "SAIL (.sea)".into(),
            context_limit: 2048,
            precision: "f32".into(),
            device: format!("{:?}", active.device),
            capabilities: BackendCapabilities::text_backend(true),
        }
    }

    fn reset_session(&self, session_id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(session_id);
        Ok(())
    }

    fn generate_stream(
        &self,
        request: GenerateRequest,
        chunk_sender: &mut dyn FnMut(GenerationChunk) -> Result<(), String>,
    ) -> Result<GenerationMetadata, String> {
        self.validate_request(&request).map_err(|e| e.to_string())?;
        let start = Instant::now();
        let gen_id = {
            let mut c = self.generation_counter.lock().unwrap();
            let cur = *c;
            *c += 1;
            cur
        };

        // Pin active model generation for snapshot isolation during this turn
        let active = {
            let engine = self.engine.lock().unwrap();
            engine.active_generation()
        };

        let prompt_ids = active
            .tokenizer
            .encode(&request.prompt_text)
            .map_err(|e| format!("Tokenization failed: {e}"))?;
        let prompt_tokens = prompt_ids.len();

        let cfg = SamplingConfig {
            temperature: request.temperature,
            top_k: request.top_k,
            seed: request.seed.unwrap_or(42),
            ..Default::default()
        };

        // Retrieve or initialize session state
        let (existing_state, existing_row, mut rng, prior_history) = {
            let mut sessions = self.sessions.write().unwrap();
            let session = sessions
                .entry(request.session_id.clone())
                .or_insert_with(|| SessionState {
                    history_tokens: Vec::new(),
                    recurrent_state: None,
                    last_logits_row: None,
                    rng: StdRng::seed_from_u64(cfg.seed),
                    generation_id: active.generation_id,
                });

            // If an explicit seed was supplied in the request, refresh session RNG
            if let Some(s) = request.seed {
                session.rng = StdRng::seed_from_u64(s);
            }

            // If the model was promoted (newer generation id), invalidate stale recurrent state
            if session.generation_id != active.generation_id {
                session.recurrent_state = None;
                session.last_logits_row = None;
                session.generation_id = active.generation_id;
            }

            // Context window bounding: keep recent history if total tokens would exceed 2048
            let max_ctx = 2048;
            if session.history_tokens.len() + prompt_ids.len() > max_ctx {
                let keep_len =
                    max_ctx.saturating_sub(prompt_ids.len() + request.max_new_tokens.min(256));
                let drain_count = session.history_tokens.len().saturating_sub(keep_len);
                if drain_count > 0 {
                    session.history_tokens.drain(0..drain_count);
                    session.recurrent_state = None;
                    session.last_logits_row = None;
                }
            }

            let prior_history = session.history_tokens.clone();
            let existing_state = session.recurrent_state.take();
            let existing_row = session.last_logits_row.take();
            let rng = session.rng.clone();

            (existing_state, existing_row, rng, prior_history)
        };

        let vocab = active.model.vocab_size;

        // Prefill once on fresh/truncated session, or recurrently step prompt tokens on continuation
        let (mut state, mut current_row, prefill_count) = match existing_state {
            None => {
                // Initial prefill on combined history + prompt tokens
                let mut full_prompt = prior_history.clone();
                full_prompt.extend_from_slice(&prompt_ids);
                if full_prompt.is_empty() {
                    return Err("Prompt text cannot be empty for initial session turn".to_string());
                }
                let (logits, init_state) = active
                    .model
                    .prefill(tokens_tensor(&full_prompt, &active.device));
                let [_, len, _v] = logits.dims();
                let last: Tensor<B, 1> = logits.narrow(1, len - 1, 1).reshape([vocab]);
                (init_state, row_to_vec(last), 1)
            }
            Some(mut s) => {
                // Continuation: step new prompt tokens into recurrent state without whole-prompt prefill
                let mut row = existing_row.unwrap_or_default();
                for &tok in &prompt_ids {
                    let token = Tensor::<B, 1, Int>::from_data(
                        TensorData::new(vec![tok as i64], [1]),
                        &active.device,
                    );
                    let (logits, new_state) = active.model.step(token, s);
                    s = new_state;
                    row = row_to_vec(logits.reshape([vocab]));
                }
                (s, row, 0)
            }
        };

        // Autoregressive decoding with cooperative cancellation and streaming UTF-8 chunking
        let mut generated_ids = Vec::new();
        let mut finish_reason = FinishReason::Length;
        let mut accumulated_text = String::new();
        let mut stream_decoder = StreamingUtf8Decoder::new();

        for _step in 0..request.max_new_tokens {
            if request.cancellation_token.is_cancelled() {
                finish_reason = FinishReason::Cancelled;
                break;
            }

            let next_tok = sample_logits(&current_row, &cfg, &mut rng);
            generated_ids.push(next_tok);

            // Check optional EOS stop condition
            if request.stop_on_eos && active.tokenizer.is_eos(next_tok) {
                finish_reason = FinishReason::Stop;
                break;
            }

            let raw_bytes = active.tokenizer.token_bytes(next_tok);
            let text_chunk = stream_decoder.push_bytes(&raw_bytes);
            if !text_chunk.is_empty() {
                accumulated_text.push_str(&text_chunk);
            }

            chunk_sender(GenerationChunk::Token {
                text: text_chunk,
                token_id: next_tok,
            })?;

            // Check for tool call proposal
            if accumulated_text.contains("<tool_call>") && accumulated_text.contains("</tool_call>")
            {
                if let Some(start_idx) = accumulated_text.find("<tool_call>") {
                    if let Some(end_idx) = accumulated_text.find("</tool_call>") {
                        let call_body = &accumulated_text[start_idx + 11..end_idx].trim();
                        let proposal = parse_tool_call_string(call_body);
                        chunk_sender(GenerationChunk::ToolCall(proposal))?;
                        finish_reason = FinishReason::ToolCall;
                        break;
                    }
                }
            }

            if _step + 1 == request.max_new_tokens {
                finish_reason = FinishReason::Length;
                break;
            }

            // Single recurrent O(1) step
            let token = Tensor::<B, 1, Int>::from_data(
                TensorData::new(vec![next_tok as i64], [1]),
                &active.device,
            );
            let (step_logits, new_state) = active.model.step(token, state);
            state = new_state;
            current_row = row_to_vec(step_logits.reshape([vocab]));
        }

        // Flush any trailing multibyte characters at end of stream
        let trailing = stream_decoder.finish();
        if !trailing.is_empty() {
            accumulated_text.push_str(&trailing);
            chunk_sender(GenerationChunk::Token {
                text: trailing,
                token_id: u32::MAX,
            })?;
        }

        // Commit turn prefix and updated recurrent state to session
        {
            let mut sessions = self.sessions.write().unwrap();
            if let Some(sess) = sessions.get_mut(&request.session_id) {
                sess.history_tokens.extend_from_slice(&prompt_ids);
                sess.history_tokens.extend_from_slice(&generated_ids);
                sess.recurrent_state = Some(state);
                sess.last_logits_row = Some(current_row);
                sess.rng = rng;
                sess.generation_id = active.generation_id;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let tokens_per_sec = if elapsed > 0.0 {
            generated_ids.len() as f64 / elapsed
        } else {
            0.0
        };

        let metadata = GenerationMetadata {
            backend_name: "Sail Native Engine".into(),
            model_id: self.model_name.clone(),
            generation_id: gen_id,
            prompt_tokens,
            generated_tokens: generated_ids.len(),
            prefill_count,
            elapsed_ms: elapsed * 1000.0,
            tokens_per_sec,
            finish_reason,
        };

        chunk_sender(GenerationChunk::Complete(metadata.clone()))?;
        Ok(metadata)
    }
}

// ---------------------------------------------------------------------------
// Pretrained Local Adapter (GGUF / Standard Local Runner)
// ---------------------------------------------------------------------------

/// Supported binary model container formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    SailSea,
    SailSsmLegacy,
    GgufV2V3,
}

/// Strictly inspects file headers to prevent silent corruption or loading unsupported formats.
pub fn validate_model_format(path: impl AsRef<Path>) -> Result<ModelFormat, String> {
    let path = path.as_ref();
    let metadata =
        std::fs::metadata(path).map_err(|e| format!("cannot stat {}: {e}", path.display()))?;
    let file_len = metadata.len();

    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut magic_buffer = [0u8; 8];
    file.read_exact(&mut magic_buffer)
        .map_err(|e| format!("file too small to determine format {}: {e}", path.display()))?;

    // 1. Prohibit Python pickle / raw PyTorch .pt explicitly
    if magic_buffer.starts_with(b"\x80\x02")
        || magic_buffer.starts_with(b"\x80\x03")
        || magic_buffer.starts_with(b"\x80\x04")
        || magic_buffer.starts_with(b"\x80\x05")
    {
        return Err(format!(
            "Unsupported format in '{}': Python pickle / raw PyTorch .pt is prohibited for security.",
            path.display()
        ));
    }

    // 2. Reject ZIP archives (e.g. renamed .zip / .docx / .jar)
    if magic_buffer.starts_with(b"PK\x03\x04") || magic_buffer.starts_with(b"PK") {
        return Err(format!(
            "Rejected ZIP container in '{}': expected native SEAPACK1 or GGUF format, arbitrary ZIP files are prohibited.",
            path.display()
        ));
    }

    // 3. Reject raw JSON
    if magic_buffer.starts_with(b"{") {
        return Err(format!(
            "Rejected raw JSON in '{}': expected packed binary artifact (.sea or .gguf).",
            path.display()
        ));
    }

    // 4. Native SAIL artifact (SEAPACK1 or legacy SSMPACK1)
    if &magic_buffer == crate::export::SEA_ARTIFACT_MAGIC || &magic_buffer == b"SSMPACK1" {
        if file_len < crate::export::SeaHeader::SIZE as u64 {
            return Err(format!(
                "Corrupted native artifact in '{}': file size ({} bytes) is smaller than header size ({} bytes)",
                path.display(),
                file_len,
                crate::export::SeaHeader::SIZE
            ));
        }
        let mut rest_of_header = [0u8; 16];
        file.read_exact(&mut rest_of_header)
            .map_err(|e| format!("failed reading header in {}: {e}", path.display()))?;

        let config_len = u32::from_le_bytes(rest_of_header[0..4].try_into().unwrap());
        let tokenizer_len = u32::from_le_bytes(rest_of_header[4..8].try_into().unwrap());
        let weights_len = u64::from_le_bytes(rest_of_header[8..16].try_into().unwrap());

        if config_len == 0 || tokenizer_len == 0 || weights_len == 0 {
            return Err(format!(
                "Invalid native artifact header in '{}': non-empty config, tokenizer, and weights required",
                path.display()
            ));
        }

        let total_required = crate::export::SeaHeader::SIZE as u64
            + config_len as u64
            + tokenizer_len as u64
            + weights_len;
        if file_len < total_required {
            return Err(format!(
                "Truncated native artifact in '{}': expected at least {} bytes from header, file has {} bytes",
                path.display(),
                total_required,
                file_len
            ));
        }

        if &magic_buffer == b"SSMPACK1" {
            return Ok(ModelFormat::SailSsmLegacy);
        } else {
            return Ok(ModelFormat::SailSea);
        }
    }

    // 5. GGUF format (v2 or v3)
    if &magic_buffer[0..4] == b"GGUF" {
        if file_len < 24 {
            return Err(format!(
                "Corrupted GGUF file in '{}': file size ({} bytes) is smaller than minimum GGUF header (24 bytes)",
                path.display(),
                file_len
            ));
        }
        let version = u32::from_le_bytes(magic_buffer[4..8].try_into().unwrap());
        if version != 2 && version != 3 {
            return Err(format!(
                "Unsupported GGUF version {} in '{}': only GGUF v2 and v3 are supported",
                version,
                path.display()
            ));
        }

        let mut counts_buffer = [0u8; 16];
        file.read_exact(&mut counts_buffer)
            .map_err(|e| format!("failed reading GGUF counts in {}: {e}", path.display()))?;
        let tensor_count = u64::from_le_bytes(counts_buffer[0..8].try_into().unwrap());
        let metadata_kv_count = u64::from_le_bytes(counts_buffer[8..16].try_into().unwrap());

        // Basic structural completeness check: a real GGUF model must contain tensors or metadata
        if tensor_count == 0 && metadata_kv_count == 0 {
            return Err(format!(
                "Malformed GGUF container in '{}': zero tensors and zero metadata KV pairs",
                path.display()
            ));
        }

        return Ok(ModelFormat::GgufV2V3);
    }

    Err(format!(
        "Unsupported or unrecognized model format in '{}' (magic {:?})",
        path.display(),
        &magic_buffer[..4]
    ))
}

/// Helper searching standard PATH for a binary name.
fn which_in_path(binary: &str) -> Option<PathBuf> {
    if let Ok(paths) = std::env::var("PATH") {
        for p in std::env::split_paths(&paths) {
            let candidate = p.join(binary);
            if candidate.is_file() {
                return Some(candidate);
            }
            #[cfg(windows)]
            {
                let candidate_exe = p.join(format!("{binary}.exe"));
                if candidate_exe.is_file() {
                    return Some(candidate_exe);
                }
            }
        }
    }
    None
}

/// Helper probing runner version via `--version`.
fn probe_runner_version(path: &Path) -> Option<String> {
    let mut cmd = Command::new(path);
    if let Some(parent) = path.parent() {
        let mut ld_path = parent.to_string_lossy().to_string();
        if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
            ld_path = format!("{ld_path}:{existing}");
        }
        cmd.env("LD_LIBRARY_PATH", ld_path);
    }
    let output = cmd.arg("--version").output().ok()?;
    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

/// Helper terminating child process tree.
fn kill_child_process_tree(child: &mut std::process::Child) {
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

/// Explicit test double for deterministic offline protocol and coordinator testing.
///
/// Clearly separated from production inference backends. Simulates token streaming,
/// tool calls, cancellation, and session isolation using pre-configured mock responses.
#[derive(Debug)]
pub struct MockModelAdapter {
    model_id: String,
    sessions: Arc<RwLock<HashMap<String, Vec<String>>>>,
    generation_counter: Arc<Mutex<u64>>,
    mock_responses: Arc<RwLock<HashMap<String, String>>>,
}

impl MockModelAdapter {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            generation_counter: Arc::new(Mutex::new(1)),
            mock_responses: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Configures a deterministic mock response when prompt contains `prompt_substring`.
    pub fn with_mock_response(self, prompt_substring: &str, response: &str) -> Self {
        self.mock_responses
            .write()
            .unwrap()
            .insert(prompt_substring.to_lowercase(), response.to_string());
        self
    }
}

impl ModelBackendAdapter for MockModelAdapter {
    fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "Mock Model Adapter (Test Double)".into(),
            model_id: self.model_id.clone(),
            format: "Mock".into(),
            context_limit: 4096,
            precision: "mock-f32".into(),
            device: "Memory Test Double".into(),
            capabilities: BackendCapabilities::text_backend(true),
        }
    }

    fn reset_session(&self, session_id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(session_id);
        Ok(())
    }

    fn generate_stream(
        &self,
        request: GenerateRequest,
        chunk_sender: &mut dyn FnMut(GenerationChunk) -> Result<(), String>,
    ) -> Result<GenerationMetadata, String> {
        self.validate_request(&request).map_err(|e| e.to_string())?;
        let start = Instant::now();
        let gen_id = {
            let mut c = self.generation_counter.lock().unwrap();
            let cur = *c;
            *c += 1;
            cur
        };

        let prompt_lower = request.prompt_text.to_lowercase();
        let response_text = {
            let mocks = self.mock_responses.read().unwrap();
            let mut matched = None;
            for (key, val) in mocks.iter() {
                if prompt_lower.contains(key) {
                    matched = Some(val.clone());
                    break;
                }
            }
            matched.unwrap_or_else(|| {
                format!(
                    "Mock test double response for: {}",
                    request.prompt_text.trim()
                )
            })
        };

        let prompt_tokens = request.prompt_text.split_whitespace().count().max(1);
        let mut tokens_sent = 0;
        let mut finish_reason = FinishReason::Stop;

        if let Some(tool_start) = response_text.find("<tool_call>") {
            let prefix = &response_text[..tool_start];
            let after = &response_text[tool_start + 11..];
            let call_body = if let Some(tool_end) = after.find("</tool_call>") {
                &after[..tool_end]
            } else {
                after
            };

            let words: Vec<&str> = prefix.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                if request.cancellation_token.is_cancelled() {
                    finish_reason = FinishReason::Cancelled;
                    break;
                }
                let text_chunk = if i == 0 {
                    word.to_string()
                } else {
                    format!(" {word}")
                };
                chunk_sender(GenerationChunk::Token {
                    text: text_chunk,
                    token_id: i as u32 + 1,
                })?;
                tokens_sent += 1;
                if tokens_sent >= request.max_new_tokens {
                    finish_reason = FinishReason::Length;
                    break;
                }
            }

            if finish_reason != FinishReason::Cancelled && finish_reason != FinishReason::Length {
                let proposal = parse_tool_call_string(call_body.trim());
                chunk_sender(GenerationChunk::ToolCall(proposal))?;
                finish_reason = FinishReason::ToolCall;
                tokens_sent += 1;
            }
        } else {
            let words: Vec<&str> = response_text.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                if request.cancellation_token.is_cancelled() {
                    finish_reason = FinishReason::Cancelled;
                    break;
                }
                let text_chunk = if i == 0 {
                    word.to_string()
                } else {
                    format!(" {word}")
                };
                chunk_sender(GenerationChunk::Token {
                    text: text_chunk,
                    token_id: i as u32 + 1,
                })?;
                tokens_sent += 1;
                if tokens_sent >= request.max_new_tokens {
                    finish_reason = FinishReason::Length;
                    break;
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let tokens_per_sec = if elapsed > 0.0 {
            tokens_sent as f64 / elapsed
        } else {
            0.0
        };

        let metadata = GenerationMetadata {
            backend_name: "Mock Model Adapter (Test Double)".into(),
            model_id: self.model_id.clone(),
            generation_id: gen_id,
            prompt_tokens,
            generated_tokens: tokens_sent,
            prefill_count: 1,
            elapsed_ms: elapsed * 1000.0,
            tokens_per_sec,
            finish_reason,
        };

        chunk_sender(GenerationChunk::Complete(metadata.clone()))?;
        Ok(metadata)
    }
}

/// Production adapter for local pretrained models (e.g. GGUF format executed via llama.cpp or local runner).
///
/// Performs full container header validation, runner discovery, supervised process execution,
/// real-time stdout streaming, and cooperative cancellation with process group termination.
/// If the runner binary is missing or not installed, fails closed with an actionable error.
#[derive(Debug)]
pub struct PretrainedLocalAdapter {
    model_path: PathBuf,
    model_id: String,
    model_format: ModelFormat,
    model_hash: String,
    tensor_count: u64,
    metadata_kv_count: u64,
    runner_path: Option<PathBuf>,
    runner_version: Option<String>,
    sessions: Arc<RwLock<HashMap<String, Vec<String>>>>,
    generation_counter: Arc<Mutex<u64>>,
}

impl PretrainedLocalAdapter {
    /// Loads and validates a local pretrained model from disk.
    pub fn open(path: impl AsRef<Path>, model_id: impl Into<String>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let format = validate_model_format(&path)?;
        if format != ModelFormat::GgufV2V3 {
            return Err(format!(
                "PretrainedLocalAdapter only supports GGUF models, but '{}' has format {:?}",
                path.display(),
                format
            ));
        }

        let mut file =
            File::open(&path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mut header = [0u8; 24];
        file.read_exact(&mut header)
            .map_err(|e| format!("failed reading GGUF header in {}: {e}", path.display()))?;

        let tensor_count = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let metadata_kv_count = u64::from_le_bytes(header[16..24].try_into().unwrap());

        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        header.hash(&mut hasher);
        let model_hash = format!("{:016x}", hasher.finish());

        // Probe for local runner in LLAMA_RUNNER_PATH, tools directory, ~/.llama-app, or PATH
        let runner_path = if let Ok(custom) = std::env::var("LLAMA_RUNNER_PATH") {
            let p = PathBuf::from(custom);
            if p.exists() { Some(p) } else { None }
        } else if let Some(found) = [
            "tools/llama/llama-cli",
            "tools/llama/llama",
            "tools/llama-cli",
            "/home/pedro/.llama-app/llama",
            "/home/pedro/.llama-app/llama-cli",
            "/home/pedro/.local/bin/llama-cli",
            "/home/pedro/.local/bin/llama",
        ]
        .iter()
        .map(PathBuf::from)
        .find(|pb| pb.is_file())
        {
            Some(found)
        } else {
            which_in_path("llama-cli").or_else(|| which_in_path("llama"))
        };

        let runner_version = runner_path.as_ref().and_then(|rp| probe_runner_version(rp));

        Ok(Self {
            model_path: path,
            model_id: model_id.into(),
            model_format: format,
            model_hash,
            tensor_count,
            metadata_kv_count,
            runner_path,
            runner_version,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            generation_counter: Arc::new(Mutex::new(1)),
        })
    }

    /// Allows explicitly configuring the path to a local runner binary (e.g. for testing or non-standard paths).
    pub fn with_runner_path(mut self, path: impl AsRef<Path>) -> Self {
        let p = path.as_ref().to_path_buf();
        self.runner_version = probe_runner_version(&p);
        self.runner_path = Some(p);
        self
    }

    pub fn model_format(&self) -> ModelFormat {
        self.model_format
    }

    pub fn model_hash(&self) -> &str {
        &self.model_hash
    }

    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    pub fn metadata_kv_count(&self) -> u64 {
        self.metadata_kv_count
    }

    pub fn runner_version(&self) -> Option<&str> {
        self.runner_version.as_deref()
    }
}

impl ModelBackendAdapter for PretrainedLocalAdapter {
    fn backend_info(&self) -> BackendInfo {
        let runner_desc = match (&self.runner_path, &self.runner_version) {
            (Some(p), Some(v)) => format!("llama.cpp ({}, runner: {})", v, p.display()),
            (Some(p), None) => format!("llama.cpp (runner: {})", p.display()),
            (None, _) => "llama.cpp (runner not found)".into(),
        };

        BackendInfo {
            name: "Pretrained Local GGUF".into(),
            model_id: format!("{}#{}", self.model_id, &self.model_hash[..8]),
            format: "GGUF v3".into(),
            context_limit: 4096,
            precision: "Q4_K_M".into(),
            device: runner_desc,
            capabilities: BackendCapabilities::text_backend(false),
        }
    }

    fn reset_session(&self, session_id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(session_id);
        Ok(())
    }

    fn generate_stream(
        &self,
        request: GenerateRequest,
        chunk_sender: &mut dyn FnMut(GenerationChunk) -> Result<(), String>,
    ) -> Result<GenerationMetadata, String> {
        self.validate_request(&request).map_err(|e| e.to_string())?;
        let start = Instant::now();
        let gen_id = {
            let mut c = self.generation_counter.lock().unwrap();
            let cur = *c;
            *c += 1;
            cur
        };

        // If local runner binary is absent, fail closed with an actionable error. Never fabricate an answer.
        let runner_path = match &self.runner_path {
            Some(p) => p.clone(),
            None => {
                return Err(format!(
                    "RunnerNotFound: Local pretrained GGUF runner ('llama-cli') was not found in PATH or LLAMA_RUNNER_PATH. \
                     To execute local GGUF models, install llama.cpp and ensure llama-cli is accessible. \
                     For deterministic offline testing without external binaries, use MockModelAdapter."
                ));
            }
        };

        let prompt_tokens = request.prompt_text.split_whitespace().count().max(1);

        // Spawn supervised child runner process
        let mut cmd = Command::new(&runner_path);
        if let Some(parent) = runner_path.parent() {
            let mut ld_path = parent.to_string_lossy().to_string();
            if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
                ld_path = format!("{ld_path}:{existing}");
            }
            cmd.env("LD_LIBRARY_PATH", ld_path);
        }
        cmd.arg("-m").arg(&self.model_path);
        cmd.arg("-p").arg(&request.prompt_text);
        cmd.arg("-n").arg(request.max_new_tokens.to_string());
        cmd.arg("--temp").arg(request.temperature.to_string());
        cmd.arg("--top-k").arg(request.top_k.to_string());
        let ngl = std::env::var("LLAMA_GPU_LAYERS").unwrap_or_else(|_| "99".to_string());
        cmd.arg("-ngl").arg(ngl);
        cmd.arg("--no-display-prompt");
        cmd.arg("--log-disable");
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn local pretrained runner '{}': {e}",
                runner_path.display()
            )
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture runner stdout".to_string())?;
        let mut reader = BufReader::new(stdout);
        let mut accumulated_text = String::new();
        let mut tokens_sent = 0;
        let mut finish_reason = FinishReason::Stop;

        let mut buffer = String::new();
        while let Ok(bytes_read) = reader.read_line(&mut buffer) {
            if bytes_read == 0 {
                break;
            }

            if request.cancellation_token.is_cancelled() {
                kill_child_process_tree(&mut child);
                finish_reason = FinishReason::Cancelled;
                break;
            }

            let line = std::mem::take(&mut buffer);
            accumulated_text.push_str(&line);

            chunk_sender(GenerationChunk::Token {
                text: line,
                token_id: tokens_sent as u32 + 1,
            })?;
            tokens_sent += 1;

            if tokens_sent >= request.max_new_tokens {
                kill_child_process_tree(&mut child);
                finish_reason = FinishReason::Length;
                break;
            }
        }

        let mut stderr_msg = String::new();
        if let Some(mut err_reader) = child.stderr.take() {
            use std::io::Read;
            let _ = err_reader.read_to_string(&mut stderr_msg);
        }

        let exit_status = child.wait();
        if tokens_sent == 0 && exit_status.as_ref().map(|s| !s.success()).unwrap_or(false) {
            let err_snippet = stderr_msg.trim();
            return Err(format!(
                "Runner process '{}' failed ({}):\n{}",
                runner_path.display(),
                exit_status.map(|s| s.to_string()).unwrap_or_else(|_| "unknown".into()),
                if err_snippet.is_empty() { "no stderr output" } else { err_snippet }
            ));
        }

        // Check for tool call proposal in output
        if finish_reason != FinishReason::Cancelled && finish_reason != FinishReason::Length {
            if accumulated_text.contains("<tool_call>") && accumulated_text.contains("</tool_call>")
            {
                if let Some(start_idx) = accumulated_text.find("<tool_call>") {
                    if let Some(end_idx) = accumulated_text.find("</tool_call>") {
                        let call_body = &accumulated_text[start_idx + 11..end_idx].trim();
                        let proposal = parse_tool_call_string(call_body);
                        chunk_sender(GenerationChunk::ToolCall(proposal))?;
                        finish_reason = FinishReason::ToolCall;
                        tokens_sent += 1;
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let tokens_per_sec = if elapsed > 0.0 {
            tokens_sent as f64 / elapsed
        } else {
            0.0
        };

        let metadata = GenerationMetadata {
            backend_name: "Pretrained Local GGUF".into(),
            model_id: format!("{}#{}", self.model_id, &self.model_hash[..8]),
            generation_id: gen_id,
            prompt_tokens,
            generated_tokens: tokens_sent,
            prefill_count: 1,
            elapsed_ms: elapsed * 1000.0,
            tokens_per_sec,
            finish_reason,
        };

        chunk_sender(GenerationChunk::Complete(metadata.clone()))?;
        Ok(metadata)
    }
}

// ---------------------------------------------------------------------------
// Assistant Coordinator
// ---------------------------------------------------------------------------

/// Coordinates high-level assistant interactions through any interchangeable `ModelBackendAdapter`.
pub struct AssistantCoordinator {
    adapter: Arc<dyn ModelBackendAdapter>,
}

impl AssistantCoordinator {
    pub fn new(adapter: Arc<dyn ModelBackendAdapter>) -> Self {
        Self { adapter }
    }

    /// Executes a scripted conversational task through the backend, capturing text and tool proposals.
    pub fn execute_turn(
        &self,
        session_id: &str,
        user_prompt: &str,
    ) -> Result<(String, Option<ToolProposal>, GenerationMetadata), String> {
        let mut accumulated_text = String::new();
        let mut proposed_tool = None;

        let req = GenerateRequest::new(session_id, user_prompt);
        let meta = self
            .adapter
            .generate_stream(req, &mut |chunk| match chunk {
                GenerationChunk::Token { text, .. } => {
                    accumulated_text.push_str(&text);
                    Ok(())
                }
                GenerationChunk::ToolCall(prop) => {
                    proposed_tool = Some(prop);
                    Ok(())
                }
                GenerationChunk::Complete(_) => Ok(()),
            })?;

        Ok((accumulated_text, proposed_tool, meta))
    }
}

/// Helper parsing a raw tool call string into `ToolProposal`.
fn parse_tool_call_string(raw: &str) -> ToolProposal {
    let mut clean = raw.trim();
    if clean.starts_with("<tool_call>") {
        clean = clean.trim_start_matches("<tool_call>");
    }
    if clean.ends_with("</tool_call>") {
        clean = clean.trim_end_matches("</tool_call>");
    }
    let raw = clean.trim();
    if let Some(paren_idx) = raw.find('(') {
        let name = raw[..paren_idx].trim();
        let rest = &raw[paren_idx + 1..];
        let end_idx = rest.rfind(')').unwrap_or(rest.len());
        let args = &rest[..end_idx];
        ToolProposal {
            call_id: format!("call_{}", rand::random::<u32>()),
            tool_name: name.to_string(),
            input: args.trim().trim_matches('"').to_string(),
        }
    } else {
        ToolProposal {
            call_id: format!("call_{}", rand::random::<u32>()),
            tool_name: raw.to_string(),
            input: String::new(),
        }
    }
}
