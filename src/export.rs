//! Single-file native artifact export and standalone execution (.sea format:
//! State space model, mixture of Experts, Active parameters).
//!
//! Packs model architecture, hyperparameters, tokenizer (BPE or character),
//! and full precision parameter weights into a single portable binary file (`.sea`).
//!
//! Enables zero-dependency distribution and execution across Linux, macOS, Windows,
//! iOS, and WebAssembly in browsers without requiring Python or training harnesses.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

use crate::data::Tokenizer;
use crate::model::{HybridLm, HybridLmConfig};
use crate::train::load_run;

pub const SEA_ARTIFACT_MAGIC: &[u8; 8] = b"SEAPACK1";
pub const SSM_ARTIFACT_MAGIC: &[u8; 8] = SEA_ARTIFACT_MAGIC;

/// Header definition for .sea (and .ssm) native standalone model artifacts.
#[derive(Debug, Clone, Copy)]
pub struct SeaHeader {
    pub config_len: u32,
    pub tokenizer_len: u32,
    pub weights_len: u64,
}

pub type SsmHeader = SeaHeader;

impl SeaHeader {
    pub const SIZE: usize = 24; // 8 (magic) + 4 (config_len) + 4 (tok_len) + 8 (weights_len)

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(SEA_ARTIFACT_MAGIC);
        buf[8..12].copy_from_slice(&self.config_len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.tokenizer_len.to_le_bytes());
        buf[16..24].copy_from_slice(&self.weights_len.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < Self::SIZE {
            return Err("Buffer too short for SeaHeader".into());
        }
        let magic = &bytes[0..8];
        if magic != SEA_ARTIFACT_MAGIC && magic != b"SSMPACK1" {
            return Err("Invalid magic bytes: expected SEAPACK1".into());
        }
        let config_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let tokenizer_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let weights_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

        Ok(Self {
            config_len,
            tokenizer_len,
            weights_len,
        })
    }
}

/// Atomically packs an in-memory model, config, and tokenizer into a standalone `.sea` binary artifact.
pub fn pack_to_artifact<B: Backend>(
    model: &HybridLm<B>,
    config: &HybridLmConfig,
    tokenizer: &Tokenizer,
    out_file: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let config_json =
        serde_json::to_string_pretty(config).map_err(|e| format!("serializing config: {e}"))?;
    let tokenizer_json = serde_json::to_string_pretty(tokenizer)
        .map_err(|e| format!("serializing tokenizer: {e}"))?;

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let weights_bytes = recorder
        .record(model.clone().into_record(), ())
        .map_err(|e| format!("recording weights: {e:?}"))?;

    let header = SeaHeader {
        config_len: config_json.len() as u32,
        tokenizer_len: tokenizer_json.len() as u32,
        weights_len: weights_bytes.len() as u64,
    };

    let out_path = out_file.as_ref().to_path_buf();
    let mut bytes = Vec::with_capacity(
        SeaHeader::SIZE + config_json.len() + tokenizer_json.len() + weights_bytes.len(),
    );
    bytes.extend_from_slice(&header.to_bytes());
    bytes.extend_from_slice(config_json.as_bytes());
    bytes.extend_from_slice(tokenizer_json.as_bytes());
    bytes.extend_from_slice(&weights_bytes);
    crate::storage::atomic_write(&out_path, &bytes)
        .map_err(|e| format!("publishing {}: {e}", out_path.display()))?;

    Ok(out_path)
}

/// Exports a trained checkpoint directory to a standalone `.sea` binary artifact.
pub fn export_to_artifact<B: Backend>(
    run_dir: impl AsRef<Path>,
    out_file: impl AsRef<Path>,
    device: &B::Device,
) -> Result<PathBuf, String> {
    let (model, config, tokenizer) = load_run::<B>(&run_dir, device)?;
    pack_to_artifact(&model, &config, &tokenizer, out_file)
}

/// Loads a complete runnable model from a standalone `.sea` binary artifact.
pub fn load_artifact<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(HybridLm<B>, HybridLmConfig, Tokenizer), String> {
    let mut file = File::open(path.as_ref())
        .map_err(|e| format!("opening {}: {e}", path.as_ref().display()))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| format!("reading {}: {e}", path.as_ref().display()))?;

    load_artifact_from_bytes::<B>(&buffer, device)
}

/// Loads a runnable model directly from an in-memory byte slice (e.g. static embedded binary or WASM).
pub fn load_artifact_from_bytes<B: Backend>(
    bytes: &[u8],
    device: &B::Device,
) -> Result<(HybridLm<B>, HybridLmConfig, Tokenizer), String> {
    let header = SeaHeader::from_bytes(bytes)?;

    let mut cursor = SeaHeader::SIZE;

    let config_end = cursor + header.config_len as usize;
    if config_end > bytes.len() {
        return Err("Unexpected EOF reading config".into());
    }
    let config_str = std::str::from_utf8(&bytes[cursor..config_end])
        .map_err(|e| format!("UTF-8 error in config: {e}"))?;
    let config: HybridLmConfig =
        serde_json::from_str(config_str).map_err(|e| format!("parsing config: {e}"))?;
    cursor = config_end;

    let tok_end = cursor + header.tokenizer_len as usize;
    if tok_end > bytes.len() {
        return Err("Unexpected EOF reading tokenizer".into());
    }
    let tok_str = std::str::from_utf8(&bytes[cursor..tok_end])
        .map_err(|e| format!("UTF-8 error in tokenizer: {e}"))?;
    let mut tokenizer: Tokenizer =
        serde_json::from_str(tok_str).map_err(|e| format!("parsing tokenizer: {e}"))?;
    tokenizer.rebuild_index();
    cursor = tok_end;

    let weights_end = cursor + header.weights_len as usize;
    if weights_end > bytes.len() {
        return Err("Unexpected EOF reading weights".into());
    }
    let weights_bytes = &bytes[cursor..weights_end];

    let mut model = config.init::<B>(device);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load(weights_bytes.to_vec(), device)
        .map_err(|e| format!("loading weights record: {e:?}"))?;

    model = model.load_record(record);

    Ok((model, config, tokenizer))
}
