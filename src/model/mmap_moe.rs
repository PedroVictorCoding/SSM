//! Memory-mapped weight streaming for Mixture of Experts (MoE).
//!
//! Decouples parameter capacity from VRAM / system RAM by persisting large banks
//! of MoE experts in a contiguous, aligned binary file (`experts.bin`) and memory-mapping
//! them via `memmap2`.
//!
//! During inference or single-token decoding, the router dynamically selects Top-K experts.
//! Only the active experts selected are streamed from page cache / disk into device tensors,
//! drastically reducing memory consumption while allowing massive parameter banks.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use burn::module::Param;
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::activation::softmax;
use memmap2::Mmap;

use super::mlp::{SwiGlu, SwiGluConfig};
use super::moe::Moe;

pub const MMAP_MOE_MAGIC: &[u8; 8] = b"SSMMOE01";

/// Header layout for binary memory-mapped expert storage.
#[derive(Debug, Clone, Copy)]
pub struct MmapHeader {
    pub version: u32,
    pub num_experts: u32,
    pub d_model: u32,
    pub d_hidden: u32,
    pub floats_per_expert: u64,
}

impl MmapHeader {
    pub const SIZE: usize = 32; // 8 (magic) + 4 + 4 + 4 + 4 + 8

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(MMAP_MOE_MAGIC);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.num_experts.to_le_bytes());
        buf[16..20].copy_from_slice(&self.d_model.to_le_bytes());
        buf[20..24].copy_from_slice(&self.d_hidden.to_le_bytes());
        buf[24..32].copy_from_slice(&self.floats_per_expert.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < Self::SIZE {
            return Err("Buffer too small for MmapHeader".into());
        }
        if &bytes[0..8] != MMAP_MOE_MAGIC {
            return Err("Invalid magic bytes in MmapHeader".into());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let num_experts = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let d_model = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let d_hidden = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let floats_per_expert = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

        Ok(Self {
            version,
            num_experts,
            d_model,
            d_hidden,
            floats_per_expert,
        })
    }
}

/// Zero-copy, memory-mapped storage for an arbitrary number of MoE expert weights.
pub struct MmapExpertStorage {
    pub header: MmapHeader,
    pub path: PathBuf,
    mmap: Mmap,
}

impl MmapExpertStorage {
    /// Creates a new storage file initialized with zeroed weights.
    pub fn create_empty(
        path: impl AsRef<Path>,
        num_experts: usize,
        d_model: usize,
        d_hidden: usize,
    ) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let floats_per_expert = (3 * d_model * d_hidden) as u64;
        let header = MmapHeader {
            version: 1,
            num_experts: num_experts as u32,
            d_model: d_model as u32,
            d_hidden: d_hidden as u32,
            floats_per_expert,
        };

        let total_bytes = MmapHeader::SIZE as u64 + (num_experts as u64 * floats_per_expert * 4);

        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)?;
            file.set_len(total_bytes)?;
            let mut file_buf = file;
            file_buf.seek(SeekFrom::Start(0))?;
            file_buf.write_all(&header.to_bytes())?;
            file_buf.flush()?;
        }

        Self::open(path)
    }

    /// Opens an existing memory-mapped expert storage file.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let header = MmapHeader::from_bytes(&mmap[0..MmapHeader::SIZE])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let expected_size = MmapHeader::SIZE
            + (header.num_experts as usize * header.floats_per_expert as usize * 4);
        if mmap.len() < expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "File size {} is smaller than expected {}",
                    mmap.len(),
                    expected_size
                ),
            ));
        }

        Ok(Self { header, path, mmap })
    }

    /// Exports all experts from an in-memory `Moe<B>` directly into an `experts.bin` mmap file.
    pub fn dump_from_moe<B: Backend>(
        path: impl AsRef<Path>,
        moe: &Moe<B>,
    ) -> std::io::Result<Self> {
        let _storage =
            Self::create_empty(path.as_ref(), moe.num_experts, moe.d_model, moe.d_hidden)?;

        let mut file = OpenOptions::new().write(true).open(path.as_ref())?;
        file.seek(SeekFrom::Start(MmapHeader::SIZE as u64))?;

        for expert in &moe.experts {
            let gate_data = expert.gate.weight.val().into_data();
            let gate_floats = gate_data.as_slice::<f32>().unwrap();

            let up_data = expert.up.weight.val().into_data();
            let up_floats = up_data.as_slice::<f32>().unwrap();

            let down_data = expert.down.weight.val().into_data();
            let down_floats = down_data.as_slice::<f32>().unwrap();

            let mut byte_buf =
                Vec::with_capacity((gate_floats.len() + up_floats.len() + down_floats.len()) * 4);
            for &f in gate_floats
                .iter()
                .chain(up_floats.iter())
                .chain(down_floats.iter())
            {
                byte_buf.extend_from_slice(&f.to_le_bytes());
            }
            file.write_all(&byte_buf)?;
        }
        file.flush()?;

        Self::open(path)
    }

    /// Reads expert weights for `expert_id` directly from memory map and instantiates a `SwiGlu<B>`.
    pub fn load_expert<B: Backend>(&self, expert_id: usize, device: &B::Device) -> SwiGlu<B> {
        assert!(
            expert_id < self.header.num_experts as usize,
            "expert_id {} out of bounds ({})",
            expert_id,
            self.header.num_experts
        );

        let d_model = self.header.d_model as usize;
        let d_hidden = self.header.d_hidden as usize;
        let floats_per_expert = self.header.floats_per_expert as usize;
        let bytes_per_expert = floats_per_expert * 4;

        let start_byte = MmapHeader::SIZE + (expert_id * bytes_per_expert);
        let expert_slice = &self.mmap[start_byte..start_byte + bytes_per_expert];

        // Zero-copy float parsing
        let mut float_vec = Vec::with_capacity(floats_per_expert);
        for chunk in expert_slice.chunks_exact(4) {
            float_vec.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        let gate_size = d_model * d_hidden;
        let up_size = d_model * d_hidden;
        let down_size = d_hidden * d_model;

        let gate_floats = &float_vec[0..gate_size];
        let up_floats = &float_vec[gate_size..gate_size + up_size];
        let down_floats = &float_vec[gate_size + up_size..gate_size + up_size + down_size];

        let gate_td = TensorData::new(gate_floats.to_vec(), [d_model, d_hidden]);
        let up_td = TensorData::new(up_floats.to_vec(), [d_model, d_hidden]);
        let down_td = TensorData::new(down_floats.to_vec(), [d_hidden, d_model]);

        let mut expert = SwiGluConfig::new(d_model, d_hidden).init(device);
        expert.gate.weight = Param::from_tensor(Tensor::from_data(gate_td, device));
        expert.up.weight = Param::from_tensor(Tensor::from_data(up_td, device));
        expert.down.weight = Param::from_tensor(Tensor::from_data(down_td, device));

        expert
    }

    pub fn num_experts(&self) -> usize {
        self.header.num_experts as usize
    }

    pub fn d_model(&self) -> usize {
        self.header.d_model as usize
    }

    pub fn d_hidden(&self) -> usize {
        self.header.d_hidden as usize
    }
}

/// Dynamic Streaming MoE layer.
///
/// Holds the router in device memory, while streaming active expert weights
/// on-demand from `MmapExpertStorage` with an optional LRU cache.
pub struct StreamingMoe<B: Backend> {
    pub router: Linear<B>,
    pub storage: Arc<MmapExpertStorage>,
    pub num_experts: usize,
    pub top_k: usize,
    pub dynamic_threshold: f64,
    pub d_model: usize,
    pub d_hidden: usize,
    cache: Arc<Mutex<HashMap<usize, SwiGlu<B>>>>,
}

impl<B: Backend> StreamingMoe<B> {
    pub fn new(
        router: Linear<B>,
        storage: Arc<MmapExpertStorage>,
        top_k: usize,
        dynamic_threshold: f64,
    ) -> Self {
        let num_experts = storage.num_experts();
        let d_model = storage.d_model();
        let d_hidden = storage.d_hidden();
        let top_k = top_k.clamp(1, num_experts);

        Self {
            router,
            storage,
            num_experts,
            top_k,
            dynamic_threshold,
            d_model,
            d_hidden,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Fetches an expert from cache or streams it from mmap.
    fn get_or_stream_expert(&self, expert_id: usize, device: &B::Device) -> SwiGlu<B> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(expert) = cache.get(&expert_id) {
            return expert.clone();
        }
        let expert = self.storage.load_expert::<B>(expert_id, device);
        // Keep working set bounded
        if cache.len() >= 8 {
            cache.clear();
        }
        cache.insert(expert_id, expert.clone());
        expert
    }

    /// Total theoretical parameters across all experts in storage and router.
    pub fn total_params(&self) -> usize {
        let router_p = self.d_model * self.num_experts;
        let per_expert_p = 3 * self.d_model * self.d_hidden;
        router_p + self.num_experts * per_expert_p
    }

    /// Average active parameters per token.
    pub fn active_params_per_token(&self) -> usize {
        let router_p = self.d_model * self.num_experts;
        let per_expert_p = 3 * self.d_model * self.d_hidden;
        router_p + self.top_k * per_expert_p
    }

    /// Forward pass with sparse expert streaming:
    /// Only experts actively chosen by tokens in `x` are loaded and executed.
    pub fn forward_with_aux<const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> (Tensor<B, D>, Tensor<B, 1>) {
        let dims = x.dims();
        let d_model = dims[D - 1];
        let num_tokens: usize = dims[..D - 1].iter().product();
        let device = x.device();

        let x_2d: Tensor<B, 2> = x.reshape([num_tokens, d_model]);

        let router_logits = self.router.forward(x_2d.clone());
        let router_probs = softmax(router_logits.clone(), 1);
        let p_e = router_probs.clone().mean_dim(0).reshape([self.num_experts]);

        let (weights, f_e) = if self.top_k == 1 {
            let max_indices = router_logits.clone().argmax(1);
            let mut one_hot_f = Vec::with_capacity(self.num_experts);
            let mut w_cols = Vec::with_capacity(self.num_experts);

            for e in 0..self.num_experts {
                let e_tensor = Tensor::<B, 2, Int>::full([num_tokens, 1], e as i64, &device);
                let is_e = max_indices.clone().equal(e_tensor).float();
                let p_col = router_probs.clone().narrow(1, e, 1);
                let w = if self.dynamic_threshold > 0.0 {
                    let thresh = Tensor::<B, 2>::full(
                        [num_tokens, 1],
                        self.dynamic_threshold as f32,
                        &device,
                    );
                    let above_thresh = p_col.clone().greater_equal(thresh).float();
                    p_col * is_e.clone() * above_thresh
                } else {
                    p_col * is_e.clone()
                };
                one_hot_f.push(is_e.mean());
                w_cols.push(w);
            }
            let f_e = Tensor::cat(one_hot_f, 0);
            let weights = Tensor::cat(w_cols, 1);
            (weights, f_e)
        } else {
            let (top_val, top_idx) = router_logits.clone().topk_with_indices(self.top_k, 1);
            let top_probs = softmax(top_val, 1);

            let mut w_cols = Vec::with_capacity(self.num_experts);
            let mut f_cols = Vec::with_capacity(self.num_experts);

            for e in 0..self.num_experts {
                let e_idx = Tensor::<B, 2, Int>::full([num_tokens, self.top_k], e as i64, &device);
                let is_e = top_idx.clone().equal(e_idx).float();
                let e_weight = (top_probs.clone() * is_e.clone()).sum_dim(1);

                let active_w = if self.dynamic_threshold > 0.0 {
                    let thresh = Tensor::<B, 2>::full(
                        [num_tokens, 1],
                        self.dynamic_threshold as f32,
                        &device,
                    );
                    let above = e_weight.clone().greater_equal(thresh).float();
                    e_weight * above
                } else {
                    e_weight
                };

                let token_used_e = is_e.sum_dim(1).clamp_max(1.0);
                f_cols.push(token_used_e.mean());
                w_cols.push(active_w);
            }

            let f_e = Tensor::cat(f_cols, 0);
            let weights = Tensor::cat(w_cols, 1);
            (weights, f_e)
        };

        let aux_loss = (f_e * p_e).sum().mul_scalar(self.num_experts as f32);

        // Stream and evaluate only experts that receive non-zero token routing weights
        let mut out_accum = Tensor::<B, 2>::zeros([num_tokens, d_model], &device);

        for e in 0..self.num_experts {
            let w_e: Tensor<B, 2> = weights.clone().narrow(1, e, 1);
            // Check if this expert has any non-zero routing weights in the current batch
            let sum_w: f32 = w_e.clone().sum().into_data().as_slice::<f32>().unwrap()[0];
            if sum_w > 0.0 {
                let expert = self.get_or_stream_expert(e, &device);
                let expert_out = expert.forward(x_2d.clone());
                out_accum = out_accum + expert_out * w_e;
            }
        }

        let out = out_accum.reshape(dims);
        (out, aux_loss)
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.forward_with_aux(x).0
    }
}
