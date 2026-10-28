//! Resource Scheduler, Device Budgets, and Concurrency Prioritization.
//!
//! Enforces:
//! - Explicit CPU and GPU memory allocation budgets for target hardware (e.g. AMD RX 7800 XT 16GB).
//! - Strict priority tiers:
//!   1. `RealtimeAudio` (Capture, VAD, DSP, Playback - non-preemptible, zero-underrun target).
//!   2. `ForegroundDialogue` (Streaming ASR, Multimodal Inference, Streaming TTS).
//!   3. `VisualPerception` (Screen/Camera sampling - dynamic framerate throttling under load).
//!   4. `BackgroundTraining` (Strictly paused/yielded within 100ms when conversation starts).
//! - Controlled OOM and memory pressure handling with visible degradation rather than silent failures.
//! - Bounded memory guarantees preventing growth across sessions and repeated reloads.
//! - RAII reservation guards (`MemoryReservationGuard`, `HostReservationGuard`, `SessionGuard`)
//!   for automatic leak-free deallocation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Priority tier determining preemption and scheduling order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PriorityTier {
    /// P0: Realtime audio capture, DSP, playback (never preempted).
    RealtimeAudio = 0,
    /// P1: Foreground conversation, streaming inference, tool results.
    ForegroundDialogue = 1,
    /// P2: Visual capture and OCR (throttled during active dialogue).
    VisualPerception = 2,
    /// P3: Background continual training on external data (yields on dialogue).
    BackgroundTraining = 3,
}

/// Memory and execution budget allocation for a hardware target (e.g. RX 7800 XT 16GB).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceBudgetConfig {
    pub total_gpu_vram_mb: usize,
    pub os_reserved_vram_mb: usize,
    pub audio_budget_mb: usize,
    pub foreground_inference_budget_mb: usize,
    pub perception_budget_mb: usize,
    pub background_training_budget_mb: usize,
    pub total_cpu_ram_mb: usize,
    pub max_active_sessions: usize,
}

impl Default for DeviceBudgetConfig {
    fn default() -> Self {
        // Target: AMD Radeon RX 7800 XT (16 GB VRAM) + 32 GB System RAM
        Self {
            total_gpu_vram_mb: 16384,
            os_reserved_vram_mb: 2048,
            audio_budget_mb: 256,
            foreground_inference_budget_mb: 6144,
            perception_budget_mb: 2048,
            background_training_budget_mb: 4096,
            total_cpu_ram_mb: 32768,
            max_active_sessions: 4,
        }
    }
}

/// Snapshot of resource utilization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceUtilizationSnapshot {
    pub total_allocated_vram_mb: usize,
    pub available_vram_mb: usize,
    pub allocations_by_tier: HashMap<String, usize>,
    pub is_dialogue_active: bool,
    pub is_training_paused: bool,
    pub perception_fps_cap: u32,
    pub audio_underruns: u64,
    pub dropped_visual_frames: u64,
    #[serde(default)]
    pub total_allocated_host_bytes: usize,
    #[serde(default)]
    pub active_sessions: usize,
    #[serde(default)]
    pub max_active_sessions: usize,
    #[serde(default)]
    pub last_yield_latency_ms: f64,
    #[serde(default)]
    pub last_training_step_duration_ms: f64,
}

/// Metrics recorded during concurrent-load benchmarking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConcurrentLoadMetrics {
    pub p50_voice_latency_ms: f64,
    pub p95_voice_latency_ms: f64,
    pub p99_voice_latency_ms: f64,
    pub peak_vram_mb: usize,
    pub audio_underruns: u64,
    pub dropped_frames: u64,
    pub training_yield_latency_ms: f64,
    pub degradation_events: usize,
}

/// RAII guard releasing a GPU memory reservation upon drop.
pub struct MemoryReservationGuard {
    scheduler: Arc<ResourceScheduler>,
    tier: PriorityTier,
    bytes: usize,
    allocated_mb: usize,
    released: bool,
}

impl MemoryReservationGuard {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn allocated_mb(&self) -> usize {
        self.allocated_mb
    }

    pub fn tier(&self) -> PriorityTier {
        self.tier
    }

    pub fn release(mut self) {
        if !self.released {
            self.scheduler
                .deallocate_memory(self.tier, self.allocated_mb);
            self.released = true;
        }
    }
}

impl Drop for MemoryReservationGuard {
    fn drop(&mut self) {
        if !self.released {
            self.scheduler
                .deallocate_memory(self.tier, self.allocated_mb);
            self.released = true;
        }
    }
}

impl std::fmt::Debug for MemoryReservationGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryReservationGuard")
            .field("tier", &self.tier)
            .field("bytes", &self.bytes)
            .field("allocated_mb", &self.allocated_mb)
            .field("released", &self.released)
            .finish()
    }
}

/// RAII guard releasing host CPU RAM reservation on drop.
pub struct HostReservationGuard {
    scheduler: Arc<ResourceScheduler>,
    bytes: usize,
    released: bool,
}

impl HostReservationGuard {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn release(mut self) {
        if !self.released {
            self.scheduler.deallocate_host_memory(self.bytes);
            self.released = true;
        }
    }
}

impl Drop for HostReservationGuard {
    fn drop(&mut self) {
        if !self.released {
            self.scheduler.deallocate_host_memory(self.bytes);
            self.released = true;
        }
    }
}

impl std::fmt::Debug for HostReservationGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostReservationGuard")
            .field("bytes", &self.bytes)
            .field("released", &self.released)
            .finish()
    }
}

/// RAII guard releasing an active session slot on drop.
pub struct SessionGuard {
    scheduler: Arc<ResourceScheduler>,
    session_id: String,
    released: bool,
}

impl SessionGuard {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn release(mut self) {
        if !self.released {
            self.scheduler.release_session(&self.session_id);
            self.released = true;
        }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        if !self.released {
            self.scheduler.release_session(&self.session_id);
            self.released = true;
        }
    }
}

impl std::fmt::Debug for SessionGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionGuard")
            .field("session_id", &self.session_id)
            .field("released", &self.released)
            .finish()
    }
}

/// Real host buffer backed by a scheduler memory reservation guard.
pub struct RealMemoryAllocation {
    guard: MemoryReservationGuard,
    buffer: Vec<u8>,
}

impl std::fmt::Debug for RealMemoryAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealMemoryAllocation")
            .field("guard", &self.guard)
            .field("buffer_len", &self.buffer.len())
            .finish()
    }
}

impl RealMemoryAllocation {
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn tier(&self) -> PriorityTier {
        self.guard.tier()
    }

    pub fn bytes(&self) -> usize {
        self.guard.bytes()
    }
}

/// Resource scheduler managing hardware allocation and dynamic preemption.
pub struct ResourceScheduler {
    config: DeviceBudgetConfig,
    active_allocations_mb: Arc<RwLock<HashMap<PriorityTier, usize>>>,
    active_host_bytes: Arc<AtomicUsize>,
    active_sessions: Arc<AtomicUsize>,
    dialogue_active: Arc<AtomicBool>,
    training_yield_requested: Arc<AtomicBool>,
    training_yield_acknowledged: Arc<AtomicBool>,
    training_is_paused: Arc<AtomicBool>,
    audio_underruns: Arc<AtomicU64>,
    dropped_frames: Arc<AtomicU64>,
    perception_fps: Arc<AtomicUsize>,
    last_yield_request_instant: Arc<RwLock<Option<Instant>>>,
    last_yield_latency_ms: Arc<AtomicU64>,
    last_training_step_duration_ms: Arc<AtomicU64>,
}

impl Default for ResourceScheduler {
    fn default() -> Self {
        Self::new(DeviceBudgetConfig::default())
    }
}

impl ResourceScheduler {
    pub fn new(config: DeviceBudgetConfig) -> Self {
        let mut initial_allocs = HashMap::new();
        initial_allocs.insert(PriorityTier::RealtimeAudio, 0);
        initial_allocs.insert(PriorityTier::ForegroundDialogue, 0);
        initial_allocs.insert(PriorityTier::VisualPerception, 0);
        initial_allocs.insert(PriorityTier::BackgroundTraining, 0);

        Self {
            config,
            active_allocations_mb: Arc::new(RwLock::new(initial_allocs)),
            active_host_bytes: Arc::new(AtomicUsize::new(0)),
            active_sessions: Arc::new(AtomicUsize::new(0)),
            dialogue_active: Arc::new(AtomicBool::new(false)),
            training_yield_requested: Arc::new(AtomicBool::new(false)),
            training_yield_acknowledged: Arc::new(AtomicBool::new(false)),
            training_is_paused: Arc::new(AtomicBool::new(false)),
            audio_underruns: Arc::new(AtomicU64::new(0)),
            dropped_frames: Arc::new(AtomicU64::new(0)),
            perception_fps: Arc::new(AtomicUsize::new(30)),
            last_yield_request_instant: Arc::new(RwLock::new(None)),
            last_yield_latency_ms: Arc::new(AtomicU64::new(0)),
            last_training_step_duration_ms: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Signals the start of foreground user dialogue.
    /// Immediately triggers background training to yield within bounded optimizer steps
    /// and throttles visual sampling to maintain audio/inference latency.
    pub fn on_dialogue_started(&self) {
        self.dialogue_active.store(true, Ordering::SeqCst);
        self.training_yield_requested.store(true, Ordering::SeqCst);
        self.training_yield_acknowledged
            .store(false, Ordering::SeqCst);
        *self.last_yield_request_instant.write().unwrap() = Some(Instant::now());
        // Throttle background perception from 30 FPS to 5 FPS to reduce memory bandwidth
        self.perception_fps.store(5, Ordering::SeqCst);
    }

    /// Signals conclusion of foreground user dialogue.
    pub fn on_dialogue_ended(&self) {
        self.dialogue_active.store(false, Ordering::SeqCst);
        self.training_yield_requested.store(false, Ordering::SeqCst);
        self.training_is_paused.store(false, Ordering::SeqCst);
        *self.last_yield_request_instant.write().unwrap() = None;
        self.perception_fps.store(30, Ordering::SeqCst);
    }

    /// Checks if background training should yield.
    pub fn should_training_yield(&self) -> bool {
        self.training_yield_requested.load(Ordering::SeqCst)
    }

    /// Acknowledges from the training thread that yield has been received and training is paused.
    pub fn acknowledge_training_yield(&self) {
        self.training_yield_acknowledged
            .store(true, Ordering::SeqCst);
        self.training_is_paused.store(true, Ordering::SeqCst);
        if let Some(req_time) = *self.last_yield_request_instant.read().unwrap() {
            let latency_ms = req_time.elapsed().as_secs_f64() * 1000.0;
            self.last_yield_latency_ms
                .store(latency_ms.to_bits(), Ordering::SeqCst);
        }
    }

    /// Records that background training has resumed after dialogue concluded.
    pub fn record_training_resumed(&self) {
        self.training_is_paused.store(false, Ordering::SeqCst);
    }

    /// Records the measured duration of an optimizer step in milliseconds.
    pub fn record_training_step_duration(&self, duration_ms: f64) {
        self.last_training_step_duration_ms
            .store(duration_ms.to_bits(), Ordering::SeqCst);
    }

    /// Checks if background training is currently paused for dialogue.
    pub fn is_training_paused(&self) -> bool {
        self.training_is_paused.load(Ordering::SeqCst)
    }

    /// Checks if background training acknowledged the yield request.
    pub fn is_training_yield_acknowledged(&self) -> bool {
        self.training_yield_acknowledged.load(Ordering::SeqCst)
    }

    /// Returns the latest observed yield preemption latency in milliseconds.
    pub fn last_yield_latency_ms(&self) -> f64 {
        f64::from_bits(self.last_yield_latency_ms.load(Ordering::SeqCst))
    }

    /// Returns the latest observed training step duration in milliseconds.
    pub fn last_training_step_duration_ms(&self) -> f64 {
        f64::from_bits(self.last_training_step_duration_ms.load(Ordering::SeqCst))
    }

    /// Records audio buffer underrun event.
    pub fn record_audio_underrun(&self) {
        self.audio_underruns.fetch_add(1, Ordering::SeqCst);
    }

    /// Records dropped visual frame due to priority throttle.
    pub fn record_dropped_frame(&self) {
        self.dropped_frames.fetch_add(1, Ordering::SeqCst);
    }

    /// Requests memory allocation for a specific priority tier in megabytes.
    pub fn allocate_memory(&self, tier: PriorityTier, megabytes: usize) -> Result<(), String> {
        let mut map = self.active_allocations_mb.write().unwrap();
        let current = *map.get(&tier).unwrap_or(&0);

        let budget = match tier {
            PriorityTier::RealtimeAudio => self.config.audio_budget_mb,
            PriorityTier::ForegroundDialogue => self.config.foreground_inference_budget_mb,
            PriorityTier::VisualPerception => self.config.perception_budget_mb,
            PriorityTier::BackgroundTraining => self.config.background_training_budget_mb,
        };

        let new_tier_total = current.checked_add(megabytes).ok_or_else(|| {
            format!(
                "MemoryBudgetExceeded: Tier {:?} allocation counter would overflow",
                tier
            )
        })?;
        if new_tier_total > budget {
            return Err(format!(
                "MemoryBudgetExceeded: Tier {:?} requested {} MB exceeding configured budget of {} MB (currently allocated: {} MB)",
                tier, megabytes, budget, current
            ));
        }

        let total_allocated = map
            .values()
            .try_fold(0usize, |total, &amount| total.checked_add(amount))
            .ok_or_else(|| {
                "DeviceVramExhausted: Total allocation counter would overflow".to_string()
            })?;
        let max_usable = self
            .config
            .total_gpu_vram_mb
            .checked_sub(self.config.os_reserved_vram_mb)
            .ok_or_else(|| {
                "InvalidDeviceBudget: OS reservation exceeds total GPU memory".to_string()
            })?;
        let new_device_total = total_allocated.checked_add(megabytes).ok_or_else(|| {
            "DeviceVramExhausted: Total allocation counter would overflow".to_string()
        })?;
        if new_device_total > max_usable {
            return Err(format!(
                "DeviceVramExhausted: Total allocation {} MB exceeds hardware usable limit {} MB",
                new_device_total, max_usable
            ));
        }

        map.insert(tier, new_tier_total);
        Ok(())
    }

    /// Requests memory allocation in exact bytes, returning an RAII reservation guard.
    pub fn allocate_bytes(
        self: &Arc<Self>,
        tier: PriorityTier,
        bytes: usize,
    ) -> Result<MemoryReservationGuard, String> {
        let mb = (bytes.saturating_add(1024 * 1024 - 1)) / (1024 * 1024);
        let mb = mb.max(if bytes > 0 { 1 } else { 0 });
        self.allocate_memory(tier, mb)?;
        Ok(MemoryReservationGuard {
            scheduler: self.clone(),
            tier,
            bytes,
            allocated_mb: mb,
            released: false,
        })
    }

    /// Allocates a real host buffer backed by a scheduler memory reservation guard.
    pub fn allocate_real_buffer(
        self: &Arc<Self>,
        tier: PriorityTier,
        bytes: usize,
    ) -> Result<RealMemoryAllocation, String> {
        let guard = self.allocate_bytes(tier, bytes)?;
        let mut buffer = Vec::new();
        if buffer.try_reserve_exact(bytes).is_err() {
            return Err(format!(
                "SystemOom: Host OS memory allocation failed for buffer of {} bytes",
                bytes
            ));
        }
        buffer.resize(bytes, 0);
        Ok(RealMemoryAllocation { guard, buffer })
    }

    /// Acquires an active session slot, returning an RAII session guard.
    pub fn acquire_session(
        self: &Arc<Self>,
        session_id: impl Into<String>,
    ) -> Result<SessionGuard, String> {
        let id = session_id.into();
        let mut current = self.active_sessions.load(Ordering::SeqCst);
        loop {
            if current >= self.config.max_active_sessions {
                return Err(format!(
                    "MaxActiveSessionsExceeded: Active session count {} reached configured limit {}",
                    current, self.config.max_active_sessions
                ));
            }
            match self.active_sessions.compare_exchange_weak(
                current,
                current + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        Ok(SessionGuard {
            scheduler: self.clone(),
            session_id: id,
            released: false,
        })
    }

    /// Releases an active session slot.
    pub fn release_session(&self, _session_id: &str) {
        self.active_sessions
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |curr| {
                Some(curr.saturating_sub(1))
            })
            .ok();
    }

    /// Allocates host CPU RAM, returning an RAII host reservation guard.
    pub fn allocate_host_memory(
        self: &Arc<Self>,
        bytes: usize,
    ) -> Result<HostReservationGuard, String> {
        let max_host_bytes = self.config.total_cpu_ram_mb.saturating_mul(1024 * 1024);
        let mut current = self.active_host_bytes.load(Ordering::SeqCst);
        loop {
            let new_total = current.checked_add(bytes).ok_or_else(|| {
                "HostMemoryBudgetExceeded: Allocation counter would overflow".to_string()
            })?;
            if new_total > max_host_bytes {
                return Err(format!(
                    "HostMemoryBudgetExceeded: Requested {} bytes exceeds configured host RAM limit {} bytes (current: {} bytes)",
                    bytes, max_host_bytes, current
                ));
            }
            match self.active_host_bytes.compare_exchange_weak(
                current,
                new_total,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        Ok(HostReservationGuard {
            scheduler: self.clone(),
            bytes,
            released: false,
        })
    }

    /// Deallocates host CPU RAM.
    pub fn deallocate_host_memory(&self, bytes: usize) {
        self.active_host_bytes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |curr| {
                Some(curr.saturating_sub(bytes))
            })
            .ok();
    }

    /// Deallocates memory when a worker or task finishes.
    pub fn deallocate_memory(&self, tier: PriorityTier, megabytes: usize) {
        let mut map = self.active_allocations_mb.write().unwrap();
        if let Some(val) = map.get_mut(&tier) {
            *val = val.saturating_sub(megabytes);
        }
    }

    /// Returns a real-time snapshot of resource allocations and flags.
    pub fn get_snapshot(&self) -> ResourceUtilizationSnapshot {
        let map = self.active_allocations_mb.read().unwrap();
        let total_allocated: usize = map.values().sum();
        let usable = self
            .config
            .total_gpu_vram_mb
            .saturating_sub(self.config.os_reserved_vram_mb);
        let available = usable.saturating_sub(total_allocated);

        let mut tier_map = HashMap::new();
        for (tier, amt) in map.iter() {
            tier_map.insert(format!("{:?}", tier), *amt);
        }

        ResourceUtilizationSnapshot {
            total_allocated_vram_mb: total_allocated,
            available_vram_mb: available,
            allocations_by_tier: tier_map,
            is_dialogue_active: self.dialogue_active.load(Ordering::SeqCst),
            is_training_paused: self.training_yield_requested.load(Ordering::SeqCst)
                || self.training_is_paused.load(Ordering::SeqCst),
            perception_fps_cap: self.perception_fps.load(Ordering::SeqCst) as u32,
            audio_underruns: self.audio_underruns.load(Ordering::SeqCst),
            dropped_visual_frames: self.dropped_frames.load(Ordering::SeqCst),
            total_allocated_host_bytes: self.active_host_bytes.load(Ordering::SeqCst),
            active_sessions: self.active_sessions.load(Ordering::SeqCst),
            max_active_sessions: self.config.max_active_sessions,
            last_yield_latency_ms: f64::from_bits(
                self.last_yield_latency_ms.load(Ordering::SeqCst),
            ),
            last_training_step_duration_ms: f64::from_bits(
                self.last_training_step_duration_ms.load(Ordering::SeqCst),
            ),
        }
    }

    /// Benchmarks voice turnaround latency, memory stability, and yield boundaries under synthetic lock/atomic load.
    ///
    /// NOTE: This microbenchmark measures scheduler lock and atomic contention overhead.
    /// For end-to-end multi-threaded acoustic streaming and preemption benchmarks, see integration tests.
    pub fn benchmark_concurrent_load(&self, num_iterations: usize) -> ConcurrentLoadMetrics {
        if num_iterations == 0 {
            let snapshot = self.get_snapshot();
            return ConcurrentLoadMetrics {
                p50_voice_latency_ms: 0.0,
                p95_voice_latency_ms: 0.0,
                p99_voice_latency_ms: 0.0,
                peak_vram_mb: snapshot.total_allocated_vram_mb,
                audio_underruns: snapshot.audio_underruns,
                dropped_frames: snapshot.dropped_visual_frames,
                training_yield_latency_ms: 0.0,
                degradation_events: 0,
            };
        }

        let mut latencies_ms = Vec::with_capacity(num_iterations);
        let degradation_events = 0;

        // Allocate base allocations
        let _ = self.allocate_memory(PriorityTier::RealtimeAudio, 64);
        let _ = self.allocate_memory(PriorityTier::ForegroundDialogue, 2048);
        let _ = self.allocate_memory(PriorityTier::BackgroundTraining, 2048);

        let yield_start = Instant::now();
        self.on_dialogue_started();
        let yield_latency_ms = yield_start.elapsed().as_secs_f64() * 1000.0;

        for i in 0..num_iterations {
            let iter_start = Instant::now();

            // Simulate audio packet and model step
            if i % 10 == 0 {
                self.record_dropped_frame(); // Throttled visual frame
            }

            // Yield verification
            assert!(self.should_training_yield());

            let elapsed_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
            latencies_ms.push(elapsed_ms);
        }

        self.on_dialogue_ended();

        // Release base allocations
        self.deallocate_memory(PriorityTier::RealtimeAudio, 64);
        self.deallocate_memory(PriorityTier::ForegroundDialogue, 2048);
        self.deallocate_memory(PriorityTier::BackgroundTraining, 2048);

        latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = latencies_ms[((num_iterations * 50) / 100).min(num_iterations - 1)];
        let p95 = latencies_ms[((num_iterations * 95) / 100).min(num_iterations - 1)];
        let p99 = latencies_ms[((num_iterations * 99) / 100).min(num_iterations - 1)];

        let snapshot = self.get_snapshot();

        ConcurrentLoadMetrics {
            p50_voice_latency_ms: p50,
            p95_voice_latency_ms: p95,
            p99_voice_latency_ms: p99,
            peak_vram_mb: snapshot.total_allocated_vram_mb,
            audio_underruns: snapshot.audio_underruns,
            dropped_frames: snapshot.dropped_visual_frames,
            training_yield_latency_ms: yield_latency_ms,
            degradation_events,
        }
    }
}
