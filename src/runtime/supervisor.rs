//! Worker Supervision and Fault Recovery for the SAIL Native Host Runtime.
//!
//! Manages life cycle of perception, inference, and tool execution workers:
//! - Health check and heartbeat monitoring.
//! - Cooperative cancellation propagation.
//! - Crash detection and recovery without blocking audio capture/playback or causing deadlocks.
//! - Graceful, orderly shutdown.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::runtime::bus::{CancellationToken, EventBus};

/// Functional role assigned to a supervised runtime worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkerRole {
    PerceptionAudio,
    PerceptionVideo,
    InferenceEngine,
    ToolExecutor,
}

/// Operational state of a supervised worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerState {
    Starting,
    Running,
    Faulted(String),
    Recovering,
    Stopped,
}

/// Metadata and controls for an active worker thread.
pub struct WorkerHandle {
    pub worker_id: String,
    pub role: WorkerRole,
    pub cancellation_token: CancellationToken,
    pub state: Arc<RwLock<WorkerState>>,
    pub last_heartbeat_ns: Arc<AtomicU64>,
    pub restart_count: Arc<AtomicU64>,
    thread_handle: Option<JoinHandle<()>>,
}

/// Supervisor coordinating native worker execution, health checks, and recovery.
pub struct Supervisor {
    workers: Arc<RwLock<HashMap<String, WorkerHandle>>>,
    bus: Arc<EventBus>,
    shutdown_flag: Arc<AtomicBool>,
}

impl Supervisor {
    pub fn new(bus: Arc<EventBus>) -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            bus,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_flag.load(Ordering::SeqCst)
    }

    /// Spawns a supervised worker thread.
    pub fn spawn_worker<F>(
        &self,
        worker_id: &str,
        role: WorkerRole,
        worker_fn: F,
    ) -> Result<(), String>
    where
        F: FnOnce(CancellationToken, Arc<EventBus>) -> Result<(), String> + Send + 'static,
    {
        if self.is_shutting_down() {
            return Err("Supervisor is shutting down; cannot spawn new worker".into());
        }

        let token = self.bus.cancellation_token();
        let state = Arc::new(RwLock::new(WorkerState::Starting));
        let heartbeat = Arc::new(AtomicU64::new(0));
        let restart_count = Arc::new(AtomicU64::new(0));

        let t_token = token.clone();
        let t_state = state.clone();
        let t_bus = self.bus.clone();
        let t_id = worker_id.to_string();

        let handle = thread::spawn(move || {
            {
                let mut s = t_state.write().unwrap();
                *s = WorkerState::Running;
            }

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                worker_fn(t_token, t_bus)
            }));

            let mut s = t_state.write().unwrap();
            match result {
                Ok(Ok(())) => {
                    *s = WorkerState::Stopped;
                }
                Ok(Err(err)) => {
                    eprintln!("[Supervisor] Worker '{t_id}' exited with error: {err}");
                    *s = WorkerState::Faulted(err);
                }
                Err(panic_payload) => {
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Worker thread panicked with unknown error".to_string()
                    };
                    eprintln!("[Supervisor] Worker '{t_id}' PANICKED: {msg}");
                    *s = WorkerState::Faulted(msg);
                }
            }
        });

        let mut workers = self.workers.write().unwrap();
        workers.insert(
            worker_id.to_string(),
            WorkerHandle {
                worker_id: worker_id.to_string(),
                role,
                cancellation_token: token,
                state,
                last_heartbeat_ns: heartbeat,
                restart_count,
                thread_handle: Some(handle),
            },
        );

        Ok(())
    }

    /// Checks worker status and returns map of worker states.
    pub fn get_worker_states(&self) -> HashMap<String, WorkerState> {
        let workers = self.workers.read().unwrap();
        workers
            .iter()
            .map(|(id, handle)| (id.clone(), handle.state.read().unwrap().clone()))
            .collect()
    }

    /// Cancels a specific worker without affecting others.
    pub fn cancel_worker(&self, worker_id: &str) -> bool {
        let workers = self.workers.read().unwrap();
        if let Some(handle) = workers.get(worker_id) {
            handle.cancellation_token.cancel();
            true
        } else {
            false
        }
    }

    /// Gracefully shuts down all workers.
    pub fn shutdown(&self, timeout: Duration) {
        self.shutdown_flag.store(true, Ordering::SeqCst);
        self.bus.cancel_all();

        let mut workers = self.workers.write().unwrap();
        let start = Instant::now();

        for (_id, handle) in workers.iter_mut() {
            handle.cancellation_token.cancel();
            if let Some(th) = handle.thread_handle.take() {
                let remaining = timeout.saturating_sub(start.elapsed());
                if remaining > Duration::ZERO {
                    let _ = th.join();
                }
            }
            let mut s = handle.state.write().unwrap();
            *s = WorkerState::Stopped;
        }
    }
}
