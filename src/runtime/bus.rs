//! Bounded Timestamped Event Bus and Non-Blocking Real-Time Queues.
//!
//! Enforces:
//! - Strict queue bounds preventing unbounded memory growth.
//! - Non-blocking execution for audio callbacks (`try_push` without blocking or dynamic reallocation).
//! - Overflow policies (`DropOldest`, `DropNewest`, `Reject`) with explicit atomic drop counters.
//! - Per-source sequence verification and gap detection.
//! - Hierarchical cancellation tokens.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::runtime::event::RuntimeEvent;

/// Policy applied when an event queue exceeds its bounded capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Discards the oldest event in the queue to make room for the new one (ideal for video/sensors).
    DropOldest,
    /// Discards the newly incoming event (ideal for batch jobs under backpressure).
    DropNewest,
    /// Rejects the event and returns an overflow error without modifying queue state.
    Reject,
}

/// Errors returned by bus and queue operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BusError {
    QueueFull { capacity: usize },
    QueueClosed,
    EventExpired { event_id: u64, deadline_ns: u64 },
    Cancelled,
}

impl std::fmt::Display for BusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BusError::QueueFull { capacity } => {
                write!(f, "Queue reached maximum capacity {capacity}")
            }
            BusError::QueueClosed => write!(f, "Queue is closed"),
            BusError::EventExpired {
                event_id,
                deadline_ns,
            } => {
                write!(
                    f,
                    "Event #{event_id} expired before dispatch (deadline {deadline_ns})"
                )
            }
            BusError::Cancelled => write!(f, "Operation cancelled"),
        }
    }
}

impl std::error::Error for BusError {}

/// Sequence discrepancy detected in per-source event stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SequenceDiscrepancy {
    /// Missing events between last received and current event.
    Gap {
        source: String,
        expected: u64,
        actual: u64,
    },
    /// Out-of-order or duplicate event arrived late.
    LateOrDuplicate {
        source: String,
        expected: u64,
        actual: u64,
    },
}

/// Thread-safe bounded FIFO queue with overflow policies and non-blocking operations.
pub struct BoundedEventQueue {
    capacity: usize,
    queue: Mutex<VecDeque<RuntimeEvent>>,
    dropped_count: AtomicU64,
    enqueued_count: AtomicU64,
    closed: AtomicBool,
}

impl BoundedEventQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            queue: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
            dropped_count: AtomicU64::new(0),
            enqueued_count: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    pub fn enqueued_count(&self) -> u64 {
        self.enqueued_count.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Non-blocking push designed specifically for real-time audio callbacks.
    /// Never blocks waiting for locks; if contention or overflow occurs, it safely drops
    /// or returns an error without stalling the audio engine.
    pub fn try_push(&self, event: RuntimeEvent, policy: OverflowPolicy) -> Result<(), BusError> {
        if self.is_closed() {
            return Err(BusError::QueueClosed);
        }

        // Try lock without blocking
        let mut q = match self.queue.try_lock() {
            Ok(guard) => guard,
            Err(_) => {
                // Mutex is contended; drop to protect audio thread deadline
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
                return Err(BusError::QueueFull {
                    capacity: self.capacity,
                });
            }
        };

        if q.len() >= self.capacity {
            match policy {
                OverflowPolicy::DropOldest => {
                    q.pop_front();
                    self.dropped_count.fetch_add(1, Ordering::Relaxed);
                    q.push_back(event);
                    self.enqueued_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                OverflowPolicy::DropNewest => {
                    self.dropped_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                OverflowPolicy::Reject => {
                    self.dropped_count.fetch_add(1, Ordering::Relaxed);
                    Err(BusError::QueueFull {
                        capacity: self.capacity,
                    })
                }
            }
        } else {
            q.push_back(event);
            self.enqueued_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Blocking push that acquires mutex normally, adhering to the specified overflow policy.
    pub fn push(&self, event: RuntimeEvent, policy: OverflowPolicy) -> Result<(), BusError> {
        if self.is_closed() {
            return Err(BusError::QueueClosed);
        }

        let mut q = self.queue.lock().map_err(|_| BusError::QueueClosed)?;
        if q.len() >= self.capacity {
            match policy {
                OverflowPolicy::DropOldest => {
                    q.pop_front();
                    self.dropped_count.fetch_add(1, Ordering::Relaxed);
                    q.push_back(event);
                    self.enqueued_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                OverflowPolicy::DropNewest => {
                    self.dropped_count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                OverflowPolicy::Reject => {
                    self.dropped_count.fetch_add(1, Ordering::Relaxed);
                    Err(BusError::QueueFull {
                        capacity: self.capacity,
                    })
                }
            }
        } else {
            q.push_back(event);
            self.enqueued_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Pops the next event from the queue.
    pub fn pop(&self) -> Option<RuntimeEvent> {
        let mut q = self.queue.lock().ok()?;
        q.pop_front()
    }

    /// Drains all currently queued events.
    pub fn drain(&self) -> Vec<RuntimeEvent> {
        let q = self
            .queue
            .lock()
            .map(|mut g| g.drain(..).collect())
            .unwrap_or_default();
        q
    }
}

/// Hierarchical cancellation token supporting cooperative worker interruption.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    parent: Option<Arc<AtomicBool>>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            parent: None,
        }
    }

    pub fn child_token(&self) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            parent: Some(self.cancelled.clone()),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        if self.cancelled.load(Ordering::SeqCst) {
            return true;
        }
        if let Some(parent) = &self.parent {
            if parent.load(Ordering::SeqCst) {
                return true;
            }
        }
        false
    }
}

/// Central high-integrity event bus coordinating perception, inference, and tools.
pub struct EventBus {
    next_event_id: AtomicU64,
    source_sequences: RwLock<HashMap<String, u64>>,
    subscribers: RwLock<HashMap<String, Vec<Arc<BoundedEventQueue>>>>,
    global_cancellation: CancellationToken,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            next_event_id: AtomicU64::new(1),
            source_sequences: RwLock::new(HashMap::new()),
            subscribers: RwLock::new(HashMap::new()),
            global_cancellation: CancellationToken::new(),
        }
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.global_cancellation.child_token()
    }

    pub fn cancel_all(&self) {
        self.global_cancellation.cancel();
    }

    /// Generates next globally unique event ID.
    pub fn next_event_id(&self) -> u64 {
        self.next_event_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Allocates the next strictly monotonic sequence number for a given source.
    pub fn next_sequence_for_source(&self, source: &str) -> u64 {
        let mut map = self.source_sequences.write().unwrap();
        let counter = map.entry(source.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Validates an incoming event against the expected per-source sequence,
    /// returning any gap or out-of-order discrepancy.
    pub fn validate_sequence(&self, event: &RuntimeEvent) -> Option<SequenceDiscrepancy> {
        let map = self.source_sequences.read().unwrap();
        let expected = map.get(&event.source_id).copied().unwrap_or(0);
        if event.sequence_number > expected + 1 {
            Some(SequenceDiscrepancy::Gap {
                source: event.source_id.clone(),
                expected: expected + 1,
                actual: event.sequence_number,
            })
        } else if event.sequence_number <= expected {
            Some(SequenceDiscrepancy::LateOrDuplicate {
                source: event.source_id.clone(),
                expected: expected + 1,
                actual: event.sequence_number,
            })
        } else {
            None
        }
    }

    /// Subscribes a bounded queue to an event topic (e.g. "audio", "video", "model", "tools").
    pub fn subscribe(&self, topic: &str, capacity: usize) -> Arc<BoundedEventQueue> {
        let queue = Arc::new(BoundedEventQueue::new(capacity));
        let mut subs = self.subscribers.write().unwrap();
        subs.entry(topic.to_string())
            .or_default()
            .push(queue.clone());
        queue
    }

    /// Publishes an event to all subscriber queues for the specified topic.
    pub fn publish(
        &self,
        topic: &str,
        event: RuntimeEvent,
        policy: OverflowPolicy,
    ) -> Result<(), BusError> {
        if self.global_cancellation.is_cancelled() {
            return Err(BusError::Cancelled);
        }

        let subs = self.subscribers.read().unwrap();
        if let Some(queues) = subs.get(topic) {
            for q in queues {
                let _ = q.push(event.clone(), policy);
            }
        }
        Ok(())
    }
}
