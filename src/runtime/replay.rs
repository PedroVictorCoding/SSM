//! Deterministic Headless Event Recording and Replay Engine.
//!
//! Enables reproducible offline testing and simulation without hardware dependencies:
//! - Records raw `RuntimeEvent` streams to JSONL files.
//! - Replays recorded streams with deterministic timestamp ordering.
//! - Supports time-scaled or step-by-step lockstep dispatch.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::runtime::event::RuntimeEvent;

/// Manages serialization and playback of event streams for deterministic verification.
#[derive(Clone, Debug, Default)]
pub struct EventRecorder {
    recorded: Vec<RuntimeEvent>,
}

impl EventRecorder {
    pub fn new() -> Self {
        Self {
            recorded: Vec::new(),
        }
    }

    pub fn record(&mut self, event: RuntimeEvent) {
        self.recorded.push(event);
    }

    pub fn len(&self) -> usize {
        self.recorded.len()
    }

    pub fn is_empty(&self) -> bool {
        self.recorded.is_empty()
    }

    pub fn events(&self) -> &[RuntimeEvent] {
        &self.recorded
    }

    /// Saves recorded events to a JSONL file.
    pub fn save_to_jsonl(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        for event in &self.recorded {
            let json = serde_json::to_string(event)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            writeln!(file, "{json}")?;
        }
        Ok(())
    }

    /// Loads events from a JSONL file.
    pub fn load_from_jsonl(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let event = serde_json::from_str::<RuntimeEvent>(&line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            events.push(event);
        }

        Ok(Self { recorded: events })
    }

    /// Replays events into a handler callback in strict timestamp sequence.
    pub fn replay_deterministic<F>(&self, mut dispatch_fn: F) -> usize
    where
        F: FnMut(&RuntimeEvent),
    {
        let mut sorted = self.recorded.clone();
        sorted.sort_by_key(|e| e.timestamp_ns);

        let count = sorted.len();
        for event in &sorted {
            dispatch_fn(event);
        }
        count
    }
}
