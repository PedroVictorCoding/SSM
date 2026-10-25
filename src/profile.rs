//! Opt-in synchronized stage timings. Profiling changes scheduling; benchmark
//! throughput separately with profiling disabled.
use burn::tensor::backend::Backend;
use std::{fs::File, io::Write, path::Path, time::Instant};

pub(crate) struct StepProfiler {
    file: Option<File>,
    last: Instant,
    step: usize,
    stages: Vec<(&'static str, f64)>,
}

impl StepProfiler {
    pub fn new(dir: &Path, enabled: bool) -> std::io::Result<Self> {
        Ok(Self {
            file: if enabled {
                Some(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(dir.join("profile.jsonl"))?,
                )
            } else {
                None
            },
            last: Instant::now(),
            step: 0,
            stages: Vec::new(),
        })
    }
    pub fn begin<B: Backend>(&mut self, step: usize, device: &B::Device) {
        if self.file.is_none() {
            return;
        }
        B::sync(device);
        self.last = Instant::now();
        self.step = step;
        self.stages.clear();
    }
    pub fn stage<B: Backend>(&mut self, name: &'static str, device: &B::Device) {
        if self.file.is_none() {
            return;
        }
        B::sync(device);
        self.stages
            .push((name, self.last.elapsed().as_secs_f64() * 1000.0));
        self.last = Instant::now();
    }
    pub fn finish(&mut self) -> std::io::Result<()> {
        if let Some(file) = &mut self.file {
            let stages: std::collections::BTreeMap<_, _> = self.stages.iter().copied().collect();
            writeln!(
                file,
                "{}",
                serde_json::json!({"step": self.step, "stages_ms": stages})
            )?;
            file.flush()?;
        }
        Ok(())
    }
}
