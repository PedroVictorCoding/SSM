# RX 7800 XT reliability and performance goal

Binding constraint: [ALIGNMENT.md](../ALIGNMENT.md). No autonomous recursive self-improvement, ever, with no exceptions. This work improves the human-defined training implementation; runtime learning remains context-dependent training on human-approved external data.

Objective: reliable GPU regular/streaming training and inference, verified CPU/GPU
agreement, reproducible profiling and memory measurements, a measured optimization,
checkpoint recovery, and concurrent live inference. 20,000 tok/s is an experimental
target, not an acceptance gate. Existing user checkpoints must remain untouched.

## Acceptance evidence to collect

- [x] CPU/GPU full-model outputs, parameter gradients, and validation losses;
      odd lengths, batched sequences, Mamba/attention/MoE, observed and unobserved paths.
- [x] Regular and streaming loops apply the same numerical guards and telemetry;
      streaming validation uses held-out data excluded from replay.
- [x] Checkpoint recovery restores consistent model/optimizer/step state;
      replay/RNG continuation semantics are tested and documented.
- [x] Live .sea readers succeed while streaming training publishes new versions,
      including native Windows execution and injected publication failures.
- [x] End-to-end regular/streaming throughput, GPU memory samples, visualization
      overhead, and stage profiling with reproducible workload and environment metadata.
- [x] Largest measured bottleneck optimized; before/after timings and numerical
      equivalence tests cover the changed behavior.
- [x] GPU selection is observable; commands select the dedicated AMD device rather
      than a CPU fallback or integrated graphics.
- [x] Current test results, benchmark artifacts, commands and limits documented.

## Initial audit

The following describes the starting state, before the reliability changes below.

The earlier forward-only GPU regression exists and layout workarounds are present.
Streaming training currently skips finite checks, validation and visualization;
its CSV schema does not match the dashboard. Restore/export errors are silently
ignored. `.sea` publication uses a shared temporary filename and keeps the writer
file open across rename, which needs native Windows testing. The advertised CubeCL
scan dispatcher currently calls its CPU slice fallback and is not integrated into
the autodiff Mamba training path. None of these observations proves the complete goal.

Work artifacts live under dedicated test/benchmark directories, never existing runs.

## Verified evidence

Completed evidence and limitations: [GPU_RESULTS.md](GPU_RESULTS.md).
Copy-paste PowerShell commands: [GPU_TRAINING.md](GPU_TRAINING.md).
Raw reproducible benchmark metadata, configuration and logs:
[`benchmarks/rx7800xt_training_and_inference.json`](benchmarks/rx7800xt_training_and_inference.json).
The final recurrent GPU reference rerun also passed after adding per-token
CPU/GPU comparisons for sequence lengths 7, 32 and 128.

Current checks: full CPU release suite **40 passed**. Native Windows
`--no-default-features --features wgpu,ndarray --lib --test test_gpu_reference
--test test_gpu_training --test test_checkpoint_reliability`: **11 passed**.
The loop tests compare final updated logits and held-out validation from common
initial weights for regular and streaming training, including comparison of the
new clipping path with Burn's original clipping. The library tests include an
injected failure immediately before atomic replacement and successful publication
afterward. All test checkpoints are isolated from existing user runs.

Baseline synchronized RX 7800 XT profiling (380,033 parameters, batch 8, sequence
64, width 64, 3 layers, 4 experts/top-1) found optimizer updates averaged
103.29 ms, **73.18%** of warm step time. See
[`benchmarks/rx7800xt_baseline_profile.json`](benchmarks/rx7800xt_baseline_profile.json).
The measured bottleneck motivated `--device-clip`, which preserves per-parameter
norm clipping while removing each norm's host scalar read. Three profiling-disabled
runs per path measured a 3.47× median warm throughput improvement (3,827 to 13,298
tok/s). Optimizer time fell to 9.03 ms. A larger 200-update run reached 26,646 warm
and 23,754 end-to-end tok/s, with finite validation loss 2.76344.

- CPU release tests passed for checkpoint reliability (3), continual learning (1),
  checkpoint resume (1), and streaming ingestion (3). The reliability cases compare
  interrupted and uninterrupted regular/streaming runs, restore replay and sampler
  state, and read complete `.sea` artifacts concurrently with training publication.
- The full-model GPU reference compares parameter gradients and validation as
  well as forward logits; the current native Windows rerun passed.
- `--profile` now records synchronized stage timings to `profile.jsonl` for both
  training loops. It is disabled by default; throughput measurements must run with
  profiling disabled because stage synchronization changes scheduling.
- `scripts/benchmark_gpu.ps1` launches isolated native runs and captures logs,
  configuration, binary hash, throughput windows and sampled Windows process GPU
  memory counters. `scripts/summarize_profile.py` summarizes warmed stage timings.
- Native WGPU initialization logs the selected physical adapter and driver.

Full visualization every update reduced warm throughput by 86–87%. GPU execution
is correct for the tested inference paths, but CPU is faster for this small model's
single-sequence decode (851 versus 98.5 tok/s); the documentation recommends CPU
for that workload. No universal GPU speedup or untested-backend claim is made.
