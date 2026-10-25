# RX 7800 XT verification and measured results

Measured on 2026-10-24 using native Windows 11 (build 26200), Ryzen 5 7600X,
RX 7800 XT 16 GB, WGPU/Vulkan, AMD driver 26.3.1 (LLPC), Burn 0.16.1, f32,
release builds. [Raw benchmark evidence](benchmarks/rx7800xt_training_and_inference.json)
contains arguments, configurations, binary/data hashes, validation results and logs.
[Reproduction commands](GPU_TRAINING.md) preserve existing runs.

## Correctness and recovery

- Full CPU release suite: **40 passed**.
- Native Windows GPU library, reference, training-loop and checkpoint checks:
  **11 passed**. Tests compare logits, every parameter gradient and held-out
  validation for Mamba/attention/MoE configurations. Both scan strategies and
  observation paths are exercised. Actual regular and streaming loops start from
  common CPU weights and compare updated outputs and validation.
- Recurrent GPU logits also match CPU sequence logits at every token for lengths
  7, 32 and 128, covering SSM state, convolution state and attention KV updates.
- Device clipping matches Burn's clipping for zero, small and large norms,
  including zero thresholds. Four-update regular/streaming runs also compare
  against Burn's optimizer path; the updated-logit tolerance is 2e-5.
- Regular and streaming resume match uninterrupted training after restoring
  optimizer and sampler RNG. Streaming additionally restores replay, ingested-file
  history and frozen validation. Unpublished incomplete generations are ignored.
- Concurrent GPU inference reads complete `.sea` versions while training publishes
  replacements. An injected interruption before atomic rename preserves the prior
  file and cleans the temporary file; subsequent publication succeeds.
- CPU and GPU generated **identical greedy text** for 128 new tokens from the
  same trained `.sea` artifact in separate native CLI processes.

These checks establish behavior on the tested configurations. The larger run below
completed 200 guarded updates with finite validation loss 2.76344; it is not a
claim about arbitrary future datasets, model sizes or training durations.

## Training throughput and memory

Small workload: **380,033 parameters**, batch 8, sequence 64, width 64, 3 layers
`[M A M]`, state 8, 4 experts/top-1, TinyShakespeare, 60 updates, 8 validation
windows, final checkpoint, no generated samples. Full visualization captures every
parameter and gradient value each update plus activation probes.

| Workload | Warm tok/s | End-to-end tok/s | Sampled dedicated GPU MiB |
| --- | ---: | ---: | ---: |
| Native CPU, default clipping | 1,697 | 1,605 | 0 |
| GPU, original clipping | 3,827 | 3,341 | 630–647 |
| GPU, `--device-clip` | 13,298 | 8,408 | 806–811 |
| GPU, device clipping + full visualization every step | 1,678 | 1,560 | 836 |
| GPU streaming, device clipping | 12,631 | 7,834 | 779 |
| GPU streaming + full visualization every step | 1,749 | 1,605 | 807 |
| Larger GPU workload: B32 L128, 4 layers, state 16, 200 updates | 26,646 | 23,754 | 3,794 |

The clipping comparison uses the median of three independent runs per path. Other
rows are single runs. Warm tok/s is the mean of five-update metric windows after
the first window. End-to-end includes process startup, evaluation and final saves;
it uses observer wall time, with polling resolution up to approximately 250 ms
plus counter-query latency. Training benchmarks use matching configurations and
seed with each backend's initialization; correctness tests use shared weights.

The measured optimization improves warm throughput **3.47×** over the original
GPU clipping path, and **7.84×** over the native CPU baseline on the small workload.
The larger workload reached the experimental 20,000 tok/s target in this run.
Full visualization every update reduced warm throughput by **87%** for regular
training and **86%** for streaming. Increase `--viz-every` when throughput matters.

Memory values are sampled Windows process dedicated allocations, including runtime
pools, not exact peak tensor storage or total board use. Shared allocations and
individual samples remain in each local benchmark directory. Other applications,
driver caches, and shared-host load can change timings and allocation counts.

## Measured bottleneck and optimization

Synchronized stage profiling found that Burn 0.16's per-parameter norm clipping
performed a host scalar read for every parameter tensor. The optimizer averaged
**103.29 ms**, **73.18%** of warm step time. `--device-clip` computes the same scale
on-device, then supplies the clipped gradients to AdamW with its internal clipping
disabled. It preserves per-tensor clipping semantics and existing numerical guards.
This uses the established clipping formula; it is not a new learning algorithm.

Optimizer time fell to **9.03 ms**; synchronized full-step time fell from **141.15
to 43.68 ms**. See the [baseline](benchmarks/rx7800xt_baseline_profile.json) and
[optimized](benchmarks/rx7800xt_optimized_profile.json) stage records. These use
steps 6–19 of 20-update runs and exclude final checkpoint work. Profiling inserts
GPU synchronization and is deliberately separate from throughput measurements.

## Inference and remaining limits

For this small model, native CPU single-sequence decoding is faster:

| Same artifact, prompt `ROMEO:`, 128 greedy new tokens | Decode tok/s | Prefill ms | Process tok/s |
| --- | ---: | ---: | ---: |
| CPU | 851.0 | 3.3 | 418.5 |
| GPU | 98.5 | 444.5 | 59.8 |

Use GPU for the measured batched training workloads and CPU for this small
single-sequence inference workload. GPU launch/transfer overhead matters at this
scale; throughput is not universally higher just because GPU execution is enabled.
The timings include cold process behavior and do not establish larger-model or
batched-inference performance.

The standalone CubeCL scan definition is not integrated into autodiff training;
these results use Burn tensor operations. MoE routing currently computes all
expert outputs before masking them, so active-parameter counts do not imply sparse
execution speedups. Checkpoint generations are retained and require disk space.
Source/tokenizer/replay settings must remain consistent for exact continuation;
legacy checkpoints cannot restore state that was never saved. Other GPUs and CUDA
have not been verified by this work.

[The foundational alignment rule](../ALIGNMENT.md) applies permanently: no
autonomous recursive self-improvement, with no exceptions. All optimization here
is an implementation change made under the user's direction.
