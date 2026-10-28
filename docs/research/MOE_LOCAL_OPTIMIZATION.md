# MoE and local execution research plan

Status: proposed experiments and independent implementation review, 2026-10-25.
No new optimization or paper speedup is claimed as reproduced in SAIL.
Follow [ALIGNMENT.md](../../ALIGNMENT.md) permanently, with no exceptions: all
configuration, experiments, external training data and promotion are human-defined.
Runtime cache placement is host resource management, not model-controlled learning.

## User-supplied thesis

[Jan Ludziejewski, *Improving Performance of Mixture of Experts Large Language Models*](https://repozytorium.uw.edu.pl/server/api/core/bitstreams/44fbb658-4a84-4c15-9ddd-521aaabddf83/content),
University of Warsaw, version 1.0.1, September 2025; repository cover dated 2026.
The downloaded document has 123 PDF pages. Its identity, hash, reviewed sections and
related primary sources are in [the source manifest](moe_research_sources.json).
The thesis points to [the authors' code](https://github.com/llm-random/llm-random).

| Thesis topic | Location: printed pages / PDF pages | Implication for SAIL experiments |
| --- | --- | --- |
| Fine-grained experts | Chapter 2, pp. 22–35 / PDF 23–36 | Vary expert width and number selected together; measure routing overhead as well as quality. |
| Joint compute and memory scaling | Chapter 3, pp. 37–47 / PDF 38–48 | Choose total capacity, active capacity and token budget jointly under measured memory limits. |
| Decoupled relative learning-rate schedules (RLRS) | Chapter 4, pp. 55–67 / PDF 56–68 | Test different optimizer schedules for router, experts and other components. |
| Mixture of Tokens (MoT) | Chapter 5, pp. 69–80 / PDF 70–81 | Research cross-example aggregation separately; batch-one inference and session isolation matter. |
| MoE-Mamba | Chapter 6, §§6.3–6.4, pp. 83–89 / PDF 84–90 | Compare SSM/MoE allocations and sequential placement against matched controls. |

The thesis reports up to 23% RLRS training acceleration and 2.35× fewer training
steps for a MoE-Mamba comparison. Those are convergence results under its experimental
conditions, not promises of faster SAIL kernels or local token generation. The
2.35× comparison uses substantially more total parameters at similar active capacity.
See also the original [RLRS paper](https://arxiv.org/abs/2507.03526) and
[MoE-Mamba paper](https://arxiv.org/abs/2401.04081v2).

## Current SAIL prerequisites

Inspection of [moe.rs](../../src/model/moe.rs),
[mmap_moe.rs](../../src/model/mmap_moe.rs),
[hybrid.rs](../../src/model/hybrid.rs), and
[the sparse tests](../../tests/test_moe_sparse.rs) establishes:

- `sparse_dispatch` skips an expert only when its weight is zero for **every** token
  in the batch. Every nonempty expert still executes over the full token matrix.
  It does not gather just the selected rows. When all E experts are used, expert
  rows processed are N×E, versus the desired N×K for N tokens and top-K routing.
  This is a computation-count comparison, not a measured wall-clock speedup.
- The skip decision reads a scalar through `into_data()` once per expert. On GPU,
  that can force synchronization. More experts can increase this cost.
- `HybridLmConfig::init` does not pass a sparse-dispatch option to `MoeConfig`;
  toggling it in an isolated test does not enable it in the ordinary trainer.
- The current equivalence test checks outputs, auxiliary loss and **router**
  gradients. It does not compare input gradients, every expert gradient or optimizer
  updates. Its CPU timing test does not establish native Windows GPU performance.
- `StreamingMoe` clears its entire expert cache at eight entries. It has no byte
  budget or locality-aware replacement. Loading parses bytes into vectors and
  constructs tensors; the mmap file alone does not make GPU uploads zero-copy.
- The normal hybrid model owns ordinary `Moe` blocks. Standalone `StreamingMoe`
  tests do not establish that `.sea` inference uses streamed experts end to end.

These are the first targets; fine-grained architecture changes on top of redundant
expert computation could make throughput worse.

## Ordered implementation requests

Record claims, owned files, measurements and review replies in
[LLM_HANDOFF.md](../../LLM_HANDOFF.md). Existing foundational boundary fixes retain
their priority. This research track can proceed as isolated benchmarks without
expanding tool permissions or changing default production training policy.

| ID | Order | Applies to | Deliverable / acceptance evidence |
| --- | --- | --- | --- |
| PERF-MOE-00 | First | Training + inference | Truthful baseline and routing/expert-work instrumentation. |
| PERF-MOE-01 | First implementation experiment | Training + inference | Actual token dispatch with output, gradient and optimizer parity. |
| PERF-MOE-02 | Next, independent of kernel experiments | Training convergence | Explicit component optimizer schedules, resumable and opt-in. |
| PERF-MOE-03 | After 00/01 | Training and model selection | Granularity, SSM/MoE allocation and memory-constrained architecture sweeps. |
| PERF-MOE-04 | After 00 and generation-correctness fixes | Local inference | CPU/GPU execution choice based on measured transfer and compute costs. |
| PERF-MOE-05 | After 04 | Local inference | Bounded locality-aware expert cache with optional prefetch. |
| PERF-MOE-06 | After exact baselines | Local inference | Calibrated INT8/INT4 experiment with quality and backend gates. |
| PERF-MOE-07 | After recurrent streaming correctness | Local inference | Exact speculative verification and state rollback prototype. |
| PERF-MOE-08 | Deferred architecture research | Offline training | Causal MoT experiment, isolated from live multi-user sessions. |

### PERF-MOE-00 — Baseline before optimization

Add a dedicated benchmark that records expert assignments, rows actually processed,
zero-work experts, routing/gather/scatter time, expert GEMM time, synchronization
count, transfers, peak memory and full-step/decode timing. Separate prefill, batch-one
decode and batched training. Include cases where every expert receives work so an
empty-expert optimization cannot masquerade as token dispatch.

Compare dense reference, current empty-expert skipping and proposed true dispatch
using identical weights/inputs. Correctness tests precede timed runs. Suggested
coverage: E={4,8,16}, K={1,2}; decoded N=1, prefill N={128,512}, training N={512,4096},
plus larger widths only if they fit a declared memory budget. This is a suggested
experiment grid, not a requirement to run a large sweep before basic validation.

### PERF-MOE-01 — Token dispatch and grouped computation

[MegaBlocks](https://arxiv.org/abs/2211.15841) reformulates dynamic MoE work using
block-sparse operations; [its implementation](https://github.com/databricks/megablocks)
is a reference, not an automatically compatible Burn/WGPU dependency.

Start with a correctness reference: route → bucket selected token/expert pairs →
gather rows → expert computation → weighted scatter-add. Then consider grouped
GEMMs/block-sparse kernels if profiling warrants them. Avoid per-expert host reads
in the GPU path; measure dispatch overhead and workload-dependent crossover points.

Acceptance: output and auxiliary-loss parity; input, router and **all expert**
gradient parity; one/multiple AdamW-update parity including inactive-expert weight
decay/moment behavior; top-1/top-2, imbalanced/empty experts, duplicate destinations
and threshold behavior. Preserve existing gate weights and normalization. Do not
quietly drop or pad away tokens to produce an apparent speedup. Make dispatch mode
explicit in model configuration, checkpoint/export metadata and the actual CLI
path, with the existing mode retained for reproducibility and rollback.

### PERF-MOE-02 — Relative optimizer schedules

Reproduce RLRS as an optimizer-group experiment, not a kernel speed claim. Start
with router/expert groups; define embedding, output head, attention and SSM groups
explicitly. The thesis does not establish tuned SSM multipliers. For component m,
use independently defined start/end learning rates and a cosine path between them;
the paper's constants are candidate seeds, not validated defaults for SAIL.

Acceptance: all multipliers 1 reproduce the baseline; every parameter belongs to
exactly one group; base schedule, warmup, minimum rate, optimizer state, decay and
resume remain defined and deterministic. **Do not implement this by scaling the
gradient alone:** Adam's moment normalization makes that different from changing
the update learning rate. Compare equal data/seeds and separately tuned base rates,
then report steps, tokens and **wall time to the same held-out quality target**,
including tuning cost. A different final loss at a fixed step count is insufficient
to claim a local end-to-end training speedup. Use finite-loss recovery gates.

### PERF-MOE-03 — Fine-grained and memory-matched models

Use [the granularity paper](https://arxiv.org/abs/2402.07871) and thesis Chapters 2/3/6.
For a baseline expert width H, expert count E and selected count K, test granularity
g∈{1,2,4} with H'=H/g, E'=E×g and K'=K×g where valid. This approximately preserves
expert-bank and active expert parameters; router size and actual runtime still grow.
Count those costs. In SAIL, top-1 and top-K use different weight normalization;
control or explicitly report that confound in architectural comparisons.

Also compare pure SSM, the current hybrid and sequential SSM+MoE allocations,
including the thesis's roughly balanced active SSM/MoE allocation as a hypothesis.
Run both active-compute-matched and total-memory-matched comparisons; those are
different controls. Include tokenizer, embeddings, attention, optimizer moments,
activations, replay, runtime state and temporary workspaces in the budget.

Acceptance: fixed human-approved data split, same tokenizer and token budget,
reported total/active/allocation counts, ≥3 seeds for retained candidates, held-out
quality versus wall time and peak measured RAM/VRAM. Do not extrapolate the thesis's
large-scale fitted coefficients or NVIDIA A100 optimum into guaranteed RX 7800 XT
results. Architecture changes are explicit new human-defined experiments.

### PERF-MOE-04 — Move activations when cheaper than weights

[Fiddler](https://arxiv.org/abs/2402.07033v3),
[official code](https://github.com/efeslab/fiddler), compares CPU execution of
nonresident experts with moving those experts to GPU. Port the placement idea
using SAIL CPU and WGPU backends; do not assume its implementation runs on AMD.

Measure both paths on this machine for each relevant expert shape/token count:
activation copy + CPU compute + result copy, versus weight upload + GPU compute.
Include synchronization and fixed transfer overhead; choose the cheaper path via
a frozen human-defined cost table. Maintain an all-CPU/all-GPU control. Mixed
backends must preserve routed weights, output order and model generation identity.

Acceptance: output parity, cold/warm measurements, real transfer-byte counters,
end-to-end prefill and batch-one decode latency. A memory-mapped file does not
eliminate host RAM/page-cache pressure, page faults or SSD bandwidth constraints.
This is initially inference-only; offloaded autodiff/optimizer state is separate work.

### PERF-MOE-05 — Bounded expert cache and prefetch

[MoE-Infinity](https://arxiv.org/abs/2401.14361v3),
[official code](https://github.com/EfficientMoE/MoE-Infinity), uses activation locality
to guide expert caching/prefetching. Start with byte-budgeted LRU and frequency
baselines before a more complicated trace-guided policy. Count cold misses and
prefetch pollution rather than reporting warm-cache throughput alone.

Acceptance: budget in bytes, stable generation+layer+expert+device keys, no eviction
of in-flight tensors, bounded asynchronous transfers and cancellation, clean
invalidation on model promotion, and correctness under concurrent inference.
Compare miss rate, transferred bytes, page faults, peak host/VRAM and p95 token
latency. Integrate into the actual artifact/engine path. Runtime routing traces may
guide cache residency under human-defined rules; they must not feed training or
automatically train a replacement model/predictor.

### PERF-MOE-06 — Activation-aware quantization

[AWQ](https://arxiv.org/abs/2306.00978),
[official code](https://github.com/mit-han-lab/llm-awq), is a candidate for low-bit
weight-only inference. Use independent, human-approved external calibration data;
do not use live assistant outputs as automatic training/calibration feedback.

Acceptance: versioned artifact metadata and compatible dequantization/matmul kernels,
declared quantization groups/scales, measured quality on held-out tasks and long
recurrent sequences, and actual load/memory/latency results. Consider keeping router,
SSM recurrence-sensitive parameters and normalization at higher precision. Packed
storage alone is not a fast quantized inference implementation. Preserve the
full-precision training checkpoint; a separate quantized export changes numerical
values and cannot be called lossless. Coordinate with PACK-01 in the handoff.

### PERF-MOE-07 — Exact speculative decoding

[Leviathan et al.](https://arxiv.org/abs/2211.17192v2) use draft proposals and target
verification with correction sampling to preserve the target output distribution.
The transformer result motivates an experiment; SAIL's recurrent state requires
additional rollback machinery. No upstream code implementation was verified here.

Acceptance: same tokenizer and sampling policy, greedy reference equality,
statistical sampling-distribution tests, correct rejection/residual probabilities,
and restoration of Mamba state, convolution history, attention state and RNG to the
accepted prefix. Preserve distribution, not necessarily token-for-token equality
under the same seed with a different sampling algorithm. Never emit/execute an
unverified draft tool call or speak rejected draft text. Measure draft cost,
acceptance rate, rollback memory and real latency before retaining the method.
Draft outputs remain inference intermediates and never automatically enter training.

### PERF-MOE-08 — MoT remains a separate research branch

The thesis's Chapter 5 groups same-position tokens across examples and notes
batch-one limitations. Do not mix unrelated user/session context or merge future
positions. An offline external-data experiment needs causality, no train/eval mixing,
cross-example dependence tests and a defined inference strategy. Any transition
tuning is a human-defined training run with explicit architecture/version changes.
Given SAIL's interactive batch-one goal, rank this after dispatch, schedules and
memory placement, rather than treating it as a drop-in inference optimization.

## Reproduction and reporting contract

These current commands exercise existing correctness tests only:

```bash
cargo test --offline --test test_moe_sparse --test test_mmap_moe -- --nocapture
```

Use optimized builds for timing and native Windows WGPU for the target GPU.
Pin upstream paper/code versions before reproduction; verify dependency/kernel
support and licenses before any code reuse. The cited implementations are research
references, not installed or proven compatible with this project.

For every experiment, record source/config/model/data/tokenizer hashes, actual
adapter and CPU, OS/driver/backend/features, precision, batch/sequence shapes,
warmup/repeats and synchronization points. Report median and p95 from raw samples,
host RAM/VRAM peaks, cold/warm distinctions and confidence/variation. Distinguish
training tokens/s, decode tokens/s and time-to-quality. Compare with the same
baseline configuration and include unsuccessful cases; do not multiply paper
speedups together or present cross-hardware numbers as local measurements.

The target matrix's numeric VRAM allocations are planning assumptions until real
allocator and concurrent-load evidence exists. Counting requested megabytes or
timing an atomic flag is not measured VRAM or voice latency. In particular, current
target-matrix entries do not establish support for FP16 100M–500M SAIL models or
real 7B/14B GGUF inference. Correct those claims as part of QA-04 before using them
as research baselines.
