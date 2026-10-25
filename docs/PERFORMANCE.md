# Performance experiments

The CPU timings below predate the GPU tensor-layout fixes documented in
[NUMERICAL_RECOVERY.md](NUMERICAL_RECOVERY.md#verified-gpu-fix). Those fixes add
layout materialization and correct convolution-prefix gradients; rerun benchmarks
on the current code before treating the earlier timings as current performance.

## Work-efficient selective scan

The existing chunked Hillis–Steele scan repeatedly composes full chunk tensors.
The new opt-in implementation pairs adjacent affine recurrences, recursively
scans the pair summaries, then reconstructs intermediate states. Its tensor work
and saved tensor elements follow T(L) = T(floor(L/2)) + O(L), with logarithmic
dependent depth. Odd lengths use an explicit tail; no power-of-two padding or
division by cumulative products is needed.

This is an application of established work-efficient parallel scan ideas, not a
claim of a new research algorithm. See [Blelloch, Prefix Sums and Their
Applications](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html).
The implementation uses portable Burn tensor operations; it is not a fused GPU kernel.

### Enable

Add `--work-efficient-scan` to a training command. It overrides `--scan-chunk`
and is persisted in the model/train configuration. It can also be selected with
`--resume`; only the execution strategy changes, preserving checkpoint parameters.
An omitted option keeps the saved strategy when resuming. Older configurations
without the optional field continue using their existing strategy.

For a small experiment in a **new** output directory:

```bash
cargo run --release -- train \
  --data data/tinyshakespeare.txt --out runs/tree_experiment \
  --steps 100 --batch-size 4 --seq-len 64 --d-model 64 --d-state 8 \
  --n-layers 4 --attn-layers 2 --mlp-ratio 2 --moe-experts 4 --moe-top-k 2 \
  --work-efficient-scan --viz-every 0 --sample-chars 0
```

Telemetry is disabled in this command to isolate compute performance; it remains
available with the new strategy. Single-token recurrent decoding is unchanged.

### CPU measurements, 2026-10-24

Host: AMD Ryzen 5 7600X, Linux virtualized environment; release build; Burn 0.16.1
ndarray CPU backend, f32. These are local measurements, not GPU results or model
quality evaluations. Shared-host load can affect timing.

Scan forward + backward, shape `[batch=8, length, inner_width=256, state=16]`.
One warmup per strategy, five timed repetitions each, alternating order; median
latency. Both input gradients are explicitly read to synchronize execution.
The baseline is chunked scan with chunk size 64 **and the gradient fix below**.

| Length | Corrected chunked (ms) | Pair-reduction tree (ms) | Speedup |
| --- | --- | --- | --- |
| 16 | 46.351 | 23.258 | 1.99× |
| 32 | 108.548 | 56.062 | 1.94× |
| 64 | 285.783 | 126.711 | 2.26× |
| 127 | 726.035 | 329.636 | 2.20× |
| 128 | 717.295 | 273.004 | 2.63× |
| 256 | 4226.805 | 952.984 | 4.44× |

Complete synthetic training steps, including forward, cross-entropy and MoE
auxiliary loss, backward, and AdamW: batch 4, sequence 64, width 64, state 8,
4 layers `[M M A M]`, 4 experts/top-2, vocabulary 65, MLP ratio 2. One warmup step
then nine timed steps per strategy, alternating execution order. Both models
start from identical weights and see identical synthetic batches; each has its
own optimizer. Updated embedding parameters are read to synchronize the step.

| Corrected chunked median | Tree median | Speedup | Maximum loss difference over 10 updates |
| --- | --- | --- | --- |
| 187.282 ms | 117.059 ms | **1.60×** | 0.00000024 |

This is a compute benchmark on synthetic tokens; data loading, checkpoints,
validation, sample generation, and telemetry are excluded. Long-run validation
quality, peak memory, and CUDA/wgpu timings have not been measured.

### Reproduce

```bash
# Arguments: batch, inner width, state width, repetitions
cargo run --release --example bench_scan -- 8 256 16 5

# Arguments: batch, sequence length, model width, repetitions
cargo run --release --example bench_train_step -- 4 64 64 9

cargo test --release
```

Both benchmarks print CSV rows after their configuration header. GPU builds can
use the corresponding Cargo backend features, but must be measured independently.

### Correctness finding

Burn 0.16.1's `float_cat` backward implementation advances gradient slice offsets
only for tracked inputs. Concatenating a constant prefix with a tracked suffix
therefore routes suffix gradients to the wrong positions. The original chunked
scan's shift helper used that pattern.

The shared shift helper now constructs a tracked prefix with an identically zero
derivative. For finite recurrence inputs this preserves forward values while
correctly aligning gradients. This affects training gradients in the existing
chunked scan too; it is a correctness fix, not an approximation. Previously
trained checkpoint weights still load, but subsequent training follows corrected
gradients and need not reproduce old training trajectories.

Regression checks compare tree and chunked outputs and both input gradients
against the sequential recurrence, including odd lengths, chunk boundaries,
signed/zero/identity transitions, and nonuniform upstream gradients. Tests also
check recurrent decoding, legacy configuration loading, and switching strategy
while resuming. An existing `runs/moe_hybrid` checkpoint was loaded and used for
generation successfully during verification.

## Next experiments

1. **Actual sparse MoE dispatch.** Currently every expert processes every token,
   with routing weights applied afterward. Gather routed tokens per expert and
   scatter results, measuring dispatch overhead and gradients against the dense
   reference. A smaller active-parameter count currently does not imply sparse
   expert FLOPs.
2. **Fused scan and recomputation.** Reduce materialized `[B,L,D,N]` tensors and
   backward state traffic with backend-specific kernels and explicit recomputation.
3. **Telemetry delivery.** Measure host synchronization/serialization costs and
   decouple visualization capture frequency from training work.

These are backlog experiments, not implemented speedups. The brain-inspired
circuit proposals remain deferred in [the product backlog](../PRODUCT_BACKLOG.md).
