# Numerical failure investigation — 2026-10-24

**Update:** the GPU layout discrepancies described below are fixed in source.
See “Verified GPU fix” for the evidence and restart instructions. The original
observations are retained as investigation history.

## Observed evidence

- `runs/moe_hybrid/metrics.csv` first logs non-finite training loss at step 1970,
  following finite loss 2.200725 at step 1965. Log windows contain five steps, so
  the exact first failing update is not established.
- A subsequent neural capture at step 2161 reports non-finite values in all 93
  parameter tensors. Later latest-checkpoint files were saved during this period;
  they should not be used for recovery without verification.
- The separate best `model.mpk` is associated with validation loss 1.9756899476 at
  step 1500. A copy has all 517,505 parameters finite on the CPU health probe.
- A deterministic next-character loss probe using the preserved best weights
  gives CPU loss 1.2612762 (saved tree strategy), 1.2612764 (chunked), and
  1.2612762 (sequential).
- A diagnostic linked against the existing Windows/WGPU release libraries gives
  2.4422383 (saved tree), 3.17734 (chunked), and 2.3633986 (sequential) on the same
  input and checkpoint. These discrepancies exceed ordinary rounding differences.
  The native Windows build must be reconciled with the CPU reference before
  resuming GPU training. This does not isolate the first failing kernel or prove
  which operation caused the original training NaN.

The read-only probe is `examples/inspect_run.rs`. It examines `model.mpk`, not the
latest `checkpoint_model.mpk`, and writes no run files:

```bash
cargo run --release --example inspect_run -- runs/moe_hybrid_recovery_1500
```

## Preserved recovery copy

`runs/moe_hybrid_recovery_1500` contains copies of the healthy best model and
vocabulary, saved architecture, and step-1500 metadata. The tree scan is disabled
for initial diagnosis. No original run files were replaced.

Healthy optimizer moments from step 1500 are unavailable: the recovery copy has
no optimizer file, so AdamW moments start fresh. This is recovery from known-good
weights rather than an exact optimizer-state continuation. A disposable copy
passed a CPU training step with finite loss 1.8448 and finite validation loss.

Stop the original trainer with Ctrl+C before launching another writer to its
directory. Until the Windows/GPU discrepancy is resolved, a conservative CPU
recovery command, run in WSL from the repository, is:

```bash
cargo run --release -- train \
  --data data/tinyshakespeare.txt --out runs/moe_hybrid_recovery_1500 \
  --resume --steps 3750 --batch-size 32 --seq-len 128 \
  --lr 0.0002 --min-lr 0.00005 --warmup 0 \
  --eval-every 250 --save-every 50 --log-every 5 \
  --viz-every 25 --viz-max-values 0 --sample-chars 200
```

The reduced learning rate is a recovery precaution, not a demonstrated fix for
the GPU discrepancy. The recovery directory remains at step 1500 until the user
starts training there.

## Implemented guards

The trainer now checks finite startup weights, loss every step before backward,
gradients before AdamW, and weights before checkpoint saving. Non-finite
validation loss also stops the run. It records `phase: failed` and a
`failure.json` message instead of continuing to save non-finite checkpoints.
Restore errors are fatal rather than silently continuing with un-restored state.
The dashboard displays the failure reason.

These checks add reductions and synchronization. They were verified with CPU
regression tests; GPU guard execution has not yet been verified. Existing
long-running executables must be stopped and rebuilt to receive the guards.

During investigation, building a new native Windows probe through Cargo hit a
build-script copy access error on the WSL UNC path. A standalone diagnostic was
linked against the existing Windows artifacts using their recorded native library
search paths. Fresh-build GPU equivalence, per-operation tests, and backward-pass
comparison remain necessary to isolate the discrepancy.

## Verified GPU fix

Two noncontiguous tensor paths in Burn 0.16.1's JIT backend caused the forward
discrepancy:

1. **Concatenation of sliced sources.** Its slice-assign kernel iterates using
   source strides. Narrow views can leave parts of concatenated outputs unwritten,
   causing run-to-run variation. `model::cat_dense` materializes source layouts
   through reshape operations before concatenation in the Mamba convolution,
   scans, and attention KV cache. The convolution's zero prefix is also tracked
   with zero derivative to avoid the separately identified cat-backward offset bug.
2. **Timestep projection of a strided view.** The low-rank dt input was sliced
   from the joint `[dt, B, C]` projection. The GPU read later timestep rows
   incorrectly during its matmul. Flattening this input to a dense 2D matrix
   before applying the existing linear projection fixes the result. Shapes,
   checkpoint parameters, and the intended model equations are preserved.

The weight fingerprint was identical on CPU and GPU throughout investigation.
With these fixes, the preserved checkpoint gives next-character loss **1.7177176**
on the deterministic 128-token probe, matching the CPU reference with observation
enabled or disabled and with chunked or sequential scans. Disabling fusion alone
did not fix the issue; the normal fused WGPU backend passes the new CPU/GPU logits
equivalence regression at sequence lengths 7, 32, and 128.

The full CPU test suite passes. A fresh native Windows build using the normal
`wgpu` feature also completed two actual training updates on a **scratch copy**
of the step-1500 checkpoint, with batch 32, sequence length 128, visualization
every step, and validation after each update:

| Step | Train loss | Validation loss |
| --- | --- | --- |
| 1501 | 1.8501 | 2.0101 |
| 1502 | 1.8931 | 2.0081 |

This short smoke test validates finite forward/backward, optimizer updates,
telemetry, validation and checkpoint saving on the GPU. It does not establish
long-run convergence or guarantee the entire remaining training run.
The scratch run used 32 validation windows; the equivalence test checks forward
logits rather than equality of every parameter gradient.

The user's `runs/moe_hybrid_recovery_1500` checkpoint was not changed by these
tests. Re-run the existing PowerShell GPU training command using `cargo run`
to rebuild the fixes and resume there from step 1500. Keep
`CARGO_TARGET_DIR` on the Windows filesystem. Neither `cargo clean` nor switching
to `wgpu-unfused` is required. Do not resume the original NaN-contaminated
`runs/moe_hybrid` checkpoint.

Reproduce on the Windows GPU host:

```powershell
cargo test --release --no-default-features --features wgpu,ndarray --test test_gpu_reference
cargo run --release --no-default-features --features wgpu --example inspect_run -- runs/moe_hybrid_recovery_1500 128 1
```

The optional `wgpu-unfused` feature and `inspect_run`'s trailing `--trace` flag
are diagnostic tools. The trace retains a comparison against the original
strided timestep projection to demonstrate the issue without modifying weights.
