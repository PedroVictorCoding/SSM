# Native AMD GPU training and recovery

All work follows [ALIGNMENT.md](../ALIGNMENT.md). Learning remains continuous,
context-dependent training on human-approved external data through the configured
pipeline. The model does not control its own implementation or training policy.

## Device and build

The verified device is an AMD Radeon RX 7800 XT (16 GB), on native Windows WGPU
with Vulkan. WSL's available software Vulkan adapter is not this GPU. Use a native
Windows Rust toolchain from PowerShell, with build artifacts on the Windows disk:

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\pedro\Dev\SSM'
$env:CARGO_TARGET_DIR = Join-Path $env:LOCALAPPDATA 'SSM\cargo-target'
$env:CUBECL_WGPU_DEFAULT_DEVICE = 'DiscreteGpu(0)'
cargo build --release --offline --no-default-features --features wgpu,ndarray
if ($LASTEXITCODE -ne 0) { throw 'GPU build failed' }
$sail = Join-Path $env:CARGO_TARGET_DIR 'release\sail.exe'
```

The executable prints the physical adapter name, device type, graphics backend,
and driver. Confirm `AMD Radeon RX 7800 XT` and `DiscreteGpu`. CUDA is for NVIDIA
hardware and is not the backend for this AMD card. `--offline` requires dependencies
to be cached. No driver or system configuration changes are needed by these commands.

## Isolated training

Use a fresh output directory for a new run. This example leaves existing checkpoints
untouched, captures every neural snapshot, saves every 200 updates, and logs every 25:

```powershell
$run = 'runs\amd_training_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $sail train --data data/tinyshakespeare.txt --out $run `
  --steps 4000 --batch-size 8 --seq-len 64 `
  --d-model 64 --n-layers 3 --d-state 8 --d-conv 4 `
  --attn-layers 1 --n-heads 4 --mlp-ratio 2 --moe-experts 4 --moe-top-k 1 `
  --lr 0.0002 --min-lr 0.00005 --warmup 100 --grad-clip 1.0 --device-clip `
  --save-every 200 --eval-every 200 --eval-windows 64 `
  --log-every 25 --viz-every 1 --viz-max-values 0 --sample-chars 0 --seed 42
```

`--device-clip` opts into the on-device equivalent of Burn 0.16's **per-parameter
tensor** norm clipping. The existing Burn implementation remains the default for
comparison. This does not change the objective, data, optimizer moments, or learning
rate schedule. `--profile` enables synchronized per-stage measurements; leave it off
when measuring production throughput.

## Checkpoint recovery

Resume from the same run directory, using `--resume --out $run` and the original
training settings. `--steps` is the total target step, not a count of additional
updates. Keeping it unchanged preserves the original learning-rate schedule.

For the example above, resume with:

```powershell
& $sail train --data data/tinyshakespeare.txt --out $run --resume `
  --steps 4000 --batch-size 8 --seq-len 64 `
  --lr 0.0002 --min-lr 0.00005 --warmup 100 --grad-clip 1.0 --device-clip `
  --save-every 200 --eval-every 200 --eval-windows 64 `
  --log-every 25 --viz-every 1 --viz-max-values 0 --sample-chars 0 --seed 42
```

`checkpoint.json` points to an immutable generation under `checkpoints/`. The
generation contains the matching model, optimizer, model configuration, sampler
RNG, and, for streaming runs, replay buffer, ingested-file history and frozen
validation tokens. The root pointer changes only after the generation is complete.
Incomplete unpublished generations are ignored. Generation directories are retained
and consume disk space; no automatic checkpoint deletion is performed.

Legacy checkpoints without optimizer or RNG state print explicit fallback messages;
they cannot provide exact continuation of the missing state. A numerical failure
stops before committing that update and writes failure details. Existing committed
checkpoints remain available. Do not resume a checkpoint with non-finite weights.

## Streaming and concurrent inference

To download larger batches before and during training, use the resumable,
disk-bounded feeder in [HF_STREAMING.md](HF_STREAMING.md). `stream-train` watches
local files; downloading remains a separate human-configured process.

Use `stream-train --initial-data PATH --incoming DIR --out NEW_RUN` with the same
model/training flags. The initial replay buffer reserves a frozen validation tail;
`--validation-data PATH` supplies a separate held-out corpus instead. Without enough
held-out tokens, validation fields remain empty. Validation tokens are not added to
the replay buffer by this reservation step.

```powershell
$run = 'runs\amd_stream_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$incoming = "$run\incoming"
& $sail stream-train --initial-data data/tinyshakespeare.txt --incoming $incoming --out $run `
  --steps 4000 --batch-size 8 --seq-len 64 `
  --d-model 64 --n-layers 3 --d-state 8 --d-conv 4 `
  --attn-layers 1 --n-heads 4 --mlp-ratio 2 --moe-experts 4 --moe-top-k 1 `
  --lr 0.0002 --min-lr 0.00005 --warmup 100 --grad-clip 1.0 --device-clip `
  --replay-ratio 0.30 --save-every 200 --eval-every 200 --eval-windows 64 `
  --log-every 25 --viz-every 25 --sample-chars 0 --seed 42
```

Streaming publishes complete `model.sea` replacements at saves. A reader opens one
complete version while the trainer continues publishing later versions:

```powershell
& $sail run --model "$run\model.sea" --prompt 'The theory' --tokens 100
```

The `.sea` path above applies to streaming runs; regular runs can use `sail export`
to create an artifact. Training snapshots show weights before the labeled update,
with the corresponding gradients. They are measurements of the configured training
pipeline, not a mechanism for self-directed training or modification.

## Reproduce the checks and measurements

```powershell
cargo test --release --offline --no-default-features --features wgpu,ndarray `
  --lib --test test_gpu_reference --test test_gpu_training --test test_checkpoint_reliability

# Keep the script local to avoid UNC RemoteSigned restrictions; no policy change.
$bench = Join-Path $env:CARGO_TARGET_DIR 'benchmark_gpu.ps1'
Copy-Item scripts/benchmark_gpu.ps1 $bench -Force
$benchRun = 'runs\amd_bench_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $bench -Binary $sail -Out $benchRun -Steps 60 -DeviceClip
```

The harness requires a new output directory. Repeat with `-VizEvery 1`,
`-Mode stream-train`, and `-Profile` in separate directories. Save a baseline binary
before rebuilding and repeat without `-DeviceClip` for the clipping comparison.
`benchmark.json` records arguments, binary/data hashes, process wall time, throughput
windows, validation and sampled GPU memory. `memory_samples.json` retains individual
Windows process counter samples. These are sampled process allocations, not exact
instantaneous tensor-memory peaks; other applications and shared-host load affect
timings. Profiling runs additionally write `profile.jsonl`, summarized with:

```bash
python3 scripts/summarize_profile.py runs/YOUR_PROFILE_RUN/profile.jsonl
```

The larger measured workload is reproducible with:

```powershell
$largeRun = 'runs\amd_large_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $bench -Binary $sail -Out $largeRun -Steps 200 -DeviceClip `
  -BatchSize 32 -SeqLen 128 -DModel 64 -NLayers 4 -DState 16
```

To preserve separate native binaries for comparison:

```powershell
$release = Join-Path $env:CARGO_TARGET_DIR 'release'
# After the GPU build above:
Copy-Item $sail (Join-Path $release 'sail-gpu.exe') -Force
cargo build --release --offline --no-default-features --features ndarray
if ($LASTEXITCODE -ne 0) { throw 'CPU build failed' }
Copy-Item (Join-Path $release 'sail.exe') (Join-Path $release 'sail-cpu.exe') -Force
$sail = Join-Path $release 'sail-gpu.exe'
$cpu = Join-Path $release 'sail-cpu.exe'

$cpuRun = 'runs\cpu_bench_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $bench -Binary $cpu -Out $cpuRun -Steps 60

# Both inference commands load the same streaming artifact:
$gpuInference = 'runs\gpu_infer_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$cpuInference = 'runs\cpu_infer_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $bench -Binary $sail -Out $gpuInference -Mode run -Model "$run\model.sea" -Tokens 128
& $bench -Binary $cpu -Out $cpuInference -Mode run -Model "$run\model.sea" -Tokens 128
```
