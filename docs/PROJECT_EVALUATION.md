# Sail framework evaluation — 2026-10-25

## Recheck after the follow-up changes

The implementation statuses here supersede the corresponding findings in the
original audit below. Current verification: **52 CPU Rust tests passed** and
**18 Python feeder tests passed**, including the real local-Parquet test and
text train/checkpoint/cleanup/resume round trip. GPU kernels and throughput were
not rebenchmarked for this recheck. Evidence is in
[`benchmarks/project_recheck_20261025.json`](benchmarks/project_recheck_20261025.json).

| Original concern | Current finding |
| --- | --- |
| Routing described as sparse compute | README and code comments now distinguish routing capacity from compute. Actual expert execution still evaluates every expert. Documentation correction, not a sparse-compute implementation. |
| Universal O(1) hybrid memory claim | README now explicitly explains that unbounded attention grows its KV cache. Existing checkpoint still has `attn_window=0`. |
| Experimental CubeCL kernel described as integrated | README and dispatcher comments now identify the CPU fallback and lack of autodiff integration. Kernel acceleration remains unfinished. |
| Voice/search presented as real services | README and tool-module descriptions identify prototypes. Actual ASR still returns a fixed sentence, TTS is harmonic synthesis, and search is a local lookup. Backlog still contains stronger implemented/calibrated/neural/live-verification claims. |
| Lock-free/Redox descriptions | Engine docs now describe `RwLock` snapshot swapping; Redox docs identify a simulator. Real OS deployment and ABI integration remain unverified. |
| Omnimodal waiting consumes scheduled steps | Fixed in the loop: step increments after a batch is available. |
| Omnimodal replay/frozen validation absent | Snapshot and frozen validation files plus restore code now exist. Strong continuation tests, strict missing-state handling, and text-loop-equivalent numerical guards are still incomplete. |
| Independent validation/model quality | Unresolved. The saved step-4000 checkpoint still has 7/7 validation windows present in training replay. No new clean quality benchmark was found. |
| Persistent cross-source deduplication | Partially implemented: hashes filter repeats during one producer process, but are reset on restart and cleared after 100,000 entries. No durable split-level exclusion was found. |
| Competitive external benchmarks | Still absent; the archived within-Sail benchmarks do not establish superiority over PyTorch/Mamba/llama.cpp. |

Three targeted probes expose remaining functional gaps despite passing suites:

1. **Deduplication does not survive restart.** A source with two identical text
   records was fetched with `--batch-rows 1 --max-chunks 1`, then restarted using
   the same persisted spool. Result: **two published files, one unique content**.
   The iterator cursor resumed correctly; the deduplication set did not.
2. **Multimodal deduplication drops valid distinct examples.** Two different
   32×32 RGB arrays with the same class label produced **one accepted record**.
   `collect` hashes only `prompt|target`, excluding image/audio content. For
   unprompted classification this can retain only one example per class within
   the remembered hash set. The key must include the actual example identity.
3. **Multimodal checkpoint cleanup does not match the feeder protocol.** A fresh
   one-step `stream-train --multimodal` run succeeded and committed replay. Its
   generation wrote `processed_files.json`, but feeder cleanup requires
   `ingestion.json` with the matching spool ID. Result: **zero bytes reclaimed**
   and the managed `.omni.jsonl` input retained. A bounded producer will eventually
   wait at its budget even after these inputs have been checkpointed.

These are review findings; production source code and existing checkpoints were
not changed by the recheck. Fixing descriptions is useful, but does not complete
the corresponding capabilities or establish improved trained-model quality.

## Original audit

## Assessment

Sail is a technically credible experimental Rust/Burn training and inference
framework. Its text-model implementation, GPU execution, checkpoint recovery and
observability have useful evidence. Competitive language understanding,
instruction following, multimodal understanding and superiority over other
frameworks have not been established.

The implementation's value is the combination of native Rust, AMD/WGPU training,
instrumentation, streaming ingestion and inspectable recurrent models. The hybrid
architecture itself follows established research: AI21's
[Jamba](https://arxiv.org/abs/2403.19887) already combines Mamba, attention and MoE.
An architecture's published validity does not validate every implementation or
training run derived from it.

This evaluation concerns the current working tree, including uncommitted engine,
multimodal and OS work. Its HEAD and raw measurements are saved in
[`benchmarks/project_evaluation_20261025.json`](benchmarks/project_evaluation_20261025.json).
The foundational [alignment rule](../ALIGNMENT.md) remains unchanged.

## Direct local evidence

Fresh CPU release suite: **52 passed, 0 failed**. This includes recurrence/scan,
training/recovery, streaming, concurrency and synthetic multimodal tests. GPU-only
tests are feature-gated out of that CPU command. Earlier native GPU checks are
documented separately in [GPU_RESULTS.md](GPU_RESULTS.md); they do not certify all
new multimodal or OS paths.

Fresh same-model CPU training-step comparison:

| Implementation | Median step ms | Relative throughput |
| --- | ---: | ---: |
| Chunked scan | 189.990 | 1.00× |
| Work-efficient tree scan | 107.826 | 1.76× |

Batch 4, sequence 64, width 64, four layers `[M M A M]`, four experts/top-2,
f32/ndarray. Both paths start from identical weights, use identical synthetic
batches and separate AdamW optimizers. One warmup plus nine measured updates,
alternating order. Maximum loss difference: **2.4e-7**. Includes forward, loss,
backward and optimizer; excludes ingestion, checkpointing, validation and telemetry.
This compares two implementations within Sail, not Sail against PyTorch.

Archived native Windows measurements on Ryzen 5 7600X / RX 7800 XT, 2026-10-24:

| Configuration | Warm training tok/s | End-to-end tok/s |
| --- | ---: | ---: |
| CPU, original clipping | 1,697 | 1,605 |
| GPU, original clipping | 3,827 | 3,341 |
| GPU, device clipping | 13,298 | 8,408 |
| GPU, device clipping, all weights/gradients captured every update | 1,678 | 1,560 |
| GPU streaming, device clipping | 12,631 | 7,834 |
| Larger GPU workload, device clipping | 26,646 | 23,754 |

First five rows: 380,033 parameters, B8/L64, vocabulary 65. Larger row: 517,505
parameters, B32/L128, 200 updates, vocabulary 65. Clipping rows summarize three
runs per path; other rows are single measurements. GPU clipping improved warm
throughput **3.47×** over its original path. Full visualization reduced that
throughput by **87%**. Larger-workload sampled dedicated allocation peaked at
**3,794 MiB**. These are runtime allocations, not exact live-tensor memory peaks.

Same-artifact single-sequence inference, 380,033 parameters, 128 new tokens:

| Backend | Decode tok/s | Prefill ms |
| --- | ---: | ---: |
| CPU | 851.0 | 3.3 |
| GPU | 98.5 | 444.5 |

CPU was **8.64× faster at decode** on that workload. This is a useful result for
deployment selection; it contradicts a blanket claim that GPU inference is faster.
These historical benchmarks have not been rerun for every current working-tree
change. The user's 575,168-parameter BPE checkpoint is a different model from the
65-character-vocabulary models in these tables.

## Current checkpoint quality audit

Run: `runs/hf_gpu_20261025_075531`, checkpoint step 4000.

| Measurement | Observed value |
| --- | ---: |
| Parameters | 575,168 |
| Token presentations, steps × batch × sequence | 16,384,000 |
| Final training replay buffer tokens | 110,877 |
| Frozen validation tokens | 929 |
| Actual evaluated windows, despite a configured maximum of 64 | 7 |
| Validation windows whose full 129-token input/target span occurs in replay | **7 / 7** |

The audit reproduces `eval_starts`, then searches the saved replay for each exact
input-plus-target token span, enforcing token alignment. This directly establishes
train/validation overlap for every evaluated window in this run. The final
validation loss **0.03689** is not independent evidence of generalization.

The old incoming directory also contains 524 files with only 11 unique byte
contents; its three most repeated texts occur 172, 171 and 171 times. Repeated
fallback data explains the memorized binary-search completion. Those counts and
the window match locations are recorded in the evaluation JSON.

The replacement downloader removes the automatic fallback mechanism and preserves
source progress. It does not by itself provide cross-source deduplication or an
independent evaluation split. Split by source/document identity and deduplicate
across splits before making quality claims.

## External numerical context

These published model results supply scale and capability context. Sail has not
been evaluated under their matching protocols; missing scores are **unmeasured**.

| Model | Parameters | Training token presentations |
| --- | ---: | ---: |
| Current Sail checkpoint | 0.575 million | 16.384 million |
| Official Mamba-130M | 130 million | 300 billion |
| SmolLM2-135M | 135 million | 2 trillion pretraining tokens |
| Jamba-v0.1 | 52 billion total / 12 billion active | Not compared here |

Sources: [official Mamba repository](https://github.com/state-spaces/mamba),
[SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
[Jamba model card](https://huggingface.co/ai21labs/Jamba-v0.1).
SmolLM2-135M has approximately **235×** this Sail checkpoint's parameters. Its
reported pretraining token count is approximately **122,070×** this run's token
presentations. Different tokenizers and repeated data prevent interpreting that
ratio as an equivalent ratio of information learned. The instruct model also has
post-training beyond its base pretraining count.

| Published quality metric | Sail checkpoint | SmolLM2-135M-Instruct | Jamba-v0.1 base |
| --- | ---: | ---: | ---: |
| HellaSwag | Unmeasured | 40.9% | 87.1% |
| PIQA | Unmeasured | 66.3% | 83.2% |
| IFEval average prompt/instruction | Unmeasured | 29.9% | Not reported here |

Values come from the model cards linked above. They are author-reported results
for different models/protocols; this table does not establish an experimental
ranking of Sail. Small capacity and a short repeated-data run are poor evidence
for expecting comparable assistant capability.

For external runtime context, the official
[llama-bench documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
includes a CSV example for **Qwen2.5-7B-Instruct Q4_K_M, 7.62B parameters**, on
**RTX 4080 CUDA / Ryzen 7 7800X3D**: **7,285.68 prompt tokens/s** for 512 prompt
tokens and **119.92 generated tokens/s** for 128 output tokens. These are
documentation examples, not results measured on this machine. `llama-bench`
excludes tokenization and sampling. Hardware, quantization, model, tokenizer and
timing boundaries differ from Sail; no speedup/slowdown ratio is justified.

No matched Sail-versus-PyTorch, official Mamba, or llama.cpp benchmark was executed.
Those runtimes/models were not installed here. A large-model inference engine and
a small-model training framework also serve different purposes.

## Implementation findings that affect validity

1. **Sparse routing does not yet imply sparse expert compute.**
   [`moe.rs`](../src/model/moe.rs) evaluates every expert and applies routing
   weights afterward. The reported active parameter count is a routing statistic,
   not proof of reduced expert FLOPs. The separate mmap prototype does not change
   the backbone's dense execution path.
2. **Whole-model memory is not universally O(1).**
   [`attention.rs`](../src/model/attention.rs) explicitly grows the KV cache when
   `attn_window = 0`, as in the user's current model. SSM state is fixed-size;
   unbounded attention state is not. Windowed attention bounds that component.
3. **A defined CubeCL kernel is not a measured integrated accelerator.**
   [`kernel/scan.rs`](../src/kernel/scan.rs) still dispatches slice inputs through
   its CPU fallback. The production tensor scan uses Burn operations, not this
   experimental launch/backward path.
4. **Several assistant-facing features are demonstrations.**
   [`voice.rs`](../src/voice.rs) returns a fixed transcription after an energy
   check; its TTS creates harmonic tones and does not synthesize linguistic
   content. [`tool.rs`](../src/tool.rs) implements a local keyword/fact lookup,
   not live Internet search. Entropy calculation is not a measured hallucination
   detector or guarantee of factual verification.
5. **New multimodal work has plumbing evidence, not semantic quality results.**
   Projectors, paired-batch ingestion, training/export and streaming tests now
   exist in the working tree and CPU tests pass. Random/synthetic tensors and
   short loss-reduction tests establish execution, not image understanding, ASR
   accuracy or grounded video reasoning. No WER/VQA/captioning benchmark was found.
6. **OS and concurrency descriptions overstate their implementation level.**
   [`engine/mod.rs`](../src/engine/mod.rs) uses `RwLock`; immutable per-request
   snapshots are useful, but acquisition is not lock-free. The
   [Redox handler](../src/os/redox.rs) simulates packet/handle behavior; tests do
   not demonstrate an installed, ABI-verified Redox service.
7. **Multimodal reliability is behind text-loop reliability.**
   The new omnimodal streaming loop does not yet provide the text path's frozen
   validation or full replay snapshot restoration. Its loop advances the step
   before finding a batch, so waiting can consume scheduled steps. Existing text
   throughput/recovery measurements must not be attributed to that new path.

## Benchmarks needed for a defensible competitive claim

1. Fix evaluation validity first: deduplicate documents before the split, retain
   source IDs, and evaluate on unseen tasks/documents. Preserve instruction and
   answer pairs for instruction-following training. Report repetitions and overlap.
2. Compare pure SSM, pure Transformer and hybrid variants at matched parameter
   count or training FLOPs, using the same tokenizer, dataset, schedule and token
   budget. Report held-out loss and task accuracy alongside throughput.
3. Compare Sail and a faithful PyTorch reference with identical exported weights,
   shapes, precision and clipping semantics. Check logits/gradients first, then
   measure forward, backward, optimizer, peak memory and complete training time.
4. Benchmark inference separately: prefill and decode at batch 1/8/32, several
   context lengths, the same model/tokenizer/precision, warmup and repeated trials.
   Include time-to-first-token, inter-token latency and p50/p95/p99 for serving.
5. For multimodal claims, use real held-out paired data and modality-specific
   metrics, plus tests that masking/shuffling each modality changes task accuracy.

The project warrants continued focused development. Its strongest supported claim
is an inspectable native Rust experimental framework with working small-model GPU
training. General-purpose assistant quality and competitive cross-framework
efficiency remain open empirical questions.
