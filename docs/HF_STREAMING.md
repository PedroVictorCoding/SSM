# Bulk dataset prefetch and GPU training

`hf_streamer.py` streams dataset shards through Hugging Face Datasets, batches
records locally, and publishes plain UTF-8 training chunks atomically. It no longer
calls the dataset viewer's two-row sampling API. Sources, topics, filters and
budgets are human-configured under [ALIGNMENT.md](../ALIGNMENT.md).

## Start downloading ahead of training

Use a fresh incoming directory to avoid mixing older prototype fallback files into
this run. In **WSL**, from the repository root:

```bash
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install -r scripts/requirements-hf.txt

.venv-hf/bin/python -u scripts/hf_streamer.py \
  --topic code \
  --incoming data/hf_bulk_code \
  --consumer-run runs/hf_bulk_gpu \
  --prefetch-mib 64 \
  --chunk-mib 4 \
  --batch-rows 2048
```

Let a few chunks accumulate, then start the trainer below. Leave the downloader
running: it pauses at its text-spool budget and resumes after the trainer commits
ingestion receipts. `--once` instead fills the budget and exits. `--max-chunks N`
stops after publishing N chunks during that invocation. Restarting the same command
with the same incoming directory resumes persisted progress automatically.

In **native Windows PowerShell**, rebuild the trainer to enable ingestion receipts:

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\pedro\Dev\SSM'
$env:CARGO_TARGET_DIR = Join-Path $env:LOCALAPPDATA 'SSM\cargo-target'
$env:CUBECL_WGPU_DEFAULT_DEVICE = 'DiscreteGpu(0)'
cargo build --release --offline --no-default-features --features wgpu,ndarray
if ($LASTEXITCODE -ne 0) { throw 'GPU build failed' }
$sail = Join-Path $env:CARGO_TARGET_DIR 'release\sail.exe'

& $sail stream-train --incoming data/hf_bulk_code --out runs/hf_bulk_gpu `
  --initial-data data/tinyshakespeare.txt --bpe --vocab-size 512 `
  --steps 4000 --batch-size 32 --seq-len 128 `
  --d-model 64 --n-layers 4 --d-state 16 --d-conv 4 `
  --attn-layers 1 --n-heads 4 --mlp-ratio 2 --moe-experts 4 --moe-top-k 1 `
  --lr 0.0002 --min-lr 0.00005 --warmup 100 --device-clip --grad-clip 1.0 `
  --replay-ratio 0.30 --min-tokens 4096 --scan-interval 1000 `
  --save-every 200 --eval-every 200 --eval-windows 64 `
  --log-every 25 --viz-every 25 --viz-max-values 0 --sample-chars 0 --seed 42
```

Use a new `runs/hf_bulk_gpu` directory for the first run. To continue it, add
`--resume` and keep the other training settings. The example uses the small local
Shakespeare corpus to train the initial BPE vocabulary quickly and seed replay;
byte fallback covers later incoming text. Omitting `--initial-data` instead trains
the vocabulary from all prefetched files, which costs more startup time as the
prefetch size grows. For general GPU details see [GPU_TRAINING.md](GPU_TRAINING.md).

## Budgets, batching and cleanup

| Option | Default | Meaning |
| --- | ---: | --- |
| `--prefetch-mib` | 256 | Maximum queued top-level `.txt`, `.jsonl`, `.md` payload bytes |
| `--chunk-mib` | 4 | Target UTF-8 payload size per published file |
| `--batch-rows` | 2048 | Maximum records processed per durable batch; not an API row request |
| `--max-record-mib` | 1 | Oversized records are explicitly skipped and logged |
| `--min-chars` | 150 | Minimum accepted text length |
| `--poll-seconds` | 5 | Local budget/checkpoint polling interval |
| `--interval` | 0 | Optional pause between bulk batches |
| `--max-retries` | 8 | Retries per source after consecutive transient failures |

A chunk ends at a record boundary or the row-count limit, so it can be smaller
than the target or exceed it by at most one accepted record. The producer reserves
that headroom before advancing the network iterator. Budget polling does not make
network requests. Dataset iterators are reused between local batches; the shard
reader controls network requests instead of making a viewer API call per chunk.

`--consumer-run` points to the trainer whose replay checkpoint will retain this
incoming data. Each complete streaming checkpoint includes `ingestion.json` in its
generation. The receipt becomes valid only when `checkpoint.json` commits that
generation. A stable spool ID matches a WSL downloader to the native Windows
trainer, despite their different path representations.

The downloader removes **only its own files named in that committed receipt**.
Files merely read into RAM are retained until checkpointed. Files from other
feeders or manually supplied datasets are never removed. Failed tokenization is
not acknowledged. Without a consumer run, or with an older trainer lacking
receipts, prefetch stops growing at the budget and keeps the files.

After cleanup, resume the selected run from its latest committed checkpoint.
Starting another run or rolling back to an older replay snapshot requires its own
data plan/spool; deleted inputs are backed by the checkpoint that acknowledged
them, not necessarily by earlier generations.

The budget covers queued training text, not the entire disk: journal/state writes
need additional room (up to roughly two maximum chunks during publication), and
HF metadata/cache, library read-ahead, trainer checkpoints and replay storage are
separate. The trainer currently retains an append-only replay buffer; bounded
prefetch does not bound its RAM or checkpoint growth over an indefinite run.

## Sources and progress

Presets rotate between the named topic's datasets, with independent progress for
each source. Gated The Stack is no longer included in the default code preset;
select it explicitly after arranging access if needed. An explicit Hub source:

```bash
.venv-hf/bin/python scripts/hf_streamer.py \
  --dataset HuggingFaceTB/smollm-corpus --config python-edu --field text \
  --incoming data/hf_bulk_code --consumer-run runs/hf_bulk_gpu
```

For multiple human-approved sources, `--sources PATH` accepts a JSON array:

```json
[
  {"dataset": "HuggingFaceTB/smollm-corpus", "config": "python-edu", "split": "train", "field": "text"},
  {"dataset": "parquet", "data_files": ["/path/to/approved-shard.parquet"], "split": "train", "field": "text"}
]
```

`--dataset parquet --data-files URL_OR_PATH ... --field text` also accepts explicit
shards from other approved sources. Built-in loaders `json`, `text`, `csv` and
`arrow` are available. Use immutable/versioned shard URLs or unchanged local files
for reproducible continuation. Records with a missing configured field fail clearly;
`messages` fields are converted to role/text lines. The code preset's `output`
source emits that field, not an automatically constructed instruction pair.

State lives in `INCOMING/.hf_streamer/state.json`: source specification, resolved
Hub commit, iterator cursor, row counts and chunk sequence. Hub revisions are pinned
on first successful ingestion. The dependency version is checked on resume; keep
the pinned `datasets` version. Changing topic selects the configured sources while
preserving earlier sources' cursors. Changing record filters requires a new spool.
Only one producer may own an incoming directory at a time.

Cursor advancement and pending output bytes are committed together before a file
is published. A restart completes an interrupted publication without publishing
the same batch twice. Shuffling is deliberately absent. Datasets may reread part
of the current shard to restore an iterator cursor, but previously committed
records are not republished. This uses Hugging Face's documented
[streaming and iterator checkpoint APIs](https://huggingface.co/docs/datasets/v4.8.4/stream).

## Failures and verification

Transient network errors and HTTP 408/425/429/5xx receive exponential backoff with
jitter. Numeric and HTTP-date `Retry-After` headers are honored even when longer
than the exponential-backoff cap. Other ready sources can proceed while one waits.
Persistent failures stop with an error after the retry limit; permission, dataset,
configuration and field errors fail immediately. No fallback examples are emitted,
and exhausted sources do not restart from their beginning.

Tests use synthetic records, actual local Parquet shards, injected publication
interruptions, rate-limit responses, and a full Rust train/checkpoint/prune/resume
round trip without relying on live corpus downloads:

```bash
cargo build --release --offline
HF_HOME=/tmp/sail-hf-tests \
SAIL_TEST_BINARY="$PWD/target/release/sail" \
  .venv-hf/bin/python -m unittest discover -s scripts -p 'test_hf_streamer.py' -v
cargo test --release --offline --test test_stream_data \
  --test test_checkpoint_reliability --test test_continual_learning
```
