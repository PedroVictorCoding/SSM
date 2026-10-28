# Storage Lifecycle, Bounded Replay, and TTSA (Time-To-Stay-Away) Design

**Document Status**: Specification & Design Proposal for Independent Review (DATA-01 / #22, #24)  
**Permanent Alignment Mandate**: All data ingestion, replay, retention, and training policies are human-configured and human-governed. **SAIL must never perform autonomous recursive self-improvement.** Model outputs and self-assessments are never fed back into training or policy adaptation.

---

## 1. Executive Summary & Problem Definition

During continuous multimodal training on local hardware (e.g. AMD Ryzen 5 7600X / Radeon RX 7800 XT with 16 GB VRAM and limited NVMe staging capacity):
1. **Unbounded Storage Growth**: Naive append-only token buffers in RAM and staging directories (`data/incoming/`) will eventually exhaust available memory and disk.
2. **Premature Payload Deletion**: Deleting staging files merely upon loading into RAM or after a single optimizer step causes unrecoverable data loss if the training process crashes before a durable checkpoint is committed to disk.
3. **Redundant Re-downloads & Split Leakage**: Without durable re-download cooldowns and permanent split tagging, external ingestion scripts may repeatedly re-download the same shards or contaminate frozen evaluation benchmarks with training data.
4. **Upstream Disappearance**: External datasets on Hugging Face or public repositories can be deleted, renamed, or altered upstream. Training recoverability must not rely on the external network remaining immutable.

To resolve these challenges, this specification formalizes:
- **Receipt-based cleanup**: Payload files are deleted from the staging spool *only* when a committed checkpoint receipt guarantees recoverability.
- **TTSA (Time-To-Stay-Away) Cooldown**: A deterministic, human-configured cooldown period separating re-download eligibility from deduplication and replay.
- **Bounded Replay & Checkpoint Budgets**: Explicit token capacity limits in RAM and fixed generational retention on disk ($K$-checkpoint sliding window).
- **Decoupled Recovery**: Checkpoint self-sufficiency guarantees that local training can restart and rollback independently of upstream network availability.

---

## 2. Storage Architecture & Provenance Schema

```
+-----------------------------------------------------------------------------------+
| External Data Feeder (scripts/hf_streamer.py)                                     |
| - Probes human-configured Hugging Face datasets / shards                          |
| - Checks TTSA cooldown & deduplication SQLite store (`dedup.db`)                   |
| - Writes staged chunks to `data/incoming/` with spool identity                     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ Atomic Spool Deposit
+-----------------------------------------------------------------------------------+
| Staging Spool (`data/incoming/hf_<spool_id>_chunk_*.txt`)                         |
| - Monitored by `IncomingDirectoryWatcher`                                         |
| - Preserved across crashes until durable checkpoint receipt is emitted            |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ Scan & Tokenize
+-----------------------------------------------------------------------------------+
| Live Training Engine (`StreamDataset` / `train.rs`)                               |
| - Bounded Experience Replay Buffer (FIFO + Watermark Stratification)              |
| - Generates Checkpoint Generation `step-XXXXXXXX-TIMESTAMP`                        |
| - Writes `ingestion.json` committing consumed spool chunk IDs                      |
| - Saves atomic checkpoint (`model.mpk`, `optimizer.mpk`, `state.json`)             |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ Receipt-Based Reclaim
+-----------------------------------------------------------------------------------+
| Feeder Reclaim (`spool.reclaim(consumer_run)`)                                    |
| - Validates receipt `spool_id` matches active feeder spool                         |
| - Safely unlinks *only* committed and acknowledged chunk files                     |
+-----------------------------------------------------------------------------------+
```

### 2.1 Compact Provenance Record Schema

Every ingested record is tracked in a local transactional SQLite store (`dedup.db`) with the following compact relational schema:

```sql
CREATE TABLE IF NOT EXISTS dataset_provenance (
    content_hash        TEXT PRIMARY KEY,  -- SHA-256 of normalized text or media payload
    source_dataset      TEXT NOT NULL,     -- e.g. "HuggingFaceFW/fineweb-edu"
    source_config       TEXT,              -- e.g. "default"
    source_split        TEXT NOT NULL,     -- e.g. "train"
    source_revision     TEXT NOT NULL,     -- Git commit hash / Hub revision ID
    record_id           TEXT NOT NULL,     -- Canonical row identifier within shard
    split_assignment    TEXT NOT NULL,     -- 'TRAIN' or 'FROZEN_EVAL' (Permanent)
    downloaded_at_unix  INTEGER NOT NULL,  -- Timestamp when fetched from external network
    last_trained_at_unix INTEGER,          -- Timestamp of last optimizer exposure
    exposure_count      INTEGER NOT NULL DEFAULT 0, -- Total forward-backward passes
    eligible_after_unix INTEGER NOT NULL,  -- TTSA cooldown boundary: earliest re-fetch time
    committed_receipt   TEXT               -- Associated checkpoint generation ID
);

CREATE INDEX IF NOT EXISTS idx_provenance_split ON dataset_provenance(split_assignment);
CREATE INDEX IF NOT EXISTS idx_provenance_cooldown ON dataset_provenance(source_dataset, eligible_after_unix);
```

---

## 3. Receipt-Based Cleanup Protocol

### 3.1 Strict Deletion Preconditions
A staging payload file in `data/incoming/` is eligible for disk unlinking **if and only if** all of the following conditions hold:
1. **Committed Checkpoint Receipt**: The trainer has emitted an `ingestion.json` file inside a valid checkpoint directory (`runs/<run_id>/checkpoints/step-XXXXXXXX-*/`).
2. **Spool Identity Match**: The `spool_id` recorded in `ingestion.json` matches the active streamer's UUID in `data/incoming/.hf_streamer/state.json`.
3. **Exact Filename Acknowledgment**: The filename is explicitly listed in `processed_files` within `ingestion.json`.
4. **Durable Optimizer & State Sync**: The checkpoint directory contains non-empty `model.mpk`, `optimizer.mpk`, `stream.json`, and `state.json` files, verified via `fsync`.

> [!CAUTION]
> Ingesting tokens into volatile RAM, updating an in-memory watermark, or completing a single training step does **not** constitute a deletion receipt. If training terminates abruptly before `ingestion.json` is committed, all staging files remain intact in `data/incoming/` and are re-ingested upon restart.

### 3.2 Spool Ownership Isolation
To support mixed-host workflows (e.g. WSL streamer feeding a Windows native trainer across `/mnt/c/` or shared SMB volumes):
- File prefixes strictly encode spool ownership: `hf_<spool_uuid>_<timestamp>_<counter>.txt`.
- A feeder process will **never** unlink files that do not match its own `spool_id` prefix, protecting multi-producer or manual drops.

---

## 4. TTSA (Time-To-Stay-Away) Cooldown Mechanics

### 4.1 Separation of Deduplication, Cooldown, and Split Membership
TTSA separates operational network traffic from mathematical dataset integrity:

| Mechanism | Scope | Persistence | Behavior on Expiry |
| :--- | :--- | :--- | :--- |
| **Split Guard** | 'TRAIN' vs 'FROZEN_EVAL' | Permanent | **Never expires**. An evaluation record is permanently barred from entering training queues. |
| **Content Dedup** | SHA-256 Hash | Permanent | **Never expires**. Identical content is never duplicated within the active buffer. |
| **TTSA Cooldown** | Source record re-fetch | Time-bounded | Once `current_time >= eligible_after_unix`, streamer *may* sample this record again if curriculum requires, updating `exposure_count`. |
| **Experience Replay** | In-memory token pool | Session-bounded | Bounded circular buffer blending historical tokens with fresh tokens at `replay_ratio`. |

### 4.2 Mathematical Cooldown Calculation
When record $r$ is processed, its next eligible download timestamp is set by a human-configured policy:

$$\text{eligible\_after}(r) = t_{\text{trained}} + \Delta_{\text{base}} \cdot \gamma^{\text{exposure\_count}(r)}$$

Where:
- $t_{\text{trained}}$ is the current system timestamp.
- $\Delta_{\text{base}}$ is the base cooldown interval (e.g. 7 days / $604,800\text{ seconds}$).
- $\gamma \ge 1.0$ is the optional exponential backoff multiplier for frequently sampled items (default $\gamma = 1.5$).
- Expiry allows re-download *only* if the external source is actively requested by human configuration; it does not trigger autonomous pull requests.

---

## 5. Bounded Replay and Checkpoint Retention Budgets

### 5.1 Bounded Experience Replay Buffer (`StreamDataset`)
The current append-only buffer is replaced with a **Bounded Stratified Replay Buffer**:
- **Budget**: Human-configured `max_replay_tokens` (e.g. $1,000,000\text{ tokens} \approx 4\text{ MB}$ RAM).
- **Partitioning**:
  - `Historical Pool` ($0 \dots W$): tokens from earlier training windows.
  - `Incoming Pool` ($W \dots N$): freshly tokenized stream tokens.
- **Eviction Policy**: When `total_tokens() > max_replay_tokens`:
  - Evict oldest tokens from the historical pool ($0 \dots \Delta$).
  - Advance `new_data_watermark` accordingly.
  - Memory bounds are strictly maintained at $O(\text{budget})$ without fragmentation.

### 5.2 Sliding-Window Checkpoint Retention
Disk space in `runs/<run_id>/checkpoints/` is bounded by a human-configured retention budget:
- **Parameter**: `--checkpoint-retention <K>` (default: $K = 3$).
- **Lifecycle**:
  1. Checkpoint at step $S_N$ is generated and atomically published.
  2. The trainer checks the list of committed generations: $[S_{N-K}, \dots, S_N]$.
  3. Generations older than $S_{N-K}$ are unlinked *only after* $S_N$ has been verified valid and its `ingestion.json` receipt recorded.
  4. Best-validation checkpoint (`best_model.mpk`) is preserved separately in a dedicated non-expiring slot.
- **Rollback Limits**: Documented rollback limit is exactly $K$ generations. Rollbacks beyond $K$ require restoring from cold archival backups.

---

## 6. Resilience to Upstream Disappearance

1. **Immutable Local Checkpoint Self-Sufficiency**:
   - Checkpoints contain full model weights (`model.mpk`), optimizer state (`optimizer.mpk`), model architecture (`model_config.json`), tokenizer vocabulary (`vocab.json`), and stream replay state (`stream.json`).
   - Resuming training never requires communicating with Hugging Face. If an external dataset is taken offline, deleted, or changed, training proceeds uninterrupted on existing spooled and historical replay data.
2. **Revision Pinning**:
   - Every external source requires an explicit commit hash or revision tag (`revision: "e1f8a2..."`).
   - If upstream changes the revision, the streamer logs an actionable notice and halts ingestion for that source until human confirmation, preventing silent distribution shifts.

---

## 7. Deterministic Testing with Fake Clock

To verify cooldown boundaries and time-dependent edge cases without sleeping:
- Introduce `sail::data::clock::Clock` trait with `SystemClock` and `SimulatedClock`.
- Integration tests (`tests/test_data_lifecycle_and_ttsa.rs`) advance the simulated clock by hours/days instantaneously to assert:
  - Records within cooldown window are rejected by the feeder.
  - Records past cooldown window become eligible.
  - Evaluation splits remain permanently excluded regardless of clock advancement.
  - Checkpoint receipts delete only committed chunks.
  - Overloaded replay buffers evict deterministically within the token budget.

---

## 8. Alignment Invariant Verification

In accordance with [ALIGNMENT.md](../ALIGNMENT.md):
- **Model Cannot Alter Policy**: The model has no write access to `dedup.db`, `StreamConfig`, `retention_budget`, or TTSA parameters.
- **Human Governance**: All cooldowns, retention numbers, and dataset sources are defined via command-line arguments or static JSON configurations.
- **Data Segregation**: Ephemeral assistant runtime outputs (camera frames, audio buffers, tool stdout) carry provenance tags that are strictly rejected by `IncomingDirectoryWatcher`.

