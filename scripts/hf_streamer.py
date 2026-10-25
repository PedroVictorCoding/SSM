#!/usr/bin/env python3
"""
Continuous Hugging Face & Open Dataset Ingestion Daemon for SAIL.

Slowly and progressively retrieves datasets from Hugging Face Hub (and curated open sources),
filters and chunks text samples, and deposits them into `data/incoming/`.

The SAIL continual engine (`sail stream-train`) automatically watches this folder,
tokenizes newly arrived chunks, and updates the living model using Experience Replay.

Usage Examples:
  # Stream coding datasets (Python, Rust, algorithms)
  python3 scripts/hf_streamer.py --topic code --interval 10 --incoming data/incoming

  # Stream scientific and STEM papers / textbooks
  python3 scripts/hf_streamer.py --topic science --interval 15 --incoming data/incoming

  # Stream conversational & instruction dialogue
  python3 scripts/hf_streamer.py --topic conversational --interval 12 --incoming data/incoming

  # Stream custom Hugging Face dataset
  python3 scripts/hf_streamer.py --dataset HuggingFaceFW/fineweb-edu --interval 10
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime

# Curated Hugging Face Datasets by Domain
DOMAIN_DATASETS = {
    "code": [
        {"dataset": "iamtarun/python_code_instructions_18k_alpaca", "config": "default", "split": "train", "field": "output"},
        {"dataset": "bigcode/the-stack-v2", "config": "default", "split": "train", "field": "content"},
        {"dataset": "HuggingFaceTB/smollm-corpus", "config": "python-edu", "split": "train", "field": "text"},
    ],
    "science": [
        {"dataset": "HuggingFaceTB/cosmopedia-v2", "config": "default", "split": "train", "field": "text"},
        {"dataset": "arxiv-community/arxiv_dataset", "config": "default", "split": "train", "field": "abstract"},
        {"dataset": "HuggingFaceFW/fineweb-edu", "config": "default", "split": "train", "field": "text"},
    ],
    "conversational": [
        {"dataset": "HuggingFaceH4/ultrachat_200k", "config": "default", "split": "train_sft", "field": "messages"},
        {"dataset": "Open-Orca/OpenOrca", "config": "default", "split": "train", "field": "response"},
    ],
    "general": [
        {"dataset": "HuggingFaceFW/fineweb-edu", "config": "default", "split": "train", "field": "text"},
        {"dataset": "HuggingFaceTB/cosmopedia-v2", "config": "default", "split": "train", "field": "text"},
        {"dataset": "iamtarun/python_code_instructions_18k_alpaca", "config": "default", "split": "train", "field": "output"},
    ],
}

# Robust offline curated knowledge fallback banks if network is unavailable or rate-limited
FALLBACK_CORPUS = {
    "code": [
        """// Rust memory safety and ownership model
pub struct BufferPool<T> {
    storage: Vec<T>,
    capacity: usize,
}

impl<T: Clone> BufferPool<T> {
    pub fn new(capacity: usize) -> Self {
        Self { storage: Vec::with_capacity(capacity), capacity }
    }
    
    pub fn push(&mut self, item: T) -> Result<(), &'static str> {
        if self.storage.len() >= self.capacity {
            return Err("Capacity exceeded");
        }
        self.storage.push(item);
        Ok(())
    }
}
// Ownership XOR mutability guarantees that data races cannot occur at runtime.""",

        """def binary_search(arr: list[int], target: int) -> int:
    \"\"\"Performs logarithmic O(log N) lookup in a sorted sequence.\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",

        """// Fused Associative Parallel Prefix Scan in CubeCL / Rust
#[cube]
pub fn parallel_prefix_scan<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    shared_scratch: &mut SharedMemory<F>,
) {
    let local_id = UNIT_POS_X;
    shared_scratch[local_id] = input[local_id];
    sync_units();
    
    let mut stride = 1;
    while stride < BLOCK_SIZE {
        let val = if local_id >= stride { shared_scratch[local_id - stride] } else { F::new(0.0) };
        sync_units();
        shared_scratch[local_id] = shared_scratch[local_id] + val;
        sync_units();
        stride *= 2;
    }
    output[local_id] = shared_scratch[local_id];
}""",
    ],
    "science": [
        """In statistical mechanics and thermodynamics, entropy S represents a measure of microscopic 
configurations corresponding to a macroscopic state: S = k_B * ln(Omega), where k_B is Boltzmann's 
constant and Omega is the multiplicity of microstates. The Second Law dictates that the total entropy 
of an isolated thermodynamic system never decreases over time, defining the thermodynamic arrow of time.""",

        """Quantum electrodynamics (QED) describes the relativistic quantum field theory of electrodynamics. 
Electrons and positrons interact through the exchange of virtual photons, the gauge bosons of the U(1) 
gauge symmetry. Feynman diagrams provide a perturbative calculation method for scattering cross-sections 
and anomalous magnetic moments with unprecedented experimental precision exceeding ten decimal places.""",

        """CRISPR-Cas9 operates as an adaptive immune mechanism in bacteria against bacteriophages. 
The guide RNA (gRNA) directs the Cas9 endonuclease to introduce a targeted double-strand break in genomic 
DNA adjacent to a protospacer adjacent motif (PAM). Subsequent cellular repair mechanisms (NHEJ or HDR) 
enable precise genome engineering and gene knockout.""",
    ],
    "conversational": [
        """User: Can you explain how selective state space models achieve constant memory during inference?
Assistant: Certainly! Traditional Transformers require caching all past Key-Value states in memory (the KV-cache), 
which expands linearly O(L) with every new generated token. In contrast, Selective SSMs (like Mamba and SAIL) compress 
the entire historical context into a fixed-size continuous hidden state vector h_t. When generating the next token, 
the model simply updates h_t = A * h_{t-1} + B * x_t and computes y_t = C * h_t. Because the state size never grows, 
the model operates with strictly constant O(1) memory and constant per-token latency regardless of context length.""",

        """User: What is the main defense against catastrophic forgetting in continual learning?
Assistant: The most effective defense is Experience Replay (or replay buffers). Instead of training purely on 
the newly arriving data stream, the sampling engine continuously blends fresh data with historical memory windows. 
In SAIL, our dynamic streaming buffer maintains a watermark, reserving e.g. 30% of each training batch for historical 
tokens and 70% for newly acquired knowledge. This guarantees that new concepts are acquired while core reasoning 
remains intact.""",
    ],
    "general": [
        """The Library of Alexandria in Ptolemaic Egypt stood as one of the largest and most significant 
libraries of the ancient world. Founded in the 3rd century BCE under the patronage of Ptolemy I and II, 
it functioned as a major intellectual center housing hundreds of thousands of papyrus scrolls spanning 
astronomy, mathematics, grammar, philosophy, and medicine.""",

        """Modern operating systems implement virtual memory to isolate process address spaces and simulate 
more memory than physically available. The Translation Lookaside Buffer (TLB) acts as a hardware cache for 
page table entries, accelerating virtual-to-physical address translation. When a page fault occurs, the 
kernel retrieves the requested memory frame from secondary storage.""",
    ],
}


def fetch_huggingface_rows(dataset_id: str, config: str = "default", split: str = "train", offset: int = 0, limit: int = 5):
    """
    Attempts to fetch sample rows from the Hugging Face Datasets Server REST API.
    Zero external dependencies; uses standard Python urllib.
    """
    url = f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset_id)}&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}&offset={offset}&limit={limit}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SAIL-Continual-Ingestion/1.0", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("rows", [])
    except Exception:
        return []
    return []


def extract_text(row_data, preferred_field="text"):
    """Extracts raw text string from diverse dataset record formats."""
    row = row_data.get("row", {})
    if preferred_field in row and isinstance(row[preferred_field], str):
        return row[preferred_field]
    for key in ["text", "content", "output", "abstract", "response", "code"]:
        if key in row and isinstance(row[key], str):
            return row[key]
    # Check messages array (dialogue format)
    if "messages" in row and isinstance(row["messages"], list):
        turns = []
        for msg in row["messages"]:
            role = msg.get("role", "User")
            content = msg.get("content", "")
            turns.append(f"{role}: {content}")
        return "\n".join(turns)
    return ""


def main():
    parser = argparse.ArgumentParser(description="SAIL Continuous Hugging Face Dataset Streamer")
    parser.add_argument("--incoming", default="data/incoming", help="Target incoming directory for SAIL watcher")
    parser.add_argument("--topic", default="code", choices=["code", "science", "conversational", "general"], help="Topic domain to stream")
    parser.add_argument("--dataset", default="", help="Optional specific Hugging Face dataset ID override")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between dropped knowledge chunks")
    parser.add_argument("--max-chunks", type=int, default=0, help="Maximum chunks to stream (0 = run indefinitely)")
    parser.add_argument("--min-chars", type=int, default=150, help="Minimum character length per chunk")
    args = parser.parse_args()

    os.makedirs(args.incoming, exist_ok=True)
    print(f"=== SAIL Continuous Ingestion Streamer ===")
    print(f"Target Directory: {args.incoming}")
    print(f"Active Topic:     {args.topic}")
    print(f"Interval:         {args.interval}s")
    if args.dataset:
        print(f"Dataset Override: {args.dataset}")
    print(f"Streaming started. Press Ctrl+C to stop.\n")

    chunk_count = 0
    hf_offset = 0
    fallback_idx = 0

    sources = DOMAIN_DATASETS.get(args.topic, DOMAIN_DATASETS["general"])
    if args.dataset:
        sources = [{"dataset": args.dataset, "config": "default", "split": "train", "field": "text"}]

    source_idx = 0

    while True:
        chunk_count += 1
        current_source = sources[source_idx % len(sources)]
        dataset_id = current_source["dataset"]
        config = current_source.get("config", "default")
        split = current_source.get("split", "train")
        field = current_source.get("field", "text")

        text_content = ""
        source_label = ""

        # Attempt to fetch live from Hugging Face Datasets API
        rows = fetch_huggingface_rows(dataset_id, config=config, split=split, offset=hf_offset, limit=2)
        if rows:
            for r in rows:
                candidate = extract_text(r, field)
                if len(candidate.strip()) >= args.min_chars:
                    text_content = candidate.strip()
                    source_label = f"hf:{dataset_id}"
                    hf_offset += 1
                    break

        # Fallback to curated topic knowledge if offline / throttled
        if not text_content:
            topic_fallbacks = FALLBACK_CORPUS.get(args.topic, FALLBACK_CORPUS["general"])
            text_content = topic_fallbacks[fallback_idx % len(topic_fallbacks)]
            source_label = f"curated:{args.topic}"
            fallback_idx += 1

        # Write clean chunk to data/incoming/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_id = uuid.uuid4().hex[:6]
        filename = f"chunk_hf_{args.topic}_{timestamp}_{rand_id}.txt"
        filepath = os.path.join(args.incoming, filename)

        temporary = filepath + ".tmp"
        with open(temporary, "x", encoding="utf-8") as f:
            f.write(text_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, filepath)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Chunk #{chunk_count:03d} -> {filename} ({len(text_content)} chars | {source_label})")

        if args.max_chunks > 0 and chunk_count >= args.max_chunks:
            print(f"\nReached max chunk limit ({args.max_chunks}). Exiting.")
            break

        source_idx += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

