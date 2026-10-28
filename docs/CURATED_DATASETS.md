# Curated Multimodal Datasets for Continual SAIL Training

All ingestion and training must follow [the foundational alignment rule](../ALIGNMENT.md): no autonomous recursive self-improvement, ever, with no exceptions. Context may select data from human-approved external sources within the human-defined training pipeline; the model does not control that pipeline.

A catalog of candidate external datasets for continual learning in **SAIL** (State Space, Mixture of Experts & Active Parameters Hybrid LM). Access, availability and ingestion support vary by source.

The feeder retrieves human-configured sources into `data/incoming/`. This catalog is broader than its built-in presets; listing a dataset here does not configure ingestion or establish that its media format is supported.

## Machine-readable catalog and size budget

The corresponding [JSON catalog](../data/catalogs/curated_datasets.json) contains **27 distinct entries**, with configurations, source URLs, available Hub revision IDs, split counts, typed size measurements, provenance and availability notes. FineWeb-Edu appears in two sections below but is counted once. GSM8K/MATH and VoxCeleb1/2 each have separate entries.

Metadata checked **2026-10-25**, without downloading the corpora:

| Measurement | Known amount | Meaning |
| :--- | ---: | :--- |
| Selected-file download subtotal | **13.414 TB** | Available byte counts for 18/27 entries; one representation per entry. Includes some annotation-only files and excludes their external media/code. Not the complete catalog download size. |
| Published text-token subtotal | **1.582 trillion** | Python-Edu 4B + Cosmopedia v2 28B + StarCoder Data 250B + FineWeb-Edu 1.3T. Approximate publisher figures with different tokenizer/release scopes; cross-source overlap remains. |
| Exact SAIL training tokens | **Unknown** | Requires the selected tokenizer, extracted fields, formatting, filters, deduplication and training split selection. |
| Complete download / working disk budget | **Unknown** | Nine entries lack selected-file byte counts; external payloads, decoded media, caches, tokenized output and checkpoints are additional. |

Largest verified selected-file contributions:

| Selection | Download bytes, decimal | Scope |
| :--- | ---: | :--- |
| OpenVid-1M | 7.493 TB | 202 archive files covering parts 0–185, including split archives; excludes overlapping OpenVidHD packages and extracted copies. |
| FineWeb-Edu / `default` | 4.523 TB | Parquet files for the full configuration; science is not a separate subset. |
| GigaSpeech / `xl` | 1.152 TB | Publisher download size including evaluation splits. |
| Cosmopedia v2 | 122.361 GB | Selected configuration, not the entire SmolLM repository. |
| LibriSpeech / `all` | 61.838 GB | Parquet files including evaluation splits; 960 hours of training audio. |

These figures come from the linked publisher cards, [Hub metadata API](https://huggingface.co/docs/hub/api#get-apidatasetsrepo_id), [Dataset Viewer size API](https://huggingface.co/docs/dataset-viewer/size), and public file listings. Each populated JSON measurement records its own source and scope. Missing quantities are `null`, never zero. Counts of records, conversations, clips and tokens are distinct units.

The Stack v2 separately reports **67.53 TB of uncompressed code**, which is not its download size and is excluded from the byte subtotal. Its often quoted 900B training tokens describe a filtered training version rather than the full listed corpus, so they are also excluded. Python-Edu's approximately 644 MB Parquet selection contains blob metadata: the actual code is retrieved separately from Software Heritage. LLaVA, ShareGPT4V and VideoInstruct annotation sizes likewise exclude their external media. VoxCeleb's original host no longer distributes its audio.

Reproduce the summary locally, without network access:

```bash
python3 scripts/dataset_catalog.py
python3 scripts/dataset_catalog.py --json
```

The catalog is a sizing manifest, **not** an `hf_streamer.py --sources` file or a download-all plan. Its `null` configurations and unavailable entries need concrete selections before use. Existing bounded prefetch limits still control the active feeder; there is no need to download this entire catalog before training.

---

## 1. Code & Formal Reasoning Datasets

These datasets build software engineering skills, syntax precision, algorithmic logic, and tool usage.

| Dataset Name | Hugging Face ID / URL | Modality | Primary Focus |
| :--- | :--- | :--- | :--- |
| **Python-Edu** | [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (subset `python-edu`) | Text (Code) | Educational Python code; published metadata points to separately retrieved Software Heritage contents. |
| **The Stack v2** | [`bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2) | Text (Code) | Multi-language open source code (Rust, C++, Python, Shell, WGSL, CUDA). |
| **StarCoder Data** | [`bigcode/starcoderdata`](https://huggingface.co/datasets/bigcode/starcoderdata) | Text (Code) | Permissively licensed code repositories, documentation, Git commits, and Jupyter notebooks. |
| **Python Instructions 18k** | [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) | Text (Dialogue) | Input-output pairs showing algorithmic problem solving in Python. |
| **GSM8K & MATH** | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) & [`hendrycks/competition_math`](https://huggingface.co/datasets/hendrycks/competition_math) | Text (Math) | Multi-step mathematical reasoning, algebra, calculus, and step-by-step scratchpads. |
| **Rust Crates Index** | [`rust-lang/crates.io-index`](https://github.com/rust-lang/crates.io-index) | Package metadata | Package/version index, not API documentation. Rustdoc/docs.rs content requires a separate source. |

---

## 2. Scientific Knowledge & Technical Reasoning

Datasets for physical sciences, systems engineering, biology, quantum physics, and cosmology.

| Dataset Name | Hugging Face ID / URL | Modality | Primary Focus |
| :--- | :--- | :--- | :--- |
| **FineWeb-Edu (Science Candidate)** | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Text | Educational web text; a science-only filter has not been specified. Same corpus as the general entry. |
| **Cosmopedia v2** | [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (subset `cosmopedia-v2`) | Text (Synthetic) | Synthetic educational text across subjects; old `cosmopedia-v2` repository redirects here. |
| **arXiv Metadata & Abstracts** | [`arxiv-community/arxiv_dataset`](https://huggingface.co/datasets/arxiv-community/arxiv_dataset) | Text | Paper metadata and abstracts; full paper PDFs are separate. |
| **PubMed Abstracts** | [`ncbi/pubmed`](https://huggingface.co/datasets/ncbi/pubmed) (configuration `2024`) | Text | Structured biomedical publication metadata and abstracts. |
| **Wikipedia (English)** | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (subset `20231101.en`) | Text | English encyclopedic snapshot; a STEM-only filter has not been specified. |

---

## 3. General Pretraining & Conversational Instruction

Core language foundations, dialogue coherence, instruction-following, and anti-hallucination.

| Dataset Name | Hugging Face ID / URL | Modality | Primary Focus |
| :--- | :--- | :--- | :--- |
| **FineWeb-Edu** | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Text | 1.3T token corpus filtered by Llama-3 educational classifiers. |
| **UltraChat 200k** | [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Text (Dialogue) | Multi-turn instruction following, roleplay, and complex conversational reasoning. |
| **OpenOrca** | [`Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca) | Text (Chain-of-Thought) | Rich step-by-step reasoning explanations guiding complex question answering. |
| **HotpotQA** | [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa) (configuration `distractor`) | Text (Multi-Hop) | Multi-hop question answering requiring cross-document factual verification. |
| **ToolACE** | [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) | Text (Tool-Calling) | Function calling and environment interaction. ToolBench is a separate dataset, not included by this link. |
| **Bash Instruct 55k** | [`Frost2o24/bash-instruct-II-55k`](https://huggingface.co/datasets/Frost2o24/bash-instruct-II-55k) | Text (Linux Administration) | 55,000 multi-turn GNU coreutils, bash shell scripting, and sysadmin interaction dialogues. |
| **Bash Commands** | [`g-ronimo/bash-commands-dataset`](https://huggingface.co/datasets/g-ronimo/bash-commands-dataset) | Text (Linux CLI) | 840 natural language to shell command pairs for direct command-line synthesis. |

---

## 4. Audio & Speech Datasets (For End-to-End Voice)

Raw and transcribed speech corpora for continuous audio token modeling and real-time conversation.

| Dataset Name | Hugging Face ID / URL | Modality | Format & Details |
| :--- | :--- | :--- | :--- |
| **Mozilla Common Voice 17** | [`mozilla-foundation/common_voice_17_0`](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Audio + Text | Language selection and current distribution availability must be verified; no aggregate hours assigned in the sizing manifest. |
| **LibriSpeech ASR** | [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr) | Audio + Text | 1,000 hours of 16kHz audiobooks cleanly read with aligned sentence transcripts. |
| **VoxCeleb 1 & 2** | [`voxceleb`](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | Audio | Speaker recognition corpora; original host no longer distributes audio. Requires an available approved distribution. |
| **GigaSpeech** | [`speechcolab/gigaspeech`](https://huggingface.co/datasets/speechcolab/gigaspeech) | Audio + Text | 10,000 hours of multi-domain audio from podcasts, audiobooks, and YouTube interviews. |
| **AudioSet** | [`google/audioset`](https://research.google.com/audioset/) | Audio + Labels | 2M sound clips (human sounds, music, animals, machinery) for acoustic representation. |

---

## 5. Vision & Video Datasets (For Multi-Frame Context)

Image-text pairs, multi-frame video sequences, and visual question answering for multi-frame window modeling.

| Dataset Name | Hugging Face ID / URL | Modality | Format & Details |
| :--- | :--- | :--- | :--- |
| **LLaVA Instruct 150K** | [`liuhaotian/LLaVA-Instruct-150K`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Image + Text | Multi-turn conversational visual reasoning, spatial reasoning, and detailed descriptions. |
| **ShareGPT4V** | [`Lin-Chen/ShareGPT4V`](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) | Image + Text | Dense, 800-word granular captions describing every visual object, texture, and relationship. |
| **WebVid-10M** | [`webvid`](https://m-bain.github.io/webvid-dataset/) | Video + Text | 10.7M video clips with text descriptions capturing temporal progression and motion dynamics. |
| **OpenVid-1M** | [`nkp37/OpenVid-1M`](https://huggingface.co/datasets/nkp37/OpenVid-1M) | Video + Text | High-resolution video-caption corpus distributed as archives. |
| **Video-ChatGPT / VideoInstruct-100K** | [`MBZUAI/VideoInstruct-100K`](https://huggingface.co/datasets/MBZUAI/VideoInstruct-100K) | Video + Dialogue | Conversation annotations with separately retrieved underlying videos. |

---

## 6. Recommended Streaming Ingestion Strategy

When using the `scripts/hf_streamer.py` background ingester:

See [bulk prefetch and resume instructions](HF_STREAMING.md). The feeder now uses
shard streaming, durable per-source cursors, bounded local batches, and checkpoint
receipts for cleanup. It does not substitute bundled fallback examples on errors.

1. **Curriculum Scheduling by Mode / Task**:
   - **Coding Sessions**: `--topic code` currently selects `smollm-corpus (python-edu)` and Python Instructions 18k. Python-Edu needs external code resolution; its metadata has no `text` column. The Stack v2 is not in this preset.
   - **Science Sessions**: `--topic science` currently lists the old Cosmopedia repository ID and FineWeb-Edu. Use the canonical `smollm-corpus` ID with explicit `cosmopedia-v2` configuration when configuring that source; arXiv is not in the preset.
   - **Conversational Sessions**: Set `--topic conversational` to stream from `ultrachat_200k` and `openorca`.
   - **General Balanced**: Round-robin streams across all three domains.

2. **Stratified Experience Replay**:
   - SAIL's [`StreamDataset`](../src/data/stream.rs) retains a historical watermark (30%–40% replay ratio).
   - When shifting from coding data to science data, the model retains previous coding abilities by continually re-sampling historical coding tokens alongside fresh science tokens.

