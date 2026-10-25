# Curated Multimodal Datasets for Continual SAIL Training

All ingestion and training must follow [the foundational alignment rule](../ALIGNMENT.md): no autonomous recursive self-improvement, ever, with no exceptions. Context may select data from human-approved external sources within the human-defined training pipeline; the model does not control that pipeline.

A master catalog of open-access, high-quality datasets for continual online learning in **SAIL** (State Space, Mixture of Experts & Active Parameters Hybrid LM).

The continual ingestion engine slowly retrieves text, speech, audio, and video chunks from these sources into `data/incoming/`, allowing SAIL to grow continuously across reasoning, coding, science, spoken dialogue, and visual understanding.

---

## 1. Code & Formal Reasoning Datasets

These datasets build software engineering skills, syntax precision, algorithmic logic, and tool usage.

| Dataset Name | Hugging Face ID / URL | Modality | Primary Focus |
| :--- | :--- | :--- | :--- |
| **FineWeb-Edu Python** | [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (subset `python-edu`) | Text (Code) | High educational value Python code with explanations. |
| **The Stack v2** | [`bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2) | Text (Code) | Multi-language open source code (Rust, C++, Python, Shell, WGSL, CUDA). |
| **StarCoder Data** | [`bigcode/starcoderdata`](https://huggingface.co/datasets/bigcode/starcoderdata) | Text (Code) | Permissively licensed code repositories, documentation, Git commits, and Jupyter notebooks. |
| **Python Instructions 18k** | [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) | Text (Dialogue) | Input-output pairs showing algorithmic problem solving in Python. |
| **GSM8K & MATH** | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) & [`hendrycks/competition_math`](https://huggingface.co/datasets/hendrycks/competition_math) | Text (Math) | Multi-step mathematical reasoning, algebra, calculus, and step-by-step scratchpads. |
| **Rust Crates Docs** | [`rust-lang/crates.io-index`](https://github.com/rust-lang/crates.io-index) | Text (Docs) | Official documentation and signatures for top Rust crates and stdlib APIs. |

---

## 2. Scientific Knowledge & Technical Reasoning

Datasets for physical sciences, systems engineering, biology, quantum physics, and cosmology.

| Dataset Name | Hugging Face ID / URL | Modality | Primary Focus |
| :--- | :--- | :--- | :--- |
| **FineWeb-Edu (Science Subset)** | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Text | Academic papers, textbooks, and educational science articles. |
| **Cosmopedia v2** | [`HuggingFaceTB/cosmopedia-v2`](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2) | Text (Synthetic) | Deep synthetic textbooks and courses generated across STEM disciplines. |
| **arXiv Papers & Abstracts** | [`arxiv-community/arxiv_dataset`](https://huggingface.co/datasets/arxiv-community/arxiv_dataset) | Text | Scientific preprints spanning Machine Learning, Physics, Mathematics, and Astrophysics. |
| **PubMed Abstracts** | [`pubmed`](https://huggingface.co/datasets/pubmed) | Text | Biomedical and biological literature covering neuroscience, genetics, and pharmacology. |
| **Wikipedia (STEM Articles)** | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (subset `20231101.en`) | Text | Verified encyclopedic knowledge across history, science, geography, and philosophy. |

---

## 3. General Pretraining & Conversational Instruction

Core language foundations, dialogue coherence, instruction-following, and anti-hallucination.

| Dataset Name | Hugging Face ID / URL | Modality | Primary Focus |
| :--- | :--- | :--- | :--- |
| **FineWeb-Edu** | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Text | 1.3T token corpus filtered by Llama-3 educational classifiers. |
| **UltraChat 200k** | [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Text (Dialogue) | Multi-turn instruction following, roleplay, and complex conversational reasoning. |
| **OpenOrca** | [`Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca) | Text (Chain-of-Thought) | Rich step-by-step reasoning explanations guiding complex question answering. |
| **HotpotQA** | [`hotpot_qa`](https://huggingface.co/datasets/hotpot_qa) | Text (Multi-Hop) | Multi-hop question answering requiring cross-document factual verification. |
| **ToolBench / ToolACE** | [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) | Text (Tool-Calling) | Function calling, API calls (`<tool_call>...</tool_call>`), and environment interaction. |

---

## 4. Audio & Speech Datasets (For End-to-End Voice)

Raw and transcribed speech corpora for continuous audio token modeling and real-time conversation.

| Dataset Name | Hugging Face ID / URL | Modality | Format & Details |
| :--- | :--- | :--- | :--- |
| **Mozilla Common Voice 17** | [`mozilla-foundation/common_voice_17_0`](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Audio + Text | 30,000+ hours of diverse accents, background acoustics, and verified transcripts (MP3/WAV). |
| **LibriSpeech ASR** | [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr) | Audio + Text | 1,000 hours of 16kHz audiobooks cleanly read with aligned sentence transcripts. |
| **VoxCeleb 1 & 2** | [`voxceleb`](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | Audio | Real-world conversational audio with diverse ambient noise and natural speaker cadences. |
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
| **OpenVid-1M** | [`NKU-VIC/OpenVid-1M`](https://huggingface.co/datasets/NKU-VIC/OpenVid-1M) | Video + Text | High-resolution video clips with rich temporal event annotations. |
| **Video-ChatGPT** | [`MBZUAI/Video-ChatGPT-100K`](https://huggingface.co/datasets/MBZUAI/Video-ChatGPT-100K) | Video + Dialogue | Video QA and time-grounded conversational reasoning across multi-second time windows. |

---

## 6. Recommended Streaming Ingestion Strategy

When using the `scripts/hf_streamer.py` background ingester:

See [bulk prefetch and resume instructions](HF_STREAMING.md). The feeder now uses
shard streaming, durable per-source cursors, bounded local batches, and checkpoint
receipts for cleanup. It does not substitute bundled fallback examples on errors.

1. **Curriculum Scheduling by Mode / Task**:
   - **Coding Sessions**: Set `--topic code` to stream from `smollm-corpus (python-edu)` and `the-stack-v2`.
   - **Science Sessions**: Set `--topic science` to stream from `cosmopedia-v2` and `arxiv_dataset`.
   - **Conversational Sessions**: Set `--topic conversational` to stream from `ultrachat_200k` and `openorca`.
   - **General Balanced**: Round-robin streams across all three domains.

2. **Stratified Experience Replay**:
   - SAIL's [`StreamDataset`](../src/data/stream.rs) retains a historical watermark (30%–40% replay ratio).
   - When shifting from coding data to science data, the model retains previous coding abilities by continually re-sampling historical coding tokens alongside fresh science tokens.

