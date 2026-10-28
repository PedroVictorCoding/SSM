#!/usr/bin/env python3
"""Parses downloaded AnyInstruct conversations into training chunks for SAIL.

Converts multi-turn cross-modal dialogues into formatted plain UTF-8 text chunks
with role markers, scenario context, and multimodal tokens, placing them into
data/incoming/ for live stream training.
"""

import json
from pathlib import Path

def process_anyinstruct(input_dir, output_dir, chunk_size_kb=512):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sources = [
        in_path / "part1_metadata.json",
        in_path / "part2_metadata.json",
    ]

    total_conversations = 0
    current_chunk = []
    current_size = 0
    chunk_index = 0

    def flush_chunk():
        nonlocal current_chunk, current_size, chunk_index
        if not current_chunk:
            return
        filename = out_path / f"anyinstruct_chunk_{chunk_index:04d}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(current_chunk) + "\n")
        chunk_index += 1
        current_chunk = []
        current_size = 0

    for src in sources:
        if not src.exists():
            continue
        print(f"Reading {src.name}...")
        with open(src, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items = raw_data.values() if isinstance(raw_data, dict) else raw_data

        for item in items:
            if not isinstance(item, dict):
                continue
            scenario = str(item.get("scenario", "")).strip()
            desc = str(item.get("description", "")).strip()
            chat = item.get("chat", [])
            if not chat and not desc:
                continue

            turns = []
            if scenario:
                turns.append(f"[Topic: {scenario}]")
            if desc and not chat:
                turns.append(desc)
            elif desc and isinstance(chat, list) and chat and isinstance(chat[0], str) and chat[0].startswith("AnyGPT:"):
                # Description often contains the user prompt
                turns.append(desc)

            for msg in chat:
                if isinstance(msg, dict):
                    role = msg.get("role", "User").capitalize()
                    content = msg.get("message", "").strip()
                    if content.startswith("AnyGPT: "):
                        content = content[8:]
                    turns.append(f"{role}: {content}")
                elif isinstance(msg, str):
                    turns.append(msg.strip())

            if turns:
                conv_text = "\n".join(turns)
                conv_bytes = len(conv_text.encode("utf-8"))
                current_chunk.append(conv_text)
                current_size += conv_bytes
                total_conversations += 1

                if current_size >= chunk_size_kb * 1024:
                    flush_chunk()

    # Also process speech_conv if available
    speech_path = in_path / "speech_conv_metadata.jsonl"
    if speech_path.exists():
        print(f"Reading {speech_path.name}...")
        with open(speech_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    chat = item.get("chat", [])
                    turns = []
                    for msg in chat:
                        if isinstance(msg, dict):
                            role = msg.get("role", "User").capitalize()
                            content = msg.get("message", "").strip()
                            turns.append(f"{role}: {content}")
                        elif isinstance(msg, str):
                            turns.append(msg.strip())
                    if turns:
                        conv_text = "\n".join(turns)
                        conv_bytes = len(conv_text.encode("utf-8"))
                        current_chunk.append(conv_text)
                        current_size += conv_bytes
                        total_conversations += 1
                        if current_size >= chunk_size_kb * 1024:
                            flush_chunk()
                except Exception:
                    continue

    flush_chunk()
    print(f"Processed {total_conversations} total conversations into {chunk_index} chunks in {out_path}")

if __name__ == "__main__":
    process_anyinstruct("data/external/AnyInstruct", "data/incoming")

