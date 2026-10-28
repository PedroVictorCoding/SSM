#!/usr/bin/env python3
"""
Convert downloaded Linux & Assistant skill datasets into standard streaming chunks
in `data/incoming/skills_chunk_XXXX.txt` for SAIL continual training.

Sources processed:
1. data/external/linux_skills/bash_commands_dataset.json (840 direct NL -> Bash commands)
2. data/external/linux_skills/bash_instruct_bash_dataset.jsonl (55,000 Linux/coreutils dialogues)
3. data/external/assistant_skills/toolace_data.json (11,300 multi-turn tool-calling dialogues)
"""

import os
import json
import glob

INCOMING_DIR = "data/incoming"
CHUNK_SIZE_TARGET = 512 * 1024  # 512 KB per chunk

class ChunkWriter:
    def __init__(self, prefix="skills_chunk", target_size=CHUNK_SIZE_TARGET):
        self.prefix = prefix
        self.target_size = target_size
        self.chunk_idx = 0
        self.current_buffer = []
        self.current_bytes = 0
        self.total_conversations = 0
        self.total_bytes = 0
        self.created_files = []

    def add_conversation(self, text):
        encoded = (text.strip() + "\n\n").encode("utf-8")
        self.current_buffer.append(encoded)
        self.current_bytes += len(encoded)
        self.total_conversations += 1
        self.total_bytes += len(encoded)

        if self.current_bytes >= self.target_size:
            self.flush()

    def flush(self):
        if not self.current_buffer:
            return
        filename = f"{self.prefix}_{self.chunk_idx:04d}.txt"
        filepath = os.path.join(INCOMING_DIR, filename)
        with open(filepath, "wb") as f:
            for piece in self.current_buffer:
                f.write(piece)
        self.created_files.append((filepath, self.current_bytes))
        self.chunk_idx += 1
        self.current_buffer = []
        self.current_bytes = 0

    def finish(self):
        self.flush()
        return self.created_files

def format_turn(role, content):
    return f"<|im_start|>{role}\n{content.strip()}\n<|im_end|>\n"

def process_bash_commands(writer):
    path = "data/external/linux_skills/bash_commands_dataset.json"
    if not os.path.exists(path):
        print(f"[Skip] {path} not found.")
        return 0
    print(f"[Processing] {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    count = 0
    for item in data:
        prompt = item.get("prompt", "").strip()
        cmd = item.get("response", "").strip()
        if not prompt or not cmd:
            continue
        conv = (
            format_turn("system", "You are a Linux and Bash command assistant.") +
            format_turn("user", prompt) +
            format_turn("assistant", f"```bash\n{cmd}\n```")
        )
        writer.add_conversation(conv)
        count += 1
    print(f"  Processed {count} bash command pairs.")
    return count

def process_bash_instruct(writer):
    path = "data/external/linux_skills/bash_instruct_bash_dataset.jsonl"
    if not os.path.exists(path):
        print(f"[Skip] {path} not found.")
        return 0
    print(f"[Processing] {path}...")
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            
            messages = record.get("messages", [])
            if not messages:
                continue
            
            conv_pieces = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if not content:
                    continue
                conv_pieces.append(format_turn(role, content))
            
            if conv_pieces:
                writer.add_conversation("".join(conv_pieces))
                count += 1
    print(f"  Processed {count} multi-turn bash instructions.")
    return count

def process_toolace(writer):
    path = "data/external/assistant_skills/toolace_data.json"
    if not os.path.exists(path):
        print(f"[Skip] {path} not found.")
        return 0
    print(f"[Processing] {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    count = 0
    for item in data:
        system = item.get("system", "").strip()
        conversations = item.get("conversations", [])
        if not conversations:
            continue
        
        conv_pieces = []
        if system:
            conv_pieces.append(format_turn("system", system))
        
        for turn in conversations:
            from_role = turn.get("from", "user")
            val = turn.get("value", "").strip()
            if not val:
                continue
            
            role_map = {
                "human": "user",
                "user": "user",
                "assistant": "assistant",
                "gpt": "assistant",
                "tool": "tool",
                "system": "system",
            }
            role = role_map.get(from_role.lower(), "user")
            
            # Format tool calls specifically
            if role == "assistant" and (val.startswith("[") and ("(" in val or "{" in val)):
                conv_pieces.append(format_turn("assistant", f"<tool_call>{val}</tool_call>"))
            elif role == "tool":
                conv_pieces.append(format_turn("tool", f"<tool_response>{val}</tool_response>"))
            else:
                conv_pieces.append(format_turn(role, val))
        
        if conv_pieces:
            writer.add_conversation("".join(conv_pieces))
            count += 1
    print(f"  Processed {count} ToolACE tool-calling conversations.")
    return count

def main():
    os.makedirs(INCOMING_DIR, exist_ok=True)
    writer = ChunkWriter(prefix="skills_chunk", target_size=CHUNK_SIZE_TARGET)
    
    n_bash = process_bash_commands(writer)
    n_instruct = process_bash_instruct(writer)
    n_tool = process_toolace(writer)
    
    created = writer.finish()
    
    print("\n=== Summary of Processed Linux & Assistant Skills ===")
    print(f"Total Conversations: {writer.total_conversations:,}")
    print(f"  - Linux Coreutils / Bash Commands: {n_bash + n_instruct:,}")
    print(f"  - Multi-Turn Tool Calling (ToolACE): {n_tool:,}")
    print(f"Total Bytes: {writer.total_bytes / (1024*1024):.2f} MB")
    print(f"Chunks Written to {INCOMING_DIR}: {len(created)} files")
    if created:
        print(f"  First chunk: {created[0][0]} ({created[0][1]} bytes)")
        print(f"  Last chunk:  {created[-1][0]} ({created[-1][1]} bytes)")

if __name__ == "__main__":
    main()

