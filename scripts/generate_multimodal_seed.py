#!/usr/bin/env python3
"""Generates deterministic, lightweight multimodal seed records for out-of-the-box training.

Creates data/multimodal_seed.jsonl containing paired visual, acoustic, and text records:
- Geometric shapes with distinct RGB colors (32x32 patches).
- Acoustic frequency sweeps and harmonic tones (80 Mel channels x 16 frames).
- Paired text prompts and descriptive target answers.

Zero external dependencies (pure Python standard library).
"""

import json
import math
from pathlib import Path

def generate_shape_image(shape_type, color_rgb, size=32):
    """Generates a 3x32x32 normalized RGB array in [-1.0, 1.0]."""
    r_chan = [-1.0] * (size * size)
    g_chan = [-1.0] * (size * size)
    b_chan = [-1.0] * (size * size)

    norm_r = (color_rgb[0] / 127.5) - 1.0
    norm_g = (color_rgb[1] / 127.5) - 1.0
    norm_b = (color_rgb[2] / 127.5) - 1.0

    cx, cy = size // 2, size // 2
    radius = size // 3

    for y in range(size):
        for x in range(size):
            idx = y * size + x
            inside = False
            if shape_type == "circle":
                inside = ((x - cx) ** 2 + (y - cy) ** 2) <= (radius ** 2)
            elif shape_type == "square":
                inside = abs(x - cx) <= radius and abs(y - cy) <= radius
            elif shape_type == "cross":
                inside = (abs(x - cx) <= 3 and abs(y - cy) <= radius) or (abs(y - cy) <= 3 and abs(x - cx) <= radius)
            elif shape_type == "stripe":
                inside = abs(y - cy) <= 4

            if inside:
                r_chan[idx] = norm_r
                g_chan[idx] = norm_g
                b_chan[idx] = norm_b

    return {
        "channels": 3,
        "height": size,
        "width": size,
        "data": r_chan + g_chan + b_chan
    }

def generate_audio_mel(sound_type, channels=80, frames=16):
    """Generates synthetic 80-channel x 16-frame Mel-scale acoustic features."""
    data = [[-5.0 for _ in range(frames)] for _ in range(channels)]

    for f in range(frames):
        if sound_type == "high_tone":
            target_bin = int(channels * 0.75)
        elif sound_type == "low_hum":
            target_bin = int(channels * 0.20)
        elif sound_type == "ascending_sweep":
            target_bin = int(channels * 0.15 + (channels * 0.70) * (f / frames))
        elif sound_type == "descending_chirp":
            target_bin = int(channels * 0.85 - (channels * 0.70) * (f / frames))
        else:
            target_bin = channels // 2

        for c in range(channels):
            diff = abs(c - target_bin)
            energy = math.exp(-0.5 * (diff / 2.5) ** 2)
            data[c][f] = -5.0 + 6.0 * energy  # Peak around +1.0, floor around -5.0

    # Flat channel-first or frame-first
    flat_data = []
    for c in range(channels):
        flat_data.extend(data[c])

    return {
        "channels": channels,
        "frames": frames,
        "data": flat_data
    }

def build_dataset(output_path, num_samples=200):
    shapes = [
        ("circle", (255, 0, 0), "red circle"),
        ("circle", (0, 0, 255), "blue circle"),
        ("square", (0, 255, 0), "green square"),
        ("square", (255, 255, 0), "yellow square"),
        ("cross", (255, 0, 255), "magenta cross"),
        ("stripe", (0, 255, 255), "cyan horizontal stripe"),
    ]

    sounds = [
        ("high_tone", "high-frequency pure tone"),
        ("low_hum", "low-frequency steady hum"),
        ("ascending_sweep", "ascending acoustic pitch sweep"),
        ("descending_chirp", "descending frequency chirp"),
    ]

    records = []

    # 1. Vision-focused records
    for i in range(num_samples // 3):
        shape_type, color, desc = shapes[i % len(shapes)]
        vis = generate_shape_image(shape_type, color)
        prompt = "Describe visual content:"
        target = f"Visual shows a {desc} on a dark background."
        records.append({
            "text": prompt,
            "target": target,
            "vision": vis,
            "audio": None,
            "provenance": "curated_geometric_seed"
        })

    # 2. Audio-focused records
    for i in range(num_samples // 3):
        sound_type, desc = sounds[i % len(sounds)]
        aud = generate_audio_mel(sound_type)
        prompt = "Transcribe acoustic signal:"
        target = f"Acoustic spectrogram contains a {desc}."
        records.append({
            "text": prompt,
            "target": target,
            "vision": None,
            "audio": aud,
            "provenance": "curated_acoustic_seed"
        })

    # 3. Interleaved Joint Multimodal records
    for i in range(num_samples - len(records)):
        shape_type, color, shape_desc = shapes[i % len(shapes)]
        sound_type, sound_desc = sounds[i % len(sounds)]
        vis = generate_shape_image(shape_type, color)
        aud = generate_audio_mel(sound_type)
        prompt = "Analyze paired multimodal inputs:"
        target = f"Input contains a {shape_desc} paired with a {sound_desc}."
        records.append({
            "text": prompt,
            "target": target,
            "vision": vis,
            "audio": aud,
            "provenance": "curated_joint_multimodal_seed"
        })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Generated {len(records)} multimodal seed records in {output_path}")

if __name__ == "__main__":
    build_dataset("data/multimodal_seed.jsonl", num_samples=180)

