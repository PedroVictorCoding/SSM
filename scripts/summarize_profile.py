#!/usr/bin/env python3
"""Summarize synchronized stage timing; exclude cold steps and final I/O by default."""
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("profile", type=Path)
parser.add_argument("--skip", type=int, default=5)
args = parser.parse_args()
rows = [json.loads(line) for line in args.profile.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
selected = rows[args.skip:-1]
if not selected:
    parser.error("not enough steps after skipping warmup and final checkpoint")
stages = {}
for row in selected:
    for name, value in row["stages_ms"].items():
        stages.setdefault(name, []).append(value)
means = {name: statistics.mean(values) for name, values in stages.items()}
total = sum(means.values())
print(json.dumps({
    "selected_steps": [row["step"] for row in selected],
    "mean_step_ms": total,
    "stages": sorted([
        {"stage": name, "mean_ms": value, "percent": 100 * value / total,
         "median_ms": statistics.median(stages[name])}
        for name, value in means.items()
    ], key=lambda item: -item["mean_ms"]),
    "note": "Synchronized timings; measure production throughput with --profile disabled.",
}, indent=2))
