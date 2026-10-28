#!/usr/bin/env python3
"""Summarize the curated sizing catalog without downloading dataset contents."""

import argparse
import json
from pathlib import Path


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "data/catalogs/curated_datasets.json"


def summarize(catalog):
    datasets = catalog["datasets"]
    ids = [entry["id"] for entry in datasets]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dataset IDs would double-count catalog entries")
    byte_total = 0
    byte_coverage = []
    token_total = 0
    token_coverage = []
    for entry in datasets:
        for measure in entry["metrics"].values():
            if measure is not None and (measure["value"] <= 0 or not measure["source"]):
                raise ValueError(f"invalid metric in {entry['id']}")
        # Pick one representation per entry. Never add original and Parquet copies.
        selected = entry["metrics"]["parquet_bytes"] or entry["metrics"]["download_bytes"]
        if selected is not None:
            byte_total += selected["value"]
            byte_coverage.append(entry["id"])
        tokens = entry["metrics"]["text_tokens"]
        if tokens is not None:
            token_total += tokens["value"]
            token_coverage.append(entry["id"])
    return {
        "catalog_entries": len(datasets),
        "verified_on": catalog["verified_on"],
        "selected_files_gross_bytes": byte_total,
        "selected_files_covered_entries": byte_coverage,
        "selected_files_unknown_entries": [key for key in ids if key not in byte_coverage],
        "publisher_reported_gross_text_tokens": token_total,
        "publisher_reported_token_entries": token_coverage,
        "exact_sail_tokens": None,
        "complete_download_bytes": None,
        "caveats": catalog["aggregation_policy"] + [
            "Selected-file subtotal includes annotation/metadata-only entries and excludes their external payloads.",
            "Publisher token subtotal combines different tokenizer/release scopes and is not a unique SAIL token count.",
            "Download/cache, decoded working data, tokenized data and replay/checkpoints require separate disk budgets.",
        ],
    }


def human(value, suffix=""):
    if value is None:
        return "unknown"
    for scale, prefix in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]:
        if value >= scale:
            return f"{value / scale:.3f} {prefix}{suffix}".strip()
    return f"{value:g} {suffix}".strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--json", action="store_true", help="emit machine-readable totals and coverage")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    totals = summarize(catalog)
    if args.json:
        print(json.dumps(totals, indent=2))
        return
    print(f"Catalog: {totals['catalog_entries']} entries; metadata checked {totals['verified_on']}")
    print(f"Known selected-file subtotal: {human(totals['selected_files_gross_bytes'], 'B')} "
          f"({len(totals['selected_files_covered_entries'])}/{totals['catalog_entries']} entries)")
    print(f"Publisher text-token subtotal: {human(totals['publisher_reported_gross_text_tokens'])} tokens "
          f"({len(totals['publisher_reported_token_entries'])} entries; mixed tokenizers/scopes)")
    print("Complete download size: unknown. Exact SAIL token count: unknown.\n")
    print("| Dataset / selected config | Selected files | Published text tokens | Audio hours |")
    print("| --- | ---: | ---: | ---: |")
    for entry in catalog["datasets"]:
        metrics = entry["metrics"]
        selected = metrics["parquet_bytes"] or metrics["download_bytes"]
        tokens = metrics["text_tokens"]
        hours = metrics["audio_hours"]
        config = (entry["hf"] or {}).get("config")
        label = entry["name"] + (f" / {config}" if config else "")
        print(f"| {label} | {human(selected['value'] if selected else None, 'B')} | "
              f"{human(tokens['value'] if tokens else None)} | "
              f"{human(hours['value'] if hours else None)} |")
    print("\n" + "\n".join("- " + note for note in totals["caveats"]))


if __name__ == "__main__":
    main()
