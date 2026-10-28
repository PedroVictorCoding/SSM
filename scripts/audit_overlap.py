#!/usr/bin/env python3
"""Audit a training run for train/validation overlap and contamination."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def audit_text_run(run_dir, state):
    stream_file = state.get("stream_file")
    val_file = state.get("validation_file")
    if not stream_file or not val_file:
        return {"status": "skipped", "reason": "Run does not have stream_file or validation_file"}

    stream_path = run_dir / stream_file
    val_path = run_dir / val_file

    if not stream_path.exists():
        return {"status": "error", "reason": f"stream_file missing: {stream_path}"}
    if not val_path.exists():
        return {"status": "error", "reason": f"validation_file missing: {val_path}"}

    stream_data = json.loads(stream_path.read_text(encoding="utf-8"))
    val_tokens = json.loads(val_path.read_text(encoding="utf-8"))
    train_tokens = stream_data.get("tokens", [])

    seq_len = 128
    window_len = seq_len + 1  # input + target
    total_val_tokens = len(val_tokens)
    if total_val_tokens < window_len:
        eval_windows = []
    else:
        # Match eval_starts from train.rs: stride 128, up to 64 windows
        stride = seq_len
        starts = list(range(0, total_val_tokens - seq_len, stride))[:64]
        eval_windows = [val_tokens[s:s + window_len] for s in starts]

    train_token_str = " " + " ".join(str(t) for t in train_tokens) + " "
    overlapping_windows = 0
    for w in eval_windows:
        w_str = " " + " ".join(str(t) for t in w) + " "
        if w_str in train_token_str:
            overlapping_windows += 1

    total_windows = len(eval_windows)
    overlap_pct = (overlapping_windows / total_windows * 100.0) if total_windows > 0 else 0.0

    return {
        "status": "completed",
        "modality": "text",
        "total_train_tokens": len(train_tokens),
        "total_val_tokens": total_val_tokens,
        "evaluated_windows": total_windows,
        "overlapping_windows": overlapping_windows,
        "overlap_percentage": round(overlap_pct, 2),
        "is_contaminated": overlapping_windows > 0,
    }


def record_hash(rec):
    hasher = hashlib.sha256()
    hasher.update((rec.get("text") or "").encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(str(rec.get("target") or "").encode("utf-8"))
    hasher.update(b"\x00")
    if rec.get("vision"):
        hasher.update(b"vis:")
        hasher.update(hashlib.sha256(json.dumps(rec["vision"].get("data", [])).encode("utf-8")).digest())
    if rec.get("audio"):
        hasher.update(b"aud:")
        hasher.update(hashlib.sha256(json.dumps(rec["audio"].get("data", [])).encode("utf-8")).digest())
    return hasher.hexdigest()


def audit_omnimodal_run(run_dir, state):
    stream_file = state.get("stream_file")
    val_file = state.get("validation_file")
    if not stream_file or not val_file:
        return {"status": "skipped", "reason": "Omnimodal run missing stream_file or validation_file"}

    stream_path = run_dir / stream_file
    val_path = run_dir / val_file

    if not stream_path.exists() or not val_path.exists():
        return {"status": "error", "reason": "Stream or validation files not found on disk"}

    stream_data = json.loads(stream_path.read_text(encoding="utf-8"))
    train_records = stream_data.get("records", [])
    val_records = json.loads(val_path.read_text(encoding="utf-8"))

    train_hashes = {record_hash(r) for r in train_records}
    overlapping_records = sum(1 for r in val_records if record_hash(r) in train_hashes)
    total_val = len(val_records)
    overlap_pct = (overlapping_records / total_val * 100.0) if total_val > 0 else 0.0

    return {
        "status": "completed",
        "modality": "omnimodal",
        "total_train_records": len(train_records),
        "total_val_records": total_val,
        "overlapping_records": overlapping_records,
        "overlap_percentage": round(overlap_pct, 2),
        "is_contaminated": overlapping_records > 0,
    }


def audit_spool_db(incoming_dir):
    db_path = incoming_dir / ".hf_streamer" / "dedup.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("""
            SELECT doc_hash, COUNT(DISTINCT split) as split_count
            FROM documents
            GROUP BY doc_hash
            HAVING split_count > 1
        """)
        cross_split_dups = len(cursor.fetchall())
        conn.close()
        return {"database": str(db_path), "cross_split_duplicates": cross_split_dups}
    except Exception as e:
        return {"database": str(db_path), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Audit training runs and spools for train/validation overlap.")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    parser.add_argument("--incoming", type=str, default=None, help="Optional path to incoming spool directory")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    cp_path = run_dir / "checkpoint.json"
    if not cp_path.exists():
        print(f"Error: checkpoint.json not found in {run_dir}", file=sys.stderr)
        sys.exit(2)

    state = json.loads(cp_path.read_text(encoding="utf-8"))
    stream_file = state.get("stream_file", "")

    if "omni_stream.json" in stream_file or (run_dir / "omnimodal_config.json").exists():
        result = audit_omnimodal_run(run_dir, state)
    else:
        result = audit_text_run(run_dir, state)

    result["run"] = str(run_dir)
    result["step"] = state.get("step")

    if args.incoming:
        db_audit = audit_spool_db(Path(args.incoming).resolve())
        if db_audit:
            result["spool_dedup_audit"] = db_audit

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"=== SAIL Overlap Audit ===")
        print(f"Run Directory: {run_dir}")
        print(f"Step: {result.get('step')}")
        print(f"Modality: {result.get('modality')}")
        if result.get("modality") == "text":
            print(f"Evaluated Windows: {result.get('evaluated_windows')}")
            print(f"Overlapping Windows: {result.get('overlapping_windows')} ({result.get('overlap_percentage')}%)")
        else:
            print(f"Validation Records: {result.get('total_val_records')}")
            print(f"Overlapping Records: {result.get('overlapping_records')} ({result.get('overlap_percentage')}%)")

        if result.get("is_contaminated"):
            print("\n[AUDIT REJECTED] Evaluation set is contaminated with training data! Results invalid.")
        else:
            print("\n[AUDIT PASSED] Evaluation set is clean. 0% overlap with training replay.")

    sys.exit(1 if result.get("is_contaminated") else 0)


if __name__ == "__main__":
    main()

