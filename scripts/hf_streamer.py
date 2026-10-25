#!/usr/bin/env python3
"""Resumable, bounded text prefetch from human-configured HF datasets/shards."""
import argparse
import copy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import uuid

LOG = logging.getLogger("hf_streamer")
MIB = 1024 * 1024
FORMAT = 1
PRESETS = {
    "code": [
        {"dataset": "HuggingFaceTB/smollm-corpus", "config": "python-edu", "field": "text"},
        {"dataset": "iamtarun/python_code_instructions_18k_alpaca", "field": "output"},
    ],
    "science": [
        {"dataset": "HuggingFaceTB/cosmopedia-v2", "field": "text"},
        {"dataset": "HuggingFaceFW/fineweb-edu", "field": "text"},
    ],
    "conversational": [
        {"dataset": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "field": "messages"},
        {"dataset": "Open-Orca/OpenOrca", "field": "response"},
    ],
}
PRESETS["general"] = [PRESETS["code"][0], PRESETS["science"][1], PRESETS["conversational"][0]]


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DirectoryLock:
    """OS-held lock, released on exit/crash; no stale PID-file approval needed."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.file = self.path.open("a+b")
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            raise RuntimeError("another downloader owns this incoming directory") from error
        return self

    def __exit__(self, *_):
        self.file.close()


def normalize_source(source):
    allowed = {"dataset", "config", "split", "field", "revision", "data_files"}
    if not isinstance(source, dict) or set(source) - allowed:
        raise ValueError(f"source must contain only {sorted(allowed)}")
    result = {"split": "train", "field": "text", **source}
    if not all(isinstance(result.get(key), str) and result[key] for key in ("dataset", "split", "field")):
        raise ValueError("source requires nonempty dataset, split and field strings")
    for key in ("config", "revision"):
        if key in result and (not isinstance(result[key], str) or not result[key]):
            raise ValueError(f"source {key} must be a nonempty string")
    return result


def source_key(source):
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:20]


def extract_text(row, field):
    content = row.get(field)
    if isinstance(content, str):
        return content.strip()
    if field == "messages" and isinstance(content, list):
        return "\n".join(f"{msg.get('role', 'user')}: {msg['content']}" for msg in content
                         if isinstance(msg, dict) and isinstance(msg.get("content"), str)).strip()
    return ""


def error_status(error):
    """Libraries wrap HTTP failures; inspect causes without printing credentials/URLs."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None) or getattr(error, "code", None)
        if isinstance(status, int):
            headers = getattr(response, "headers", None) or getattr(error, "headers", {}) or {}
            return status, headers.get("Retry-After") or headers.get("retry-after")
        error = error.__cause__ or error.__context__
    return None, None


def retry_delay(error, attempt, base=2.0, cap=300.0, now=None):
    _, retry_after = error_status(error)
    backoff = min(cap, base * 2 ** min(attempt, 20))
    delay = random.uniform(backoff / 2, backoff)
    if retry_after:
        try:
            requested = float(retry_after)
        except ValueError:
            try:
                requested = parsedate_to_datetime(retry_after).timestamp() - (time.time() if now is None else now)
            except (ValueError, TypeError, OverflowError):
                requested = 0.0
        delay = max(delay, requested)  # Never shorten a server's requested delay.
    return delay


def retryable(error):
    status, _ = error_status(error)
    if status is not None:
        return status in (408, 425, 429) or status >= 500
    if isinstance(error, (ValueError, TypeError, ImportError, FileNotFoundError, PermissionError)):
        return False
    return isinstance(error, (OSError, ConnectionError, TimeoutError)) or error.__class__.__module__.split(".")[0] in ("httpx", "requests", "aiohttp", "fsspec", "huggingface_hub")


class HFReader:
    def __init__(self, source, saved, cache_dir):
        from datasets import load_dataset
        from huggingface_hub import HfApi
        self.revision = saved.get("revision")
        builtin = source["dataset"] in ("parquet", "json", "text", "csv", "arrow")
        if not builtin and not self.revision:
            self.revision = HfApi().dataset_info(source["dataset"], revision=source.get("revision")).sha
        kwargs = dict(split=source["split"], streaming=True, cache_dir=str(cache_dir))
        if source.get("config"):
            kwargs["name"] = source["config"]
        if self.revision:
            kwargs["revision"] = self.revision
        if "data_files" in source:
            kwargs["data_files"] = source["data_files"]
        self.dataset = load_dataset(source["dataset"], **kwargs)
        if saved.get("cursor") is not None:
            self.dataset.load_state_dict(copy.deepcopy(saved["cursor"]))
        self.iterator = iter(self.dataset)

    def next(self):
        return next(self.iterator)

    def cursor(self):
        return copy.deepcopy(self.dataset.state_dict())


class Spool:
    def __init__(self, incoming, sources, version, min_chars, max_record_bytes):
        self.incoming = Path(incoming).resolve()
        self.control = self.incoming / ".hf_streamer"
        self.control.mkdir(parents=True, exist_ok=True)
        for stale in self.control.glob("state.json.*.tmp"):
            stale.unlink()
        self.path = self.control / "state.json"
        self.sources = sources
        self.version = version
        self.filters = {"min_chars": min_chars, "max_record_bytes": max_record_bytes}
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state.get("format") != FORMAT or self.state.get("datasets_version") != version:
                raise ValueError("downloader state format/datasets version differs; use the original version or a new incoming directory")
            if self.state.get("filters") != self.filters:
                raise ValueError("record filters changed; use original filters or a new incoming directory")
        else:
            self.state = {"format": FORMAT, "datasets_version": version, "filters": self.filters,
                          "id": uuid.uuid4().hex, "sources": {}, "pending": None}
        for source in sources:
            key = source_key(source)
            self.state["sources"].setdefault(key, {"source": source, "cursor": None, "rows": 0,
                                                   "accepted": 0, "sequence": 0, "exhausted": False})
        self.save()
        self.recover()
        (self.control / "chunk.tmp").unlink(missing_ok=True)

    def save(self):
        atomic_json(self.path, self.state)

    def recover(self):
        pending = self.state["pending"]
        if pending is None:
            return
        target = self.incoming / pending["name"]
        data = pending["text"].encode("utf-8")
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f"published chunk differs from transaction: {target.name}")
        else:
            temp = self.control / "chunk.tmp"
            with temp.open("wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, target)
        self.state["pending"] = None
        self.save()

    def commit(self, key, saved, text):
        self.state["sources"][key] = saved
        if text:
            name = f"hf_{self.state['id']}_{key}_{saved['sequence']:010d}.txt"
            self.state["pending"] = {"name": name, "text": text}
        # Cursor and pending bytes are committed together before publication.
        self.save()
        self.recover()

    def payload_bytes(self):
        # Count all local input files, including other feeders; never delete those.
        return sum(p.stat().st_size for p in self.incoming.iterdir()
                   if p.is_file() and p.suffix in (".txt", ".jsonl", ".md"))

    def reclaim(self, consumer_run):
        if not consumer_run:
            return 0
        run = Path(consumer_run).resolve()
        pointer = run / "checkpoint.json"
        if not pointer.exists():
            return 0
        state = json.loads(pointer.read_text(encoding="utf-8"))
        stream = state.get("stream_file")
        if not stream:
            return 0
        relative = Path(stream)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid consumer checkpoint path")
        receipt = (run / relative).parent / "ingestion.json"
        if not receipt.exists():
            LOG.warning("consumer has no ingestion receipt; rebuild trainer; retaining chunks")
            return 0
        ack = json.loads(receipt.read_text(encoding="utf-8"))
        # A stable identity works across WSL paths and native Windows UNC paths.
        if ack.get("spool_id") != self.state["id"]:
            raise ValueError("consumer receipt belongs to a different incoming spool")
        removed = 0
        prefix = f"hf_{self.state['id']}_"
        for name in ack["processed_files"]:
            if Path(name).name != name or not name.startswith(prefix) or not name.endswith(".txt"):
                continue
            path = self.incoming / name
            if path.is_file():
                removed += path.stat().st_size
                path.unlink()
        return removed


def collect(reader, saved, source, chunk_bytes, batch_rows, min_chars, max_record_bytes):
    saved = copy.deepcopy(saved)
    parts, size = [], 0
    for _ in range(batch_rows):
        try:
            row = reader.next()
        except StopIteration:
            saved["exhausted"] = True
            break
        saved["rows"] += 1
        if source["field"] not in row:
            raise ValueError(f"configured field {source['field']!r} is missing from dataset {source['dataset']}")
        text = extract_text(row, source["field"])
        if len(text) >= min_chars:
            byte_count = len(text.encode("utf-8")) + 2
            if byte_count <= max_record_bytes:
                parts.append(text + "\n\n")
                size += byte_count
                saved["accepted"] += 1
            else:
                LOG.warning("dataset=%s skipped oversized record row=%d bytes=%d", source["dataset"], saved["rows"], byte_count)
        if size >= chunk_bytes:
            break
    saved["cursor"] = reader.cursor()
    saved["revision"] = reader.revision
    if parts:
        saved["sequence"] += 1
    return saved, "".join(parts)


def run(args, sources, version, reader_factory=HFReader):
    incoming = Path(args.incoming).resolve()
    incoming.mkdir(parents=True, exist_ok=True)
    control = incoming / ".hf_streamer"
    control.mkdir(exist_ok=True)
    with DirectoryLock(control / "lock"):
        spool = Spool(incoming, sources, version, args.min_chars, args.max_record_bytes)
        readers, errors, retry_at = {}, {}, {}
        emitted, last_source = 0, -1
        while True:
            reclaimed = spool.reclaim(args.consumer_run)
            if reclaimed:
                LOG.info("reclaimed %d checkpointed input bytes", reclaimed)
            active = [(i, source) for i, source in enumerate(sources)
                      if not spool.state["sources"][source_key(source)]["exhausted"]]
            if not active:
                LOG.info("all configured sources exhausted; no automatic dataset restart")
                return
            # Reserve for the worst-case final record before touching the network.
            if spool.payload_bytes() + args.chunk_bytes + args.max_record_bytes > args.prefetch_bytes:
                LOG.info("prefetch full; waiting for checkpoint-backed ingestion (budget %.1f MiB)", args.prefetch_bytes / MIB)
                if args.once:
                    return
                time.sleep(args.poll_seconds)
                continue
            available = [(i, source) for i, source in active if retry_at.get(source_key(source), 0) <= time.monotonic()]
            if not available:
                time.sleep(min(args.poll_seconds, max(0.01, min(retry_at.values()) - time.monotonic())))
                continue
            index, source = next(((i, s) for i, s in available if i > last_source), available[0])
            last_source = index
            key = source_key(source)
            saved = spool.state["sources"][key]
            try:
                if key not in readers:
                    readers[key] = reader_factory(source, saved, control / "cache")
                updated, text = collect(readers[key], saved, source, args.chunk_bytes, args.batch_rows,
                                        args.min_chars, args.max_record_bytes)
            except Exception as error:
                readers.pop(key, None)  # Reopen from the last durable cursor, including lost partial batches.
                if not retryable(error):
                    status, _ = error_status(error)
                    raise RuntimeError(f"dataset={source['dataset']} failed ({type(error).__name__}, HTTP {status}); check source/config/split/access; no fallback data written") from error
                attempt = errors.get(key, 0)
                if attempt >= args.max_retries:
                    raise RuntimeError(f"dataset={source['dataset']} exhausted {args.max_retries} retries; progress preserved, no fallback data written") from error
                delay = retry_delay(error, attempt)
                errors[key] = attempt + 1
                retry_at[key] = time.monotonic() + delay
                LOG.warning("dataset=%s network failure %s HTTP=%s; retry %d/%d in %.1fs", source["dataset"], type(error).__name__, error_status(error)[0], attempt+1, args.max_retries, delay)
                continue
            spool.commit(key, updated, text)
            errors[key] = 0
            retry_at.pop(key, None)
            if updated["exhausted"]:
                readers.pop(key, None)
            if text:
                emitted += 1
                LOG.info("hf:%s revision=%s rows=%d accepted=%d chunk=%d bytes=%d queued=%.1f MiB", source["dataset"], updated.get("revision"), updated["rows"], updated["accepted"], updated["sequence"], len(text.encode("utf-8")), spool.payload_bytes()/MIB)
                if args.max_chunks and emitted >= args.max_chunks:
                    return
            elif updated["rows"] > saved["rows"]:
                LOG.warning("dataset=%s batch had no eligible text; rows scanned=%d (check filters)", source["dataset"], updated["rows"])
            if args.interval:
                time.sleep(args.interval)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", default="data/incoming")
    parser.add_argument("--topic", choices=tuple(PRESETS), default="code")
    parser.add_argument("--dataset", help="Explicit Hub dataset, or parquet/json/text/csv with --data-files")
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--field", default="text")
    parser.add_argument("--revision")
    parser.add_argument("--data-files", nargs="+", help="Human-selected shard URLs or local files")
    parser.add_argument("--sources", type=Path, help="JSON array of human-configured source specifications")
    parser.add_argument("--consumer-run", help="Trainer run whose committed replay permits removal of this downloader's chunks")
    parser.add_argument("--prefetch-mib", type=float, default=256)
    parser.add_argument("--chunk-mib", type=float, default=4)
    parser.add_argument("--max-record-mib", type=float, default=1)
    parser.add_argument("--batch-rows", type=int, default=2048, help="Maximum rows per local commit, not API request count")
    parser.add_argument("--min-chars", type=int, default=150)
    parser.add_argument("--interval", type=float, default=0, help="Optional pause between bulk batches")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=0, help="Published chunks this invocation; 0 = unlimited")
    parser.add_argument("--once", action="store_true", help="Stop when prefetch budget is full (or sources exhausted)")
    args = parser.parse_args(argv)
    import math
    numbers = [args.prefetch_mib, args.chunk_mib, args.max_record_mib, args.interval, args.poll_seconds]
    if not all(math.isfinite(value) for value in numbers) or min(numbers[:3]) <= 0 or args.interval < 0 or args.poll_seconds <= 0 or args.batch_rows <= 0 or min(args.min_chars, args.max_chunks, args.max_retries) < 0:
        parser.error("sizes, batch rows and polling must be positive; counts and interval must be nonnegative")
    args.prefetch_bytes = int(args.prefetch_mib * MIB)
    args.chunk_bytes = int(args.chunk_mib * MIB)
    args.max_record_bytes = int(args.max_record_mib * MIB)
    if min(args.prefetch_bytes, args.chunk_bytes, args.max_record_bytes) < 1:
        parser.error("sizes must represent at least one byte")
    if args.prefetch_bytes < args.chunk_bytes + args.max_record_bytes:
        parser.error("prefetch budget must hold chunk-mib + max-record-mib")
    if args.sources and args.dataset:
        parser.error("choose --sources or --dataset")
    if not args.dataset and any((args.config, args.revision, args.data_files)):
        parser.error("--config, --revision and --data-files require --dataset")
    if args.sources:
        sources = json.loads(args.sources.read_text(encoding="utf-8"))
    elif args.dataset:
        source = {"dataset": args.dataset, "split": args.split, "field": args.field}
        source.update({key: value for key, value in {"config": args.config, "revision": args.revision, "data_files": args.data_files}.items() if value is not None})
        sources = [source]
    else:
        sources = PRESETS[args.topic]
    if not isinstance(sources, list) or not sources:
        parser.error("sources must be a nonempty JSON array")
    sources = [normalize_source(source) for source in sources]
    if len({source_key(source) for source in sources}) != len(sources):
        parser.error("duplicate sources")
    return args, sources


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        args, sources = parse_args()
        version = importlib.metadata.version("datasets")
        run(args, sources, version)
    except importlib.metadata.PackageNotFoundError:
        LOG.error("install dependencies: python3 -m pip install -r scripts/requirements-hf.txt")
        return 1
    except KeyboardInterrupt:
        LOG.info("stopped; restart with the same incoming directory to resume committed progress")
        return 0
    except Exception as error:
        # No traceback: upstream exceptions may contain signed URLs or tokens.
        LOG.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
