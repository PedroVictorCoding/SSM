#!/usr/bin/env python3
"""Local, read-only training dashboard. Uses only the Python standard library."""

import argparse
import csv
import io
import json
import math
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import time
from urllib.parse import unquote, urlsplit

STATIC = Path(__file__).parent / "static"
FIELDS = ("step", "lr", "train_loss", "val_loss", "val_bpc", "tokens_per_sec", "elapsed_secs")


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, name):
        path = (self.root / name).resolve()
        if path.parent != self.root or not path.is_dir() or name.startswith("."):
            raise FileNotFoundError("Run not found")
        return path

    def runs(self):
        if not self.root.exists():
            return []
        items = []
        for path in self.root.iterdir():
            try:
                path = self.path(path.name)
                if not (path / "train_config.json").is_file():
                    continue
                metrics = path / "metrics.csv"
                items.append({"id": path.name, "modified": metrics.stat().st_mtime if metrics.exists() else 0})
            except (OSError, ValueError):
                continue
        return sorted(items, key=lambda item: item["modified"], reverse=True)

    def snapshot(self, name):
        path = self.path(name)
        train = read_json(path / "train_config.json")
        model = read_json(path / "model_config.json") or train.get("model", {})
        rows = []
        try:
            # Ignore an unterminated final row: the trainer may be writing it now.
            raw = (path / "metrics.csv").read_text()
            raw = raw[:raw.rfind("\n") + 1]
            for row in csv.DictReader(io.StringIO(raw)):
                if None in row or any(key not in row for key in FIELDS):
                    continue
                clean = {key: number(row[key]) for key in FIELDS}
                if clean["step"] is not None:
                    rows.append(clean)
        except OSError:
            pass
        samples = []
        try:
            for line in (path / "samples.jsonl").read_text().splitlines():
                try:
                    sample = json.loads(line)
                    if isinstance(sample, dict) and isinstance(sample.get("text"), str):
                        samples.append(sample)
                except ValueError:
                    pass
        except OSError:
            pass
        status = read_json(path / "status.json")
        step = max((row["step"] for row in rows), default=0)
        # Older runs have no lifecycle telemetry; freshness does not prove liveness.
        if not status:
            status = {"phase": "finished" if train.get("steps") and step >= train["steps"] else "untracked"}
        status["report_age_secs"] = max(0, time() - status["updated_at"]) if number(status.get("updated_at")) is not None else None
        try:
            neural_version = (path / "neural.json").stat().st_mtime_ns
        except OSError:
            neural_version = None
        return {"id": name, "train_config": train, "model_config": model,
                "telemetry": read_json(path / "telemetry.json"), "status": status,
                "failure": read_json(path / "failure.json"),
                "metrics": rows, "samples": samples[-30:], "server_time": time(),
                "neural_version": str(neural_version) if neural_version else None}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path.startswith("/api/"):
            try:
                if path == "/api/runs":
                    result = {"runs": self.store.runs()}
                elif path.startswith("/api/runs/"):
                    name = path[len("/api/runs/"):]
                    if name.endswith("/neural"):
                        file = self.store.path(name[:-len("/neural")]) / "neural.json"
                        body = file.read_bytes()
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    result = self.store.snapshot(name)
                else:
                    raise FileNotFoundError("Endpoint not found")
                self.send_json(HTTPStatus.OK, result)
            except (FileNotFoundError, ValueError):
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Run or endpoint not found"})
            except OSError:
                self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Run files temporarily unavailable"})
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, status, value):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, store=RunStore(args.runs)))
    print(f"SSM Observatory → http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
