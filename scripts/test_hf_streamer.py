"""Offline behavioral tests plus optional real local-Parquet integration."""
import copy
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile
import os
import subprocess
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import hf_streamer as hf


class FakeReader:
    def __init__(self, rows, saved):
        self.rows = rows
        self.position = (saved.get("cursor") or {}).get("position", 0)
        self.revision = saved.get("revision") or "immutable-commit"

    def next(self):
        if self.position == len(self.rows):
            raise StopIteration
        row = self.rows[self.position]
        self.position += 1
        return row

    def cursor(self):
        return {"position": self.position}


class FeederTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.incoming = self.root / "incoming"
        self.source = hf.normalize_source({"dataset": "test/source"})
        self.rows = [{"text": f"row-{i:03d}-" + "x"*40} for i in range(20)]

    def args(self, *extra):
        return hf.parse_args(["--incoming", str(self.incoming), "--dataset", "test/source",
                              "--min-chars", "1", "--chunk-mib", "0.0001",
                              "--max-record-mib", "0.0001", "--prefetch-mib", "0.01", *extra])[0]

    def factory(self, source, saved, cache):
        return FakeReader(self.rows, saved)

    def payload(self):
        return "".join(p.read_text() for p in sorted(self.incoming.glob("hf_*.txt")))

    def spool(self):
        return hf.Spool(self.incoming, [self.source], "test-version", 1, int(.0001*hf.MIB))

    def test_bulk_commits_and_restart_continue_without_duplicate_rows(self):
        args = self.args("--max-chunks", "2")
        hf.run(args, [self.source], "test-version", self.factory)
        first = self.payload()
        hf.run(self.args(), [self.source], "test-version", self.factory)
        self.assertTrue(self.payload().startswith(first))
        self.assertEqual(self.payload(), "".join(row["text"]+"\n\n" for row in self.rows))
        calls = []
        hf.run(self.args(), [self.source], "test-version", lambda *a: calls.append(a))
        self.assertEqual(calls, [], "exhausted sources must not reopen or loop")

    def test_each_source_has_independent_progress_and_revision(self):
        other = hf.normalize_source({"dataset": "test/other"})
        positions = []
        def factory(source, saved, _):
            positions.append((source["dataset"], copy.deepcopy(saved.get("cursor"))))
            return FakeReader(self.rows, saved)
        hf.run(self.args("--max-chunks", "2"), [self.source, other], "test-version", factory)
        state = self.spool().state
        for source in [self.source, other]:
            saved = state["sources"][hf.source_key(source)]
            self.assertGreater(saved["rows"], 0)
            self.assertEqual(saved["revision"], "immutable-commit")
        hf.run(self.args("--max-chunks", "2"), [self.source, other], "test-version", factory)
        self.assertTrue(all(cursor is not None for _, cursor in positions[2:]))

    def test_budget_stops_without_further_network_reads(self):
        args = self.args("--prefetch-mib", "0.0005", "--once")
        hf.run(args, [self.source], "test-version", self.factory)
        spool = self.spool()
        self.assertLessEqual(spool.payload_bytes(), args.prefetch_bytes)
        saved_rows = spool.state["sources"][hf.source_key(self.source)]["rows"]
        self.assertGreater(saved_rows, 0)
        self.assertLess(saved_rows, len(self.rows))
        with patch.object(hf, "HFReader", side_effect=AssertionError("network touched")):
            hf.run(args, [self.source], "test-version", lambda *a: self.fail("network touched"))

    def test_consumption_requires_committed_matching_receipt(self):
        hf.run(self.args("--max-chunks", "1"), [self.source], "test-version", self.factory)
        spool = self.spool()
        chunk = next(self.incoming.glob("hf_*.txt"))
        unrelated = self.incoming / "human.txt"
        unrelated.write_text("keep")
        run = self.root / "run"
        generation = run / "checkpoints" / "step1"
        generation.mkdir(parents=True)
        receipt = generation / "ingestion.json"
        receipt.write_text(json.dumps({"spool_id": spool.state["id"], "processed_files": [chunk.name, unrelated.name, "../escape.txt"]}))
        self.assertEqual(spool.reclaim(run), 0)
        self.assertTrue(chunk.exists(), "unpublished receipt cannot authorize cleanup")
        (run / "checkpoint.json").write_text(json.dumps({"stream_file": "checkpoints/step1/stream.json"}))
        self.assertGreater(spool.reclaim(run), 0)
        self.assertFalse(chunk.exists())
        self.assertTrue(unrelated.exists())
        receipt.write_text(json.dumps({"spool_id": "another-spool", "processed_files": []}))
        with self.assertRaisesRegex(ValueError, "different incoming spool"):
            spool.reclaim(run)

    def test_checkpointed_cleanup_allows_prefetch_to_resume(self):
        args = self.args("--prefetch-mib", "0.0005", "--once")
        hf.run(args, [self.source], "test-version", self.factory)
        spool = self.spool()
        position = spool.state["sources"][hf.source_key(self.source)]["rows"]
        run = self.root / "run"
        gen = run / "checkpoints" / "one"
        gen.mkdir(parents=True)
        (gen/"ingestion.json").write_text(json.dumps({"spool_id":spool.state["id"], "processed_files":[p.name for p in self.incoming.glob("hf_*.txt")]}))
        (run/"checkpoint.json").write_text(json.dumps({"stream_file":"checkpoints/one/stream.json"}))
        args.consumer_run = str(run)
        hf.run(args, [self.source], "test-version", self.factory)
        self.assertGreater(self.spool().state["sources"][hf.source_key(self.source)]["rows"], position)
        self.assertLessEqual(self.spool().payload_bytes(), args.prefetch_bytes)

    def test_interruption_after_cursor_commit_recovers_publication(self):
        spool = self.spool()
        saved = copy.deepcopy(spool.state["sources"][hf.source_key(self.source)])
        saved.update(cursor={"position":3}, rows=3, sequence=1)
        with patch.object(spool, "recover", side_effect=OSError("injected crash")):
            with self.assertRaises(OSError):
                spool.commit(hf.source_key(self.source), saved, "committed batch\n")
        self.assertEqual(list(self.incoming.glob("hf_*.txt")), [])
        recovered = self.spool()
        self.assertEqual(self.payload(), "committed batch\n")
        self.assertIsNone(recovered.state["pending"])
        self.assertEqual(recovered.state["sources"][hf.source_key(self.source)]["cursor"], {"position":3})
        self.spool()
        self.assertEqual(len(list(self.incoming.glob("hf_*.txt"))), 1)

    def test_interruption_after_publication_does_not_duplicate_file(self):
        spool = self.spool()
        saved = copy.deepcopy(spool.state["sources"][hf.source_key(self.source)])
        saved.update(cursor={"position":1}, rows=1, sequence=1)
        original_save = spool.save
        count = 0
        def fail_second_save():
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("crash after rename")
            original_save()
        with patch.object(spool, "save", side_effect=fail_second_save):
            with self.assertRaises(OSError):
                spool.commit(hf.source_key(self.source), saved, "published once")
        self.spool()
        self.assertEqual(self.payload(), "published once")
        self.assertEqual(len(list(self.incoming.glob("hf_*.txt"))), 1)

    def test_retry_discards_partial_batch_and_reopens_committed_cursor(self):
        opened = 0
        def factory(source, saved, cache):
            nonlocal opened
            opened += 1
            reader = self.factory(source, saved, cache)
            if opened == 1:
                original = reader.next
                def interrupted():
                    if reader.position == 1:
                        raise HTTPError("https://example.invalid",429,"throttled",{"Retry-After":"0"},None)
                    return original()
                reader.next = interrupted
            return reader
        with patch.object(hf, "retry_delay", return_value=0):
            hf.run(self.args(), [self.source], "test-version", factory)
        self.assertEqual(opened,2)
        self.assertEqual(self.payload(), "".join(row["text"]+"\n\n" for row in self.rows))

    def test_permanent_failure_writes_no_fallback(self):
        def denied(*_):
            raise HTTPError("https://example.invalid",403,"gated",{},None)
        with self.assertRaisesRegex(RuntimeError, "no fallback"):
            hf.run(self.args(), [self.source], "test-version", denied)
        self.assertEqual(list(self.incoming.glob("*.txt")), [])

    def test_missing_field_fails_without_advancing_committed_cursor(self):
        def factory(source, saved, cache):
            return FakeReader([{"different_field":"data"}], saved)
        with self.assertRaisesRegex(RuntimeError,"no fallback"):
            hf.run(self.args(),[self.source],"test-version",factory)
        state = self.spool().state["sources"][hf.source_key(self.source)]
        self.assertIsNone(state["cursor"])
        self.assertEqual(list(self.incoming.glob("*.txt")),[])

    def test_retry_budget_is_finite_and_progress_is_preserved(self):
        calls = []
        def failing(*_):
            calls.append(1)
            raise OSError("connection reset")
        with patch.object(hf,"retry_delay",return_value=0):
            with self.assertRaisesRegex(RuntimeError,"exhausted 2 retries"):
                hf.run(self.args("--max-retries","2"),[self.source],"test-version",failing)
        self.assertEqual(len(calls),3)
        self.assertEqual(list(self.incoming.glob("*.txt")),[])

    def test_rate_limit_honors_seconds_and_http_date(self):
        error = HTTPError("https://example.invalid",429,"rate limited",{"Retry-After":"123"},None)
        self.assertGreaterEqual(hf.retry_delay(error,0),123)
        error.headers["Retry-After"] = "Thu, 01 Jan 1970 00:03:00 GMT"
        self.assertGreaterEqual(hf.retry_delay(error,0,now=60),120)
        self.assertFalse(hf.retryable(HTTPError("",401,"",{},None)))
        self.assertTrue(hf.retryable(HTTPError("",503,"",{},None)))

    def test_hub_revision_is_pinned_and_cursor_loaded_on_reopen(self):
        calls = []
        class Dataset:
            position = 0
            def __iter__(self): return self
            def __next__(self):
                self.position += 1
                return {"text":"test"}
            def state_dict(self): return {"position":self.position}
            def load_state_dict(self, state): self.position = state["position"]
        def load_dataset(name, **kwargs):
            calls.append((name,kwargs))
            return Dataset()
        api = unittest.mock.Mock()
        api.dataset_info.return_value.sha = "pinned-hub-commit"
        with patch.dict("sys.modules", {"datasets":types.SimpleNamespace(load_dataset=load_dataset),
                                       "huggingface_hub":types.SimpleNamespace(HfApi=lambda:api)}):
            first = hf.HFReader(self.source,{},self.root)
            first.next()
            second = hf.HFReader(self.source,{"revision":first.revision,"cursor":first.cursor()},self.root)
            self.assertEqual(second.cursor(),{"position":1})
        self.assertEqual(api.dataset_info.call_count,1)
        self.assertTrue(all(kwargs["streaming"] and kwargs["revision"] == "pinned-hub-commit" for _,kwargs in calls))

    @unittest.skipUnless(os.environ.get("SAIL_TEST_BINARY"), "set SAIL_TEST_BINARY for Rust trainer roundtrip")
    def test_trainer_checkpoint_allows_cleanup_and_resume_without_original_files(self):
        hf.run(self.args("--max-chunks","2"),[self.source],"test-version",self.factory)
        run = self.root/"run"
        command = [os.environ["SAIL_TEST_BINARY"],"stream-train","--incoming",str(self.incoming),
                   "--out",str(run),"--bpe","--vocab-size","262","--steps","2",
                   "--batch-size","2","--seq-len","8","--d-model","8","--n-layers","2",
                   "--d-state","3","--attn-layers","1","--n-heads","2","--mlp-ratio","1",
                   "--moe-experts","2","--min-tokens","16","--save-every","1","--eval-every","1",
                   "--eval-windows","2","--log-every","1","--viz-every","0","--sample-chars","0",
                   "--max-minutes","0.1"]
        result = subprocess.run(command,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        spool = self.spool()
        self.assertGreater(spool.reclaim(run),0)
        self.assertEqual(list(self.incoming.glob("hf_*.txt")),[])
        command[command.index("--steps")+1] = "3"
        result = subprocess.run(command+["--resume"],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        checkpoint = json.loads((run/"checkpoint.json").read_text())
        self.assertEqual(checkpoint["step"],3)
        replay = json.loads((run/checkpoint["stream_file"]).read_text())
        self.assertGreater(len(replay["tokens"]),16)

    def test_single_producer_lock_and_resume_version_guard(self):
        self.incoming.mkdir()
        with hf.DirectoryLock(self.incoming/"lock"):
            with self.assertRaisesRegex(RuntimeError,"another downloader"):
                with hf.DirectoryLock(self.incoming/"lock"):
                    pass
        self.spool()
        with self.assertRaisesRegex(ValueError,"version differs"):
            hf.Spool(self.incoming,[self.source],"new-version",1,int(.0001*hf.MIB))

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "install scripts/requirements-hf.txt for real Parquet test")
    def test_real_parquet_shard_iterator_resume(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        shards = []
        for i in range(2):
            shard = self.root/f"shard-{i}.parquet"
            pq.write_table(pa.Table.from_pylist(self.rows[i*10:(i+1)*10]),shard,row_group_size=3)
            shards.append(str(shard))
        source = hf.normalize_source({"dataset":"parquet","data_files":shards})
        version = importlib.metadata.version("datasets")
        hf.run(self.args("--max-chunks","2"),[source],version)
        hf.run(self.args(),[source],version)
        self.assertEqual(self.payload(), "".join(row["text"]+"\n\n" for row in self.rows))
        self.assertEqual(len(list(self.incoming.glob("hf_*.txt"))),7)


if __name__ == "__main__":
    unittest.main()
