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

    @unittest.skipUnless(os.environ.get("SAIL_TEST_BINARY"), "set SAIL_TEST_BINARY for Rust trainer roundtrip")
    def test_multimodal_trainer_roundtrip_with_compiled_binary(self):
        omni_incoming = self.root / "omni_incoming"
        audio_source = hf.normalize_source({"dataset": "test/audio", "field": "audio", "target_field": "label"})
        audio_rows = [
            {"audio": [0.05 * (j % 10) for j in range(800)], "label": f"audio-command-{i}"}
            for i in range(10)
        ]
        audio_args = hf.parse_args([
            "--incoming", str(omni_incoming), "--dataset", "test/audio",
            "--field", "audio", "--target-field", "label",
            "--modality", "audio", "--min-chars", "1",
            "--chunk-mib", "0.02", "--max-record-mib", "0.05", "--prefetch-mib", "1.0", "--once"
        ])[0]
        hf.run(audio_args, [audio_source], "test-version", lambda s, saved, c: FakeReader(audio_rows, saved))

        omni_run = self.root / "omni_run"
        command = [
            os.environ["SAIL_TEST_BINARY"], "stream-train", "--multimodal",
            "--incoming", str(omni_incoming), "--out", str(omni_run),
            "--steps", "2", "--batch-size", "2", "--seq-len", "8",
            "--d-model", "16", "--n-layers", "2", "--d-state", "4",
            "--audio-channels", "80", "--patch-size", "16",
            "--save-every", "1", "--log-every", "1", "--viz-every", "0",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((omni_run / "model.sea").exists())
        self.assertTrue((omni_run / "checkpoint.json").exists())

        # Verify that multimodal checkpoint receipts enable durable cleanup of committed chunks
        spool = hf.Spool(omni_incoming, [audio_source], "test-version", 1, int(0.05 * hf.MIB))
        reclaimed_bytes = spool.reclaim(omni_run)
        self.assertGreater(reclaimed_bytes, 0)
        remaining_chunks = list(omni_incoming.glob("hf_*.omni.jsonl"))
        self.assertEqual(len(remaining_chunks), 0)

    def test_multimodal_audio_and_vision_streaming(self):
        # 1. Test Audio Modality streaming
        audio_source = hf.normalize_source({"dataset": "test/audio", "field": "audio", "target_field": "label"})
        audio_rows = [
            {"audio": [0.1 * ((i + j) % 10) for j in range(800)], "label": f"label-{i}"}
            for i in range(5)
        ]
        audio_args = hf.parse_args([
            "--incoming", str(self.incoming), "--dataset", "test/audio",
            "--field", "audio", "--target-field", "label",
            "--modality", "audio", "--min-chars", "1",
            "--chunk-mib", "0.01", "--max-record-mib", "0.05", "--prefetch-mib", "1.0", "--once"
        ])[0]

        hf.run(audio_args, [audio_source], "test-version", lambda s, saved, c: FakeReader(audio_rows, saved))
        audio_chunks = list(self.incoming.glob("hf_*_*.omni.jsonl"))
        self.assertGreater(len(audio_chunks), 0)

        # Inspect generated record
        first_line = audio_chunks[0].read_text().splitlines()[0]
        record = json.loads(first_line)
        self.assertIn("audio", record)
        self.assertIsNotNone(record["audio"])
        self.assertEqual(record["audio"]["channels"], 80)
        self.assertIn("target", record)
        self.assertTrue(record["target"].startswith("label-"))

        # 2. Test Vision Modality streaming
        incoming_vision = self.root / "incoming_vision"
        vision_source = hf.normalize_source({"dataset": "test/vision", "field": "img", "target_field": "class"})
        vision_rows = [
            {"img": [128 for _ in range(3 * 32 * 32)], "class": f"cat-{i}"}
            for i in range(5)
        ]
        vision_args = hf.parse_args([
            "--incoming", str(incoming_vision), "--dataset", "test/vision",
            "--field", "img", "--target-field", "class",
            "--modality", "vision", "--min-chars", "1",
            "--chunk-mib", "0.2", "--max-record-mib", "0.2", "--prefetch-mib", "2.0", "--once"
        ])[0]

        hf.run(vision_args, [vision_source], "test-version", lambda s, saved, c: FakeReader(vision_rows, saved))
        vision_chunks = list(incoming_vision.glob("hf_*_*.omni.jsonl"))
        self.assertGreater(len(vision_chunks), 0)
        v_line = vision_chunks[0].read_text().splitlines()[0]
        v_record = json.loads(v_line)
        self.assertIn("vision", v_record)
        self.assertIsNotNone(v_record["vision"])
        self.assertEqual(v_record["vision"]["channels"], 3)
        self.assertEqual(v_record["vision"]["height"], 32)
        self.assertEqual(v_record["vision"]["width"], 32)
        self.assertTrue(v_record["target"].startswith("cat-"))

    def test_single_producer_lock_and_resume_version_guard(self):
        self.incoming.mkdir()
        with hf.DirectoryLock(self.incoming/"lock"):
            with self.assertRaisesRegex(RuntimeError,"another downloader"):
                with hf.DirectoryLock(self.incoming/"lock"):
                    pass
        self.spool()
        with self.assertRaisesRegex(ValueError,"version differs"):
            hf.Spool(self.incoming,[self.source],"new-version",1,int(.0001*hf.MIB))

    def test_sqlite_deduplication_persists_across_restarts(self):
        identical_rows = [
            {"text": "identical record for testing restart deduplication" * 2},
            {"text": "identical record for testing restart deduplication" * 2},
        ]
        args1 = self.args("--max-chunks", "1", "--batch-rows", "1")
        hf.run(args1, [self.source], "test-version", lambda s, saved, c: FakeReader(identical_rows, saved))
        chunks_after_first = list(self.incoming.glob("hf_*.txt"))
        self.assertEqual(len(chunks_after_first), 1)

        # Second run with identical data against the same spool: should detect duplicate via SQLite and publish 0 new chunks
        args2 = self.args("--max-chunks", "1", "--batch-rows", "1")
        hf.run(args2, [self.source], "test-version", lambda s, saved, c: FakeReader(identical_rows, saved))
        chunks_after_second = list(self.incoming.glob("hf_*.txt"))
        self.assertEqual(len(chunks_after_second), 1)

    def test_multimodal_different_images_same_label_both_survive(self):
        vision_source = hf.normalize_source({"dataset": "test/vision", "field": "img", "target_field": "label"})
        vision_rows = [
            {"img": [0.2] * 3072, "label": "1"},
            {"img": [0.8] * 3072, "label": "1"},
        ]
        saved = {"cursor": None, "rows": 0, "accepted": 0, "sequence": 0, "exhausted": False}
        reader = FakeReader(vision_rows, saved)
        dedup_store = hf.DedupStore(self.incoming / ".hf_streamer" / "dedup.db")
        updated, text, is_omni, batch_hashes = hf.collect(
            reader, saved, vision_source,
            chunk_bytes=100000, batch_rows=2, min_chars=1, max_record_bytes=100000,
            modality="vision", dedup_store=dedup_store
        )
        self.assertEqual(updated["accepted"], 2)
        self.assertEqual(len(batch_hashes), 2)
        lines = [line for line in text.strip().split("\n") if line]
        self.assertEqual(len(lines), 2)

    def test_split_contamination_guard(self):
        val_source = hf.normalize_source({"dataset": "test/source", "split": "validation"})
        train_source = hf.normalize_source({"dataset": "test/source", "split": "train"})
        shared_text = "held out text"
        val_rows = [{"text": shared_text}]
        train_rows = [{"text": shared_text}, {"text": "clean training"}]

        val_args = self.args("--max-chunks", "1")
        hf.run(val_args, [val_source], "test-version", lambda s, saved, c: FakeReader(val_rows, saved))

        db = hf.DedupStore(self.incoming / ".hf_streamer" / "dedup.db")
        doc_hash = __import__("hashlib").sha256(shared_text.encode("utf-8")).hexdigest()[:32]
        self.assertEqual(db.get_split(doc_hash), "validation")

        train_args = self.args("--max-chunks", "1")
        hf.run(train_args, [train_source], "test-version", lambda s, saved, c: FakeReader(train_rows, saved))

        all_text = "".join(p.read_text(encoding="utf-8") for p in self.incoming.glob("hf_*.txt"))
        self.assertIn("clean training", all_text)

    def test_ttsa_cooldown_boundary_and_exposure_backoff(self):
        db = hf.DedupStore(self.incoming / ".hf_streamer" / "dedup.db")
        doc_hash = "abc00011122233344455566677788899"
        cooldown = 1000  # 1000 seconds

        # 1. First stage at t = 1000: novel document is permitted
        self.assertTrue(db.check_and_stage(doc_hash, "train", now=1000, cooldown_secs=cooldown))
        db.commit_records([(doc_hash, "test/src", "train", "chunk1.txt")], now=1000, cooldown_secs=cooldown)

        # 2. Re-check at t = 1500 (before eligible_after = 2000): must be rejected (cooldown active)
        self.assertFalse(db.check_and_stage(doc_hash, "train", now=1500, cooldown_secs=cooldown))

        # 3. Check across split at any time: must be rejected (contamination guard never expires)
        self.assertFalse(db.check_and_stage(doc_hash, "validation", now=5000, cooldown_secs=cooldown))

        # 4. Re-check at t = 2000 (cooldown boundary reached): must be permitted
        self.assertTrue(db.check_and_stage(doc_hash, "train", now=2000, cooldown_secs=cooldown))

        # 5. Commit second exposure at t = 2000: backoff increases eligible_after
        db.commit_records([(doc_hash, "test/src", "train", "chunk2.txt")], now=2000, cooldown_secs=cooldown)
        cursor = db.conn.execute("SELECT exposure_count, eligible_after FROM documents WHERE doc_hash = ?", (doc_hash,))
        exp, eligible_after = cursor.fetchone()
        self.assertEqual(exp, 2)
        # For exp=2, backoff = 1000 * (1.5**2) = 2250 -> eligible_after = 2000 + 2250 = 4250
        self.assertEqual(eligible_after, 4250)
        self.assertFalse(db.check_and_stage(doc_hash, "train", now=4000, cooldown_secs=cooldown))
        self.assertTrue(db.check_and_stage(doc_hash, "train", now=4300, cooldown_secs=cooldown))

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
