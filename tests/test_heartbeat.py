"""Stalls must be visible remotely, and a stuck upload must not freeze training."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orpheus_utils import HubOperationError
from pipeline_recovery import ACTIVITY, Heartbeat, set_activity
from test_pipeline import FakeHub


class HeartbeatTests(unittest.TestCase):
    def test_beat_uploads_stage_log_tail_and_stacks_on_its_own_lane(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "run.log").write_text("\n".join(f"line {i}" for i in range(1000)))
            (run / "stacks.log").write_text("Thread 0x1 (most recent call first):\n  File x.py")
            (run / "metrics.jsonl").write_text('{"step": 5}\n')
            store = SimpleNamespace(remote="hf://x", put_many=Mock())
            set_activity("uploading checkpoint-5", 5)
            Heartbeat(run, store).beat()
            items, = store.put_many.call_args.args
            self.assertEqual(store.put_many.call_args.kwargs, {"lane": "heartbeat"})
            names = {relative for _, relative in items}
            self.assertTrue({"monitoring/heartbeat.json", "monitoring/run_tail.log",
                             "monitoring/stacks_tail.log", "monitoring/metrics.jsonl"} <= names)
            beat = json.loads((run / "monitoring/heartbeat.json").read_text())
            self.assertEqual((beat["stage"], beat["step"]), ("uploading checkpoint-5", 5))
            tail = (run / "monitoring/run_tail.log").read_text().splitlines()
            self.assertEqual((len(tail), tail[-1]), (400, "line 999"))

    def test_beat_failure_is_optional(self):
        with tempfile.TemporaryDirectory() as d:
            store = SimpleNamespace(remote="hf://x", put_many=Mock(side_effect=OSError("offline")))
            Heartbeat(Path(d), store).beat()  # must not raise

    def test_start_and_stop_watchdog(self):
        with tempfile.TemporaryDirectory() as d:
            store = SimpleNamespace(remote=None, put_many=Mock())
            beat = Heartbeat(Path(d), store, interval=3600, stack_interval=3600).start()
            beat.stop()
            self.assertTrue((Path(d) / "stacks.log").exists())
            self.assertTrue((Path(d) / "monitoring/heartbeat.json").exists())  # final beat


class UploadDeadlineTests(unittest.TestCase):
    def test_stalled_upload_times_out_and_blocks_only_its_lane(self):
        hub = FakeHub(private=True)
        release = threading.Event()
        hub.create_commit = lambda **kwargs: release.wait(5)
        store = hub.store()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.json"; f.write_text("{}")
            with patch.object(type(store), "UPLOAD_TIMEOUTS", {"checkpoint": 0.05, "files": 0.05, "heartbeat": 0.05}):
                with self.assertRaises(HubOperationError) as caught:
                    store.put_many([(f, "a.json")])
                self.assertEqual(caught.exception.diagnostics["category"], "timeout")
                with self.assertRaises(HubOperationError) as caught:
                    store.put_many([(f, "b.json")])  # same lane still busy: refuse, don't pile up
                self.assertTrue(caught.exception.diagnostics["still_running"])
                with self.assertRaises(HubOperationError) as caught:
                    store.put_many([(f, "c.json")], lane="heartbeat")  # other lane is independent
                self.assertFalse(caught.exception.diagnostics["still_running"])
            release.set()


if __name__ == "__main__":
    unittest.main()
