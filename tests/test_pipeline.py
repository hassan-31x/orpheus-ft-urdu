"""CPU regression tests for codec order, data safety and resumable artifacts."""
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from orpheus_utils import (SPECIAL, DriveStore, HuggingFaceStore, SnapshotStore, atomic_json, clean_text, contained,
                           decode_frames, fingerprint, interleave_codes, latest_checkpoint,
                           safe_unzip, safe_untar, seal_checkpoint, training_row, verify_checkpoint)
from evaluate_asr import distance, normalize


class HubError(Exception):
    def __init__(self, status, message="simulated Hub failure"):
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status)


class FakeHub:
    """In-memory Hub API double: synchronous commits are all-or-nothing."""
    def __init__(self, private=None):
        self.private = private
        self.files = {}
        self.commits = []
        self.commit_failure = None
        self.created = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)
        if self.private is None:
            self.private = kwargs["private"]

    def repo_info(self, **kwargs):
        if self.private is None:
            raise HubError(404)
        return SimpleNamespace(private=self.private)

    def upload_file(self, **kwargs):
        assert kwargs["repo_type"] == "model" and kwargs["run_as_future"] is False
        self.files[kwargs["path_in_repo"]] = Path(kwargs["path_or_fileobj"]).read_bytes()

    def create_commit(self, **kwargs):
        if self.commit_failure:
            raise self.commit_failure
        assert kwargs["repo_type"] == "model" and kwargs["run_as_future"] is False
        updates = {o.path_in_repo: Path(o.path_or_fileobj).read_bytes() for o in kwargs["operations"]}
        self.files.update(updates)
        self.commits.append(tuple(updates))

    def list_repo_files(self, **kwargs):
        return list(self.files)

    def download(self, **kwargs):
        dest = Path(kwargs["local_dir"]) / kwargs["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[kwargs["filename"]])
        return str(dest)

    def store(self, run_id="experiment"):
        return HuggingFaceStore("student/private-checkpoints", run_id, token="test-secret",
                                api=self, download=self.download, commit_add=SimpleNamespace)


def fake_checkpoint(path):
    path.mkdir(parents=True)
    for name in ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                 "optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"]:
        (path / name).write_bytes(name.encode())
    seal_checkpoint(path)
    return path


class CodecTests(unittest.TestCase):
    def test_roundtrip_and_codebook_order(self):
        codes = ([10, 11], [20, 21, 22, 23], list(range(30, 38)))
        ids = interleave_codes(*codes)
        decoded, status = decode_frames(ids + [SPECIAL["end_speech"], SPECIAL["end_ai"]])
        self.assertEqual(decoded, list(codes))
        self.assertTrue(status["ended_with_speech_stop"])
        self.assertEqual(status["complete_frames"], 2)

    def test_coarse_dedup_is_explicit_and_lossy(self):
        codes = ([1, 1], [2, 3, 4, 5], list(range(8)))
        self.assertEqual(len(interleave_codes(*codes)), 14)
        self.assertEqual(len(interleave_codes(*codes, dedup="coarse")), 7)

    def test_invalid_frame_and_partial_tokens_reported(self):
        ids = interleave_codes([1], [2, 3], [4, 5, 6, 7])
        _, status = decode_frames(ids + [1]*7 + [SPECIAL["end_speech"]])
        self.assertEqual(status["invalid_frame_token_offset"], 7)
        _, status = decode_frames(ids + [128266])
        self.assertEqual(status["trailing_tokens"], 1)
        self.assertFalse(status["ended_with_speech_stop"])

    def test_audio_loss_masks_context_but_learns_delimiters(self):
        prefix = [128259, 128000, 50, 128009, 128260, 128261, 128257]
        ids = interleave_codes([1], [2, 3], [4, 5, 6, 7])
        row = training_row(prefix, ids, "audio", 100, 160000)
        self.assertEqual(row["labels"][:6], [-100]*6)
        self.assertEqual(row["labels"][6], SPECIAL["start_speech"])
        self.assertEqual(row["labels"][-2:], [128258, 128262])
        with self.assertRaises(ValueError):
            training_row(prefix, ids, "all", 10, 160000)

    def test_reject_invalid_codebook(self):
        with self.assertRaises(ValueError):
            interleave_codes([4096], [0, 0], [0]*4)


class DataTests(unittest.TestCase):
    def test_normalization_preserves_urdu(self):
        self.assertEqual(clean_text("آج\x00  موسم\n خوشگوار ہے۔"), "آج موسم خوشگوار ہے۔")
        self.assertEqual(normalize("آج، موسم!"), "آج موسم")
        self.assertEqual(distance("abc", "axc"), 1)

    def test_fingerprint_independent_of_dict_key_order(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(fingerprint([1, 2]), fingerprint([2, 1]))

    def test_paths_and_archives_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            for path in ("../x", "/tmp/x", "C:/x", "..\\x"):
                with self.assertRaises(ValueError):
                    contained(d, path)
            archive = Path(d) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escaped", "bad")
            with self.assertRaises(ValueError):
                safe_unzip(archive, Path(d) / "out")
            archive = Path(d) / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as t:
                m = tarfile.TarInfo("link")
                m.type = tarfile.SYMTYPE
                m.linkname = "/tmp"
                t.addfile(m)
            with self.assertRaises(ValueError):
                safe_untar(archive, Path(d) / "out")


class ArtifactTests(unittest.TestCase):
    def test_latest_ignores_unfinished_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cp = fake_checkpoint(root / "checkpoint-10")
            (root / "checkpoint-20").mkdir()
            self.assertEqual(latest_checkpoint(root), cp)
            (cp / "optimizer.pt").write_bytes(b"corrupted")
            with self.assertRaises(RuntimeError):
                verify_checkpoint(cp)

    def test_adapter_only_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "adapter_model.safetensors").write_bytes(b"weights")
            with self.assertRaises(RuntimeError):
                seal_checkpoint(d)

    def test_upload_failure_does_not_publish_latest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            cp = fake_checkpoint(root / "checkpoint-10")
            store = DriveStore("gdrive:study")
            with patch.object(store, "put", side_effect=RuntimeError("network down")) as upload:
                with self.assertRaises(RuntimeError):
                    store.backup(root, cp)
                self.assertEqual(upload.call_count, 1)
                self.assertTrue(upload.call_args.args[1].startswith("snapshots/"))

    def test_snapshot_roundtrip(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            run = base / "original"
            cp = fake_checkpoint(run / "checkpoint-12")
            atomic_json(run / "status.json", {"status": "checkpoint_saved"})
            remote = base / "remote"
            remote.mkdir()
            store = DriveStore("mock:study")

            def put(src, relative):
                dst = remote / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            def get(relative, dst):
                shutil.copy2(remote / relative, dst)

            with patch.object(store, "put", side_effect=put):
                store.backup(run, cp)
            restored = base / "restored"
            restored.mkdir()
            with patch.object(store, "names", return_value=["latest.json"]), patch.object(store, "get", side_effect=get):
                store.restore(restored)
            self.assertTrue(verify_checkpoint(restored / "checkpoint-12"))
            self.assertEqual(json.loads((restored / "status.json").read_text())["status"], "checkpoint_saved")


class HuggingFaceTests(unittest.TestCase):
    def test_preflight_creates_private_repo_and_checks_read_write(self):
        hub = FakeHub()
        store = hub.store()
        store.preflight()
        self.assertTrue(hub.private)
        self.assertTrue(hub.created[0]["private"])
        self.assertIn("runs/experiment/connection_probe.json", hub.files)
        self.assertEqual(store.names(), ["connection_probe.json"])

    def test_public_repo_is_rejected_before_any_upload(self):
        hub = FakeHub(private=False)
        with self.assertRaisesRegex(RuntimeError, "must be private"):
            hub.store().preflight()
        self.assertEqual(hub.files, {})

    def test_existing_private_repo_does_not_require_creation_permission(self):
        hub = FakeHub(private=True)
        hub.store().preflight()
        self.assertEqual(hub.created, [])

    def test_full_checkpoint_roundtrip_with_atomic_pointer(self):
        hub = FakeHub(private=True)
        store = hub.store()
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "original"
            cp = fake_checkpoint(run / "checkpoint-12")
            atomic_json(run / "status.json", {"status": "checkpoint_saved"})
            store.backup(run, cp)
            self.assertEqual(len(hub.commits), 1)
            self.assertGreater(len(hub.commits[0]), 2)
            self.assertFalse(any(k.endswith(".tar.gz") for k in hub.files))
            self.assertIn("runs/experiment/latest.json", hub.commits[0])
            restored = Path(d) / "restored"
            restored.mkdir()
            # New client/session must list and download the remote files.
            hub.store().restore(restored)
            self.assertTrue(verify_checkpoint(restored / "checkpoint-12"))
            self.assertEqual((restored / "checkpoint-12/optimizer.pt").read_bytes(), b"optimizer.pt")
            self.assertFalse(any(b"test-secret" in v for v in hub.files.values()))

    def test_failed_checkpoint_commit_preserves_previous_pointer(self):
        hub = FakeHub(private=True)
        store = hub.store()
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            store.backup(run, fake_checkpoint(run / "checkpoint-10"))
            previous = dict(hub.files)
            hub.commit_failure = HubError(403, "token test-secret must never be logged")
            with self.assertRaises(RuntimeError) as error:
                store.backup(run, fake_checkpoint(run / "checkpoint-20"))
            self.assertEqual(hub.files, previous)
            self.assertNotIn("test-secret", str(error.exception))

    def test_corrupted_remote_snapshot_is_rejected(self):
        hub = FakeHub(private=True)
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            hub.store().backup(run, fake_checkpoint(run / "checkpoint-10"))
            artifact = next(k for k in hub.files if k.endswith("optimizer.pt"))
            hub.files[artifact] = b"broken optimizer"
            restored = Path(d) / "restored"
            restored.mkdir()
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                hub.store().restore(restored)
            self.assertFalse((restored / "checkpoint-10").exists())

    def test_legacy_archive_restore_remains_supported(self):
        hub = FakeHub(private=True)
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "original"
            cp = fake_checkpoint(run / "checkpoint-12")
            from pipeline_recovery import export_checkpoint_adapter
            export_checkpoint_adapter(cp, run / 'adapter_final')
            legacy = DriveStore("mock:study")
            def put(path, relative):
                hub.files["runs/experiment/" + relative] = Path(path).read_bytes()
            with patch.object(legacy, "put", side_effect=put):
                legacy.backup(run, cp)
            restored = Path(d) / "restored"
            restored.mkdir()
            hub.store().restore(restored)
            self.assertTrue(verify_checkpoint(restored / "checkpoint-12"))
            self.assertEqual((restored / 'adapter_final/adapter_model.safetensors').read_bytes(),
                             (cp / 'adapter_model.safetensors').read_bytes())

    def test_preparation_and_cache_restore_are_scoped_to_run(self):
        hub = FakeHub(private=True)
        hub.files.update({"runs/experiment/preparation/resolved_config.json": b'{"seed":3407}',
                          "runs/experiment/cache/abc/train/0000000.parquet": b"tokens",
                          "runs/other/preparation/resolved_config.json": b'{"seed":42}'})
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            run.mkdir()
            hub.store().restore(run)
            self.assertEqual(json.loads((run / "resolved_config.json").read_text())["seed"], 3407)
            (run / "resolved_config.json").write_text('{"seed":123}')
            hub.store().restore(run)
            self.assertEqual(json.loads((run / "resolved_config.json").read_text())["seed"], 123)
            cache = Path(d) / "cache"
            hub.store().restore_files("cache/abc", cache)
            self.assertEqual((cache / "train/0000000.parquet").read_bytes(), b"tokens")

    def test_transient_errors_retry_but_auth_errors_stop(self):
        from unittest.mock import Mock
        store = FakeHub(private=True).store()
        call = Mock(side_effect=[HubError(503), HubError(503), "ok"])
        with patch("orpheus_utils.time.sleep") as sleep:
            self.assertEqual(store._call("test", call), "ok")
            self.assertEqual(sleep.call_count, 2)
        call = Mock(side_effect=HubError(401))
        with patch("orpheus_utils.time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                store._call("test", call)
            sleep.assert_not_called()


class StorageSelectionTests(unittest.TestCase):
    def test_local_only_does_not_require_any_credentials(self):
        from finetune_aslp_50h import checkpoint_store
        with patch.dict("os.environ", {}, clear=True):
            with checkpoint_store({"checkpoint_backend": "local"}) as store:
                self.assertIsInstance(store, SnapshotStore)
                self.assertIsNone(store.remote)

    def test_hub_requires_secret(self):
        from finetune_aslp_50h import checkpoint_store
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                with checkpoint_store({"checkpoint_backend": "huggingface"}):
                    pass

    def test_cli_storage_overrides(self):
        from finetune_aslp_50h import cli
        with patch("sys.argv", ["train", "--hf-repo-id", "student/checkpoints"]), patch.dict("os.environ", {}, clear=True):
            _, cfg = cli()
            self.assertEqual(cfg["checkpoint_backend"], "huggingface")
            self.assertEqual(cfg["hf_repo_id"], "student/checkpoints")
        with patch("sys.argv", ["train", "--local-only"]), patch.dict("os.environ", {}, clear=True):
            _, cfg = cli()
            self.assertEqual(cfg["checkpoint_backend"], "local")
        with patch("sys.argv", ["train", "--checkpoint-backend", "drive"]), patch.dict("os.environ", {}, clear=True):
            _, cfg = cli()
            self.assertEqual(cfg["checkpoint_backend"], "drive")

    def test_storage_check_exits_before_gpu_or_data_processing(self):
        from finetune_aslp_50h import main
        hub = FakeHub(private=True)
        with tempfile.TemporaryDirectory() as d:
            args = ["train", "--mode", "storage-check", "--work-dir", d,
                    "--hf-repo-id", "student/private-checkpoints"]
            with patch("sys.argv", args), patch.dict("os.environ", {"HF_TOKEN": "test-secret"}, clear=True), \
                    patch.dict("sys.modules", {"unsloth": None, "torch": None}), \
                    patch("finetune_aslp_50h.HuggingFaceStore", return_value=hub.store()), \
                    patch("finetune_aslp_50h.find_data") as find_data:
                main()
                find_data.assert_not_called()
            self.assertIn("runs/experiment/connection_probe.json", hub.files)


if __name__ == "__main__":
    unittest.main()
