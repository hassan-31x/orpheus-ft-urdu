"""CPU regression tests for codec order, data safety and resumable artifacts."""
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from orpheus_utils import (SPECIAL, DriveStore, atomic_json, clean_text, contained,
                           decode_frames, fingerprint, interleave_codes, latest_checkpoint,
                           safe_unzip, safe_untar, seal_checkpoint, training_row, verify_checkpoint)
from evaluate_asr import distance, normalize


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


if __name__ == "__main__":
    unittest.main()
