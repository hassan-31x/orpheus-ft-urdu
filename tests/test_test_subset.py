"""TEST mode: a seeded clip subset is applied before audit/encoding and labelled as smoke."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE_AUDIO = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "soundfile"))


class SubsetConfigTests(unittest.TestCase):
    def test_subset_requires_smoke_run_id(self):
        from finetune_aslp_50h import cli
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "c.json"
            config.write_text(json.dumps({"test_train_clips": 50}))
            with patch("sys.argv", ["train", "--config", str(config), "--local-only"]), \
                    patch("sys.stderr"), self.assertRaises(SystemExit):
                cli()
            config.write_text(json.dumps({"test_train_clips": 50, "run_id": "base-smoke-test50"}))
            with patch("sys.argv", ["train", "--config", str(config), "--local-only"]):
                _, cfg = cli()
            self.assertEqual(cfg["test_train_clips"], 50)

    def test_full_defaults_have_no_subset(self):
        from finetune_aslp_50h import DEFAULTS
        cfg = json.loads(DEFAULTS.read_text())
        self.assertIsNone(cfg["test_train_clips"])
        self.assertIsNone(cfg["test_validation_clips"])


@unittest.skipUnless(HAVE_AUDIO, "numpy/pandas/soundfile not installed")
class SubsetAuditTests(unittest.TestCase):
    def test_audit_uses_seeded_subset_with_original_row_numbers(self):
        import numpy as np
        import pandas as pd
        import soundfile as sf
        from finetune_aslp_50h import DEFAULTS, audit_data
        with tempfile.TemporaryDirectory() as d:
            d = str(Path(d).resolve())  # find_data() returns a resolved root
            root, run = Path(d) / "data", Path(d) / "run"
            root.mkdir(), run.mkdir()
            for offset, (split, count) in enumerate((("train", 20), ("validation", 6), ("test", 6))):
                rows = []
                for i in range(count):
                    name = f"{split}_{i}.wav"
                    wav = (np.sin(np.arange(2400) * (1 + i + 100 * offset) / 1000) * 0.1).astype("float32")
                    sf.write(root / name, wav, 24000, subtype="PCM_16")
                    rows.append(dict(audio=name, text="آج موسم اچھا ہے"))
                pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)
            cfg = json.loads(DEFAULTS.read_text())
            cfg.update(test_train_clips=5, test_validation_clips=2, minimum_train_hours=0)
            splits = audit_data(root, cfg, run)
            self.assertEqual([len(splits[s]) for s in ("train", "validation", "test")], [5, 2, 2])
            # source_row must index the ORIGINAL CSV, and the subset is deterministic.
            for row in splits["train"]:
                self.assertEqual(row["audio"], f"train_{row['source_row']}.wav")
            report = json.loads((run / "dataset_report.json").read_text())
            self.assertEqual(report["training_cap"], {"train": 5, "validation_and_test": 2})
            self.assertEqual(report["manifests"]["train"]["source_rows"], 20)
            (Path(d) / "run2").mkdir()
            again = audit_data(root, cfg, Path(d) / "run2")
            self.assertEqual([r["audio"] for r in again["train"]], [r["audio"] for r in splits["train"]])


if __name__ == "__main__":
    unittest.main()
