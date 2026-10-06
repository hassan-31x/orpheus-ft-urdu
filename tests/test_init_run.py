"""init_from_run: a new run inherits a finished run's adapter and token cache."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orpheus_utils import fingerprint, seal_checkpoint


class FakePrior:
    def __init__(self, status="complete", with_checkpoint=True):
        self.status, self.with_checkpoint, self.cache_requests = status, with_checkpoint, []

    def restore(self, run):
        run = Path(run)
        (run / "status.json").write_text(json.dumps(dict(status=self.status, step=3000)))
        (run / "token_format.json").write_text(json.dumps(dict(format_version=1)))
        (run / "source").mkdir()
        (run / "source" / "orpheus_utils.py").write_text("X = 1\n")
        if self.with_checkpoint:
            cp = run / "checkpoint-3000"
            cp.mkdir()
            for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                         "optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"):
                (cp / name).write_text("weights" if name.startswith("adapter_model") else name)
            seal_checkpoint(cp)

    def restore_files(self, prefix, destination, skip_existing=False):
        self.cache_requests.append(prefix)


class InitRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name).resolve()
        self.run = self.work / "runs" / "epoch2"
        self.run.mkdir(parents=True)
        self.cfg = dict(init_from_run="epoch1", run_id="epoch2", hf_repo_id="u/r")

    def tearDown(self):
        self.tmp.cleanup()

    def stage(self, prior):
        from finetune_aslp_50h import stage_init_run
        return stage_init_run(self.cfg, self.work, self.run, "t", store_factory=lambda run_id: prior)

    def test_inherits_adapter_and_cache_identity(self):
        prior = FakePrior()
        target = self.stage(prior)
        self.assertEqual((target / "adapter_model.safetensors").read_text(), "weights")
        info = json.loads((self.run / "init_source.json").read_text())
        self.assertEqual((info["init_from_run"], info["step"]), ("epoch1", 3000))
        self.assertEqual(prior.cache_requests, ["cache/" + fingerprint(dict(format_version=1))])
        self.assertTrue((self.run / "token_format.json").is_file())
        self.assertFalse((self.work / "init_runs" / "epoch1").exists())  # staging disk released

    def test_second_call_does_not_refetch(self):
        self.stage(FakePrior())
        class Boom:
            def restore(self, run):
                raise AssertionError("must not download again")
        self.assertEqual(self.stage(Boom()), self.run / "init_adapter")

    def test_unfinished_or_smoke_source_refused(self):
        for status in ("smoke_complete", "paused_for_resume"):
            with self.assertRaisesRegex(RuntimeError, "not 'complete'"):
                self.stage(FakePrior(status=status))
            self.assertFalse((self.run / "init_adapter").exists())

    def test_missing_checkpoint_refused(self):
        with self.assertRaisesRegex(RuntimeError, "no verified checkpoint"):
            self.stage(FakePrior(with_checkpoint=False))

    def test_own_checkpoint_wins(self):
        cp = self.run / "checkpoint-1"
        FakePrior().restore(self.run)  # plants a sealed checkpoint-3000 in the new run
        self.assertIsNone(self.stage(FakePrior()))

    def test_config_validation(self):
        from finetune_aslp_50h import cli
        config = self.work / "c.json"
        config.write_text(json.dumps(dict(init_from_run="x", run_id="x", hf_repo_id="u/r")))
        with patch("sys.argv", ["t", "--config", str(config)]), patch("sys.stderr"), self.assertRaises(SystemExit):
            cli()
        config.write_text(json.dumps(dict(init_from_run="a", run_id="b", hf_repo_id="u/r")))
        with patch("sys.argv", ["t", "--config", str(config)]):
            self.assertEqual(cli()[1]["init_from_run"], "a")


if __name__ == "__main__":
    unittest.main()
