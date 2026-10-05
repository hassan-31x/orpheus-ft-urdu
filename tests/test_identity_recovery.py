import tempfile
import unittest
from pathlib import Path
from orpheus_utils import atomic_json, read_json, seal_checkpoint
from pipeline_recovery import reconcile_run_identity

class IdentityRecoveryTests(unittest.TestCase):
    def test_preparation_only_updated_code_continues_and_archives_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            atomic_json(run / 'run_identity.json', {'source': 'old'})
            atomic_json(run / 'training_selection.json', {'accepted_audio': ['old.wav']})
            (run / 'metrics.jsonl').write_text('old loss\n')
            cache = run / 'cache'; cache.mkdir(); (cache / 'chunk').write_text('keep')
            partial = run / 'checkpoint-100'; partial.mkdir(); (partial / 'adapter_model.safetensors').write_text('partial')
            self.assertTrue(reconcile_run_identity(run, {'source': 'new'}))
            evidence = next((run / 'attempt_history').iterdir())
            self.assertEqual(read_json(evidence / 'run_identity.json'), {'source': 'old'})
            self.assertTrue((evidence / 'training_selection.json').exists())
            self.assertFalse((run / 'training_selection.json').exists())
            self.assertFalse((run / 'metrics.jsonl').exists())
            self.assertEqual((cache / 'chunk').read_text(), 'keep')
            self.assertTrue(partial.exists())

    def test_verified_progress_is_preserved_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); atomic_json(run / 'run_identity.json', {'source': 'old'})
            cp = run / 'checkpoint-1'; cp.mkdir()
            for name in ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json',
                         'optimizer.pt', 'scheduler.pt', 'training_args.bin', 'rng_state.pth'):
                (cp / name).write_text('state')
            seal_checkpoint(cp)
            with self.assertRaisesRegex(RuntimeError, 'Verified checkpoint'):
                reconcile_run_identity(run, {'source': 'new'})
            self.assertTrue((cp / 'COMPLETE.json').exists())
            self.assertEqual(read_json(run / 'run_identity.json'), {'source': 'old'})
            self.assertFalse((run / 'attempt_history').exists())

    def test_identical_identity_does_not_reset_selection(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); atomic_json(run / 'run_identity.json', {'source': 'same'})
            atomic_json(run / 'training_selection.json', {'accepted_audio': ['keep.wav']})
            self.assertFalse(reconcile_run_identity(run, {'source': 'same'}))
            self.assertTrue((run / 'training_selection.json').exists())
