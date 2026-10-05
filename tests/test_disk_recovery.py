import errno
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pipeline_recovery import checkpoint_budget, ensure_checkpoint_space, save_with_space_retry

class DiskTests(unittest.TestCase):
    def test_budget_includes_frozen_embeddings(self):
        p = SimpleNamespace(numel=lambda: 100, element_size=lambda: 2, requires_grad=False)
        layer = SimpleNamespace(parameters=lambda: [p])
        model = SimpleNamespace(parameters=lambda: [], get_input_embeddings=lambda: layer,
                                get_output_embeddings=lambda: layer)
        self.assertEqual(checkpoint_budget(model), 800 + 256 * 1024**2)

    def test_rust_enospc_retries_current_state(self):
        operation = Mock(side_effect=[RuntimeError('I/O error: No space left on device (os error 28)'), 'saved'])
        recovery = Mock()
        self.assertEqual(save_with_space_retry(operation, recovery), 'saved')
        recovery.assert_called_once()
        self.assertEqual(operation.call_count, 2)

    def test_other_failures_not_hidden(self):
        recovery = Mock()
        with self.assertRaises(ValueError):
            save_with_space_retry(Mock(side_effect=ValueError('bad tensors')), recovery)
        recovery.assert_not_called()

    def test_low_disk_preserves_latest_and_pending(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            cps = [run / f'checkpoint-{i}' for i in (1,2,3)]
            for p in cps: p.mkdir()
            with patch('orpheus_utils.latest_checkpoint', return_value=cps[2]), patch('pipeline_recovery.verify_checkpoint', return_value=True), patch('shutil.disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaises(OSError) as exc:
                    ensure_checkpoint_space(run, 1, remote=True, protected=[cps[1]])
            self.assertEqual(exc.exception.errno, errno.ENOSPC)
            self.assertFalse(cps[0].exists())
            self.assertTrue(cps[1].exists())
            self.assertTrue(cps[2].exists())

    def test_reported_kaggle_capacity_supports_direct_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=11344220160)):
                report = ensure_checkpoint_space(d, 4903436288, remote=True, archive_snapshots=False)
            self.assertTrue(report['enough'])
            self.assertEqual(report['required_bytes'], 4903436288 + 512 * 1024**2)
            self.assertEqual(report['snapshot_format'], 'direct_files')

    def test_archive_backend_still_reserves_upload_copy(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('shutil.disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)):
                report = ensure_checkpoint_space(d, 4903436288, remote=True)
            self.assertEqual(report['required_bytes'], 2 * 4903436288 + 512 * 1024**2)

    def test_later_save_reclaims_old_checkpoint_without_deleting_latest(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            first, latest = run / 'checkpoint-1', run / 'checkpoint-100'
            first.mkdir(); latest.mkdir()
            estimate = 4903436288
            def capacity(_):
                count = len(list(run.glob('checkpoint-*')))
                return SimpleNamespace(free=11344220160 - count * estimate)
            with patch('orpheus_utils.latest_checkpoint', return_value=latest), patch('pipeline_recovery.verify_checkpoint', return_value=True), patch('shutil.disk_usage', side_effect=capacity):
                report = ensure_checkpoint_space(run, estimate, remote=True, archive_snapshots=False)
            self.assertEqual(report['removed_old_checkpoints'], ['checkpoint-1'])
            self.assertTrue(latest.exists())
            self.assertTrue(report['enough'])
