import copy
import errno
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orpheus_utils import atomic_json, latest_checkpoint, verify_checkpoint
from pipeline_recovery import (DeferredUploads, compatible_encoding_identity,
                               export_checkpoint_adapter, optional_evaluation)
from test_pipeline import fake_checkpoint, FakeHub


class LateFailureTests(unittest.TestCase):
    def test_corrupt_newest_uses_older_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            good = fake_checkpoint(run / 'checkpoint-1')
            bad = fake_checkpoint(run / 'checkpoint-100')
            (bad / 'optimizer.pt').write_text('broken')
            (run / 'checkpoint-invalid').mkdir()
            self.assertEqual(latest_checkpoint(run), good)
            self.assertTrue(bad.exists())

    def test_no_valid_checkpoint_does_not_silently_restart_training(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); bad = fake_checkpoint(run / 'checkpoint-100')
            (bad / 'COMPLETE.json').write_text('{broken')
            with self.assertRaisesRegex(RuntimeError, 'No verified checkpoint'):
                latest_checkpoint(run)

    def test_empty_manifest_cannot_mark_adapter_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            atomic_json(Path(d) / 'COMPLETE.json', {'files': {}})
            with self.assertRaisesRegex(RuntimeError, 'Incomplete checkpoint manifest'):
                verify_checkpoint(d)

    def test_adapter_export_reuses_bytes_and_survives_checkpoint_removal(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); cp = fake_checkpoint(run / 'checkpoint-1')
            final = export_checkpoint_adapter(cp, run / 'adapter_final')
            self.assertEqual((cp / 'adapter_model.safetensors').stat().st_ino,
                             (final / 'adapter_model.safetensors').stat().st_ino)
            shutil.rmtree(cp)
            self.assertEqual((final / 'adapter_model.safetensors').read_bytes(), b'adapter_model.safetensors')

    def test_export_falls_back_on_filesystem_without_hardlinks(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); cp = fake_checkpoint(run / 'checkpoint-1')
            with patch('os.link', side_effect=OSError(errno.EOPNOTSUPP, 'no hardlinks')):
                final = export_checkpoint_adapter(cp, run / 'adapter_final')
            self.assertEqual((final / 'adapter_model.safetensors').read_bytes(), b'adapter_model.safetensors')

    def test_upload_status_disk_error_does_not_lose_in_memory_queue(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); source = run / 'report.json'; source.write_text('{}')
            store = DeferredUploads(SimpleNamespace(remote='hf://private', put=Mock(side_effect=TimeoutError())), run)
            with patch('pipeline_recovery.atomic_json', side_effect=OSError(errno.ENOSPC, 'full')):
                store.put(source, 'preparation/report.json')
            self.assertIn('preparation/report.json', store.queue['files'])

    def test_remote_restore_outage_uses_verified_local_state(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); cp = fake_checkpoint(run / 'checkpoint-1')
            store = DeferredUploads(SimpleNamespace(remote='hf://private', restore=Mock(side_effect=TimeoutError())), run)
            store.restore(run)
            self.assertEqual(latest_checkpoint(run), cp)

    def test_restore_outage_without_local_state_still_fails(self):
        with tempfile.TemporaryDirectory() as d:
            store = DeferredUploads(SimpleNamespace(remote='hf://private', restore=Mock(side_effect=TimeoutError())), Path(d))
            with self.assertRaises(TimeoutError):
                store.restore(Path(d))

    def test_corrupted_optional_upload_queue_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); (run / 'pending_uploads.json').write_text('{broken')
            store = DeferredUploads(SimpleNamespace(remote='hf://private'), run)
            self.assertEqual(store.queue['files'], {})
            self.assertIsNone(store.queue['checkpoint'])

    def test_nonfinite_eval_is_recorded_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            result = optional_evaluation(lambda: {'eval_loss': float('nan')}, d, 100, lambda: None)
            self.assertEqual(result, {})
            self.assertEqual(json.loads((Path(d) / 'evaluation_status.json').read_text())['status'], 'failed')

    def test_eval_disk_error_and_failed_error_logging_do_not_abort(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('pipeline_recovery.atomic_json', side_effect=OSError(errno.ENOSPC, 'full')), patch('orpheus_utils.append_jsonl', side_effect=OSError(errno.ENOSPC, 'full')):
                result = optional_evaluation(Mock(side_effect=OSError('eval output failed')), d, 100, lambda: None)
            self.assertEqual(result, {})

    def test_storage_only_edit_can_reuse_tokens_but_encoding_change_cannot(self):
        import shutil
        root = Path(__file__).resolve().parents[1]
        old = {'model': 'same', 'splits': 'same', 'encoding_source_sha256': {'old': 'hash'}}
        new = dict(old, encoding_source_sha256={'new': 'hash'})
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)
            for name in ('finetune_aslp_50h.py', 'orpheus_utils.py', 'pipeline_recovery.py'):
                shutil.copy2(root / name, source / name)
            p = source / 'orpheus_utils.py'
            p.write_text(p.read_text() + '\ndef unrelated_storage_function():\n    pass\n')
            self.assertIs(compatible_encoding_identity(old, new, source, root), old)
            p.write_text(p.read_text().replace('len(ids) > max_length', 'len(ids) >= max_length'))
            self.assertIs(compatible_encoding_identity(old, new, source, root), new)
            changed_data = dict(new, splits='different')
            self.assertIs(compatible_encoding_identity(old, changed_data, root, root), changed_data)

    def test_changing_live_logs_cannot_corrupt_snapshot_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'run'; cp = fake_checkpoint(run / 'checkpoint-1')
            log = run / 'run.log'; log.write_text('before upload')
            events = run / 'tensorboard/events'; events.parent.mkdir(); events.write_text('old events')
            hub = FakeHub(private=True)
            original_commit = hub.create_commit
            def commit(**kwargs):
                log.write_text('log changed by upload library')
                events.write_text('new events flushed asynchronously')
                return original_commit(**kwargs)
            hub.create_commit = commit
            hub.store().backup(run, cp)
            restored = Path(d) / 'restored'; restored.mkdir()
            hub.store().restore(restored)
            self.assertEqual((restored / 'run.log').read_text(), 'before upload')
            self.assertEqual((restored / 'tensorboard/events').read_text(), 'old events')
            self.assertTrue(verify_checkpoint(restored / 'checkpoint-1'))
