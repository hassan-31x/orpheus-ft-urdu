"""Fault-injection tests for unattended IO, cache, upload and memory recovery."""
import errno
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from orpheus_utils import atomic_json, seal_checkpoint, sha256
from pipeline_recovery import DeferredUploads, memory_selection, retry_io, valid_chunk, optional_evaluation


class Rows:
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __iter__(self):
        return iter(self.rows)
    def __getitem__(self, key):
        return [r[key] for r in self.rows] if isinstance(key, str) else self.rows[key]
    def filter(self, predicate):
        return Rows([row for row in self.rows if predicate(row)])


class OutOfMemoryError(RuntimeError):
    pass


def checkpoint(run):
    cp = run / 'checkpoint-1'
    cp.mkdir()
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json',
                 'optimizer.pt', 'scheduler.pt', 'training_args.bin', 'rng_state.pth'):
        (cp / name).write_text('state')
    seal_checkpoint(cp)
    return cp


class RecoveryTests(unittest.TestCase):
    def test_network_retry_then_success(self):
        operation = Mock(side_effect=[TimeoutError(), OSError(errno.ECONNRESET, 'reset'), 'ok'])
        sleep = Mock()
        self.assertEqual(retry_io(operation, sleep=sleep), 'ok')
        self.assertEqual(operation.call_count, 3)

    def test_disk_full_and_programming_errors_do_not_retry(self):
        for error in (OSError(errno.ENOSPC, 'full'), TypeError('bug')):
            operation = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                retry_io(operation, sleep=Mock())
            self.assertEqual(operation.call_count, 1)

    def test_invalid_and_corrupt_cache_can_be_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'chunk.parquet'; meta = p.with_suffix('.json')
            p.write_text('valid')
            atomic_json(meta, dict(row_identity='id', sha256=sha256(p), rows=1, excluded=[]))
            self.assertTrue(valid_chunk(p, meta, 'id', 1, lambda _: [1]))
            p.write_text('damaged')
            self.assertFalse(valid_chunk(p, meta, 'id', 1, lambda _: [1]))
            meta.write_text('broken json')
            self.assertFalse(valid_chunk(p, meta, 'id', 1, lambda _: [1]))

    def test_upload_outage_queues_files_and_checkpoint_then_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            now = [0]
            store = SimpleNamespace(remote='hf://private', put=Mock(side_effect=TimeoutError()),
                                    backup=Mock(side_effect=TimeoutError()))
            wrapper = DeferredUploads(store, run, clock=lambda: now[0])
            artifact = run / 'data.json'; artifact.write_text('{}')
            wrapper.put(artifact, 'preparation/data.json')
            wrapper.put(artifact, 'preparation/another.json')
            self.assertEqual(store.put.call_count, 1)  # Circuit breaker limits retries.
            cp = checkpoint(run)
            self.assertFalse(wrapper.backup(run, cp))
            self.assertTrue(cp.is_dir())
            self.assertFalse(json.loads((run / 'backup_status.json').read_text())['remote_snapshot_current'])
            now[0] = 61
            store.put.side_effect = None; store.backup.side_effect = None
            self.assertTrue(wrapper.backup(run, cp))
            status = json.loads((run / 'backup_status.json').read_text())
            self.assertTrue(status['remote_snapshot_current'])
            self.assertEqual(status['pending_files'], 0)

    def test_incomplete_checkpoint_cannot_be_claimed_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SimpleNamespace(remote='hf://private')
            with self.assertRaises(RuntimeError):
                DeferredUploads(store, directory).backup(directory, Path(directory) / 'checkpoint-1')

    def test_memory_probe_drops_padded_length_and_reuses_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Rows([dict(audio='long', length=31), dict(audio='same-pad', length=29),
                         dict(audio='short', length=16)])
            def probe(row):
                if row['length'] > 20:
                    raise OutOfMemoryError('out of memory')
            selected, rejected = memory_selection(data, directory, probe)
            self.assertEqual(selected['audio'], ['short'])
            self.assertEqual(len(rejected), 2)
            selected, rejected = memory_selection(data, directory, lambda _: None, resuming=True)
            self.assertEqual(selected['audio'], ['short'])

    def test_resume_refuses_to_change_training_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Rows([dict(audio='a', length=16)])
            with self.assertRaisesRegex(RuntimeError, 'cannot change'):
                memory_selection(data, directory, Mock(side_effect=OutOfMemoryError('out of memory')), resuming=True)

    def test_memory_probe_does_not_swallow_model_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                memory_selection(Rows([dict(audio='a', length=16)]), directory, Mock(side_effect=TypeError('bug')))

    def test_final_backup_retries_even_during_upload_cooldown(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            backend = SimpleNamespace(remote='hf://private', put=Mock(side_effect=TimeoutError()), backup=Mock())
            wrapper = DeferredUploads(backend, run, clock=lambda: 0)
            artifact = run / 'data.json'; artifact.write_text('{}')
            wrapper.put(artifact, 'data.json')
            self.assertTrue(wrapper.backup(run, checkpoint(run), force=True))
            backend.backup.assert_called_once()

    def test_failed_optional_evaluation_does_not_return_fake_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            cleanup = Mock()
            result = optional_evaluation(Mock(side_effect=OutOfMemoryError('out of memory')), directory, 10, cleanup)
            self.assertEqual(result, {})
            self.assertEqual(json.loads((Path(directory) / 'evaluation_status.json').read_text())['status'], 'failed')
            cleanup.assert_called_once()

    def test_successful_optional_evaluation_keeps_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = optional_evaluation(lambda: {'eval_loss': 1.5}, directory, 10, Mock())
            self.assertEqual(result['eval_loss'], 1.5)



if __name__ == '__main__':
    unittest.main()
