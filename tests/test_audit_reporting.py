"""CPU tests for audit diagnostics without changing dataset acceptance policy."""
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import finetune_aslp_50h as pipeline
from orpheus_utils import SnapshotStore


class AuditReportingTests(unittest.TestCase):
    def test_summary_preserves_all_rows_and_split_specific_counts(self):
        problems = [dict(split='test', row=4, audio='a.wav', error='Duplicate path across/in splits: train'),
                    dict(split='test', row=5, audio='b.wav', error='Duplicate path across/in splits: train'),
                    dict(split='train', row=8, audio='c.wav', error='Expected 24kHz mono PCM16; found stereo')]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            original = json.dumps(problems)
            summary = pipeline.report_audit_errors(run, problems)
            self.assertEqual(json.dumps(problems), original)
            self.assertEqual(summary['total'], 3)
            self.assertIn(dict(split='test', reason='duplicate_path', count=2), summary['groups'])
            self.assertEqual(json.loads((run / 'data_error_summary.json').read_text()), summary)

    def test_backup_failure_preserves_original_audit_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'data_errors.json').write_text('[]')
            store = SimpleNamespace(put=lambda *args: (_ for _ in ()).throw(RuntimeError('offline')))
            with patch.object(pipeline, 'audit_data', side_effect=ValueError('invalid rows')):
                with self.assertRaisesRegex(ValueError, 'invalid rows'):
                    pipeline.audit_with_reports(run, {}, run, store)

    def test_audit_mode_exits_before_model_imports_or_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(work_dir=directory, run_id='audit-test', checkpoint_backend='local')
            with patch.object(pipeline, 'cli', return_value=(SimpleNamespace(mode='audit'), cfg)), \
                 patch.object(pipeline, 'checkpoint_store', return_value=nullcontext(SnapshotStore())), \
                 patch.object(pipeline, 'find_data', return_value=Path(directory)), \
                 patch.object(pipeline, 'audit_with_reports') as audit, \
                 patch.object(pipeline, 'train') as train:
                pipeline.main()
                audit.assert_called_once()
                train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
