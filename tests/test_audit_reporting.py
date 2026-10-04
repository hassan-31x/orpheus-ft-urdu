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

class RetentionPolicyTests(unittest.TestCase):
    def base(self):
        splits = {'train': [dict(duration=60)] * 99, 'validation': [dict(duration=10)], 'test': []}
        manifests = {'train': dict(rows=100), 'validation': dict(rows=1), 'test': dict(rows=423)}
        problems = [dict(split='train', duration_seconds=60)] + [dict(split='test', duration_seconds=10)] * 423
        cfg = dict(invalid_row_policy='skip', maximum_rejected_train_fraction=.05, minimum_train_hours=0)
        return splits, manifests, problems, cfg

    def test_small_training_loss_and_all_bad_test_rows_continue(self):
        summary, errors = pipeline.audit_retention(*self.base())
        self.assertEqual(errors, [])
        self.assertEqual(summary['train']['rejected_row_fraction'], .01)
        self.assertEqual(summary['test']['rejected_rows'], 423)

    def test_small_row_count_can_remove_too_many_audio_hours(self):
        splits, manifests, problems, cfg = self.base()
        problems[0]['duration_seconds'] = 3600
        _, errors = pipeline.audit_retention(splits, manifests, problems, cfg)
        self.assertTrue(any('hour_fraction' in e for e in errors))

    def test_unknown_duration_is_reported_and_counts_toward_row_limit(self):
        splits, manifests, problems, cfg = self.base()
        problems = [dict(split='train', duration_seconds=None)] * 6
        summary, errors = pipeline.audit_retention(splits, manifests, problems, cfg)
        self.assertEqual(summary['train']['rejected_unknown_duration_rows'], 6)
        self.assertTrue(any('row_fraction' in e for e in errors))

    def test_strict_mode_keeps_previous_behavior(self):
        splits, manifests, problems, cfg = self.base()
        cfg['invalid_row_policy'] = 'strict'
        _, errors = pipeline.audit_retention(splits, manifests, problems, cfg)
        self.assertTrue(any('strict' in e for e in errors))

    def test_empty_validation_still_fails(self):
        splits, manifests, problems, cfg = self.base()
        splits['validation'] = []
        _, errors = pipeline.audit_retention(splits, manifests, problems, cfg)
        self.assertTrue(any('validation' in e for e in errors))

    def test_optional_plot_failure_is_logged(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, 'plot_logs', side_effect=RuntimeError('plot')):
            pipeline.optional_plots(Path(directory))
            self.assertIn('plotting', (Path(directory) / 'optional_errors.jsonl').read_text())

class EncodingFilterTests(unittest.TestCase):
    def test_context_exclusion_is_reproduced_from_cache_without_codec_loading(self):
        from contextlib import nullcontext
        import sys
        from unittest.mock import MagicMock

        class Column(list):
            def tolist(self):
                return list(self)

        class Frame:
            def __init__(self, rows, columns=None):
                self.rows = rows
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, name):
                return Column(row[name] for row in self.rows)
            def to_parquet(self, path, **kwargs):
                Path(path).write_text(json.dumps(self.rows))

        def codes(frames):
            return [
                [SimpleNamespace(cpu=lambda values=[0]*n: SimpleNamespace(tolist=lambda: values))]
                for n in (frames, 2*frames, 4*frames)]

        codec = MagicMock()
        codec.encode.side_effect = [codes(1), codes(10), codes(1)]
        snac = SimpleNamespace(from_pretrained=MagicMock(return_value=codec))
        codec.eval.return_value.cuda.return_value = codec
        tensor = MagicMock()
        torch = SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda _: tensor,
                                cuda=SimpleNamespace(empty_cache=lambda: None))
        pandas = SimpleNamespace(DataFrame=Frame, read_parquet=lambda p: Frame(json.loads(Path(p).read_text())))
        datasets = SimpleNamespace(load_dataset=lambda kind, data_files, **kw: {
            split: [row for path in paths for row in json.loads(Path(path).read_text())]
            for split, paths in data_files.items()})
        modules = {'torch': torch, 'pandas': pandas, 'soundfile': SimpleNamespace(read=lambda *a, **k: ([], 24000)),
                   'snac': SimpleNamespace(SNAC=snac), 'datasets': datasets}
        rows = lambda: {'train': [dict(audio=name, text='اردو', speaker=None, duration=10, source_row=i)
                                 for i, name in enumerate(['good.wav', 'long.wav'])],
                         'validation': [dict(audio='val.wav', text='اردو', speaker=None, duration=10, source_row=0)]}
        cfg = dict(chunk_size=100, codec_revision='fixed', dedup='none', objective='all', max_length=32,
                   invalid_row_policy='skip', maximum_rejected_train_fraction=.75, minimum_train_hours=0)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
             patch.object(pipeline, 'prompt_ids', return_value=[1]):
            run = Path(directory)
            (run / 'data_errors.json').write_text('[]')
            (run / 'dataset_report.json').write_text(json.dumps({'manifests': {'train': {'rows': 2}, 'validation': {'rows': 1}}}))
            store = SimpleNamespace(restore_files=lambda *args: None, put=lambda *args: None)
            first_rows = rows()
            result = pipeline.encode_data(run, first_rows, cfg, None, 200000, run / 'cache', store, 'id', run)
            self.assertEqual(len(result['train']), 1)
            self.assertEqual(first_rows['train'][0]['audio'], 'good.wav')
            self.assertEqual(len(json.loads((run / 'encoding_errors.json').read_text())), 1)
            second_rows = rows()
            result = pipeline.encode_data(run, second_rows, cfg, None, 200000, run / 'cache', store, 'id', run)
            self.assertEqual(len(result['train']), 1)
            snac.from_pretrained.assert_called_once()
            # The same rejected rows are restored and count against strict mode.
            with self.assertRaisesRegex(ValueError, 'strict'):
                pipeline.encode_data(run, rows(), dict(cfg, invalid_row_policy='strict'), None,
                                     200000, run / 'cache', store, 'id', run)


if __name__ == '__main__':
    unittest.main()
