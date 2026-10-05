"""Full training control-flow tests with fake GPU/framework objects (no GPU claim)."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import finetune_aslp_50h as pipeline
from pipeline_recovery import DeferredUploads
from test_recovery import Rows


class Parameter:
    requires_grad = True
    device = 'cuda'
    shape = (1,)
    def numel(self):
        return 1


class Tensor:
    def to(self, device):
        return self
    def item(self):
        return True
    def backward(self):
        pass


class Model:
    config = SimpleNamespace(use_cache=False)
    def parameters(self):
        return iter([Parameter()])
    def named_parameters(self):
        return iter([('lora', Parameter())])
    def train(self):
        return self
    def zero_grad(self, **kwargs):
        pass
    def __call__(self, **kwargs):
        return SimpleNamespace(loss=Tensor())


class TrainingFlowTests(unittest.TestCase):
    def execute(self, directory, fail_evaluation=False, fail_checkpoint_once=False, fail_reporting=False, fail_tensorboard=False):
        run = Path(directory)
        cfg = json.loads(pipeline.DEFAULTS.read_text())
        cfg.update(max_steps=1, sample_count=0, sample_at_start=False)
        train_rows = [dict(audio='train.wav', text='اردو', speaker=None, duration=10, source_row=0)]
        val_rows = [dict(audio='val.wav', text='اردو', speaker=None, duration=10, source_row=0)]
        splits = dict(train=train_rows, validation=val_rows)
        ds = {split: Rows([dict(audio=rows[0]['audio'], input_ids=[1, 2], labels=[1, 2],
                               attention_mask=[1, 1], length=2)]) for split, rows in splits.items()}
        for name, value in [('dataset_report.json', {'manifests': {'train': {'rows': 1}, 'validation': {'rows': 1}}}),
                            ('data_errors.json', []), ('encoding_errors.json', []),
                            ('training_manifests.json', splits)]:
            (run / name).write_text(json.dumps(value))
        tokenizer = SimpleNamespace(get_vocab=lambda: {'token': 1},
                                    save_pretrained=lambda p: (Path(p) / 'tokenizer.json').write_text('{}'))
        model = Model()
        fast = SimpleNamespace(from_pretrained=lambda **kw: (model, tokenizer),
                               get_peft_model=lambda model, **kw: model)
        cuda = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None,
                               memory_allocated=lambda: 0, max_memory_allocated=lambda: 0)
        torch = SimpleNamespace(cuda=cuda, tensor=lambda *a, **kw: Tensor(), empty=lambda *a, **kw: Tensor(),
                                long='long', uint8='uint8', float16='fp16', bfloat16='bf16',
                                isfinite=lambda loss: Tensor(), autocast=lambda **kw: nullcontext(),
                                random=SimpleNamespace(fork_rng=lambda **kw: nullcontext()))

        class Trainer:
            def __init__(self, **kwargs):
                self.args = kwargs['args']; self.callbacks = kwargs['callbacks']
                self.model = kwargs['model']; self.control = SimpleNamespace()
                self.save_attempts = 0
                self.state = SimpleNamespace(global_step=1, max_steps=1, epoch=1.)
            def train(self, resume_from_checkpoint=None):
                self.callbacks[0].on_log(self.args, self.state, self.control, logs={'loss': 1.})
                self.callbacks[0].on_log(self.args, self.state, self.control, logs={'loss': 0.9})
                self._save_checkpoint(self.model, None)
                self.callbacks[0].on_save(self.args, self.state, self.control)
                return SimpleNamespace(metrics={'train_loss': 1.})
            def _save_checkpoint(self, model, trial):
                cp = run / 'checkpoint-1'
                self.save_model(cp, _internal_call=True)
                for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt',
                             'training_args.bin', 'rng_state.pth'):
                    (cp / name).write_text('state')
            def save_metrics(self, name, value):
                if fail_reporting:
                    raise OSError('optional metrics writer unavailable')
                (run / (name + '_metrics.json')).write_text(json.dumps(value))
            def save_state(self):
                if fail_reporting:
                    raise OSError('root trainer-state export unavailable')
            def save_model(self, destination, **kwargs):
                destination = Path(destination)
                destination.mkdir(exist_ok=True)
                self.save_attempts += 1
                (destination / 'adapter_model.safetensors').write_text('model')
                if fail_checkpoint_once and self.save_attempts == 1:
                    raise RuntimeError('PytorchStreamWriter failed writing file: file write failed')
                (destination / 'adapter_config.json').write_text('{}')
            def evaluate(self, **kwargs):
                if fail_evaluation:
                    # Export must already be safe before optional final evaluation.
                    if not (run / 'adapter_final' / 'adapter_model.safetensors').exists():
                        raise AssertionError('Adapter export happened after evaluation')
                    raise RuntimeError('evaluation kernel unavailable')
                return {'full_validation_loss': 1.2}

        modules = {'torch': torch, 'numpy': SimpleNamespace(), 'psutil': SimpleNamespace(
                       Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))),
                   'unsloth': SimpleNamespace(FastLanguageModel=fast, is_bfloat16_supported=lambda: False),
                   'transformers': SimpleNamespace(Trainer=Trainer, TrainingArguments=lambda **kw: SimpleNamespace(**kw),
                                                   TrainerCallback=object)}
        if fail_tensorboard:
            writer = Mock()
            writer.add_scalar.side_effect = OSError('event writer failed')
            modules['torch.utils.tensorboard'] = SimpleNamespace(SummaryWriter=lambda **kwargs: writer)
        backend = SimpleNamespace(remote='hf://private', put=Mock(), backup=Mock())
        self.backend = backend
        store = DeferredUploads(backend, run)
        with patch.dict('sys.modules', modules), patch.object(pipeline, 'optional_plots'):
            pipeline.train(cfg, ds, tokenizer, splits, run, store, {'config': cfg}, pipeline.time.monotonic())
        if fail_tensorboard:
            self.assertEqual(writer.add_scalar.call_count, 1)
        return run

    def test_full_training_control_flow_seals_saves_and_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(directory)
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'smoke_complete')
            self.assertTrue((run / 'checkpoint-1' / 'COMPLETE.json').exists())
            self.assertTrue((run / 'adapter_final' / 'adapter_model.safetensors').exists())
            self.assertTrue(json.loads((run / 'backup_status.json').read_text())['remote_snapshot_current'])
            # One full snapshot per saved checkpoint (+ the final forced backup); sample
            # refreshes upload small monitoring files instead of a second full snapshot.
            snapshots = [Path(c.args[1]).name for c in self.backend.backup.call_args_list]
            self.assertEqual(snapshots, sorted(set(snapshots), key=snapshots.index) + snapshots[-1:])
            monitored = [c.args[1] for c in self.backend.put.call_args_list]
            self.assertIn('monitoring/status.json', monitored)

    def test_optional_evaluation_failure_keeps_completed_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(directory, fail_evaluation=True)
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'smoke_complete')
            self.assertEqual(json.loads((run / 'status.json').read_text())['validation'], {})
            self.assertEqual(json.loads((run / 'evaluation_status.json').read_text())['status'], 'failed')

    def test_checkpoint_write_failure_retries_and_finishes_same_training_call(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(directory, fail_checkpoint_once=True)
            self.assertTrue((run / 'checkpoint-1' / 'COMPLETE.json').exists())
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'smoke_complete')

    def test_reporting_failures_do_not_abort_final_export(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(directory, fail_reporting=True)
            self.assertTrue((run / 'adapter_final' / 'adapter_model.safetensors').exists())
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'smoke_complete')

    def test_live_tensorboard_failure_disables_only_that_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(directory, fail_tensorboard=True)
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'smoke_complete')
            self.assertEqual(len((run / 'metrics.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
