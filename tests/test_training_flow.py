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
    def execute(self, directory, fail_evaluation=False):
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
        cuda = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
        torch = SimpleNamespace(cuda=cuda, tensor=lambda *a, **kw: Tensor(), empty=lambda *a, **kw: Tensor(),
                                long='long', uint8='uint8', float16='fp16', bfloat16='bf16',
                                isfinite=lambda loss: Tensor(), autocast=lambda **kw: nullcontext(),
                                random=SimpleNamespace(fork_rng=lambda **kw: nullcontext()))

        class Trainer:
            def __init__(self, **kwargs):
                self.args = kwargs['args']; self.callbacks = kwargs['callbacks']
                self.state = SimpleNamespace(global_step=1, max_steps=1, epoch=1.)
            def train(self, resume_from_checkpoint=None):
                cp = run / 'checkpoint-1'; cp.mkdir()
                for name in ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json',
                             'optimizer.pt', 'scheduler.pt', 'training_args.bin', 'rng_state.pth'):
                    (cp / name).write_text('state')
                self.callbacks[0].on_save(self.args, self.state, None)
                return SimpleNamespace(metrics={'train_loss': 1.})
            def save_metrics(self, name, value):
                (run / (name + '_metrics.json')).write_text(json.dumps(value))
            def save_state(self):
                pass
            def save_model(self, destination):
                destination.mkdir()
                (destination / 'adapter_model.safetensors').write_text('model')
            def evaluate(self, **kwargs):
                if fail_evaluation:
                    # Export must already be safe before optional final evaluation.
                    if not (run / 'adapter_final' / 'adapter_model.safetensors').exists():
                        raise AssertionError('Adapter export happened after evaluation')
                    raise RuntimeError('evaluation kernel unavailable')
                return {'full_validation_loss': 1.2}

        modules = {'torch': torch, 'numpy': SimpleNamespace(), 'psutil': SimpleNamespace(),
                   'unsloth': SimpleNamespace(FastLanguageModel=fast, is_bfloat16_supported=lambda: False),
                   'transformers': SimpleNamespace(Trainer=Trainer, TrainingArguments=lambda **kw: SimpleNamespace(**kw),
                                                   TrainerCallback=object)}
        backend = SimpleNamespace(remote='hf://private', put=Mock(), backup=Mock())
        store = DeferredUploads(backend, run)
        with patch.dict('sys.modules', modules), patch.object(pipeline, 'optional_plots'):
            pipeline.train(cfg, ds, tokenizer, splits, run, store, {'config': cfg}, pipeline.time.monotonic())
        return run

    def test_full_training_control_flow_seals_saves_and_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(directory)
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'smoke_complete')
            self.assertTrue((run / 'checkpoint-1' / 'COMPLETE.json').exists())
            self.assertTrue((run / 'adapter_final' / 'adapter_model.safetensors').exists())
            self.assertTrue(json.loads((run / 'backup_status.json').read_text())['remote_snapshot_current'])

    def test_optional_evaluation_failure_keeps_completed_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.execute(directory, fail_evaluation=True)
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'smoke_complete')
            self.assertEqual(json.loads((run / 'status.json').read_text())['validation'], {})
            self.assertEqual(json.loads((run / 'evaluation_status.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
