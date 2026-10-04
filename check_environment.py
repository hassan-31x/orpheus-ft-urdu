"""Validate the project's active dependency graph, with optional CUDA smoke checks."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def requirements(path):
    return [Requirement(line.split('#', 1)[0].strip())
            for line in Path(path).read_text().splitlines()
            if line.split('#', 1)[0].strip()]


def check_dependencies(roots, distribution=metadata.distribution):
    """Follow active dependency edges, including extras; ignore unrelated packages."""
    pending = [(None, req) for req in roots]
    checked, activated, errors = {}, {}, set()
    while pending:
        parent, req = pending.pop()
        name = canonicalize_name(req.name)
        if parent is None and req.marker and not req.marker.evaluate():
            continue
        try:
            dist = distribution(name)
        except metadata.PackageNotFoundError:
            errors.add(f'{parent or "project"} requires {req}: missing package')
            continue
        checked[name] = dist.version
        if req.specifier and not req.specifier.contains(dist.version, prereleases=True):
            errors.add(f'{parent or "project"} requires {req}: installed {dist.version}')
        extras = set(req.extras)
        previous = activated.get(name)
        if previous is not None and extras <= previous:
            continue
        extras |= previous or set()
        activated[name] = extras
        for value in dist.requires or []:
            dependency = Requirement(value)
            if dependency.marker is None or any(
                dependency.marker.evaluate({'extra': extra}) for extra in {''} | extras
            ):
                pending.append((name, dependency))
    return {'python': sys.version, 'packages': dict(sorted(checked.items())),
            'errors': sorted(errors), 'scope': 'active project dependency graph'}


def smoke(gpu=False):
    # CPU checks do not import Unsloth: its import requires an accelerator.
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'smoke.parquet'
        pq.write_table(pa.Table.from_pandas(pd.DataFrame({'text': ['اردو']})), path)
        assert pq.read_table(path)['text'][0].as_py() == 'اردو'
    wave = io.BytesIO()
    sf.write(wave, np.zeros(2400, dtype=np.float32), 24000, format='WAV', subtype='PCM_16')
    wave.seek(0)
    assert sf.read(wave)[1] == 24000
    if gpu:
        from unsloth import FastLanguageModel  # Must precede Torch/Transformers.
        import torch
        from transformers import Trainer, TrainingArguments, AutoTokenizer
        from snac import SNAC
        import bitsandbytes
        import xformers.ops
        from packaging.version import Version
        if Version(torch.__version__.split('+')[0]) < Version('2.6'):
            raise RuntimeError('Torch 2.6+ is required for resumable Trainer checkpoints')
        if not torch.cuda.is_available():
            raise RuntimeError('Enable a Kaggle NVIDIA GPU before running --gpu')
        x = torch.randn(1, 16, 2, 32, device='cuda', dtype=torch.float16, requires_grad=True)
        xformers.ops.memory_efficient_attention(x, x, x).float().sum().backward()
        torch.cuda.synchronize()
        return {'torch': torch.__version__, 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(0), 'attention_backward': 'passed'}
    return {'parquet': 'passed', 'audio': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements', type=Path, default=Path(__file__).with_name('requirements-kaggle.txt'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    report = check_dependencies(requirements(args.requirements))
    try:
        if not report['errors']:
            report['smoke'] = smoke(args.gpu)
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Checked {len(report['packages'])} project packages; report: {args.report}")
    for error in report['errors']:
        print(error, file=sys.stderr)
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
