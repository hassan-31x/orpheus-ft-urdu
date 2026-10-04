"""Install into a dedicated venv without changing Kaggle's notebook packages."""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys
import venv

CUDA_PACKAGES = ('torch', 'torchvision', 'torchaudio', 'triton', 'xformers')


def cuda_constraints(version=metadata.version):
    constraints = []
    for package in CUDA_PACKAGES:
        try:
            value = version(package)
        except metadata.PackageNotFoundError:
            continue
        constraints.append(f'{package}=={value}')
    if not any(item.startswith('torch==') for item in constraints):
        raise RuntimeError('Kaggle must provide Torch; setup will not download a replacement CUDA stack')
    return constraints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('/kaggle/working/orpheus-env'))
    parser.add_argument('--gpu', action='store_true', help='Also run CUDA imports and attention backward')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    target = args.venv.resolve()
    if Path(sys.prefix).resolve() == target:
        raise RuntimeError('Run this bootstrap using the notebook Python, not the training venv')
    # Inherit CUDA wheels to avoid several GB of duplicate downloads. Pip installs
    # shadowing project packages into the venv; the host kernel stays unchanged.
    locked = cuda_constraints()
    target.mkdir(parents=True, exist_ok=True)
    evidence = target.parent / (target.name + '-setup')
    evidence.mkdir(parents=True, exist_ok=True)
    constraints = evidence / 'cuda-constraints.txt'
    constraints.write_text('\n'.join(locked) + '\n')
    venv.EnvBuilder(system_site_packages=True, with_pip=True).create(target)
    python = str(target / 'bin' / 'python')
    command = [python, '-m', 'pip', 'install', '--upgrade', '--upgrade-strategy',
               'only-if-needed', '--constraint', str(constraints), '--report',
               str(evidence / 'installation.json'), '-r', str(repo / 'requirements-kaggle.txt')]
    # Stream output to both Kaggle and a persistent setup log, preserving failure.
    with (evidence / 'installation.log').open('a') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f'Installation failed. See {evidence}/installation.log. '
                               'CUDA wheels are locked; do not upgrade them blindly.')
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    (evidence / 'resolved-requirements.txt').write_text(freeze)
    command = [python, str(repo / 'check_environment.py'), '--report',
               str(evidence / ('gpu-check.json' if args.gpu else 'cpu-check.json'))]
    if args.gpu:
        command.append('--gpu')
    subprocess.run(command, check=True)
    print(f'TRAIN_PYTHON = {python!r}')


if __name__ == '__main__':
    main()
