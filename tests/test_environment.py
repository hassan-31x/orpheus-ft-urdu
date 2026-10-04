"""Regression checks for scoped dependency validation and CUDA preservation."""
import ast
import importlib.metadata as metadata
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import subprocess
import tempfile
from unittest.mock import patch

from packaging.requirements import Requirement
from check_environment import check_dependencies
from setup_kaggle import cuda_constraints, create_environment


class EnvironmentTests(unittest.TestCase):
    def graph(self, values):
        def distribution(name):
            if name not in values:
                raise metadata.PackageNotFoundError(name)
            version, requires = values[name]
            return SimpleNamespace(version=version, requires=requires)
        return distribution

    def test_unrelated_kaggle_conflict_does_not_block(self):
        report = check_dependencies([Requirement('trainer==1')], self.graph({
            'trainer': ('1', ['hub<1']), 'hub': ('0.36', []),
            'gradio': ('6', ['hub>=1.16']), 'moviepy': ('1', ['decorator<5'])}))
        self.assertEqual(report['errors'], [])
        self.assertEqual(set(report['packages']), {'trainer', 'hub'})

    def test_actual_transitive_conflict_and_missing_dependency_fail(self):
        report = check_dependencies([Requirement('trainer')], self.graph({
            'trainer': ('1', ['hub<1', 'codec']), 'hub': ('2', [])}))
        self.assertEqual(len(report['errors']), 2)
        self.assertTrue(any('missing package' in error for error in report['errors']))

    def test_extras_activate_dependencies_and_markers(self):
        report = check_dependencies([Requirement('filesystem[http]')], self.graph({
            'filesystem': ('1', ['client>=2; extra == "http"',
                                 'unused; extra == "s3"', 'windows; sys_platform == "win32"']),
            'client': ('1', [])}))
        self.assertTrue(any('client' in error for error in report['errors']))
        self.assertNotIn('unused', report['packages'])

    def test_later_extra_on_same_distribution_is_processed(self):
        report = check_dependencies([Requirement('filesystem[http]'), Requirement('filesystem')],
            self.graph({'filesystem': ('1', ['client; extra == "http"'])}))
        self.assertEqual(len(report['errors']), 1)

    def test_cuda_versions_locked_including_local_build(self):
        def version(name):
            if name == 'torch':
                return '2.8.0+cu128'
            if name == 'xformers':
                return '0.0.32'
            raise metadata.PackageNotFoundError(name)
        self.assertEqual(cuda_constraints(version), ['torch==2.8.0+cu128', 'xformers==0.0.32'])

    def test_missing_torch_stops_bootstrap(self):
        def version(name):
            raise metadata.PackageNotFoundError(name)
        with self.assertRaises(RuntimeError):
            cuda_constraints(version)

    def test_venv_creation_uses_inherited_cuda_packages(self):
        with patch('setup_kaggle.venv.EnvBuilder') as builder:
            create_environment(Path('/tmp/env'), Path('/tmp/evidence'))
        builder.assert_called_once_with(system_site_packages=True, with_pip=True)
        builder.return_value.create.assert_called_once_with(Path('/tmp/env'))

    def test_missing_ensurepip_uses_separate_virtualenv_bootstrap(self):
        with patch('setup_kaggle.venv.EnvBuilder') as builder, patch('setup_kaggle.subprocess.run') as run:
            builder.return_value.create.side_effect = subprocess.CalledProcessError(1, 'ensurepip')
            create_environment(Path('/tmp/env'), Path('/tmp/evidence'))
        self.assertEqual(run.call_count, 2)
        self.assertIn('--target', run.call_args_list[0].args[0])
        self.assertIn('--system-site-packages', run.call_args_list[1].args[0])
        self.assertEqual(run.call_args_list[1].kwargs['env']['PYTHONPATH'], '/tmp/evidence/bootstrap')

    def test_fresh_notebook_installs_before_using_training_python(self):
        notebook = json.loads(Path('kaggle_run.ipynb').read_text())
        source = ''.join(notebook['cells'][4]['source'])
        with tempfile.TemporaryDirectory() as directory:
            # Simulate a fresh runtime: no training Python has been installed yet.
            scope = {'Path': Path, 'REPO': Path(directory), 'AUDIT_ONLY': False,
                     'sys': SimpleNamespace(executable='/host/python'), 'subprocess': subprocess}
            def simulate(command, **kwargs):
                if 'setup_kaggle.py' in str(command):
                    target = scope['ENV'] / 'bin' / 'python'
                    self.assertFalse(target.exists())
                    target.parent.mkdir(parents=True)
                    target.touch()
                return SimpleNamespace(returncode=0)
            with patch('subprocess.run', side_effect=simulate) as run:
                # Substitute only the environment destination for the test filesystem.
                exec(source.replace('/kaggle/working/orpheus-env', directory + '/env'), scope)
            first = run.call_args_list[0].args[0]
            self.assertEqual(first[0], '/host/python')
            self.assertIn('--gpu', first)
            self.assertIn('setup_kaggle.py', str(first))

    def test_notebook_accepts_real_repository_owner_without_blacklist(self):
        import re
        notebook = json.loads(Path('kaggle_run.ipynb').read_text())
        tree = ast.parse(''.join(notebook['cells'][2]['source']))
        guard = next(node for node in tree.body if isinstance(node, ast.If))
        condition = compile(ast.Expression(guard.test), '<repo-validation>', 'eval')
        for repo in ('hassan-31x/orpheus-urdu-checkpoints', 'student/private-model'):
            self.assertFalse(eval(condition, {'re': re, 'HF_REPO_ID': repo}))
        for repo in ('invalid', 'a/b/c', 'username/'):
            self.assertTrue(eval(condition, {'re': re, 'HF_REPO_ID': repo}))

    def test_notebook_uses_training_interpreter(self):
        notebook = json.loads(Path('kaggle_run.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            ast.parse(source)
            self.assertNotIn('"pip", "check"', source)
            if 'finetune_aslp_50h.py' in source or 'synthesize.py' in source:
                self.assertNotIn('[sys.executable', source)
                self.assertIn('TRAIN_PYTHON', source)


if __name__ == '__main__':
    unittest.main()
