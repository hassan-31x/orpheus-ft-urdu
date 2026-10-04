"""Regression checks for scoped dependency validation and CUDA preservation."""
import ast
import importlib.metadata as metadata
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from packaging.requirements import Requirement
from check_environment import check_dependencies
from setup_kaggle import cuda_constraints


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
