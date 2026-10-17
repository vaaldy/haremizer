"""Installed-wheel boundary checks; run after a non-editable package installation."""
import importlib.resources
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest

import haremizer
from haremizer import prepare, refinement_loop


class PackageTests(unittest.TestCase):
    def run_child(self, argv, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(argv, cwd=directory, capture_output=True, text=True,
                                  timeout=30, **kwargs)

    def test_origin_is_installed_not_checkout(self):
        origin = Path(haremizer.__file__).resolve()
        self.assertTrue(origin.is_relative_to(Path(sysconfig.get_path('purelib')).resolve()), origin)
        result = self.run_child([sys.executable, '-I', '-c',
                                 'import haremizer; print(haremizer.__file__)'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()).resolve(), origin)

    def test_all_imports_and_cli_help_are_ml_lazy(self):
        modules = sorted(p.stem for p in Path(haremizer.__file__).parent.glob('*.py')
                         if p.stem != '__init__')
        code = '''
import builtins, importlib, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','diffusers','transformers','peft','accelerate','safetensors'}:
        raise AssertionError('ML import forbidden: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
for name in sys.argv[1:]:
    module = importlib.import_module('haremizer.' + name)
    if hasattr(module, 'main'):
        sys.argv = [module.__name__, '--help']
        try:
            runpy.run_module(module.__name__, run_name='__main__')
        except SystemExit as error:
            assert error.code == 0, (name, error.code)
'''
        result = self.run_child([sys.executable, '-I', '-B', '-c', code, *modules])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--workspace', result.stdout)

    def test_evaluation_resource_is_usable(self):
        resource = importlib.resources.files('haremizer').joinpath('evaluation-suite.json')
        with importlib.resources.as_file(resource) as path:
            frozen = prepare.freeze_suite(path)
        self.assertTrue(json.loads(frozen['frozen_json'])['cases'])

    def test_missing_installed_resource_fails_without_checkout_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / 'haremizer'
            shutil.copytree(Path(haremizer.__file__).parent, copied,
                            ignore=shutil.ignore_patterns('__pycache__'))
            (copied / 'evaluation-suite.json').unlink()
            code = (
                'import pathlib,sys; sys.path.insert(0,sys.argv[1]); '
                'from haremizer import prepare; '
                'prepare.freeze_suite(pathlib.Path(prepare.__file__).with_name("evaluation-suite.json"))'
            )
            result = self.run_child([sys.executable, '-I', '-B', '-c', code, directory])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FileNotFoundError', result.stderr)
            self.assertIn(str(copied / 'evaluation-suite.json'), result.stderr)
            output = Path(directory) / 'preparation.json'
            cli = (
                'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); '
                'sys.argv[0]="haremizer.prepare"; '
                'runpy.run_module("haremizer.prepare",run_name="__main__")'
            )
            result = self.run_child([sys.executable, '-I', '-B', '-c', cli, directory,
                                     '--output', str(output)])
            self.assertEqual(result.returncode, 2, result.stderr)
            plan = json.loads(output.read_text())
            self.assertIsNone(plan['evaluation_suite'])
            self.assertTrue(any('evaluation suite:' in blocker and
                                str(copied / 'evaluation-suite.json') in blocker
                                for blocker in plan['blockers']), plan['blockers'])

    def test_coordinator_module_launcher_uses_same_installation(self):
        result = self.run_child([*refinement_loop.module_argv(sys.executable, 'sample'), '--help'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--model', result.stdout)

    def test_coordinator_rejects_another_installation(self):
        argv = refinement_loop.module_argv(sys.executable, 'sample')
        argv[4] = argv[4].replace(str(refinement_loop.SOURCE), str(refinement_loop.SOURCE / 'wrong'))
        result = self.run_child([*argv, '--help'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('package origin mismatch', result.stderr)

    def test_matrix_regenerator_runs_outside_checkout(self):
        from haremizer.matrix import build
        with tempfile.TemporaryDirectory(prefix='matrix with spaces ') as directory:
            root = Path(directory)
            run = root / 'run'
            (run / 'baseline-primary').mkdir(parents=True)
            output = root / 'output'
            build(run, output, cell=40)
            result = self.run_child([str(output / 'regenerate.sh')])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / 'page-01.png').is_file())
