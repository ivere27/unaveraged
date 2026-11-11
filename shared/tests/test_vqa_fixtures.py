"""Public timing fixtures require explicit models and preserve their identities."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks import vqa_fixtures
from shared.model_support import sha256


class VqaFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'source'
        (self.source/'examples').mkdir(parents=True)
        examples = {'examples': [{'language': 'en', 'image': 'receipt.png', 'cases': [
            {'question': f'What is the {name}?', 'expected_answer': 'ok'}
            for name in ('phone number', 'store address', 'store name')]}]}
        (self.source/'examples/family_examples.json').write_text(json.dumps(examples))
        self.packages = {}
        for name in ('f32', 'w8a8', 'w4a8'):
            package = self.root/name
            package.mkdir()
            (package/'manifest.json').write_text(json.dumps({'graphs': {'encoder': {}, 'decoder': {}}}))
            for file in ('config.json', 'vocab.json', 'encoder/graph.json', 'decoder/graph.json',
                         'encoder/model.safetensors', 'decoder/model.safetensors'):
                path = package/file
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(f'{name}:{file}'.encode())
            self.packages[name] = package
        self.output = self.root/'output'
        self.arguments = ['vqa_fixtures.py', '--source', str(self.source), '--out', str(self.output)]
        for argument, name in (('fp32', 'f32'), ('w8a8', 'w8a8'), ('w4a8', 'w4a8')):
            self.arguments.extend([f'--{argument}-package', str(self.packages[name])])

    def run_fixture(self, *, mutate=False):
        closed = []
        tokens = {'f32': [1, 10, 2], 'w8a8': [1, 11, 2], 'w4a8': [1, 12, 2]}
        class Runner:
            def __init__(self, package):
                self.package = package
            def close(self):
                closed.append(self.package.name)
        class Tokenizer:
            def encode(self, question, *, add_eos):
                return [2]
            def decode(self, values, *, errors):
                return '<field>phone</field><op>identity</op><answer>ok</answer>'
        def generate(runner, image, ids):
            if mutate:
                (runner.package/'encoder/model.safetensors').write_bytes(b'changed during evaluation')
            return {'tokens': tokens[runner.package.name]}
        image = np.zeros((1, 1, 320, 672), dtype=np.float32)
        with patch.object(sys, 'argv', self.arguments), \
                patch.object(vqa_fixtures, 'load_tokenizer', return_value=Tokenizer()), \
                patch.object(vqa_fixtures, 'preprocess', return_value=image), \
                patch.object(vqa_fixtures, 'ExperimentRunner', Runner), \
                patch.object(vqa_fixtures, 'generate', side_effect=generate), \
                redirect_stdout(io.StringIO()):
            vqa_fixtures.main()
        return closed, tokens, image.tobytes()

    def test_each_model_path_is_required_without_creating_output(self):
        for option in ('--fp32-package', '--w8a8-package', '--w4a8-package'):
            arguments = self.arguments.copy()
            index = arguments.index(option)
            del arguments[index:index + 2]
            error = io.StringIO()
            with self.subTest(option=option), patch.object(sys, 'argv', arguments), redirect_stderr(error):
                with self.assertRaises(SystemExit) as raised:
                    vqa_fixtures.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(option, error.getvalue())
            self.assertFalse(self.output.exists())

    def test_three_model_fixture_contains_exact_hashes_and_shared_fp32_prefix(self):
        closed, tokens, image = self.run_fixture()
        fixture = json.loads((self.output/'fixtures.json').read_text())
        report = json.loads((self.output/'public-native.json').read_text())
        self.assertEqual(set(fixture['models']), {'f32', 'w8a8', 'w4a8'})
        self.assertEqual(closed, ['f32', 'w8a8', 'w4a8'])
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(len(fixture['cases']), 3)
        for name, model in fixture['models'].items():
            expected = {str(file.relative_to(self.packages[name])): sha256(file.read_bytes())
                        for file in self.packages[name].rglob('*') if file.is_file()}
            self.assertEqual(model['sha256'], expected)
            self.assertEqual(Path(model['directory']), self.packages[name])
            self.assertEqual(report['models'][name]['correct'], 3)
        for case in fixture['cases']:
            self.assertEqual((self.output/case['file']).read_bytes(), image)
            self.assertEqual(case['sha256'], sha256(image))
            self.assertEqual(case['decoder_prefix'], tokens['f32'])
            self.assertEqual(case['references'], {name: {'tokens': value} for name, value in tokens.items()})

    def test_package_changes_during_reference_generation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Model package changed'):
            self.run_fixture(mutate=True)
        self.assertFalse((self.output/'fixtures.json').exists())


if __name__ == '__main__':
    unittest.main()
