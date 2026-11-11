"""Public dataset auditing and fresh Digit QAT orchestration, without training."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from digit import rebuild

ROOT = Path(__file__).resolve().parents[2]


class DigitRebuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='digit public recipe ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.synth = self.root/'synth'
        self.release, self.prepared = self.root/'release', self.root/'prepared'
        for directory in (self.synth/'images', self.synth/'annotations', self.root/'digit',
                          self.release, self.prepared):
            directory.mkdir(parents=True)
        for name in ('package.json', 'digit/extension.json'):
            (self.root/name).write_bytes((ROOT/name).read_bytes())
        records = []
        for index in range(256):
            name = f'{index:06d}'
            image = self.synth/'images'/f'{name}.png'
            annotation = self.synth/'annotations'/f'{name}.json'
            Image.new('L', (1, 1), index).save(image)
            rebuild.write(annotation, {'answers': {'phone_digits': '0123456', 'street_no': index}})
            records.append({'id': name, 'image': f'images/{name}.png',
                            'annotation': f'annotations/{name}.json',
                            'sha256': rebuild.digest(image), 'annotation_sha256': rebuild.digest(annotation)})
        rebuild.write(self.synth/'manifest.json', {'count': 256, 'records': records})
        self.config = {'input': {'channels': 1, 'height': 320, 'width': 672,
                                 'normalization': 'fixture'}}
        rebuild.write(self.release/'config.json', self.config)
        rebuild.write(self.release/'manifest.json', {'release': 'fixture'})
        (self.release/'model.onnx').write_bytes(b'pinned-public-onnx-fixture')
        self.base = {
            'format': 'volvoxai-receipt-digit-reader-onnx-package-v1', 'variant': 'fp32',
            'source_format': 'receipt_digit_reader_onnx_v1',
            'source': {'sha256': rebuild._identity(self.release, rebuild.SOURCE_FILES)},
            'converter': {'package': 'volvoxai', 'version': rebuild.read(ROOT/'package.json')['config']['volvoxai_version'],
                          'source': 'pypi', 'prepare_for_ptq': True},
            'abi': {'input': {'name': 'input0', 'shape': [1, 1, 320, 672]}, 'output': 'slot_logits'},
            'decode': {'slots': 16, 'phone_slots': 12, 'street_slots': 4, 'num_classes': 11, 'blank_class': 10},
            'preprocess': self.config['input'], 'private_experiment_path': '/private/excluded'}
        rebuild.write(self.prepared/'manifest.json', self.base)
        (self.prepared/'graph.json').write_text('{}')
        (self.prepared/'model.safetensors').write_bytes(b'converted-fp32-fixture')
        self.graph = {'inputs': {'input0': {'shape': [1, 1, 320, 672], 'dtype': 'float32'}},
            'outputs': ['slot_logits'], 'nodes':
            [{'opType': 'QConv2D', 'params': {'weight_bits': 4}} for _ in range(11)] +
            [{'opType': 'PackedLinear', 'params': {'weight_bits': 2, 'group_size': 32}} for _ in range(7)]}
        self.graph['nodes'][-1]['outputs'] = {
            'output': {'tensor': 'slot_logits', 'shape': [1, 16, 11], 'dtype': 'float32'}}
        self.addCleanup(patch.stopall)
        patch.object(rebuild, 'ROOT', self.root).start()
        patch('shared.model_support.packaged_library_paths', return_value=()).start()

    def fake_training(self, command, **kwargs):
        self.command = command
        self.assertTrue(kwargs['check'])
        self.assertEqual(Path(command[command.index('--model')+1]), self.prepared)
        self.assertEqual(command[command.index('--calibration-backend')+1], 'cpu')
        self.assertIn('--allow-calibration-overlap', command)
        for name, value in rebuild.RECIPE.items():
            self.assertEqual(command[command.index('--'+name.replace('_', '-'))+1], str(value))
        out = Path(command[command.index('--out')+1])
        best = out/'best'
        best.mkdir(parents=True)
        rebuild.write(best/'graph.json', self.graph)
        (best/'model.safetensors').write_bytes(b'fresh-qat-fixture')
        rebuild.write(out/'report.json', {'status': 'passed', 'private': True,
            'settings': {'cpu_threads': int(command[command.index('--cpu-threads')+1])},
            'source_sha256': rebuild._identity(self.prepared), 'best_epoch': 2,
            'epochs': [{'epoch': 2, 'sha256': rebuild._identity(best),
                        'validation': {'all': {'samples': 32, 'record_accuracy': 1.0, 'cross_entropy': .001}}}]})

    def build(self, backend='cpu', cpu_threads=None):
        with patch.object(rebuild.subprocess, 'run', side_effect=self.fake_training):
            return rebuild.rebuild_model(self.release, self.root/'output', self.prepared,
                                         synth_root=self.synth, backend=backend, cpu_threads=cpu_threads)

    def test_split_and_targets_are_derived_from_verified_public_data(self):
        audit = rebuild.prepare_manifests(self.synth, self.root/'inputs')
        train = rebuild.read(self.root/'inputs/train.json')['items']
        calibration = rebuild.read(self.root/'inputs/calibration.json')['items']
        validation = rebuild.read(self.root/'inputs/validation.json')['items']
        self.assertEqual(train, calibration)
        self.assertEqual((len(train), len(validation)), (224, 32))
        self.assertEqual(train[0]['targets'], [0, 1, 2, 3, 4, 5, 6, 10, 10, 10, 10, 10, 0, 10, 10, 10])
        self.assertFalse({r['input_sha256'] for r in train} & {r['input_sha256'] for r in validation})
        self.assertEqual(audit['images'][0]['path'], 'synth/images/000000.png')
        self.assertEqual(audit['validation_images'][-1]['path'], 'synth/images/000255.png')
        self.assertNotIn(str(self.root), json.dumps(audit))

    def test_modified_image_or_annotation_is_rejected_before_training(self):
        for name, message in [('images/000000.png', 'image hash'), ('annotations/000000.json', 'annotation hash')]:
            with self.subTest(name=name):
                path = self.synth/name
                original = path.read_bytes()
                path.write_bytes(original+b'changed')
                with self.assertRaisesRegex(ValueError, message):
                    rebuild.prepare_manifests(self.synth, self.root/'inputs')
                path.write_bytes(original)

    def test_duplicate_validation_input_is_rejected(self):
        manifest = rebuild.read(self.synth/'manifest.json')
        first = self.synth/'images/000000.png'
        last = self.synth/'images/000255.png'
        last.write_bytes(first.read_bytes())
        manifest['records'][-1]['sha256'] = rebuild.digest(last)
        rebuild.write(self.synth/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'overlap or repeat'):
            rebuild.prepare_manifests(self.synth, self.root/'inputs')

    def test_cpu_default_invokes_exact_recipe_and_exports_sanitized_identity(self):
        manifest = self.build()
        self.assertEqual(self.command[self.command.index('--backend')+1], 'cpu')
        self.assertEqual(manifest['quantization']['training']['selected_epoch'], 2)
        self.assertEqual(manifest['model_sha256'], rebuild._identity(self.root/'output'))
        self.assertEqual(manifest['source_build']['prepared_sha256'], rebuild._identity(self.prepared))
        self.assertEqual(rebuild.validate_package(self.root/'output', self.release), manifest)
        self.assertNotIn('frozen_candidate', manifest)
        self.assertNotIn('/private', json.dumps(manifest))
        self.assertNotIn(str(self.root), json.dumps(manifest))

    def test_cuda_is_explicit_and_failed_subprocess_leaves_no_package(self):
        self.build('cuda')
        self.assertEqual(self.command[self.command.index('--backend')+1], 'cuda')
        with patch.object(rebuild.subprocess, 'run', side_effect=RuntimeError('training failed')):
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                rebuild.rebuild_model(self.release, self.root/'failed-output', self.prepared,
                                      synth_root=self.synth)
        self.assertFalse((self.root/'failed-output').exists())

    def test_jobs_environment_reaches_training_and_calibration_metadata(self):
        with patch.dict(os.environ, {'JOBS': '3'}):
            manifest = self.build()
        self.assertEqual(self.command[self.command.index('--cpu-threads')+1], '3')
        for name in ('training', 'calibration'):
            self.assertEqual(manifest['quantization'][name]['cpu_threads'], 3)

    def test_explicit_threads_override_jobs_and_legacy_manifests_remain_valid(self):
        with patch.dict(os.environ, {'JOBS': '7'}):
            manifest = self.build(cpu_threads=2)
        self.assertEqual(self.command[self.command.index('--cpu-threads')+1], '2')
        source_hashes = rebuild._identity(self.release, rebuild.SOURCE_FILES)
        for name in ('training', 'calibration'):
            self.assertEqual(manifest['quantization'][name].pop('cpu_threads'), 2)
        self.assertEqual(rebuild.validate_manifest(manifest, source_hashes), manifest)
        for name in ('training', 'calibration'):
            for invalid in (0, -1, True, '2', 1.5):
                changed = deepcopy(manifest)
                changed['quantization'][name]['cpu_threads'] = invalid
                with self.subTest(section=name, value=invalid), self.assertRaisesRegex(ValueError, 'CPU thread count'):
                    rebuild.validate_manifest(changed, source_hashes)

    def test_invalid_or_misreported_thread_count_does_not_publish_a_package(self):
        with patch.object(rebuild.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                rebuild.rebuild_model(self.release, self.root/'output', self.prepared,
                                      synth_root=self.synth, cpu_threads=0)
            run.assert_not_called()
        def wrong_count(command, **kwargs):
            self.fake_training(command, **kwargs)
            path = Path(command[command.index('--out')+1])/'report.json'
            report = rebuild.read(path)
            report['settings']['cpu_threads'] = 99
            rebuild.write(path, report)
        with patch.object(rebuild.subprocess, 'run', side_effect=wrong_count):
            with self.assertRaisesRegex(ValueError, 'CPU thread count changed'):
                rebuild.rebuild_model(self.release, self.root/'output', self.prepared,
                                      synth_root=self.synth, cpu_threads=2)
        self.assertFalse((self.root/'output').exists())

    def test_tampered_runtime_recipe_source_or_weights_is_rejected(self):
        original = self.build()
        source_hashes = rebuild._identity(self.release, rebuild.SOURCE_FILES)
        for mutate, message in [
            (lambda m: m['converter'].update(version='wrong'), 'runtime'),
            (lambda m: m['quantization']['training'].update(seed=1), 'recipe'),
            (lambda m: m['source_build'].update(model_revision='wrong'), 'provenance'),
            (lambda m: m['source']['sha256'].update({'model.onnx': '0'*64}), 'release'),
            (lambda m: m['quantization']['training'].update(validation_cross_entropy=float('nan')), 'metrics'),
        ]:
            manifest = deepcopy(original)
            mutate(manifest)
            with self.assertRaisesRegex(ValueError, message):
                rebuild.validate_manifest(manifest, source_hashes)
        (self.root/'output/model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'model hash'):
            rebuild.validate_package(self.root/'output', self.release)


class DigitQatThreadTests(unittest.TestCase):
    def test_both_quantization_stages_and_validation_receive_the_thread_budget(self):
        from digit import qat
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masters = root/'masters'
            masters.mkdir()
            graph = {'nodes': [{'opType': 'Linear'}, {'opType': 'Conv2D'}]}
            (masters/'graph.json').write_text(json.dumps(graph))
            def quantize(source, *, output, **kwargs):
                output.mkdir()
                (output/'graph.json').write_text(json.dumps(graph))
                (output/'model.safetensors').write_bytes(b'quantized fixture')
                return SimpleNamespace(calibration_backend=kwargs.get('calibration_backend'))
            with patch.object(qat.vx, 'quantize', side_effect=quantize) as convert:
                qat.quantize_candidate(masters, root/'candidate', [], 4, 2, 'cpu', cpu_threads=5)
            self.assertEqual([call.kwargs['cpu_threads'] for call in convert.call_args_list], [5, 5])
            with patch.object(qat.vx, 'InferenceSession') as session:
                qat.evaluate(root/'candidate', [], cpu_threads=5)
            self.assertEqual(session.call_args.kwargs['cpu_threads'], 5)

    def test_qat_cli_uses_one_synchronized_trainer_and_records_its_thread_count(self):
        from digit import qat
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            graph = {'inputs': {}, 'nodes': [
                {'id': str(index), 'opType': 'Conv2D', 'inputs': {}, 'outputs': {}} for index in range(11)]}
            (model/'graph.json').write_text(json.dumps(graph))
            (model/'model.safetensors').write_bytes(b'FP32 fixture')
            arguments = ['qat.py', '--model', str(model), '--out', str(root/'output'),
                         '--conv-bits', '4', '--linear-bits', '2', '--backend', 'cpu',
                         '--batch-size', '1', '--smoke', '--cpu-threads', '5']
            for index, name in enumerate(('train', 'validation', 'calibration')):
                path = root/f'{name}.json'
                path.write_text(json.dumps({'items': [{'input_sha256': str(index),
                                                       'source': 'synthetic', 'targets': [0] * 16}]}))
                arguments.extend(['--'+name, str(path)])
            trainer = MagicMock()
            trainer.backend = 'cpu'
            trainer.step.return_value.__enter__.return_value = SimpleNamespace(backend='cpu', loss=.25)
            with patch.object(sys, 'argv', arguments), patch.object(qat, 'load_file', return_value={}), \
                 patch.object(qat, 'packaged_library_paths', return_value=()), \
                 patch.object(qat, 'pixels', return_value=np.zeros((1, 1, 1, 1), np.float32)), \
                 patch.object(qat.vx, 'TrainingSession') as session, redirect_stdout(io.StringIO()):
                session.return_value.__enter__.return_value = trainer
                qat.main()
            session.assert_called_once()
            self.assertEqual(session.call_args.kwargs['cpu_threads'], 5)
            self.assertEqual(trainer.step.call_count, 3)
            report = json.loads((root/'output/report.json').read_text())
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['settings']['cpu_threads'], 5)


if __name__ == '__main__':
    unittest.main()
