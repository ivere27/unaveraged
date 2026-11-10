"""Default builds select training/PTQ from source even when old models exist."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from shared import build

ROOT = Path(__file__).resolve().parents[2]


class SourceBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='unaveraged source build ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for project in ('digit', 'vqa'):
            (self.root/project).mkdir()
            shutil.copy2(ROOT/project/'extension.json', self.root/project/'extension.json')
        # The source export test separately exercises copied, self-contained data.
        (self.root/'synth').symlink_to(ROOT/'synth', target_is_directory=True)
        self.args = argparse.Namespace(
            model='all', precision='default', variant='ptq', model_source=None,
            cpu_threads=3,
            package_only=False, fixtures=False, calibration_images=self.root/'synth/images',
            calibration_count=256, calibration_seed=20261027, calibration_questions=None, exclude_images=[])

    def selections(self):
        with patch.object(build, 'ROOT', self.root):
            return build.selected_builds(self.args)

    def test_clean_defaults_require_no_quantized_model_files(self):
        self.assertFalse((self.root/'models').exists())
        self.assertFalse((self.root/'.cache').exists())
        selected = self.selections()
        self.assertEqual([(c['id'], a.precision) for c, a in selected],
                         [('digit', 'conv4-linear2'), ('vqa', 'w4a8')])
        for _, args in selected:
            self.assertIsNone(args.model_source)
            self.assertFalse(args.frozen_default)

    def test_default_build_never_silently_imports_an_old_checkpoint(self):
        for project, precision in (('digit', 'conv4-linear2'), ('vqa', 'w4a8')):
            for directory in (self.root/'models'/project/precision,
                              self.root/'.cache'/project/'packages'/build.RUNTIME_VERSION/'ptq'/precision):
                directory.mkdir(parents=True)
                (directory/'manifest.json').write_text(json.dumps({'frozen_candidate': True}))
        for _, args in self.selections():
            self.assertIsNone(args.model_source)

    def test_explicit_frozen_import_remains_available(self):
        self.args.digit_model_source = self.root/'explicit-digit'
        self.args.vqa_model_source = self.root/'explicit-vqa'
        self.assertEqual([a.model_source for _, a in self.selections()],
                         [self.args.digit_model_source, self.args.vqa_model_source])

    def test_digit_checks_public_training_inputs_before_dependency_install(self):
        self.args.model = 'digit'
        (self.root/'synth').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing public QAT input'):
            self.selections()

    def test_source_archive_has_rebuild_code_without_fitted_model_binaries(self):
        self.assertIn('digit/rebuild.py', build.SOURCE_FILES)
        self.assertIn('digit/qat.py', build.SOURCE_FILES)
        self.assertFalse(any(name.startswith('models/') or name.endswith('.safetensors')
                             for name in build.SOURCE_FILES))

    def test_default_build_dispatches_training_and_ptq_and_preserves_cache_on_failure(self):
        from digit import rebuild
        runtime = self.root/'runtime'
        runtime.mkdir()
        for name in ('volvoxai.lite.min.js', 'volvoxai.lite.wasm'):
            (runtime/name).write_bytes(b'runtime fixture')
        source = self.root/'public-hf'
        for project in ('digit', 'vqa'):
            with self.subTest(project=project):
                self.args.model = project
                config, args = self.selections()[0]
                args.qat_backend = 'cuda'
                package = self.root/'.cache'/project/'packages'/build.RUNTIME_VERSION/'ptq'/args.precision
                package.mkdir(parents=True)
                (package/'manifest.json').write_text('previous completed model')
                converter = SimpleNamespace(convert_model=Mock(),
                    quantize_model=Mock(side_effect=RuntimeError('calibration failed')))
                with patch.object(build, 'ROOT', self.root), \
                     patch.object(build, 'installed_packages', return_value=(runtime, {})), \
                     patch.object(build, 'select_runtime', return_value=(runtime, {})), \
                     patch.object(build, 'fetch_release', return_value=source), \
                     patch.object(rebuild, 'rebuild_model', side_effect=RuntimeError('training failed')) as train, \
                     patch.object(build.importlib, 'import_module', return_value=converter):
                    with self.assertRaisesRegex(RuntimeError, 'training failed' if project == 'digit' else 'calibration failed'):
                        build.build(config, args)
                call = converter.convert_model.call_args
                self.assertEqual((call.args[0], *call.args[2:]), (source, 'fp32', build.RUNTIME_VERSION))
                self.assertTrue(call.kwargs['prepare_for_ptq'])
                if project == 'digit':
                    train.assert_called_once()
                    self.assertEqual(train.call_args.args[0], source)
                    self.assertEqual(train.call_args.kwargs, {'synth_root': self.root/'synth', 'backend': 'cuda',
                                                             'cpu_threads': 3})
                    converter.quantize_model.assert_not_called()
                else:
                    train.assert_not_called()
                    self.assertEqual(converter.quantize_model.call_args.kwargs['precision'], 'w4a8')
                    self.assertEqual(converter.quantize_model.call_args.kwargs['count'], 256)
                    self.assertEqual(converter.quantize_model.call_args.kwargs['cpu_threads'], 3)
                self.assertEqual((package/'manifest.json').read_text(), 'previous completed model')


if __name__ == '__main__':
    unittest.main()
