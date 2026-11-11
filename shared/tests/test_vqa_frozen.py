"""Selected VQA artifact identity, source provenance and clean package import."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vqa.frozen import import_model, profiles, validate_package
from shared.model_support import VERSION

SOURCE = ROOT/'.cache/benchmarks/vqa-lowbit-2026-10-26/models/w4a8'
RELEASE = ROOT/'.cache/vqa/huggingface'/profiles()['w4a8']['model_revision']


class FrozenVqaTests(unittest.TestCase):
    def test_imported_build_checks_prepare_missing_reference_once_and_publish_after_success(self):
        from digit.check import ensure_fp32_reference as digit_reference
        from vqa.check import ensure_fp32_reference as vqa_reference
        for project, prepare in (('digit', digit_reference), ('vqa', vqa_reference)):
            with self.subTest(project=project), tempfile.TemporaryDirectory() as temporary:
                packages = Path(temporary)/'packages'
                def convert(source, destination, variant, version, **options):
                    self.assertEqual((variant, version, options), ('fp32', VERSION, {'prepare_for_ptq': True}))
                    destination.mkdir()
                    (destination/'manifest.json').write_text('{}')
                with patch(f'{project}.model.convert_model', side_effect=RuntimeError('conversion failed')):
                    with self.assertRaisesRegex(RuntimeError, 'conversion failed'):
                        prepare(RELEASE, packages)
                    self.assertFalse((packages/'fp32-prepared').exists())
                with patch(f'{project}.model.convert_model', side_effect=convert) as conversion:
                    prepare(RELEASE, packages)
                    prepare(RELEASE, packages)
                    self.assertEqual(conversion.call_count, 1)
                    self.assertTrue((packages/'fp32-prepared/manifest.json').is_file())

    def test_import_refuses_missing_or_corrupt_candidate_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'output'
            with self.assertRaisesRegex(ValueError, 'requires --model-source'):
                import_model(None, destination, RELEASE)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                import_model(Path(temporary), destination, RELEASE)
            self.assertFalse(destination.exists())

    @unittest.skipUnless((SOURCE/'manifest.json').is_file() and (RELEASE/'manifest.json').is_file(),
                         'Original frozen VQA candidate/release is not cached')
    def test_real_import_preserves_identity_and_excludes_experiment_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'output'
            manifest = import_model(SOURCE, destination, RELEASE)
            actual = {str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
            self.assertEqual(actual, {*profiles()['w4a8']['sha256'], 'manifest.json'})
            text = (destination/'manifest.json').read_text()
            for excluded in ('ordered_inputs_sha256', 'excluded_inputs_sha256', '/home/',
                             'calibration-inputs', 'images_directory', 'questions_file'):
                self.assertNotIn(excluded, text)
            self.assertEqual(manifest['variant'], 'ptq')
            self.assertEqual(manifest['quantization']['packed_arithmetic'], 'dynamic-i8')
            self.assertEqual(validate_package(destination), manifest)
            # Ordinary subsequent builds reuse the clean cached candidate.
            reused = Path(temporary)/'reimported'
            self.assertEqual(import_model(destination, reused, RELEASE), manifest)
            self.assertEqual(validate_package(reused), manifest)
            # A cached manifest cannot silently select float arithmetic or add private records.
            manifest['quantization']['packed_arithmetic'] = 'f32'
            (destination/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                validate_package(destination)
            manifest['quantization']['packed_arithmetic'] = 'dynamic-i8'
            manifest['private_records'] = ['private evaluation detail']
            (destination/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                validate_package(destination)

    @unittest.skipUnless((SOURCE/'manifest.json').is_file(), 'Original frozen VQA candidate is not cached')
    def test_import_rejects_a_different_source_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                import_model(SOURCE, root/'output', root)
            self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
