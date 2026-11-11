"""Pages model staging verifies Release bytes before extracting public assets."""
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zipfile
import zlib
from unittest.mock import patch

from shared import pages


class PagesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = {
            'model/graph.json': b'{}', 'model/model.safetensors': b'weights',
            'model/manifest.json': b'{"format":"digit-fixture","abi":{"output":"slot_logits"},"quantization":{"packed_arithmetic":"dynamic-i8","calibration_path":"private/example"},"calibration_path":"private/example"}',
            'model/adapter.js': b'export const modelAdapter = {};',
            'model/LICENSE.txt': b'Apache-2.0',
            'vendor/volvoxai.lite.min.js': b'runtime', 'vendor/volvoxai.lite.wasm': b'wasm',
            'vendor/LICENSE.txt': b'runtime license',
            'shared/extension/runtime.js': b'runtime import',
            'shared/extension/preprocess.js': b'preprocess',
            'LICENSE.txt': b'extension license', 'NOTICE.txt': b'extension notice',
            'shared/extension/background.js': b'not needed on Pages',
        }
        self.info = {
            'extension': 'digit', 'runtime_profile': 'lite', 'model_precision': 'conv4-linear2',
            'runtime_version': '0.9.0',
            'model_sha256': {name[6:]: pages.digest(value) for name, value in self.payload.items()
                             if name in ('model/graph.json', 'model/model.safetensors', 'model/manifest.json')},
            'runtime_sha256': {name[7:]: pages.digest(value) for name, value in self.payload.items()
                              if name[7:] in pages.RUNTIME_FILES},
        }

    def archive(self, extra=None):
        path = self.root/'model.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, value in {**self.payload, 'build-info.json': json.dumps(self.info), **(extra or {})}.items():
                archive.writestr(name, value)
        return path

    def source_overlay_fixture(self, project):
        repository = Path(__file__).resolve().parents[2]
        sources = ['package.json', 'LICENSE', 'NOTICE', f'{project}/adapter.js',
                   'shared/extension/runtime.js', 'shared/extension/preprocess.js']
        if project == 'vqa':
            sources.append('vqa/tokenizer.js')
        for name in sources:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository/name, path)
        notice = (self.root/'NOTICE').read_bytes()
        self.payload['vendor/LICENSE.txt'] = b'MIT License\n\n' + notice.split(b'MIT License\n\n', 1)[1]
        self.payload = {name: data for name, data in self.payload.items() if not name.startswith('model/')}
        self.payload.update({f'model/{name}': f'unchanged {name}'.encode() for name in pages.MODEL_FILES[project]})
        self.payload.update({'model/adapter.js': b'old packaged adapter', 'model/LICENSE.txt': b'upstream model license'})
        if project == 'digit':
            manifest = {'format': 'volvoxai-receipt-digit-reader-onnx-package-v1',
                        'abi': {'input': {'shape': [1, 1, 320, 672], 'name': 'input0'}, 'output': 'slot_logits'},
                        'decode': {'slots': 16, 'num_classes': 11, 'phone_slots': 12,
                                   'street_slots': 4, 'blank_class': 10}}
        else:
            tokenizer = {'type': 'byte_fallback_bpe', 'version': 1, 'vocab_size': 1536,
                         'normalization': 'NFC', 'tokenizer_hash': 'a' * 64}
            cross = [f'cross_{kind}_{index}' for index in range(4) for kind in ('k', 'v')]
            inputs = {'encoder': ['image', 'question_ids', 'family_ids'],
                      'decoder': ['decoder_input_ids', 'position_ids', 'family_ids', 'memory_padding_mask',
                                  'past_padding_mask', *cross, *(name.replace('cross_', 'past_') for name in cross)]}
            manifest = {'format': 'receipt-vqa-volvoxai/v1', 'question_capacity': 192,
                        'max_decoder_steps': 191, 'initial_masked_cache_rows': 1, 'tokenizer': tokenizer,
                        'graphs': {role: {'inputs': {name: f'input{index}' for index, name in enumerate(names)},
                                          'weights': [f'{role}/model.safetensors', 'shared/token-embedding.safetensors']}
                                   for role, names in inputs.items()}}
            self.payload['model/vocab.json'] = json.dumps(tokenizer).encode()
            self.payload['model/tokenizer.js'] = b'old packaged tokenizer'
            self.payload['model/NOTICE.txt'] = b'upstream model notice'
        manifest['quantization'] = {'packed_arithmetic': 'dynamic-i8'}
        self.payload['model/manifest.json'] = json.dumps(manifest).encode()
        self.info.update({'extension': project, 'model_precision': 'conv4-linear2' if project == 'digit' else 'w4a8',
                          'runtime_version': json.loads((self.root/'package.json').read_text())['config']['volvoxai_version']})
        self.info['model_sha256'] = {name: pages.digest(self.payload[f'model/{name}'])
                                     for name in pages.MODEL_FILES[project] | {'manifest.json'}}
        return manifest

    def update_model_manifest(self, manifest):
        raw = json.dumps(manifest).encode()
        self.payload['model/manifest.json'] = raw
        self.info['model_sha256']['manifest.json'] = pages.digest(raw)

    def sample_fixture(self):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(b'\x00\xff\xff\xff')) + chunk(b'IEND', b''))
        examples = []
        for project in pages.MODELS:
            asset = f'{project}-receipt-ko.jpg'
            image = self.root/'docs/assets/examples'/asset
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'public JPEG fixture')
            examples.append({'asset': asset, 'model': project, 'synthetic': True,
                             'sha256': pages.digest(image.read_bytes()), 'width': 670, 'height': 320,
                             'questions': [{'question': '주소 숫자?', 'answer': '868', 'private': 'not published'}],
                             'receipt': {'phone_digits': '1234567', 'street_no': '868', 'private': 'not published'},
                             'private': '/private/example'})
        (image.parent/'sources.json').write_text(json.dumps({
            'format': 'unaveraged-public-synthetic-examples/v1', 'examples': examples}))
        synth = self.root/'synth'
        (synth/'images').mkdir(parents=True)
        (synth/'annotations').mkdir()
        annotation = json.dumps({
            'answers': {'phone_digits': '1611627', 'street_no': 868, 'address': '합성길 868', 'store': '합성 가게',
                        'items': [{'name': '상품', 'quantity': 2, 'unit_price': 100, 'total': 200,
                                   'private': '/private/item'}],
                        'private': '/private/receipt', 'layout': 'not needed'},
            'fields': [{'glyphs': [{'char': 'private glyph', 'poly': [[1, 2]]}]}],
            'private': '/private/annotation',
        }, ensure_ascii=False).encode()
        records = []
        for index in range(pages.SYNTH_COUNT):
            identifier = f'{index:06d}'
            (synth/f'images/{identifier}.png').write_bytes(png)
            (synth/f'annotations/{identifier}.json').write_bytes(annotation)
            records.append({'id': identifier, 'image': f'images/{identifier}.png', 'sha256': pages.digest(png),
                            'bytes': len(png), 'width': 1, 'height': 1,
                            'annotation': f'annotations/{identifier}.json', 'annotation_sha256': pages.digest(annotation),
                            'annotation_bytes': len(annotation), 'private': '/private/record'})
        manifest = {'schema_version': 1, 'id': 'receipt-initial-256', 'count': pages.SYNTH_COUNT,
                    'license': 'Apache-2.0', 'path_base': 'manifest_directory', 'records': records,
                    'total_image_bytes': len(png) * pages.SYNTH_COUNT,
                    'total_annotation_bytes': len(annotation) * pages.SYNTH_COUNT,
                    'generation': {'private': '/private/generator'}}
        (synth/'manifest.json').write_text(json.dumps(manifest))
        (self.root/'LICENSE').write_text('Apache-2.0 fixture')
        (self.root/'NOTICE').write_text('Project attribution fixture')
        return manifest

    def write_sample_manifest(self, manifest):
        (self.root/'synth/manifest.json').write_text(json.dumps(manifest))

    def test_preserves_relative_imports_and_licenses_with_exact_metadata(self):
        path = self.archive()
        target = self.root/'output'
        result = pages.stage_model(path, target, 'digit', pages.digest(path.read_bytes()))
        self.assertEqual(result['adapter'], 'runtime/digit/model/adapter.js')
        self.assertEqual(result['precision'], 'conv4-linear2')
        self.assertEqual((target/'model/model.safetensors').read_bytes(), b'weights')
        self.assertTrue((target/'model/LICENSE.txt').exists())
        self.assertTrue((target/'shared/extension/runtime.js').exists())
        self.assertFalse((target/'shared/extension/background.js').exists())
        contract = json.loads((target/'model/manifest.json').read_text())
        self.assertEqual(contract['format'], 'digit-fixture')
        self.assertEqual(contract['abi']['output'], 'slot_logits')
        self.assertEqual(contract['quantization'], {'packed_arithmetic': 'dynamic-i8'})
        self.assertNotIn('private/example', json.dumps(contract))
        self.assertFalse((target/'build-info.json').exists())

    def test_private_files_in_model_or_vendor_trees_are_not_published(self):
        private = {'model/private-input.png': b'private receipt', 'model/private.json': b'question/answer',
                   'vendor/.env': b'private-token', 'vendor/debug.js': b'private debug code'}
        target = self.root/'output'
        pages.stage_model(self.archive(private), target, 'digit')
        for name in private:
            self.assertFalse((target/name).exists())

    def test_current_source_overlay_preserves_uploaded_model_runtime_and_upstream_notices(self):
        for project in pages.MODELS:
            with self.subTest(project=project):
                self.source_overlay_fixture(project)
                path = self.archive()
                target = self.root/f'output-{project}'
                expected_hash = pages.digest(path.read_bytes())
                result = pages.stage_model(path, target, project, expected_hash, source_root=self.root)
                immutable = {*(f'model/{name}' for name in pages.MODEL_FILES[project]),
                             *(f'vendor/{name}' for name in pages.RUNTIME_FILES),
                             'model/LICENSE.txt', 'vendor/LICENSE.txt'}
                if project == 'vqa':
                    immutable.add('model/NOTICE.txt')
                for name in immutable:
                    self.assertEqual((target/name).read_bytes(), self.payload[name], name)
                for name, source in [('model/adapter.js', f'{project}/adapter.js'),
                                     ('shared/extension/runtime.js', 'shared/extension/runtime.js'),
                                     ('shared/extension/preprocess.js', 'shared/extension/preprocess.js'),
                                     ('LICENSE.txt', 'LICENSE'), ('NOTICE.txt', 'NOTICE'),
                                     *([('model/tokenizer.js', 'vqa/tokenizer.js')] if project == 'vqa' else [])]:
                    data = (self.root/source).read_bytes()
                    self.assertEqual((target/name).read_bytes(), data)
                    self.assertEqual(result['currentSourceSha256'][name], pages.digest(data))
                self.assertIn(self.payload['vendor/LICENSE.txt'].strip(), (target/'NOTICE.txt').read_bytes())
                self.assertEqual(result['archiveSha256'], expected_hash)
                self.assertEqual(result['downloadBytes'], sum(file.stat().st_size for file in target.rglob('*') if file.is_file()))

    def test_current_source_rejects_runtime_abi_or_license_mismatch_before_writing(self):
        manifest = self.source_overlay_fixture('digit')
        target = self.root/'output'
        original_version = self.info['runtime_version']
        self.info['runtime_version'] = '0.0.0'
        with self.assertRaisesRegex(ValueError, 'runtime version'):
            pages.stage_model(self.archive(), target, 'digit', source_root=self.root)
        self.assertFalse(target.exists())
        self.info['runtime_version'] = original_version
        manifest['abi']['input']['shape'] = [1, 1, 640, 672]
        self.update_model_manifest(manifest)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            pages.stage_model(self.archive(), target, 'digit', source_root=self.root)
        self.assertFalse(target.exists())
        manifest['abi']['input']['shape'] = [1, 1, 320, 672]
        self.update_model_manifest(manifest)
        (self.root/'NOTICE').write_text('License attribution accidentally omitted')
        with self.assertRaisesRegex(ValueError, 'full uploaded runtime license'):
            pages.stage_model(self.archive(), target, 'digit', source_root=self.root)
        self.assertFalse(target.exists())

    def test_current_vqa_source_rejects_incompatible_tokenizer_and_weight_paths(self):
        manifest = self.source_overlay_fixture('vqa')
        original = json.loads(json.dumps(manifest))
        for change in ('tokenizer', 'weights', 'decoder'):
            manifest = json.loads(json.dumps(original))
            if change == 'tokenizer':
                manifest['tokenizer']['tokenizer_hash'] = 'b' * 64
            elif change == 'weights':
                manifest['graphs']['decoder']['weights'] = ['../private.safetensors']
            else:
                manifest['max_decoder_steps'] = 192
            self.update_model_manifest(manifest)
            target = self.root/f'output-{change}'
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'incompatible'):
                pages.stage_model(self.archive(), target, 'vqa', source_root=self.root)
            self.assertFalse(target.exists())

    def test_verified_archive_bytes_are_used_without_reopening_mutable_source(self):
        path = self.archive()
        original = path.read_bytes()
        expected = pages.digest(original)
        def read_verified(_):
            path.write_bytes(b'changed after checksum read')
            return original
        with patch.object(Path, 'read_bytes', read_verified):
            result = pages.stage_model(path, self.root/'output', 'digit', expected)
        self.assertEqual(result['archiveSha256'], expected)
        self.assertEqual((self.root/'output/model/model.safetensors').read_bytes(), b'weights')

    def test_missing_integrity_entries_are_rejected_before_extracting(self):
        self.info['model_sha256'].pop('model.safetensors')
        with self.assertRaisesRegex(ValueError, 'asset hashes'):
            pages.stage_model(self.archive(), self.root/'output', 'digit')
        self.assertFalse((self.root/'output').exists())

    def site_fixture(self):
        source_root = Path(__file__).resolve().parents[2]
        for name in set(pages.docs.DOCUMENTS) | set(pages.docs.ASSETS):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root/name, path)
        for name in pages.EXAMPLE_FILES:
            path = self.root/'example'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('public frontend')
        for name in ('docs/.env', 'docs/private.png', 'example/browser.mjs', 'example/private.json'):
            (self.root/name).write_text('must not publish')
        self.sample_fixture()
        renderer = self.root/'shared/extension/heatmap-view.js'
        renderer.parent.mkdir(parents=True)
        renderer.write_text('public renderer')
        for project in pages.MODELS:
            config = self.root/project/'extension.json'
            config.parent.mkdir(exist_ok=True)
            config.write_text(json.dumps({'slug': f'unaveraged-{project}', 'default_precision': 'fixture'}))

    def test_frontend_staging_uses_explicit_public_allowlist_and_preserves_old_output_on_failure(self):
        self.site_fixture()
        output = self.root/'staged'
        with patch.object(pages, 'stage_model', return_value={'adapter': 'fixture'}):
            pages.stage(output, root=self.root)
        for name in ('docs/.env', 'docs/private.png', 'example/browser.mjs', 'example/private.json'):
            staged_name = name.removeprefix('docs/')
            self.assertFalse((output/staged_name).exists())
        self.assertTrue((output/'example/app.js').exists())
        self.assertTrue((output/'example/questions.js').exists())
        self.assertEqual(len(json.loads((output/'example/samples.json').read_text())['samples']), 258)
        (self.root/'PRIVACY.md').write_text('# Revised policy\n\nFresh source-only policy.\n')
        (self.root/'docs/privacy.html').write_text('stale generated policy')
        with patch.object(pages, 'stage_model', return_value={'adapter': 'fixture'}):
            pages.stage(output, root=self.root)
        self.assertIn('Fresh source-only policy.', (output/'privacy.html').read_text())
        self.assertNotIn('stale generated policy', (output/'privacy.html').read_text())
        with patch.object(pages, 'stage_model', side_effect=ValueError('invalid archive')):
            with self.assertRaisesRegex(ValueError, 'invalid archive'):
                pages.stage(output, root=self.root)
        self.assertEqual((output/'example/app.js').read_text(), 'public frontend')
        previous_catalog = (output/'example/samples.json').read_bytes()
        (self.root/'synth/annotations/000000.json').write_text('corrupt annotation')
        with patch.object(pages, 'stage_model', return_value={'adapter': 'fixture'}):
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                pages.stage(output, root=self.root)
        self.assertEqual((output/'example/samples.json').read_bytes(), previous_catalog)
        self.assertTrue((output/'example/synth/000255.png').exists())
        for forbidden in (self.root, self.root/'docs', self.root/'example', self.root/'synth', self.root/'synth/images'):
            with self.assertRaisesRegex(ValueError, 'dedicated output'):
                pages.stage(forbidden, root=self.root)

    def test_source_commit_stages_local_packages_without_a_release_and_pins_all_source_links(self):
        self.site_fixture()
        (self.root/'README.md').write_text('# Site\n\n[Build source](shared/build.py)\n')
        commit = 'a1' * 20
        output = self.root/'staged'
        with patch.object(pages, 'stage_model', return_value={'adapter': 'fixture'}) as stage_model:
            manifest = pages.stage(output, root=self.root, repo='owner/project',
                                   source_commit=commit.upper(), tag='not-an-actual-release', current_source=True)
        self.assertEqual(manifest['format'], 'unaveraged-demo/v1')
        self.assertEqual(manifest['release'], {'tag': 'local', 'url': ''})
        self.assertEqual(manifest['source'], {
            'repository': 'owner/project', 'commit': commit,
            'url': f'https://github.com/owner/project/commit/{commit}',
        })
        self.assertEqual(json.loads((output/'example/release.json').read_text()), manifest)
        for call, project in zip(stage_model.call_args_list, pages.MODELS):
            self.assertEqual(call.args[0], self.root/f'dist/{project}/fixture/unaveraged-{project}-chrome.zip')
            self.assertIsNone(call.args[3], 'Locally built packages do not require a Release checksum file')
            self.assertEqual(call.kwargs, {'source_root': self.root})
        self.assertIn(f'https://github.com/owner/project/blob/{commit}/shared/build.py',
                      (output/'index.html').read_text())
        source = json.loads((output/'example/synth/source.json').read_text())
        self.assertEqual(source['source_url'], f'https://github.com/owner/project/tree/{commit}/synth')

    def test_release_staging_without_source_commit_remains_compatible(self):
        self.site_fixture()
        release = self.root/'downloaded-release'
        release.mkdir()
        checksum = 'a' * 64
        (release/'SHA256SUMS').write_text(''.join(
            f'{checksum}  unaveraged-{project}-chrome.zip\n' for project in pages.MODELS))
        with patch.object(pages, 'stage_model', return_value={'adapter': 'fixture'}) as stage_model:
            manifest = pages.stage(self.root/'staged', root=self.root, repo='owner/project',
                                   release_dir=release, tag='v1.0')
        self.assertNotIn('source', manifest)
        self.assertEqual(manifest['release'], {
            'tag': 'v1.0', 'url': 'https://github.com/owner/project/releases/tag/v1.0',
        })
        for call, project in zip(stage_model.call_args_list, pages.MODELS):
            self.assertEqual(call.args[0], release/f'unaveraged-{project}-chrome.zip')
            self.assertEqual(call.args[3], checksum)

    def test_invalid_source_commit_is_rejected_before_creating_output(self):
        output = self.root/'staged'
        for commit in ('', 'main', 'a' * 7, 'g' * 40, '../private', 123):
            with self.subTest(commit=commit), self.assertRaisesRegex(ValueError, 'full 40-character'):
                pages.stage(output, root=self.root, source_commit=commit)
            self.assertFalse(output.exists())

    def test_samples_include_defaults_and_all_verified_synth_without_raw_annotations(self):
        manifest = self.sample_fixture()
        (self.root/'synth/images/private.png').write_bytes(b'not in manifest')
        target = self.root/'output/example'
        result = pages.stage_samples(self.root, target)
        self.assertEqual(result['format'], 'unaveraged-demo-samples/v1')
        self.assertEqual([row['defaultFor'] for row in result['samples'][:2]], list(pages.MODELS))
        self.assertEqual(result['samples'][0]['receipt'], {'phone_digits': '1234567', 'street_no': '868'})
        synth = result['samples'][2:]
        self.assertEqual(len(synth), pages.SYNTH_COUNT)
        self.assertEqual(synth[0]['id'], 'synth-000000')
        self.assertEqual(synth[-1]['id'], 'synth-000255')
        self.assertEqual(synth[0]['models'], ['digit', 'vqa'])
        self.assertEqual(synth[0]['receipt'], {
            'phone_digits': '1611627', 'street_no': 868, 'address': '합성길 868', 'store': '합성 가게',
            'items': [{'name': '상품', 'quantity': 2, 'unit_price': 100, 'total': 200}],
        })
        for row, original in zip(synth, manifest['records']):
            self.assertEqual(pages.digest((target/row['image']).read_bytes()), original['sha256'])
        self.assertEqual(len(list((target/'synth').glob('*.png'))), pages.SYNTH_COUNT)
        self.assertFalse((target/'synth/annotations').exists())
        self.assertFalse((target/'synth/private.png').exists())
        self.assertFalse((target/'synth/manifest.json').exists())
        catalog = (target/'samples.json').read_text()
        attribution = (target/'synth/source.json').read_text()
        for forbidden in ('private', 'glyph', 'poly', 'layout', 'generation'):
            self.assertNotIn(forbidden, catalog)
        self.assertNotIn('/private/', attribution)
        self.assertEqual(json.loads(attribution)['manifest_sha256'], pages.digest((self.root/'synth/manifest.json').read_bytes()))
        self.assertEqual((target/'synth/LICENSE.txt').read_text(), 'Apache-2.0 fixture')
        self.assertEqual((target/'synth/NOTICE.txt').read_text(), 'Project attribution fixture')

    def test_samples_reject_corrupted_image_and_annotation_bytes(self):
        self.sample_fixture()
        for relative in ('synth/images/000000.png', 'synth/annotations/000000.json',
                         'docs/assets/examples/digit-receipt-ko.jpg'):
            path = self.root/relative
            original = path.read_bytes()
            path.write_bytes(b'corrupt')
            with self.subTest(path=relative), self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                pages.stage_samples(self.root, self.root/'output/example')
            self.assertFalse((self.root/'output/example/samples.json').exists())
            path.write_bytes(original)
            shutil.rmtree(self.root/'output', ignore_errors=True)

    def test_samples_reject_path_aliases_duplicate_ids_and_wrong_dimensions(self):
        original = self.sample_fixture()
        for field, value, message in (
                ('image', '../private.png', 'path/order'), ('annotation', '/private.json', 'path/order'),
                ('image', 'images\\000000.png', 'path/order'), ('id', '000001', 'path/order'),
                ('width', 2, 'dimensions'), ('bytes', 1, 'size mismatch'),
                ('annotation_sha256', None, 'checksum')):
            manifest = json.loads(json.dumps(original))
            manifest['records'][0][field] = value
            self.write_sample_manifest(manifest)
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                pages.stage_samples(self.root, self.root/f'output-{field}-{len(str(value))}/example')
        self.write_sample_manifest(original)
        image = self.root/'synth/images/000000.png'
        outside = self.root/'private.png'
        outside.write_bytes(image.read_bytes())
        image.unlink()
        image.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Unsafe public sample path'):
            pages.stage_samples(self.root, self.root/'symlink-output/example')

    def test_sample_copy_uses_verified_snapshot_without_reopening_source(self):
        manifest = self.sample_fixture()
        image = self.root/'synth/images/000000.png'
        original = image.read_bytes()
        real_read = Path.read_bytes
        def change_after_read(path):
            raw = real_read(path)
            if path == image:
                path.write_bytes(b'changed after checksum read')
            return raw
        target = self.root/'output/example'
        with patch.object(Path, 'read_bytes', change_after_read):
            pages.stage_samples(self.root, target)
        self.assertEqual((target/'synth/000000.png').read_bytes(), original)
        self.assertEqual(pages.digest(original), manifest['records'][0]['sha256'])

    def test_receipt_metadata_rejects_objects_in_public_scalar_fields(self):
        for source in ({'store': {'private': '/secret'}}, {'phone_digits': ['123']},
                       {'street_no': True}, {'items': [{'name': '상품', 'quantity': '1', 'unit_price': 100, 'total': 100}]}):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'Invalid public receipt'):
                pages.receipt_metadata(source)

    def test_corrupt_release_hash_is_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, 'Release checksum'):
            pages.stage_model(self.archive(), self.root/'output', 'digit', '0' * 64)
        self.assertFalse((self.root/'output').exists())

    def test_corrupt_model_and_wrong_project_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid packaged asset'):
            pages.stage_model(self.archive({'model/model.safetensors': b'changed'}), self.root/'output', 'digit')
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            pages.stage_model(self.archive(), self.root/'output', 'vqa')

    def test_unsafe_paths_rejected_even_outside_selected_subtrees(self):
        for name in ('../escape', '/absolute', 'model/../../escape', 'model\\escape'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsafe ZIP path'):
                pages.stage_model(self.archive({name: b'bad'}), self.root/'output', 'digit')


if __name__ == '__main__':
    unittest.main()
