"""Release selection, artifact integrity and mocked GitHub CLI publishing."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import build, release


def encoded(value):
    return (json.dumps(value, ensure_ascii=False) + '\n').encode()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='release repository with spaces ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.version = '0.9.0'
        self.commit = 'a' * 40
        self.lock = {'version': self.version,
                     'resolved': f'https://registry.npmjs.org/volvoxai-lite/-/volvoxai-lite-{self.version}.tgz',
                     'integrity': 'sha512-fixture-integrity'}
        write(self.root/'package.json', encoded({'config': {'volvoxai_version': self.version}}))
        write(self.root/'package-lock.json', encoded({'packages': {'node_modules/volvoxai-lite': self.lock}}))
        npm = self.root/'node_modules/volvoxai-lite'
        write(npm/'package.json', encoded({'name': 'volvoxai-lite', 'version': self.version}))
        write(npm/'LICENSE', 'Fixture runtime license\n')
        self.runtime = {'volvoxai.lite.min.js': b'export const fixture = true;\n',
                        'volvoxai.lite.wasm': b'\0asm\x01\0\0\0'}
        for name, data in self.runtime.items():
            write(npm/'dist'/self.version/name, data)
        for name in ('shared/sites.json', 'shared/extension/manifest.json', 'PRIVACY.md'):
            write(self.root/name, (ROOT/name).read_bytes())
        self.paths, self.contents = {}, {}
        for project in ('digit', 'vqa'):
            for browser in ('chrome', 'firefox'):
                self.make_archive(project, browser)

    def make_archive(self, project, browser='chrome'):
        config = json.loads((ROOT/project/'extension.json').read_text())
        precision = config['default_precision']
        write(self.root/project/'extension.json', encoded(config))
        frozen_file = ROOT/project/'frozen-profiles.json'
        raw_profiles = json.loads(frozen_file.read_text())['profiles'] if frozen_file.is_file() else {}
        profile = deepcopy(raw_profiles[precision]) if precision in raw_profiles else None
        if profile is not None:
            payloads = {name: f'{project} selected fixture {name}'.encode() for name in profile['sha256']}
            profile['sha256'] = {name: release.digest(data) for name, data in payloads.items()}
            write(self.root/project/'frozen-profiles.json', encoded({'profiles': {precision: profile}}))
            quantization = {'precision': precision, 'packed_arithmetic': 'dynamic-i8', 'version': '0.8.0'}
            model = {'frozen_candidate': profile, 'quantization': quantization}
        else:
            model_names = ['graph.json', 'model.safetensors'] if project == 'digit' else [
                'config.json', 'vocab.json', 'encoder/graph.json', 'encoder/model.safetensors',
                'shared/token-embedding.safetensors', 'decoder/graph.json', 'decoder/model.safetensors']
            payloads = {name: f'{project} selected fixture {name}'.encode() for name in model_names}
            quantization = {'precision': precision, 'method': 'volvoxai-ptq',
                            'calibration': {'selected_images': 256}}
            model = {'quantization': quantization}
            release.DEFAULT_PROFILES[project][precision] = {
                'sha256': {name: release.digest(data) for name, data in payloads.items()}
            }
        payloads['manifest.json'] = encoded(model)
        copied = release.source_files(self.root, project, config)
        for name, target in copied.items():
            if name != 'vendor/LICENSE.txt':
                write(target, (ROOT/target.relative_to(self.root)).read_bytes())
        entries = {name: target.read_bytes() for name, target in copied.items()}
        entries.update({'model/' + name: data for name, data in payloads.items()})
        entries['model/LICENSE.txt'] = b'Fixture model license\n'
        if 'NOTICE' in config['model']['common_files']:
            entries['model/NOTICE.txt'] = b'Fixture model notice\n'
        sites = build.site_config(config, root=self.root)
        entries['manifest.json'] = encoded(build.extension_manifest(config, sites, root=self.root, browser=browser))
        entries['privacy.html'] = build.privacy_page(root=self.root).encode()
        entries['shared/extension/site-config.js'] = (
            'globalThis.RECEIPT_READER_SITES = ' + json.dumps(sites, ensure_ascii=False) + ';\n').encode()
        entries.update({'vendor/' + name: data for name, data in self.runtime.items()})
        entries['build-info.json'] = encoded({
            'extension': project, 'extension_version': config['version'], 'browser': browser,
            'model_precision': precision, 'model_variant': 'ptq',
            'model_repository': config['model']['repository'],
            'model_revision': config['model']['revision'],
            'runtime_profile': 'lite', 'runtime_version': self.version,
            'dependencies': {
                'javascript': {'source': 'published', **self.lock},
                'python': {'version': self.version, 'source': 'published'},
            },
            'model_quantization': quantization,
            'model_sha256': {name: release.digest(data) for name, data in payloads.items()},
            'runtime_sha256': {name: release.digest(data) for name, data in self.runtime.items()},
        })
        key = project if browser == 'chrome' else project + '-firefox'
        self.paths[key] = self.root/'dist'/project/precision/f"{config['slug']}-{browser}.zip"
        self.contents[key] = entries
        self.replace_archive(key, entries)

    def replace_archive(self, project, entries):
        path = self.paths[project]
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in entries.items():
                archive.writestr(name, data)

    def mutate_json(self, entries, name, mutate):
        value = json.loads(entries[name])
        mutate(value)
        entries[name] = encoded(value)

    def fresh_archive(self, project):
        """Regenerated weights keep the source/ABI but have no frozen identity."""
        config = json.loads((self.root/project/'extension.json').read_text())
        source = self.root/'.cache'/project/'huggingface'/config['model']['revision']
        source_names = (('config.json', 'manifest.json', 'model.onnx') if project == 'digit' else
                        ('encoder_model.onnx', 'decoder_model.onnx', 'config.json', 'vocab.json'))
        preprocess = {'channels': 1, 'height': 320, 'width': 672}
        hashes = {}
        for name in source_names:
            value = encoded({'input': preprocess}) if name == 'config.json' else f'public source {name}'.encode()
            write(source/name, value)
            hashes[name] = release.digest(value)
        write(source/'SHA256SUMS', ''.join(f'{value}  {name}\n' for name, value in hashes.items()))
        entries = deepcopy(self.contents[project])
        if project == 'digit':
            from digit.rebuild import QUANTIZATION, RECIPE
            write(self.root/'synth/manifest.json', (ROOT/'synth/manifest.json').read_bytes())
            graph = {'inputs': {'input0': {'shape': [1, 1, 320, 672], 'dtype': 'float32'}},
                     'outputs': ['slot_logits'], 'nodes':
                     [{'opType': 'QConv2D', 'params': {'weight_bits': 4}} for _ in range(11)] +
                     [{'opType': 'PackedLinear', 'params': {'weight_bits': 2, 'group_size': 32}} for _ in range(7)]}
            graph['nodes'][-1]['outputs'] = {'out': {'tensor': 'slot_logits', 'shape': [1, 16, 11], 'dtype': 'float32'}}
            entries['model/graph.json'] = encoded(graph)
            entries['model/model.safetensors'] = b'newly fitted weights'
            model = {
                'format': 'volvoxai-receipt-digit-reader-onnx-package-v1',
                'source_format': 'receipt_digit_reader_onnx_v1', 'variant': 'ptq',
                'source': {'sha256': hashes}, 'preprocess': preprocess,
                'converter': {'package': 'volvoxai', 'version': self.version, 'source': 'pypi',
                              'prepare_for_ptq': True},
                'abi': {'input': {'name': 'input0', 'shape': [1, 1, 320, 672]}, 'output': 'slot_logits'},
                'decode': {'slots': 16, 'phone_slots': 12, 'street_slots': 4, 'num_classes': 11, 'blank_class': 10},
                'source_build': {'format': 'unaveraged-public-synth-qat/v1',
                    'model_repository': config['model']['repository'], 'model_revision': config['model']['revision'],
                    'dataset_manifest_sha256': release.digest((self.root/'synth/manifest.json').read_bytes()),
                    'prepared_sha256': {'graph.json': 'a' * 64, 'model.safetensors': 'b' * 64}},
                'quantization': {**QUANTIZATION, 'version': self.version,
                    'training': {**RECIPE, 'method': 'joint-weight-QAT-synth', 'backend': 'cpu',
                        'calibration_backend': 'cpu', 'activation_qat': False,
                        'training_samples': 224, 'validation_samples': 32,
                        'calibration_overlaps_training': True, 'selected_epoch': 2,
                        'validation_record_accuracy': 0.95, 'validation_cross_entropy': 0.1},
                    'calibration': {'samples': 224, 'coverage_complete': True, 'validation_overlap': 0,
                        'evaluation_checked': False, 'evaluation_overlap': None, 'audit_sha256': 'c' * 64}},
            }
        else:
            from vqa.frozen import _manifest
            profile = json.loads((self.root/project/'frozen-profiles.json').read_text())['profiles']['w4a8']
            model = _manifest(profile, 'w4a8')
            del model['frozen_candidate']
            model['source_sha256'] = hashes
            model['converter'] = {'package': 'volvoxai', 'version': self.version, 'source': 'pypi'}
            for name in ('config.json', 'vocab.json'):
                entries['model/' + name] = (source/name).read_bytes()
            quantization = model['quantization']
            quantization['version'] = self.version
            # Fresh PTQ reports dtypes; unlike old frozen manifests, it has no
            # redundant weight_bits or convolution_weight_bits metadata.
            del quantization['weight_bits'], quantization['convolution_weight_bits']
            quantization['calibration'] = {'samples': 256, 'selected_images': 256,
                                         'coverage_complete': True, 'evaluation_overlap': None}
            for role, dense, conv in (('encoder', 34, 13), ('decoder', 33, 0)):
                contract = model['graphs'][role]
                graph = {'inputs': {name: {} for name in contract['inputs'].values()},
                         'outputs': contract['outputs'], 'nodes':
                         [{'opType': 'PackedLinear', 'params': {'weight_bits': 4, 'group_size': 32}} for _ in range(dense)] +
                         [{'opType': 'QConv2D', 'params': {'weight_bits': 8}} for _ in range(conv)]}
                entries[f'model/{role}/graph.json'] = encoded(graph)
                entries[f'model/{role}/model.safetensors'] = f'fresh {role} weights'.encode()
        entries['model/manifest.json'] = encoded(model)
        self.refresh_model_identity(project, entries)
        self.replace_archive(project, entries)
        return entries

    def refresh_model_identity(self, project, entries):
        info = json.loads(entries['build-info.json'])
        model = json.loads(entries['model/manifest.json'])
        if project == 'digit':
            model['model_sha256'] = {name: release.digest(entries['model/' + name])
                                    for name in ('graph.json', 'model.safetensors')}
            entries['model/manifest.json'] = encoded(model)
        info['model_quantization'] = model['quantization']
        info['model_sha256'] = {name: release.digest(entries['model/' + name]) for name in info['model_sha256']}
        entries['build-info.json'] = encoded(info)

    def test_accepts_fresh_default_weights_without_the_frozen_candidate_hashes(self):
        for project in ('digit', 'vqa'):
            with self.subTest(project=project):
                self.fresh_archive(project)
                self.assertEqual(release.validate_archive(self.root, project)['project'], project)

    def test_fresh_vqa_allows_custom_calibration_count(self):
        entries = self.fresh_archive('vqa')
        self.mutate_json(entries, 'model/manifest.json', lambda model:
                         model['quantization']['calibration'].update(samples=128, selected_images=128))
        self.refresh_model_identity('vqa', entries)
        self.replace_archive('vqa', entries)
        release.validate_archive(self.root, 'vqa')

    def test_rebuilt_architecture_and_recipe_cannot_be_changed_by_rehashing(self):
        for project in ('digit', 'vqa'):
            with self.subTest(project=project):
                entries = self.fresh_archive(project)
                graph_name = 'model/graph.json' if project == 'digit' else 'model/encoder/graph.json'
                self.mutate_json(entries, graph_name,
                                 lambda graph: graph['nodes'][0]['params'].update(weight_bits=3))
                self.refresh_model_identity(project, entries)
                self.replace_archive(project, entries)
                with self.assertRaisesRegex(ValueError, 'precision changed'):
                    release.validate_archive(self.root, project)
        entries = self.fresh_archive('digit')
        self.mutate_json(entries, 'model/manifest.json', lambda model:
                         model['quantization']['training'].update(epochs=0))
        self.refresh_model_identity('digit', entries)
        self.replace_archive('digit', entries)
        with self.assertRaisesRegex(ValueError, 'training recipe mismatch'):
            release.validate_archive(self.root, 'digit')

    def test_rebuilt_source_and_dataset_provenance_are_checked(self):
        for project in ('digit', 'vqa'):
            with self.subTest(project=project):
                entries = self.fresh_archive(project)
                config = json.loads((self.root/project/'extension.json').read_text())
                source = self.root/'.cache'/project/'huggingface'/config['model']['revision']
                (source/'config.json').write_bytes(b'changed public source')
                with self.assertRaisesRegex(ValueError, 'public source checksum mismatch'):
                    release.validate_archive(self.root, project)
        entries = self.fresh_archive('digit')
        self.mutate_json(entries, 'model/manifest.json', lambda model:
                         model['source_build'].update(dataset_manifest_sha256='d' * 64))
        self.refresh_model_identity('digit', entries)
        self.replace_archive('digit', entries)
        with self.assertRaisesRegex(ValueError, 'public source provenance mismatch'):
            release.validate_archive(self.root, 'digit')
        entries = self.fresh_archive('vqa')
        entries['model/vocab.json'] = b'changed tokenizer'
        self.refresh_model_identity('vqa', entries)
        self.replace_archive('vqa', entries)
        with self.assertRaisesRegex(ValueError, 'rebuilt source asset differs'):
            release.validate_archive(self.root, 'vqa')

    def test_collects_only_selected_production_archives_with_original_model_provenance(self):
        write(self.root/'dist/private-inputs.zip', b'do not upload')
        write(self.root/'dist/digit/w8a8/unaveraged-digit-chrome.zip', b'wrong model')
        assets = release.collect_assets(self.root)
        self.assertEqual([(a['project'], a['browser'], a['precision']) for a in assets],
                         [('digit', 'chrome', 'conv4-linear2'), ('digit', 'firefox', 'conv4-linear2'),
                          ('vqa', 'chrome', 'w4a8'), ('vqa', 'firefox', 'w4a8')])
        self.assertEqual([a['path'] for a in assets],
                         [self.paths[key] for key in ('digit', 'digit-firefox', 'vqa', 'vqa-firefox')])
        for asset in assets:
            self.assertEqual(asset['runtime_version'], self.version)
            self.assertEqual(asset['sha256'], release.digest(asset['path'].read_bytes()))

    def test_browser_selection_and_firefox_manifest_identity(self):
        self.assertEqual([asset['browser'] for asset in release.collect_assets(self.root, 'chrome')], ['chrome', 'chrome'])
        self.assertEqual([asset['browser'] for asset in release.collect_assets(self.root, 'firefox')], ['firefox', 'firefox'])
        self.assertEqual(release.validate_archive(self.root, 'digit')['browser'], 'chrome')
        entries = deepcopy(self.contents['digit-firefox'])
        self.mutate_json(entries, 'build-info.json', lambda info: info.update(browser='chrome'))
        self.replace_archive('digit-firefox', entries)
        with self.assertRaisesRegex(ValueError, 'selected model/runtime'):
            release.validate_archive(self.root, 'digit', 'firefox')
        entries = deepcopy(self.contents['digit-firefox'])
        entries['manifest.json'] = self.contents['digit']['manifest.json']
        self.replace_archive('digit-firefox', entries)
        with self.assertRaisesRegex(ValueError, 'non-production firefox manifest'):
            release.validate_archive(self.root, 'digit', 'firefox')

    def test_refuses_wrong_model_runtime_or_dependency_provenance(self):
        changes = [
            lambda d: d.update(model_precision='w8a8'),
            lambda d: d.update(browser='firefox'),
            lambda d: d.update(model_variant='fp32'),
            lambda d: d.update(runtime_version='0.8.0'),
            lambda d: d['dependencies']['javascript'].update(source='local-dist'),
            lambda d: d['dependencies']['python'].update(version='0.8.0'),
            lambda d: d['dependencies']['javascript'].update(integrity='wrong'),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                entries = deepcopy(self.contents['digit'])
                self.mutate_json(entries, 'build-info.json', change)
                self.replace_archive('digit', entries)
                with self.assertRaises(ValueError):
                    release.validate_archive(self.root, 'digit')

    def test_refuses_development_configuration_even_with_production_filename(self):
        entries = deepcopy(self.contents['digit'])
        self.mutate_json(entries, 'manifest.json',
                         lambda d: d['content_scripts'][0]['matches'].append('http://localhost/*'))
        self.replace_archive('digit', entries)
        with self.assertRaisesRegex(ValueError, 'non-production'):
            release.validate_archive(self.root, 'digit')
        entries = deepcopy(self.contents['digit'])
        entries['shared/extension/site-config.js'] += b'// development override\n'
        self.replace_archive('digit', entries)
        with self.assertRaisesRegex(ValueError, 'site configuration'):
            release.validate_archive(self.root, 'digit')

    def test_refuses_private_extra_missing_duplicate_and_unsafe_entries(self):
        for name in ('predictions.private.json', '../outside.txt', '/absolute.txt'):
            with self.subTest(name=name):
                entries = {**self.contents['digit'], name: b'private fixture'}
                self.replace_archive('digit', entries)
                with self.assertRaisesRegex(ValueError, 'ZIP entries'):
                    release.validate_archive(self.root, 'digit')
        entries = deepcopy(self.contents['digit'])
        del entries['model/model.safetensors']
        self.replace_archive('digit', entries)
        with self.assertRaisesRegex(ValueError, 'ZIP entries'):
            release.validate_archive(self.root, 'digit')
        self.replace_archive('digit', self.contents['digit'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.paths['digit'], 'a') as archive:
                archive.writestr('model/model.safetensors', b'duplicate')
        with self.assertRaisesRegex(ValueError, 'ZIP entries'):
            release.validate_archive(self.root, 'digit')

    def test_refuses_tampering_even_when_self_reported_model_hash_is_rewritten(self):
        for rewrite_hash in (False, True):
            with self.subTest(rewrite_hash=rewrite_hash):
                entries = deepcopy(self.contents['digit'])
                entries['model/model.safetensors'] = b'unselected weights'
                if rewrite_hash:
                    self.mutate_json(entries, 'build-info.json', lambda d:
                        d['model_sha256'].update({'model.safetensors': release.digest(b'unselected weights')}))
                self.replace_archive('digit', entries)
                with self.assertRaisesRegex(ValueError, 'model hash mismatch|unselected model'):
                    release.validate_archive(self.root, 'digit')
        entries = deepcopy(self.contents['digit'])
        entries['vendor/volvoxai.lite.wasm'] = b'changed runtime'
        self.replace_archive('digit', entries)
        with self.assertRaisesRegex(ValueError, 'runtime mismatch'):
            release.validate_archive(self.root, 'digit')

    def test_refuses_stale_source_and_missing_default_archive(self):
        write(self.root/'digit/adapter.js', b'changed application source')
        with self.assertRaisesRegex(ValueError, 'stale packaged source'):
            release.validate_archive(self.root, 'digit')
        self.paths['vqa'].unlink()
        with self.assertRaisesRegex(ValueError, 'Missing production ZIP'):
            release.validate_archive(self.root, 'vqa')

    def run_publish(self, action, **options):
        assets = release.collect_assets(self.root)
        calls, staged = [], []
        def run(command, **kwargs):
            calls.append(command)
            self.assertEqual(kwargs, {'cwd': self.root, 'check': True})
            if command[0] == 'gh':
                for asset, path in zip(assets, command[-len(assets)-1:-1]):
                    staged.append(Path(path))
                    self.assertEqual(Path(path).read_bytes(), asset['path'].read_bytes())
                checksums = Path(command[-1])
                staged.append(checksums)
                self.assertEqual(checksums.name, 'SHA256SUMS')
                self.assertEqual(checksums.read_text(), release.checksum_text(assets))
                lines = checksums.read_text().splitlines()
                self.assertEqual(len(lines), 4)
                for line in lines:
                    digest, filename = line.split('  ', 1)
                    self.assertEqual(release.digest((checksums.parent/filename).read_bytes()), digest)
                if action == 'create':
                    notes = Path(command[command.index('--notes-file') + 1]).read_text()
                    self.assertIn('conv4-linear2', notes)
                    self.assertIn('0.9.0', notes)
                    self.assertIn('Firefox', notes)
                    self.assertIn('SHA256SUMS', notes)
            return subprocess.CompletedProcess(command, 0)
        with patch.object(release.subprocess, 'run', side_effect=run), \
             patch.object(release.subprocess, 'check_output', return_value=self.commit+'\n'):
            release.publish(action, assets, 'v0.5.0', repo='owner/repository', root=self.root, **options)
        self.assertTrue(all(not path.exists() for path in staged), 'Temporary upload copies must be cleaned')
        return calls

    def test_create_uses_explicit_commit_selected_assets_and_notes_file(self):
        calls = self.run_publish('create', draft=True)
        self.assertEqual(calls[0], ['git', 'check-ref-format', 'refs/tags/v0.5.0'])
        command = calls[1]
        self.assertEqual(command[:11], ['gh', 'release', 'create', 'v0.5.0', '--repo',
            'owner/repository', '--target', self.commit, '--title', 'unaveraged v0.5.0', '--draft'])
        self.assertNotIn('--clobber', command)
        self.assertEqual([Path(p).name for p in command[-5:]],
                         ['unaveraged-digit-chrome.zip', 'unaveraged-digit-firefox.zip',
                          'unaveraged-vqa-chrome.zip', 'unaveraged-vqa-firefox.zip', 'SHA256SUMS'])

    def test_upload_has_no_implicit_overwrite_and_supports_explicit_clobber(self):
        command = self.run_publish('upload')[-1]
        self.assertEqual(command[:6], ['gh', 'release', 'upload', 'v0.5.0', '--repo', 'owner/repository'])
        self.assertNotIn('--clobber', command)
        self.assertNotIn('--target', command)
        self.assertIn('--clobber', self.run_publish('upload', clobber=True)[-1])

    def test_dry_run_does_not_invoke_github_cli(self):
        calls = self.run_publish('create', dry_run=True)
        self.assertTrue(calls)
        self.assertTrue(all(command[0] == 'git' for command in calls))

    def run_models_publish(self, status, *, repo='', change_original=False, upload_failure=False):
        assets = release.collect_assets(self.root)
        originals = {asset['path'].name: asset['path'].read_bytes() for asset in assets}
        calls, staged = [], []
        def resolve(command, **kwargs):
            self.assertEqual(command, ['gh', 'repo', 'view', *([repo] if repo else []),
                                       '--json', 'nameWithOwner', '--jq', '.nameWithOwner'])
            self.assertEqual(kwargs, {'cwd': self.root, 'text': True})
            if change_original:
                self.paths['digit'].write_bytes(b'changed after snapshot')
            return 'owner/resolved\n'
        def run(command, **kwargs):
            calls.append(command)
            if command[:2] == ['gh', 'api']:
                self.assertEqual(command, ['gh', 'api', '--include', 'repos/owner/resolved/releases/tags/models'])
                self.assertEqual(kwargs, {'cwd': self.root, 'capture_output': True, 'text': True, 'check': False})
                output = (f'HTTP/2.0 {status} response\r\nContent-Type: application/json\r\n\r\n{{}}'
                          if status else '')
                return subprocess.CompletedProcess(command, 0 if status == 200 else 1, output, 'request failed')
            self.assertEqual(kwargs, {'cwd': self.root, 'check': True})
            self.assertEqual(command[:4], ['gh', 'release', 'upload' if status == 200 else 'create', 'models'])
            self.assertEqual(command[command.index('--repo') + 1], 'owner/resolved')
            self.assertNotIn('--target', command)
            for name in command[-5:]:
                path = Path(name)
                staged.append(path)
                if path.name == 'SHA256SUMS':
                    self.assertEqual(path.read_text(), release.checksum_text(assets))
                else:
                    self.assertEqual(path.read_bytes(), originals[path.name])
            if upload_failure:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        with patch.object(release.subprocess, 'run', side_effect=run), \
                patch.object(release.subprocess, 'check_output', side_effect=resolve):
            if status not in (200, 404):
                with self.assertRaisesRegex(ValueError, 'Cannot determine'):
                    release.publish('models', assets, '', repo=repo, root=self.root)
            elif upload_failure:
                with self.assertRaises(subprocess.CalledProcessError):
                    release.publish('models', assets, '', repo=repo, root=self.root)
            else:
                release.publish('models', assets, '', repo=repo, root=self.root)
        self.assertTrue(all(not path.exists() for path in staged))
        return calls

    def test_models_creates_missing_prerelease_without_targeting_local_head(self):
        calls = self.run_models_publish(404, repo='owner/repository')
        self.assertEqual(len(calls), 2)
        command = calls[-1]
        self.assertEqual(command[2:4], ['create', 'models'])
        self.assertIn('--prerelease', command)
        self.assertIn('--latest=false', command)
        self.assertEqual(command[command.index('--title') + 1], 'Local model packages')
        self.assertIn('--notes-file', command)
        self.assertNotIn('--draft', command)

    def test_models_replaces_existing_assets_using_verified_snapshots(self):
        calls = self.run_models_publish(200, change_original=True)
        self.assertEqual(len(calls), 2)
        command = calls[-1]
        self.assertEqual(command[2:4], ['upload', 'models'])
        self.assertIn('--clobber', command)
        self.assertEqual([Path(path).name for path in command[-5:]],
                         ['unaveraged-digit-chrome.zip', 'unaveraged-digit-firefox.zip',
                          'unaveraged-vqa-chrome.zip', 'unaveraged-vqa-firefox.zip', 'SHA256SUMS'])

    def test_models_auth_network_and_server_failures_never_trigger_create(self):
        for status in (401, 403, 500, None):
            with self.subTest(status=status):
                calls = self.run_models_publish(status)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][:2], ['gh', 'api'])
        assets = release.collect_assets(self.root)
        with patch.object(release.subprocess, 'run') as run, \
                patch.object(release.subprocess, 'check_output',
                             side_effect=subprocess.CalledProcessError(1, ['gh', 'repo', 'view'])):
            with self.assertRaises(subprocess.CalledProcessError):
                release.publish('models', assets, '', root=self.root)
            run.assert_not_called()

    def test_models_upload_failure_does_not_fall_back_to_creating_release(self):
        calls = self.run_models_publish(200, upload_failure=True)
        self.assertEqual([command[2] for command in calls if command[:2] == ['gh', 'release']], ['upload'])

    def test_models_dry_run_is_offline_and_shows_both_conditional_actions(self):
        assets = release.collect_assets(self.root)
        output = io.StringIO()
        with patch.object(release.subprocess, 'run') as run, \
                patch.object(release.subprocess, 'check_output') as lookup, redirect_stdout(output):
            release.publish('models', assets, '', repo='owner/repository', dry_run=True, root=self.root)
        run.assert_not_called()
        lookup.assert_not_called()
        self.assertIn('models release absent', output.getvalue())
        self.assertIn('models release exists', output.getvalue())
        self.assertIn('gh release create models --repo owner/repository', output.getvalue())
        self.assertIn('gh release upload models --repo owner/repository --clobber', output.getvalue())
        self.assertNotIn('--target', output.getvalue())

    def test_models_requires_all_four_archives_and_fixed_tag_before_github(self):
        assets = release.collect_assets(self.root)
        with patch.object(release.subprocess, 'run') as run, \
                patch.object(release.subprocess, 'check_output') as lookup:
            for selected, tag in ((assets[:2], ''), (assets, 'v1.0')):
                with self.subTest(tag=tag, count=len(selected)), self.assertRaises(ValueError):
                    release.publish('models', selected, tag, root=self.root)
            self.paths['digit'].write_bytes(b'changed after validation')
            with self.assertRaisesRegex(ValueError, 'ZIP changed after validation'):
                release.publish('models', assets, '', root=self.root)
        run.assert_not_called()
        lookup.assert_not_called()

    def test_validation_to_upload_race_refuses_changed_zip_before_github_call(self):
        assets = release.collect_assets(self.root)
        self.paths['digit'].write_bytes(b'changed after validation')
        with patch.object(release.subprocess, 'run') as run, \
             patch.object(release.subprocess, 'check_output', return_value=self.commit):
            with self.assertRaisesRegex(ValueError, 'ZIP changed after validation'):
                release.publish('create', assets, 'v0.5.0', root=self.root)
            self.assertFalse(any(call.args[0][0] == 'gh' for call in run.call_args_list))

    def test_invalid_arguments_and_github_failure_are_not_silently_reinterpreted(self):
        assets = release.collect_assets(self.root)
        with patch.object(release.subprocess, 'run') as run:
            for tag in ('', '--draft'):
                with self.subTest(tag=tag), self.assertRaises(ValueError):
                    release.publish('create', assets, tag, root=self.root)
            run.assert_not_called()
        def fail_github(command, **kwargs):
            if command[0] == 'gh':
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        with patch.object(release.subprocess, 'run', side_effect=fail_github) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                release.publish('upload', assets, 'v0.5.0', root=self.root)
            self.assertEqual([call.args[0][2] for call in run.call_args_list if call.args[0][0]=='gh'], ['upload'])


class CiReleaseTests(ReleaseTests):
    def setUp(self):
        super().setUp()
        self.ext_version = '0.1.0'
        self.set_project_versions(self.ext_version)

    def set_project_versions(self, version, *, rebuild_archives=False):
        self.ext_version = version
        pkg = json.loads((self.root / 'package.json').read_text())
        pkg['name'] = 'unaveraged'
        pkg['version'] = version
        write(self.root / 'package.json', encoded(pkg))
        lock = json.loads((self.root / 'package-lock.json').read_text())
        lock['name'] = 'unaveraged'
        lock['version'] = version
        lock.setdefault('packages', {}).setdefault('', {})['version'] = version
        write(self.root / 'package-lock.json', encoded(lock))
        for project in ('digit', 'vqa'):
            config = json.loads((self.root / project / 'extension.json').read_text())
            config['version'] = version
            write(self.root / project / 'extension.json', encoded(config))
            if rebuild_archives:
                for browser in ('chrome', 'firefox'):
                    entries = deepcopy(self.contents[project if browser == 'chrome' else f'{project}-firefox'])
                    self.mutate_json(entries, 'build-info.json', lambda d: d.update(extension_version=version))
                    manifest = json.loads(entries['manifest.json'])
                    manifest['version'] = version
                    entries['manifest.json'] = encoded(manifest)
                    self.contents[project if browser == 'chrome' else f'{project}-firefox'] = entries
                    self.replace_archive(project if browser == 'chrome' else f'{project}-firefox', entries)

    def make_source_archive(self, *, path=None, version=None, overrides=None, skip=()):
        version = version or self.ext_version
        target = path or (self.root / 'dist/unaveraged-source.zip')
        target.parent.mkdir(parents=True, exist_ok=True)
        payloads = {}
        for name in build.SOURCE_FILES:
            if name in skip:
                continue
            on_disk = self.root / name
            if name in ('package.json', 'package-lock.json', 'digit/extension.json', 'vqa/extension.json'):
                data = json.loads(on_disk.read_text())
                if name == 'package-lock.json':
                    data['version'] = version
                    data.setdefault('packages', {}).setdefault('', {})['version'] = version
                else:
                    data['version'] = version
                payloads[name] = encoded(data)
            elif on_disk.is_file():
                payloads[name] = on_disk.read_bytes()
            else:
                payloads[name] = (ROOT / name).read_bytes()
                write(on_disk, payloads[name])
        if overrides:
            payloads.update(overrides)
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in payloads.items():
                archive.writestr(name, data)
        return target

    def test_check_tag_validates_strict_semver_and_all_version_files(self):
        self.assertEqual(release.validate_release_tag('v0.1.0', root=self.root), '0.1.0')
        with patch.object(release, 'ROOT', self.root), redirect_stdout(io.StringIO()) as stdout:
            release.main(['check-tag', '--tag', 'v0.1.0'])
        self.assertIn('v0.1.0', stdout.getvalue())

        for bad_tag in ('', '0.1.0', 'v01.0.0', 'v0.1', 'v0.1.0-beta.1', 'v0.1.1', '--help'):
            with self.subTest(bad_tag=bad_tag), self.assertRaises(ValueError):
                release.validate_release_tag(bad_tag, root=self.root)

        mutations = [
            ('package.json', lambda d: d.update(version='0.2.0')),
            ('package-lock.json', lambda d: d.update(version='0.2.0')),
            ('package-lock.json', lambda d: d['packages'][''].update(version='0.2.0')),
            ('digit/extension.json', lambda d: d.update(version='0.2.0')),
            ('vqa/extension.json', lambda d: d.update(version='0.2.0')),
        ]
        for rel_path, mutate in mutations:
            with self.subTest(rel_path=rel_path):
                self.set_project_versions('0.1.0')
                data = json.loads((self.root / rel_path).read_text())
                mutate(data)
                write(self.root / rel_path, encoded(data))
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    release.validate_release_tag('v0.1.0', root=self.root)

    def test_stage_ci_and_discover_ci_assets_verify_archives_without_build_caches(self):
        self.make_source_archive()
        staged_dir = self.root / 'release-dist'
        write(staged_dir / 'stale.zip', b'old leftover')

        staged_assets = release.stage_ci(staged_dir, include_source=True, root=self.root)
        self.assertFalse((staged_dir / 'stale.zip').exists())
        expected_names = {
            'unaveraged-digit-chrome.zip',
            'unaveraged-digit-firefox.zip',
            'unaveraged-vqa-chrome.zip',
            'unaveraged-vqa-firefox.zip',
            'unaveraged-source.zip',
        }
        self.assertEqual({asset['path'].name for asset in staged_assets}, expected_names)
        checksum_lines = (staged_dir / 'SHA256SUMS').read_text().splitlines()
        self.assertEqual(len(checksum_lines), 5)

        # Simulate the CI release job where node_modules and .cache do not exist.
        shutil.rmtree(self.root / 'node_modules')
        discovered = release.discover_ci_assets(
            staged_dir, require_source=True, expected_version=self.ext_version, root=self.root,
        )
        self.assertEqual({asset['path'].name for asset in discovered}, expected_names)
        self.assertEqual(
            [asset['kind'] for asset in discovered],
            ['extension', 'extension', 'extension', 'extension', 'source'],
        )

    def test_discover_ci_assets_rejects_missing_tampered_symlink_and_unsafe_archives(self):
        self.make_source_archive()
        staged_dir = self.root / 'release-dist'
        release.stage_ci(staged_dir, include_source=True, root=self.root)
        shutil.rmtree(self.root / 'node_modules')

        with self.assertRaisesRegex(ValueError, 'SHA256SUMS'):
            (staged_dir / 'SHA256SUMS').write_text('0' * 64 + '  unaveraged-digit-chrome.zip\n')
            release.discover_ci_assets(staged_dir, require_source=True, root=self.root)

        # Recreate a clean staged directory using the valid staged ZIPs.
        clean_dist = self.root / 'clean-dist'
        clean_dist.mkdir()
        for src in (staged_dir / n for n in (
            'unaveraged-digit-chrome.zip', 'unaveraged-digit-firefox.zip',
            'unaveraged-vqa-chrome.zip', 'unaveraged-vqa-firefox.zip', 'unaveraged-source.zip',
        )):
            shutil.copy2(src, clean_dist / src.name)

        # Symlink SHA256SUMS rejected
        outside_sums = self.root / 'outside-sums'
        write(outside_sums, b'outside')
        (clean_dist / 'SHA256SUMS').symlink_to(outside_sums)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)
        (clean_dist / 'SHA256SUMS').unlink()

        # Symlink ZIP rejected
        outside_zip = self.root / 'outside.zip'
        outside_zip.write_bytes(b'not a zip')
        (clean_dist / 'symlinked.zip').symlink_to(outside_zip)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)
        (clean_dist / 'symlinked.zip').unlink()

        # Extra unexpected ZIP rejected
        write(clean_dist / 'unexpected.zip', b'PK\x05\x06' + b'\x00' * 18)
        with self.assertRaisesRegex(ValueError, 'Unexpected archive'):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)
        (clean_dist / 'unexpected.zip').unlink()

        # Duplicate ZIP in subdir rejected
        sub = clean_dist / 'dup'
        sub.mkdir()
        shutil.copy2(clean_dist / 'unaveraged-source.zip', sub / 'unaveraged-source.zip')
        with self.assertRaisesRegex(ValueError, 'Duplicate archive'):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)
        shutil.rmtree(sub)

        # Missing source archive rejected when required
        saved_source = (clean_dist / 'unaveraged-source.zip').read_bytes()
        (clean_dist / 'unaveraged-source.zip').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing required source archive'):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)

        # Source archive with symlink entry rejected
        bad_source = clean_dist / 'unaveraged-source.zip'
        with zipfile.ZipFile(bad_source, 'w') as archive:
            info = zipfile.ZipInfo(build.SOURCE_FILES[0])
            info.external_attr = 0o120777 << 16
            archive.writestr(info, b'../secret')
            for name in build.SOURCE_FILES[1:]:
                archive.writestr(name, b'x')
        with self.assertRaisesRegex(ValueError, 'symlink member'):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)

        # Source archive missing file rejected
        self.make_source_archive(path=bad_source, skip=('.github/workflows/build.yml',))
        with self.assertRaisesRegex(ValueError, 'unexpected or missing source entries'):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)

        # Corrupt ZIP CRC rejected
        bad_source.write_bytes(saved_source)
        raw = bytearray(bad_source.read_bytes())
        raw[-30] ^= 0xFF
        bad_source.write_bytes(bytes(raw))
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)
        bad_source.write_bytes(saved_source)

        # Development manifest in staged extension ZIP rejected
        chrome_zip = clean_dist / 'unaveraged-digit-chrome.zip'
        entries = deepcopy(self.contents['digit'])
        self.mutate_json(entries, 'manifest.json',
                         lambda d: d['content_scripts'][0]['matches'].append('http://localhost/*'))
        with zipfile.ZipFile(chrome_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in entries.items():
                archive.writestr(name, data)
        with self.assertRaisesRegex(ValueError, 'non-production chrome manifest'):
            release.discover_ci_assets(clean_dist, require_source=True, root=self.root)

    def _ci_assets(self):
        self.make_source_archive()
        staged_dir = self.root / 'release-dist'
        release.stage_ci(staged_dir, include_source=True, root=self.root)
        return release.discover_ci_assets(
            staged_dir, require_source=True, expected_version=self.ext_version, root=self.root,
        )

    def _expected_uploaded_records(self, assets):
        records = [
            {
                'name': asset['path'].name,
                'size': asset['bytes'],
                'state': 'uploaded',
                'digest': f"sha256:{asset['sha256']}",
            }
            for asset in assets
        ]
        checksums_raw = release.checksum_text(assets).encode()
        records.append({
            'name': 'SHA256SUMS',
            'size': len(checksums_raw),
            'state': 'uploaded',
            'digest': f'sha256:{release.digest(checksums_raw)}',
        })
        return records

    def test_publish_ci_stable_publishes_verified_draft_and_rejects_regression(self):
        assets = self._ci_assets()
        records = self._expected_uploaded_records(assets)
        sha = 'A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2'
        events = []

        def run(command, **kwargs):
            if command[:2] == ['gh', 'api']:
                if 'releases?per_page=100' in command[4]:
                    if 'create-draft' in events:
                        events.append('verify')
                        return subprocess.CompletedProcess(command, 0, json.dumps([[{
                            'tag_name': 'v0.1.0',
                            'draft': True,
                            'prerelease': False,
                            'assets': records,
                        }]]), '')
                    # Prerelease and draft newer versions do not block v0.1.0; v0.0.9 is older.
                    return subprocess.CompletedProcess(command, 0, json.dumps([[
                        {'tag_name': 'v0.0.9', 'draft': False, 'prerelease': False},
                        {'tag_name': 'v9.9.9', 'draft': False, 'prerelease': True},
                        {'tag_name': 'v9.9.8', 'draft': True, 'prerelease': False},
                    ]]), '')
            if command[:4] == ['gh', 'release', 'create', 'v0.1.0']:
                events.append('create-draft')
                self.assertIn('--draft', command)
                self.assertIn('--target', command)
                self.assertEqual(command[command.index('--target') + 1], sha.lower())
                notes = Path(command[command.index('--notes-file') + 1]).read_text()
                self.assertIn('unaveraged-source.zip', notes)
                self.assertIn('SHA256SUMS', notes)
                self.assertIn(sha.lower(), notes)
                return subprocess.CompletedProcess(command, 0, '', '')
            if command[:4] == ['gh', 'release', 'edit', 'v0.1.0']:
                events.append('publish')
                self.assertIn('--draft=false', command)
                self.assertIn('--latest', command)
                return subprocess.CompletedProcess(command, 0, '', '')
            raise AssertionError(f'Unexpected command: {command}')

        with patch.object(release.subprocess, 'run', side_effect=run), redirect_stdout(io.StringIO()):
            release.publish_ci('stable', assets, sha, tag='v0.1.0', repo='owner/repository', root=self.root)
        self.assertEqual(events, ['create-draft', 'verify', 'publish'])

        # Existing tag or higher published stable release refuses before gh release create.
        for existing_list, pattern in (
            ([{'tag_name': 'v0.1.0', 'draft': True, 'prerelease': False}], 'already exists'),
            ([{'tag_name': 'v0.2.0', 'draft': False, 'prerelease': False}], 'must be newer'),
        ):
            with self.subTest(pattern=pattern):
                calls = []
                def block_run(command, **kwargs):
                    calls.append(command)
                    if command[:2] == ['gh', 'api']:
                        return subprocess.CompletedProcess(command, 0, json.dumps([existing_list]), '')
                    raise AssertionError(f'Should not invoke mutation: {command}')
                with patch.object(release.subprocess, 'run', side_effect=block_run):
                    with self.assertRaisesRegex(ValueError, pattern):
                        release.publish_ci('stable', assets, sha, tag='v0.1.0', repo='owner/repository', root=self.root)
                self.assertEqual(len(calls), 1)

    def test_publish_ci_stable_leaves_draft_unpublished_when_verification_fails(self):
        assets = self._ci_assets()
        valid_records = self._expected_uploaded_records(assets)
        sha = 'a' * 40
        bad_cases = []

        r0 = deepcopy(valid_records)
        del r0[0]['digest']
        bad_cases.append((r0, 'valid GitHub SHA-256 digest'))

        r1 = deepcopy(valid_records)
        r1[0]['digest'] = 'sha256:' + '0' * 64
        bad_cases.append((r1, 'checksum does not match'))

        r2 = deepcopy(valid_records)
        r2[0]['state'] = 'starter'
        bad_cases.append((r2, 'incomplete'))

        r3 = deepcopy(valid_records)
        r3[0]['size'] += 1
        bad_cases.append((r3, 'incomplete'))

        r4 = valid_records[:-1]
        bad_cases.append((r4, 'complete package set'))

        for bad_assets, message in bad_cases:
            with self.subTest(message=message):
                commands = []
                def run(command, **kwargs):
                    commands.append(command)
                    if command[:2] == ['gh', 'api']:
                        created = any(c[:3] == ['gh', 'release', 'create'] for c in commands)
                        payload = [[{'tag_name': 'v0.1.0', 'draft': True, 'prerelease': False, 'assets': bad_assets}]] if created else [[]]
                        return subprocess.CompletedProcess(command, 0, json.dumps(payload), '')
                    if command[:3] == ['gh', 'release', 'create']:
                        return subprocess.CompletedProcess(command, 0, '', '')
                    raise AssertionError(f'Publish edit must not run on failed verify: {command}')
                with patch.object(release.subprocess, 'run', side_effect=run):
                    with self.assertRaisesRegex(ValueError, message):
                        release.publish_ci('stable', assets, sha, tag='v0.1.0', repo='owner/repository', root=self.root)
                self.assertEqual([c[2] for c in commands if c[:2] == ['gh', 'release']], ['create'])

    def test_publish_ci_nightly_creates_initial_and_rolls_existing_prerelease(self):
        assets = self._ci_assets()
        records = self._expected_uploaded_records(assets)
        sha = 'abcdef1234567890abcdef1234567890abcdef12'

        # Case 1: initial creation
        create_events = []
        def run_create(command, **kwargs):
            if command[:2] == ['gh', 'api']:
                if 'create-draft' in create_events:
                    create_events.append('verify')
                    return subprocess.CompletedProcess(command, 0, json.dumps([[{
                        'tag_name': 'nightly', 'draft': True, 'prerelease': True, 'assets': records,
                    }]]), '')
                return subprocess.CompletedProcess(command, 0, '[[]]', '')
            if command[:4] == ['gh', 'release', 'create', 'nightly']:
                create_events.append('create-draft')
                self.assertIn('--draft', command)
                self.assertIn('--prerelease', command)
                self.assertIn('--latest=false', command)
                self.assertIn('unaveraged nightly (abcdef1)', command)
                return subprocess.CompletedProcess(command, 0, '', '')
            if command[:4] == ['gh', 'release', 'edit', 'nightly']:
                create_events.append('publish')
                self.assertIn('--draft=false', command)
                self.assertIn('--prerelease', command)
                self.assertIn('--latest=false', command)
                return subprocess.CompletedProcess(command, 0, '', '')
            raise AssertionError(f'Unexpected command: {command}')

        with patch.object(release.subprocess, 'run', side_effect=run_create), redirect_stdout(io.StringIO()):
            release.publish_ci('nightly', assets, sha, repo='owner/repository', root=self.root)
        self.assertEqual(create_events, ['create-draft', 'verify', 'publish'])

        # Case 2: rolling update deletes stale asset, verifies, patches git ref, edits release
        update_events = []
        def run_update(command, **kwargs):
            if command[:2] == ['gh', 'api']:
                path = command[4]
                if 'releases?per_page=100' in path:
                    if 'upload' in update_events:
                        update_events.append('verify')
                        return subprocess.CompletedProcess(command, 0, json.dumps([[{
                            'tag_name': 'nightly', 'draft': False, 'prerelease': True, 'assets': records,
                        }]]), '')
                    return subprocess.CompletedProcess(command, 0, json.dumps([[{
                        'tag_name': 'nightly',
                        'draft': False,
                        'prerelease': True,
                        'assets': [{'name': 'unaveraged-digit-chrome.zip'}, {'name': 'stale-old.zip'}],
                    }]]), '')
                if 'matching-refs/tags/nightly' in path:
                    return subprocess.CompletedProcess(command, 0, json.dumps([{'ref': 'refs/tags/nightly'}]), '')
                if path.endswith('/git/refs/tags/nightly') and '--method' in command and 'PATCH' in command:
                    update_events.append('patch-ref')
                    self.assertIn(f'sha={sha}', command)
                    return subprocess.CompletedProcess(command, 0, '{}', '')
            if command[:4] == ['gh', 'release', 'upload', 'nightly']:
                update_events.append('upload')
                self.assertIn('--clobber', command)
                return subprocess.CompletedProcess(command, 0, '', '')
            if command[:5] == ['gh', 'release', 'delete-asset', 'nightly', 'stale-old.zip']:
                update_events.append('delete-stale')
                self.assertIn('--yes', command)
                return subprocess.CompletedProcess(command, 0, '', '')
            if command[:4] == ['gh', 'release', 'edit', 'nightly']:
                update_events.append('edit')
                self.assertIn('--target', command)
                self.assertEqual(command[command.index('--target') + 1], sha)
                self.assertIn('--draft=false', command)
                self.assertIn('--prerelease', command)
                self.assertIn('--latest=false', command)
                return subprocess.CompletedProcess(command, 0, '', '')
            raise AssertionError(f'Unexpected command: {command}')

        with patch.object(release.subprocess, 'run', side_effect=run_update), redirect_stdout(io.StringIO()):
            release.publish_ci('nightly', assets, sha, repo='owner/repository', root=self.root)
        self.assertEqual(update_events, ['upload', 'delete-stale', 'verify', 'patch-ref', 'edit'])

        # Case 2b: update_nightly_git_ref creates refs/tags/nightly via POST when matching-refs is empty
        git_ref_calls = []
        def run_post_ref(command, **kwargs):
            git_ref_calls.append(command)
            if 'matching-refs/tags/nightly' in command[4]:
                return subprocess.CompletedProcess(command, 0, '[]', '')
            if command[4].endswith('/git/refs') and '--method' in command and 'POST' in command:
                self.assertIn('ref=refs/tags/nightly', command)
                self.assertIn(f'sha={sha}', command)
                return subprocess.CompletedProcess(command, 0, '{}', '')
            raise AssertionError(f'Unexpected command: {command}')
        with patch.object(release.subprocess, 'run', side_effect=run_post_ref):
            release.update_nightly_git_ref('owner/repository', sha, root=self.root)
        self.assertEqual(len(git_ref_calls), 2)

        # Case 3: refuses invalid SHA or non-prerelease nightly before calling gh upload
        with patch.object(release.subprocess, 'run') as mock_run:
            with self.assertRaisesRegex(ValueError, '7..=40'):
                release.publish_ci('nightly', assets, 'abc', repo='owner/repository', root=self.root)
            mock_run.assert_not_called()

        with patch.object(release.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps([[{
            'tag_name': 'nightly', 'draft': False, 'prerelease': False, 'assets': [],
        }]]), '')) as mock_run:
            with self.assertRaisesRegex(ValueError, 'nightly prerelease'):
                release.publish_ci('nightly', assets, sha, repo='owner/repository', root=self.root)
            self.assertEqual(mock_run.call_count, 1)

    def test_release_ci_dry_run_cli_is_offline(self):
        self.make_source_archive()
        staged_dir = self.root / 'release-dist'
        release.stage_ci(staged_dir, include_source=True, root=self.root)
        stdout = io.StringIO()
        with patch.object(release, 'ROOT', self.root), \
             patch.object(release.subprocess, 'run') as mock_run, \
             redirect_stdout(stdout):
            release.main([
                'release-ci',
                '--channel', 'nightly',
                '--sha', 'abcdef1',
                '--dist', str(staged_dir),
                '--repo', 'owner/repository',
                '--require-source',
                '--dry-run',
            ])
        mock_run.assert_not_called()
        self.assertIn('Dry run (nightly -> nightly @ abcdef1 on owner/repository)', stdout.getvalue())
        self.assertIn('unaveraged-source.zip', stdout.getvalue())

    def test_verify_ci_validates_staged_archives_and_rejects_mismatched_tag(self):
        self.make_source_archive()
        staged_dir = self.root / 'release-dist'
        release.stage_ci(staged_dir, include_source=True, root=self.root)
        shutil.rmtree(self.root / 'node_modules')
        stdout = io.StringIO()
        with patch.object(release, 'ROOT', self.root), redirect_stdout(stdout):
            release.main([
                'verify-ci',
                '--tag', 'v0.1.0',
                '--dist', str(staged_dir),
                '--require-source',
                '--require-checksums',
            ])
        self.assertIn('unaveraged-digit-chrome.zip', stdout.getvalue())
        self.assertIn('unaveraged-vqa-chrome.zip', stdout.getvalue())
        with patch.object(release, 'ROOT', self.root), self.assertRaises(SystemExit) as cm:
            release.main([
                'verify-ci',
                '--tag', 'v0.1.1',
                '--dist', str(staged_dir),
                '--require-source',
                '--require-checksums',
            ])
        self.assertNotEqual(cm.exception.code, 0)
        (staged_dir / 'SHA256SUMS').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing required SHA256SUMS'):
            release.discover_ci_assets(
                staged_dir, require_source=True, require_checksums=True, root=self.root,
            )
        with patch.object(release, 'ROOT', self.root), self.assertRaises(SystemExit) as cm_missing:
            release.main([
                'verify-ci',
                '--tag', 'v0.1.0',
                '--dist', str(staged_dir),
                '--require-source',
                '--require-checksums',
            ])
        self.assertNotEqual(cm_missing.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
