"""Release selection, artifact integrity and mocked GitHub CLI publishing."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
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


if __name__ == '__main__':
    unittest.main()
