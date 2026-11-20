#!/usr/bin/env python3
"""Validate and upload the selected Chrome/Firefox extension ZIPs and SHA256SUMS."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import build


DEFAULT_PROFILES = {
    'digit': {
        'w8a8': {
            'sha256': {
                'graph.json': '8e89b05bae252847ba66e4a654f3b0bec4f25f0c5d444e86dedefa752e026c2a',
                'model.safetensors': 'eabb21a06231703abd0d140b8483b5af932b0603bcdfcd932312dcf37a4c4079',
            }
        }
    },
    'vqa': {
        'w8a8': {
            'sha256': {
                'config.json': 'e18e266da200b9b915628c1b8f633cb523520287e2d8f75c8093e3f4d6575f7e',
                'vocab.json': 'ae19bf56556649d75e72450ff6c7d43895190404c15dffeb6412dce5cc4d2c80',
                'encoder/graph.json': '28fb62de5eee6865fdb30ae362ae3fd466be5e42d0af749b72e5896c5af70727',
                'encoder/model.safetensors': 'b229c3e0f928708685ffc80066c2fb3d0253ba0568cf08992e9810b8e536a7bf',
                'shared/token-embedding.safetensors': '2ddc31c77c47498d65ee5604f5ee202704aa32a4878b10cf1c171618306a4a60',
                'decoder/graph.json': '2b1e0a54b16dbb57ae9315f66b7212e3e986435c4a080922807e5cb0e4d375c7',
                'decoder/model.safetensors': '42318118d2c79f13735bd52803f27cf197676eb03d359456a2266d21154f0361',
            }
        }
    }
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_files(root, project, config):
    files = {f'shared/extension/{name}': root/'shared/extension'/name
             for name in build.EXTENSION_FILES if name != 'manifest.json'}
    files.update({f'model/{name}': root/project/name for name in config['scripts']})
    files.update({name + '.txt': root/name for name in ('LICENSE', 'NOTICE')})
    files.update({f'shared/assets/{name}': root/'shared/assets'/name for name in
                  (f'icons/icon{size}.png' for size in (16, 32, 48, 128))})
    files['vendor/LICENSE.txt'] = root/'node_modules/volvoxai-lite/LICENSE'
    return files


def fresh_source_hashes(root, project, config):
    """Use the checked public release cached by the build as source provenance."""
    directory = root/'.cache'/project/'huggingface'/config['model']['revision']
    names = (('config.json', 'manifest.json', 'model.onnx') if project == 'digit' else
             ('encoder_model.onnx', 'decoder_model.onnx', 'config.json', 'vocab.json'))
    require((directory/'SHA256SUMS').is_file(),
            f'{project}: missing public source checksums; run make first')
    checksums = {}
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        if line.strip():
            expected, name = line.split(maxsplit=1)
            checksums[name.lstrip('*')] = expected
    hashes = {}
    for name in names:
        path = directory/name
        require(path.is_file() and not path.is_symlink(), f'{project}: missing public source: {name}')
        hashes[name] = digest(path.read_bytes())
        require(hashes[name] == checksums.get(name), f'{project}: public source checksum mismatch: {name}')
    return hashes


def validate_fresh_model(root, project, config, model, archive, version):
    """Validate a source rebuild's recipe and ABI without requiring old weights."""
    precision = config['default_precision']
    quantization = model['quantization']
    require(model.get('variant') == 'ptq'
            and model.get('converter', {}).get('version') == version
            and model['converter'].get('source') == 'pypi'
            and quantization.get('version') == version,
            f'{project}: fresh model must use the current published converter')
    source_hashes = fresh_source_hashes(root, project, config)
    if project == 'digit':
        from digit.rebuild import validate_manifest
        from digit.frozen import validate_graph
        validate_manifest(model, source_hashes, root=root)
        source = root/'.cache'/project/'huggingface'/config['model']['revision']
        require(model.get('preprocess') == read(source/'config.json')['input'],
                'digit: rebuilt preprocessing contract differs')
        graph = json.loads(archive.read('model/graph.json'))
        validate_graph(graph,
                       {'conv_weight_bits': 4, 'linear_weight_bits': 2, 'linear_group_size': 32})
        outputs = [value for node in graph['nodes'] for value in node.get('outputs', {}).values()
                   if value.get('tensor') == 'slot_logits']
        require(len(outputs) == 1 and outputs[0].get('shape') == [1, 16, 11]
                and outputs[0].get('dtype') == 'float32', 'digit: rebuilt slot output contract differs')
        for name, expected in model['model_sha256'].items():
            require(digest(archive.read('model/' + name)) == expected,
                    f'{project}: rebuilt model hash mismatch: {name}')
        return
    require(project == 'vqa' and precision == 'w4a8', 'Unsupported fresh release model')
    require(model.get('source_sha256') == source_hashes,
            'vqa: rebuilt model source provenance differs')
    for name in ('config.json', 'vocab.json'):
        require(digest(archive.read('model/' + name)) == source_hashes[name],
                f'vqa: rebuilt source asset differs: {name}')
    require(quantization.get('method') == 'volvoxai-ptq'
            and quantization.get('packed_arithmetic') == 'dynamic-i8'
            and quantization.get('weight_dtype') == 'int4'
            and quantization.get('activation_dtype') == 'int8'
            and quantization.get('group_size') == 32
            and quantization.get('convolution_weight_dtype') == 'int8',
            'vqa: rebuilt quantization precision differs')
    calibration = quantization.get('calibration', {})
    count = calibration.get('selected_images')
    require(isinstance(count, int) and count > 0 and calibration.get('samples') == count
            and calibration.get('coverage_complete') is True
            and calibration.get('evaluation_overlap') in (None, 0),
            'vqa: incomplete or overlapping calibration')
    from vqa.frozen import _manifest, _validate_graphs
    profile = read(root/'vqa/frozen-profiles.json')['profiles'][precision]
    expected = _manifest(profile, precision)
    for key in ('format', 'graphs', 'tokenizer', 'family_names', 'question_capacity',
                'max_decoder_steps', 'initial_masked_cache_rows'):
        require(model.get(key) == expected[key], f'vqa: rebuilt {key} contract differs')
    with tempfile.TemporaryDirectory(prefix='release-model-check-') as temporary:
        package = Path(temporary)
        for role in ('encoder', 'decoder'):
            (package/role).mkdir()
            (package/role/'graph.json').write_bytes(archive.read(f'model/{role}/graph.json'))
        _validate_graphs(package, model)


def validate_archive(root, project, browser='chrome'):
    require(browser in build.BROWSERS, f'Unknown browser: {browser}')
    config = read(root/project/'extension.json')
    precision = config['default_precision']
    frozen_file = root/project/'frozen-profiles.json'
    profile = read(frozen_file)['profiles'].get(precision) if frozen_file.is_file() else None
    if profile is None:
        profile = DEFAULT_PROFILES.get(project, {}).get(precision)
    version = read(root/'package.json')['config']['volvoxai_version']
    npm = root/'node_modules/volvoxai-lite'
    installed, lock = read(npm/'package.json'), read(root/'package-lock.json')['packages']['node_modules/volvoxai-lite']
    require(installed['version'] == lock['version'] == version
            and installed['name'] == 'volvoxai-lite'
            and lock['resolved'] == f'https://registry.npmjs.org/volvoxai-lite/-/volvoxai-lite-{version}.tgz'
            and lock.get('integrity'), 'Run make deps to install the pinned npm runtime')
    path = root/'dist'/project/precision/f"{config['slug']}-{browser}.zip"
    require(path.is_file() and not path.is_symlink(), f'Missing production ZIP: {path}; run make first')
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        copied = source_files(root, project, config)
        model_manifest_name = 'model/model-manifest.json' if 'model/model-manifest.json' in names else 'model/manifest.json'
        model_manifest_file = 'model-manifest.json' if 'model/model-manifest.json' in names else 'manifest.json'
        info, model = (json.loads(archive.read(name)) for name in ('build-info.json', model_manifest_name))
        frozen = 'frozen_candidate' in model
        if frozen:
            require(profile is not None and 'sha256' in profile, f'{project}: unknown frozen candidate')
            model_files = {model_manifest_file, *profile['sha256']}
        elif project == 'digit':
            model_files = {model_manifest_file, 'graph.json', 'model.safetensors'}
        else:
            model_files = {model_manifest_file, 'config.json', 'vocab.json',
                           'encoder/graph.json', 'encoder/model.safetensors',
                           'decoder/graph.json', 'decoder/model.safetensors',
                           'shared/token-embedding.safetensors'}
        allowed = {*copied, 'manifest.json', 'build-info.json', 'privacy.html',
                   'shared/extension/site-config.js', 'model/LICENSE.txt',
                   *(f'model/{name}' for name in model_files),
                   'vendor/volvoxai.lite.min.js', 'vendor/volvoxai.lite.wasm'}
        if 'NOTICE' in config['model']['common_files']:
            allowed.add('model/NOTICE.txt')
        require(len(names) == len(set(names)) and set(names) == allowed,
                f'{project}: unexpected or missing ZIP entries; rebuild the selected production package')
        require(info['extension'] == project and info['extension_version'] == config['version']
                and info.get('browser') == browser
                and info['model_precision'] == precision and info['model_variant'] == 'ptq'
                and info.get('model_repository') == config['model']['repository']
                and info.get('model_revision') == config['model']['revision']
                and info['runtime_profile'] == 'lite' and info['runtime_version'] == version,
                f'{project}: ZIP does not match the selected model/runtime')
        require(all(info['dependencies'][kind]['version'] == version
                    and info['dependencies'][kind]['source'] == 'published' for kind in ('javascript', 'python')),
                f'{project}: release requires published npm/wheel dependencies')
        require(info['dependencies']['javascript'].get('integrity') == lock['integrity']
                and info['dependencies']['javascript'].get('resolved') == lock['resolved'],
                f'{project}: npm provenance differs from the lockfile')
        require(model['quantization']['precision'] == precision
                and info['model_quantization'] == model['quantization'],
                f'{project}: selected quantization metadata differs')
        if frozen:
            require(model.get('frozen_candidate') == profile and model['quantization']['precision'] == precision
                    and model['quantization']['packed_arithmetic'] == 'dynamic-i8'
                    and info['model_quantization'] == model['quantization'],
                    f'{project}: selected quantization metadata differs')
        else:
            validate_fresh_model(root, project, config, model, archive, version)
        if frozen:
            for name, expected in profile['sha256'].items():
                require(digest(archive.read('model/' + name)) == expected, f'{project}: unselected model: {name}')
        require(set(info['model_sha256']) == model_files, f'{project}: incomplete model identity')
        for name, expected in info['model_sha256'].items():
            require(digest(archive.read('model/' + name)) == expected, f'{project}: model hash mismatch: {name}')
        for name in ('volvoxai.lite.min.js', 'volvoxai.lite.wasm'):
            payload = archive.read('vendor/' + name)
            require(digest(payload) == info['runtime_sha256'][name]
                    and payload == (npm/'dist'/version/name).read_bytes(), f'{project}: runtime mismatch: {name}')
        for name, source in copied.items():
            require(archive.read(name) == source.read_bytes(), f'{project}: stale packaged source: {name}; run make')
        # Reuse the production configuration factories used by the builder.
        sites = build.site_config(config, root=root)
        require(json.loads(archive.read('manifest.json')) == build.extension_manifest(config, sites, root=root, browser=browser),
                f'{project}: non-production {browser} manifest')
        require(archive.read('privacy.html').decode() == build.privacy_page(root=root), f'{project}: stale privacy page')
        expected_sites = 'globalThis.RECEIPT_READER_SITES = ' + json.dumps(sites, ensure_ascii=False) + ';\n'
        require(archive.read('shared/extension/site-config.js').decode() == expected_sites,
                f'{project}: non-production site configuration')
    return {'path': path, 'project': project, 'browser': browser, 'precision': precision, 'version': config['version'],
            'runtime_version': version, 'sha256': digest(raw), 'bytes': len(raw)}


def collect_assets(root=ROOT, browser='all'):
    require(browser in ('all', *build.BROWSERS), f'Unknown browser: {browser}')
    browsers = build.BROWSERS if browser == 'all' else (browser,)
    return [validate_archive(root, project, selected) for project in build.PROJECTS for selected in browsers]


_SEMVER_RE = re.compile(r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')
_REPO_RE = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')
_SHA_RE = re.compile(r'[0-9a-fA-F]{7,40}')
_SHA256_HEX_RE = re.compile(r'[0-9a-f]{64}')
_GITHUB_DIGEST_RE = re.compile(r'sha256:[0-9a-f]{64}')


def parse_semver_tuple(value):
    if not isinstance(value, str):
        return None
    match = _SEMVER_RE.fullmatch(value)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def project_versions(root=ROOT):
    package = read(root / 'package.json')
    lock = read(root / 'package-lock.json')
    versions = {
        'package.json': package.get('version'),
        'package-lock.json': lock.get('version'),
        'package-lock.json packages[""].version': lock.get('packages', {}).get('', {}).get('version'),
    }
    for project in build.PROJECTS:
        config = read(root / project / 'extension.json')
        versions[f'{project}/extension.json'] = config.get('version')
    for source, value in versions.items():
        require(parse_semver_tuple(value) is not None,
                f'{source} version {value!r} must be a stable MAJOR.MINOR.PATCH version')
    distinct = set(versions.values())
    require(len(distinct) == 1,
            'Project version mismatch: ' + ', '.join(f'{key}={value}' for key, value in versions.items()))
    return next(iter(distinct))


def validate_release_tag(tag, root=ROOT):
    require(isinstance(tag, str) and tag.startswith('v'),
            f"Release tag '{tag}' must start with v and use vMAJOR.MINOR.PATCH (e.g. v0.1.0)")
    stripped = tag[1:]
    require(parse_semver_tuple(stripped) is not None,
            f"Release tag '{tag}' must match vMAJOR.MINOR.PATCH (e.g. v0.1.0)")
    version = project_versions(root)
    require(stripped == version,
            f"Release tag '{tag}' does not match project version v{version}")
    return version


def _validate_zip_container(raw, filename):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        bad = archive.testzip()
        require(bad is None, f'{filename}: corrupt ZIP member {bad}')
        infos = archive.infolist()
        require(infos, f'{filename}: empty archive')
        seen = set()
        seen_folded = set()
        names = []
        for info in infos:
            name = info.filename
            require(bool(name), f'{filename}: empty ZIP entry name')
            require(not name.startswith('/') and '\\' not in name and ':' not in name,
                    f'{filename}: unsafe ZIP member path: {name}')
            require(all(ord(ch) >= 32 and ord(ch) != 127 for ch in name),
                    f'{filename}: control characters in ZIP member path')
            trimmed = name[:-1] if name.endswith('/') else name
            parts = trimmed.split('/')
            require(all(part not in ('', '.', '..') for part in parts),
                    f'{filename}: unsafe ZIP member path: {name}')
            normalized = '/'.join(parts)
            folded = normalized.lower()
            require(name not in seen and folded not in seen_folded,
                    f'{filename}: duplicate ZIP member {normalized}')
            seen.add(name)
            seen_folded.add(folded)
            mode = info.external_attr >> 16
            if mode:
                fmt = stat.S_IFMT(mode)
                require(not stat.S_ISLNK(mode), f'{filename}: symlink member not allowed: {name}')
                require(fmt in (0, stat.S_IFREG, stat.S_IFDIR),
                        f'{filename}: special ZIP member not allowed: {name}')
                require((mode & 0o6000) == 0,
                        f'{filename}: setuid/setgid ZIP member not allowed: {name}')
            if info.is_dir():
                continue
            names.append(name)
        return names


def validate_ci_source_archive(path, *, expected_version=None, verify_against_root=None):
    require(path.name == 'unaveraged-source.zip', f'Unexpected source archive name: {path.name}')
    require(path.is_file() and not path.is_symlink(), f'Missing source archive: {path}')
    raw = path.read_bytes()
    names = _validate_zip_container(raw, path.name)
    expected_files = set(build.SOURCE_FILES)
    require(set(names) == expected_files,
            f'{path.name}: unexpected or missing source entries')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if expected_version is not None:
            pkg = json.loads(archive.read('package.json'))
            lock = json.loads(archive.read('package-lock.json'))
            archived_versions = {
                'package.json': pkg.get('version'),
                'package-lock.json': lock.get('version'),
                'package-lock.json packages[""].version': lock.get('packages', {}).get('', {}).get('version'),
            }
            for project in build.PROJECTS:
                ext = json.loads(archive.read(f'{project}/extension.json'))
                archived_versions[f'{project}/extension.json'] = ext.get('version')
            for source, value in archived_versions.items():
                require(value == expected_version,
                        f'{path.name}: {source} version {value!r} does not match expected {expected_version!r}')
        if verify_against_root is not None:
            for name in build.SOURCE_FILES:
                source = verify_against_root / name
                require(source.is_file() and not source.is_symlink(),
                        f'{path.name}: missing source file on disk: {name}')
                require(archive.read(name) == source.read_bytes(),
                        f'{path.name}: stale source entry: {name}; run make source')
    return {
        'path': path,
        'kind': 'source',
        'project': 'source',
        'browser': 'source',
        'precision': 'source',
        'version': expected_version or '',
        'runtime_version': '',
        'sha256': digest(raw),
        'bytes': len(raw),
    }


def validate_ci_extension_archive(path, project, browser='chrome', *, expected_version=None, root=ROOT):
    require(project in build.PROJECTS, f'Unknown extension: {project}')
    require(browser in build.BROWSERS, f'Unknown browser: {browser}')
    config = read(root / project / 'extension.json')
    precision = config['default_precision']
    expected_name = f"{config['slug']}-{browser}.zip"
    require(path.name == expected_name, f'{project}: expected filename {expected_name}, got {path.name}')
    require(path.is_file() and not path.is_symlink(), f'Missing extension ZIP: {path}')
    frozen_file = root / project / 'frozen-profiles.json'
    profile = read(frozen_file)['profiles'].get(precision) if frozen_file.is_file() else None
    if profile is None:
        profile = DEFAULT_PROFILES.get(project, {}).get(precision)
    runtime_version = read(root / 'package.json')['config']['volvoxai_version']
    lock = read(root / 'package-lock.json')['packages']['node_modules/volvoxai-lite']
    require(lock.get('version') == runtime_version
            and lock.get('resolved') == f'https://registry.npmjs.org/volvoxai-lite/-/volvoxai-lite-{runtime_version}.tgz'
            and lock.get('integrity'),
            'Invalid pinned runtime entry in package-lock.json')
    raw = path.read_bytes()
    names = _validate_zip_container(raw, path.name)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        copied = source_files(root, project, config)
        model_manifest_name = 'model/model-manifest.json' if 'model/model-manifest.json' in names else 'model/manifest.json'
        model_manifest_file = 'model-manifest.json' if 'model/model-manifest.json' in names else 'manifest.json'
        info = json.loads(archive.read('build-info.json'))
        model = json.loads(archive.read(model_manifest_name))
        frozen = 'frozen_candidate' in model
        if frozen:
            require(profile is not None and 'sha256' in profile, f'{project}: unknown frozen candidate')
            model_files = {model_manifest_file, *profile['sha256']}
        elif project == 'digit':
            model_files = {model_manifest_file, 'graph.json', 'model.safetensors'}
        else:
            model_files = {model_manifest_file, 'config.json', 'vocab.json',
                           'encoder/graph.json', 'encoder/model.safetensors',
                           'decoder/graph.json', 'decoder/model.safetensors',
                           'shared/token-embedding.safetensors'}
        allowed = {*copied, 'manifest.json', 'build-info.json', 'privacy.html',
                   'shared/extension/site-config.js', 'model/LICENSE.txt',
                   *(f'model/{name}' for name in model_files),
                   'vendor/volvoxai.lite.min.js', 'vendor/volvoxai.lite.wasm'}
        if 'NOTICE' in config['model']['common_files']:
            allowed.add('model/NOTICE.txt')
        require(len(names) == len(set(names)) and set(names) == allowed,
                f'{project}: unexpected or missing ZIP entries in {path.name}')
        require(info['extension'] == project and info['extension_version'] == config['version']
                and info.get('browser') == browser
                and info['model_precision'] == precision and info['model_variant'] == 'ptq'
                and info.get('model_repository') == config['model']['repository']
                and info.get('model_revision') == config['model']['revision']
                and info['runtime_profile'] == 'lite' and info['runtime_version'] == runtime_version,
                f'{project}: ZIP does not match the selected model/runtime')
        if expected_version is not None:
            require(info['extension_version'] == expected_version,
                    f'{project}: extension version {info["extension_version"]} does not match expected {expected_version}')
        require(all(info['dependencies'][kind]['version'] == runtime_version
                    and info['dependencies'][kind]['source'] == 'published' for kind in ('javascript', 'python')),
                f'{project}: release requires published npm/wheel dependencies')
        require(info['dependencies']['javascript'].get('integrity') == lock['integrity']
                and info['dependencies']['javascript'].get('resolved') == lock['resolved'],
                f'{project}: npm provenance differs from the lockfile')
        require(model['quantization']['precision'] == precision
                and info['model_quantization'] == model['quantization'],
                f'{project}: selected quantization metadata differs')
        if frozen:
            require(model.get('frozen_candidate') == profile
                    and model['quantization'].get('packed_arithmetic') == 'dynamic-i8',
                    f'{project}: selected quantization metadata differs')
            for name, expected in profile['sha256'].items():
                require(digest(archive.read('model/' + name)) == expected, f'{project}: unselected model: {name}')
        require(set(info['model_sha256']) == model_files, f'{project}: incomplete model identity')
        for name, expected in info['model_sha256'].items():
            require(digest(archive.read('model/' + name)) == expected, f'{project}: model hash mismatch: {name}')
        runtime_names = ('volvoxai.lite.min.js', 'volvoxai.lite.wasm')
        require(set(info['runtime_sha256']) == set(runtime_names), f'{project}: incomplete runtime identity')
        for name in runtime_names:
            payload = archive.read('vendor/' + name)
            require(digest(payload) == info['runtime_sha256'][name] and len(payload) > 0,
                    f'{project}: runtime mismatch: {name}')
        for name, source in copied.items():
            if name == 'vendor/LICENSE.txt':
                require(len(archive.read(name).strip()) > 0, f'{project}: empty vendor license')
            elif source.is_file():
                require(archive.read(name) == source.read_bytes(), f'{project}: stale packaged source: {name}')
        sites = build.site_config(config, root=root)
        require(json.loads(archive.read('manifest.json')) == build.extension_manifest(config, sites, root=root, browser=browser),
                f'{project}: non-production {browser} manifest')
        expected_sites = 'globalThis.RECEIPT_READER_SITES = ' + json.dumps(sites, ensure_ascii=False) + ';\n'
        require(archive.read('shared/extension/site-config.js').decode() == expected_sites,
                f'{project}: non-production site configuration')
        privacy_bytes = archive.read('privacy.html')
        privacy_text = privacy_bytes.decode()
        require(privacy_text.startswith('<!doctype html>') and len(privacy_text.strip()) > 0,
                f'{project}: invalid privacy page')
        if (root / 'PRIVACY.md').is_file():
            try:
                expected_privacy = build.privacy_page(root=root)
            except RuntimeError:
                expected_privacy = None
            if expected_privacy is not None:
                require(privacy_text == expected_privacy, f'{project}: stale privacy page')
    return {
        'path': path,
        'kind': 'extension',
        'project': project,
        'browser': browser,
        'precision': precision,
        'version': config['version'],
        'runtime_version': runtime_version,
        'sha256': digest(raw),
        'bytes': len(raw),
    }


def _parse_checksums(raw_text, expected_names):
    parsed = {}
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f'Invalid SHA256SUMS line: {line!r}')
        digest_value, name = parts[0], parts[1].lstrip('*')
        require(_SHA256_HEX_RE.fullmatch(digest_value) is not None,
                f'Invalid SHA-256 digest in SHA256SUMS: {digest_value!r}')
        require(name not in parsed, f'Duplicate SHA256SUMS entry: {name}')
        parsed[name] = digest_value
    require(set(parsed) == set(expected_names),
            'SHA256SUMS entries do not match staged archives: expected '
            + ', '.join(sorted(expected_names)) + '; got ' + ', '.join(sorted(parsed)))
    return parsed


def _walk_sorted(directory, out):
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            if path.name.endswith('.zip') or path.name == 'SHA256SUMS':
                raise ValueError(f'Package must be a regular file inside dist: {path.name}')
            continue
        if path.is_dir():
            _walk_sorted(path, out)
        elif path.is_file():
            out.append(path)


def discover_ci_assets(dist, *, require_source=False, require_checksums=False, expected_version=None, root=ROOT):
    dist = Path(dist)
    require(dist.exists() and dist.is_dir() and not dist.is_symlink(),
            f'No release archives in {dist}; build packages first')
    dist_canon = dist.resolve()
    expected_ext_map = {}
    for project in build.PROJECTS:
        config = read(root / project / 'extension.json')
        for browser in build.BROWSERS:
            expected_ext_map[f"{config['slug']}-{browser}.zip"] = (project, browser)
    allowed_zips = set(expected_ext_map) | {'unaveraged-source.zip'}

    files = []
    _walk_sorted(dist, files)
    discovered = {}
    for path in files:
        if path.suffix != '.zip':
            continue
        require(path.resolve().is_relative_to(dist_canon),
                f'Package must stay inside dist: {path.name}')
        require(path.name in allowed_zips,
                f'Unexpected archive in {dist}: {path.name}')
        require(path.name not in discovered,
                f'Duplicate archive for {path.name}; keep only one in {dist}')
        discovered[path.name] = path

    missing = [name for name in expected_ext_map if name not in discovered]
    require(not missing, 'Missing required extension ZIP(s): ' + ', '.join(missing))
    if require_source:
        require('unaveraged-source.zip' in discovered,
                f'Missing required source archive unaveraged-source.zip in {dist}')

    assets = []
    for name, (project, browser) in expected_ext_map.items():
        assets.append(validate_ci_extension_archive(
            discovered[name], project, browser, expected_version=expected_version, root=root,
        ))
    if 'unaveraged-source.zip' in discovered:
        assets.append(validate_ci_source_archive(
            discovered['unaveraged-source.zip'], expected_version=expected_version,
        ))

    checksum_path = dist / 'SHA256SUMS'
    require(not checksum_path.is_symlink(), 'SHA256SUMS must be a regular file')
    if require_checksums:
        require(checksum_path.is_file(), f'Missing required SHA256SUMS in {dist}')
    if checksum_path.is_file():
        expected_checksums = _parse_checksums(
            checksum_path.read_text(),
            [asset['path'].name for asset in assets],
        )
        for asset in assets:
            name = asset['path'].name
            require(asset['sha256'] == expected_checksums[name],
                    f'SHA256SUMS mismatch for {name}')
    return assets


def stage_ci(out_dir, *, browser='all', include_source=True, dry_run=False, root=ROOT):
    require(browser in ('all', *build.BROWSERS), f'Unknown browser: {browser}')
    version = project_versions(root)
    assets = collect_assets(root=root, browser=browser)
    for asset in assets:
        require(asset['version'] == version,
                f"{asset['project']} version {asset['version']} does not match project version {version}")
    if include_source:
        source_zip = root / 'dist/unaveraged-source.zip'
        assets.append(validate_ci_source_archive(
            source_zip, expected_version=version, verify_against_root=root,
        ))
    out_resolved = Path(out_dir).resolve()
    protected = [(root / name).resolve() for name in (
        'docs', 'example', 'shared', 'digit', 'vqa', 'synth', 'benchmarks', '.github',
    )]
    require(out_resolved != root.resolve()
            and not any(src.is_relative_to(out_resolved) or out_resolved.is_relative_to(src) for src in protected),
            'Choose a dedicated output directory for staged CI assets')
    if dry_run:
        return assets
    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stage-ci-', dir=out_resolved.parent) as temporary:
        stage = Path(temporary) / 'dist'
        stage.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            data = asset['path'].read_bytes()
            require(digest(data) == asset['sha256'],
                    f"{asset['path'].name} changed after validation; rebuild first")
            (stage / asset['path'].name).write_bytes(data)
        checksums = checksum_text(assets)
        _parse_checksums(checksums, [asset['path'].name for asset in assets])
        (stage / 'SHA256SUMS').write_text(checksums)
        if out_resolved.exists():
            shutil.rmtree(out_resolved)
        shutil.move(str(stage), out_resolved)
    return assets


def checksum_text(assets):
    """Standard sha256sum format for exactly the validated release ZIPs."""
    return ''.join(f"{asset['sha256']}  {asset['path'].name}\n"
                   for asset in sorted(assets, key=lambda asset: asset['path'].name))


def resolve_repository(repo, root):
    """Confirm repository access before interpreting a missing release as 404."""
    command = ['gh', 'repo', 'view', *([repo] if repo else []),
               '--json', 'nameWithOwner', '--jq', '.nameWithOwner']
    resolved = subprocess.check_output(command, cwd=root, text=True).strip()
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', resolved),
            'GitHub did not return a valid repository name')
    return resolved


def models_release_exists(repo, root):
    """Only a confirmed HTTP 404 means this rolling release must be created."""
    command = ['gh', 'api', '--include', f'repos/{repo}/releases/tags/models']
    response = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    statuses = re.findall(r'^HTTP/\S+\s+(\d{3})(?:\s|$)', response.stdout, re.MULTILINE)
    status = int(statuses[-1]) if statuses else None
    if status == 404:
        return False
    require(status == 200 and response.returncode == 0,
            'Cannot determine whether the models release exists: '
            + (f'HTTP {status}' if status is not None else 'GitHub authentication or network error'))
    return True


def publish(action, assets, tag, repo='', draft=False, clobber=False, dry_run=False, root=ROOT):
    require(action in ('create', 'upload', 'models'), 'Unknown release action')
    require(assets and len({asset['path'].name for asset in assets}) == len(assets), 'Release ZIP names must be nonempty and unique')
    rolling = action == 'models'
    if rolling:
        require(tag in ('', 'models'), 'The models action uses the fixed models tag; omit TAG')
        tag = 'models'
        require(len(assets) == len(build.PROJECTS) * len(build.BROWSERS)
                and {(asset['project'], asset['browser']) for asset in assets}
                == {(project, browser) for project in build.PROJECTS for browser in build.BROWSERS},
                'The models action requires all four Chrome/Firefox production ZIPs')
    else:
        require(tag and not tag.startswith('-'), 'Set TAG, for example: make release TAG=v0.5.0')
        subprocess.run(['git', 'check-ref-format', 'refs/tags/' + tag], cwd=root, check=True)
    require(not repo or re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo), 'REPO must be OWNER/REPO')
    require(not draft or action == 'create', 'DRAFT is only supported when creating a release')
    require(not clobber or action in ('upload', 'models'), 'CLOBBER is only supported by release-upload')
    command = ['gh', 'release', 'create' if rolling else action, tag]
    if repo:
        command += ['--repo', repo]
    if rolling:
        # The storage tag is created on GitHub's default branch. Local HEAD may
        # not have been pushed and is not a prerequisite for publishing bytes.
        command += ['--title', 'Local model packages', '--prerelease', '--latest=false']
    elif action == 'create':
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        command += ['--target', commit, '--title', 'unaveraged ' + tag]
        if draft:
            command.append('--draft')
    elif clobber:
        command.append('--clobber')
    with tempfile.TemporaryDirectory(prefix='unaveraged-release-') as temporary:
        staging = Path(temporary)
        # Upload exactly the validated bytes, even if dist is rebuilt concurrently.
        paths = []
        for asset in assets:
            data = asset['path'].read_bytes()
            require(digest(data) == asset['sha256'], 'ZIP changed after validation; run release again')
            target = staging/asset['path'].name
            target.write_bytes(data)
            paths.append(str(target))
        checksums = staging/'SHA256SUMS'
        checksums.write_text(checksum_text(assets))
        paths.append(str(checksums))
        if action in ('create', 'models'):
            notes = staging/'notes.md'
            notes.write_text('Chrome and Firefox extensions with bundled quantized models and '
                f"VolvoxAI {assets[0]['runtime_version']}.\n\n"
                '| Extension | Browser | Model | Extension version |\n| --- | --- | --- | --- |\n'
                + ''.join(f"| {a['project'].title()} | {a['browser'].title()} | {a['precision']} | {a['version']} |\n" for a in assets)
                + '\nDownload the ZIP for your browser and extract it. In Chrome, open '
                '`chrome://extensions`, enable Developer mode and choose **Load unpacked**. '
                'In Firefox, open `about:debugging#/runtime/this-firefox`, choose **Load Temporary Add-on** '
                'and select the extracted `manifest.json`. Temporary Firefox installs last until browser restart; '
                'regular installation requires Mozilla signing.\n\n'
                'The accompanying `SHA256SUMS` verifies the exact ZIP downloads.\n')
            command += ['--notes-file', str(notes)]
        command += paths
        if rolling:
            upload = ['gh', 'release', 'upload', tag, *(['--repo', repo] if repo else []), '--clobber', *paths]
            if dry_run:
                print('Dry run (models release absent): ' + shlex.join(command), flush=True)
                print('Dry run (models release exists; replace these assets): ' + shlex.join(upload), flush=True)
                return
            resolved = resolve_repository(repo, root)
            # Both the lookup and mutation explicitly target the same resolved
            # repository, including when REPO was omitted.
            for candidate in (command, upload):
                if '--repo' in candidate:
                    candidate[candidate.index('--repo') + 1] = resolved
                else:
                    candidate[4:4] = ['--repo', resolved]
            if models_release_exists(resolved, root):
                command = upload
        print(('Dry run: ' if dry_run else 'Running: ') + shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=root, check=True)


def _gh_api(path, *, root=ROOT, paginate=False, method=None, fields=None):
    command = ['gh', 'api', '--hostname', 'github.com', path]
    if paginate:
        command += ['--paginate', '--slurp']
    if method:
        command += ['--method', method]
    if fields:
        for flag, value in fields:
            command += [flag, value]
    response = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if response.returncode != 0:
        detail = response.stderr.strip() or response.stdout.strip() or f'exit {response.returncode}'
        raise ValueError(f'gh api failed ({path}): {detail}')
    raw = response.stdout.strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError('GitHub returned an invalid JSON response') from error
    if paginate:
        require(isinstance(value, list) and all(isinstance(page, list) for page in value),
                'GitHub returned invalid paginated data')
        return [item for page in value for item in page]
    return value


def list_releases(repo, root=ROOT):
    return _gh_api(f'repos/{repo}/releases?per_page=100', root=root, paginate=True)


def verify_release_assets(repo, tag, upload_paths, *, expected_draft, expected_prerelease, root=ROOT):
    found = [rel for rel in list_releases(repo, root=root) if rel.get('tag_name') == tag]
    draft_ok = expected_draft is None or (len(found) == 1 and found[0].get('draft') is expected_draft)
    prerelease_ok = len(found) == 1 and found[0].get('prerelease') is expected_prerelease
    if len(found) != 1 or not draft_ok or not prerelease_ok:
        if expected_draft is True and not expected_prerelease:
            raise ValueError('Expected one unpublished stable draft after upload; refusing to publish')
        if expected_draft is True and expected_prerelease:
            raise ValueError('Expected one unpublished nightly prerelease draft after upload; refusing to publish')
        if expected_prerelease:
            raise ValueError('Expected one nightly prerelease after upload; refusing to update nightly release')
        raise ValueError('Expected one release matching expected publication state after upload; refusing to publish')
    expected = {}
    for path in upload_paths:
        raw = path.read_bytes()
        expected[path.name] = (len(raw), 'sha256:' + digest(raw))
    actual = found[0].get('assets')
    require(isinstance(actual, list),
            'Draft assets do not match the complete package set; refusing to publish')
    actual_names = {asset.get('name') for asset in actual if isinstance(asset, dict) and asset.get('name')}
    require(len(actual) == len(expected) and actual_names == set(expected),
            'Draft assets do not match the complete package set; refusing to publish')
    for asset in actual:
        name = asset.get('name', '')
        exp_size, exp_digest = expected[name]
        if asset.get('state') != 'uploaded' or asset.get('size') != exp_size:
            raise ValueError(f'Draft asset is incomplete: {name}')
        reported = asset.get('digest')
        if not isinstance(reported, str) or not _GITHUB_DIGEST_RE.fullmatch(reported):
            raise ValueError(f'Draft asset has no valid GitHub SHA-256 digest: {name}; refusing to publish')
        if reported != exp_digest:
            raise ValueError(f'Draft asset checksum does not match: {name}; refusing to publish')


def update_nightly_git_ref(repo, sha, root=ROOT):
    refs = _gh_api(f'repos/{repo}/git/matching-refs/tags/nightly', root=root)
    has_exact = isinstance(refs, list) and any(
        isinstance(item, dict) and item.get('ref') == 'refs/tags/nightly' for item in refs
    )
    if has_exact:
        _gh_api(
            f'repos/{repo}/git/refs/tags/nightly',
            root=root,
            method='PATCH',
            fields=[('-f', f'sha={sha}'), ('-F', 'force=true')],
        )
    else:
        _gh_api(
            f'repos/{repo}/git/refs',
            root=root,
            method='POST',
            fields=[('-f', 'ref=refs/tags/nightly'), ('-f', f'sha={sha}')],
        )


def _ci_release_notes(channel, assets, *, sha='', tag=''):
    ext_assets = [asset for asset in assets if asset.get('kind', 'extension') != 'source']
    has_source = any(asset.get('kind') == 'source' for asset in assets)
    ext_version = ext_assets[0]['version']
    runtime_version = ext_assets[0]['runtime_version']
    if channel == 'nightly':
        intro = (
            f'Rolling nightly pre-release built transparently from source commit `{sha}`.\n\n'
            f'Version: `{ext_version}-nightly ({sha[:7]})` · VolvoxAI `{runtime_version}`.'
        )
    else:
        intro = (
            f'Stable release `{tag}` (extension version `{ext_version}`, VolvoxAI `{runtime_version}`) '
            f'built transparently from source commit `{sha}`.'
        )
    rows = ''.join(
        f"| {a['project'].title()} | {a['browser'].title()} | {a['precision']} | "
        f"{a['version']} | `{a['path'].name}` |\n"
        for a in ext_assets
    )
    source_note = (
        '\nThe accompanying `unaveraged-source.zip` contains the exported open-source tree for this build, '
        if has_source else '\n'
    )
    checksum_note = (
        ('and ' if has_source else 'The accompanying ')
        + '`SHA256SUMS` verifies the exact ZIP downloads (`sha256sum -c SHA256SUMS`).\n'
    )
    return (
        f'{intro}\n\n'
        '| Extension | Browser | Model | Extension version | Archive |\n'
        '| --- | --- | --- | --- | --- |\n'
        f'{rows}\n'
        'Download the ZIP for your browser and extract it. In Chrome, open '
        '`chrome://extensions`, enable Developer mode and choose **Load unpacked**. '
        'In Firefox, open `about:debugging#/runtime/this-firefox`, choose **Load Temporary Add-on** '
        'and select the extracted `manifest.json`. Temporary Firefox installs last until browser restart; '
        'regular installation requires Mozilla signing.\n'
        f'{source_note}{checksum_note}'
    )


def publish_ci(channel, assets, sha, *, tag='', repo='', dry_run=False, root=ROOT):
    require(channel in ('nightly', 'stable'), f'Unsupported CI release channel: {channel!r}')
    require(isinstance(sha, str) and _SHA_RE.fullmatch(sha) is not None,
            'Commit SHA must be a 7..=40 character hexadecimal string')
    sha_lower = sha.lower()
    require(not repo or _REPO_RE.fullmatch(repo) is not None, 'REPO must be OWNER/REPO')
    require(assets and len({asset['path'].name for asset in assets}) == len(assets),
            'Release archive names must be nonempty and unique')
    ext_assets = [asset for asset in assets if asset.get('kind', 'extension') != 'source']
    require(len(ext_assets) == len(build.PROJECTS) * len(build.BROWSERS)
            and {(asset['project'], asset['browser']) for asset in ext_assets}
            == {(project, browser) for project in build.PROJECTS for browser in build.BROWSERS},
            'CI release requires all four Chrome/Firefox production extension ZIPs')
    version = project_versions(root)
    for asset in ext_assets:
        require(asset['version'] == version,
                f"{asset['project']} version {asset['version']} does not match project version {version}")
    if channel == 'stable':
        validate_release_tag(tag, root=root)
        target_tag = tag
        plan_tuple = parse_semver_tuple(version)
    else:
        require(tag in ('', 'nightly'), '--tag must be nightly (or omitted) when --channel nightly is used')
        target_tag = 'nightly'

    if dry_run:
        display_repo = repo or '<resolved-repo>'
        names = [asset['path'].name for asset in assets] + ['SHA256SUMS']
        print(f'Dry run ({channel} -> {target_tag} @ {sha_lower[:7]} on {display_repo}): '
              + ', '.join(names), flush=True)
        return

    target_repo = repo or resolve_repository(repo, root)
    existing_releases = list_releases(target_repo, root=root)

    if channel == 'stable':
        for existing in existing_releases:
            existing_tag = existing.get('tag_name', '')
            if existing_tag == target_tag:
                raise ValueError(
                    f'Release {target_tag} already exists (including drafts); it will not be overwritten'
                )
            if not existing.get('draft') and not existing.get('prerelease'):
                previous = existing_tag[1:] if existing_tag.startswith('v') else existing_tag
                prev_tuple = parse_semver_tuple(previous)
                if prev_tuple is not None and prev_tuple >= plan_tuple:
                    raise ValueError(
                        f'Release {target_tag} version must be newer than published stable release {existing_tag}'
                    )
        existing_nightly = None
    else:
        matches = [rel for rel in existing_releases if rel.get('tag_name') == 'nightly']
        require(len(matches) <= 1, 'Multiple nightly releases found; remove duplicates before updating')
        existing_nightly = matches[0] if matches else None
        if existing_nightly is not None:
            require(existing_nightly.get('prerelease') is True,
                    'Expected one nightly prerelease; existing nightly release is not a prerelease')

    with tempfile.TemporaryDirectory(prefix='unaveraged-release-ci-') as temporary:
        work = Path(temporary)
        upload_dir = work / 'assets'
        upload_dir.mkdir(parents=True)
        upload_paths = []
        for asset in assets:
            data = asset['path'].read_bytes()
            require(digest(data) == asset['sha256'],
                    f"{asset['path'].name} changed after validation; refusing upload")
            staged = upload_dir / asset['path'].name
            staged.write_bytes(data)
            upload_paths.append(staged)
        checksums = upload_dir / 'SHA256SUMS'
        checksums.write_text(checksum_text(assets))
        upload_paths.append(checksums)
        notes_file = work / 'notes.md'
        notes_file.write_text(_ci_release_notes(channel, assets, sha=sha_lower, tag=target_tag))

        if channel == 'stable':
            create_cmd = [
                'gh', 'release', 'create', target_tag,
                *(str(path) for path in upload_paths),
                '--repo', target_repo,
                '--target', sha_lower,
                '--draft',
                '--title', f'unaveraged {target_tag}',
                '--notes-file', str(notes_file),
            ]
            edit_cmd = [
                'gh', 'release', 'edit', target_tag,
                '--repo', target_repo,
                '--draft=false',
                '--latest',
            ]
            try:
                subprocess.run(create_cmd, cwd=root, check=True)
                verify_release_assets(
                    target_repo, target_tag, upload_paths,
                    expected_draft=True, expected_prerelease=False, root=root,
                )
                subprocess.run(edit_cmd, cwd=root, check=True)
            except (ValueError, subprocess.CalledProcessError) as error:
                raise ValueError(
                    f'{error}\nInspect https://github.com/{target_repo}/releases before retrying. '
                    'A failed upload may have left a draft; existing releases are never overwritten.'
                ) from error
            print(f'Published https://github.com/{target_repo}/releases/tag/{target_tag}', flush=True)
            return

        title = f'unaveraged nightly ({sha_lower[:7]})'
        if existing_nightly is None:
            create_cmd = [
                'gh', 'release', 'create', 'nightly',
                *(str(path) for path in upload_paths),
                '--repo', target_repo,
                '--target', sha_lower,
                '--draft',
                '--prerelease',
                '--latest=false',
                '--title', title,
                '--notes-file', str(notes_file),
            ]
            edit_cmd = [
                'gh', 'release', 'edit', 'nightly',
                '--repo', target_repo,
                '--draft=false',
                '--prerelease',
                '--latest=false',
            ]
            try:
                subprocess.run(create_cmd, cwd=root, check=True)
                verify_release_assets(
                    target_repo, 'nightly', upload_paths,
                    expected_draft=True, expected_prerelease=True, root=root,
                )
                subprocess.run(edit_cmd, cwd=root, check=True)
            except (ValueError, subprocess.CalledProcessError) as error:
                raise ValueError(
                    f'{error}\nInspect https://github.com/{target_repo}/releases before retrying.'
                ) from error
        else:
            expected_names = {path.name for path in upload_paths}
            stale_names = [
                item['name'] for item in (existing_nightly.get('assets') or [])
                if isinstance(item, dict) and item.get('name') and item['name'] not in expected_names
            ]
            upload_cmd = [
                'gh', 'release', 'upload', 'nightly',
                *(str(path) for path in upload_paths),
                '--repo', target_repo,
                '--clobber',
            ]
            edit_cmd = [
                'gh', 'release', 'edit', 'nightly',
                '--repo', target_repo,
                '--draft=false',
                '--prerelease',
                '--latest=false',
                '--target', sha_lower,
                '--title', title,
                '--notes-file', str(notes_file),
            ]
            try:
                subprocess.run(upload_cmd, cwd=root, check=True)
                for stale in stale_names:
                    subprocess.run([
                        'gh', 'release', 'delete-asset', 'nightly', stale,
                        '--repo', target_repo,
                        '--yes',
                    ], cwd=root, check=True)
                verify_release_assets(
                    target_repo, 'nightly', upload_paths,
                    expected_draft=None, expected_prerelease=True, root=root,
                )
                update_nightly_git_ref(target_repo, sha_lower, root=root)
                subprocess.run(edit_cmd, cwd=root, check=True)
            except (ValueError, subprocess.CalledProcessError) as error:
                raise ValueError(
                    f'{error}\nInspect https://github.com/{target_repo}/releases before retrying.'
                ) from error
        print(f'Updated https://github.com/{target_repo}/releases/tag/nightly ({sha_lower[:7]})', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'action',
        choices=('check', 'create', 'upload', 'models', 'check-tag', 'stage-ci', 'verify-ci', 'release-ci'),
    )
    parser.add_argument('--browser', choices=('all', *build.BROWSERS), default=os.environ.get('BROWSER', 'all'))
    parser.add_argument('--tag', default=os.environ.get('TAG', ''))
    parser.add_argument('--repo', default=os.environ.get('REPO', ''))
    parser.add_argument('--draft', action='store_true', default=os.environ.get('DRAFT') == '1')
    parser.add_argument('--clobber', action='store_true', default=os.environ.get('CLOBBER') == '1')
    parser.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN') == '1')
    parser.add_argument('--channel', choices=('nightly', 'stable'), default='')
    parser.add_argument('--sha', default='')
    parser.add_argument('--dist', type=Path, default=ROOT / 'dist')
    parser.add_argument('--out', type=Path, default=ROOT / 'release-dist')
    parser.add_argument('--include-source', action='store_true')
    parser.add_argument('--require-source', action='store_true')
    parser.add_argument('--require-checksums', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.action == 'check-tag':
            version = validate_release_tag(args.tag, root=ROOT)
            print(f'Tag {args.tag} matches project version v{version}', flush=True)
            return
        if args.action == 'stage-ci':
            assets = stage_ci(
                args.out,
                browser=args.browser,
                include_source=args.include_source,
                dry_run=args.dry_run,
                root=ROOT,
            )
            for asset in assets:
                print(
                    f"{asset['path'].name} · {asset['precision']} · "
                    f"{asset['bytes']/1048576:.2f} MiB · SHA256 {asset['sha256']}"
                )
            return
        if args.action == 'verify-ci':
            expected_version = (
                validate_release_tag(args.tag, root=ROOT) if args.tag else project_versions(ROOT)
            )
            dist = args.dist if args.dist.is_absolute() else ROOT / args.dist
            assets = discover_ci_assets(
                dist,
                require_source=args.require_source,
                require_checksums=args.require_checksums,
                expected_version=expected_version,
                root=ROOT,
            )
            for asset in assets:
                print(
                    f"{asset['path'].name} · {asset['precision']} · "
                    f"{asset['bytes']/1048576:.2f} MiB · SHA256 {asset['sha256']}"
                )
            return
        if args.action == 'release-ci':
            require(args.channel in ('nightly', 'stable'),
                    '--channel <nightly|stable> is required for release-ci')
            require(isinstance(args.sha, str) and _SHA_RE.fullmatch(args.sha) is not None,
                    '--sha must be a 7..=40 character hexadecimal commit SHA')
            expected_version = (
                validate_release_tag(args.tag, root=ROOT)
                if args.channel == 'stable' else project_versions(ROOT)
            )
            dist = args.dist if args.dist.is_absolute() else ROOT / args.dist
            assets = discover_ci_assets(
                dist,
                require_source=args.require_source,
                require_checksums=args.require_checksums,
                expected_version=expected_version,
                root=ROOT,
            )
            for asset in assets:
                print(
                    f"{asset['path'].name} · {asset['precision']} · "
                    f"{asset['bytes']/1048576:.2f} MiB · SHA256 {asset['sha256']}"
                )
            publish_ci(
                args.channel,
                assets,
                args.sha,
                tag=args.tag,
                repo=args.repo,
                dry_run=args.dry_run,
                root=ROOT,
            )
            return
        require(args.action != 'models' or args.browser == 'all',
                'The models action requires --browser all')
        assets = collect_assets(browser=args.browser)
        for asset in assets:
            print(f"{asset['path'].relative_to(ROOT)} · {asset['precision']} · {asset['bytes']/1048576:.2f} MiB · SHA256 {asset['sha256']}")
        if args.action != 'check':
            publish(args.action, assets, args.tag, args.repo, args.draft, args.clobber, args.dry_run)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Release failed: {error}\n')


if __name__ == '__main__':
    main()
