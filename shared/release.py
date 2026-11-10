#!/usr/bin/env python3
"""Validate and upload the selected Chrome/Firefox extension ZIPs and SHA256SUMS."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
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
        info, model = (json.loads(archive.read(name)) for name in ('build-info.json', 'model/manifest.json'))
        frozen = 'frozen_candidate' in model
        if frozen:
            require(profile is not None and 'sha256' in profile, f'{project}: unknown frozen candidate')
            model_files = {'manifest.json', *profile['sha256']}
        elif project == 'digit':
            model_files = {'manifest.json', 'graph.json', 'model.safetensors'}
        else:
            model_files = {'manifest.json', 'config.json', 'vocab.json',
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'create', 'upload', 'models'))
    parser.add_argument('--browser', choices=('all', *build.BROWSERS), default=os.environ.get('BROWSER', 'all'))
    parser.add_argument('--tag', default=os.environ.get('TAG', ''))
    parser.add_argument('--repo', default=os.environ.get('REPO', ''))
    parser.add_argument('--draft', action='store_true', default=os.environ.get('DRAFT') == '1')
    parser.add_argument('--clobber', action='store_true', default=os.environ.get('CLOBBER') == '1')
    parser.add_argument('--dry-run', action='store_true', default=os.environ.get('DRY_RUN') == '1')
    args = parser.parse_args(argv)
    try:
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
