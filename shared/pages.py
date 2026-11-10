#!/usr/bin/env python3
"""Stage the static demo with quantized models from local or Release ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys
import tempfile
from urllib.parse import quote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('digit', 'vqa')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared import docs

EXAMPLE_FILES = ('index.html', 'app.js', 'worker.js', 'questions.js', 'style.css')
MODEL_FILES = {
    'digit': {'graph.json', 'model.safetensors'},
    'vqa': {'config.json', 'vocab.json', 'encoder/graph.json', 'encoder/model.safetensors',
            'decoder/graph.json', 'decoder/model.safetensors', 'shared/token-embedding.safetensors'},
}
RUNTIME_FILES = {'volvoxai.lite.min.js', 'volvoxai.lite.wasm'}
SYNTH_COUNT = 256


def digest(data):
    return hashlib.sha256(data).hexdigest()


def public_bytes(root, relative):
    """Read an explicitly selected source, never an alias to another file."""
    root = root.resolve()
    path = root/relative
    if not path.resolve().is_relative_to(root) or any(
            part.is_symlink() for part in (path, *path.parents) if part != root and root in part.parents):
        raise ValueError('Unsafe public sample path')
    return path.read_bytes()


def verified_bytes(root, relative, expected_hash, expected_size=None):
    if not isinstance(expected_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_hash):
        raise ValueError('Invalid public sample checksum')
    raw = public_bytes(root, relative)
    if digest(raw) != expected_hash:
        raise ValueError(f'Public sample checksum mismatch: {relative}')
    if expected_size is not None and (type(expected_size) is not int or len(raw) != expected_size):
        raise ValueError(f'Public sample size mismatch: {relative}')
    return raw


def sample_text(value, *, maximum=256):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError('Invalid public receipt text')
    return value


def receipt_metadata(source):
    """Publish content needed for questions, excluding geometry and provenance."""
    if not isinstance(source, dict):
        raise ValueError('Invalid public receipt metadata')
    result = {}
    for key in ('phone_digits', 'street_no', 'address', 'store'):
        if key not in source:
            continue
        value = source[key]
        if key == 'phone_digits':
            if not isinstance(value, str) or not re.fullmatch(r'[0-9]{1,16}', value):
                raise ValueError('Invalid public receipt phone digits')
        elif key == 'street_no':
            if not (type(value) is int and 0 <= value <= 10**12 or
                    isinstance(value, str) and re.fullmatch(r'[0-9]{1,12}', value)):
                raise ValueError('Invalid public receipt street number')
        else:
            value = sample_text(value)
        result[key] = value
    if 'items' in source:
        if not isinstance(source['items'], list) or len(source['items']) > 32:
            raise ValueError('Invalid public receipt items')
        result['items'] = []
        for item in source['items']:
            if not isinstance(item, dict):
                raise ValueError('Invalid public receipt item')
            selected = {'name': sample_text(item.get('name'))}
            for key in ('quantity', 'unit_price', 'total'):
                value = item.get(key)
                if type(value) is not int or not 0 <= value <= 10**12:
                    raise ValueError('Invalid public receipt item number')
                selected[key] = value
            result['items'].append(selected)
    return result


def stage_samples(root, demo, *, repo='ivere27/unaveraged', source_ref='main'):
    """Stage only public, hash-verified receipts and a compact question catalog."""
    sources = json.loads(public_bytes(root, 'docs/assets/examples/sources.json'))
    examples = sources.get('examples', [])
    if (sources.get('format') != 'unaveraged-public-synthetic-examples/v1'
            or not isinstance(examples, list) or len(examples) != len(MODELS)
            or any(not isinstance(row, dict) for row in examples)):
        raise ValueError('Invalid public model examples')
    samples = []
    for project in MODELS:
        matches = [row for row in examples if row.get('model') == project]
        if len(matches) != 1:
            raise ValueError('Missing or duplicate public model example')
        row = matches[0]
        asset = f'{project}-receipt-ko.jpg'
        if row.get('asset') != asset or row.get('synthetic') is not True:
            raise ValueError('Unexpected public model example asset')
        raw = verified_bytes(root, f'docs/assets/examples/{asset}', row.get('sha256'))
        if any(type(row.get(key)) is not int or not 0 < row[key] <= 8192 for key in ('width', 'height')):
            raise ValueError('Invalid public model example dimensions')
        questions = row.get('questions')
        if (not isinstance(questions, list) or not 1 <= len(questions) <= 64
                or any(not isinstance(item, dict) for item in questions)):
            raise ValueError('Invalid public model example questions')
        sample = {
            'id': f'model-{project}', 'label': f'{project.upper() if project == "vqa" else "Digit"} 공개 예제',
            'image': f'../assets/examples/{asset}', 'width': row['width'], 'height': row['height'],
            'source': 'model-example', 'models': [project], 'defaultFor': project,
            'questions': [{'question': sample_text(item.get('question')), 'answer': sample_text(item.get('answer'))}
                          for item in questions],
        }
        if 'receipt' in row:
            sample['receipt'] = receipt_metadata(row['receipt'])
        samples.append(sample)
        target = demo.parent/'assets/examples'/asset
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)

    manifest_raw = public_bytes(root, 'synth/manifest.json')
    manifest = json.loads(manifest_raw)
    records = manifest.get('records')
    if (manifest.get('schema_version') != 1 or manifest.get('id') != 'receipt-initial-256'
            or manifest.get('license') != 'Apache-2.0' or manifest.get('path_base') != 'manifest_directory'
            or manifest.get('count') != SYNTH_COUNT or not isinstance(records, list)
            or len(records) != SYNTH_COUNT):
        raise ValueError('Invalid public synthetic sample manifest')
    synth = demo/'synth'
    synth.mkdir(parents=True)
    total_images = total_annotations = 0
    for index, row in enumerate(records):
        identifier = f'{index:06d}'
        if (not isinstance(row, dict) or row.get('id') != identifier or row.get('image') != f'images/{identifier}.png'
                or row.get('annotation') != f'annotations/{identifier}.json'):
            raise ValueError('Unsafe or duplicate synthetic sample path/order')
        if (type(row.get('bytes')) is not int or not 0 < row['bytes'] <= 16 * 1024 * 1024
                or type(row.get('annotation_bytes')) is not int or not 0 < row['annotation_bytes'] <= 2 * 1024 * 1024):
            raise ValueError('Invalid synthetic sample byte count')
        raw = verified_bytes(root, f'synth/{row["image"]}', row.get('sha256'), row.get('bytes'))
        annotation = verified_bytes(root, f'synth/{row["annotation"]}', row.get('annotation_sha256'), row.get('annotation_bytes'))
        if (len(raw) < 33 or raw[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
                or any(type(row.get(key)) is not int or not 0 < row[key] <= 8192 for key in ('width', 'height'))
                or struct.unpack('>II', raw[16:24]) != (row['width'], row['height'])):
            raise ValueError('Invalid synthetic sample image dimensions')
        receipt = receipt_metadata(json.loads(annotation).get('answers'))
        if set(receipt) != {'phone_digits', 'street_no', 'address', 'store', 'items'}:
            raise ValueError('Incomplete synthetic receipt contents')
        samples.append({
            'id': f'synth-{identifier}', 'label': f'{identifier} · {receipt["store"][:60] or "합성 영수증"}',
            'image': f'synth/{identifier}.png', 'width': row['width'], 'height': row['height'],
            'source': 'synth', 'models': list(MODELS), 'receipt': receipt,
        })
        (synth/f'{identifier}.png').write_bytes(raw)
        total_images += len(raw)
        total_annotations += len(annotation)
    if (total_images != manifest.get('total_image_bytes')
            or total_annotations != manifest.get('total_annotation_bytes')):
        raise ValueError('Synthetic sample total size mismatch')
    # Keep a small, explicit attribution record instead of the full generation
    # manifest or annotations (which include glyphs and other unused metadata).
    attribution = {
        'format': 'unaveraged-demo-synth-source/v1', 'dataset': manifest['id'], 'count': SYNTH_COUNT,
        'license': 'Apache-2.0', 'license_file': 'LICENSE.txt', 'notice_file': 'NOTICE.txt',
        'source_url': f'https://github.com/{repo}/tree/{quote(source_ref, safe="")}/synth',
        'manifest_sha256': digest(manifest_raw),
        'scope': 'Initial synthetic calibration receipts; not an independent accuracy benchmark.',
        'pixels': 'Procedural paper/background and rendered font glyphs, with watermarks; no captured receipt images or crops.',
        'text': 'Store names, address prefixes and item names come from an annotation text bank. Phone digits and street numbers are randomly generated.',
    }
    (synth/'source.json').write_text(json.dumps(attribution, ensure_ascii=False, indent=2) + '\n')
    for original, name in (('LICENSE', 'LICENSE.txt'), ('NOTICE', 'NOTICE.txt')):
        (synth/name).write_bytes(public_bytes(root, original))
    catalog = {'format': 'unaveraged-demo-samples/v1', 'samples': samples}
    (demo/'samples.json').write_text(json.dumps(catalog, ensure_ascii=False, separators=(',', ':')) + '\n')
    return catalog


def current_source_files(root, project, info, manifest, archive):
    """Overlay authored code only when it can consume the uploaded model ABI."""
    version = json.loads(public_bytes(root, 'package.json'))['config']['volvoxai_version']
    if info.get('runtime_version') != version:
        raise ValueError(f'{project}: uploaded runtime version does not match current source ({version})')
    try:
        if manifest.get('quantization', {}).get('packed_arithmetic', 'f32') not in ('f32', 'dynamic-i8'):
            raise ValueError('unsupported packed arithmetic')
        if project == 'digit':
            abi, decode = manifest['abi'], manifest['decode']
            valid = (manifest['format'] == 'volvoxai-receipt-digit-reader-onnx-package-v1'
                     and abi['input']['shape'] == [1, 1, 320, 672]
                     and isinstance(abi['input']['name'], str) and bool(abi['input']['name'])
                     and isinstance(abi['output'], str) and bool(abi['output'])
                     and all(decode[key] == value for key, value in
                             {'slots': 16, 'num_classes': 11, 'phone_slots': 12,
                              'street_slots': 4, 'blank_class': 10}.items()))
        else:
            tokenizer, vocab = manifest['tokenizer'], json.loads(archive.read('model/vocab.json'))
            valid = (manifest['format'] == 'receipt-vqa-volvoxai/v1'
                     and manifest['question_capacity'] == 192 and manifest['max_decoder_steps'] == 191
                     and manifest['initial_masked_cache_rows'] == 1
                     and all(tokenizer[key] == value and vocab[key] == value for key, value in
                             {'type': 'byte_fallback_bpe', 'version': 1, 'vocab_size': 1536,
                              'normalization': 'NFC'}.items())
                     and re.fullmatch(r'[0-9a-f]{64}', tokenizer['tokenizer_hash']) is not None
                     and tokenizer['tokenizer_hash'] == vocab['tokenizer_hash'])
            cross = [f'cross_{kind}_{index}' for index in range(4) for kind in ('k', 'v')]
            required = {'encoder': {'image', 'question_ids', 'family_ids'},
                        'decoder': {'decoder_input_ids', 'position_ids', 'family_ids',
                                    'memory_padding_mask', 'past_padding_mask', *cross,
                                    *(name.replace('cross_', 'past_') for name in cross)}}
            for role, inputs in required.items():
                graph = manifest['graphs'][role]
                weights = graph.get('weights', [f'{role}/model.safetensors'])
                valid = (valid and set(graph['inputs']) == inputs
                         and all(isinstance(value, str) and value for value in graph['inputs'].values())
                         and isinstance(weights, list) and bool(weights)
                         and all(isinstance(name, str) for name in weights)
                         and len(set(weights)) == len(weights)
                         and all(name in {f'{role}/model.safetensors', 'shared/token-embedding.safetensors'}
                                 for name in weights))
        if not valid:
            raise ValueError('unsupported execution contract')
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f'{project}: uploaded model is incompatible with current source: {error}') from error
    sources = {'model/adapter.js': f'{project}/adapter.js',
               'shared/extension/runtime.js': 'shared/extension/runtime.js',
               'shared/extension/preprocess.js': 'shared/extension/preprocess.js',
               'LICENSE.txt': 'LICENSE', 'NOTICE.txt': 'NOTICE'}
    if project == 'vqa':
        sources['model/tokenizer.js'] = 'vqa/tokenizer.js'
    payloads = {target: public_bytes(root, source) for target, source in sources.items()}
    license_text = archive.read('vendor/LICENSE.txt').strip()
    if not license_text or license_text not in payloads['NOTICE.txt']:
        raise ValueError('Current NOTICE must retain the full uploaded runtime license for adapted JavaScript')
    return payloads


def stage_model(archive_path, target, project, expected_hash=None, *, source_root=None):
    if project not in MODELS:
        raise ValueError('Unexpected model/runtime package')
    raw = archive_path.read_bytes()
    if expected_hash is not None and digest(raw) != expected_hash:
        raise ValueError(f'{archive_path.name}: Release checksum mismatch')
    # Extract exactly the bytes whose Release hash was verified above.
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('Unsafe ZIP path')
        if sum(entry.file_size for entry in archive.infolist()) > 512 * 1024 * 1024:
            raise ValueError('Demo package is too large')
        info = json.loads(archive.read('build-info.json'))
        if info['extension'] != project or info['runtime_profile'] != 'lite':
            raise ValueError('Unexpected model/runtime package')
        if set(info['model_sha256']) != MODEL_FILES[project] | {'manifest.json'} or set(info['runtime_sha256']) != RUNTIME_FILES:
            raise ValueError('Incomplete or unexpected packaged asset hashes')
        for prefix, hashes in [('model/', info['model_sha256']), ('vendor/', info['runtime_sha256'])]:
            for name, expected in hashes.items():
                if digest(archive.read(prefix + name)) != expected:
                    raise ValueError(f'{project}: invalid packaged asset: {name}')
        # Publish only inference dependencies and licenses. Do not copy arbitrary
        # model/vendor files or full provenance that can include calibration paths.
        selected = {
            *(f'model/{name}' for name in MODEL_FILES[project]),
            *(f'vendor/{name}' for name in RUNTIME_FILES),
            'model/adapter.js', 'model/LICENSE.txt', 'vendor/LICENSE.txt',
            'shared/extension/runtime.js', 'shared/extension/preprocess.js',
            'LICENSE.txt', 'NOTICE.txt',
        }
        if project == 'vqa':
            selected.update({'model/tokenizer.js', 'model/NOTICE.txt'})
        if not selected.issubset(names):
            raise ValueError('Missing public inference dependency or license')
        # Adapters require this execution contract. Retain it without build or
        # calibration provenance, which is not needed by the browser runtime.
        source_manifest = json.loads(archive.read('model/manifest.json'))
        contract_keys = ('format', 'variant', 'abi', 'decode') if project == 'digit' else (
            'format', 'variant', 'graphs', 'tokenizer', 'family_names',
            'question_capacity', 'max_decoder_steps', 'initial_masked_cache_rows',
        )
        contract = {key: source_manifest[key] for key in contract_keys if key in source_manifest}
        contract['quantization'] = {
            'packed_arithmetic': source_manifest.get('quantization', {}).get('packed_arithmetic', 'f32'),
        }
        overrides = (current_source_files(source_root, project, info, source_manifest, archive)
                     if source_root is not None else {})
        manifest_bytes = (json.dumps(contract, ensure_ascii=False) + '\n').encode()
        for name in sorted(selected):
            path = target/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(overrides[name] if name in overrides else archive.read(name))
        (target/'model/manifest.json').write_bytes(manifest_bytes)
        result = {
            'adapter': f'runtime/{project}/model/adapter.js',
            'precision': info['model_precision'], 'runtimeVersion': info['runtime_version'],
            'downloadBytes': len(manifest_bytes) + sum(len(overrides[name]) if name in overrides
                                                     else archive.getinfo(name).file_size for name in selected),
            'archive': archive_path.name, 'archiveSha256': digest(raw),
        }
        if overrides:
            result['currentSourceSha256'] = {name: digest(data) for name, data in sorted(overrides.items())}
        return result


def stage(output, *, release_dir=None, tag='local', repo='ivere27/unaveraged', source_commit=None,
          current_source=False, root=ROOT):
    if source_commit is not None:
        if not isinstance(source_commit, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', source_commit):
            raise ValueError('source_commit must be a full 40-character hexadecimal Git commit')
        source_commit = source_commit.lower()
    source_ref = source_commit or 'main'
    output = Path(output).resolve()
    # Only the destination produced by this command is replaced, after every
    # package is verified successfully in a sibling temporary directory.
    protected = [(root/source).resolve() for source in ('docs', 'example', 'shared', 'digit', 'vqa', 'synth')]
    if any(source.is_relative_to(output) or output.is_relative_to(source) for source in protected):
        raise ValueError('Choose a dedicated output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}
    if release_dir is not None:
        for line in (release_dir/'SHA256SUMS').read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            name = name.lstrip('*')
            if not re.fullmatch(r'[0-9a-f]{64}', expected) or name in checksums:
                raise ValueError('Invalid Release checksums')
            checksums[name] = expected
    with tempfile.TemporaryDirectory(prefix='.pages-', dir=output.parent) as temporary:
        destination = Path(temporary)/'site'
        docs.write(destination, root=root, repo=repo, source_ref=source_ref)
        demo = destination/'example'
        demo.mkdir()
        for name in EXAMPLE_FILES:
            shutil.copyfile(root/'example'/name, demo/name)
        shutil.copyfile(root/'shared/extension/heatmap-view.js', demo/'heatmap-view.js')
        stage_samples(root, demo, repo=repo, source_ref=source_ref)
        models = {}
        for project in MODELS:
            config = json.loads((root/project/'extension.json').read_text())
            name = f"{config['slug']}-chrome.zip"
            archive = (release_dir/name if release_dir is not None else
                       root/'dist'/project/config['default_precision']/name)
            if archive.resolve().is_relative_to(output):
                raise ValueError('Output directory would replace an input archive')
            if release_dir is not None and name not in checksums:
                raise ValueError(f'Missing Release checksum: {name}')
            options = {'source_root': root} if current_source else {}
            models[project] = stage_model(archive, demo/'runtime'/project, project, checksums.get(name), **options)
        release_tag = tag if release_dir is not None else 'local'
        manifest = {'format': 'unaveraged-demo/v1',
                    'release': {'tag': release_tag, 'url': f'https://github.com/{repo}/releases/tag/{quote(release_tag, safe="")}'
                                if release_tag != 'local' else ''},
                    'models': models}
        if source_commit is not None:
            manifest['source'] = {'repository': repo, 'commit': source_commit,
                                  'url': f'https://github.com/{repo}/commit/{source_commit}'}
        (demo/'release.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        if output.exists():
            shutil.rmtree(output)
        destination.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'dist/pages')
    parser.add_argument('--release-dir', type=Path)
    parser.add_argument('--tag', default='local')
    parser.add_argument('--repo', default='ivere27/unaveraged')
    parser.add_argument('--source-commit', help='Full Git commit of the authored site sources')
    parser.add_argument('--current-source', action='store_true',
                        help='Use current adapters and shared JavaScript with compatible uploaded models/runtime')
    args = parser.parse_args()
    manifest = stage(args.out, release_dir=args.release_dir, tag=args.tag, repo=args.repo,
                     source_commit=args.source_commit, current_source=args.current_source)
    print(f'Pages staged at {args.out}: ' + ', '.join(manifest['models']))


if __name__ == '__main__':
    main()
