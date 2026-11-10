"""Rebuild Digit QAT from the pinned FP32 release and public synthetic receipts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRECISION = 'conv4-linear2'
FILES = ('graph.json', 'model.safetensors')
SOURCE_FILES = ('config.json', 'manifest.json', 'model.onnx')
RECIPE = {'epochs': 2, 'samples_per_epoch': 512, 'batch_size': 4,
          'learning_rate': 3e-5, 'seed': 20261025}
QUANTIZATION = {'method': 'volvoxai-qat', 'source_variant': 'fp32',
    'precision': PRECISION, 'activation_dtype': 'int8', 'packed_arithmetic': 'dynamic-i8',
    'conv_weight_bits': 4, 'linear_weight_bits': 2, 'group_size': 32,
    'conv_algorithm': 'mse-clip', 'linear_algorithm': 'mse-clip',
    'quantized_convolutions': 11, 'packed_linear_nodes': 7, 'quantized_nodes': 18}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(directory, names=FILES):
    return {name: digest(Path(directory)/name) for name in names}


def _hashes(value, names):
    return isinstance(value, dict) and set(value) == set(names) and all(
        isinstance(item, str) and re.fullmatch('[0-9a-f]{64}', item) for item in value.values())


def _dataset_records(synth_root):
    manifest = read(synth_root/'manifest.json')
    rows = manifest.get('records', [])
    _require(manifest.get('count') == 256 and len(rows) == 256,
             'Digit QAT requires the complete 256-receipt public dataset')
    for index, row in enumerate(rows):
        expected = f'{index:06d}'
        _require((row.get('id'), row.get('image'), row.get('annotation')) ==
                 (expected, f'images/{expected}.png', f'annotations/{expected}.json'),
                 'Synthetic dataset order or paths changed')
    return rows


def prepare_manifests(synth_root, destination):
    """Verify public image/label bytes and audit the fixed 224/32 split."""
    from shared.model_support import preprocess
    synth_root, destination = Path(synth_root).resolve(), Path(destination)
    rows = _dataset_records(synth_root)
    items = []
    for row in rows:
        image, annotation = synth_root/row['image'], synth_root/row['annotation']
        _require(digest(image) == row['sha256'], 'Synthetic image hash mismatch: ' + row['image'])
        _require(digest(annotation) == row['annotation_sha256'],
                 'Synthetic annotation hash mismatch: ' + row['annotation'])
        answers = read(annotation)['answers']
        targets = []
        for value, width in ((answers['phone_digits'], 12), (answers['street_no'], 4)):
            digits = str(value)
            _require(re.fullmatch('[0-9]+', digits) is not None and len(digits) <= width,
                     'Synthetic numeric label exceeds the Digit slot contract')
            targets.extend([int(digit) for digit in digits] + [10] * (width-len(digits)))
        items.append({'path': str(image), 'file_sha256': row['sha256'],
                      'input_sha256': hashlib.sha256(preprocess(image).tobytes()).hexdigest(),
                      'targets': targets, 'source': 'synthetic'})
    _require(len({item['input_sha256'] for item in items}) == 256,
             'Public synthetic train/validation inputs overlap or repeat')
    destination.mkdir(parents=True, exist_ok=True)
    splits = {'train': items[:224], 'calibration': items[:224], 'validation': items[224:]}
    for name, selected in splits.items():
        write(destination/f'{name}.json', {'items': selected})
    portable = [{'path': 'synth/' + row['image'], 'file_sha256': row['sha256'],
                 'annotation_sha256': row['annotation_sha256'],
                 'input_sha256': item['input_sha256']} for row, item in zip(rows, items)]
    return {'format': 'unaveraged-public-synth-calibration/v1', 'count': 224,
            'images': portable[:224], 'validation_images': portable[224:],
            'validation_input_sha256': [item['input_sha256'] for item in items[224:]]}


def validate_manifest(manifest, source_hashes, *, root=None):
    """Validate fresh model metadata without importing native ML dependencies."""
    root = ROOT if root is None else Path(root)
    version = read(root/'package.json')['config']['volvoxai_version']
    model = read(root/'digit/extension.json')['model']
    _require(manifest.get('format') == 'volvoxai-receipt-digit-reader-onnx-package-v1'
             and manifest.get('source_format') == 'receipt_digit_reader_onnx_v1'
             and manifest.get('variant') == 'ptq' and 'frozen_candidate' not in manifest,
             'Invalid fresh Digit QAT package format')
    _require(_hashes(source_hashes, SOURCE_FILES)
             and manifest.get('source', {}).get('sha256') == source_hashes,
             'Digit QAT source release mismatch')
    converter = manifest.get('converter', {})
    _require(converter.get('package') == 'volvoxai' and converter.get('version') == version
             and converter.get('prepare_for_ptq') is True,
             'Digit QAT converter runtime mismatch')
    provenance = manifest.get('source_build', {})
    _require(provenance.get('format') == 'unaveraged-public-synth-qat/v1'
             and provenance.get('model_repository') == model['repository']
             and provenance.get('model_revision') == model['revision']
             and provenance.get('dataset_manifest_sha256') == digest(root/'synth/manifest.json')
             and _hashes(provenance.get('prepared_sha256'), FILES),
             'Digit QAT public source provenance mismatch')
    _require(_hashes(manifest.get('model_sha256'), FILES), 'Missing Digit QAT model identity')
    quantization = manifest.get('quantization', {})
    _require(all(quantization.get(key) == value for key, value in QUANTIZATION.items())
             and quantization.get('version') == version, 'Digit QAT precision or runtime mismatch')
    training = quantization.get('training', {})
    _require(all(training.get(key) == value for key, value in RECIPE.items())
             and training.get('method') == 'joint-weight-QAT-synth'
             and training.get('backend') in ('cpu', 'cuda')
             and training.get('calibration_backend') == 'cpu'
             and training.get('activation_qat') is False
             and training.get('training_samples') == 224
             and training.get('validation_samples') == 32
             and training.get('calibration_overlaps_training') is True
             and training.get('selected_epoch') in (1, 2), 'Digit QAT training recipe mismatch')
    accuracy, entropy = training.get('validation_record_accuracy'), training.get('validation_cross_entropy')
    _require(isinstance(accuracy, (int, float)) and math.isfinite(accuracy) and 0 <= accuracy <= 1
             and isinstance(entropy, (int, float)) and math.isfinite(entropy) and entropy >= 0,
             'Invalid Digit QAT validation metrics')
    calibration = quantization.get('calibration', {})
    for name, metadata in (('training', training), ('calibration', calibration)):
        if 'cpu_threads' in metadata:
            _require(type(metadata['cpu_threads']) is int and metadata['cpu_threads'] > 0,
                     f'Invalid Digit QAT {name} CPU thread count')
    _require(calibration.get('samples') == 224 and calibration.get('coverage_complete') is True
             and calibration.get('validation_overlap') == 0
             and calibration.get('evaluation_checked') is False
             and calibration.get('evaluation_overlap') is None
             and _hashes({'audit': calibration.get('audit_sha256')}, ('audit',)),
             'Digit QAT calibration provenance mismatch')
    _require(manifest.get('abi', {}).get('input') == {'name': 'input0', 'shape': [1, 1, 320, 672]}
             and manifest.get('abi', {}).get('output') == 'slot_logits'
             and manifest.get('decode') == {'slots': 16, 'phone_slots': 12, 'street_slots': 4,
                                          'num_classes': 11, 'blank_class': 10},
             'Digit QAT input/output contract mismatch')
    return manifest


def validate_package(package, release, precision=PRECISION):
    from digit.frozen import validate_graph
    package, release = Path(package), Path(release)
    _require(precision == PRECISION, 'Unsupported fresh Digit QAT precision')
    manifest = validate_manifest(read(package/'manifest.json'), _identity(release, SOURCE_FILES))
    _require(_identity(package) == manifest['model_sha256'], 'Digit QAT model hash mismatch')
    _require(manifest['preprocess'] == read(release/'config.json')['input'],
             'Digit QAT preprocessing contract mismatch')
    graph = read(package/'graph.json')
    validate_graph(graph,
                   {'conv_weight_bits': 4, 'linear_weight_bits': 2, 'linear_group_size': 32})
    outputs = [value for node in graph['nodes'] for value in node.get('outputs', {}).values()
               if value.get('tensor') == 'slot_logits']
    _require(len(outputs) == 1 and outputs[0].get('shape') == [1, 16, 11]
             and outputs[0].get('dtype') == 'float32', 'Digit QAT slot output contract mismatch')
    audit_path = package/'calibration-inputs.json'
    _require(digest(audit_path) == manifest['quantization']['calibration']['audit_sha256'],
             'Digit QAT calibration audit hash mismatch')
    audit = read(audit_path)
    images, validation = audit.get('images', []), audit.get('validation_images', [])
    _require(audit.get('format') == 'unaveraged-public-synth-calibration/v1'
             and audit.get('count') == 224 and len(images) == 224 and len(validation) == 32,
             'Digit QAT calibration audit count mismatch')
    hashes = []
    for item, row in zip([*images, *validation], _dataset_records(ROOT/'synth')):
        _require(item.get('path') == 'synth/' + row['image']
                 and item.get('file_sha256') == row['sha256']
                 and item.get('annotation_sha256') == row['annotation_sha256']
                 and _hashes({'input': item.get('input_sha256')}, ('input',)),
                 'Digit QAT calibration audit dataset mismatch')
        hashes.append(item['input_sha256'])
    _require(len(set(hashes)) == 256 and audit.get('validation_input_sha256') == hashes[224:],
             'Digit QAT calibration/validation overlap')
    return manifest


def rebuild_model(source, destination, prepared, *, synth_root, backend='cpu', cpu_threads=None):
    """Fit and export the public recipe; GPU training is optional."""
    from shared.cpu_policy import resolve_cpu_threads
    from shared.model_support import VERSION, packaged_library_paths
    cpu_threads = resolve_cpu_threads(cpu_threads)
    source, destination, prepared = Path(source), Path(destination), Path(prepared)
    synth_root = Path(synth_root)
    _require(backend in ('cpu', 'cuda'), 'Digit QAT backend must be cpu or cuda')
    _require(not destination.exists(), 'Digit QAT output already exists')
    _require(digest(synth_root/'manifest.json') == digest(ROOT/'synth/manifest.json'),
             'Digit QAT requires the repository synthetic dataset')
    packaged_library_paths()
    base = read(prepared/'manifest.json')
    _require(base.get('variant') == 'fp32' and base.get('converter', {}).get('version') == VERSION
             and base.get('converter', {}).get('prepare_for_ptq') is True
             and base.get('source', {}).get('sha256') == _identity(source, SOURCE_FILES),
             'Digit QAT requires a prepared conversion of the pinned FP32 release')
    prepared_hashes = _identity(prepared)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.digit-qat-', dir=destination.parent) as temporary:
        work = Path(temporary).resolve()
        audit = prepare_manifests(synth_root, work/'inputs')
        command = [sys.executable, str(ROOT/'digit/qat.py'), '--model', str(prepared.resolve()),
                   '--out', str(work/'training'), '--conv-bits', '4', '--linear-bits', '2',
                   '--backend', backend, '--calibration-backend', 'cpu',
                   '--cpu-threads', str(cpu_threads), '--allow-calibration-overlap']
        for name in ('train', 'validation', 'calibration'):
            command += ['--'+name, str(work/'inputs'/f'{name}.json')]
        for key, value in RECIPE.items():
            command += ['--'+key.replace('_', '-'), str(value)]
        subprocess.run(command, check=True, cwd=ROOT)
        report = read(work/'training/report.json')
        _require(report.get('status') == 'passed' and report.get('source_sha256') == prepared_hashes,
                 'Digit QAT training failed or used a different FP32 model')
        _require(report.get('settings', {}).get('cpu_threads') == cpu_threads,
                 'Digit QAT CPU thread count changed')
        selected = next((row for row in report['epochs'] if row['epoch'] == report['best_epoch']), None)
        _require(selected is not None and selected['sha256'] == _identity(work/'training/best'),
                 'Digit QAT selected checkpoint identity mismatch')
        metrics = selected['validation']['all']
        _require(metrics['samples'] == 32 and all(math.isfinite(metrics[key])
                 for key in ('record_accuracy', 'cross_entropy')), 'Invalid Digit QAT validation metrics')
        stage = work/'package'
        stage.mkdir()
        for name in FILES:
            shutil.copy2(work/'training/best'/name, stage/name)
        write(stage/'calibration-inputs.json', audit)
        configured = read(ROOT/'digit/extension.json')['model']
        manifest = {key: base[key] for key in
                    ('format', 'source_format', 'source', 'converter', 'abi', 'decode', 'preprocess')}
        manifest.update(variant='ptq', model_sha256=_identity(stage), source_build={
            'format': 'unaveraged-public-synth-qat/v1', 'model_repository': configured['repository'],
            'model_revision': configured['revision'], 'dataset_manifest_sha256': digest(synth_root/'manifest.json'),
            'prepared_sha256': prepared_hashes}, quantization={**QUANTIZATION, 'version': VERSION,
            'training': {**RECIPE, 'method': 'joint-weight-QAT-synth', 'backend': backend,
                'calibration_backend': 'cpu', 'cpu_threads': cpu_threads,
                'activation_qat': False, 'training_samples': 224,
                'validation_samples': 32, 'calibration_overlaps_training': True,
                'selected_epoch': report['best_epoch'],
                'validation_record_accuracy': metrics['record_accuracy'],
                'validation_cross_entropy': metrics['cross_entropy']},
            'calibration': {'samples': 224, 'cpu_threads': cpu_threads,
                'coverage_complete': True, 'validation_overlap': 0,
                'evaluation_checked': False, 'evaluation_overlap': None,
                'audit_sha256': digest(stage/'calibration-inputs.json')}},
            qualification={'status': 'fresh-build', 'independent_accuracy_verified': False,
                           'scope': 'Checkpoint selected on 32 public synthetic validation receipts.'})
        write(stage/'manifest.json', manifest)
        validate_package(stage, source)
        stage.replace(destination)
    return manifest
