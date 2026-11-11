"""Import the exact evaluated W4 PTQ model without recalibration or private data."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('encoder', 'decoder')
CROSS = [f'cross_{kind}_{index}' for index in range(4) for kind in ('k', 'v')]
PAST = [name.replace('cross_', 'past_') for name in CROSS]


def profiles():
    return json.loads((ROOT/'vqa/frozen-profiles.json').read_text())['profiles']


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_files(directory, hashes):
    for name, expected in hashes.items():
        path = directory/name
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f'Frozen VQA hash mismatch: {name}')


def _manifest(profile, precision):
    """Construct a closed public contract instead of copying experiment metadata."""
    inputs = {
        'encoder': ['image', 'question_ids', 'family_ids'],
        'decoder': ['decoder_input_ids', 'position_ids', 'family_ids',
                    'memory_padding_mask', 'past_padding_mask', *CROSS, *PAST],
    }
    outputs = {
        'encoder': ['memory', 'memory_padding_mask', 'router_logits', 'selected_family_ids', *CROSS],
        'decoder': ['logits', 'present_padding_mask',
                    *[name.replace('past_', 'present_') for name in PAST]],
    }
    return {
        'format': 'receipt-vqa-volvoxai/v1', 'variant': 'ptq',
        'graphs': {role: {
            'inputs': {name: f'input{index}' for index, name in enumerate(inputs[role])},
            'outputs': outputs[role],
            'weights': [f'{role}/model.safetensors', 'shared/token-embedding.safetensors'],
        } for role in ROLES},
        'tokenizer': profile['tokenizer'],
        'family_names': ['phone', 'address', 'store', 'item_row', 'item_math', 'item_lookup', 'math', 'other'],
        'question_capacity': 192, 'max_decoder_steps': 191, 'initial_masked_cache_rows': 1,
        'converter': {'package': 'volvoxai', 'version': '0.8.0', 'source': 'frozen-candidate-import'},
        'source_sha256': profile['source_sha256'],
        'quantization': {
            'method': 'volvoxai-ptq', 'source_variant': 'fp32', 'package': 'volvoxai', 'version': '0.8.0',
            'precision': precision, 'weight_bits': 4, 'weight_dtype': 'int4',
            'weight_storage': 'packed-int32', 'activation_dtype': 'int8',
            'activation_scheme': 'dynamic-per-row', 'packed_arithmetic': 'dynamic-i8',
            'group_size': 32, 'algorithm': 'mse-clip', 'convolution_weight_bits': 8,
            'convolution_weight_dtype': 'int8', 'convolution_activation_scheme': 'static-asymmetric',
            'batch_size': 1,
            'scope': 'All 67 dense layers use W4/G32 with dynamic INT8; 13 convolutions use static INT8; other operations retain FP32.',
            'roles': {
                'encoder': {'packed_linear_nodes': 34, 'quantized_convolutions': 13,
                            'quantized_nodes': 47, 'samples': profile['calibration_samples'], 'coverage_complete': True},
                'decoder': {'packed_linear_nodes': 33, 'quantized_convolutions': 0,
                            'quantized_nodes': 33, 'samples': 0, 'coverage_complete': True},
            },
            'calibration': {'samples': profile['calibration_samples'], 'coverage_complete': True,
                            'evaluation_checked': True, 'evaluation_overlap': 0,
                            'provenance': f"Original frozen experiment used {profile['calibration_samples']} disjoint calibration records; import performs no recalibration."},
        },
        'frozen_candidate': profile,
        'qualification': {'status': 'candidate', 'independent_accuracy_verified': False,
                          'scope': 'Existing 2000-record phone-digit/street-number evaluation; not general VQA accuracy.'},
    }


def _validate_graphs(package, manifest):
    for role, dense_count, conv_count in (('encoder', 34, 13), ('decoder', 33, 0)):
        graph = json.loads((package/role/'graph.json').read_text())
        contract = manifest['graphs'][role]
        if list(graph.get('inputs', {})) != list(contract['inputs'].values()) or graph.get('outputs') != contract['outputs']:
            raise ValueError(f'Frozen VQA {role} input/output contract changed')
        dense = [node for node in graph['nodes'] if node['opType'] == 'PackedLinear']
        conv = [node for node in graph['nodes'] if node['opType'] == 'QConv2D']
        if len(dense) != dense_count or len(conv) != conv_count or any(
                node['opType'] in ('Linear', 'Conv2D', 'QLinear', 'QGemm', 'QMatMul') for node in graph['nodes']):
            raise ValueError(f'Frozen VQA {role} operator contract changed')
        if any((node['params'].get('weight_bits'), node['params'].get('group_size')) != (4, 32) for node in dense):
            raise ValueError(f'Frozen VQA {role} dense precision changed')
        if any(node['params'].get('weight_bits', 8) != 8 for node in conv):
            raise ValueError('Frozen VQA convolution precision changed')


def validate_package(package, precision='w4a8'):
    """Verify cached selected artifacts and the complete sanitized manifest."""
    package = Path(package)
    profile = profiles()[precision]
    _verify_files(package, profile['sha256'])
    manifest = json.loads((package/'manifest.json').read_text())
    if manifest != _manifest(profile, precision):
        raise ValueError('Frozen VQA manifest differs from the public candidate contract')
    _validate_graphs(package, manifest)
    return manifest


def import_model(source, destination, release, precision='w4a8'):
    """Copy only pinned deployable files, rebuilding a privacy-safe PTQ manifest."""
    profile = profiles()[precision]
    configured = json.loads((ROOT/'vqa/extension.json').read_text())['model']
    if (configured['repository'], configured['revision']) != (profile['model_repository'], profile['model_revision']):
        raise ValueError('Frozen VQA candidate belongs to a different source-model revision')
    if source is None:
        raise ValueError(f'{precision} frozen import requires --model-source=/path/to/frozen/model')
    source, destination, release = Path(source).resolve(), Path(destination), Path(release)
    _verify_files(source, profile['sha256'])
    _verify_files(release, profile['source_sha256'])
    source_manifest = source/'manifest.json'
    if not source_manifest.is_file():
        raise ValueError('Frozen VQA hash mismatch: manifest.json')
    if _digest(source_manifest) == profile['source_manifest_sha256']:
        original = json.loads(source_manifest.read_text())
        if any(original.get('source_sha256', {}).get(name) != expected
               for name, expected in profile['source_sha256'].items() if name != 'manifest.json'):
            raise ValueError('Frozen VQA source-model provenance mismatch')
    else:
        # A previously sanitized cache is also an exact identity, never a generic
        # candidate with merely similar precision settings.
        validate_package(source, precision)
    manifest = _manifest(profile, precision)
    _validate_graphs(source, manifest)
    destination.mkdir(parents=True)
    for name in profile['sha256']:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return validate_package(destination, precision)
