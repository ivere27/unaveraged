"""Identify the public-build baselines and immutable low-bit comparison models."""
from collections import Counter
import json
from pathlib import Path

from digit.frozen import profiles as frozen_profiles, validate_graph
from shared.model_support import sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ('f32', 'w8a8', 'conv4-linear2')
# Immutable artifact provenance, independent of the runtime selected in package.json.
MODEL_EXPORT_VERSION = '0.8.0'


def validate_baseline_graph(graph, name):
    if graph.get('inputs') != {'input0': {'shape': [1, 1, 320, 672], 'dtype': 'float32'}} or graph.get('outputs') != ['slot_logits']:
        raise ValueError('Baseline input/output contract changed')
    counts = Counter(n['opType'] for n in graph['nodes'])
    if counts['Linear'] != 7 or counts['PackedLinear']:
        raise ValueError('F32 and W8A8 baselines require seven FP32 Linear layers')
    if name == 'f32':
        if counts['Conv2D'] != 11 or any(counts[op] for op in ('QConv2D', 'QuantizeLinear', 'DequantizeLinear')):
            raise ValueError('F32 baseline contains quantized operators')
    elif name == 'w8a8':
        conv = [n for n in graph['nodes'] if n['opType'] == 'QConv2D']
        if len(conv) != 11 or counts['Conv2D'] or any(n['params'].get('weight_bits', 8) != 8 for n in conv):
            raise ValueError('W8A8 baseline requires eleven INT8 convolutions')
    else:
        raise ValueError('Unknown baseline: ' + name)


def model_definitions(root=ROOT, model_export_version=MODEL_EXPORT_VERSION):
    """Read frozen cached artifacts without changing their exporter provenance."""
    config = json.loads((root/'digit/extension.json').read_text())['model']
    source = root/'.cache/digit/huggingface'/config['revision']
    packages = root/'.cache/digit/packages'/model_export_version
    paths = {'f32': packages/'fp32-prepared', 'w8a8': packages/'ptq/w8a8',
             **{name: packages/'ptq'/name for name in frozen_profiles()}}
    definitions = {}
    for name in MODEL_ORDER:
        directory = paths[name]
        graph = json.loads((directory/'graph.json').read_text())
        if name in ('f32', 'w8a8'):
            validate_baseline_graph(graph, name)
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['converter']['version'] != model_export_version or not manifest['converter'].get('prepare_for_ptq'):
                raise ValueError('Baseline must use the public FP32 PTQ-preparation build')
            expected_source = {n: sha256((source/n).read_bytes()) for n in ('config.json', 'manifest.json', 'model.onnx')}
            if manifest['source']['sha256'] != expected_source:
                raise ValueError('Baseline source differs from the pinned FP32 release')
            quant = manifest.get('quantization', {})
            if name == 'w8a8' and (quant.get('precision') != name or quant.get('packed_arithmetic') != 'f32'):
                raise ValueError('W8A8 baseline manifest precision/arithmetic changed')
            if name == 'f32' and (manifest['variant'] != 'fp32' or quant):
                raise ValueError('F32 baseline manifest contains quantization')
            definitions[name] = {
                'role': 'floating-reference' if name == 'f32' else 'w8a8-reference',
                'description': 'All FP32; SiLU prepared identically to PTQ input' if name == 'f32' else '11 static W8A8 convolutions; seven FP32 Linear layers',
                'conv_weight_bits': 32 if name == 'f32' else 8,
                'linear_weight_bits': 32, 'packed_arithmetic': 'f32',
                'calibration_samples': quant.get('calibration', {}).get('samples', 0),
                'model_repository': config['repository'], 'model_revision': config['revision'],
                'source_exporter_version': manifest['converter']['version'],
                'sha256': {n: sha256((directory/n).read_bytes()) for n in ('graph.json', 'model.safetensors')},
                'graph_weight_bytes': sum((directory/n).stat().st_size for n in ('graph.json', 'model.safetensors')),
                'source_preparation': 'FP32 export with --allow-silu-numerical-migration; same graph preparation for both baselines',
            }
        else:
            profile = frozen_profiles()[name]
            validate_graph(graph, profile)
            manifest = json.loads((directory/'manifest.json').read_text())
            definitions[name] = {**profile, 'source_exporter_version': manifest['converter']['version']}
        for file, expected in definitions[name]['sha256'].items():
            if sha256((directory/file).read_bytes()) != expected:
                raise ValueError('Benchmark model hash mismatch: ' + name)
    return definitions, paths
