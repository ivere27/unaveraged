"""Model-independent preprocessing and published VolvoxAI native-library loading."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import tempfile
import numpy as np
from PIL import Image
import volvoxai as vx

VERSION = json.loads((Path(__file__).resolve().parents[1] / 'package.json').read_text())['config']['volvoxai_version']
PRECISIONS = ('w8a8', 'w4a8')
SUPPORTED_PRECISIONS = (*PRECISIONS, 'w4a8-conv4')
PACKED_GROUP_SIZE = 32
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def packaged_library_paths():
    distribution = importlib.metadata.distribution('volvoxai')
    package = Path(vx.__file__).resolve().parent
    if distribution.version != VERSION or package != Path(distribution.locate_file('volvoxai')).resolve():
        raise RuntimeError(f'PTQ requires installed volvoxai=={VERSION}')
    engine = package / vx.library_filename()
    loader = package / vx.library_filename().replace('volvoxai', 'synurang_module_host')
    if not engine.is_file() or not loader.is_file():
        raise RuntimeError('The selected wheel must include its native engine and loader')
    if vx.find_library().resolve() != engine.resolve():
        raise RuntimeError('Native library override differs from the selected wheel; unset VOLVOXAI_LIBRARY/VOLVOXAI_LIBRARY_DIR')
    loader_override = os.environ.get('SYNURANG_MODULE_HOST_LIBRARY')
    if loader_override and Path(loader_override).expanduser().resolve() != loader.resolve():
        raise RuntimeError('Native loader override differs from the selected wheel; unset SYNURANG_MODULE_HOST_LIBRARY')
    return engine, loader

def preprocess(path):
    with Image.open(path) as image:
        values = np.asarray(image.convert('L').resize((672,320), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.ascontiguousarray(((values / 255.0 - .5) / .5)[None,None])

def image_paths(directory):
    if not directory.is_dir(): raise ValueError(f'Image directory does not exist: {directory}')
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

def open_packaged_host():
    engine, loader = packaged_library_paths()
    return vx.open_library(engine, loader=loader)

def package_source():
    distribution = importlib.metadata.distribution('volvoxai')
    return 'local-wheel' if distribution.read_text('direct_url.json') else 'pypi'

def quantization_config(graph, precision, static_operators, *, static_convolution_bits=8):
    """Select stored weights separately from the compilation arithmetic.

    This describes one plan. W4A8 packs dense weights and requires dynamic-i8
    compilation. create_quantization_pipeline follows it with a separate
    calibrated Conv-I8 plan when the model contains convolutions.
    """
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f'Unsupported precision: {precision}')
    if static_convolution_bits not in (4, 8) or (static_convolution_bits == 4 and
            (precision != 'w8a8' or set(static_operators) != {'Conv2D'})):
        raise ValueError('Static INT4 is supported only for the explicit Conv2D stage')
    pb = vx.pb
    packed = precision in ('w4a8', 'w4a8-conv4')
    operators = {'Linear', 'MatMul'} if packed else set(static_operators)
    selected = [node for node in graph['nodes'] if node['opType'] in operators]
    if not selected:
        raise ValueError(f'No eligible operators for {precision}')
    # Explicitly retain every other operator, preserving the conservative
    # model recipes instead of applying new weightless static conversions.
    rules = [pb.QuantizationRule(operator_name=name, keep_float=pb.Empty())
             for name in sorted({node['opType'] for node in graph['nodes']} - operators)]
    config = pb.QuantizationConfig(
        weights=pb.WeightQuantization(format=pb.WeightFormat(
            bits=4 if packed else static_convolution_bits, group_size=PACKED_GROUP_SIZE if packed else 0),
            algorithm=pb.WeightAlgorithm.WEIGHT_ALGORITHM_MSE_CLIP if packed or static_convolution_bits == 4
                      else pb.WeightAlgorithm.WEIGHT_ALGORITHM_RTN),
        rules=rules)
    if not packed:
        config.static_activations.CopyFrom(pb.StaticActivations(
            dtype=pb.DataType.DATA_TYPE_I8,
            scheme=pb.AffineScheme.AFFINE_SCHEME_ASYMMETRIC))
    return config, selected

def create_quantization_plan(client, source, precision, static_operators, *, static_convolution_bits=8,
                             cpu_threads=1):
    graph = json.loads((source / 'graph.json').read_text())
    config, selected = quantization_config(graph, precision, static_operators,
        static_convolution_bits=static_convolution_bits)
    plan = client.create_quantization_plan(vx.pb.CreateQuantizationPlanRequest(
        files=vx.pb.ModelFiles(graph_path=str(source / 'graph.json'),
            weight_paths=[str(source / 'model.safetensors')]),
        config=config, calibration=vx.pb.CalibrationOptions(cpu_threads=cpu_threads)))
    if sum(item.nodes for item in plan.info.quantized) != len(selected):
        client.release_quantization_plan(vx.pb.QuantizationPlanRef(
            quantization_plan_id=plan.quantization_plan_id))
        raise ValueError('Quantization plan did not select every requested weight node')
    return plan

def export_quantization_plan(client, plan, destination, samples):
    ref = vx.pb.QuantizationPlanRef(quantization_plan_id=plan.quantization_plan_id)
    info = client.inspect_quantization_plan(ref).info
    if not info.ready or info.calibration.samples != samples:
        raise ValueError('Quantization calibration coverage is incomplete')
    if info.requires_calibration and not info.calibration.coverage.complete:
        raise ValueError('Quantization calibration coverage is incomplete')
    packed = client.export_quantized_model(vx.pb.ExportQuantizedModelRequest(
        quantization_plan_id=plan.quantization_plan_id,
        output_files=vx.pb.ModelFiles(graph_path=str(destination / 'graph.json'),
            weight_paths=[str(destination / 'model.safetensors')])))
    return json.loads((destination / 'graph.json').read_text())

def create_quantization_pipeline(client, source, precision, static_operators, *, cpu_threads=1):
    """Prepare a final plan, count and stage audit using only public PTQ calls.

    The 0.8 API has one activation policy per plan. Pack dense matrices first,
    then calibrate only Conv2D while retaining those packed tensors exactly.
    Native plan creation snapshots the intermediate files, so they can be
    removed immediately. A packed dense ancestor of Conv is refused: private
    calibration uses F32 packed arithmetic and would then observe a different
    convolution input from the final dynamic-I8 computation.
    """
    graph = json.loads((source / 'graph.json').read_text())
    convolutions = [node for node in graph['nodes'] if node['opType'] == 'Conv2D']
    mixed = precision in ('w4a8', 'w4a8-conv4') and 'Conv2D' in static_operators and convolutions
    if mixed:
        packed_dependencies = set()
        for node in graph['nodes']:
            depends_on_packed = any(name in packed_dependencies for name in node.get('inputs', {}).values())
            if node['opType'] == 'Conv2D' and depends_on_packed:
                raise ValueError('Mixed calibration requires convolutions independent of packed dense outputs')
            if depends_on_packed or node['opType'] in ('Linear', 'MatMul'):
                packed_dependencies.update(value['tensor'] for value in node.get('outputs', {}).values())
    plan = create_quantization_plan(client, source, precision, static_operators, cpu_threads=cpu_threads)
    count = sum(item.nodes for item in plan.info.quantized)
    stages = [{'policy': 'packed-linear-i4-group32-mse' if precision in ('w4a8', 'w4a8-conv4') else 'static-i8-rtn',
               'quantized_nodes': count, 'requires_calibration': plan.info.requires_calibration}]
    if not mixed:
        return plan, count, stages
    try:
        with tempfile.TemporaryDirectory(prefix='receipt-packed-source-') as temporary:
            packed_source = Path(temporary)
            export_quantization_plan(client, plan, packed_source, 0)
            conv_plan = create_quantization_plan(client, packed_source, 'w8a8', ('Conv2D',),
                static_convolution_bits=4 if precision == 'w4a8-conv4' else 8, cpu_threads=cpu_threads)
    finally:
        client.release_quantization_plan(vx.pb.QuantizationPlanRef(
            quantization_plan_id=plan.quantization_plan_id))
    conv_count = sum(item.nodes for item in conv_plan.info.quantized)
    stages.append({'policy': 'static-conv-i4-per-channel-mse' if precision == 'w4a8-conv4' else 'static-conv-i8-rtn', 'quantized_nodes': conv_count,
                   'requires_calibration': conv_plan.info.requires_calibration})
    return conv_plan, count + conv_count, stages

def validate_quantized_weights(directory, graph, precision, expected_nodes):
    from safetensors import safe_open
    packed = precision in ('w4a8', 'w4a8-conv4')
    names = ('PackedLinear', 'QConv2D') if packed else ('QConv2D', 'QLinear', 'QGemm', 'QMatMul')
    nodes = [node for node in graph['nodes'] if node['opType'] in names]
    if len(nodes) != expected_nodes:
        raise ValueError('Exported quantized weight node count changed')
    with safe_open(directory / 'model.safetensors', framework='numpy') as weights:
        for node in nodes:
            packed_node = node['opType'] == 'PackedLinear'
            conv4 = node['opType'] == 'QConv2D' and precision == 'w4a8-conv4'
            stored = weights.get_slice(node['inputs']['weight'])
            if stored.get_dtype() != ('I32' if packed_node else 'U8' if conv4 else 'I8'):
                raise ValueError('Quantized weight storage does not match the requested precision')
            if node['opType'] == 'QConv2D':
                if node['params'].get('weight_bits', 8) != (4 if conv4 else 8):
                    raise ValueError('Convolution weight bits do not match the requested precision')
                if conv4:
                    affine = graph['quantization']['tensors'][node['inputs']['weight']]
                    if len(stored.get_shape()) != 2 or len(node['params'].get('kernel_size', [])) != 2 \
                            or affine['scheme'] != 'per_axis' or affine['axis'] != 0 \
                            or weights.get_slice(affine['zero_point_tensor']).get_dtype() != 'U8' \
                            or np.any(weights.get_tensor(affine['zero_point_tensor']) != 8):
                        raise ValueError('Packed Conv4 storage or per-channel affine is invalid')
            if packed_node:
                if node['params'].get('weight_bits') != 4 or node['params'].get('group_size') != PACKED_GROUP_SIZE:
                    raise ValueError('PackedLinear format does not match W4A8')
                if weights.get_slice(node['inputs']['scale']).get_dtype() != 'F32':
                    raise ValueError('PackedLinear scales must be F32')
    return nodes

def precision_metadata(precision):
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f'Unsupported precision: {precision}')
    packed = precision in ('w4a8', 'w4a8-conv4')
    conv4 = precision == 'w4a8-conv4'
    return {'precision': precision, 'activation_dtype': 'int8',
        'weight_dtype': 'int4' if packed else 'int8',
        'weight_storage': 'linear-packed-int32+conv-packed-uint8' if conv4 else 'packed-int32' if packed else 'int8',
        'group_size': PACKED_GROUP_SIZE if packed else 0,
        'algorithm': 'mse-clip' if packed else 'rtn',
        'activation_scheme': 'dynamic-per-row' if packed else 'asymmetric',
        'packed_arithmetic': 'dynamic-i8' if packed else 'f32',
        'convolution_weight_dtype': 'int4' if conv4 else 'int8',
        'convolution_activation_scheme': 'static-asymmetric',
        'scope': 'Dense layers use INT4/group32 with dynamic-I8; selected convolutions use packed INT4/per-output-channel MSE with static INT8 activations' if conv4 else
                 'Dense layers use INT4/group32 with dynamic-I8; selected convolutions use static INT8' if packed
                 else 'Selected convolutions/dense layers use static W8A8; other operators remain FP32'}
