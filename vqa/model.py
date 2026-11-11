"""Pinned HF FP32 -> VolvoxAI 0.8.0 exporter -> native W8A8/W4A8.

No local VolvoxAI checkout, pretrained INT8 ONNX, or private exporter imports.
The encoder uses the source model's masked PAD token to fix Q=192. The decoder
keeps a variable cache, starting with one masked zero row (positive dimensions).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata

import numpy as np
import onnx
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
import volvoxai as vx
from shared.model_support import (VERSION, sha256, write_json, preprocess,
    image_paths, packaged_library_paths, open_packaged_host, package_source,
    create_quantization_pipeline, export_quantization_plan, validate_quantized_weights,
    precision_metadata)

ROLES = ('encoder', 'decoder')
FAMILIES = ['phone', 'address', 'store', 'item_row', 'item_math', 'item_lookup', 'math', 'other']
QUESTION_CAPACITY = 192
MAX_STEPS = 191  # Source max_out_len=192 includes BOS.
CROSS = [f'cross_{kind}_{layer}' for layer in range(4) for kind in ('k', 'v')]
PAST = [name.replace('cross_', 'past_') for name in CROSS]
PRESENT = [name.replace('cross_', 'present_') for name in CROSS]
pb = vx.pb


def clean_text(value):
    return unicodedata.normalize('NFC', re.sub(r'\s+', ' ', str(value)).strip())


def write_graph(path, graph):
    # Graphs are shipped model data; avoid increasing storage through indentation.
    path.write_text(json.dumps(graph, ensure_ascii=False, separators=(',', ':')) + '\n')


def load_tokenizer(source):
    # byte_bpe.py is downloaded from the pinned HF commit and checksum-verified.
    spec = importlib.util.spec_from_file_location('receipt_hf_byte_bpe', source / 'byte_bpe.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_bpe1536_vocab(json.loads((source / 'vocab.json').read_text()))


def prepare_onnx(source, destination, role):
    model = onnx.load(source / f'{role}_model.onnx')
    if role == 'decoder':
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    fixed = {'batch': 1, 'question_length': 192, 'question_length + 210': 402, 'memory_length': 402}
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        dimensions = value.type.tensor_type.shape.dim
        for axis, dimension in enumerate(dimensions):
            symbol = dimension.dim_param
            if symbol in fixed:
                dimension.dim_value = fixed[symbol]
            elif symbol in ('past_length', 'present_length'):
                dimension.dim_param = 'P' if symbol == 'past_length' else 'R'
            elif role == 'decoder' and symbol.startswith('unk__'):
                if axis == 0 and len(dimensions) >= 2:
                    dimension.dim_value = 1
                elif axis == 3 and len(dimensions) == 4:
                    dimension.dim_value = 40
                elif axis == 2 and value.name.startswith('present_v_'):
                    dimension.dim_param = 'R'
    for value in model.graph.output:
        if role == 'encoder' and value.name in CROSS:
            for dimension, extent in zip(value.type.tensor_type.shape.dim, [1, 8, 402, 40]):
                dimension.dim_value = extent
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return model


def deduplicate_weights(directory):
    """Compact immutable pointwise constants without changing values or outputs.

    A uniform F32 Mul operand becomes one scalar only when its other, nonweight
    operand already has the exact declared output shape. Constants that supply
    a broadcast extent therefore keep their shape. Identical Mul/Where weights
    can then share one dtype/shape/byte-identical tensor. Public, bank and affine
    metadata references are excluded from both transformations.
    """
    graph_path, weights_path = directory/'graph.json', directory/'model.safetensors'
    graph = json.loads(graph_path.read_text())
    weights = load_file(weights_path)
    with safe_open(weights_path, framework='numpy') as reader:
        metadata = reader.metadata()
    source_bytes = graph_path.stat().st_size + weights_path.stat().st_size
    protected = set(graph['inputs']) | set(graph['outputs'])
    protected.update(graph.get('banks', {}))
    protected.update(graph.get('banks', {}).values())
    for name, affine in graph.get('quantization', {}).get('tensors', {}).items():
        protected.add(name)
        protected.update(affine[key] for key in ('scale_tensor', 'zero_point_tensor') if key in affine)
    declarations = dict(graph['inputs'])
    consumers = {}
    for node in graph['nodes']:
        declarations.update({output['tensor']:output for output in node.get('outputs', {}).values()})
        for port, name in node['inputs'].items():
            consumers.setdefault(name, []).append((node, port))
    scalars = {}
    for name, value in list(weights.items()):
        users = consumers.get(name, [])
        if name in protected or value.dtype != np.dtype('float32') or value.size <= 1 or not users:
            continue
        first = value.reshape(-1)[:1].copy()
        if not np.isfinite(first).all() or value.tobytes() != first.tobytes() * value.size:
            continue
        for node, port in users:
            if node['opType'] != 'Mul' or len(node['inputs']) != 2 or len(node.get('outputs', {})) != 1:
                break
            other = next(item for key, item in node['inputs'].items() if key != port)
            declared = declarations.get(other, {})
            output = next(iter(node['outputs'].values()))
            if other in weights or declared.get('dtype') != 'float32' or not declared.get('shape') \
                    or declared['shape'] != output.get('shape'):
                break
        else:
            scalars[name] = {'original_shape':list(value.shape), 'stored_shape':[1],
                'original_bytes':value.nbytes, 'stored_bytes':first.nbytes,
                'value_bytes_sha256':sha256(first.tobytes()),
                'shape_proof':'Every other nonweight Mul operand has the declared output shape',
                'nodes':[node['id'] for node, _ in users]}
            weights[name] = first
    canonical, duplicates = {}, {}
    for name, value in weights.items():
        operators = {node['opType'] for node, _ in consumers.get(name, [])}
        if name in protected or not operators or not operators <= {'Mul', 'Where'}:
            continue
        raw = value.tobytes()
        identity = (value.dtype.str, value.shape, sha256(raw))
        previous = canonical.get(identity)
        if previous is None:
            canonical[identity] = name
        elif weights[previous].tobytes() == raw:
            duplicates[name] = previous
    if duplicates or scalars:
        for node in graph['nodes']:
            node['inputs'] = {port:duplicates.get(name,name) for port,name in node['inputs'].items()}
        for name in duplicates:
            del weights[name]
        save_file(weights, weights_path, metadata=metadata)
        write_graph(graph_path, graph)
    result_bytes = graph_path.stat().st_size + weights_path.stat().st_size
    return {'method':'exact-pointwise-constant-storage-v2', 'duplicates':duplicates,
        'uniform_mul_scalars':scalars,
        'source_bytes':source_bytes, 'result_bytes':result_bytes, 'saved_bytes':source_bytes-result_bytes}


def convert_model(source, destination, variant='fp32', version=VERSION, **_):
    if variant != 'fp32' or version != VERSION:
        raise ValueError(f'Only HF FP32 and VolvoxAI {VERSION} are supported')
    config = json.loads((source / 'config.json').read_text())
    if any(config[key] != value for key, value in {'vocab_size':1536, 'd_model':320, 'heads':8,
            'dec_layers':4, 'max_q_len':192, 'max_out_len':192, 'img_tokens':210}.items()):
        raise ValueError('Unexpected HF model contract')
    destination.mkdir(parents=True)
    contracts = {}
    deduplication = {}
    for role in ROLES:
        directory = destination / role
        directory.mkdir()
        prepared = destination / f'{role}.onnx'
        model = prepare_onnx(source, prepared, role)
        command = [sys.executable, '-I', '-m', 'volvoxai', 'export', '--model', str(prepared),
            '--out', str(directory / 'model.safetensors'), '--weight-dtype', 'float32',
            '--target', 'native-cpu', '--target', 'wasm', '--allow-silu-numerical-migration',
            '--enable-exact-common-subexpression-elimination', '--report', str(directory / 'export-report.json'),
            '--report-format', 'json']
        if role == 'decoder':
            command += ['--dimension-bound', 'P=1:191', '--dimension-bound', 'R=2:192']
        for value in model.graph.input:
            if value.type.tensor_type.elem_type in (onnx.TensorProto.INT64, onnx.TensorProto.BOOL):
                command += ['--input-dtype', f'{value.name}=int32']
        for value in model.graph.output:
            command += ['--output-name', value.name]
            if value.type.tensor_type.elem_type in (onnx.TensorProto.INT64, onnx.TensorProto.BOOL):
                command += ['--output-dtype', f'{value.name}=int32']
        subprocess.run(command, check=True)
        deduplication[role] = deduplicate_weights(directory)
        graph = json.loads((directory / 'graph.json').read_text())
        if list(graph['inputs']) != [f'input{i}' for i in range(len(model.graph.input))]:
            raise ValueError('Exporter input mapping changed')
        contracts[role] = {'inputs': {value.name: f'input{i}' for i,value in enumerate(model.graph.input)},
                           'outputs': [value.name for value in model.graph.output]}
        prepared.unlink()
    upstream = json.loads((source / 'manifest.json').read_text())
    for name in ('config.json', 'vocab.json'):
        shutil.copy2(source / name, destination / name)
    write_json(destination / 'manifest.json', {
        'format':'receipt-vqa-volvoxai/v1', 'variant':'fp32', 'graphs':contracts,
        'tokenizer': upstream['tokenizer'], 'family_names':FAMILIES,
        'question_capacity':192, 'max_decoder_steps':191, 'initial_masked_cache_rows':1,
        'changes':['Masked question padding to 192 tokens; batch size 1',
                   'Int32 public IDs/masks; positive-length masked KV sentinel',
                   'VolvoxAI SiLU numerical migration and exact common subexpression elimination',
                   'Compact uniform Mul constants with proven output-shape preservation; share identical Mul/Where constants'],
        'converter': {'package':'volvoxai','version':VERSION,'source':package_source(),
                      'weight_deduplication':deduplication},
        'source_sha256':{name:sha256((source/name).read_bytes()) for name in
                         ('encoder_model.onnx','decoder_model.onnx','config.json','vocab.json')},
    })


def tensors(feed, mapping):
    return [pb.Tensor(name=mapping[name], shape=list(value.shape),
        dtype=pb.DataType.DATA_TYPE_F32 if value.dtype.kind == 'f' else pb.DataType.DATA_TYPE_I32,
        inline=np.ascontiguousarray(value, dtype=np.float32 if value.dtype.kind == 'f' else np.int32).tobytes())
        for name,value in feed.items()]


def role_weight_paths(package, role, manifest=None):
    """Resolve explicit shards relative to the package root; accept legacy files."""
    package = Path(package)
    manifest = manifest or json.loads((package/'manifest.json').read_text())
    if role not in ROLES:
        raise ValueError(f'Unknown model role: {role}')
    names = manifest['graphs'][role].get('weights', [f'{role}/model.safetensors'])
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names) \
            or len(set(names)) != len(names):
        raise ValueError('A model role requires distinct weight paths')
    for name in names:
        if '..' in name or not re.fullmatch(rf'(?:{role}|shared)/[a-zA-Z0-9._-]+\.safetensors', name):
            raise ValueError('Weight paths must stay inside their role or shared package directory')
    return [package/name for name in names]


def package_model_files(package):
    """Return each deployable model asset once, including shared weight shards."""
    package = Path(package)
    manifest = json.loads((package/'manifest.json').read_text())
    files = [package/name for name in ('manifest.json', 'config.json', 'vocab.json')]
    for role in ROLES:
        files.append(package/role/'graph.json')
        files.extend(role_weight_paths(package, role, manifest))
    return list(dict.fromkeys(files))


def share_token_embeddings(package):
    """Store byte-identical encoder/decoder embeddings once after PTQ export.

    Both public LoadModel calls still own their source snapshots. This shares
    package/download storage, with no claim that runtime allocations are shared.
    Calibration sources keep their original single-shard format.
    """
    package = Path(package)
    manifest = json.loads((package/'manifest.json').read_text())
    if manifest.get('storage', {}).get('format') == 'receipt-shared-weights/v1':
        return manifest['storage']
    config = json.loads((package/'config.json').read_text())
    expected_shape = (config['vocab_size'], config['d_model'])
    graphs, weights, metadata, names = {}, {}, {}, {}
    for role in ROLES:
        graphs[role] = json.loads((package/role/'graph.json').read_text())
        weights[role] = load_file(package/role/'model.safetensors')
        with safe_open(package/role/'model.safetensors', framework='numpy') as reader:
            metadata[role] = reader.metadata()
        candidates = {node['inputs']['weight'] for node in graphs[role]['nodes']
            if node['opType'] == 'Embedding'
            and weights[role][node['inputs']['weight']].shape == expected_shape}
        if len(candidates) != 1:
            raise ValueError(f'Expected one token embedding in {role}')
        names[role] = candidates.pop()
        protected = set(graphs[role]['inputs']) | set(graphs[role]['outputs']) | set(graphs[role].get('banks', {}))
        protected.update(graphs[role].get('banks', {}).values())
        for name, affine in graphs[role].get('quantization', {}).get('tensors', {}).items():
            protected.add(name)
            protected.update(affine[key] for key in ('scale_tensor', 'zero_point_tensor') if key in affine)
        if names[role] in protected or weights[role][names[role]].dtype != np.dtype('float32'):
            raise ValueError('Only private F32 token embedding weights can be shared')
    embedding = weights['encoder'][names['encoder']]
    if embedding.tobytes() != weights['decoder'][names['decoder']].tobytes():
        raise ValueError('Encoder and decoder token embeddings differ; exact sharing is unavailable')
    tensor_name = 'receipt.shared.token_embedding'
    relative = 'shared/token-embedding.safetensors'
    if any(tensor_name in weights[role] for role in ROLES):
        raise ValueError('Shared embedding tensor name already exists')
    (package/'shared').mkdir(exist_ok=True)
    save_file({tensor_name:embedding}, package/relative,
              metadata={'format':'receipt-shared-weights/v1', 'arithmetic':'unchanged'})
    for role in ROLES:
        for node in graphs[role]['nodes']:
            node['inputs'] = {port:tensor_name if name == names[role] else name
                              for port, name in node['inputs'].items()}
        del weights[role][names[role]]
        save_file(weights[role], package/role/'model.safetensors', metadata=metadata[role])
        write_graph(package/role/'graph.json', graphs[role])
        manifest['graphs'][role]['weights'] = [f'{role}/model.safetensors', relative]
    storage = {'format':'receipt-shared-weights/v1', 'shared_weight_files':[relative],
        'token_embedding':{'tensor':tensor_name, 'dtype':'float32', 'shape':list(embedding.shape),
            'payload_sha256':sha256(embedding.tobytes()), 'removed_duplicate_payload_bytes':embedding.nbytes},
        'arithmetic':'unchanged', 'runtime_memory_sharing':False}
    manifest['storage'] = storage
    write_json(package/'manifest.json', manifest)
    return storage


class Runner:
    def __init__(self, package, *, precision=None, cpu_threads=2):
        packaged_library_paths()
        manifest = json.loads((package / 'manifest.json').read_text())
        if manifest.get('frozen_candidate'):
            from vqa.frozen import validate_package
            validate_package(package, manifest.get('quantization', {}).get('precision'))
        elif manifest.get('converter', {}).get('version') != VERSION:
            raise ValueError(f'Expected a VolvoxAI {VERSION} model package')
        quantization = manifest.get('quantization', {})
        if precision is not None and quantization.get('precision') != precision:
            raise ValueError(f'Cached model precision does not match {precision}')
        if manifest.get('variant') == 'ptq':
            expected_settings = precision_metadata(quantization.get('precision'))
            for key in ('packed_arithmetic', 'activation_dtype', 'weight_dtype', 'group_size'):
                if quantization.get(key) != expected_settings[key]:
                    raise ValueError(f'Cached model precision has inconsistent {key}')
            for role in ROLES:
                graph = json.loads((package/role/'graph.json').read_text())
                validate_quantized_weights(package/role, graph, quantization['precision'],
                    quantization['roles'][role]['quantized_nodes'])
        self.contracts = manifest['graphs']
        arithmetic = quantization.get('packed_arithmetic', 'f32')
        self.sessions = {}
        try:
            for role in ROLES:
                session = vx.InferenceSession(package / role / 'graph.json', cpu_threads=cpu_threads,
                    weights=role_weight_paths(package, role, manifest),
                    packed_arithmetic=arithmetic)
                self.sessions[role] = session
                if session.backend != 'cpu':
                    raise RuntimeError('Expected CPU execution for calibration')
                expected = pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8 if arithmetic == 'dynamic-i8' \
                    else pb.PackedArithmetic.PACKED_ARITHMETIC_F32
                if session.packed_arithmetic != expected:
                    raise RuntimeError('Compiled arithmetic differs from the model precision')
        except BaseException:
            self.close()
            raise

    def run(self, role, feed, *, output_names=None, reuse_inputs=(), feedback=None):
        mapping = self.contracts[role]['inputs']
        values = {mapping[name]: np.ascontiguousarray(value,
            dtype=np.float32 if value.dtype.kind == 'f' else np.int32) for name, value in feed.items()}
        outputs = self.sessions[role].run(values, output_names=output_names,
            reuse_inputs=[mapping[name] for name in reuse_inputs],
            feedback={mapping[name]: output for name, output in (feedback or {}).items()})
        for name, value in outputs.items():
            if not np.isfinite(value).all():
                raise ValueError(f'Non-finite {role} output: {name}')
        return outputs

    def close(self):
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()


def encoder_feed(image, ids):
    if not 1 <= len(ids) <= 192:
        raise ValueError('Question length exceeds the model capacity')
    return {'image':image, 'question_ids':np.asarray([list(ids)+[0]*(192-len(ids))],np.int32),
            'family_ids':np.asarray([-1],np.int32)}


def decoder_feed(encoded):
    return {**{name:encoded[name] for name in CROSS},
        'memory_padding_mask':encoded['memory_padding_mask'], 'family_ids':encoded['selected_family_ids'],
        'decoder_input_ids':np.asarray([[1]],np.int32), 'position_ids':np.asarray([0],np.int32),
        'past_padding_mask':np.ones((1,1),np.int32),
        **{name:np.zeros((1,8,1,40),np.float32) for name in PAST}}


def advance(feed, output, token, position):
    return {**feed, 'decoder_input_ids':np.asarray([[token]],np.int32),
        'position_ids':np.asarray([position+1],np.int32),
        'past_padding_mask':output['present_padding_mask'],
        **{past:output[present] for past,present in zip(PAST,PRESENT)}}


def generate(runner, image, ids, observe=None, max_steps=MAX_STEPS):
    feed = encoder_feed(image,ids)
    if observe: observe('encoder',feed,0)
    encoded = runner.run('encoder',feed, output_names=None if observe else
        ['memory_padding_mask', 'selected_family_ids', *CROSS])
    feed = decoder_feed(encoded)
    generated = [1]
    for position in range(max_steps):
        if observe: observe('decoder',feed,position)
        # Calibration observers need complete host batches. Ordinary generation
        # keeps fixed conditioning and growing KV inside the execution context.
        bindings = {} if observe or position == 0 else {
            'reuse_inputs': ['memory_padding_mask', 'family_ids', *CROSS],
            'feedback': {'past_padding_mask':'present_padding_mask', **dict(zip(PAST, PRESENT))}}
        output = runner.run('decoder',feed, output_names=None if observe else ['logits'], **bindings)
        token = int(output['logits'][0,0].argmax())
        generated.append(token)
        if token == 2: break
        feed = advance(feed,output,token,position) if observe else {
            'decoder_input_ids':np.asarray([[token]],np.int32),
            'position_ids':np.asarray([position+1],np.int32)}
    return {'tokens':generated,'family':FAMILIES[int(encoded['selected_family_ids'][0])]}


DEFAULT_QUESTIONS = [
    'What is the store phone number?', '가게 전화번호는 무엇입니까?',
    'What is the street number in the store address?', '가게 주소의 마지막 숫자는 무엇입니까?',
    'What is the store name?', '가게 상호명은 무엇입니까?',
    'What is the name of the first item?', '첫 번째 상품의 이름은 무엇입니까?',
    'What is the sum of all item totals?', '모든 상품의 금액 합계는 얼마입니까?',
    'What is the quantity of milk?', '우유의 수량은 몇 개입니까?',
]


def select_calibration(images, excluded, count, seed, questions=None):
    question_map = {}
    if questions:
        for line in questions.read_text().splitlines():
            if not line.strip(): continue
            record = json.loads(line)
            name, question = record['image'], clean_text(record['question'])
            if Path(name).name != name or not question or name in question_map:
                raise ValueError('Questions JSONL requires unique image basenames and nonempty questions')
            question_map[name] = question
    excluded_hashes = {sha256(preprocess(path).tobytes()) for d in excluded for path in image_paths(d)}
    candidates = image_paths(images)
    random.Random(seed).shuffle(candidates)
    selected, seen = [], set()
    for path in candidates:
        if questions and path.name not in question_map: continue
        identity = sha256(preprocess(path).tobytes())
        if identity in excluded_hashes or identity in seen: continue
        seen.add(identity)
        selected.append({'path':path,'input_sha256':identity,'file_sha256':sha256(path.read_bytes()),
            'question': question_map.get(path.name,DEFAULT_QUESTIONS[len(selected)%len(DEFAULT_QUESTIONS)])})
        if len(selected) == count: break
    if len(selected) != count:
        raise ValueError(f'Need {count} distinct calibration images/questions outside evaluation; found {len(selected)}')
    return selected, excluded_hashes


def quantize_model(fp32, destination, *, source, images, excluded, count, seed,
                   evaluation_checked=False, questions=None, precision='w8a8', cpu_threads=None):
    settings = precision_metadata(precision)
    # Keep calls from already-running builds compatible with their previously
    # imported helper, and preserve the legacy calibration/session defaults.
    thread_options = {} if cpu_threads is None else {'cpu_threads': cpu_threads}
    selected, excluded_hashes = select_calibration(images,excluded,count,seed,questions)
    destination.mkdir(parents=True)
    tok = load_tokenizer(source)
    runner = None
    host = open_packaged_host()
    quantization = vx.VxQuantizationServiceClient(host)
    plans, reports, counts, families = {}, {}, Counter(), Counter()
    contracts = json.loads((fp32/'manifest.json').read_text())['graphs']
    started = time.perf_counter()
    try:
        for role in ROLES:
            (destination/role).mkdir()
            plans[role], count_nodes, stages = create_quantization_pipeline(quantization, fp32/role, precision,
                ('Conv2D', 'Linear', 'Gemm'), **thread_options)
            reports[role] = {'quantized_nodes':count_nodes, 'stages':stages}
        requires_calibration = any(plan.info.requires_calibration for plan in plans.values())
        requires_decoder_calibration = plans['decoder'].info.requires_calibration
        if requires_decoder_calibration:
            runner = Runner(fp32, **thread_options)
        for index,item in enumerate(selected if requires_calibration else []):
            image = preprocess(item['path'])
            if sha256(image.tobytes()) != item['input_sha256']: raise ValueError('Calibration image changed')
            ids = tok.encode(clean_text(item['question']),add_eos=True)
            if len(ids) > 192:
                raise ValueError('A calibration question exceeds the model capacity of 192 tokens')
            def observe(role,feed,position):
                if role == 'decoder' and position not in (0,4,12): return
                if not plans[role].info.requires_calibration: return
                quantization.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                    quantization_plan_id=plans[role].quantization_plan_id,
                    profile_name='default',sample_name=f'receipt-{index}-step-{position}',sample_count=1,
                    inputs=tensors(feed,contracts[role]['inputs'])))
                counts[role] += 1
            if requires_decoder_calibration:
                # Only prefixes through position 12 are observed; no target answers used.
                prediction = generate(runner,image,ids,observe,max_steps=13)
                families[prediction['family']] += 1
            else:
                observe('encoder',encoder_feed(image,ids),0)
            if (index+1)%16 == 0 or index+1 == count:
                print(f'VolvoxAI VQA PTQ {index+1}/{count} ({time.perf_counter()-started:.1f}s)',flush=True)
        for role in ROLES:
            graph = export_quantization_plan(quantization, plans[role], destination/role, counts[role])
            validate_quantized_weights(destination/role, graph, precision, reports[role]['quantized_nodes'])
            reports[role].update(samples=counts[role], coverage_complete=True,
                quantized_convolutions=sum(node['opType'] == 'QConv2D' for node in graph['nodes']),
                packed_linear_nodes=sum(node['opType'] == 'PackedLinear' for node in graph['nodes']),
                retained_float_convolutions=sum(node['opType'] == 'Conv2D' for node in graph['nodes']),
                operators=dict(Counter(node['opType'] for node in graph['nodes'])))
    finally:
        try:
            for plan in plans.values():
                quantization.release_quantization_plan(pb.QuantizationPlanRef(
                    quantization_plan_id=plan.quantization_plan_id))
        finally:
            host.close()
            if runner is not None:
                runner.close()
    metadata = {'method':'volvoxai-ptq','source_variant':'fp32','package':'volvoxai','version':VERSION,
        **settings,
        'selected_operators':['Conv2D','Linear','MatMul'] if precision in ('w4a8', 'w4a8-conv4') else ['Conv2D','Linear','Gemm'], 'roles':reports,
        'calibration':{'samples':count if requires_calibration else 0, 'required':requires_calibration,
            **thread_options,
            'selected_images':count,'seed':seed,'coverage_complete':True,
            'questions':'external-jsonl' if questions else 'bilingual-template-v1',
            'auto_families':dict(families),'decoder_positions':[0,4,12] if requires_decoder_calibration else [],
            'evaluation_checked':evaluation_checked,'evaluation_overlap':0 if evaluation_checked else None},
        'native_sha256':{path.name:sha256(path.read_bytes()) for path in packaged_library_paths()}}
    manifest = json.loads((fp32/'manifest.json').read_text())
    manifest.update(variant='ptq',quantization=metadata)
    write_json(destination/'manifest.json',manifest)
    for name in ('config.json','vocab.json'): shutil.copy2(fp32/name,destination/name)
    share_token_embeddings(destination)
    write_json(destination/'calibration-inputs.json',{'images_directory':str(images),
        'questions_file':str(questions) if questions else None,
        'excluded_directories':[str(p) for p in excluded], 'excluded_input_sha256':sorted(excluded_hashes),
        'images':[{**item,'path':str(item['path'])} for item in selected]})
    write_json(destination/'ptq-report.json',{**metadata,'seconds':time.perf_counter()-started})
    return metadata
