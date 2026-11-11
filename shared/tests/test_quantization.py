"""Exercise the 0.8 native plan lifecycle and actual I8 execution contracts."""
from pathlib import Path
from contextlib import ExitStack
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from safetensors.numpy import save_file
from safetensors import safe_open
import volvoxai as vx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.model_support import (create_quantization_plan, create_quantization_pipeline, export_quantization_plan,
    open_packaged_host, packaged_library_paths, precision_metadata, quantization_config,
    validate_quantized_weights, VERSION)
from vqa.model import deduplicate_weights, Runner, share_token_embeddings, role_weight_paths, package_model_files
from digit import model as digit_model
from vqa import model as vqa_model


class QuantizationTests(unittest.TestCase):
    def test_model_thread_options_preserve_calibration_inputs_and_legacy_helper_calls(self):
        for module in (digit_model, vqa_model):
            with self.subTest(model=module.__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root/'source'
                source.mkdir()
                image = np.linspace(-1, 1, 320*672, dtype=np.float32).reshape(1, 1, 320, 672)
                selected = [{'path':root/'receipt.png', 'input_sha256':module.sha256(image.tobytes()),
                             'question':'What is the phone number?'}]
                contracts = {'encoder':{'inputs':{'image':'image_input'}},
                             'decoder':{'inputs':{'token':'token_input'}}}
                (source/'manifest.json').write_text(json.dumps({'graphs':contracts}))
                for name in ('graph.json', 'config.json', 'vocab.json'):
                    (source/name).write_text('{}')
                for name in ('model.safetensors', 'engine.so', 'loader.so'):
                    (source/name).write_bytes(b'fixture')
                graph = {'inputs':{'input0':{'shape':digit_model.INPUT_SHAPE, 'dtype':'float32'}},
                    'outputs':['slot_logits'], 'nodes':[{'opType':'QConv2D', 'params':{'stride':[2, 1]},
                        'outputs':{'out':{'tensor':'slot_logits' if index == 10 else f'v{index}',
                            'shape':digit_model.OUTPUT_SHAPE, 'dtype':'float32'}}} for index in range(11)]}
                baseline = None
                for threads in (None, 10):
                    with self.subTest(cpu_threads=threads), ExitStack() as stack:
                        calls = []
                        client, host, runner = Mock(), Mock(), Mock()
                        def legacy_pipeline(client_arg, directory, precision, operators):
                            # Deliberately no cpu_threads parameter: a running
                            # parent may retain this older function signature.
                            calls.append((directory, {}))
                            plan = SimpleNamespace(quantization_plan_id=2 if directory.name == 'decoder' else 1,
                                info=SimpleNamespace(requires_calibration=True))
                            return plan, 11 if module is digit_model else 1, []
                        def threaded_pipeline(client_arg, directory, precision, operators, *, cpu_threads=1):
                            result = legacy_pipeline(client_arg, directory, precision, operators)
                            calls[-1] = (directory, {'cpu_threads':cpu_threads})
                            return result
                        patches = {
                            'select_calibration':Mock(return_value=(selected, set())),
                            'preprocess':Mock(return_value=image),
                            'packaged_library_paths':Mock(return_value=(source/'engine.so', source/'loader.so')),
                            'open_packaged_host':Mock(return_value=host),
                            'create_quantization_pipeline':legacy_pipeline if threads is None else threaded_pipeline,
                            'export_quantization_plan':Mock(return_value=graph),
                            'validate_quantized_weights':Mock(),
                        }
                        if module is vqa_model:
                            def generate(runner_arg, values, ids, observe, *, max_steps):
                                self.assertIs(runner_arg, runner)
                                self.assertEqual(ids, [7, 9])
                                self.assertEqual(max_steps, 13)
                                np.testing.assert_array_equal(values, image)
                                observe('encoder', {'image':values}, 0)
                                for position in (0, 1, 4, 12, 13):
                                    observe('decoder', {'token':np.array([[position]], np.int32)}, position)
                                return {'family':'phone'}
                            patches.update(load_tokenizer=Mock(return_value=Mock(encode=Mock(return_value=[7, 9]))),
                                Runner=Mock(return_value=runner), generate=generate, share_token_embeddings=Mock())
                        stack.enter_context(patch.multiple(module, **patches))
                        stack.enter_context(patch.object(module.vx, 'VxQuantizationServiceClient', return_value=client))
                        options = {} if threads is None else {'cpu_threads':threads}
                        result = module.quantize_model(source, root/f'output-{threads}', source=source,
                            images=root, excluded=[], count=1, seed=7, **options)
                        self.assertEqual([options for _, options in calls],
                            [options]*(2 if module is vqa_model else 1))
                        self.assertEqual(result['calibration'].get('cpu_threads'), threads)
                        self.assertEqual('cpu_threads' in result['calibration'], threads is not None)
                        requests = [call.args[0] for call in client.calibrate_quantization_plan.call_args_list]
                        self.assertEqual(len(requests), 4 if module is vqa_model else 1)
                        self.assertEqual(requests[0].inputs[0].inline, image.tobytes())
                        serialized = [request.SerializeToString() for request in requests]
                        if baseline is None:
                            baseline = serialized
                        else:
                            self.assertEqual(serialized, baseline)
                        if module is vqa_model:
                            patches['Runner'].assert_called_once_with(source, **options)
                            self.assertEqual([request.sample_name for request in requests],
                                ['receipt-0-step-0', 'receipt-0-step-0', 'receipt-0-step-4', 'receipt-0-step-12'])
                            runner.close.assert_called_once_with()
                        host.close.assert_called_once_with()

    def test_vqa_runner_uses_requested_threads_for_both_sessions_and_keeps_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source/'manifest.json').write_text(json.dumps({'converter':{'version':VERSION},
                'graphs':{'encoder':{}, 'decoder':{}}}))
            for options, expected in (({}, 2), ({'cpu_threads':10}, 10)):
                with self.subTest(options=options), \
                        patch.object(vqa_model, 'packaged_library_paths'), \
                        patch.object(vqa_model.vx, 'InferenceSession', side_effect=lambda *args, **kwargs:
                            Mock(backend='cpu', packed_arithmetic=vx.pb.PackedArithmetic.PACKED_ARITHMETIC_F32)) as session:
                    runner = Runner(source, **options)
                    self.assertEqual([call.kwargs['cpu_threads'] for call in session.call_args_list], [expected]*2)
                    runner.close()

    def test_native_overrides_must_match_the_reported_wheel(self):
        engine, loader = packaged_library_paths()
        with patch.dict(os.environ, {'VOLVOXAI_LIBRARY':str(engine),
                                     'SYNURANG_MODULE_HOST_LIBRARY':str(loader)}):
            self.assertEqual(packaged_library_paths(), (engine, loader))
            with tempfile.TemporaryDirectory() as temporary:
                other = Path(temporary)/engine.name
                other.touch()
                for name in ('VOLVOXAI_LIBRARY', 'SYNURANG_MODULE_HOST_LIBRARY'):
                    with self.subTest(variable=name), patch.dict(os.environ, {name:str(other)}):
                        with self.assertRaisesRegex(RuntimeError, 'override differs'):
                            packaged_library_paths()

    def test_runner_rejects_mislabeled_cached_precision_and_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = {'variant':'ptq', 'converter':{'version':VERSION},
                        'quantization':precision_metadata('w8a8')}
            (directory/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'precision does not match w4a8'):
                Runner(directory, precision='w4a8')
            manifest['quantization']['packed_arithmetic'] = 'dynamic-i8'
            (directory/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'inconsistent packed_arithmetic'):
                Runner(directory, precision='w8a8')
            manifest['converter']['version'] = '0.6.0'
            (directory/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Expected a VolvoxAI'):
                Runner(directory)

    def test_exact_constant_sharing_preserves_shape_dtype_bits_and_metadata(self):
        values = np.array([0., 1.], dtype=np.float32)
        tensors = {'a_constant':values, 'b_duplicate':values.copy(),
            'different_shape':values.reshape(1, 2), 'different_dtype':values.astype(np.float64),
            'signed_zero':np.array([-0., 1.], dtype=np.float32),
            'public_weight':values.copy(), 'non_mul_weight':values.copy()}
        graph = {'inputs':{'input0':{}, 'public_weight':{}}, 'outputs':['output'],
            'nodes':[{'id':name, 'opType':'Add' if name == 'non_mul_weight' else 'Mul',
                'inputs':{'input':'input0','weight':name}} for name in tensors]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'graph.json').write_text(json.dumps(graph))
            save_file(tensors, directory/'model.safetensors', metadata={'producer':'regression-test'})
            result = deduplicate_weights(directory)
            self.assertEqual(result['duplicates'], {'b_duplicate':'a_constant'})
            converted = json.loads((directory/'graph.json').read_text())
            self.assertEqual(next(node for node in converted['nodes'] if node['id']=='b_duplicate')
                ['inputs']['weight'], 'a_constant')
            with safe_open(directory/'model.safetensors', framework='numpy') as reader:
                self.assertEqual(reader.metadata(), {'producer':'regression-test'})
                self.assertEqual(set(reader.keys()), set(tensors)-{'b_duplicate'})
                for name in reader.keys():
                    actual = reader.get_tensor(name)
                    self.assertEqual(actual.dtype, tensors[name].dtype)
                    self.assertEqual(actual.shape, tensors[name].shape)
                    self.assertEqual(actual.tobytes(), tensors[name].tobytes())
            again = deduplicate_weights(directory)
            self.assertEqual(again['duplicates'], {})
            self.assertEqual(again['saved_bytes'], 0)

    def test_unknown_precision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported precision'):
            quantization_config({'nodes': []}, 'w4a16', ('Linear',))

    def test_uniform_mul_compaction_preserves_bytes_and_broadcast_output_extent(self):
        for label, input_shape, weight, compact in (
                ('uniform', [2, 3], np.full((2, 3), 1.25, np.float32), True),
                ('negative_zero', [2, 3], np.full((2, 3), -0., np.float32), True),
                ('broadcast_driver', [1, 3], np.full((2, 3), 1.25, np.float32), False),
                ('mixed_zero_bits', [2, 3], np.array([[0., -0., 0.], [0., 0., 0.]], np.float32), False)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                graph = {'format':'volvox-graph/v1', 'dimensions':{},
                    'inputs':{'input0':{'shape':input_shape, 'dtype':'float32'}},
                    'nodes':[{'id':'mul', 'opType':'Mul', 'inputs':{'a':'input0', 'b':'constant'},
                        'outputs':{'out':{'tensor':'result', 'shape':[2, 3], 'dtype':'float32'}}, 'params':{}}],
                    'outputs':['result'], 'execution':{'stateless':{}, 'entrypoints':[{
                        'name':'evaluate', 'effect':1, 'allowed_ranges':[1], 'inputs':['input0'], 'outputs':['result']}]}}
                (directory/'graph.json').write_text(json.dumps(graph))
                save_file({'constant':weight}, directory/'model.safetensors')
                values = np.linspace(-2., 2., np.prod(input_shape), dtype=np.float32).reshape(input_shape)
                with vx.InferenceSession(directory, cpu_threads=1) as session:
                    before = session.run({'input0':values})['result']
                optimization = deduplicate_weights(directory)
                self.assertEqual(bool(optimization['uniform_mul_scalars']), compact)
                self.assertEqual(json.loads((directory/'graph.json').read_text()), graph)
                with safe_open(directory/'model.safetensors', framework='numpy') as weights:
                    self.assertEqual(weights.get_tensor('constant').shape, (1,) if compact else (2, 3))
                with vx.InferenceSession(directory, cpu_threads=1) as session:
                    after = session.run({'input0':values})['result']
                self.assertEqual(after.shape, (2, 3))
                self.assertEqual(after.tobytes(), before.tobytes())
                self.assertEqual(after.tobytes(), np.multiply(values, weight).tobytes())

    def test_duplicate_where_masks_share_storage_without_changing_infinity_bits(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            graph = {'format':'volvox-graph/v1', 'dimensions':{},
                'inputs':{'condition':{'shape':[2, 3], 'dtype':'int32'},
                          'input0':{'shape':[2, 3], 'dtype':'float32'}},
                'nodes':[{'id':str(index), 'opType':'Where',
                    'inputs':{'condition':'condition', 'a':f'mask{index}', 'b':'input0'},
                    'outputs':{'out':{'tensor':f'result{index}', 'shape':[2, 3], 'dtype':'float32'}},
                    'params':{}} for index in range(2)], 'outputs':['result0', 'result1'],
                'execution':{'stateless':{}, 'entrypoints':[{'name':'evaluate', 'effect':1,
                    'allowed_ranges':[1], 'inputs':['condition', 'input0'], 'outputs':['result0', 'result1']}]}}
            (directory/'graph.json').write_text(json.dumps(graph))
            mask = np.full((2, 3), -np.inf, np.float32)
            save_file({'mask0':mask, 'mask1':mask.copy()}, directory/'model.safetensors')
            feed = {'condition':np.array([[0, 1, 0], [1, 0, 1]], np.int32),
                    'input0':np.arange(6, dtype=np.float32).reshape(2, 3)}
            with vx.InferenceSession(directory, cpu_threads=1) as session:
                before = session.run(feed)
            optimization = deduplicate_weights(directory)
            self.assertEqual(optimization['duplicates'], {'mask1':'mask0'})
            self.assertEqual(optimization['uniform_mul_scalars'], {})
            with vx.InferenceSession(directory, cpu_threads=1) as session:
                after = session.run(feed)
            for name in before:
                self.assertEqual(after[name].shape, before[name].shape)
                self.assertEqual(after[name].tobytes(), before[name].tobytes())

    def test_shared_token_embedding_uses_real_multishard_loading_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            manifest = {'graphs':{role:{} for role in ('encoder','decoder')}}
            (package/'manifest.json').write_text(json.dumps(manifest))
            (package/'config.json').write_text(json.dumps({'vocab_size':4, 'd_model':3}))
            (package/'vocab.json').write_text('{}')
            expected = {}
            for role in ('encoder','decoder'):
                directory = package/role
                directory.mkdir()
                weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 7
                graph = {'format':'volvox-graph/v1', 'dimensions':{},
                    'inputs':{'input0':{'shape':[1, 2], 'dtype':'int32'}},
                    'nodes':[{'id':'embedding', 'opType':'Embedding',
                        'inputs':{'input':'input0', 'weight':role+'_tokens'},
                        'outputs':{'out':{'tensor':'result', 'shape':[1, 2, 3], 'dtype':'float32'}}, 'params':{}}],
                    'outputs':['result'], 'execution':{'stateless':{}, 'entrypoints':[{
                        'name':'evaluate', 'effect':1, 'allowed_ranges':[1], 'inputs':['input0'], 'outputs':['result']}]}}
                (directory/'graph.json').write_text(json.dumps(graph))
                save_file({role+'_tokens':weight, 'unused_other':np.array([3.], np.float32)},
                          directory/'model.safetensors', metadata={'producer':role})
                with vx.InferenceSession(directory, cpu_threads=1) as session:
                    expected[role] = session.run({'input0':np.array([[3, 1]], np.int32)})['result']
            encoder_graph = package/'encoder/graph.json'
            original = encoder_graph.read_text()
            for extra in ({'banks':{'logical_tokens':'encoder_tokens'}},
                          {'quantization':{'tensors':{'other':{'scale_tensor':'encoder_tokens'}}}}):
                with self.subTest(protected_reference=extra):
                    encoder_graph.write_text(json.dumps({**json.loads(original), **extra}))
                    with self.assertRaisesRegex(ValueError, 'Only private F32'):
                        share_token_embeddings(package)
                    self.assertFalse((package/'shared').exists())
            encoder_graph.write_text(original)
            storage = share_token_embeddings(package)
            self.assertEqual(storage['token_embedding']['removed_duplicate_payload_bytes'], 48)
            self.assertEqual(share_token_embeddings(package), storage)
            files = package_model_files(package)
            self.assertEqual(files.count(package/'shared/token-embedding.safetensors'), 1)
            for role in ('encoder','decoder'):
                self.assertEqual(role_weight_paths(package, role),
                    [package/role/'model.safetensors', package/'shared/token-embedding.safetensors'])
                with safe_open(package/role/'model.safetensors', framework='numpy') as reader:
                    self.assertEqual(reader.metadata(), {'producer':role})
                    self.assertEqual(set(reader.keys()), {'unused_other'})
                with vx.InferenceSession(package/role/'graph.json', cpu_threads=1,
                                         weights=role_weight_paths(package, role)) as session:
                    actual = session.run({'input0':np.array([[3, 1]], np.int32)})['result']
                self.assertEqual(actual.shape, expected[role].shape)
                self.assertEqual(actual.tobytes(), expected[role].tobytes())

    def test_shared_weight_paths_reject_escape_and_duplicate_entries(self):
        for names in (['../model.safetensors'], ['shared/a.safetensors']*2,
                      ['https://example.com/model.safetensors'], [['shared/a.safetensors']], []):
            with self.subTest(paths=names), self.assertRaises(ValueError):
                role_weight_paths(Path('/tmp/package'), 'encoder', {'graphs':{'encoder':{'weights':names}}})

    def test_packed_stage_retains_convolutions_for_the_static_stage(self):
        graph = {'nodes': [{'id': 'conv', 'opType': 'Conv2D'},
                           {'id': 'dense', 'opType': 'Linear'}]}
        config, nodes = quantization_config(graph, 'w4a8', ('Conv2D', 'Linear'))
        self.assertFalse(config.HasField('static_activations'))
        self.assertEqual([node['id'] for node in nodes], ['dense'])
        self.assertEqual([(rule.operator_name, rule.WhichOneof('action')) for rule in config.rules],
                         [('Conv2D', 'keep_float')])
        with self.assertRaisesRegex(ValueError, 'No eligible operators'):
            quantization_config({'nodes': graph['nodes'][:1]}, 'w4a8', ('Conv2D',))

    def test_mixed_pipeline_preserves_packing_and_calibrates_convolutions(self):
        self.mixed_pipeline('w4a8')

    def test_conv4_pipeline_preserves_linear_packing_and_dynamic_i8(self):
        self.mixed_pipeline('w4a8-conv4')

    def mixed_pipeline(self, precision):
        pb = vx.pb
        values = np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 1, 32, 1)
        dense = np.linspace(-.7, .9, 64, dtype=np.float32).reshape(2, 32)
        graph = {'format':'volvox-graph/v1', 'dimensions':{},
            'inputs':{'input0':{'shape':[1, 1, 32, 1], 'dtype':'float32'}},
            'nodes':[
                {'id':'conv', 'opType':'Conv2D', 'inputs':{'input':'input0', 'weight':'conv_weight'},
                 'outputs':{'out':{'tensor':'convolved', 'shape':[1, 1, 32, 1], 'dtype':'float32'}},
                 'params':{'stride':[1, 1], 'dilation':[1, 1], 'groups':1,
                           'pads':[0, 0, 0, 0], 'padding':[0, 0], 'data_layout':'NHWC', 'weight_layout':'HWIO'}},
                {'id':'view', 'opType':'Reshape', 'inputs':{'input':'convolved'},
                 'outputs':{'out':{'tensor':'flattened', 'shape':[1, 32], 'dtype':'float32'}},
                 'params':{'shape':[1, 32]}},
                {'id':'dense', 'opType':'Linear', 'inputs':{'input':'flattened', 'weight':'dense_weight'},
                 'outputs':{'out':{'tensor':'result', 'shape':[1, 2], 'dtype':'float32'}},
                 'params':{'weight_layout':'dout_din'}}],
            'outputs':['convolved', 'result'],
            'execution':{'stateless':{}, 'entrypoints':[{'name':'evaluate', 'effect':1,
                'allowed_ranges':[1], 'inputs':['input0'], 'outputs':['convolved', 'result']}]}}
        with tempfile.TemporaryDirectory() as temporary:
            source, packed, mixed = (Path(temporary)/name for name in ('source', 'packed', 'mixed'))
            for directory in (source, packed, mixed):
                directory.mkdir()
            (source/'graph.json').write_text(json.dumps(graph))
            save_file({'conv_weight':np.array([[[[.75]]]], np.float32), 'dense_weight':dense},
                      source/'model.safetensors')
            with open_packaged_host() as host:
                client = vx.VxQuantizationServiceClient(host)
                packed_plan = create_quantization_plan(client, source, 'w4a8', ('Conv2D',))
                try:
                    packed_graph = export_quantization_plan(client, packed_plan, packed, 0)
                finally:
                    client.release_quantization_plan(pb.QuantizationPlanRef(
                        quantization_plan_id=packed_plan.quantization_plan_id))
                with patch.object(client, 'create_quantization_plan', wraps=client.create_quantization_plan) as create:
                    plan, count, stages = create_quantization_pipeline(
                        client, source, precision, ('Conv2D',), cpu_threads=2)
                    self.assertEqual([call.args[0].calibration.cpu_threads for call in create.call_args_list], [2, 2])
                try:
                    self.assertTrue(plan.info.requires_calibration)
                    self.assertEqual(count, 2)
                    self.assertEqual([stage['requires_calibration'] for stage in stages], [False, True])
                    client.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                        quantization_plan_id=plan.quantization_plan_id, sample_name='mixed', sample_count=1,
                        inputs=[pb.Tensor(name='input0', shape=list(values.shape),
                            dtype=pb.DataType.DATA_TYPE_F32, inline=values.tobytes())]))
                    mixed_graph = export_quantization_plan(client, plan, mixed, 1)
                finally:
                    client.release_quantization_plan(pb.QuantizationPlanRef(
                        quantization_plan_id=plan.quantization_plan_id))
            nodes = validate_quantized_weights(mixed, mixed_graph, precision, count)
            self.assertEqual({node['opType'] for node in nodes}, {'PackedLinear', 'QConv2D'})
            packed_node = next(node for node in packed_graph['nodes'] if node['opType']=='PackedLinear')
            self.assertEqual(packed_node, next(node for node in mixed_graph['nodes'] if node['opType']=='PackedLinear'))
            with safe_open(packed/'model.safetensors', framework='numpy') as before, \
                    safe_open(mixed/'model.safetensors', framework='numpy') as after:
                for port in ('weight', 'scale'):
                    name = packed_node['inputs'][port]
                    self.assertEqual(before.get_tensor(name).tobytes(), after.get_tensor(name).tobytes())
            with vx.InferenceSession(mixed, cpu_threads=1, packed_arithmetic='dynamic-i8') as session:
                self.assertEqual(session.packed_arithmetic, pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8)
                actual = session.run({'input0':values})
            np.testing.assert_allclose(actual['convolved'], values*.75, atol=.01)
            np.testing.assert_allclose(actual['result'], values.reshape(1, 32)*.75 @ dense.T, rtol=.04, atol=.04)

    def test_mixed_pipeline_refuses_dense_dependent_convolution_calibration(self):
        graph = {'nodes':[
            {'id':'dense', 'opType':'Linear', 'inputs':{'input':'x'}, 'outputs':{'out':{'tensor':'y'}}},
            {'id':'view', 'opType':'Reshape', 'inputs':{'input':'y'}, 'outputs':{'out':{'tensor':'z'}}},
            {'id':'conv', 'opType':'Conv2D', 'inputs':{'input':'z'}, 'outputs':{}}]}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source/'graph.json').write_text(json.dumps(graph))
            with self.assertRaisesRegex(ValueError, 'independent of packed dense outputs'):
                create_quantization_pipeline(None, source, 'w4a8', ('Conv2D',))

    def test_exported_formats_execute_the_requested_arithmetic(self):
        pb = vx.pb
        values = np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 32)
        weights = np.linspace(-.7, .9, 64, dtype=np.float32).reshape(2, 32)
        expected = values @ weights.T
        graph = {'format': 'volvox-graph/v1', 'dimensions': {},
            'inputs': {'input0': {'shape': [1, 32], 'dtype': 'float32'}},
            'nodes': [{'id': 'dense', 'opType': 'Linear',
                'inputs': {'input': 'input0', 'weight': 'weight'},
                'outputs': {'out': {'tensor': 'out', 'shape': [1, 2], 'dtype': 'float32'}},
                'params': {'weight_layout': 'dout_din'}}], 'outputs': ['out'],
            'execution': {'entrypoints': [{'name': 'evaluate', 'effect': 1,
                'allowed_ranges': [1], 'inputs': ['input0'], 'outputs': ['out']}], 'stateless': {}}}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source'
            source.mkdir()
            (source / 'graph.json').write_text(json.dumps(graph))
            save_file({'weight': weights}, source / 'model.safetensors')
            for precision in ('w8a8', 'w4a8'):
                with self.subTest(precision=precision):
                    output = Path(temporary) / precision
                    output.mkdir()
                    with open_packaged_host() as host:
                        client = vx.VxQuantizationServiceClient(host)
                        plan = create_quantization_plan(client, source, precision, ('Linear',))
                        try:
                            self.assertEqual(plan.info.requires_calibration, precision == 'w8a8')
                            samples = 0
                            if plan.info.requires_calibration:
                                with self.assertRaisesRegex(ValueError, 'incomplete'):
                                    export_quantization_plan(client, plan, output, samples)
                                client.calibrate_quantization_plan(pb.CalibrateQuantizationPlanRequest(
                                    quantization_plan_id=plan.quantization_plan_id,
                                    sample_name='sample-1', sample_count=1,
                                    inputs=[pb.Tensor(name='input0', shape=[1, 32],
                                        dtype=pb.DataType.DATA_TYPE_F32, inline=values.tobytes())]))
                                samples = 1
                            converted = export_quantization_plan(client, plan, output, samples)
                        finally:
                            client.release_quantization_plan(pb.QuantizationPlanRef(
                                quantization_plan_id=plan.quantization_plan_id))
                    validate_quantized_weights(output, converted, precision, 1)
                    arithmetic = precision_metadata(precision)['packed_arithmetic']
                    with vx.InferenceSession(output, cpu_threads=1, packed_arithmetic=arithmetic) as session:
                        self.assertEqual(session.packed_arithmetic,
                            pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8 if precision == 'w4a8'
                            else pb.PackedArithmetic.PACKED_ARITHMETIC_F32)
                        actual = session.run({'input0': values})['out']
                    np.testing.assert_allclose(actual, expected, rtol=.04, atol=.04)


if __name__ == '__main__':
    unittest.main()
