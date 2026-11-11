#!/usr/bin/env python3
"""Compare native VolvoxAI receipt models with the pinned ONNX release.

The default evaluates the two public release examples. --evaluation-root reads
external images/ and annotations/ directories without copying private inputs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort
import volvoxai as vx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.model_support import VERSION, PRECISIONS, SUPPORTED_PRECISIONS, packaged_library_paths, preprocess, sha256, write_json
from digit.frozen import profiles as frozen_profiles, validate_graph, validate_package


def decode(logits):
    logits = np.asarray(logits)
    if logits.shape != (1, 16, 11) or not np.isfinite(logits).all():
        raise ValueError(f'Expected finite slot logits [1, 16, 11]; got {logits.shape}')
    slots = logits.argmax(axis=-1)[0]
    def digits(values):
        end = np.flatnonzero(values == 10)
        return ''.join(str(int(value)) for value in values[:int(end[0]) if len(end) else len(values)])
    return {'phone': digits(slots[:12]), 'street': digits(slots[12:])}, slots


def load_cases(source, router, evaluation_root=None, limit=None):
    if evaluation_root is None:
        examples = json.loads((source / 'examples/examples.json').read_text())
        return [{'id': item['image'], 'image': source / 'examples' / item['image'],
                 'record': item['expected_record'], 'questions': item['questions']}
                for item in examples[:limit]]
    images = evaluation_root / 'images'
    annotations = evaluation_root / 'annotations'
    if not images.is_dir() or not annotations.is_dir():
        raise ValueError('--evaluation-root must contain images/ and annotations/')
    cases = []
    for label in sorted(annotations.glob('*.json'))[:limit]:
        item = json.loads(label.read_text())
        image_name = Path(item.get('image_path', f'{label.stem}.jpg')).name
        image = images / image_name
        if not image.is_file():
            raise ValueError(f'Missing evaluation image: {image}')
        store = item['receipt']['store']
        record = {'phone': router.digits_only(store['phone']),
                  'street': router.street_no_from_address(store['address'])}
        if not record['phone'] or not record['street']:
            raise ValueError(f'Incomplete numeric label: {label.name}')
        cases.append({'id': str(item.get('id', label.stem)), 'image': image, 'record': record,
                      'questions': [{'question': item['question'], 'answer': str(item['answer'])}]})
    if not cases:
        raise ValueError('No evaluation annotations found')
    return cases


def model_metrics():
    return {'correct': dict(phone=0, street=0, record=0, answer=0, routable=0),
            'inference_seconds': 0.0, 'record_agreement_with_fp32': 0,
            'matching_slots_with_fp32': 0, 'maximum_logit_difference_from_onnx': 0.0}


def ensure_fp32_reference(source, packages):
    """Create the public FP32 comparison model when a frozen-only build lacks it."""
    reference = packages/'fp32-prepared'
    if (reference/'manifest.json').is_file():
        return
    from digit.model import convert_model
    packages.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-', dir=packages) as temporary:
        converted = Path(temporary)/'fp32-prepared'
        convert_model(source, converted, 'fp32', VERSION, prepare_for_ptq=True)
        if reference.exists():
            shutil.rmtree(reference)
        converted.replace(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('w8a8', 'all', *frozen_profiles()), default='conv4-linear2')
    parser.add_argument('--evaluation-root', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-answer-drop', type=float, default=.005,
                        help='Maximum absolute answer-accuracy loss vs ONNX on external evaluation (default .005)')
    args = parser.parse_args()
    packaged_library_paths()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not 0 <= args.max_answer_drop <= 1:
        parser.error('--max-answer-drop must be between 0 and 1')
    precisions = ['w8a8'] if args.precision == 'all' else [args.precision]
    config = json.loads((ROOT / 'digit/extension.json').read_text())
    source = ROOT / '.cache/digit/huggingface' / config['model']['revision']
    packages = ROOT / '.cache/digit/packages' / VERSION
    spec = importlib.util.spec_from_file_location('released_digit_question_router', source / 'question_router.py')
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)
    cases = load_cases(source, router, args.evaluation_root, args.limit)
    question_count = sum(len(case['questions']) for case in cases)
    scope = 'external-heldout' if args.evaluation_root else 'public-examples'
    filename = 'heldout-check.json' if args.evaluation_root else 'model-check.json'
    ensure_fp32_reference(source, packages)
    paths = {'fp32-prepared': packages / 'fp32-prepared',
             **{precision: packages / 'ptq' / precision for precision in precisions}}
    manifests = {name: json.loads((directory / 'manifest.json').read_text()) for name, directory in paths.items()}
    for name, manifest in manifests.items():
        if 'frozen_candidate' in manifest:
            validate_package(paths[name], source, name)
        elif name == 'conv4-linear2':
            from digit.rebuild import validate_package as validate_rebuilt_package
            validate_rebuilt_package(paths[name], source, name)
        elif manifest['converter']['version'] != VERSION:
            raise ValueError(f'{name}: package runtime version mismatch')
        if name in precisions and manifest['quantization']['precision'] != name:
            raise ValueError(f'{name}: quantization precision mismatch')
    calibration = {}
    for precision in precisions:
        audit_path = paths[precision] / 'calibration-inputs.json'
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text())
            calibration[precision] = {item['input_sha256'] for item in audit['images']}
        else:
            if 'frozen_candidate' not in manifests[precision]:
                raise ValueError('Missing calibration audit')
            profile = frozen_profiles()[precision]
            validate_graph(json.loads((paths[precision]/'graph.json').read_text()), profile)
            if any(sha256((paths[precision]/name).read_bytes()) != expected for name, expected in profile['sha256'].items()):
                raise ValueError('Frozen model hash mismatch')
            calibration[precision] = None
    metrics = {name: model_metrics() for name in ['onnx', *paths]}
    overlaps = {precision: [] for precision in precisions}
    details = []
    parity = {'passed': True, 'max_absolute_difference': 0.0, 'rtol': .0005, 'atol': .001}
    report = {'runtime': VERSION, 'backend': 'cpu', 'cpu_threads': 1, 'scope': scope,
              'samples': len(cases), 'questions': question_count, 'reference': 'Pinned HF FP32 ONNX',
              'model_revision': config['model']['revision'],
              'onnx_sha256': sha256((source / 'model.onnx').read_bytes()),
              'max_answer_accuracy_drop': args.max_answer_drop if args.evaluation_root else 0,
              'fp32_parity': parity, 'models': metrics}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(source / 'model.onnx'), options, providers=['CPUExecutionProvider'])
    reference_input = reference.get_inputs()[0].name
    report['reference_runtime'] = f'ONNX Runtime {ort.__version__}'
    with ExitStack() as stack:
        sessions = {}
        for name, directory in paths.items():
            arithmetic = manifests[name].get('quantization', {}).get('packed_arithmetic', 'f32')
            sessions[name] = stack.enter_context(vx.InferenceSession(directory, backend='cpu',
                cpu_threads=1, packed_arithmetic=arithmetic))
            expected = (vx.pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8 if arithmetic == 'dynamic-i8'
                        else vx.pb.PackedArithmetic.PACKED_ARITHMETIC_F32)
            if sessions[name].packed_arithmetic != expected:
                raise ValueError(f'{name}: compiled arithmetic differs from the requested precision')
            metrics[name].update(packed_arithmetic=arithmetic,
                sha256={file: sha256((directory / file).read_bytes()) for file in ('graph.json', 'model.safetensors')},
                size_bytes=sum((directory / file).stat().st_size for file in ('graph.json', 'model.safetensors')))
        for index, case in enumerate(cases):
            values = preprocess(case['image'])
            input_hash = sha256(values.tobytes())
            for precision in precisions:
                if calibration[precision] is not None and input_hash in calibration[precision]:
                    overlaps[precision].append(case['id'])
            started = time.perf_counter()
            outputs = {'onnx': reference.run(None, {reference_input: values})[0]}
            metrics['onnx']['inference_seconds'] += time.perf_counter() - started
            for name, session in sessions.items():
                started = time.perf_counter()
                outputs[name] = session.run({'input0': values})['slot_logits']
                metrics[name]['inference_seconds'] += time.perf_counter() - started
            records = {name: decode(output) for name, output in outputs.items()}
            difference = float(np.max(np.abs(outputs['fp32-prepared'] - outputs['onnx'])))
            parity['max_absolute_difference'] = max(parity['max_absolute_difference'], difference)
            if not np.allclose(outputs['fp32-prepared'], outputs['onnx'], rtol=parity['rtol'], atol=parity['atol']):
                parity['passed'] = False
            item = {'id': case['id'], 'expected_record': case['record'], 'models': {}}
            for name, output in outputs.items():
                record, slots = records[name]
                metric = metrics[name]
                for field in ('phone', 'street'):
                    metric['correct'][field] += record[field] == case['record'][field]
                metric['correct']['record'] += record == case['record']
                metric['record_agreement_with_fp32'] += record == records['fp32-prepared'][0]
                metric['matching_slots_with_fp32'] += int((slots == records['fp32-prepared'][1]).sum())
                metric['maximum_logit_difference_from_onnx'] = max(metric['maximum_logit_difference_from_onnx'],
                    float(np.max(np.abs(output - outputs['onnx']))))
                answers = []
                for question in case['questions']:
                    predicted = router.answer_from_record(question['question'], **record)
                    expected = str(question['answer'])
                    metric['correct']['routable'] += bool(predicted)
                    metric['correct']['answer'] += predicted == expected
                    answers.append({'expected': expected, 'actual': predicted, 'correct': predicted == expected})
                item['models'][name] = {'record': record, 'answers': answers}
            details.append(item)
            if (index + 1) % 100 == 0 or index + 1 == len(cases):
                print(f'Digit {scope}: {index + 1}/{len(cases)} images', flush=True)
    for name, metric in metrics.items():
        for field in ('phone', 'street', 'record'):
            metric[f'{field}_accuracy'] = metric['correct'][field] / len(cases)
        metric['answer_accuracy'] = metric['correct']['answer'] / question_count
        metric['record_agreement_with_fp32'] /= len(cases)
        metric['slot_agreement_with_fp32'] = metric['matching_slots_with_fp32'] / (len(cases) * 16)
        metric['mean_inference_ms'] = metric['inference_seconds'] * 1000 / len(cases)
    all_passed = True
    for precision in precisions:
        selected = metrics[precision]
        delta = selected['answer_accuracy'] - metrics['onnx']['answer_accuracy']
        accuracy_ok = delta >= -args.max_answer_drop - 1e-12 if args.evaluation_root else (
            selected['correct']['answer'] == question_count and selected['correct']['record'] == len(cases))
        passed = parity['passed'] and not overlaps[precision] and accuracy_ok
        all_passed = all_passed and passed
        result = {**report, 'precision': precision, 'passed': passed,
                  'answer_accuracy_delta_vs_onnx': delta,
                  'calibration_evaluation_overlap': len(overlaps[precision]) if calibration[precision] is not None else None,
                  'calibration_overlap_status': 'checked' if calibration[precision] is not None else 'original frozen experiment audit; not rechecked against this input set',
                  'calibration_evaluation_overlap_ids': overlaps[precision],
                  'quantization': manifests[precision]['quantization'], 'cases': details}
        directory = ROOT / 'test-results/digit' / precision
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / filename, result)
        print(f'{precision}: answers {selected["correct"]["answer"]}/{question_count}; '
              f'records {selected["correct"]["record"]}/{len(cases)}; '
              f'{selected["mean_inference_ms"]:.2f} ms/image; '
              f'answer delta {delta:+.4%}; {"PASS" if passed else "FAIL"}', flush=True)
        print(f'Report: {directory / filename}', flush=True)
    if not all_passed:
        raise SystemExit('Digit validation failed; inspect the precision reports')


if __name__ == '__main__':
    main()
