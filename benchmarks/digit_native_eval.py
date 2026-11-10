#!/usr/bin/env python3
"""Evaluate frozen Digit models with an explicitly installed runtime wheel.

Fixtures and output references are private. Public per-model reports contain
only aggregate accuracy/latency, artifact hashes and measurement conditions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def decode(logits):
    values = np.asarray(logits)
    if values.shape != (1, 16, 11) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Expected finite float32 output [1,16,11]')
    slots = values.argmax(axis=-1)[0].tolist()
    def digits(part):
        end = part.index(10) if 10 in part else len(part)
        return ''.join(str(value) for value in part[:end])
    return {'phone': digits(slots[:12]), 'street': digits(slots[12:])}, slots


def latency_stats(values):
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Latency samples must be finite and positive')
    return {'samples': len(values), 'median_ms': statistics.median(values),
            'mean_ms': statistics.mean(values),
            'p95_ms': float(np.percentile(values, 95, method='linear'))}


def installed_runtime(version):
    import volvoxai as vx
    distribution = importlib.metadata.distribution('volvoxai')
    package = Path(vx.__file__).resolve().parent
    if (vx.__version__ != version or distribution.version != version
            or package != Path(distribution.locate_file('volvoxai')).resolve()):
        raise RuntimeError('Imported VolvoxAI does not match the requested installed wheel')
    engine = package / vx.library_filename()
    loader = package / vx.library_filename().replace('volvoxai', 'synurang_module_host')
    if not engine.is_file() or not loader.is_file() or vx.find_library().resolve() != engine.resolve():
        raise RuntimeError('Native engine/loader must come from the selected wheel')
    override = os.environ.get('SYNURANG_MODULE_HOST_LIBRARY')
    if override and Path(override).expanduser().resolve() != loader.resolve():
        raise RuntimeError('Native loader override differs from the selected wheel')
    # Make the already-verified loader choice explicit for every new session.
    os.environ['SYNURANG_MODULE_HOST_LIBRARY'] = str(loader)
    sources = [engine, loader, package/'__init__.py', package/'_library.py', package/'_session.py']
    return vx, {'package': 'volvoxai', 'version': version,
                'sha256': {p.name: sha(p.read_bytes()) for p in sources}}


def load_input(root, item):
    file = (root / item['file']).resolve()
    if not file.is_relative_to(root):
        raise ValueError('Input must remain inside the fixture directory')
    raw = file.read_bytes()
    if len(raw) != 320*672*4 or sha(raw) != item['sha256']:
        raise ValueError('Input hash or byte length changed: ' + item['id'])
    values = np.frombuffer(raw, dtype='<f4').copy().reshape(1, 1, 320, 672)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite input: ' + item['id'])
    return values


def verified_complete(out, name, identity, cases):
    report_file = out / (name + '.json')
    if not report_file.exists():
        return False
    report = json.loads(report_file.read_text())
    if (report.get('status') != 'passed' or report.get('identity') != identity
            or report.get('samples') != len(cases)):
        raise ValueError('Existing report is incomplete or belongs to a different run: ' + name)
    for key in ['references', 'private_samples']:
        artifact = report[key]
        file = out / artifact['file']
        if file.parent != out or not file.is_file() or sha(file.read_bytes()) != artifact['sha256']:
            raise ValueError('Completed model has missing or changed private evidence: ' + name)
    references = json.loads((out/report['references']['file']).read_text())
    if (references.get('identity') != identity
            or [item['id'] for item in references.get('cases', [])] != [item['id'] for item in cases]):
        raise ValueError('Completed reference identity/order differs: ' + name)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runtime-version', required=True)
    parser.add_argument('--models', help='Comma-separated fixture model aliases')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    fixtures_file = args.fixtures.resolve()
    fixture_bytes = fixtures_file.read_bytes()
    fixture = json.loads(fixture_bytes)
    if fixture.get('format') != 'unaveraged-digit-evaluation-fixtures/v1':
        raise ValueError('Unexpected Digit fixture format')
    cases = fixture['cases']
    if len(cases) != fixture['samples'] or len(cases) != 2000:
        raise ValueError('This qualification requires all 2000 heldout records')
    if len({item['id'] for item in cases}) != len(cases):
        raise ValueError('Duplicate evaluation case ID')
    inputs_digest, labels_digest = hashlib.sha256(), hashlib.sha256()
    for item in cases:
        inputs_digest.update(bytes.fromhex(item['sha256']))
        labels_digest.update(json.dumps(item['record'], sort_keys=True).encode())
        if (not isinstance(item['question'], str) or not isinstance(item['answer'], str)
                or any(not isinstance(item['record'][key], str) for key in ['phone', 'street'])):
            raise ValueError('Invalid evaluation labels')
    if (inputs_digest.hexdigest() != fixture['ordered_inputs_sha256']
            or labels_digest.hexdigest() != fixture['ordered_labels_sha256']):
        raise ValueError('Ordered corpus identity mismatch')
    selected = args.models.split(',') if args.models else list(fixture['models'])
    if (not selected or len(set(selected)) != len(selected)
            or any(name not in fixture['models'] or not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in selected)):
        raise ValueError('Unknown, repeated or unsafe model alias')
    source = ROOT/'.cache/digit/huggingface'/fixture['model_revision']/'question_router.py'
    if sha(source.read_bytes()) != fixture['question_router_sha256']:
        raise ValueError('Pinned question router changed')
    spec = importlib.util.spec_from_file_location('digit_evaluation_router', source)
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)
    vx, runtime = installed_runtime(args.runtime_version)
    caller_sha = sha(Path(__file__).read_bytes())
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    condition = {'backend': 'cpu', 'cpu_threads': 1, 'batch_size': 1,
                 'warmups': 4, 'warmup_input': 'First ordered heldout input',
                 'samples_per_model': len(cases), 'repeats_per_input': 1,
                 'order': 'Sequential models in fixture order; all ordered heldout inputs once per model',
                 'timer_includes': ['InferenceSession.run, input/output transport and owned output arrays'],
                 'timer_excludes': ['input file read', 'preprocessing', 'hashing', 'scoring',
                                    'loading/compilation', 'warmups', 'browser/UI'],
                 'p95_method': 'Linear interpolation over 2000 warm execution samples',
                 'host_idle_claimed': False}
    for name in selected:
        model = fixture['models'][name]
        directory = Path(model['directory']).resolve()
        if model['packed_arithmetic'] not in ['f32', 'dynamic-i8']:
            raise ValueError('Unknown packed arithmetic: ' + name)
        for file, expected in model['sha256'].items():
            if file not in ['graph.json', 'model.safetensors'] or sha((directory/file).read_bytes()) != expected:
                raise ValueError('Frozen model artifact changed: ' + name)
        if set(model['sha256']) != {'graph.json', 'model.safetensors'}:
            raise ValueError('Model must identify graph and weights')
        identity = {'fixture_sha256': sha(fixture_bytes), 'runtime': runtime,
                    'model_sha256': model['sha256'], 'packed_arithmetic': model['packed_arithmetic'],
                    'caller_sha256': caller_sha, 'question_router_sha256': fixture['question_router_sha256'],
                    'conditions': condition}
        if args.resume and verified_complete(args.out, name, identity, cases):
            print('NATIVE_SKIP', name, 'verified completed model', flush=True)
            continue
        if (args.out/(name+'.json')).exists():
            raise ValueError('Refusing to overwrite a report; use --resume or a new output directory')
        references, timings, private = [], [], []
        counts = {'record_correct': 0, 'phone_correct': 0, 'street_correct': 0, 'answer_correct': 0}
        started = datetime.now(timezone.utc).isoformat()
        with vx.InferenceSession(directory, backend='cpu', cpu_threads=1,
                                 packed_arithmetic=model['packed_arithmetic']) as session:
            arithmetic = (vx.pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8
                          if model['packed_arithmetic'] == 'dynamic-i8'
                          else vx.pb.PackedArithmetic.PACKED_ARITHMETIC_F32)
            if session.backend != 'cpu' or session.packed_arithmetic != arithmetic:
                raise RuntimeError('Compiled backend/arithmetic differs from the requested execution')
            warm = load_input(fixtures_file.parent, cases[0])
            warm_signature = None
            for _ in range(4):
                logits = session.run({'input0': warm})['slot_logits']
                decode(logits)
                signature = sha(logits.tobytes())
                if warm_signature is not None and warm_signature != signature:
                    raise ValueError('Warmup output changed')
                warm_signature = signature
            for index, item in enumerate(cases):
                values = load_input(fixtures_file.parent, item)
                before = time.perf_counter_ns()
                logits = session.run({'input0': values})['slot_logits']
                elapsed = (time.perf_counter_ns()-before)/1e6
                record, slots = decode(logits)
                signature = sha(logits.tobytes())
                if index == 0 and signature != warm_signature:
                    raise ValueError('First measured output differs from its warmups')
                good = record == item['record']
                counts['record_correct'] += good
                counts['phone_correct'] += record['phone'] == item['record']['phone']
                counts['street_correct'] += record['street'] == item['record']['street']
                answer_correct = str(router.answer_from_record(item['question'], **record)) == item['answer']
                counts['answer_correct'] += answer_correct
                references.append({'id': item['id'], 'values': logits.reshape(-1).tolist(),
                                   'slots': slots, 'sha256': signature})
                timings.append(elapsed)
                private.append({'id': item['id'], 'input_sha256': item['sha256'], 'record': record,
                                'record_correct': good, 'answer_correct': answer_correct, 'inference_ms': elapsed})
                if (index+1) % 100 == 0:
                    print('NATIVE', name, index+1, '/', len(cases), 'records', counts['record_correct'], flush=True)
        # Refuse to commit a successful result if files changed during the run.
        if sha(fixtures_file.read_bytes()) != identity['fixture_sha256'] or sha(Path(__file__).read_bytes()) != caller_sha:
            raise ValueError('Fixture or benchmark source changed during evaluation')
        _, after_runtime = installed_runtime(args.runtime_version)
        if runtime != after_runtime or any(sha((directory/file).read_bytes()) != value for file,value in model['sha256'].items()):
            raise ValueError('Runtime or model changed during evaluation')
        ref_file = args.out/(name+'.references.private.json')
        private_file = args.out/(name+'.samples.private.json')
        write_json(ref_file, {'format': 'unaveraged-digit-native-references/v1', 'status': 'passed',
                             'private': True, 'do_not_publish': True, 'model': name,
                             'model_sha256': model['sha256'], 'runtime': runtime,
                             'fixture_sha256': identity['fixture_sha256'],
                             'ordered_inputs_sha256': fixture['ordered_inputs_sha256'],
                             'ordered_labels_sha256': fixture['ordered_labels_sha256'],
                             'identity': identity, 'cases': references})
        write_json(private_file, {'private': True, 'do_not_publish': True, 'identity': identity, 'cases': private})
        report = {'format': 'unaveraged-digit-native-evaluation/v1', 'status': 'passed',
                  'created_at': started, 'completed_at': datetime.now(timezone.utc).isoformat(),
                  'identity': identity, 'model': name, 'method': model['method'],
                  'runtime': runtime, 'graph_weight_bytes': model['graph_weight_bytes'],
                  'sha256': model['sha256'], 'samples': len(cases), 'counts': counts,
                  'accuracy_percent': {key: 100*value/len(cases) for key,value in counts.items()},
                  'latency': latency_stats(timings), 'conditions': condition,
                  'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                  'cpu': next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                               if line.startswith('model name')), 'unknown')},
                  'evaluation': {key: fixture[key] for key in ['split', 'samples', 'ordered_inputs_sha256',
                      'ordered_labels_sha256', 'ordered_image_files_sha256', 'ordered_annotation_files_sha256', 'annotation_caveat']},
                  'references': {'file': ref_file.name, 'sha256': sha(ref_file.read_bytes())},
                  'private_samples': {'file': private_file.name, 'sha256': sha(private_file.read_bytes())}}
        write_json(args.out/(name+'.json'), report)
        print('NATIVE_COMPLETE', name, counts['record_correct'], '/', len(cases),
              'median_ms', report['latency']['median_ms'], flush=True)


if __name__ == '__main__':
    main()
