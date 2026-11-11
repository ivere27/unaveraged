#!/usr/bin/env python3
"""Interleave native Digit models using fixed heldout inputs and exact wheel."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import time
import numpy as np
from digit_native_eval import installed_runtime, load_input, decode, latency_stats, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--runtime-version', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.fixtures = args.fixtures.resolve()
    fixture_bytes = args.fixtures.read_bytes()
    fixture = json.loads(fixture_bytes)
    if (fixture.get('format') != 'unaveraged-digit-latency-fixtures/v1'
            or fixture.get('source_indices') != [0, 999, 1999]
            or len(fixture.get('cases', [])) != 3 or not fixture.get('models')
            or fixture.get('cycles') != 15 or fixture.get('warmups') != 4):
        raise ValueError('Unexpected fixed interleaved latency protocol')
    if args.out.exists():
        raise ValueError('Refusing to overwrite an existing latency report')
    vx, runtime = installed_runtime(args.runtime_version)
    if runtime != fixture['native_reference_runtime']:
        raise ValueError('Selected wheel differs from full-corpus native references')
    sources = {name: sha((ROOT/name).read_bytes()) for name in
               ['benchmarks/digit_latency.py', 'benchmarks/digit_native_eval.py']}
    inputs = [load_input(args.fixtures.parent, item) for item in fixture['cases']]
    protocol = {'cycles': 15, 'warmups_per_input': 4, 'initial_reference_executions_per_input': 1,
                'input_count': 3, 'samples_per_model': 45, 'batch_size': 1, 'cpu_threads': 1,
                'order': 'Alternate forward/reverse model order each cycle, for each fixed input',
                'input_selection': fixture['selection'], 'host_idle_claimed': False,
                'timer_includes': ['InferenceSession.run, input/output transport and owned output arrays'],
                'timer_excludes': ['input reads', 'hashing', 'preprocessing', 'loading/compilation',
                                   'warmups', 'numerical validation'],
                'timing_scope': 'Interleaved repeated model timings; separate from full-corpus accuracy and sequential evaluation-pass latency'}
    report = {'format': 'unaveraged-digit-interleaved-latency/v1', 'status': 'running',
              'engine': 'native-cpu', 'created_at': datetime.now(timezone.utc).isoformat(),
              'fixture_sha256': sha(fixture_bytes), 'source_fixture_sha256': fixture['source_fixture_sha256'],
              'sources_sha256': sources, 'source_references_sha256': fixture['source_references_sha256'],
              'input_indices': fixture['source_indices'], 'input_sha256': [x['sha256'] for x in fixture['cases']],
              'runtime': runtime, 'protocol': protocol, 'models': {}}
    report['environment'] = {'python': platform.python_version(), 'platform': platform.platform(),
        'hostname': socket.gethostname(), 'logical_cpus': os.cpu_count(), 'numpy': np.__version__,
        'cpu': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                     if line.startswith('model name')), 'unknown')}
    report['host_observations'] = [{'phase': 'before', 'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'loadavg': os.getloadavg()}]
    try:
        with ExitStack() as stack:
            sessions = {}
            signatures = {}
            for name, model in fixture['models'].items():
                directory = (args.fixtures.parent / model['directory']).resolve()
                if set(model['sha256']) != {'graph.json', 'model.safetensors'}:
                    raise ValueError('Missing graph/weight identities')
                if any(sha((directory/file).read_bytes()) != digest for file, digest in model['sha256'].items()):
                    raise ValueError('Frozen model artifact differs: ' + name)
                session = stack.enter_context(vx.InferenceSession(directory, backend='cpu', cpu_threads=1,
                    packed_arithmetic=model['packed_arithmetic']))
                expected = (vx.pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8 if model['packed_arithmetic'] == 'dynamic-i8'
                            else vx.pb.PackedArithmetic.PACKED_ARITHMETIC_F32)
                if session.backend != 'cpu' or session.packed_arithmetic != expected:
                    raise ValueError('Compiled backend/arithmetic differs')
                sessions[name] = session
                signatures[name] = []
                report['models'][name] = {'sha256': model['sha256'], 'packed_arithmetic': model['packed_arithmetic'],
                    'samples_ms_per_input': [[], [], []], 'native_reference_bytes_equal': True,
                    'native_reference_slots_equal': True, 'native_reference_logits_within_tolerance': True,
                    'maximum_absolute_logit_difference': 0.0, 'parity': []}
                for index, values in enumerate(inputs):
                    reference = fixture['cases'][index]['references'][name]
                    reference_array = np.asarray(reference['values'], dtype=np.float32).reshape(1, 16, 11)
                    _, slots = decode(reference_array)
                    if sha(reference_array.tobytes()) != reference['sha256'] or slots != reference['slots']:
                        raise ValueError('Invalid full-corpus native reference')
                    for repeat in range(1 + fixture['warmups']):
                        output = session.run({'input0': values})['slot_logits']
                        _, output_slots = decode(output)
                        output_sha = sha(output.tobytes())
                        if repeat == 0:
                            signatures[name].append(output_sha)
                            parity = {'input_index': index, 'bytes_equal': output_sha == reference['sha256'],
                                'slots_equal': output_slots == reference['slots'],
                                'logits_within_tolerance': bool(np.allclose(output, reference_array,
                                    rtol=fixture['rtol'], atol=fixture['atol'])),
                                'maximum_absolute_logit_difference': float(np.max(np.abs(output-reference_array))),
                                'reference_sha256': reference['sha256'], 'output_sha256': output_sha}
                            evidence = report['models'][name]
                            evidence['parity'].append(parity)
                            evidence['native_reference_bytes_equal'] &= parity['bytes_equal']
                            evidence['native_reference_slots_equal'] &= parity['slots_equal']
                            evidence['native_reference_logits_within_tolerance'] &= parity['logits_within_tolerance']
                            evidence['maximum_absolute_logit_difference'] = max(evidence['maximum_absolute_logit_difference'],
                                parity['maximum_absolute_logit_difference'])
                        elif output_sha != signatures[name][index]:
                            raise ValueError('Warmup changed native output')
            names = list(sessions)
            for cycle in range(fixture['cycles']):
                for index, values in enumerate(inputs):
                    for name in names if cycle % 2 == 0 else list(reversed(names)):
                        before = time.perf_counter_ns()
                        output = sessions[name].run({'input0': values})['slot_logits']
                        elapsed = (time.perf_counter_ns()-before)/1e6
                        report['models'][name]['samples_ms_per_input'][index].append(elapsed)
                        decode(output)
                        if sha(output.tobytes()) != signatures[name][index]:
                            raise ValueError('Repeated native output changed')
            for model in report['models'].values():
                stats = latency_stats(sum(model['samples_ms_per_input'], []))
                if stats['samples'] != 45:
                    raise ValueError('Incomplete interleaved samples')
                model.update(pooled_median_ms=stats['median_ms'], mean_ms=stats['mean_ms'], p95_ms=stats['p95_ms'])
        if any(sha((ROOT/name).read_bytes()) != digest for name, digest in sources.items()):
            raise ValueError('Benchmark source changed during timing')
        if sha(args.fixtures.read_bytes()) != report['fixture_sha256']:
            raise ValueError('Fixture changed during timing')
        _, after_runtime = installed_runtime(args.runtime_version)
        if runtime != after_runtime:
            raise ValueError('Runtime changed during timing')
        for model in fixture['models'].values():
            if any(sha((args.fixtures.parent/model['directory']/file).read_bytes()) != digest for file, digest in model['sha256'].items()):
                raise ValueError('Model changed during timing')
        report['slot_parity_passed'] = all(model['native_reference_slots_equal'] for model in report['models'].values())
        report['numerical_parity_passed'] = report['slot_parity_passed'] and all(
            model['native_reference_logits_within_tolerance'] for model in report['models'].values())
        report['tolerance'] = {'rtol': fixture['rtol'], 'atol': fixture['atol']}
        report['status'] = 'passed' if report['numerical_parity_passed'] else 'completed-with-numerical-differences'
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        report['host_observations'].append({'phase': 'after', 'timestamp': datetime.now(timezone.utc).isoformat(),
                                          'loadavg': os.getloadavg()})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
    print('NATIVE_INTERLEAVED', report['status'], {name: model['pooled_median_ms'] for name, model in report['models'].items()}, flush=True)


if __name__ == '__main__':
    main()
