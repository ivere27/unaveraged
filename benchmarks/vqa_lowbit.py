#!/usr/bin/env python3
"""Batch-one VQA low-bit experiments. Private examples stay in the local cache."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np
import volvoxai as vx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.model_support import packaged_library_paths, preprocess, sha256, write_json
from vqa.model import (Runner, generate, load_tokenizer, package_model_files,
                       role_weight_paths, CROSS, encoder_feed, decoder_feed, advance)


class ExperimentRunner(Runner):
    """Explicit experimental profiles without changing release precision contracts."""
    def __init__(self, package, *, cpu_threads=1, backend='cpu'):
        package = Path(package)
        self.manifest = json.loads((package / 'manifest.json').read_text())
        self.contracts = self.manifest['graphs']
        self.sessions = {}
        self.timings = defaultdict(list)
        arithmetic = self.manifest.get('quantization', {}).get('packed_arithmetic', 'f32')
        try:
            for role in ('encoder', 'decoder'):
                session = vx.InferenceSession(package / role / 'graph.json',
                    weights=role_weight_paths(package, role, self.manifest),
                    backend=backend, cpu_threads=cpu_threads, packed_arithmetic=arithmetic)
                self.sessions[role] = session
                if session.backend != backend:
                    raise RuntimeError('Unexpected inference backend')
                expected = (vx.pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8
                            if arithmetic == 'dynamic-i8' else vx.pb.PackedArithmetic.PACKED_ARITHMETIC_F32)
                if session.packed_arithmetic != expected:
                    raise RuntimeError('Compiled arithmetic differs from the experiment profile')
        except BaseException:
            self.close()
            raise

    def run(self, role, feed, **kwargs):
        started = time.perf_counter()
        result = super().run(role, feed, **kwargs)
        self.timings[role].append((time.perf_counter() - started) * 1000)
        return result


def identity(package):
    files = package_model_files(package)
    return {'bytes': sum(p.stat().st_size for p in files),
            'files': {str(p.relative_to(package)): {'bytes': p.stat().st_size,
                      'sha256': sha256(p.read_bytes())} for p in files}}


def write_report(path, report):
    """An interrupted progress write must not destroy the resume identity."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_checkpoint(path):
    """Keep committed JSONL rows, discarding only an interrupted final write."""
    path = Path(path)
    data = path.read_bytes()
    committed = data.rfind(b'\n') + 1
    # Every writer appends a newline with its row. A tail without one has not
    # committed, even if its truncated bytes happen to form valid JSON.
    rows = [json.loads(line) for line in data[:committed].splitlines() if line]
    if committed != len(data):
        with path.open('r+b') as stream:
            stream.truncate(committed)
    return rows


def verify_resume(previous, report):
    keys = ('model', 'ordered_cases_sha256', 'runtime_sha256', 'tokenizer_sha256',
            'backend', 'cpu_threads', 'batch_size', 'max_decoder_steps',
            'profile', 'split', 'shard_index', 'shards')
    changed = [key for key in keys if previous.get(key) != report.get(key)]
    if changed:
        raise ValueError('Existing report identities differ (' + ', '.join(changed)
                         + '); choose a new output directory')


def summarize(rows):
    groups = {}
    for family in ['all', *sorted({r['family'] for r in rows})]:
        selected = rows if family == 'all' else [r for r in rows if r['family'] == family]
        counts = {key: sum(bool(r.get(key)) for r in selected) for key in
                  ('answer_correct', 'structured_valid', 'target_correct', 'field_correct',
                   'op_correct', 'finished', 'error')}
        groups[family] = {'samples': len(selected), **counts,
            'answer_accuracy_percent': 100 * counts['answer_correct'] / len(selected),
            'generation_median_ms': statistics.median(r['generation_ms'] for r in selected),
            'encoder_median_ms': statistics.median(r['encoder_ms'] for r in selected),
            'decoder_total_median_ms': statistics.median(r['decoder_ms'] for r in selected),
            'generated_tokens_median': statistics.median(r['steps'] for r in selected)}
    return groups


def evaluate(args):
    from benchmarks.vqa_data import score_prediction
    manifest = json.loads(args.data.read_text())
    cases = manifest[args.split]
    if args.limit:
        from benchmarks.vqa_data import select_records
        cases = select_records(cases, args.limit, args.seed)
    cases = cases[args.shard_index::args.shards]
    if not cases:
        raise ValueError('Empty evaluation shard')
    tokenizer = load_tokenizer(args.source)
    model_id = identity(args.package)
    if sha256((args.source / 'vocab.json').read_bytes()) != sha256((args.package / 'vocab.json').read_bytes()):
        raise ValueError('Evaluation tokenizer differs from the model vocabulary')
    case_fingerprint = sha256(json.dumps(cases, sort_keys=True).encode())
    args.out.mkdir(parents=True, exist_ok=True)
    private_path = args.out / f'{args.name}-{args.split}-{args.shard_index}.private.jsonl'
    report_path = args.out / f'{args.name}-{args.split}-{args.shard_index}.json'
    report = {'format': 'unaveraged-vqa-lowbit-evaluation/v1', 'status': 'running',
        'profile': args.name, 'split': args.split, 'batch_size': 1, 'backend': args.backend,
        'cpu_threads': args.threads, 'shard_index': args.shard_index, 'shards': args.shards,
        'model': model_id, 'ordered_cases_sha256': case_fingerprint,
        'tokenizer_sha256': {name: sha256((args.source / name).read_bytes())
                             for name in ('vocab.json', 'byte_bpe.py')},
        'dataset_audit': manifest['audit'], 'max_decoder_steps': 191,
        'scope': 'Phone/address questions only; exact adapter answer; no post-hoc arithmetic repair.',
        'timing_scope': 'Diagnostic generation times during accuracy evaluation; host may be busy. '
                        'Excludes preprocessing/load/compile. Use separate isolated WASM timing for performance.',
        'runtime_sha256': {p.name: sha256(p.read_bytes()) for p in packaged_library_paths()}}
    rows, previous_elapsed = [], 0.0
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        verify_resume(previous, report)
        previous_elapsed = float(previous.get('elapsed_seconds', 0))
        if private_path.exists():
            rows = read_checkpoint(private_path)
        if len(rows) > len(cases) or [r['id'] for r in rows] != [r['id'] for r in cases[:len(rows)]]:
            raise ValueError('Partial results do not match selected records')
    elif private_path.exists():
        raise ValueError('Private results exist without a matching report')
    if len(rows) == len(cases):
        report.update(status='completed', completed=len(rows), metrics=summarize(rows),
                      elapsed_seconds=previous_elapsed)
        write_report(report_path, report)
        print('VQA_EVAL_ALREADY_COMPLETE', args.name, args.split, args.shard_index, len(rows), flush=True)
        return
    write_report(report_path, report)
    runner = None
    start = time.perf_counter()
    try:
        runner = ExperimentRunner(args.package, cpu_threads=args.threads, backend=args.backend)
        with private_path.open('a') as stream:
            for index, case in enumerate(cases[len(rows):], start=len(rows)):
                image = preprocess(Path(case['image']))
                if case.get('input_sha256') and sha256(image.tobytes()) != case['input_sha256']:
                    raise ValueError('Evaluation input changed')
                ids = tokenizer.encode(case['question'], add_eos=True)
                runner.timings.clear()
                begin = time.perf_counter()
                row = {'id': case['id'], 'family': case['family']}
                try:
                    prediction = generate(runner, image, ids)
                    finished = prediction['tokens'][-1] == 2
                    row.update(finished=finished, tokens=prediction['tokens'],
                               steps=len(prediction['tokens']) - 1, predicted_family=prediction['family'])
                    text = tokenizer.decode(prediction['tokens'], errors='strict')
                    row.update(score_prediction(case, text, finished, family=prediction['family']))
                    row.update(finished=finished, text=text, tokens=prediction['tokens'],
                               steps=len(prediction['tokens']) - 1, predicted_family=prediction['family'])
                except (ValueError, RuntimeError) as error:
                    row.update(answer_correct=False, structured_valid=False, target_correct=False,
                               field_correct=False, op_correct=False,
                               error={'type': type(error).__name__, 'message': str(error)},
                               finished=row.get('finished', False),
                               steps=row.get('steps', len(runner.timings['decoder'])))
                row.update(generation_ms=(time.perf_counter() - begin) * 1000,
                           encoder_ms=sum(runner.timings['encoder']), decoder_ms=sum(runner.timings['decoder']))
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                if (index + 1) % 25 == 0 or index + 1 == len(cases):
                    report.update(completed=len(rows), metrics=summarize(rows),
                                  elapsed_seconds=previous_elapsed + time.perf_counter()-start)
                    write_report(report_path, report)
                    print('VQA_EVAL', args.name, args.split, args.shard_index,
                          len(rows), '/', len(cases), 'correct', report['metrics']['all']['answer_correct'],
                          'seconds', round(time.perf_counter()-start, 1), flush=True)
        report.update(status='completed', completed=len(rows), metrics=summarize(rows))
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if runner is not None:
            runner.close()
        report['elapsed_seconds'] = previous_elapsed + time.perf_counter() - start
        write_report(report_path, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--split', choices=('validation', 'evaluation', 'training'), default='validation')
    p.add_argument('--source', type=Path, default=ROOT/'.cache/vqa/huggingface'/
        json.loads((ROOT/'vqa/extension.json').read_text())['model']['revision'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    if args.threads < 1 or args.shards < 1 or not 0 <= args.shard_index < args.shards:
        p.error('Invalid threads/shard settings')
    if args.limit is not None and args.limit < 1:
        p.error('Evaluation limit must be positive')
    evaluate(args)


if __name__ == '__main__':
    main()
