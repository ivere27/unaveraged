#!/usr/bin/env python3
"""Public-example native references and fixed-prefix VQA browser fixtures."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.vqa_lowbit import ExperimentRunner, identity
from benchmarks.vqa_data import parse_generated
from shared.model_support import preprocess, sha256, write_json
from vqa.model import generate, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'.cache/vqa/huggingface'/
        json.loads((ROOT/'vqa/extension.json').read_text())['model']['revision'])
    # Measurements use explicitly chosen, fixed model bytes. Do not silently
    # substitute a fresh build or require unrelated W2/W3 experiment exports.
    for profile in ('fp32', 'w8a8', 'w4a8'):
        parser.add_argument(f'--{profile}-package', type=Path, required=True,
                            help='Exact model package to measure; recorded identities are retained in the fixture')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    packages = {'f32': args.fp32_package, 'w8a8': args.w8a8_package,
                'w4a8': args.w4a8_package}
    identities = {name: identity(package) for name, package in packages.items()}
    tokenizer = load_tokenizer(args.source)
    examples = json.loads((args.source/'examples/family_examples.json').read_text())['examples']
    records = []
    for example in examples:
        for index, case in enumerate([*example['cases'], *example.get('additional_cases', [])]):
            records.append({'id': f'{example["language"]}-{index}', 'image': example['image'],
                            'question': case['question'], 'expected_answer': str(case['expected_answer'])})
    selected = []
    for keyword in ('phone number', 'store address', 'store name'):
        candidates = [r for r in records if keyword in r['question'].lower()]
        if not candidates:
            raise ValueError(f'Public example for {keyword} missing')
        selected.append(candidates[0])
    report = {'format': 'unaveraged-vqa-public-evaluation/v1', 'batch_size': 1,
              'backend': 'cpu', 'cpu_threads': 1, 'models': {}, 'cases': []}
    fixture = {'format': 'unaveraged-vqa-wasm-fixtures/v1', 'models':
        {name: {'directory': str(path.resolve()),
                'sha256': {file: details['sha256'] for file, details in identities[name]['files'].items()}}
         for name, path in packages.items()}, 'cases': []}
    for index, case in enumerate(selected):
        raw = preprocess(args.source/'examples'/case['image']).tobytes()
        filename = f'input-{index}.f32'
        (args.out/filename).write_bytes(raw)
        fixture['cases'].append({'id': case['id'], 'file': filename, 'sha256': sha256(raw),
                                 'question': case['question'], 'expected_answer': case['expected_answer'],
                                 'references': {}})
    for name, package in packages.items():
        runner = ExperimentRunner(package)
        successes = 0
        try:
            for record in records:
                image = preprocess(args.source/'examples'/record['image'])
                tokens = tokenizer.encode(record['question'], add_eos=True)
                prediction = generate(runner, image, tokens)
                text = tokenizer.decode(prediction['tokens'], errors='strict')
                finished = prediction['tokens'][-1] == 2
                try:
                    parsed = parse_generated(text) if finished else None
                except ValueError:
                    parsed = None
                correct = bool(parsed and parsed['answer'] == record['expected_answer'])
                successes += correct
                row = {**record, 'profile': name, 'tokens': prediction['tokens'],
                       'finished': finished, 'structured_valid': parsed is not None,
                       'answer_correct': correct, 'generated': text}
                report['cases'].append(row)
                for case in fixture['cases']:
                    if case['id'] == record['id']:
                        case['references'][name] = {'tokens': prediction['tokens']}
                        if name == 'f32':
                            case['decoder_prefix'] = prediction['tokens']
            if identity(package) != identities[name]:
                raise ValueError(f'Model package changed during fixture generation: {name}')
            report['models'][name] = {'correct': successes, 'samples': len(records),
                                      'identity': identities[name]}
            print('VQA_PUBLIC', name, successes, '/', len(records), flush=True)
            write_json(args.out/'public-native.json', report)
            write_json(args.out/'fixtures.json', fixture)
        finally:
            runner.close()
    report['status'] = 'completed'
    write_json(args.out/'public-native.json', report)


if __name__ == '__main__':
    main()
