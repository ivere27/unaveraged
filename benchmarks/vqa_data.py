"""Private VQA annotation loading, split isolation and product answer metrics.

The supplied real corpus labels phone-digit and street-number questions. Its
scores do not establish accuracy for store, item or arithmetic questions. This
module never writes annotations or includes record contents in aggregate audits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata


TARGET_SCHEMA = 'direct_answer_no_value_v1'
SCOPE = 'Existing real phone-digit and street-number annotations; no other-family accuracy claim.'
_STRUCTURED = re.compile(
    r'<field>([^<>]+)</field>(?:<value>([^<>]*)</value>)?'
    r'<op>([^<>]+)</op><answer>([^<>]*)</answer>')


def clean_text(value):
    return unicodedata.normalize('NFC', re.sub(r'\s+', ' ', str(value if value is not None else '')).strip())


def phone_operation(question):
    """Match the pinned model producer's real-question operation labels."""
    question = clean_text(question).lower()
    ordinals = {name: i for i, name in enumerate(
        ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'), 1)}
    pattern = '(' + '|'.join(ordinals) + ')'
    operation = 'phone_digit'
    match = re.search(pattern, question)
    if match:
        operation = f'front_{ordinals[match[1]]}'
    for side, expression in (
        ('front', r'(?:front of|from the front|digit)\D*(\d+)'),
        ('back', r'(?:from the end|from the back|from the right|last)\D*(\d+)'),
        ('front', r'앞에서\s*(\d+)번째'),
        ('front', r'(?:앞|앞자리)\s*(\d+)(?:번째|번)?'),
        ('back', r'뒤에서\s*(\d+)번째'),
        ('back', r'(?:뒤|뒷자리|끝자리)\s*(\d+)(?:번째|번)?'),
    ):
        match = re.search(expression, question)
        if match:
            operation = f'{side}_{match[1]}'
    if any(text in question for text in ('from the back', 'from the end', 'from last')):
        match = re.search(pattern, question)
        if match:
            operation = f'back_{ordinals[match[1]]}'
    return operation


def structured_target(field, value, operation, answer):
    """Current producer schema: direct-read operations omit repeated value."""
    value, operation = clean_text(value), clean_text(operation)
    direct = operation in {'identity', 'copy', 'count'} or re.fullmatch(
        r'item_\d+_(?:name|unit_price|quantity|total)', operation)
    evidence = f'<value>{value}</value>' if value and not direct else ''
    return f'<field>{clean_text(field)}</field>{evidence}<op>{operation}</op><answer>{clean_text(answer)}</answer>'


def load_records(root):
    """Read supported annotations without changing labels or their image paths."""
    root = Path(root).resolve()
    annotation_root, image_root = root / 'annotations', root / 'images'
    if not annotation_root.is_dir() or not image_root.is_dir():
        raise ValueError('Dataset requires annotations/ and images/ directories')
    records, skipped = [], Counter()
    paths = sorted(annotation_root.glob('*.json'))
    for path in paths:
        annotation = json.loads(path.read_text(encoding='utf-8'))
        image = image_root / f'{path.stem}.jpg'
        if not image.is_file():
            skipped['missing_image'] += 1
            continue
        question, answer = (clean_text(annotation.get(name)) for name in ('question', 'answer'))
        if not question or not answer:
            skipped['missing_question_or_answer'] += 1
            continue
        lower = question.lower()
        store = (annotation.get('receipt') or {}).get('store') or {}
        if 'phone number' in lower or '전화번호' in lower:
            family, field, kind, task = 'phone', 'phone', 'phone_number', 'phone_digit'
            value = ''.join(re.findall(r'\d', str(store.get('phone') or '')))
            operation = phone_operation(question)
        elif any(text in lower for text in ('location', 'address', '가게 위치', '빈 칸', 'blank')):
            family, field, kind, task = 'address', 'addr', 'address', 'address_number'
            value = re.sub(r'(\d+(?:\s+\d+)+)$', lambda m: re.sub(r'\s+', '', m[1]),
                           clean_text(store.get('address')))
            operation = 'street_no'
        else:
            skipped['unsupported_question'] += 1
            continue
        records.append({
            'id': str(annotation.get('id') or path.stem), 'receipt_id': path.stem,
            'image': image, 'question': question, 'answer': answer,
            'target': structured_target(field, value, operation, answer),
            'family': family, 'field': field, 'op': operation, 'kind': kind,
            'task': task, 'source': 'real',
            'language': 'ko' if re.search(r'[가-힣]', question) else 'en',
        })
    labels = [{key: record[key] for key in ('id', 'question', 'answer', 'target', 'family')}
              for record in records]
    summary = {
        'annotations': len(paths), 'records': len(records), 'skipped': dict(skipped),
        'by_family': dict(Counter(r['family'] for r in records)),
        'by_language': dict(Counter(r['language'] for r in records)),
        'target_schema': TARGET_SCHEMA, 'scope': SCOPE,
        'ordered_labels_sha256': hashlib.sha256(json.dumps(
            labels, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
    }
    return records, summary


def select_records(records, limit=None, seed=0):
    """Deterministic proportional family/language sample, independent of input order."""
    records = list(records)
    if limit is None or limit >= len(records):
        return sorted(records, key=lambda r: (str(r.get('receipt_id', r['id'])), str(r['image']), r['question']))
    if limit < 1:
        raise ValueError('Sample limit must be positive')
    groups = defaultdict(list)
    for record in records:
        groups[(record['family'], record.get('language', 'unknown'))].append(record)
    keys = sorted(groups)
    quotas = {key: limit * len(groups[key]) / len(records) for key in keys}
    counts = {key: math.floor(quotas[key]) for key in keys}
    for key in sorted(keys, key=lambda key: (-(quotas[key] - counts[key]), key))[:limit - sum(counts.values())]:
        counts[key] += 1
    rng = random.Random(seed)
    selected = []
    for key in keys:
        group = sorted(groups[key], key=lambda r: (str(r.get('receipt_id', r['id'])), str(r['image']), r['question']))
        rng.shuffle(group)
        selected.extend(group[:counts[key]])
    rng.shuffle(selected)
    return selected


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_sha256(path):
    """Hash the exact grayscale/resize/F32 normalization used by VQA inference."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        values = np.asarray(image.convert('L').resize((672, 320), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    values = (values - 0.5) / 0.5
    return hashlib.sha256(values.tobytes()).hexdigest()


def hash_records(records, *, input_hash=input_sha256):
    """Return copies with hashes; callers can cache this before repeated comparisons."""
    cache = {}
    result = []
    for record in records:
        path = Path(record['image']).resolve()
        if path not in cache:
            cache[path] = {'file_sha256': record.get('file_sha256') or file_sha256(path),
                           'input_sha256': record.get('input_sha256') or input_hash(path)}
        result.append({**record, **cache[path]})
    return result


def assert_disjoint(training, evaluation, *, input_hash=input_sha256):
    """Reject identical files AND differently encoded identical model inputs."""
    training = hash_records(training, input_hash=input_hash)
    evaluation = hash_records(evaluation, input_hash=input_hash)
    overlaps = {kind: len({r[kind] for r in training} & {r[kind] for r in evaluation})
                for kind in ('file_sha256', 'input_sha256')}
    if any(overlaps.values()):
        raise ValueError('Training/evaluation image overlap: ' + json.dumps(overlaps, sort_keys=True))
    return {'training_records': len(training), 'evaluation_records': len(evaluation),
            'overlap': overlaps}


def split_train_validation(records, validation_fraction=0.1, seed=0, *, input_hash=input_sha256):
    """Split by model-input identity, so duplicate receipt records stay together."""
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    records = hash_records(records, input_hash=input_hash)
    groups = defaultdict(list)
    for record in records:
        groups[record['input_sha256']].append(record)
    if len(groups) < 2:
        raise ValueError('At least two distinct model inputs are required')
    representatives = [sorted(group, key=lambda r: (r['id'], r['question']))[0] for group in groups.values()]
    count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    selected = select_records(representatives, count, seed)
    validation_hashes = {r['input_sha256'] for r in selected}
    training = [r for r in records if r['input_sha256'] not in validation_hashes]
    validation = [r for r in records if r['input_sha256'] in validation_hashes]
    summary = {'seed': seed, 'requested_validation_fraction': validation_fraction,
               'training_records': len(training), 'validation_records': len(validation),
               'training_unique_inputs': len(groups) - count, 'validation_unique_inputs': count,
               'duplicate_input_records': len(records) - len(groups), 'input_overlap': 0}
    return training, validation, summary


def prepare_calibration(training, evaluation, count=128, seed=0, *, input_hash=input_sha256):
    """Select unique train images after excluding every supplied evaluation input.

    Pass only the training partition here when QAT validation is reserved.
    Do not pass a sampled evaluation subset: exclusion must cover the full split.
    """
    if count < 1:
        raise ValueError('Calibration count must be positive')
    training = hash_records(training, input_hash=input_hash)
    evaluation = hash_records(evaluation, input_hash=input_hash)
    excluded_files = {r['file_sha256'] for r in evaluation}
    excluded_inputs = {r['input_sha256'] for r in evaluation}
    candidates, seen, excluded, duplicates = [], set(), 0, 0
    for record in sorted(training, key=lambda r: (r['id'], r['question'])):
        identity = record['input_sha256']
        if record['file_sha256'] in excluded_files or identity in excluded_inputs:
            excluded += 1
        elif identity in seen:
            duplicates += 1
        else:
            seen.add(identity)
            candidates.append(record)
    if len(candidates) < count:
        raise ValueError(f'Only {len(candidates)} isolated unique calibration images; requested {count}')
    selected = select_records(candidates, count, seed)
    result = [{'path': r['image'], 'question': r['question'], 'input_sha256': r['input_sha256'],
               'file_sha256': r['file_sha256']} for r in selected]
    audit = {'seed': seed, 'requested': count, 'selected': len(result),
             'source_records': len(training), 'excluded_evaluation_records': len(evaluation),
             'excluded_overlapping_records': excluded, 'deduplicated_training_records': duplicates,
             'by_family': dict(Counter(r['family'] for r in selected)),
             'by_language': dict(Counter(r.get('language', 'unknown') for r in selected)),
             'ordered_inputs_sha256': hashlib.sha256(''.join(r['input_sha256'] for r in selected).encode()).hexdigest()}
    return result, audit


def parse_generated(text):
    """Same full structured-output acceptance contract as vqa/adapter.js."""
    match = _STRUCTURED.fullmatch(text)
    if match is None:
        raise ValueError('Malformed VQA structured output')
    return {'field': match[1], 'value': match[2], 'op': match[3], 'answer': clean_text(match[4])}


def score_prediction(record, generated, finished=True, *, family=None):
    """An unfinished or malformed generation is always an incorrect answer."""
    try:
        parsed = parse_generated(generated) if finished else None
    except ValueError:
        parsed = None
    answer = parsed['answer'] if parsed else ''
    return {'finished': bool(finished), 'structured_valid': parsed is not None,
            'answer': answer, 'answer_correct': parsed is not None and answer == clean_text(record['answer']),
            'target_correct': parsed is not None and clean_text(generated) == clean_text(record['target']),
            'field_correct': parsed is not None and parsed['field'] == record['field'],
            'op_correct': parsed is not None and parsed['op'] == record['op'],
            'family_correct': family == record['family'] if family is not None else None}


def prepare_data_manifest(training_root, evaluation_root, *, calibration_count=128,
                          validation_fraction=0.1, seed=20261026, excluded_images=()):
    """Prepare private fixed partitions, reserving public examples as well."""
    source, source_audit = load_records(training_root)
    evaluation, evaluation_audit = load_records(evaluation_root)
    source, evaluation = hash_records(source), hash_records(evaluation)
    disjoint = assert_disjoint(source, evaluation)
    extra = hash_records([{'image': Path(path)} for path in excluded_images])
    excluded_inputs = {r['input_sha256'] for r in extra}
    source_before = len(source)
    source = [r for r in source if r['input_sha256'] not in excluded_inputs]
    train, validation, split_audit = split_train_validation(source, validation_fraction, seed)
    calibration, calibration_audit = prepare_calibration(train, evaluation + extra, calibration_count, seed)
    calibration_hashes = {r['input_sha256'] for r in calibration}
    training = [r for r in train if r['input_sha256'] not in calibration_hashes]
    calibration_records = [{'image': r['path'], **r} for r in calibration]
    assert_disjoint(training, validation)
    assert_disjoint(training, calibration_records)
    assert_disjoint(validation, calibration_records)
    audit = {
        'source': source_audit, 'evaluation': evaluation_audit, 'split': split_audit,
        'calibration': calibration_audit, 'source_vs_heldout': disjoint, 'scope': SCOPE,
        'training_records': len(training), 'validation_records': len(validation),
        'calibration_images': len(calibration), 'evaluation_records': len(evaluation),
        'training_unique_inputs': len({r['input_sha256'] for r in training}),
        'validation_unique_inputs': len({r['input_sha256'] for r in validation}),
        'evaluation_unique_inputs': len({r['input_sha256'] for r in evaluation}),
        'additional_excluded_images': len(extra), 'source_records_excluded_as_public': source_before - len(source),
        'evaluation_used_for_training_or_calibration': False,
        'label_caveat': 'Provided annotations are not a fully human-audited gold dataset; this heldout set has prior development/evaluation use.',
    }
    return {
        'format': 'unaveraged-vqa-private-data/v1', 'private': True, 'do_not_publish': True,
        'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed,
        'source_root': str(Path(training_root).resolve()), 'evaluation_root': str(Path(evaluation_root).resolve()),
        'calibration': calibration, 'validation': validation, 'evaluation': evaluation,
        'training': training, 'audit': audit,
        'excluded_input_sha256': sorted({r['input_sha256'] for r in evaluation + extra}),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--public-examples-dir', type=Path)
    parser.add_argument('--calibration-count', type=int, default=128)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output path; existing manifests are immutable')
    examples = [] if args.public_examples_dir is None else sorted(
        path for path in args.public_examples_dir.iterdir()
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    document = prepare_data_manifest(args.training_root, args.evaluation_root,
        calibration_count=args.calibration_count, validation_fraction=args.validation_fraction,
        seed=args.seed, excluded_images=examples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2, default=str) + '\n')
    print(json.dumps(document['audit'], ensure_ascii=False))


if __name__ == '__main__':
    main()
