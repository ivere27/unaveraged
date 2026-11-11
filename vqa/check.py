#!/usr/bin/env python3
"""Independent source-ONNX parity and native PTQ checks on public HF examples.

ONNX Runtime is used only as the validation reference, never for PTQ or in the
extension. Run after make: .venv/bin/python vqa/check.py
"""
from pathlib import Path
import argparse
import json
import shutil
import tempfile
import time
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
from shared.model_support import VERSION, PRECISIONS, SUPPORTED_PRECISIONS, IMAGE_SUFFIXES, sha256, packaged_library_paths, package_source
from vqa.model import (Runner, load_tokenizer, preprocess, clean_text, encoder_feed, decoder_feed,
                   advance, generate, CROSS, PAST, PRESENT, write_json, package_model_files)
MODEL_REVISION = json.loads((ROOT / "vqa/extension.json").read_text())["model"]["revision"]

SOURCE = ROOT / '.cache/vqa/huggingface' / MODEL_REVISION
PACKAGES = ROOT / '.cache/vqa/packages' / VERSION


def ort_run(session, feed):
    kinds = {'tensor(int64)':np.int64, 'tensor(bool)':np.bool_, 'tensor(float)':np.float32}
    values = {item.name:np.asarray(feed[item.name],dtype=kinds[item.type]) for item in session.get_inputs()}
    return dict(zip([item.name for item in session.get_outputs()],session.run(None,values)))


def answer(tokenizer, prediction):
    import re
    tokens = prediction['tokens']
    if not tokens or tokens[-1] != 2:
        raise ValueError('VQA generation did not finish with EOS')
    text = tokenizer.decode(tokens, errors='strict')
    match = re.fullmatch(r'<field>[^<>]+</field>(?:<value>[^<>]*</value>)?<op>[^<>]+</op><answer>([^<>]*)</answer>',text)
    if match is None:
        raise ValueError('Malformed VQA structured output')
    return clean_text(match[1])


def ensure_fp32_reference(source, packages):
    """Prepare a public reference after imports without calibrating the candidate."""
    reference = packages/'fp32-prepared'
    if (reference/'manifest.json').is_file():
        return
    from vqa.model import convert_model
    packages.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-', dir=packages) as temporary:
        converted = Path(temporary)/'fp32-prepared'
        convert_model(source, converted, 'fp32', VERSION, prepare_for_ptq=True)
        if reference.exists():
            shutil.rmtree(reference)
        converted.replace(reference)


def _init_eval_worker(package_dir_str, precision):
    global _worker_runner, _worker_tok
    _worker_tok = load_tokenizer(SOURCE)
    _worker_runner = Runner(Path(package_dir_str), precision=precision)


def _eval_record(item):
    img_path, question, expected_answer, record_id = item
    image = preprocess(Path(img_path))
    ids = _worker_tok.encode(clean_text(question), add_eos=True)
    pred = generate(_worker_runner, image, ids)
    finished = bool(pred['tokens'] and pred['tokens'][-1] == 2)
    try:
        extracted = answer(_worker_tok, pred)
        structured_valid = True
    except ValueError:
        extracted, structured_valid = '', False
    correct = structured_valid and extracted == clean_text(expected_answer)
    return {
        'id': record_id, 'question': question, 'expected': expected_answer,
        'extracted': extracted, 'correct': correct, 'family': pred.get('family', ''),
        'finished': finished, 'structured_valid': structured_valid,
    }


def run_heldout_evaluation(packages, precision, evaluation_root, limit=None, report_dir=None, workers=4):
    from multiprocessing import Pool
    from collections import defaultdict
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    if workers < 1:
        raise ValueError('--workers must be positive')
    images_dir = evaluation_root / 'images'
    annotations_dir = evaluation_root / 'annotations'
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise ValueError('--evaluation-root must contain images/ and annotations/')
    ann_files = sorted(annotations_dir.glob('*.json'))
    if limit is not None:
        ann_files = ann_files[:limit]
    if not ann_files:
        raise ValueError('Evaluation dataset contains no annotation records')
    package_dir = packages / 'ptq' / precision
    image_paths = defaultdict(list)
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths[path.stem].append(path)
    items = []
    for p in ann_files:
        d = json.loads(p.read_text(encoding='utf-8'))
        question = clean_text(d.get('question') if d.get('question') is not None else '')
        expected = clean_text(d.get('answer') if d.get('answer') is not None else '')
        if not question or not expected:
            raise ValueError(f'Evaluation annotation requires question and answer: {p.name}')
        if d.get('image_path'):
            # Explicit relative image paths are based at the dataset root.
            img_p = evaluation_root / d['image_path']
            if not img_p.is_file():
                img_p = images_dir / Path(d['image_path']).name
        else:
            matches = image_paths[p.stem]
            if len(matches) != 1:
                raise ValueError(f'Evaluation annotation requires exactly one matching image: {p.name}')
            img_p = matches[0]
        if not img_p.is_file():
            raise ValueError(f'Evaluation annotation image is missing: {p.name}')
        items.append((str(img_p), question, expected, p.stem))
    started = time.perf_counter()
    results = []
    with Pool(processes=workers, initializer=_init_eval_worker, initargs=(str(package_dir), precision)) as pool:
        for idx, res in enumerate(pool.imap(_eval_record, items, chunksize=8)):
            results.append(res)
            if (idx + 1) % 200 == 0 or (idx + 1) == len(items):
                print(f'VQA external-heldout: {idx+1}/{len(items)} images', flush=True)
    elapsed = time.perf_counter() - started
    correct_cnt = sum(r['correct'] for r in results)
    total_cnt = len(results)
    accuracy = correct_cnt / total_cnt
    ms_per_image = (elapsed / total_cnt) * 1000
    by_family = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        q_lower = r['question'].lower()
        fam = 'phone_number' if ('phone' in q_lower or '전화번호' in q_lower) else ('address' if ('address' in q_lower or '도로명' in q_lower or '위치' in q_lower) else 'other')
        by_family[fam]['total'] += 1
        if r['correct']:
            by_family[fam]['correct'] += 1
    print(f'{precision}: answers {correct_cnt}/{total_cnt} ({accuracy*100:.2f}%); {ms_per_image:.2f} ms/image', flush=True)
    report = {
        'precision': precision, 'total': total_cnt, 'correct': correct_cnt,
        'accuracy': accuracy, 'wall_seconds': elapsed, 'ms_per_image': ms_per_image,
        'by_family': {k: {'total': v['total'], 'correct': v['correct'], 'accuracy': v['correct']/max(1, v['total'])} for k, v in sorted(by_family.items())},
        'status': 'completed'
    }
    if report_dir:
        dest = report_dir / precision / 'heldout-check.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_json(dest, report)
        print(f'Report: {dest}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('w8a8', 'w4a8', 'all'), default='w4a8')
    parser.add_argument('--packages', type=Path, default=PACKAGES)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'test-results/vqa')
    parser.add_argument('--evaluation-root', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.evaluation_root:
        precisions = ('w8a8', 'w4a8') if args.precision == 'all' else (args.precision,)
        for p in precisions:
            run_heldout_evaluation(args.packages, p, args.evaluation_root, limit=args.limit,
                                   report_dir=args.report_dir, workers=args.workers)
        return
    precisions = ('w8a8',) if args.precision == 'all' else (args.precision,)
    fp32 = None
    quantized = {}
    started_check = time.perf_counter()
    report = {'reference':f'HF FP32 via ONNX Runtime {ort.__version__}', 'runtime':f'VolvoxAI {VERSION}',
              'precisions':list(precisions), 'status':'running',
              'cases':[], 'parity_max_abs':{}, 'scope':'Two public synthetic receipts; not heldout accuracy'}
    def close_values(label, actual, expected, atol=.0005):
        assert actual.shape == expected.shape, (label,actual.shape,expected.shape)
        difference = float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)),initial=0))
        report['parity_max_abs'][label] = max(report['parity_max_abs'].get(label,0),difference)
        np.testing.assert_allclose(actual,expected,rtol=.0005,atol=atol,err_msg=label)
    try:
        report['runtime_source'] = package_source()
        report['native_sha256'] = {path.name:sha256(path.read_bytes()) for path in packaged_library_paths()}
        ensure_fp32_reference(SOURCE, args.packages)
        report['model_sha256'] = {}
        for variant in ('fp32-prepared', *(f'ptq/{precision}' for precision in precisions)):
            directory = args.packages / variant
            report['model_sha256'][variant] = {str(path.relative_to(directory)):sha256(path.read_bytes())
                for path in package_model_files(directory)}
        tokenizer = load_tokenizer(SOURCE)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        sessions = {role:ort.InferenceSession(str(SOURCE/f'{role}_model.onnx'),sess_options=opts,
                                            providers=['CPUExecutionProvider']) for role in ('encoder','decoder')}
        fp32 = Runner(args.packages/'fp32-prepared')
        for precision in precisions:
            quantized[precision] = Runner(args.packages/'ptq'/precision, precision=precision)
        examples = json.loads((SOURCE/'examples/family_examples.json').read_text())['examples']
        for example in examples:
            image = preprocess(SOURCE/'examples'/example['image'])
            for case in [*example['cases'], *example.get('additional_cases',[])]:
                started = time.perf_counter()
                question = clean_text(case['question'])
                ids = tokenizer.encode(question,add_eos=True,max_len=192)
                original = ort_run(sessions['encoder'],{'image':image,'question_ids':np.asarray([ids]),
                                                        'family_ids':np.asarray([-1])})
                encoded = fp32.run('encoder',encoder_feed(image,ids))
                memory_length = 210+len(ids)
                for name in ('router_logits','selected_family_ids'):
                    close_values(name,encoded[name],original[name])
                close_values('encoder_memory',encoded['memory'][:,:memory_length],original['memory'])
                for name in CROSS:
                    close_values('encoder_cross',encoded[name][:,:,:memory_length],original[name])
                source_feed = decoder_feed(original)
                source_feed['past_padding_mask'] = np.zeros((1,0),np.int32)
                source_feed.update({name:np.zeros((1,8,0,40),np.float32) for name in PAST})
                feed = decoder_feed(encoded)
                generated = [1]
                for position in range(191):
                    expected = ort_run(sessions['decoder'],source_feed)
                    actual = fp32.run('decoder',feed)
                    close_values('decoder_logits',actual['logits'],expected['logits'],atol=.001)
                    for name in PRESENT:
                        close_values('decoder_present',actual[name][:,:,1:],expected[name],atol=.001)
                    token = int(actual['logits'][0,0].argmax())
                    assert token == int(expected['logits'][0,0].argmax()), 'Greedy token differs from HF source'
                    generated.append(token)
                    if token == 2: break
                    feed = advance(feed,actual,token,position)
                    source_feed = advance(source_feed,expected,token,position)
                expected_answer = str(case['expected_answer'])
                fp32_answer = answer(tokenizer,{'tokens':generated})
                assert fp32_answer == expected_answer, (question,fp32_answer,expected_answer)
                for precision, ptq in quantized.items():
                    prediction = generate(ptq,image,ids)
                    ptq_answer = answer(tokenizer,prediction)
                    report['cases'].append({'image':example['image'],'question':question,'expected':expected_answer,
                        'precision':precision, 'fp32':fp32_answer,'ptq':ptq_answer,'correct':ptq_answer==expected_answer,
                        'family':prediction['family'],'seconds':time.perf_counter()-started})
                    print(f'{len(report["cases"]):02d} {example["language"]} {precision}={ptq_answer!r} expected={expected_answer!r}',flush=True)
        # Full-length question and maximum legal cache/position validate the bounds.
        ids = [1]*191+[2]
        original = ort_run(sessions['encoder'],{'image':image,'question_ids':np.asarray([ids]),'family_ids':np.asarray([-1])})
        encoded = fp32.run('encoder',encoder_feed(image,ids))
        close_values('max_question_memory',encoded['memory'],original['memory'])
        feed = decoder_feed(encoded)
        feed.update(position_ids=np.asarray([190],np.int32),past_padding_mask=np.ones((1,191),np.int32),
                    **{name:np.zeros((1,8,191,40),np.float32) for name in PAST})
        source_feed = {**feed, 'past_padding_mask':np.ones((1,190),np.int32),
                       **{name:np.zeros((1,8,190,40),np.float32) for name in PAST}}
        actual, expected = fp32.run('decoder',feed), ort_run(sessions['decoder'],source_feed)
        close_values('max_cache_logits',actual['logits'],expected['logits'],atol=.001)
        assert actual['present_k_0'].shape == (1,8,192,40)
        report['maximum_question_and_cache'] = 'passed'
        report['passed'] = sum(case['correct'] for case in report['cases'])
        report['total'] = len(report['cases'])
        report['size_bytes'] = {}
        for variant in ('fp32-prepared', *(f'ptq/{precision}' for precision in precisions)):
            directory = args.packages/variant
            files = package_model_files(directory)
            # Shared shards count once in package bytes, not once per loader.
            report['size_bytes'][variant] = {role:sum(path.stat().st_size for path in files
                if path.relative_to(directory).parts[0] == role) for role in ('encoder','decoder','shared')}
        report['status'] = 'passed' if report['passed'] == report['total'] else 'failed'
        print(f'PTQ: {report["passed"]}/{report["total"]}; source padding/sentinel parity passed',flush=True)
    except BaseException as error:
        report.update(status='failed', error={'type':type(error).__name__, 'message':str(error),
            'traceback':traceback.format_exc()})
        raise
    finally:
        if fp32 is not None:
            fp32.close()
        for ptq in quantized.values():
            ptq.close()
        report['seconds'] = time.perf_counter() - started_check
        report['passed'] = sum(case['correct'] for case in report['cases'])
        report['total'] = len(report['cases'])
        args.report_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.report_dir/f'model-check-{args.precision}.json',report)
    if report['passed'] != report['total']:
        raise SystemExit(f'PTQ public-example accuracy regression; inspect {args.report_dir}/model-check-{args.precision}.json')


if __name__ == '__main__':
    main()
