"""Render benchmark tables from the public aggregate JSON documents."""
from __future__ import annotations

import html


DIGIT_LABELS = {
    'f32': 'FP32',
    'w8a8': 'W8A8 · Conv INT8 / Linear FP32',
    'conv8-linear4': 'Conv8/Linear4 · MSE PTQ',
    'conv4-linear4': 'Conv4/Linear4 · MSE PTQ',
    'conv4-linear3': 'Conv4/Linear3 · importance PTQ',
    'conv4-linear2': 'Conv4/Linear2 · joint QAT',
    'conv4-linear3-mse-ptq': 'Conv4/Linear3 · MSE PTQ',
    'conv4-linear3-qat': 'Conv4/Linear3 · joint QAT',
    'conv4-linear2-mse-ptq': 'Conv4/Linear2 · MSE PTQ',
    'conv3-linear3-mse-ptq': 'Conv3/Linear3 · MSE PTQ',
    'conv3-linear3-qat': 'Conv3/Linear3 · joint QAT',
    'conv8-linear2-importance': 'Conv8/Linear2 · importance PTQ',
    'conv8-linear2-head-qat': 'Conv8/Linear2 · head-only QAT',
}


def _table(headers, rows):
    def cells(values, tag):
        scope = ' scope="col"' if tag == 'th' else ''
        return ''.join(f'<{tag}{scope}>{html.escape(str(value))}</{tag}>' for value in values)

    return ('<table><thead><tr>' + cells(headers, 'th')
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + cells(row, 'td') + '</tr>' for row in rows)
            + '</tbody></table>')


def _number(value, spec='.2f'):
    return '—' if value is None else format(value, spec)


def _count(value):
    return _number(value, ',d')


def _match(value):
    return '일치' if value is True else '차이 있음' if value is False else '—'


def _label(model):
    return DIGIT_LABELS.get(model['id'], model['id'])


def _accuracy(model, key, same_scores):
    scores = []
    for engine in ('native', 'chrome'):
        result = model['accuracy'][engine]
        samples = result['samples']
        scores.append(f"{100 * result[key] / samples:.2f}%" if samples else '—')
    return scores[0] if same_scores else ' / '.join(scores)


def _digit_comparison(comparison):
    models = comparison.get('models', [])
    same_scores = all(model['accuracy']['native'][key] == model['accuracy']['chrome'][key]
                      for model in models for key in ('record_correct', 'answer_correct', 'samples'))
    suffix = '' if same_scores else ' · Native/Chrome'
    rows = [
        [_label(model) + (' · 기본 선택' if model.get('selected') else ''),
         _number(model['graph_weight_bytes'] / 1048576, '.3f'),
         _accuracy(model, 'record_correct', same_scores),
         _accuracy(model, 'answer_correct', same_scores),
         _number(model['latency']['native']['median_ms']),
         _number(model['latency']['chrome']['median_ms'])]
        for model in models
    ]
    return _table(['Digit 모델 · 방법', 'MiB', '전체 레코드' + suffix, '질문 응답' + suffix,
                   'Native 중앙값 ms', 'Chrome 중앙값 ms'], rows)


def _digit_parity(comparison):
    protocol = comparison.get('protocol', {})
    tolerance = protocol.get('accuracy', {}).get('chrome', {}).get('tolerance', {})
    accuracy_rows, latency_rows = [], []
    for model in comparison.get('models', []):
        parity = model.get('native_parity', {})
        accuracy_rows.append([
            _label(model), _count(model['accuracy']['chrome']['samples']),
            _count(parity.get('slot_mismatches')), _count(parity.get('numerical_mismatches')),
            _count(parity.get('byte_mismatches')),
            _number(parity.get('maximum_absolute_logit_difference'), '.6g'),
            _number(tolerance.get('rtol'), 'g'), _number(tolerance.get('atol'), 'g'),
        ])
        for engine, validation in model.get('latency_validation', {}).items():
            measured = protocol.get('latency', {}).get(engine, {})
            limits = measured.get('tolerance', {})
            latency_rows.append([
                _label(model), {'native': 'Native', 'chrome': 'Chrome'}.get(engine, engine),
                _count(measured.get('input_count')),
                _match(validation.get('native_reference_slots_equal')),
                _match(validation.get('native_reference_logits_within_tolerance')),
                _match(validation.get('native_reference_bytes_equal')),
                _number(validation.get('maximum_absolute_logit_difference'), '.6g'),
                _number(limits.get('rtol'), 'g'), _number(limits.get('atol'), 'g'),
            ])
    return '\n'.join([
        _table(['모델 · 정확도 평가 Native/Chrome', '비교 입력', '슬롯 불일치 입력',
                'Logits 허용오차 초과 입력', '바이트 불일치 입력', '최대 logits 절대 차이',
                'rtol', 'atol'], accuracy_rows),
        _table(['모델 · 별도 속도 측정의 Native 참조 비교', '실행 환경', '비교 입력',
                '슬롯 일치', 'Logits 허용오차 내', '바이트 일치', '최대 logits 절대 차이',
                'rtol', 'atol'], latency_rows),
    ])


def render_tables(results, digit_comparison):
    """Return five HTML fragments; inputs are parsed JSON and are never changed.

    ``digit-parity`` contains accuracy and latency-validation tables. Missing
    optional parity measurements are shown as an em dash, never as a pass.
    ``digit_comparison`` may be None when a comparison has not been published.
    """
    digit, vqa = results['models']['digit'], results['models']['vqa']
    timing_rows = []
    for engine, timing in digit['latency']['warm'].items():
        label = (f"네이티브 CPU · {timing['cpu_threads']}스레드" if engine == 'native_cpu_1t'
                 else 'Chrome WASM worker' if engine == 'chrome_wasm' else engine)
        timing_rows.append([label, _number(timing['median_ms']), _number(timing['mean_ms']),
                            _number(timing['p95_ms'])])
    reference = vqa['reference_w8a8']
    selected = vqa['identity']
    reference_label = reference['profile'].upper() + ' 기준'
    selected_label = selected['profile'].upper() + ' PTQ'
    vqa_timing = vqa['latency']
    vqa_rows = [
        [label, _number(vqa_timing['reference_w8a8'][key], f'.{precision}f'),
         _number(vqa_timing[selected['profile']][key], f'.{precision}f')]
        for key, label, precision in (
            ('complete_answer_median_ms', '완성된 답변 · ms', 2),
            ('encoder_median_ms', '인코더 · ms', 2),
            ('greedy_decoder_median_ms', '전체 greedy 디코더 · ms', 2),
            ('fixed_prefix_decoder_median_ms_per_token', '고정 prefix 디코더 · ms/token', 3),
        )
    ]
    return {
        'digit-comparison': _digit_comparison(digit_comparison) if digit_comparison else '',
        'digit-timing': _table(['실행 환경', '중앙값 ms', '평균 ms', '표본 p95 ms'], timing_rows),
        'digit-parity': _digit_parity(digit_comparison) if digit_comparison else '',
        'vqa-size': _table(['VQA 모델', '모델 패키지'], [
            [reference_label, _number(reference['package_bytes'] / 1e6, '.3f') + ' MB'],
            [selected_label + ' · 선택', _number(selected['package_bytes'] / 1e6, '.3f') + ' MB'],
        ]),
        'vqa-timing': _table(['측정 항목 · 중앙값', reference_label, selected_label], vqa_rows),
    }
