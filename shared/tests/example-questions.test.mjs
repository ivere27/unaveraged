import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { questionPresets, expectedAnswer } from '../../example/questions.js';
import { answerFromRecord } from '../../digit/adapter.js';

const receipt = {
  phone_digits: '0123456', street_no: 0, store: '합성 상점', address: '합성로 0',
  items: [{ name: '사과', quantity: 2, unit_price: 100, total: 200 },
    { name: '우유', quantity: 3, unit_price: 400, total: 1200 }],
};

test('Digit presets stay within its supported numeric questions, including leading zeroes', () => {
  const presets = questionPresets({ receipt }, 'digit');
  assert.equal(presets.length, 7);
  for (const item of presets) assert.equal(answerFromRecord(item.question, receipt.phone_digits, '0'), item.answer);
  assert.equal(presets.find(item => item.id === 'phone-front-1').answer, '0');
  assert.equal(presets.find(item => item.id === 'street-number').answer, '0');
});

test('VQA presets cover receipt fields, item lookup, positions and exact annotated totals', () => {
  const presets = questionPresets({ receipt }, 'vqa');
  assert.deepEqual(new Set(presets.map(item => item.family)), new Set(['전화번호', '주소', '가게', '품목', '상품별 조회', '합계']));
  assert.equal(presets.find(item => item.id === 'items-total').answer, '1400');
  assert.equal(presets.find(item => item.id === 'items-quantity').answer, '5');
  assert.equal(presets.find(item => item.id === 'lookup-2-quantity').question, '우유 개수는 몇 개입니까?');
  assert.equal(presets.find(item => item.id === 'item-2-name').answer, '우유');
});

test('missing fields and ambiguous item names do not fabricate grading answers', () => {
  const noItems = questionPresets({ receipt: { phone_digits: '12', street_no: '45' } }, 'vqa');
  assert(!noItems.some(item => ['품목', '상품별 조회', '합계', '가게'].includes(item.family)));
  assert(!noItems.some(item => item.id.endsWith('-3')));
  const repeated = questionPresets({ receipt: { ...receipt, items: [receipt.items[0], receipt.items[0]] } }, 'vqa');
  assert(!repeated.some(item => item.family === '상품별 조회'));
  assert.equal(expectedAnswer({ receipt }, 'vqa', '이 영수증에서 추천할 상품은?'), undefined);
  assert.equal(expectedAnswer({}, 'digit', '가게 전화번호의 뒤에서 1번째 숫자는 무엇입니까?'), undefined);
  assert.equal(expectedAnswer({ receipt }, 'vqa', '  가게   이름은 무엇입니까?\n'), '합성 상점');
});

test('pinned public examples retain original defaults and gain grounded diverse presets', () => {
  const sources = JSON.parse(readFileSync(new URL('../../docs/assets/examples/sources.json', import.meta.url), 'utf8'));
  for (const sample of sources.examples) {
    const presets = questionPresets(sample, sample.model);
    assert.equal(presets[0].question, sample.questions[0].question);
    assert.equal(presets[0].answer, sample.questions[0].answer);
    assert(presets[0].label.length < presets[0].question.length, 'Compact presets avoid duplicating the question editor');
    assert.equal(new Set(presets.map(item => item.question)).size, presets.length);
    assert(presets.length >= (sample.model === 'digit' ? 7 : 20));
    for (const item of presets) assert.equal(typeof item.answer, 'string');
  }
});
