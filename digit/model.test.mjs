import assert from 'node:assert/strict';
import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { answerFromRecord, phoneOpFromQuestion, routeFamilyFromQuestion, addressOpFromQuestion,
  decodeSlots, modelAdapter, digitAttentionHeatmap } from './adapter.js';

const python = fileURLToPath(new URL('../.venv/bin/python', import.meta.url));

test('PTQ manifests cannot silently fall back to unspecified packed arithmetic', async () => {
  const originalFetch=globalThis.fetch;
  try {
    globalThis.fetch=async()=>({ok:true,json:async()=>({quantization:{precision:'conv4-linear3'}})});
    await assert.rejects(modelAdapter.open(), /양자화 연산 설정/);
  } finally { globalThis.fetch=originalFetch; }
});
const config = JSON.parse(readFileSync(new URL('./extension.json', import.meta.url), 'utf8'));
const source = fileURLToPath(new URL(`../.cache/digit/huggingface/${config.model.revision}`, import.meta.url));

test('slot decoding stops at the first blank and rejects invalid output', () => {
  const logits = new Float32Array(176).fill(-10);
  const slots = [0, 2, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 3, 2, 10];
  slots.forEach((digit, slot) => { logits[slot * 11 + digit] = 10; });
  assert.deepEqual(decodeSlots(logits), { phone: '02', street: '732' });
  logits[0] = NaN;
  assert.throws(() => decodeSlots(logits), /유효한/);
});

test('digit heatmaps select the actual answer slots and exclude unrelated or blank digits', () => {
  const attention = new Float32Array(16 * 420);
  for (let slot = 0; slot < 16; slot++) attention[slot * 420 + slot] = 1;
  const record = { phone: '012345', street: '78', attention };
  const front = digitAttentionHeatmap(attention, record, 'phone digit 3');
  const back = digitAttentionHeatmap(attention, record, '전화번호 뒤에서 2번째 숫자');
  const street = digitAttentionHeatmap(attention, record, 'What is the street number?');
  assert.deepEqual(front.slots, [2]);
  assert.equal(front.values[2], 1);
  assert.deepEqual(back.slots, [4]);
  assert.equal(back.values[4], 1);
  assert.deepEqual(street.slots, [12, 13]);
  assert.equal(street.values[12], .5);
  assert.equal(street.values[13], .5);
  assert.equal(street.values[14], 0, 'The blank slot contributes no attention');
  assert.equal(street.values.reduce((sum, value) => sum + value, 0), 1);
  assert.equal(street.kind, 'attention');
  assert.equal(street.width * street.height, 420);
  const result = modelAdapter.answer(record, 'phone digit 3');
  assert.equal(result.answer, '2');
  assert.deepEqual(result.record, { phone: '012345', street: '78' });
  assert(!Object.hasOwn(result.record, 'attention'), 'Only the selected spatial map leaves the worker');
  assert.equal(digitAttentionHeatmap(attention, record, 'What is the store name?'), undefined);
  assert.equal(digitAttentionHeatmap(attention, record, 'phone digit 99'), undefined);
  attention[2 * 420] = NaN;
  assert.throws(() => digitAttentionHeatmap(attention, record, 'phone digit 3'), /어텐션 값/);
});

test('Korean and English routing preserves range checks and unsupported questions', () => {
  for (const [question, expected] of [
    ["What is the first number of the store's phone number?", '8'],
    ['가게 전화번호의 뒤에서 2번째 숫자는 무엇입니까?', '1'],
    ['전화번호 끝자리는?', '8'], ['digit 0 of the phone number', ''],
    ['digit 99 of the phone number', ''],
    ['영수증의 가게 위치는 산문길 [?] 입니다. (빈 칸을 채워주세요)', '209'],
    ['가게 주소가 무엇입니까?', ''], ['상품 가격의 합계로 얼마입니까?', ''],
    ['What is the store name?', ''],
  ]) assert.equal(answerFromRecord(question, '8845118', '209'), expected, question);
});

test('JavaScript router agrees with the downloaded HF Python router', () => {
  const questions = [];
  for (let n = 0; n <= 14; n++) {
    for (const side of ['앞', '뒤']) questions.push(`가게 전화번호의 ${side}에서 ${n}번째 숫자는 무엇입니까?`);
    questions.push(`digit ${n} of the phone number`, `phone number from the end ${n}`);
  }
  questions.push('전화번호 끝자리', '상호명이 무엇입니까?', 'What is the store address?',
    'What is the street number in the store address?', '상품 가격의 합계로 얼마입니까?');
  const reference = JSON.parse(execFileSync(python, ['-I', '-c', `
import sys, json
sys.path.insert(0, sys.argv[1])
import question_router as r
out = []
for q in json.load(sys.stdin):
 f = r.route_family_from_question(q)
 op = r.phone_op_from_question(q) if f == 'phone' else r.address_op_from_question(q) if f == 'address' else 'unsupported'
 out.append([f, op, r.answer_from_record(q, '0123456789', '321')])
print(json.dumps(out))`, source], { input: JSON.stringify(questions), encoding: 'utf8' }));
  questions.forEach((q, i) => {
    const family = routeFamilyFromQuestion(q);
    const op = family === 'phone' ? phoneOpFromQuestion(q) : family === 'address' ? addressOpFromQuestion(q) : 'unsupported';
    assert.deepEqual([family, op, answerFromRecord(q, '0123456789', '321')], reference[i], q);
  });
});
