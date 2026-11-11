import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {decodeDigitLogits, scoreDigitOutput, validateDigitReference, validateDigitCheckpoint,
  summarizeDigitRecords} from '../../benchmarks/digit-evaluation.js';

const sha = value => createHash('sha256').update(value).digest('hex');
function example() {
  const values = new Float32Array(176).fill(-10);
  const slots = [0, 2, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 3, 2, 10];
  slots.forEach((digit, slot) => { values[slot * 11 + digit] = 10; });
  const expected = {phone: '02', street: '732'};
  const reference = {id: 'one', values: [...values], slots, sha256: sha(new Uint8Array(values.buffer))};
  const item = {id: 'one', sha256: 'a'.repeat(64), record: expected,
    questions: [{question: 'What is the first digit of the phone number?', answer: '0'}]};
  const scored = scoreDigitOutput(values, expected, reference, {atol: 0.001, rtol: 0.0005, questions: item.questions});
  const row = {index: 0, model: 'test', id: item.id, input_sha256: item.sha256,
    output_sha256: reference.sha256, inference_ms: 10, ...scored, native: {...scored.native, bytes_equal: true}};
  return {values, reference, item, row};
}

test('Digit full-record scoring preserves leading zeros and ignores slots after blank', () => {
  const {values, reference, item} = example();
  assert.deepEqual(decodeDigitLogits(values).record, {phone: '02', street: '732'});
  const score = scoreDigitOutput(values, {...item.record, street: '733'}, reference, {atol: 0.001, rtol: 0.0005,
    questions: [{question: 'What is the first digit of the phone number?', answer: '0'}]});
  assert.deepEqual(score.scores, {record_correct: false, phone_correct: true, street_correct: false});
  assert.deepEqual(score.answers, [{answer: '0', correct: true}]);
  assert.equal(score.native.slots_equal, true);
});

test('Native reference ordering, shape, hashes and decoded slots must be present', () => {
  const {reference} = example();
  validateDigitReference(reference, 'one');
  for (const change of [r => { r.id = 'other'; }, r => { r.values.pop(); }, r => { r.values[0] = NaN; },
    r => { r.slots[0] = 10; }, r => { r.sha256 = ''; }]) {
    const altered = structuredClone(reference); change(altered);
    assert.throws(() => validateDigitReference(altered, 'one'));
  }
});

test('Native logit drift is counted even when prediction and accuracy stay unchanged', () => {
  const {values, reference, item} = example();
  values[0] += 1;
  const result = scoreDigitOutput(values, item.record, reference, {atol: 0.001, rtol: 0.0005});
  assert.equal(result.scores.record_correct, true);
  assert.equal(result.native.slots_equal, true);
  assert.equal(result.native.logits_within_tolerance, false);
  assert.equal(result.native.outside_tolerance, 1);
  values[0] = Infinity;
  assert.throws(() => scoreDigitOutput(values, item.record, reference, {atol: 0.001, rtol: 0.0005}), /finite/);
});

test('Checkpoint resumes reject reordered inputs, invalid timing and missing comparison fields', () => {
  const {item, row, reference} = example();
  validateDigitCheckpoint(row, 0, 'test', [item], [reference]);
  for (const change of [r => { r.index = 1; }, r => { r.input_sha256 = 'b'.repeat(64); },
    r => { r.inference_ms = NaN; }, r => { r.native.bytes_equal = undefined; },
    r => { r.scores.record_correct = false; }, r => { r.answers = []; },
    r => { r.answers[0].correct = false; }, r => { r.record.phone = '20'; },
    r => { r.native.slots_equal = false; }, r => { r.output_sha256 = 'b'.repeat(64); }]) {
    const altered = structuredClone(row); change(altered);
    assert.throws(() => validateDigitCheckpoint(altered, 0, 'test', [item], [reference]));
  }
});

test('Aggregate counts record and question accuracy separately and uses interpolated p95', () => {
  const {row} = example(), second = structuredClone(row);
  second.inference_ms = 30; second.scores.street_correct = false; second.scores.record_correct = false;
  second.native.slots_equal = false;
  const result = summarizeDigitRecords([row, second], 2000);
  assert.deepEqual(result.metrics.record_correct, {correct: 1, count: 2, percent: 50});
  assert.deepEqual(result.metrics.answer_correct, {correct: 2, count: 2, percent: 100});
  assert.equal(result.completed, 2); assert.equal(result.samples, 2000);
  assert.equal(result.latency.median_ms, 20); assert.equal(result.latency.p95_ms, 29);
  assert.equal(result.native_parity.slot_mismatches, 1);
  assert.equal(summarizeDigitRecords([], 2000).latency.median_ms, null);
});
