// Pure scoring/checkpoint helpers shared by the streaming evaluator and tests.
import {answerFromRecord} from '../digit/adapter.js';
const requireTrue = (condition, message) => { if (!condition) throw new Error(message); };
export const sha256Pattern = /^[a-f0-9]{64}$/;

const recordFromSlots = slots => {
  requireTrue(Array.isArray(slots) && slots.length === 16
    && slots.every(value => Number.isInteger(value) && value >= 0 && value <= 10), 'Invalid Digit slots');
  const digits = values => {
    const end = values.indexOf(10);
    return values.slice(0, end < 0 ? values.length : end).join('');
  };
  return {phone: digits(slots.slice(0, 12)), street: digits(slots.slice(12))};
};

export function decodeDigitLogits(values) {
  requireTrue(values.length === 176 && Array.from(values).every(Number.isFinite), 'Expected 176 finite Digit logits');
  const slots = Array.from({length: 16}, (_, row) => {
    let best = 0;
    for (let column = 1; column < 11; column++)
      if (values[row * 11 + column] > values[row * 11 + best]) best = column;
    return best;
  });
  return {slots, record: recordFromSlots(slots)};
}

export function validateDigitReference(reference, id) {
  requireTrue(reference?.id === id && Array.isArray(reference.values) && reference.values.length === 176
    && reference.values.every(Number.isFinite), 'Missing, misordered or nonfinite native reference');
  requireTrue(Array.isArray(reference.slots) && reference.slots.length === 16
    && reference.slots.every(value => Number.isInteger(value) && value >= 0 && value <= 10)
    && sha256Pattern.test(reference.sha256), 'Invalid native reference slots/hash');
  requireTrue(JSON.stringify(decodeDigitLogits(reference.values).slots) === JSON.stringify(reference.slots),
    'Native reference slots disagree with logits');
}

export function scoreDigitOutput(values, expected, reference, {atol, rtol, questions = []}) {
  requireTrue(Number.isFinite(atol) && atol > 0 && Number.isFinite(rtol) && rtol >= 0, 'Invalid logit tolerance');
  const output = decodeDigitLogits(values);
  requireTrue(expected && typeof expected.phone === 'string' && typeof expected.street === 'string', 'Missing expected record');
  let maxAbs = 0, outside = 0;
  for (let index = 0; index < values.length; index++) {
    const delta = Math.abs(values[index] - reference.values[index]);
    maxAbs = Math.max(maxAbs, delta);
    if (delta > atol + rtol * Math.abs(reference.values[index])) outside++;
  }
  const phone = output.record.phone === expected.phone, street = output.record.street === expected.street;
  const answers = questions.map(item => {
    requireTrue(typeof item.question === 'string' && typeof item.answer === 'string', 'Invalid evaluation question');
    const answer = answerFromRecord(item.question, output.record.phone, output.record.street);
    return {answer, correct: answer === item.answer};
  });
  return {...output, answers, scores: {record_correct: phone && street, phone_correct: phone, street_correct: street},
    native: {slots_equal: JSON.stringify(output.slots) === JSON.stringify(reference.slots),
      logits_within_tolerance: outside === 0, outside_tolerance: outside, max_absolute_logit_difference: maxAbs}};
}

export function validateDigitCheckpoint(row, index, model, cases, references) {
  const item = cases[index];
  requireTrue(item && row.index === index && row.model === model && row.id === item.id
    && row.input_sha256 === item.sha256 && sha256Pattern.test(row.output_sha256), 'Checkpoint identity/order differs');
  requireTrue(Number.isFinite(row.inference_ms) && row.inference_ms >= 0, 'Invalid checkpoint timing');
  requireTrue(['record_correct', 'phone_correct', 'street_correct'].every(key => typeof row.scores?.[key] === 'boolean'),
    'Invalid checkpoint score');
  requireTrue(row.scores.record_correct === (row.scores.phone_correct && row.scores.street_correct), 'Inconsistent checkpoint score');
  const record = recordFromSlots(row.slots);
  requireTrue(row.record?.phone === record.phone && row.record?.street === record.street
    && row.scores.phone_correct === (record.phone === item.record.phone)
    && row.scores.street_correct === (record.street === item.record.street), 'Checkpoint prediction/score differs');
  const questions = item.questions ?? (typeof item.question === 'string' ? [{question: item.question, answer: item.answer}] : []);
  requireTrue(Array.isArray(row.answers) && row.answers.length === questions.length
    && row.answers.every(answer => typeof answer.answer === 'string' && typeof answer.correct === 'boolean'),
    'Invalid checkpoint answers');
  requireTrue(row.answers.every((answer, position) =>
    answer.answer === answerFromRecord(questions[position].question, record.phone, record.street)
    && answer.correct === (answer.answer === questions[position].answer)), 'Checkpoint question score differs');
  requireTrue(['slots_equal', 'logits_within_tolerance', 'bytes_equal'].every(key => typeof row.native?.[key] === 'boolean')
    && Number.isFinite(row.native.max_absolute_logit_difference) && row.native.max_absolute_logit_difference >= 0
    && Number.isInteger(row.native.outside_tolerance) && row.native.outside_tolerance >= 0
    && row.native.outside_tolerance <= 176, 'Invalid checkpoint native comparison');
  const reference = references[index];
  requireTrue(reference?.id === row.id
    && row.native.slots_equal === (JSON.stringify(row.slots) === JSON.stringify(reference.slots))
    && row.native.bytes_equal === (row.output_sha256 === reference.sha256)
    && row.native.logits_within_tolerance === (row.native.outside_tolerance === 0), 'Checkpoint native comparison differs');
}

export function summarizeDigitRecords(records, total) {
  const samples = records.map(row => row.inference_ms).sort((a, b) => a - b);
  const percentile = fraction => {
    if (!samples.length) return null;
    const index = (samples.length - 1) * fraction, lower = Math.floor(index), upper = Math.ceil(index);
    return samples[lower] + (samples[upper] - samples[lower]) * (index - lower);
  };
  const answers = records.flatMap(row => row.answers), answerCorrect = answers.filter(item => item.correct).length;
  return {completed: records.length, samples: total,
    metrics: {...Object.fromEntries(['record_correct', 'phone_correct', 'street_correct'].map(key => {
      const correct = records.filter(row => row.scores[key]).length;
      return [key, {correct, count: records.length, percent: records.length ? 100 * correct / records.length : null}];
    })), answer_correct: {correct: answerCorrect, count: answers.length, percent: answers.length ? 100 * answerCorrect / answers.length : null}},
    latency: {samples: samples.length, median_ms: percentile(0.5), p95_ms: percentile(0.95),
      mean_ms: samples.length ? samples.reduce((sum, value) => sum + value, 0) / samples.length : null,
      percentile_method: 'Linear interpolation over one warm execution per heldout input'},
    native_parity: {slot_mismatches: records.filter(row => !row.native.slots_equal).length,
      numerical_mismatches: records.filter(row => !row.native.logits_within_tolerance).length,
      byte_mismatches: records.filter(row => !row.native.bytes_equal).length,
      maximum_absolute_logit_difference: records.length ? Math.max(...records.map(row => row.native.max_absolute_logit_difference)) : null}};
}
