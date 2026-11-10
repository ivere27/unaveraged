import assert from 'node:assert/strict';
import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { cleanText, parseGenerated, modelAdapter, generate, CROSS, PAST, PRESENT,
  crossAttentionImageWeights, VQA_ATTENTION } from './adapter.js';
import { TinyReceiptByteFallbackBPEVocab } from './tokenizer.js';

const python = fileURLToPath(new URL('../.venv/bin/python', import.meta.url));
const config = JSON.parse(readFileSync(new URL('./extension.json', import.meta.url), 'utf8'));
const source = fileURLToPath(new URL(`../.cache/vqa/huggingface/${config.model.revision}`, import.meta.url));
const vocab = JSON.parse(readFileSync(`${source}/vocab.json`, 'utf8'));
const tokenizer = await TinyReceiptByteFallbackBPEVocab.fromJSON(vocab, { pad: 0, bos: 1, eos: 2, unk: 3 });

test('BPE tokens and decoded text agree with the pinned HF Python tokenizer', () => {
  const fixtures = JSON.parse(readFileSync(`${source}/examples/family_examples.json`, 'utf8')).examples;
  const questions = fixtures.flatMap((receipt) => [...receipt.cases, ...(receipt.additional_cases || [])].map((item) => item.question));
  questions.push('가게 전화번호의 뒤에서 3번째 숫자는?', 'Café 0123456789 😀 𠀀', '가게 상호명',
    '  milk\u0085quantity\u001f3  ', '<field>phone</field><op>identity</op><answer>0123</answer>',
    'a'.repeat(400), 'abc\ufeffdef');
  const reference = JSON.parse(execFileSync(python, ['-I', '-c', `
import json, sys, re, unicodedata
sys.path.insert(0,sys.argv[1])
from byte_bpe import load_bpe1536_vocab
from pathlib import Path
tok=load_bpe1536_vocab(json.loads((Path(sys.argv[1])/'vocab.json').read_text()))
result=[]
for q in json.load(sys.stdin):
 q=unicodedata.normalize('NFC', re.sub(r'\\s+',' ',q).strip())
 ids=tok.encode(q,add_eos=True,max_len=192)
 result.append([q,ids,tok.decode(ids)])
print(json.dumps(result))`, source], { input: JSON.stringify(questions), encoding: 'utf8' }));
  questions.forEach((question, index) => {
    const ids = tokenizer.encodeQuestion(question, 192);
    assert.deepEqual([cleanText(question), ids, tokenizer.decode(ids)], reference[index], question);
  });
});

test('VQA cache includes the question and accepts only complete structured answers', () => {
  const key = (question) => modelAdapter.cacheKey({ imageHash: 'image', question });
  assert.notEqual(key('phone?'), key('address?'));
  assert.equal(key('  store   name? '), key('store name?'));
  assert.deepEqual(parseGenerated('<field>store</field><op>identity</op><answer>착한 가게</answer>'),
    { answer: '착한 가게', field: 'store', op: 'identity' });
  for (const text of ['<answer>123', '<answer>123</answer>',
    '<field>phone</field><op>identity</op><answer>123</answer>extra']) assert.throws(() => parseGenerated(text));
});

test('VQA heatmaps average real heads only across answer-generation tokens and retain image mass', async () => {
  const pb = { DataType: { DATA_TYPE_F32: 1, DATA_TYPE_I32: 2 } };
  const tensor = (shape, values) => ({ shape, inline: new Uint8Array(values.buffer),
    dtype: values instanceof Float32Array ? 1 : 2 });
  const sequence = tokenizer.encode('<field>phone</field><op>identity</op><answer>12345</answer>', { addEos: true });
  const begin = sequence.indexOf(tokenizer.stoi.get('<answer>')) + 1;
  const end = sequence.indexOf(tokenizer.stoi.get('</answer>'));
  assert(begin > 0 && end > begin);
  async function execute(heatmap) {
    let position = 0;
    return generate({ pb, tokenizer, image: new Float32Array(672 * 320), question: 'phone?', heatmap,
      run: async (role, feed, outputs) => {
        if (role === 'encoder') {
          const result = { selected_family_ids: tensor([1], new Int32Array([0])),
            memory_padding_mask: tensor([1, 402], new Int32Array(402)) };
          for (const name of CROSS) result[name] = tensor([1, 8, 402, 40], new Float32Array(8 * 402 * 40));
          return result;
        }
        const logits = new Float32Array(1536); logits[sequence[position]] = 1;
        const result = { logits: tensor([1, 1, 1536], logits) };
        if (outputs.includes(VQA_ATTENTION.name)) {
          assert(position >= begin && position <= end, 'No unrelated field/value/op token maps are requested');
          const attention = new Float32Array(8 * 402);
          for (let head = 0; head < 8; head++) {
            attention[head * 402 + (position < end ? position - begin : 209)] = .75;
            attention[head * 402 + 210] = .25;
          }
          result[VQA_ATTENTION.name] = tensor([1, 8, 1, 402], attention);
        }
        position++;
        return result;
      },
    });
  }
  const baseline = await execute(false);
  const { heatmap, ...captured } = await execute(true);
  assert.deepEqual(captured, baseline);
  assert.equal(captured.answer, '12345');
  assert.equal(heatmap.answerTokens, end - begin);
  assert.equal(heatmap.width * heatmap.height, 210);
  assert.equal(heatmap.imageMass, .75);
  for (let index = 0; index < end - begin; index++) assert.equal(heatmap.values[index], 1 / (end - begin));
  assert.equal(heatmap.values[209], 0, 'The closing tag is excluded');
  assert(Math.abs(heatmap.values.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  const invalid = new Float32Array(8 * 402);
  assert.throws(() => crossAttentionImageWeights(invalid), /어텐션 확률의 합/);
  invalid[0] = -1;
  assert.throws(() => crossAttentionImageWeights(invalid), /어텐션 값/);
});

test('decoder uses a masked initial cache, advances KV, and refuses unfinished generation', async () => {
  const pb = { DataType: { DATA_TYPE_F32: 1, DATA_TYPE_I32: 2 } };
  const tensor = (shape, array) => ({ shape, inline: new Uint8Array(array.buffer), dtype: array instanceof Float32Array ? 1 : 2 });
  let calls = 0;
  await assert.rejects(generate({ pb, tokenizer, image: new Float32Array(672 * 320), question: 'phone?',
    run: async (role, feed, outputs, bindings) => {
      if (role === 'encoder') {
        assert.equal(feed.question_ids.shape[1], 192);
        const result = { selected_family_ids: tensor([1], new Int32Array([0])),
          memory_padding_mask: tensor([1, 402], new Int32Array(402)) };
        for (const name of CROSS) result[name] = tensor([1, 8, 402, 40], new Float32Array(8 * 402 * 40));
        return result;
      }
      assert.equal(new Int32Array(feed.position_ids.inline.buffer)[0], calls);
      assert.deepEqual(outputs, ['logits']);
      if (calls === 0) {
        assert.equal(feed.past_padding_mask.shape[1], 1);
        assert.equal(new Int32Array(feed.past_padding_mask.inline.buffer)[0], 1);
        assert.deepEqual(bindings, {});
        for (const name of PAST) assert.deepEqual(feed[name].shape, [1, 8, 1, 40]);
      } else {
        assert.deepEqual(Object.keys(feed).sort(), ['decoder_input_ids', 'position_ids']);
        assert.deepEqual(bindings.reuse, ['memory_padding_mask', 'family_ids', ...CROSS]);
        assert.deepEqual(bindings.feedback, [
          { inputName: 'past_padding_mask', outputName: 'present_padding_mask' },
          ...PAST.map((inputName, index) => ({ inputName, outputName: PRESENT[index] })),
        ]);
      }
      const logits = new Float32Array(1536); logits[tokenizer.stoi.get('1')] = 1;
      const result = { logits: tensor([1, 1, 1536], logits) };
      calls++;
      return result;
    },
  }), /최대 길이/);
  assert.equal(calls, 191);
  await assert.rejects(generate({ pb, tokenizer, image: new Float32Array(1), question: '1'.repeat(192),
    run: () => assert.fail('Overlong questions must be rejected before inference') }), /192 토큰/);
});
