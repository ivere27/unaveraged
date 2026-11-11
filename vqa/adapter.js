// Receipt VQA adapter; worker, transport, lifecycle and UI stay model-independent.
import { TinyReceiptByteFallbackBPEVocab } from './tokenizer.js';

import { preprocessPixels } from '../shared/extension/preprocess.js';
export function cleanText(value) {
  // Python re \s includes U+001C..001F and U+0085, but excludes U+FEFF.
  return String(value ?? '').replace(/[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/g, ' ')
    .replace(/^ | $/g, '').normalize('NFC');
}

export function parseGenerated(text) {
  const match = text.match(/^<field>([^<>]+)<\/field>(?:<value>[^<>]*<\/value>)?<op>([^<>]+)<\/op><answer>([^<>]*)<\/answer>$/u);
  if (!match) throw new Error('모델 답변 형식을 확인하지 못했습니다. 영수증을 직접 확인해 주세요.');
  return { answer: cleanText(match[3]), field: match[1], op: match[2] };
}

export const CROSS = Array.from({ length: 4 }, (_, i) => [`cross_k_${i}`, `cross_v_${i}`]).flat();
export const PAST = CROSS.map((name) => name.replace('cross_', 'past_'));
export const PRESENT = CROSS.map((name) => name.replace('cross_', 'present_'));
export const VQA_ATTENTION = Object.freeze({
  name: 'v172', opType: 'Softmax', dtype: 'float32', shape: [1, 8, 1, 402], axis: -1,
});

// Encoder memory is 210 row-major image cells followed by 192 question tokens.
// Preserve its image probability mass instead of confusing question attention
// with spatial attention. Head and answer-token means remain bounded online.
export function crossAttentionImageWeights(weights) {
  if (weights.length !== 8 * 402 || !weights.every((value) => Number.isFinite(value) && value >= 0)) {
    throw new Error('VQA 교차 어텐션 값이 올바르지 않습니다.');
  }
  const values = new Float64Array(210);
  for (let head = 0; head < 8; head++) {
    const row = weights.subarray(head * 402, (head + 1) * 402);
    if (Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) > .001) {
      throw new Error('VQA 교차 어텐션 확률의 합이 올바르지 않습니다.');
    }
    for (let index = 0; index < 210; index++) values[index] += row[index] / 8;
  }
  return values;
}

export function vqaAttentionHeatmap(accumulated, answerTokens) {
  if (!answerTokens) return undefined;
  const mass = accumulated.reduce((sum, value) => sum + value, 0);
  return { kind: 'attention', width: 21, height: 10,
    values: Array.from(accumulated, (value) => mass > 0 ? value / mass : 0),
    inputWidth: 672, inputHeight: 320, imageMass: mass / answerTokens,
    tensor: VQA_ATTENTION.name, heads: 8, answerTokens, layer: 'final-decoder-cross-attention',
    normalization: 'image-mass', label: '답변 토큰 · 마지막 교차 어텐션 (8개 헤드 평균)' };
}

const asBytes = (values) => new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
const intValues = (tensor) => new Int32Array(tensor.inline.slice().buffer);

export async function generate({ run, pb, tokenizer, image, question, heatmap = false }) {
  const text = cleanText(question);
  if (!text) throw new Error('질문이 비어 있습니다.');
  const ids = tokenizer.encode(text, { addEos: true });
  if (ids.length > 192) throw new Error('질문이 모델의 최대 길이(192 토큰)를 넘었습니다.');
  const tensor = (shape, values) => ({ shape,
    dtype: values instanceof Float32Array ? pb.DataType.DATA_TYPE_F32 : pb.DataType.DATA_TYPE_I32,
    inline: asBytes(values) });
  const padded = new Int32Array(192);
  padded.set(ids);
  const encoded = await run('encoder', {
    image: tensor([1, 1, 320, 672], image), question_ids: tensor([1, 192], padded),
    family_ids: tensor([1], new Int32Array([-1])),
  }, ['memory_padding_mask', 'selected_family_ids', ...CROSS]);
  const family = intValues(encoded.selected_family_ids)[0];
  if (family < 0 || family > 7) throw new Error('모델의 질문 분류가 올바르지 않습니다.');
  const initialFeed = { memory_padding_mask: encoded.memory_padding_mask, family_ids: encoded.selected_family_ids,
    past_padding_mask: tensor([1, 1], new Int32Array([1])) };
  for (const name of CROSS) initialFeed[name] = encoded[name];
  for (const name of PAST) initialFeed[name] = tensor([1, 8, 1, 40], new Float32Array(320));
  const continuation = {
    reuse: ['memory_padding_mask', 'family_ids', ...CROSS],
    feedback: [{ inputName: 'past_padding_mask', outputName: 'present_padding_mask' },
      ...PAST.map((inputName, index) => ({ inputName, outputName: PRESENT[index] }))],
  };
  const generated = [tokenizer.bos];
  const answerOpen = tokenizer.stoi?.get('<answer>');
  const answerClose = tokenizer.stoi?.get('</answer>');
  if (heatmap && (!Number.isInteger(answerOpen) || !Number.isInteger(answerClose))) {
    throw new Error('답변 어텐션의 토큰 구간을 확인하지 못했습니다.');
  }
  const attention = new Float64Array(210);
  let inAnswer = false, attentionTokens = 0;
  for (let position = 0; position < 191; position++) {
    const feed = position === 0 ? { ...initialFeed } : {};
    feed.decoder_input_ids = tensor([1, 1], new Int32Array([generated.at(-1)]));
    feed.position_ids = tensor([1], new Int32Array([position]));
    const output = await run('decoder', feed, ['logits', ...(heatmap && inAnswer ? [VQA_ATTENTION.name] : [])],
      position === 0 ? {} : continuation);
    const logits = new Float32Array(output.logits.inline.slice().buffer);
    if (logits.length !== 1536 || !logits.every(Number.isFinite)) throw new Error('모델의 토큰 출력이 올바르지 않습니다.');
    let next = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[next]) next = i;
    if (heatmap && inAnswer && next !== answerClose && next !== tokenizer.eos) {
      const tensor = output[VQA_ATTENTION.name];
      if (tensor.dtype !== pb.DataType.DATA_TYPE_F32 || tensor.inline.byteLength !== 8 * 402 * 4) {
        throw new Error('VQA 교차 어텐션 출력 형식이 올바르지 않습니다.');
      }
      const weights = crossAttentionImageWeights(new Float32Array(tensor.inline.slice().buffer));
      for (let index = 0; index < 210; index++) attention[index] += weights[index];
      attentionTokens++;
    }
    if (next === answerOpen) inAnswer = true;
    if (next === answerClose) inAnswer = false;
    generated.push(next);
    if (next === tokenizer.eos) {
      const captured = heatmap ? vqaAttentionHeatmap(attention, attentionTokens) : undefined;
      return { ...parseGenerated(tokenizer.decode(generated, { errors: 'strict' })),
        family: ['phone', 'address', 'store', 'item_row', 'item_math', 'item_lookup', 'math', 'other'][family],
        tokens: generated.length - 1, ...(captured ? { heatmap: captured } : {}) };
    }
  }
  throw new Error('최대 길이 안에 답변 생성이 끝나지 않았습니다. 영수증을 직접 확인해 주세요.');
}

export const modelAdapter = {
  cacheKey: ({ imageHash, question }) => JSON.stringify([imageHash, cleanText(question)]),
  answer: (result) => result,
  async open() {
    const { openSessions, pb } = await import('../shared/extension/runtime.js');
    const documents = await Promise.all(['model-manifest.json', 'vocab.json'].map(async (name) => {
      let response = await fetch(new URL(`./${name}`, import.meta.url));
      if (!response.ok && response.status === 404 && name === 'model-manifest.json') {
        response = await fetch(new URL('./manifest.json', import.meta.url));
      }
      if (!response.ok) throw new Error(`모델 파일을 읽지 못했습니다: ${name}`);
      return response.json();
    }));
    const [manifest, vocab] = documents;
    if (manifest.format !== 'receipt-vqa-volvoxai/v1' || manifest.question_capacity !== 192
        || manifest.max_decoder_steps !== 191 || manifest.initial_masked_cache_rows !== 1) {
      throw new Error('지원하지 않는 VQA 모델 구성입니다.');
    }
    const tokenizer = await TinyReceiptByteFallbackBPEVocab.fromJSON(vocab, { pad: 0, bos: 1, eos: 2, unk: 3 }, {
      vocabSize: manifest.tokenizer.vocab_size, normalization: manifest.tokenizer.normalization,
      tokenizerHash: manifest.tokenizer.tokenizer_hash,
    });
    const models = Object.fromEntries(['encoder', 'decoder'].map((name) => {
      const weights = manifest.graphs[name].weights ?? [`${name}/model.safetensors`];
      if (!Array.isArray(weights) || !weights.length || new Set(weights).size !== weights.length
          || weights.some((path) => typeof path !== 'string' || path.includes('..')
            || !new RegExp(`^(?:${name}|shared)/[a-zA-Z0-9._-]+\\.safetensors$`).test(path))) {
        throw new Error('지원하지 않는 VQA 가중치 파일 구성입니다.');
      }
      return [name, { graph: new URL(`./${name}/graph.json`, import.meta.url),
        weights: weights.map((path) => new URL(`./${path}`, import.meta.url)),
        packedArithmetic: manifest.quantization?.packed_arithmetic ?? 'f32',
        exposeOutputs: name === 'decoder' ? [VQA_ATTENTION] : [] }];
    }));
    const sessions = await openSessions(models);
    return {
      infer: ({ image, question, heatmap = false }) => generate({ pb, tokenizer, image: preprocessPixels(image), question, heatmap,
        run: (role, feed, outputs, bindings = {}) => sessions.run(role,
          Object.entries(feed).map(([name, tensor]) => ({ ...tensor, name: manifest.graphs[role].inputs[name] })), outputs,
          { reuse: (bindings.reuse ?? []).map((name) => manifest.graphs[role].inputs[name]),
            feedback: (bindings.feedback ?? []).map(({ inputName, outputName }) => ({
              inputName: manifest.graphs[role].inputs[inputName], outputName,
            })) }),
      }),
      close: sessions.close,
      profile: sessions.profile,
    };
  },
};
