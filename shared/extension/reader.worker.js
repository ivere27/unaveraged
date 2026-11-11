// Model-independent worker. A different model supplies the same adapter contract.
import { modelAdapter } from '../../model/adapter.js';
import { SerialQueue, MAX_PENDING, MAX_PIXELS, MAX_IMAGE_BYTES, validateRead } from './requests.js';

const queue = new SerialQueue(MAX_PENDING);
const records = new Map();
const workerId = crypto.randomUUID();
let readerPromise;
let state = 'idle';
let initializations = 0;
let forwardPasses = 0;
let cacheHits = 0;

function getReader() {
  if (!readerPromise) {
    state = 'loading';
    readerPromise = modelAdapter.open().then((reader) => {
      initializations++;
      state = 'ready';
      return reader;
    }).catch((error) => { readerPromise = null; state = 'error'; throw error; });
  }
  return readerPromise;
}

async function read(request) {
  validateRead(request);
  const start = performance.now();
  const raw = atob(request.image.slice(request.image.indexOf(',') + 1));
  if (raw.length > MAX_IMAGE_BYTES) throw new Error('이미지는 4 MB 이하여야 합니다.');
  const bytes = Uint8Array.from(raw, (char) => char.charCodeAt(0));
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    (value) => value.toString(16).padStart(2, '0')).join('');
  // The adapter includes every input that changes the model result.
  const key = JSON.stringify([modelAdapter.cacheKey({ imageHash: hash, question: request.question }), Boolean(request.heatmap)]);
  let record = records.get(key);
  const cached = records.has(key);
  let inferenceMs = 0;
  let profile;
  if (!cached) {
    const bitmap = await createImageBitmap(new Blob([bytes]), { colorSpaceConversion: 'none' });
    let image;
    try {
      if (bitmap.width * bitmap.height > MAX_PIXELS) throw new Error('이미지 해상도가 너무 큽니다.');
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.drawImage(bitmap, 0, 0);
      image = context.getImageData(0, 0, bitmap.width, bitmap.height);
    } finally {
      bitmap.close();
    }
    const reader = await getReader();
    const started = performance.now();
    try {
      if (request.profiling && reader.profile) {
        const measured = await reader.profile(() => reader.infer({ image, question: request.question, heatmap: Boolean(request.heatmap) }));
        record = measured.value;
        profile = measured.profile;
      } else record = await reader.infer({ image, question: request.question, heatmap: Boolean(request.heatmap) });
    } catch (error) {
      await reader.close().catch(() => {});
      readerPromise = null;
      state = 'error';
      throw error;
    }
    inferenceMs = performance.now() - started;
    forwardPasses++;
    if (records.size >= 32) records.delete(records.keys().next().value);
    records.set(key, record);
  } else {
    cacheHits++;
    records.delete(key);
    records.set(key, record);
  }
  return {
    ...modelAdapter.answer(record, request.question),
    cached, inferenceMs, totalMs: performance.now() - start,
    ...(profile ? { profile } : {}),
  };
}

function workerStatus() {
  return { state, workerId, initializations, forwardPasses, cacheHits, pending: queue.pending };
}

self.onmessage = ({ data: message }) => {
  if (message.type === 'status') {
    self.postMessage({ id: message.id, ok: true, result: {
      state, workerId, initializations, forwardPasses, cacheHits, pending: queue.pending,
    } });
    return;
  }
  queue.run(() => read(message)).then(
    (result) => self.postMessage({ id: message.id, ok: true, result, workerStatus: workerStatus() }),
    (error) => self.postMessage({ id: message.id, ok: false, error: error.message || String(error), workerStatus: workerStatus() }),
  );
};
