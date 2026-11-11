// The demo starts work only for an explicit Run click. One worker owns one model.
let reader, adapter, activeAdapter, running = false;

async function imagePixels(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('예제 영수증을 불러오지 못했습니다.');
  const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none' });
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0);
    return context.getImageData(0, 0, bitmap.width, bitmap.height);
  } finally { bitmap.close(); }
}

self.onmessage = async ({ data: request }) => {
  if (running) return;
  running = true;
  const progress = stage => self.postMessage({ id: request.id, type: 'progress', stage });
  try {
    if (new URL(request.adapterUrl).origin !== self.location.origin
        || new URL(request.imageUrl).origin !== self.location.origin) throw new Error('지원하지 않는 모델 또는 이미지 경로입니다.');
    let loadMs = 0, loaded = false;
    if (!reader || activeAdapter !== request.adapterUrl) {
      progress('loading');
      await reader?.close();
      reader = null;
      const started = performance.now();
      ({ modelAdapter: adapter } = await import(request.adapterUrl));
      reader = await adapter.open();
      activeAdapter = request.adapterUrl;
      loadMs = performance.now() - started;
      loaded = true;
    }
    const image = await imagePixels(request.imageUrl);
    progress('reading');
    const started = performance.now();
    const record = await reader.infer({ image, question: request.question, heatmap: Boolean(request.heatmap) });
    const result = adapter.answer(record, request.question);
    const inferenceMs = performance.now() - started;
    self.postMessage({ id: request.id, type: 'result', ok: true, result, inferenceMs, loadMs, loaded });
  } catch (error) {
    await reader?.close().catch(() => {});
    reader = null;
    self.postMessage({ id: request.id, type: 'result', ok: false, error: error.message || '모델 실행에 실패했습니다.' });
  } finally { running = false; }
};
