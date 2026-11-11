import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { InferencePool } from '../extension/inference-pool.js';
import { matchesSite, validateRead, REQUEST_TIMEOUT_MS } from '../extension/requests.js';

const read = { type: 'read', image: 'data:image/png;base64,YQ==', question: 'private question' };
function workerFactory() {
  const workers = [];
  class Worker {
    constructor() { this.messages = []; this.workerId = `worker-${workers.length}`; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    reply(index, result = { answer: 'private answer', cached: false, inferenceMs: 3, totalMs: 5 }) {
      this.onmessage({ data: { id: this.messages[index].id, ok: true, result,
        workerStatus: { workerId: this.workerId, state: 'ready', initializations: 1, forwardPasses: index + 1, cacheHits: 0 } } });
    }
  }
  return { workers, createWorker: () => new Worker() };
}

test('pool executes parallel reads with global bounded admission and drains its FIFO', async () => {
  const f = workerFactory();
  let idle = 0;
  const pool = new InferencePool({ ...f, settings: { worker_count: 2, max_pending: 3 }, onIdle: () => idle++ });
  const requests = [pool.read(read), pool.read(read), pool.read(read)];
  assert.equal(f.workers.length, 2);
  assert.equal(pool.status().active, 2);
  assert.equal(pool.status().queued, 1);
  assert.equal(pool.status().pending, 3);
  assert.equal((await pool.read(read)).ok, false);
  f.workers[0].reply(0);
  assert.equal(f.workers[0].messages.length, 2, 'An available worker immediately takes the queued job');
  f.workers[1].reply(0);
  assert.equal(idle, 0, 'Queued work is still executing');
  f.workers[0].reply(1);
  assert((await Promise.all(requests)).every((response) => response.ok));
  assert.equal(idle, 1);
  assert.equal(pool.status().initializations, 2);
  assert.equal(pool.status().forwardPasses, 3);
  assert.equal(pool.status().pending, 0);
  assert.equal(pool.terminate(), true);
  assert(f.workers.every((worker) => worker.terminated));
});

test('pool resizing waits for admitted work to finish and recreates only the new capacity', async () => {
  const f = workerFactory();
  const pool = new InferencePool({ ...f, settings: { worker_count: 2, max_pending: 4 } });
  const requests = [pool.read(read), pool.read(read), pool.read(read)];
  pool.configure({ worker_count: 1, max_pending: 1 });
  assert.equal(pool.terminate(), false);
  assert.equal((await pool.read(read)).ok, false, 'New smaller queue limit applies immediately');
  f.workers[0].reply(0);
  f.workers[1].reply(0);
  assert(!f.workers[0].terminated, 'The running final job is protected from reconfiguration');
  f.workers[0].reply(1);
  await Promise.all(requests);
  assert(f.workers.every((worker) => worker.terminated));
  assert.equal(pool.status().state, 'idle');
  const again = pool.read(read);
  assert.equal(f.workers.length, 3);
  f.workers[2].reply(0);
  await again;
  pool.terminate();
});

test('a worker timeout frees its heap, settles its request, and ignores late events after restart', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000 });
  const f = workerFactory();
  const pool = new InferencePool(f);
  const request = pool.read(read);
  t.mock.timers.tick(REQUEST_TIMEOUT_MS);
  assert.equal((await request).ok, false);
  assert.equal(f.workers[0].terminated, true);
  assert.equal(pool.status().state, 'idle');
  const again = pool.read(read);
  f.workers[0].reply(0);
  assert.equal(pool.status().pending, 1);
  f.workers[1].reply(0);
  assert.equal((await again).ok, true);
  pool.terminate();
});

test('profiling controls bounded, clearable diagnostics that exclude receipt contents', async () => {
  const f = workerFactory();
  const pool = new InferencePool(f);
  const first = pool.read(read);
  f.workers[0].reply(0);
  await first;
  assert.equal(pool.status().diagnostics, undefined);
  pool.configure({ profiling: true });
  for (let index = 1; index <= 35; index++) {
    const request = pool.read(read);
    assert.equal(f.workers[0].messages[index].profiling, true);
    f.workers[0].reply(index);
    await request;
  }
  assert.equal(pool.status().diagnostics.length, 32);
  const serialized = JSON.stringify(pool.status().diagnostics);
  for (const secret of [read.image, read.question, 'private answer']) assert(!serialized.includes(secret));
  pool.clearDiagnostics();
  assert.equal(pool.status().diagnostics.length, 0);
  const measured = pool.read(read);
  f.workers[0].reply(36);
  await measured;
  assert.equal(pool.status().diagnostics.length, 1);
  // An obsolete stored debug flag cannot keep logging after profiling is off.
  pool.configure({ debug: true, profiling: false });
  assert.equal(pool.status().diagnostics, undefined);
  assert.equal(pool.diagnostics.length, 0);
  const unmeasured = pool.read(read);
  assert.equal(f.workers[0].messages[37].profiling, false);
  f.workers[0].reply(37);
  await unmeasured;
  assert.equal(pool.diagnostics.length, 0);
  pool.terminate();
});

function firefoxFixture(t) {
  t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000 });
  const f = workerFactory();
  let listener, settingsListener;
  const settings = { matches: ['https://custom.example/*'], worker_count: 1, max_pending: 2, idle_timeout_ms: 1_000 };
  const event = { addListener() {} };
  const api = { runtime: {
    id: 'reader@example.test', getURL: (path) => 'moz-extension://test/' + path,
    onMessage: { addListener(value) { listener = value; } }, onInstalled: event, onStartup: event,
  }, permissions: { onAdded: event, onRemoved: event } };
  const source = readFileSync(new URL('../extension/firefox-background.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '');
  vm.runInNewContext(source, {
    browser: api, RECEIPT_READER_SITES: { channel: 'reader' },
    matchesSite, validateRead, IDLE_TIMEOUT_MS: 1_000,
    InferencePool: class extends InferencePool { constructor(options) { super({ ...options, createWorker: f.createWorker }); } },
    readSettings: async () => settings, subscribeSettings(callback) { settingsListener = callback; },
    initializeContentScripts: async () => {}, syncContentScripts: async () => {},
    Date, setTimeout, clearTimeout, Promise,
  });
  const content = { id: api.runtime.id, tab: { id: 1 }, url: 'https://custom.example/receipt' };
  const popup = { id: api.runtime.id, url: 'moz-extension://test/shared/extension/popup.html' };
  return {
    ...f, content, popup,
    send(type = 'read', sender = type === 'read' ? content : popup, overrides = {}) {
      return listener({ ...read, type, channel: 'reader', target: 'background', ...overrides }, sender);
    },
    configure(value) { Object.assign(settings, value); settingsListener(settings); },
  };
}

test('one preview replaces the last result, never enters diagnostics and clears on disable or release', async () => {
  const f = workerFactory();
  const pool = new InferencePool({ ...f, settings: { profiling: true } });
  const map = { kind: 'attention', width: 2, height: 1, values: [.2, .8] };
  for (let i = 0; i < 3; i++) {
    const request = { ...read, image: `${read.image}${i}` };
    const response = pool.read({ ...request, heatmap: false });
    assert.equal(f.workers[0].messages[i].heatmap, true, 'Saved preference overrides content message');
    f.workers[0].reply(i, { answer: 'private answer', heatmap: map });
    await response;
    assert.equal(pool.visualization().image, request.image);
    assert.deepEqual(pool.visualization().heatmap, map);
    assert(!JSON.stringify(pool.status()).includes('private answer'));
    assert(!JSON.stringify(pool.status()).includes('data:image'));
  }
  pool.configure({ heatmap_enabled: false });
  assert.equal(pool.visualization(), null);
  const pending = pool.read({ ...read, heatmap: true });
  assert.equal(f.workers[0].messages[3].heatmap, false);
  f.workers[0].reply(3, { answer: 'private answer', heatmap: map });
  await pending;
  assert.equal(pool.visualization(), null, 'Late worker responses cannot restore disabled preview');
  pool.configure({ heatmap_enabled: true });
  const last = pool.read(read);
  f.workers[0].reply(4, { answer: 'private answer', heatmap: map });
  await last;
  assert(pool.visualization());
  pool.terminate();
  assert.equal(pool.visualization(), null);
});

test('Firefox authorizes current configured hosts, rejects forged messages, and status never allocates a worker', async (t) => {
  const f = firefoxFixture(t);
  assert.equal(f.send('read', f.content, { channel: 'other' }), undefined);
  for (const [type, sender, overrides] of [
    ['read', { ...f.content, id: 'another-extension' }],
    ['read', { ...f.content, url: 'https://custom.example.evil.test/' }],
    ['read', { ...f.content, url: 'https://smartstore.naver.com/' }],
    ['read', f.popup], ['read', f.content, { image: 'https://example.test/private.png' }],
    ['status', f.content], ['clear-diagnostics', f.content], ['close-idle-offscreen', f.popup],
  ]) assert.equal((await f.send(type, sender, overrides)).ok, false);
  assert.equal((await f.send('status')).result.state, 'idle');
  assert.equal(f.workers.length, 0);
  const request = f.send();
  await new Promise(setImmediate);
  f.workers[0].reply(0);
  assert.equal((await request).ok, true);
  t.mock.timers.tick(1_000);
});

test('Firefox polling cannot extend idle lifetime; settings update the deadline and active reads remain protected', async (t) => {
  const f = firefoxFixture(t);
  const first = f.send();
  await new Promise(setImmediate);
  f.workers[0].reply(0);
  await first;
  t.mock.timers.tick(500);
  assert.equal((await f.send('status')).result.state, 'ready');
  t.mock.timers.tick(500);
  assert.equal(f.workers[0].terminated, true);
  const second = f.send();
  await new Promise(setImmediate);
  t.mock.timers.tick(1_500);
  assert(!f.workers[1].terminated);
  f.workers[1].reply(0);
  await second;
  f.configure({ idle_timeout_ms: 2_000 });
  t.mock.timers.tick(1_500);
  assert(!f.workers[1].terminated);
  assert.equal((await f.send('status')).result.pending, 0);
  t.mock.timers.tick(500);
  assert.equal(f.workers[1].terminated, true);
  assert.equal((await f.send('status')).result.state, 'idle');
});
