import { MAX_PENDING, REQUEST_TIMEOUT_MS } from './requests.js';

// Each worker owns one model/WASM heap. The host admits a bounded global FIFO
// and gives each worker one request at a time, keeping queue memory bounded too.
export class InferencePool {
  constructor({ settings = {}, createWorker = () => new Worker(new URL('./reader.worker.js', import.meta.url), { type: 'module' }),
    onActivity = () => {}, onIdle = () => {}, onChange = () => {} } = {}) {
    this.settings = { worker_count: 1, max_pending: MAX_PENDING, profiling: false, heatmap_enabled: true, ...settings };
    this.capacity = this.settings.worker_count;
    this.createWorker = createWorker;
    this.onActivity = onActivity;
    this.onIdle = onIdle;
    this.onChange = onChange;
    this.pending = new Map();
    this.queue = [];
    this.slots = [];
    this.diagnostics = [];
    this.latestVisualization = null;
  }

  configure(settings) {
    this.settings = { ...this.settings, ...settings };
    if (!this.settings.profiling) this.clearDiagnostics();
    if (!this.settings.heatmap_enabled) this.latestVisualization = null;
    if (!this.pending.size && this.capacity !== this.settings.worker_count) {
      this.terminate();
      this.capacity = this.settings.worker_count;
    }
  }

  record(event, fields = {}) {
    if (!this.settings.profiling) return;
    // Never store input images, questions, answers, or untrusted error strings.
    this.diagnostics.push({ at: new Date().toISOString(), event, ...fields });
    if (this.diagnostics.length > 32) this.diagnostics.shift();
  }

  clearDiagnostics() { this.diagnostics.length = 0; }

  visualization() { return this.latestVisualization; }

  status() {
    let result;
    if (!this.slots.length) result = { state: 'idle', pending: this.pending.size };
    else {
      const workers = this.slots.map((slot) => ({ ...slot.status, active: Boolean(slot.job) }));
      result = {
        state: workers.some((item) => item.state === 'loading') ? 'loading'
          : workers.every((item) => item.state === 'error') ? 'error' : 'ready',
        workerId: workers[0].workerId,
        initializations: workers.reduce((sum, item) => sum + (item.initializations || 0), 0),
        forwardPasses: workers.reduce((sum, item) => sum + (item.forwardPasses || 0), 0),
        cacheHits: workers.reduce((sum, item) => sum + (item.cacheHits || 0), 0),
        pending: this.pending.size, active: workers.filter((item) => item.active).length,
        queued: this.queue.length, workerCount: workers.length,
        configuredWorkers: this.settings.worker_count, workers,
      };
    }
    if (this.settings.profiling) result.diagnostics = [...this.diagnostics];
    return result;
  }

  read(request) {
    if (this.pending.size >= this.settings.max_pending) {
      this.record('queue-full', { pending: this.pending.size });
      return Promise.resolve({ ok: false, error: '처리 대기 중인 영수증이 많습니다. 잠시 후 다시 읽어 주세요.' });
    }
    this.onActivity();
    return new Promise((resolve) => {
      const job = { id: crypto.randomUUID(), request, resolve, admitted: performance.now() };
      job.timer = setTimeout(() => this.fail(job, 'timeout', '읽기 시간이 초과되었습니다. 다시 읽어 주세요.'), REQUEST_TIMEOUT_MS);
      this.pending.set(job.id, job);
      this.queue.push(job);
      this.dispatch();
    });
  }

  makeSlot() {
    const worker = this.createWorker();
    const slot = { worker, status: { state: 'loading', workerId: crypto.randomUUID(), initializations: 0, forwardPasses: 0, cacheHits: 0 } };
    this.slots.push(slot);
    worker.onmessage = ({ data }) => {
      const job = slot.job;
      if (!this.slots.includes(slot) || !job || data.id !== job.id) return;
      slot.status = data.workerStatus || { ...slot.status, state: data.ok ? 'ready' : 'error' };
      if (data.ok && this.settings.heatmap_enabled) {
        // One replaceable preview in RAM; never include it in status or exports.
        this.latestVisualization = {
          image: job.request.image, heatmap: data.result.heatmap,
          resultSummary: { answer: data.result.answer, inferenceMs: data.result.inferenceMs,
            totalMs: data.result.totalMs },
        };
      }
      this.record(data.ok ? 'read' : 'read-error', {
        workerId: slot.status.workerId, queueMs: job.queueMs,
        ...(data.ok ? { cached: Boolean(data.result.cached), inferenceMs: data.result.inferenceMs,
          totalMs: data.result.totalMs, ...(data.result.profile ? { profile: data.result.profile } : {}) } : {}),
      });
      this.finish(job, { ok: data.ok, ...(data.ok ? { result: { ...data.result, queueMs: job.queueMs } } : { error: data.error }) });
    };
    worker.onerror = (event) => {
      event.preventDefault();
      if (slot.job) this.fail(slot.job, 'worker-error', '영수증 읽기가 중단되었습니다. 다시 읽어 주세요.');
    };
    worker.onmessageerror = () => {
      if (slot.job) this.fail(slot.job, 'message-error', '읽기 결과를 전달하지 못했습니다. 다시 읽어 주세요.');
    };
    this.record('worker-created', { workers: this.slots.length });
    return slot;
  }

  dispatch() {
    while (this.queue.length) {
      let slot = this.slots.find((entry) => !entry.job);
      if (!slot && this.slots.length >= this.capacity) break;
      const job = this.queue.shift();
      try {
        slot ||= this.makeSlot();
        slot.job = job;
        job.slot = slot;
        job.queueMs = performance.now() - job.admitted;
        slot.worker.postMessage({ ...job.request, id: job.id, type: 'read',
          profiling: this.settings.profiling, heatmap: this.settings.heatmap_enabled });
      } catch {
        this.fail(job, 'worker-start-error', '읽기 작업을 시작하지 못했습니다. 다시 읽어 주세요.');
      }
    }
    this.onChange();
  }

  fail(job, event, error) {
    if (!this.pending.has(job.id)) return;
    if (job.slot) {
      job.slot.worker.terminate();
      this.slots = this.slots.filter((slot) => slot !== job.slot);
    }
    this.record(event);
    this.finish(job, { ok: false, error });
  }

  finish(job, response) {
    clearTimeout(job.timer);
    this.pending.delete(job.id);
    this.queue = this.queue.filter((entry) => entry !== job);
    if (job.slot) job.slot.job = null;
    job.resolve(response);
    if (!this.pending.size) {
      if (this.capacity !== this.settings.worker_count) {
        this.terminate();
        this.capacity = this.settings.worker_count;
      }
      this.onIdle();
    }
    this.dispatch();
  }

  terminate() {
    if (this.pending.size) return false;
    for (const slot of this.slots) slot.worker.terminate();
    if (this.slots.length) this.record('workers-released', { workers: this.slots.length });
    this.slots.length = 0;
    this.latestVisualization = null;
    this.onChange();
    return true;
  }
}
