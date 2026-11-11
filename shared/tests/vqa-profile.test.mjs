import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const source = await readFile(new URL('../../benchmarks/vqa-wasm-worker.js', import.meta.url), 'utf8');

async function fixture(t, {changeEncoder = false, dropped = 0} = {}) {
  const calls = [], closed = [], summaries = [], imported = [];
  const image = new Uint8Array(320 * 672 * 4), modelBytes = new Uint8Array([1, 2, 3]);
  let clock = 0, encoderCalls = 0;
  t.mock.method(performance, 'now', () => clock);
  const realDigest = crypto.subtle.digest.bind(crypto.subtle);
  t.mock.method(crypto.subtle, 'digest', async (...args) => { clock += 100; return realDigest(...args); });
  t.mock.method(globalThis, 'fetch', async url => new Response(url === '/image' ? image
    : url.endsWith('/vocab.json') ? '{}' : modelBytes));
  class EngineHost {
    constructor({wasmUrl}) { this.url = wasmUrl; }
    async close() { closed.push(this.url); }
  }
  class VxInferenceServiceClient {
    constructor(host) { this.host = host; }
    async createRuntime() { return {runtimeId: 1n}; }
    async loadModel(request) { return {modelId: request.graphPath.includes('/encoder/') ? 1n : 2n}; }
    async compileModel(request) { return {report: {backend: 'wasm'}, compiledModelId: request.modelId,
      packedArithmetic: request.packedArithmetic}; }
    async createExecutionContext(request) { return {contextId: request.compiledModelId}; }
    async execute(request) {
      const encoder = request.contextId === 1n;
      calls.push(encoder ? 'encoder' : 'decoder'); clock += 5;
      if (encoder) encoderCalls++;
      return {report: {backend: 'wasm'}, outputs: request.outputs.selection.names.map(name => {
        const values = name === 'logits' ? new Float32Array(1536) : name === 'selected_family_ids'
          ? new Int32Array([0]) : new Float32Array([changeEncoder && encoderCalls >= 3 ? 1 : 0]);
        if (name === 'logits') values[2] = 1;
        return {name, dtype: values instanceof Int32Array ? 2 : 1, shape: [BigInt(values.length)],
          inline: new Uint8Array(values.buffer)};
      })};
    }
  }
  class VxProfilingServiceClient {
    async startTrace() { calls.push('trace-start'); return {traceId: 1n}; }
    async stopTrace() { calls.push('trace-stop'); }
    async releaseTrace() { calls.push('trace-release'); }
    async getTraceSummary(request) {
      summaries.push(request.groupBy);
      return {truncated: false, events: {dropped: BigInt(dropped)}, excludedEvents: 0n,
        rows: [{name: 'PackedLinear', backend: 'wasm', sourceNodeIds: ['dense'], count: 3n,
          totalNs: 15000000n, medianNs: 5000000n, copyBytes: 0n}]};
    }
  }
  const enums = {PackedArithmetic: {PACKED_ARITHMETIC_DYNAMIC_I8: 1, PACKED_ARITHMETIC_F32: 2},
    BackendPolicyMode: {BACKEND_POLICY_MODE_REQUIRE: 1}, OperatorFallback: {OPERATOR_FALLBACK_FORBID: 1},
    DataType: {DATA_TYPE_F32: 1, DATA_TYPE_I32: 2}, OutputStorage: {OUTPUT_STORAGE_INLINE: 1},
    TraceDetail: {TRACE_DETAIL_NODES: 1}, TraceSummaryGrouping: {TRACE_SUMMARY_GROUPING_OPERATOR: 2,
      TRACE_SUMMARY_GROUPING_NODE: 1, TRACE_SUMMARY_GROUPING_ACTIVITY: 4}};
  const pb = new Proxy(enums, {get: (target, name) => target[name] ?? class {
    constructor(fields) { Object.assign(this, fields); }
  }});
  const tokenizer = {bos: 1, eos: 2, encode: () => [2], decode: () => 'answer'};
  const CROSS = ['cross_k_0', 'cross_v_0'];
  const generate = async ({run}) => {
    await run('encoder', {}, ['memory_padding_mask', 'selected_family_ids', ...CROSS]);
    await run('decoder', {}, ['logits']);
    return {answer: 'answer'};
  };
  const fixtureKey = 'vqaProfileTest' + Math.random().toString(36).slice(2);
  const api = {EngineHost, VxInferenceServiceClient, VxProfilingServiceClient, pb};
  globalThis[fixtureKey] = {loadRuntime: async url => { imported.push(url); return api; },
    generate, cleanText: text => text.trim(), CROSS, PAST: [], PRESENT: [],
    TinyReceiptByteFallbackBPEVocab: {fromJSON: async () => tokenizer}, self: {postMessage() {}}};
  t.after(() => delete globalThis[fixtureKey]);
  const script = `const {loadRuntime, generate, cleanText, CROSS, PAST, PRESENT, TinyReceiptByteFallbackBPEVocab, self} = globalThis[${JSON.stringify(fixtureKey)}];\n`
    + source.replace(/^import .*;\n/gm, '')
      .replace("await import('/runtime/volvoxai.lite.min.js')", "await loadRuntime('/runtime/volvoxai.lite.min.js')");
  const {runVqaBenchmark} = await import('data:text/javascript;base64,' + Buffer.from(script).toString('base64'));
  const manifest = {tokenizer: {}, quantization: {packed_arithmetic: 'dynamic-i8'},
    graphs: {encoder: {inputs: {}}, decoder: {inputs: {}}}};
  const spec = {cycles: 1, warmups: 1, fixedSteps: 0, profileOperators: true, wasmUrl: 'default',
    models: {one: {manifest, base: '/model', sha256: {'graph.json': hash(modelBytes)}}},
    cases: [{id: 'public', question: 'question', url: '/image', sha256: hash(image)}]};
  return {run: () => runVqaBenchmark(spec), calls, summaries, closed, imported};
}

test('encoder validation excludes hashing from timings and traces only after measured work', async t => {
  const f = await fixture(t), report = await f.run(), model = report.models.one;
  assert.equal(report.status, 'passed');
  assert.equal(report.repeated_encoder_outputs_bit_exact, true);
  assert.equal(model.cases[0].samples[0].generation_ms, 10);
  assert.equal(model.cases[0].samples[0].encoder_rpc_ms, 5);
  assert.equal(model.cases[0].samples[0].encoder_output_copy_ms, 0);
  assert.equal(model.cases[0].samples[0].encoder_output_bytes, 16);
  assert.equal(Object.keys(model.cases[0].output.encoder_outputs).length, 4);
  assert.deepEqual(f.calls, ['encoder', 'decoder', 'encoder', 'decoder', 'encoder', 'decoder',
    'trace-start', 'encoder', 'encoder', 'encoder', 'trace-stop', 'trace-release']);
  assert.deepEqual(f.summaries, [2, 1, 4]);
  assert.equal(model.encoder_profile.operators[0].mean_per_execute_ms, 5);
  assert.deepEqual(f.closed, ['default']);
  assert.deepEqual(f.imported, ['/runtime/volvoxai.lite.min.js']);
});

test('encoder byte drift fails even with unchanged greedy tokens', async t => {
  const f = await fixture(t, {changeEncoder: true});
  await assert.rejects(f.run, /Measured encoder outputs changed/);
  assert.deepEqual(f.closed, ['default']);
});

test('incomplete traces fail and release their trace and host', async t => {
  const f = await fixture(t, {dropped: 1});
  await assert.rejects(f.run, /Incomplete encoder trace/);
  assert.equal(f.calls.at(-1), 'trace-release');
  assert.deepEqual(f.closed, ['default']);
});
