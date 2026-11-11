#!/usr/bin/env node
// Interleave frozen Digit models on the same small heldout-input subset.
import {createHash} from 'node:crypto';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import puppeteer from 'puppeteer';

const root = fileURLToPath(new URL('../', import.meta.url));
const arg = (name, fallback) => process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : fallback;
if (process.argv.includes('--help')) {
  console.log('Usage: node benchmarks/digit_latency.mjs --fixtures FILE --runtime-dir DIR --runtime-version VERSION --out FILE [--runtime-source published-npm]');
  process.exit(0);
}
for (const key of ['--fixtures', '--runtime-dir', '--runtime-version', '--out']) if (!arg(key)) throw new Error(key + ' is required');
const fixturePath = path.resolve(arg('--fixtures')), fixtureRoot = path.dirname(fixturePath);
const out = path.resolve(arg('--out')), runtime = path.resolve(arg('--runtime-dir'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const fixtureBytes = await readFile(fixturePath), fixture = JSON.parse(fixtureBytes);
if (fixture.format !== 'unaveraged-digit-latency-fixtures/v1' || fixture.cases?.length !== 3
  || JSON.stringify(fixture.source_indices) !== '[0,999,1999]' || !Object.keys(fixture.models ?? {}).length
  || fixture.cycles !== 15 || fixture.warmups !== 4 || fixture.native_reference_runtime?.version !== arg('--runtime-version'))
  throw new Error('Unexpected fixed interleaved latency protocol');
if (await readFile(out).then(() => true, error => { if (error.code === 'ENOENT') return false; throw error; }))
  throw new Error('Refusing to overwrite an existing latency report');
const files = new Map(), runtimeHashes = {}, sources = {};
for (const name of ['volvoxai.lite.min.js', 'volvoxai.lite.wasm']) {
  const bytes = await readFile(path.join(runtime, name)); runtimeHashes[name] = hash(bytes);
  files.set('/runtime/' + name, bytes);
}
const header = files.get('/runtime/volvoxai.lite.min.js').toString().split('\n')[0]
  .match(/^\/\*! volvoxai-browser-provenance-v1 (\{.*\}) \*\/$/);
if (!header) throw new Error('Runtime release provenance missing');
const provenance = JSON.parse(header[1]);
if (provenance.packageVersion !== arg('--runtime-version') || provenance.profile !== 'inference'
  || provenance.capability !== 'inference' || provenance.sidecar !== 'volvoxai.lite.wasm'
  || JSON.stringify(provenance.backends) !== '["wasm"]') throw new Error('Expected the requested lite WASM release');
if (!files.get('/runtime/volvoxai.lite.wasm').subarray(0, 8).equals(Buffer.from([0,97,115,109,1,0,0,0])))
  throw new Error('Invalid WASM module');
for (const [name, url] of Object.entries({'benchmarks/runtime-benchmark.js': '/runner.js',
  'benchmarks/worker.js': '/worker.js', 'benchmarks/digit_latency.mjs': null})) {
  const bytes = await readFile(path.join(root, name)); sources[name] = hash(bytes);
  if (url) files.set(url, bytes);
}
for (const [name, model] of Object.entries(fixture.models)) {
  if (!/^[a-zA-Z0-9_-]+$/.test(name) || !['f32', 'dynamic-i8'].includes(model.packed_arithmetic))
    throw new Error('Invalid model identity/arithmetic');
  for (const file of ['graph.json', 'model.safetensors']) {
    const bytes = await readFile(path.resolve(fixtureRoot, model.directory, file));
    if (hash(bytes) !== model.sha256?.[file]) throw new Error('Model bytes differ from fixed fixture');
    files.set('/models/' + name + '/' + file, bytes);
  }
}
for (const [index, item] of fixture.cases.entries()) {
  const file = path.resolve(fixtureRoot, item.file);
  if (!file.startsWith(fixtureRoot + path.sep)) throw new Error('Input must remain beside fixture');
  const bytes = await readFile(file);
  if (bytes.length !== 320 * 672 * 4 || hash(bytes) !== item.sha256) throw new Error('Input bytes differ');
  files.set('/inputs/' + index + '.f32', bytes);
  for (const name of Object.keys(fixture.models)) {
    const reference = item.references?.[name];
    if (reference?.values?.length !== 176 || reference.slots?.length !== 16
      || !reference.values.every(Number.isFinite) || !reference.slots.every(n => Number.isInteger(n) && n >= 0 && n <= 10)
      || hash(Buffer.from(Float32Array.from(reference.values).buffer)) !== reference.sha256)
      throw new Error('Invalid native output reference: ' + name);
    const slots = Array.from({length: 16}, (_, row) => {
      let best = 0; for (let col = 1; col < 11; col++) if (reference.values[row * 11 + col] > reference.values[row * 11 + best]) best = col;
      return best;
    });
    if (JSON.stringify(slots) !== JSON.stringify(reference.slots)) throw new Error('Native reference slots differ');
  }
}
const server = http.createServer((request, response) => {
  if (request.url === '/') { response.writeHead(200, {'Content-Type': 'text/html'}); response.end('<!doctype html><title>Digit interleaved latency</title>'); return; }
  const bytes = files.get(request.url);
  if (!bytes) { response.writeHead(404); response.end(); return; }
  response.writeHead(200, {'Content-Type': request.url.endsWith('.js') ? 'text/javascript'
    : request.url.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream', 'Cache-Control': 'no-store'});
  response.end(bytes);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = 'http://127.0.0.1:' + server.address().port;
const spec = {...fixture, wasmUrl: origin + '/runtime/volvoxai.lite.wasm',
  models: Object.fromEntries(Object.entries(fixture.models).map(([name, item]) => [name,
    {sha256: item.sha256, packed_arithmetic: item.packed_arithmetic, base: origin + '/models/' + name}])),
  cases: fixture.cases.map((item, index) => ({...item, url: origin + '/inputs/' + index + '.f32'}))};
const report = {format: 'unaveraged-digit-interleaved-latency/v1', status: 'running', engine: 'chrome',
  created_at: new Date().toISOString(), fixture_sha256: hash(fixtureBytes), source_fixture_sha256: fixture.source_fixture_sha256,
  sources_sha256: sources, source_references_sha256: fixture.source_references_sha256,
  input_indices: fixture.source_indices, input_sha256: fixture.cases.map(item => item.sha256),
  runtime: {package: 'volvoxai-lite', version: provenance.packageVersion, source: arg('--runtime-source', 'explicit-directory'),
    sha256: runtimeHashes, provenance},
  protocol: {cycles: 15, warmups_per_input: 4, initial_reference_executions_per_input: 1, input_count: 3,
    samples_per_model: 45, batch_size: 1, cpu_threads: 1, order: 'Alternate forward/reverse model order each cycle, for each fixed input',
    input_selection: fixture.selection, host_idle_claimed: false,
    timing_scope: 'Interleaved repeated model timings; separate from full-corpus accuracy and sequential evaluation-pass latency'}};
report.environment = {hostname: os.hostname(), platform: `${os.type()} ${os.release()} ${os.arch()}`,
  cpu: os.cpus()[0]?.model ?? 'unknown', logical_cpus: os.cpus().length, node: process.version};
report.host_observations = [{phase: 'before', timestamp: new Date().toISOString(), loadavg: os.loadavg()}];
let browser;
try {
  browser = await puppeteer.launch({browser: 'chrome', headless: true, executablePath: process.env.RECEIPT_CHROME || undefined,
    args: ['--no-sandbox', '--disable-dev-shm-usage'], protocolTimeout: 3600000});
  report.version = await browser.version(); report.execution_context = 'Chrome dedicated module worker';
  const page = await browser.newPage(); await page.goto(origin); await page.bringToFront();
  Object.assign(report, await page.evaluate(spec => new Promise((resolve, reject) => {
    const worker = new Worker('/worker.js', {type: 'module'});
    worker.onmessage = ({data}) => { worker.terminate(); data.ok ? resolve(data.result) : reject(new Error(data.error.message)); };
    worker.onerror = error => { worker.terminate(); reject(new Error(error.message)); };
    worker.postMessage(spec);
  }), spec));
  for (const model of Object.values(report.models)) {
    const values = model.samples_ms_per_input.flat().sort((a, b) => a - b), at = (values.length - 1) * 0.95;
    if (values.length !== 45 || values.some(n => !Number.isFinite(n) || n <= 0)) throw new Error('Incomplete/nonfinite latency samples');
    model.mean_ms = values.reduce((a, b) => a + b, 0) / values.length;
    model.p95_ms = values[Math.floor(at)] + (values[Math.ceil(at)] - values[Math.floor(at)]) * (at - Math.floor(at));
  }
  for (const [name, expected] of Object.entries(sources)) if (hash(await readFile(path.join(root, name))) !== expected)
    throw new Error('Benchmark source changed while measuring');
  if (report.status !== 'passed') process.exitCode = 1;
} catch (error) { report.status = 'failed'; report.error = {type: error.name, message: error.message}; process.exitCode = 1; }
finally {
  await browser?.close(); await new Promise(resolve => server.close(resolve));
  report.host_observations.push({phase: 'after', timestamp: new Date().toISOString(), loadavg: os.loadavg()});
  await mkdir(path.dirname(out), {recursive: true}); await writeFile(out, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
}
console.log('DIGIT_INTERLEAVED', report.status,
  Object.fromEntries(Object.entries(report.models ?? {}).map(([name, model]) => [name, model.pooled_median_ms])));
