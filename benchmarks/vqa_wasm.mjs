#!/usr/bin/env node
// Benchmark arbitrary batch-one VQA packages without changing extension assets.
import {createHash} from 'node:crypto';
import {readFile, writeFile, mkdir, stat} from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import puppeteer from 'puppeteer';

const root = fileURLToPath(new URL('../', import.meta.url));
const runtimeVersion = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8')).config.volvoxai_version;
const arg = (name, fallback) => process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : fallback;
if (process.argv.includes('--help')) {
  console.log('Usage: node benchmarks/vqa_wasm.mjs --fixtures FILE --out NEW_DIR [--runtime-dir DIR] [--cycles 5] [--warmups 1] [--fixed-steps 16] [--models w4a8,w8a8] [--profile-operators]');
  process.exit(0);
}
const installedVersion = JSON.parse(await readFile(path.join(root, 'node_modules/volvoxai-lite/package.json'), 'utf8')).version;
if (installedVersion !== runtimeVersion) throw new Error('Installed volvoxai-lite differs from the project pin; run npm ci');
if (!arg('--fixtures') || !arg('--out')) throw new Error('--fixtures and --out are required');
const fixturesFile = path.resolve(arg('--fixtures'));
const fixtureRoot = path.dirname(fixturesFile);
const out = path.resolve(arg('--out'));
const published = path.join(root, 'node_modules/volvoxai-lite/dist', runtimeVersion);
const runtime = path.resolve(arg('--runtime-dir', published));
const cycles = Number(arg('--cycles', '5')), warmups = Number(arg('--warmups', '1'));
const fixedSteps = Number(arg('--fixed-steps', '0'));
if (!Number.isInteger(cycles) || cycles < 1 || !Number.isInteger(warmups) || warmups < 0 ||
    !Number.isInteger(fixedSteps) || fixedSteps < 0 || fixedSteps > 191) throw new Error('Invalid timing counts');
const fixture = JSON.parse(await readFile(fixturesFile, 'utf8'));
if (fixture.format !== 'unaveraged-vqa-wasm-fixtures/v1' || !fixture.cases?.length || !Object.keys(fixture.models ?? {}).length)
  throw new Error('Invalid VQA fixture');
if (fixture.cases.some(item => typeof item.id !== 'string') ||
    new Set(fixture.cases.map(item => item.id)).size !== fixture.cases.length)
  throw new Error('Benchmark case IDs must be unique strings');
const selected = arg('--models', Object.keys(fixture.models).join(',')).split(',');
if (!selected.length || new Set(selected).size !== selected.length || selected.some(name => !fixture.models[name]))
  throw new Error('Unknown or duplicate --models');
fixture.models = Object.fromEntries(selected.map(name => [name, fixture.models[name]]));
await mkdir(out, {recursive: false});
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const files = new Map();
const runtimeHashes = {};
for (const file of ['volvoxai.lite.min.js', 'volvoxai.lite.wasm']) {
  files.set('/runtime/' + file, path.join(runtime, file));
  runtimeHashes[file] = hash(await readFile(path.join(runtime, file)));
}
async function runtimeIdentity(directory, sha256) {
  const js = await readFile(path.join(directory, 'volvoxai.lite.min.js'), 'utf8');
  const match = js.match(/^\/\*! volvoxai-browser-provenance-v1 (\{[^\n]+\}) \*\//);
  if (!match) throw new Error('Runtime JavaScript lacks release provenance');
  const provenance = JSON.parse(match[1]);
  if (provenance.packageVersion !== runtimeVersion || provenance.capability !== 'inference' || provenance.sidecar !== 'volvoxai.lite.wasm' ||
      !/^\d+\.\d+\.\d+(?:[-+].+)?$/.test(provenance.packageVersion))
    throw new Error('Unexpected runtime provenance');
  return {package: 'volvoxai-lite', version: provenance.packageVersion, sha256,
    source: path.resolve(directory) === path.resolve(published) ? 'published' : 'explicit-directory', provenance};
}
const runtimeInfo = await runtimeIdentity(runtime, runtimeHashes);
files.set('/worker.js', path.join(root, 'benchmarks/vqa-wasm-worker.js'));
for (const file of ['vqa/adapter.js', 'vqa/tokenizer.js', 'shared/extension/preprocess.js']) files.set('/' + file, path.join(root, file));
const models = {};
for (const [name, item] of Object.entries(fixture.models)) {
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) throw new Error('Unsafe model name');
  const directory = path.resolve(fixtureRoot, item.directory);
  const manifest = JSON.parse(await readFile(path.join(directory, 'manifest.json'), 'utf8'));
  if (manifest.format !== 'receipt-vqa-volvoxai/v1' || manifest.question_capacity !== 192 || manifest.max_decoder_steps !== 191)
    throw new Error('Unexpected VQA model contract');
  const names = new Set(['manifest.json', 'config.json', 'vocab.json']);
  for (const role of ['encoder', 'decoder']) {
    names.add(role + '/graph.json');
    for (const file of manifest.graphs[role].weights ?? [role + '/model.safetensors']) names.add(file);
  }
  const sha256 = {}; let packageBytes = 0;
  for (const file of names) {
    if (path.isAbsolute(file) || file.split('/').includes('..')) throw new Error('Unsafe model shard path');
    const local = path.join(directory, file);
    sha256[file] = hash(await readFile(local));
    if (item.sha256?.[file] && item.sha256[file] !== sha256[file]) throw new Error('Model differs from fixture hash');
    packageBytes += (await stat(local)).size;
    files.set('/models/' + name + '/' + file, local);
  }
  models[name] = {manifest, sha256, package_bytes: packageBytes};
}
const cases = fixture.cases.map((item, index) => {
  const local = path.resolve(fixtureRoot, item.file);
  if (!local.startsWith(fixtureRoot + path.sep)) throw new Error('Inputs must remain beside the fixture manifest');
  files.set('/inputs/' + index + '.f32', local);
  return {...item, servingPath: '/inputs/' + index + '.f32'};
});
const report = {format: 'unaveraged-vqa-wasm/v1', status: 'running', created_at: new Date().toISOString(),
  fixture_sha256: hash(await readFile(fixturesFile)), runtime: runtimeInfo,
  input_source: fixture.input_source ?? 'Public synthetic VQA examples',
  sources: Object.fromEntries(await Promise.all(['benchmarks/vqa_wasm.mjs', 'benchmarks/vqa-wasm-worker.js',
    'vqa/adapter.js', 'vqa/tokenizer.js'].map(async file => [file, hash(await readFile(path.join(root, file)))])))};
const server = http.createServer(async (request, response) => {
  try {
    if (request.url === '/') {
      response.writeHead(200, {'Content-Type': 'text/html'});
      response.end('<!doctype html><title>VQA WASM benchmark</title>'); return;
    }
    const file = files.get(request.url);
    if (!file) { response.writeHead(404); response.end(); return; }
    const type = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm'
      : file.endsWith('.json') ? 'application/json' : 'application/octet-stream';
    response.writeHead(200, {'Content-Type': type}); response.end(await readFile(file));
  } catch { response.writeHead(500); response.end('Fixture error'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = 'http://127.0.0.1:' + server.address().port;
const spec = {cycles, warmups, fixedSteps, profileOperators: process.argv.includes('--profile-operators'),
  wasmUrl: origin + '/runtime/volvoxai.lite.wasm',
  models: Object.fromEntries(Object.entries(models).map(([name, model]) => [name, {...model, base: origin + '/models/' + name}])),
  cases: cases.map(item => ({...item, url: origin + item.servingPath}))};
let browser;
try {
  browser = await puppeteer.launch({browser: 'chrome', headless: true,
    executablePath: process.env.RECEIPT_CHROME || undefined,
    args: ['--no-sandbox', '--disable-dev-shm-usage'], protocolTimeout: 3600000});
  report.browser = await browser.version();
  const page = await browser.newPage();
  await page.goto(origin); await page.bringToFront();
  page.on('console', message => console.log(message.text()));
  const result = await page.evaluate(async spec => new Promise((resolve, reject) => {
    const worker = new Worker('/worker.js', {type: 'module'});
    worker.onmessage = ({data}) => {
      if (data.type === 'progress') { console.log('VQA_WASM ' + JSON.stringify(data)); return; }
      worker.terminate();
      data.ok ? resolve(data.result) : reject(new Error(data.error.message));
    };
    worker.onerror = error => { worker.terminate(); reject(new Error(error.message)); };
    worker.postMessage(spec);
  }), spec);
  Object.assign(report, result);
  for (const [file, expected] of Object.entries(runtimeHashes))
    if (hash(await readFile(path.join(runtime, file))) !== expected) throw new Error('Runtime changed during measurement');
  for (const [file, expected] of Object.entries(report.sources))
    if (hash(await readFile(path.join(root, file))) !== expected) throw new Error('Benchmark source changed during measurement: ' + file);
} catch (error) {
  report.status = 'failed'; report.error = {type: error.name, message: error.message};
  process.exitCode = 1;
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
  await writeFile(path.join(out, 'wasm-chrome.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log('VQA_WASM_RESULT', report.status);
if (report.native_token_parity_passed === false) process.exitCode = 1;
