#!/usr/bin/env node
// Full-corpus Digit accuracy/timing with explicit runtime and native references.
import {createHash} from 'node:crypto';
import {readFile, writeFile, mkdir, open, rename, stat} from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import puppeteer from 'puppeteer';
import {sha256Pattern, validateDigitReference, validateDigitCheckpoint, summarizeDigitRecords} from './digit-evaluation.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const arg = (name, fallback) => process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : fallback;
if (process.argv.includes('--help')) {
  console.log('Usage: node benchmarks/digit_wasm_eval.mjs --fixtures FILE --runtime-dir DIR --out DIR [--models name,...] [--runtime-version VERSION] [--runtime-source published-npm] [--warmups 4] [--resume]');
  process.exit(0);
}
for (const key of ['--fixtures', '--runtime-dir', '--out']) if (!arg(key)) throw new Error(key + ' is required');
const fixturePath = path.resolve(arg('--fixtures')), fixtureRoot = path.dirname(fixturePath);
const runtime = path.resolve(arg('--runtime-dir')), out = path.resolve(arg('--out'));
const resume = process.argv.includes('--resume'), warmups = Number(arg('--warmups', '4'));
if (!Number.isInteger(warmups) || warmups < 1) throw new Error('Warmups must be a positive integer');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const fixtureBytes = await readFile(fixturePath), fixture = JSON.parse(fixtureBytes);
if (fixture.format !== 'unaveraged-digit-evaluation-fixtures/v1' || !fixture.cases?.length || !fixture.models)
  throw new Error('Unexpected Digit evaluation fixture');
const names = arg('--models', Object.keys(fixture.models).join(',')).split(',');
if (!names.length || new Set(names).size !== names.length || names.some(name => !/^[a-zA-Z0-9_-]+$/.test(name) || !fixture.models[name]))
  throw new Error('Unknown, duplicate or unsafe model name');
const atol = fixture.atol ?? 0.001, rtol = fixture.rtol ?? 0.0005;
if (!Number.isFinite(atol) || atol <= 0 || !Number.isFinite(rtol) || rtol < 0) throw new Error('Invalid native tolerance');
const files = new Map(), runtimeHashes = {};
for (const file of ['volvoxai.lite.min.js', 'volvoxai.lite.wasm']) {
  const bytes = await readFile(path.join(runtime, file));
  runtimeHashes[file] = hash(bytes); files.set('/runtime/' + file, {bytes});
}
const header = files.get('/runtime/volvoxai.lite.min.js').bytes.toString().split('\n')[0]
  .match(/^\/\*! volvoxai-browser-provenance-v1 (\{.*\}) \*\/$/);
if (!header) throw new Error('Runtime release provenance is missing');
const provenance = JSON.parse(header[1]);
if (provenance.format !== 'volvoxai-browser-provenance/v1' || provenance.profile !== 'inference'
  || provenance.capability !== 'inference' || provenance.sidecar !== 'volvoxai.lite.wasm'
  || JSON.stringify(provenance.backends) !== '["wasm"]' || !/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(provenance.packageVersion))
  throw new Error('Expected a versioned lite WASM release');
if (arg('--runtime-version') && arg('--runtime-version') !== provenance.packageVersion) throw new Error('Runtime version differs from requested release');
if (!files.get('/runtime/volvoxai.lite.wasm').bytes.subarray(0, 8).equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0])))
  throw new Error('Invalid WASM module');
const sources = {};
for (const name of ['benchmarks/digit_wasm_eval.mjs', 'benchmarks/digit-evaluation-worker.js', 'benchmarks/digit-evaluation.js',
  'digit/adapter.js', 'shared/extension/preprocess.js']) {
  const bytes = await readFile(path.join(root, name)); sources[name] = hash(bytes);
  const url = name.startsWith('benchmarks/') ? '/' + path.basename(name) : '/' + name;
  files.set(url, {bytes});
}
const ids = new Set(), orderedInputs = createHash('sha256');
const cases = fixture.cases.map((item, index) => {
  if (typeof item.id !== 'string' || ids.has(item.id) || !sha256Pattern.test(item.sha256)
    || !item.record || !/^\d+$/.test(item.record.phone) || !/^\d+$/.test(item.record.street))
    throw new Error('Invalid evaluation input identity or numeric label');
  ids.add(item.id); orderedInputs.update(Buffer.from(item.sha256, 'hex'));
  const file = path.resolve(fixtureRoot, item.file);
  if (!file.startsWith(fixtureRoot + path.sep)) throw new Error('Input pixels must remain beside the fixture');
  const url = '/inputs/' + index + '.f32'; files.set(url, {file});
  return {...item, servingPath: url};
});
const inputFingerprint = orderedInputs.digest('hex');
if (fixture.ordered_inputs_sha256 && fixture.ordered_inputs_sha256 !== inputFingerprint) throw new Error('Input order fingerprint differs');
const server = http.createServer(async (request, response) => {
  try {
    if (request.url === '/') { response.writeHead(200, {'Content-Type': 'text/html'}); response.end('<!doctype html><title>Digit evaluation</title>'); return; }
    const item = files.get(request.url);
    if (!item) { response.writeHead(404); response.end(); return; }
    const type = request.url.endsWith('.js') ? 'text/javascript' : request.url.endsWith('.wasm') ? 'application/wasm'
      : request.url.endsWith('.json') ? 'application/json' : 'application/octet-stream';
    response.writeHead(200, {'Content-Type': type, 'Cache-Control': 'no-store'});
    response.end(item.bytes ?? await readFile(item.file));
  } catch { response.writeHead(500); response.end('Fixture error'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = 'http://127.0.0.1:' + server.address().port;
let browser;
try {
  await mkdir(out, {recursive: true});
  for (const name of names) {
    const model = fixture.models[name], modelDir = path.resolve(fixtureRoot, model.directory), modelHashes = {};
    if (!['f32', 'dynamic-i8'].includes(model.packed_arithmetic)) throw new Error('Explicit packed arithmetic required');
    let modelBytes = 0;
    for (const file of ['graph.json', 'model.safetensors']) {
      const bytes = await readFile(path.join(modelDir, file)); modelHashes[file] = hash(bytes); modelBytes += bytes.length;
      if (!sha256Pattern.test(model.sha256?.[file]) || model.sha256[file] !== modelHashes[file]) throw new Error('Model hash differs from fixture: ' + name);
      files.set('/model/' + file, {bytes});
    }
    if (typeof model.references_file !== 'string') throw new Error('Model native references_file is required');
    const referenceBytes = await readFile(path.resolve(fixtureRoot, model.references_file));
    const referenceFile = JSON.parse(referenceBytes), referenceIdentity = referenceFile.identity;
    if (referenceFile.format !== 'unaveraged-digit-native-references/v1' || referenceFile.status !== 'passed'
      || referenceFile.model !== name || referenceIdentity?.fixture_sha256 !== hash(fixtureBytes)
      || referenceIdentity.packed_arithmetic !== model.packed_arithmetic || referenceIdentity.runtime?.version !== provenance.packageVersion
      || Object.keys(referenceIdentity.model_sha256 ?? {}).length !== 2
      || Object.entries(modelHashes).some(([file, expected]) => referenceIdentity.model_sha256[file] !== expected))
      throw new Error('Native reference data/model/arithmetic/runtime identity differs: ' + name);
    const references = referenceFile.cases;
    if (!Array.isArray(references) || references.length !== cases.length) throw new Error('Native reference count differs');
    references.forEach((reference, index) => {
      validateDigitReference(reference, cases[index].id);
      if (hash(Buffer.from(Float32Array.from(reference.values).buffer)) !== reference.sha256)
        throw new Error('Native reference logits/hash mismatch: ' + name + ' index ' + index);
    });
    const identity = {fixture_sha256: hash(fixtureBytes), references_sha256: hash(referenceBytes),
      model: name, model_sha256: modelHashes, runtime_sha256: runtimeHashes, sources_sha256: sources,
      samples: cases.length, warmups, tolerance: {atol, rtol}, ordered_inputs_sha256: inputFingerprint,
      split: fixture.split ?? 'heldout', packed_arithmetic: model.packed_arithmetic};
    const fingerprint = hash(JSON.stringify(identity)), directory = path.join(out, name);
    const recordsPath = path.join(directory, 'predictions.private.jsonl'), identityPath = path.join(directory, 'identity.json');
    const reportPath = path.join(directory, 'evaluation.json'), records = [];
    let previousReport;
    const exists = await stat(directory).then(() => true, error => {
      if (error.code === 'ENOENT') return false;
      throw error;
    });
    if (resume && exists) {
      const previous = JSON.parse(await readFile(identityPath, 'utf8'));
      if (previous.fingerprint !== fingerprint) throw new Error('Cannot resume changed data, model, runtime, protocol or code');
      const checkpoint = await readFile(recordsPath, 'utf8'), lines = checkpoint.split('\n');
      for (let index = 0; index < lines.length; index++) {
        if (!lines[index] && index === lines.length - 1) continue;
        let row;
        try { row = JSON.parse(lines[index]); }
        catch (error) {
          if (index !== lines.length - 1 || checkpoint.endsWith('\n')) throw error;
          await writeFile(recordsPath, records.map(item => JSON.stringify(item) + '\n').join(''));
          console.log('DIGIT_EVAL recovered incomplete checkpoint tail', name); break;
        }
        validateDigitCheckpoint(row, records.length, name, cases, references); records.push(row);
      }
      previousReport = JSON.parse(await readFile(reportPath, 'utf8'));
    } else {
      await mkdir(directory, {recursive: false});
      await writeFile(identityPath, JSON.stringify({fingerprint, ...identity}, null, 2) + '\n');
      await writeFile(recordsPath, '');
    }
    const report = {format: 'unaveraged-digit-wasm-evaluation/v1', status: 'running', created_at: new Date().toISOString(),
      fingerprint, ...identity, runtime: {package: 'volvoxai-lite', version: provenance.packageVersion,
        source: arg('--runtime-source', 'explicit-directory'), sha256: runtimeHashes, provenance},
      model_identity: {method: model.method, conv_weight_bits: model.conv_weight_bits, linear_weight_bits: model.linear_weight_bits,
        graph_weight_bytes: modelBytes}, backend: 'wasm', cpu_threads: 1, batch_size: 1,
      input_source: fixture.input_source ?? 'User-provided heldout images; private preprocessed pixel fixtures',
      native_reference_runtime: referenceIdentity.runtime,
      execution_context: 'Chrome dedicated module worker', host_idle_claimed: false, resumed_records: records.length,
      scoring: 'Exact complete phone+street numeric record; individual fields and supplied question answers counted separately',
      latency_scope: 'One warm execution per heldout input, after cold execution and four (or requested) warmups on input zero; no batching',
      timer_includes: ['Execute dispatch, input/output transport and copied inline slot logits'],
      timer_excludes: ['pixel loading', 'image preprocessing', 'model load/compile/context', 'request construction',
        'numeric accuracy/parity scoring and hashing', 'checkpoint writes', 'UI and extension queue'],
      private_predictions: 'predictions.private.jsonl',
      ...(previousReport?.browser ? {browser: previousReport.browser} : {}),
      ...(previousReport?.cold_verification_load_compile_context_ms !== undefined ? {
        cold_verification_load_compile_context_ms: previousReport.cold_verification_load_compile_context_ms,
        first_execute_ms: previousReport.first_execute_ms} : {})};
    const save = async () => {
      Object.assign(report, summarizeDigitRecords(records, cases.length));
      await writeFile(reportPath + '.tmp', JSON.stringify(report, null, 2) + '\n');
      await rename(reportPath + '.tmp', reportPath);
    };
    let writer, page;
    try {
      await save(); writer = await open(recordsPath, 'a');
      if (records.length < cases.length) {
        browser ??= await puppeteer.launch({browser: 'chrome', headless: true,
          executablePath: process.env.RECEIPT_CHROME || undefined,
          args: ['--no-sandbox', '--disable-dev-shm-usage'], protocolTimeout: 43200000});
        report.browser = await browser.version();
        if (previousReport?.browser && previousReport.browser !== report.browser)
          throw new Error('Cannot resume with a changed browser version');
        page = await browser.newPage();
        await page.exposeFunction('persistDigitRecord', async row => {
          validateDigitCheckpoint(row, records.length, name, cases, references);
          await writer.appendFile(JSON.stringify(row) + '\n'); await writer.sync(); records.push(row);
          if (records.length % 100 === 0 || records.length === cases.length) {
            await save(); console.log('DIGIT_EVAL', name, records.length, '/', cases.length,
              'record', report.metrics.record_correct.correct, 'median_ms', report.latency.median_ms);
          }
        });
        await page.goto(origin); await page.bringToFront();
        const spec = {startIndex: records.length, warmups, atol, rtol, jsUrl: origin + '/runtime/volvoxai.lite.min.js',
          wasmUrl: origin + '/runtime/volvoxai.lite.wasm', references,
          model: {name, sha256: modelHashes, packed_arithmetic: model.packed_arithmetic, base: origin + '/model'},
          cases: cases.map(item => ({id: item.id, sha256: item.sha256, record: item.record,
            ...(item.questions ? {questions: item.questions} : {question: item.question, answer: item.answer}),
            url: origin + item.servingPath}))};
        Object.assign(report, await page.evaluate(async spec => new Promise((resolve, reject) => {
          const worker = new Worker('/digit-evaluation-worker.js', {type: 'module'});
          worker.onmessage = async ({data}) => {
            if (data.type === 'record') {
              try { await window.persistDigitRecord(data.record); worker.postMessage({type: 'record-ack', index: data.record.index}); }
              catch (error) { worker.terminate(); reject(error); }
              return;
            }
            worker.terminate(); data.ok ? resolve(data.result) : reject(new Error(data.error.message));
          };
          worker.onerror = error => { worker.terminate(); reject(new Error(error.message)); };
          worker.postMessage(spec);
        }), spec));
      }
      if (records.length !== cases.length) throw new Error('Evaluation finished before every input was persisted');
      const parity = summarizeDigitRecords(records, cases.length).native_parity;
      report.status = parity.slot_mismatches || parity.numerical_mismatches ? 'completed-with-numerical-differences' : 'passed';
      if (report.status !== 'passed') process.exitCode = 1;
    } catch (error) {
      report.status = 'failed'; report.error = {type: error.name, message: error.message}; process.exitCode = 1;
    } finally { await page?.close(); await writer?.close(); await save(); }
    console.log('DIGIT_EVAL_RESULT', name, report.status, records.length, '/', cases.length);
  }
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
