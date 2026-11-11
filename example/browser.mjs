// Real-model smoke test for the staged Pages demo. No live receipt site is used.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';

const root = fileURLToPath(new URL('../', import.meta.url));
const arg = (name, fallback) => process.argv.includes(name) ? process.argv[process.argv.indexOf(name) + 1] : fallback;
const browserName = arg('--browser', 'chrome');
const site = path.resolve(root, arg('--site', 'dist/pages'));
const results = path.join(root, 'test-results', 'example');
const lifecycle = process.argv.includes('--lifecycle');
const idleMs = 2_000;
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.wasm': 'application/wasm', '.jpg': 'image/jpeg', '.png': 'image/png' };
const server = http.createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const resolved = path.resolve(site, '.' + pathname + (pathname.endsWith('/') ? 'index.html' : ''));
    if (!resolved.startsWith(site + path.sep)) throw new Error('Outside site');
    let data = await fs.readFile(resolved);
    if (lifecycle && pathname === '/example/app.js') {
      const source = data.toString();
      assert(source.includes('IDLE_TIMEOUT_MS = 60_000'));
      data = Buffer.from(source.replace('IDLE_TIMEOUT_MS = 60_000', `IDLE_TIMEOUT_MS = ${idleMs}`));
    }
    response.writeHead(200, { 'Content-Type': types[path.extname(resolved)] || 'application/octet-stream' });
    response.end(data);
  } catch { response.writeHead(404); response.end('Not found'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
const report = { browser: browserName, passed: false, lifecycle, productionIdleTimeoutMs: 60_000, checks: [], results: [] };
const pass = message => { report.checks.push(message); console.log('PASS', message); };
await fs.mkdir(results, { recursive: true });
try {
  browser = await puppeteer.launch({ browser: browserName, headless: true,
    ...(browserName === 'chrome' ? { args: ['--no-sandbox', '--disable-dev-shm-usage'] } : {}) });
  report.version = await browser.version();
  if (process.env.DEMO_BROWSER_DEBUG && browserName === 'firefox') browser.connection.on('log.entryAdded', entry => console.error('BIDI LOG', JSON.stringify(entry)));
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 1000 });
  const requests = [], errors = [];
  page.on('request', request => requests.push(request.url()));
  page.on('pageerror', error => { errors.push(String(error)); console.error('PAGE ERROR', error.stack || String(error)); });
  const installAudit = () => {
    const audit = globalThis.__demoAudit = { created: 0, terminated: 0, posted: 0, results: [], frames: 0, requests: [] };
    const Original = Worker;
    globalThis.Worker = class extends Original {
      constructor(...args) {
        super(...args); audit.created++;
        this.addEventListener('message', ({ data }) => {
          if (data.type === 'result' && data.ok) {
            const map = data.result.heatmap;
            audit.results.push({ answer: data.result.answer, loaded: data.loaded, inferenceMs: data.inferenceMs,
              heatmap: map ? { kind: map.kind, width: map.width, height: map.height,
                min: Math.min(...map.values), max: Math.max(...map.values), sum: map.values.reduce((a,b)=>a+b,0) } : null });
          }
        });
      }
      postMessage(...args) { audit.posted++; audit.requests.push({ question: args[0].question, imageUrl: args[0].imageUrl }); return super.postMessage(...args); }
      terminate() { audit.terminated++; return super.terminate(); }
    };
    const originalFrame = requestAnimationFrame;
    globalThis.requestAnimationFrame = (...args) => { audit.frames++; return originalFrame(...args); };
  };
  await page.goto(origin + '/example/');
  await page.waitForFunction(() => document.body.dataset.ready === 'true');
  assert.equal(page.workers().length, 0, 'No worker before explicit Run');
  await page.evaluate(installAudit);
  assert.equal(await page.evaluate(() => __demoAudit.created), 0);
  assert(!requests.some(url => url.includes('/runtime/')), 'Model assets must stay unloaded until Run');
  pass('Initial page loads no model, WASM or inference worker');
  const samples = JSON.parse(await fs.readFile(path.join(site, 'example/samples.json'), 'utf8')).samples;
  assert.equal(samples.filter(sample => sample.source === 'synth').length, 256);
  assert.equal(await page.$$eval('#sample option', options => options.length), samples.filter(sample => sample.models.includes('digit')).length);
  assert.equal(requests.filter(url => /\.(?:jpg|png)(?:$|\?)/.test(url)).length, 1);
  assert.equal(await page.$eval('#question', node => node.tagName), 'TEXTAREA');
  assert.equal(await page.$eval('#question', node => node.maxLength), 512);
  assert(await page.$$eval('#question-preset option', options => options.length) >= 8);
  pass('All 256 synth examples are selectable without preloading their images; question text is editable');
  const typeQuestion = text => page.$eval('#question', (node, value) => {
    node.value = value; node.dispatchEvent(new Event('input', { bubbles: true }));
  }, text);
  const run = async (expected, width) => {
    await page.click('#run');
    await page.waitForFunction(() => document.body.dataset.result === 'ready', { timeout: 120_000, polling: 50 });
    if (expected !== undefined) assert.equal(await page.$eval('#answer', node => node.textContent), expected);
    const result = await page.evaluate(() => __demoAudit.results.at(-1));
    if (width !== undefined) {
      assert.equal(result.heatmap?.kind, 'attention');
      assert.equal(result.heatmap.width, width);
      assert.equal(result.heatmap.height, 10);
      assert(Math.abs(result.heatmap.sum - 1) < .0001);
      assert(result.heatmap.max > result.heatmap.min);
      assert.equal(await page.$eval('[data-receipt-heatmap] canvas', canvas => canvas.width), width);
    }
    report.results.push(result);
  };
  await page.click('label.switch');
  await run('6', 42);
  assert.equal(await page.$eval('[data-receipt-heatmap]', layer => layer.style.display), 'none');
  const posted = await page.evaluate(() => __demoAudit.posted);
  await page.click('label.switch');
  assert.equal(await page.$eval('[data-receipt-heatmap]', layer => layer.style.display), 'block');
  assert.equal(await page.evaluate(() => __demoAudit.posted), posted);
  pass('Hiding before Run and showing afterward uses the captured map without more inference');
  pass('Digit gives the expected answer and a real nonuniform 42×10 attention grid');
  if (lifecycle) {
    await page.waitForFunction(() => __demoAudit.terminated === 1, { timeout: 8_000, polling: 50 });
    await run('6', 42);
    assert.equal(await page.evaluate(() => __demoAudit.created), 2);
    assert.equal((await page.evaluate(() => __demoAudit.results.at(-1))).loaded, true);
    pass('Idle terminates the worker and a later Run initializes a new model');
  } else {
    await page.screenshot({ path: path.join(results, `${browserName}-desktop.png`), fullPage: true });
    await new Promise(resolve => setTimeout(resolve, 100));
    const before = await page.evaluate(() => ({ posted: __demoAudit.posted, frames: __demoAudit.frames }));
    const network = requests.length;
    await new Promise(resolve => setTimeout(resolve, 1_200));
    const after = await page.evaluate(() => ({ posted: __demoAudit.posted, frames: __demoAudit.frames }));
    assert.deepEqual(after, before, 'Idle UI must not schedule inference or animation frames');
    assert.equal(requests.length, network, 'Idle UI must not poll the network');
    pass('Idle has no inference messages, animation loop or network polling');
    const freeform = '  영수증의 가게 주소에서 도로명 뒤 숫자는 무엇입니까?  ';
    await typeQuestion(freeform);
    assert.equal(await page.evaluate(() => __demoAudit.posted), before.posted);
    await run('699', 42);
    assert.equal((await page.evaluate(() => __demoAudit.results.at(-1))).loaded, false);
    assert.equal(await page.evaluate(() => __demoAudit.requests.at(-1).question), freeform.trim());
    assert.equal(await page.$eval('#answer-check', node => node.hidden), false);
    pass('Freeform text is sent unchanged except trimming; known answers are checked and the model is reused');
    const beforeUnknown = await page.evaluate(() => __demoAudit.posted);
    await typeQuestion('이 영수증의 배경색은 무엇입니까?');
    assert.equal(await page.evaluate(() => __demoAudit.posted), beforeUnknown);
    assert.equal(await page.$eval('#answer', node => node.textContent), '—');
    await run('답변 없음');
    assert.equal(await page.$eval('#answer-check', node => node.hidden), true);
    pass('Unknown custom questions have no fabricated expected-answer badge');
    const beforeSample = await page.evaluate(() => ({ posted: __demoAudit.posted, created: __demoAudit.created }));
    await page.select('#question-preset', 'phone-back-2');
    assert((await page.$eval('#question', node => node.value)).includes('2'));
    await page.select('#sample', 'synth-000000');
    await page.waitForFunction(() => document.querySelector('#receipt-image').complete && document.querySelector('#receipt-image').naturalWidth > 0);
    assert.deepEqual(await page.evaluate(() => ({ posted: __demoAudit.posted, created: __demoAudit.created })), beforeSample);
    assert.equal(requests.filter(url => url.includes('/synth/') && url.endsWith('.png')).length, 1);
    await run();
    assert.equal((await page.evaluate(() => __demoAudit.results.at(-1))).loaded, false);
    assert((await page.evaluate(() => __demoAudit.requests.at(-1).imageUrl)).endsWith('/synth/000000.png'));
    pass('Sample and preset changes are inert until Run, load only the selected image and reuse the model');
    await page.setViewport({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(results, `${browserName}-mobile.png`), fullPage: true });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    pass('Mobile layout fits the viewport');
    await page.setViewport({ width: 1280, height: 1000 });
  }
  await page.click('input[name=model][value=vqa]');
  await page.select('#sample', 'model-vqa');
  await page.select('#question-preset', 'source-0');
  assert(await page.$$eval('#question-preset option', options => options.length) >= 20);
  assert(await page.$$eval('#question-preset optgroup', groups => groups.length) >= 5);
  await run('착한 도매슈퍼마켓', 21);
  pass('VQA gives the expected answer and a real nonuniform 21×10 answer-token attention grid');
  if (!lifecycle) await page.screenshot({ path: path.join(results, `${browserName}-vqa.png`), fullPage: true });
  await page.click('#release-memory');
  assert.equal(await page.evaluate(() => __demoAudit.created), await page.evaluate(() => __demoAudit.terminated));
  pass('Explicit memory release terminates every created worker');
  assert(requests.every(url => url.startsWith(origin)), 'Demo must not send data or load scripts from third parties');
  assert.deepEqual(errors, []);
  report.passed = true;
} catch (error) {
  report.error = String(error.stack || error);
  throw error;
} finally {
  await fs.writeFile(path.join(results, `${browserName}${lifecycle ? '-lifecycle' : ''}.json`), JSON.stringify(report, null, 2) + '\n');
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
