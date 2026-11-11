import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { testContentSettings, testPoolSettings, testHeatmap, instrumentHeatmap, visualization } from './browser-settings.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const argument = (name, fallback) => process.argv.includes(name)
  ? process.argv[process.argv.indexOf(name) + 1] : fallback;
const requested = argument('--model', 'all');
const precisionArgument = argument('--precision', 'default');
const lifecycle = process.argv.includes('--lifecycle');
if (!['all', 'digit', 'vqa'].includes(requested)) throw new Error('Use --model digit|vqa|all');
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(check, label, timeout = 15_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await check()) return;
    await delay(75);
  }
  throw new Error(`Timed out: ${label}`);
}

async function runProject(project, precision) {
  const config = JSON.parse(await fs.readFile(path.join(root, project, 'extension.json'), 'utf8'));
  const distribution = path.join(root, 'dist', project, precision, 'firefox');
  const results = path.join(root, 'test-results', project, precision);
  const reportName = lifecycle ? 'firefox-lifecycle' : 'firefox';
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'receipt-firefox-'));
  const extension = path.join(temporary, 'extension');
  const idleMs = 2_000;
  const report = { project, precision, passed: false, checks: [], predictions: [], browserErrors: [],
    testCopyChanges: ['Allow local HTTP fixture pages'], productionIdleTimeoutMs: 60_000 };
  const pass = (message) => { report.checks.push(message); console.log(`PASS ${project}: ${message}`); };
  let browser, server;
  await fs.mkdir(results, { recursive: true });
  try {
    await fs.cp(distribution, extension, { recursive: true });
    await instrumentHeatmap(extension);
    report.testCopyChanges.push('Audit real attention raster writes and read-response map presence');
    const buildInfo = JSON.parse(await fs.readFile(path.join(extension, 'build-info.json'), 'utf8'));
    report.runtime = buildInfo.runtime_version;
    report.runtimeSha256 = buildInfo.runtime_sha256;
    report.modelSha256 = buildInfo.model_sha256;
    assert.equal(buildInfo.browser, 'firefox');
    assert.equal(buildInfo.model_precision, precision);
    for (const [folder, hashes] of [['vendor', buildInfo.runtime_sha256], ['model', buildInfo.model_sha256]]) {
      for (const [file, digest] of Object.entries(hashes)) {
        const bytes = await fs.readFile(path.join(extension, folder, file));
        assert.equal(createHash('sha256').update(bytes).digest('hex'), digest, `${folder}/${file} provenance`);
      }
    }
    const manifestFile = path.join(extension, 'manifest.json');
    const manifest = JSON.parse(await fs.readFile(manifestFile, 'utf8'));
    assert.equal(manifest.background.page, 'shared/extension/firefox-background.html');
    assert.equal(manifest.background.persistent, false);
    assert(!manifest.background.service_worker);
    assert(!manifest.permissions?.includes('offscreen'));
    const extensionId = manifest.browser_specific_settings.gecko.id;
    const extensionUuid = randomUUID();
    const localPattern = 'http://127.0.0.1/*';
    for (const script of manifest.content_scripts) script.matches.push(localPattern);
    manifest.host_permissions = [...(manifest.host_permissions || []), localPattern];
    await fs.writeFile(manifestFile, JSON.stringify(manifest));
    await fs.appendFile(path.join(extension, 'shared/extension/site-config.js'),
      `\nglobalThis.RECEIPT_READER_SITES.matches.push(${JSON.stringify(localPattern)});\n`);
    if (lifecycle) {
      report.testCopyChanges.push('Shorten idle timeout to 2 seconds', 'Delay one request past idle timeout', 'Audit Worker construction/termination');
      report.idleTimeoutMsInTestCopy = idleMs;
      const protocolFile = path.join(extension, 'shared/extension/requests.js');
      const protocol = await fs.readFile(protocolFile, 'utf8');
      assert(protocol.includes('IDLE_TIMEOUT_MS = 60_000'));
      await fs.writeFile(protocolFile, protocol.replace('IDLE_TIMEOUT_MS = 60_000', `IDLE_TIMEOUT_MS = ${idleMs}`));
      const settingsFile = path.join(extension, 'shared/extension/settings-core.js');
      const settingsSource = await fs.readFile(settingsFile, 'utf8');
      assert(settingsSource.includes('idle_timeout_ms: 60_000'));
      await fs.writeFile(settingsFile, settingsSource.replace('idle_timeout_ms: 60_000', `idle_timeout_ms: ${idleMs}`)
        .replace("['idle_timeout_ms', 10_000", "['idle_timeout_ms', 1_000"));
      const readerFile = path.join(extension, 'shared/extension/reader.worker.js');
      const reader = await fs.readFile(readerFile, 'utf8');
      assert(reader.includes('async function read(request) {'));
      await fs.writeFile(readerFile, reader.replace('async function read(request) {', `async function read(request) {
        if (request.question?.endsWith(' [delayed]')) {
          await new Promise(resolve => setTimeout(resolve, ${idleMs + 1_000}));
          request = { ...request, question: request.question.slice(0, -10) };
        }`));
      // Observe actual Worker.terminate calls without replacing inference or the lifecycle implementation.
      await fs.writeFile(path.join(extension, 'shared/extension/test-worker-audit.js'), `
        const audit = { created: 0, terminated: 0, active: 0 };
        const instanceId = crypto.randomUUID();
        const NativeWorker = globalThis.Worker;
        globalThis.Worker = class extends NativeWorker {
          #terminated = false;
          constructor(...args) { super(...args); audit.created++; audit.active++; }
          terminate() {
            if (!this.#terminated) { audit.terminated++; audit.active--; this.#terminated = true; }
            return super.terminate();
          }
        };
        browser.runtime.onMessage.addListener(message => {
          if (message?.target === 'test-worker-audit') return Promise.resolve({ ...audit });
          if (message?.target === 'test-background-instance') return Promise.resolve(instanceId);
        });\n`);
      const backgroundFile = path.join(extension, 'shared/extension/firefox-background.html');
      const html = await fs.readFile(backgroundFile, 'utf8');
      assert(html.includes('<script'));
      await fs.writeFile(backgroundFile, html.replace('<script', '<script src="test-worker-audit.js"></script>\n<script'));
    }
    const source = path.join(root, '.cache', project, 'huggingface', config.model.revision);
    const data = JSON.parse(await fs.readFile(path.join(source, config.examples), 'utf8'));
    const receipts = await Promise.all((Array.isArray(data) ? data : data.examples).map(async (receipt) => ({
      ...receipt,
      questions: receipt.questions || [...receipt.cases, ...(receipt.additional_cases || [])]
        .map(item => ({ ...item, answer: item.expected_answer })),
      imageUrl: 'data:image/jpeg;base64,' + (await fs.readFile(path.join(source, 'examples', receipt.image))).toString('base64'),
    })));
    server = http.createServer((request, response) => {
      const receipt = receipts[request.url === '/ko' ? 1 : 0];
      response.setHeader('Content-Type', 'text/html; charset=utf-8');
      response.end(`<!doctype html><html lang="ko"><meta charset="utf-8"><title>Local synthetic receipt</title>
        <main id="rcpt_form"><img id="rcpt_img" class="captcha_img" src="${receipt.imageUrl}">
        <p id="rcpt_info" class="captcha_message"></p><input id="rcpt_answer_fixture" name="captcha"></main>
        <script>window.answerEvents=0;window.requiredAnswerEvents=1;
        document.addEventListener('input',event=>{if(event.target.matches('input[name=captcha]'))window.answerEvents++;});</script></html>`);
    });
    await new Promise(resolve => server.listen(0, '0.0.0.0', resolve));
    const port = server.address().port;
    browser = await puppeteer.launch({ browser: 'firefox', headless: true, dumpio: Boolean(process.env.RECEIPT_FIREFOX_DEBUG),
      ...(process.env.RECEIPT_FIREFOX ? { executablePath: process.env.RECEIPT_FIREFOX } : {}),
      userDataDir: path.join(temporary, 'profile'), timeout: 30_000, protocolTimeout: 180_000,
      extraPrefsFirefox: { 'extensions.webextensions.uuids': JSON.stringify({ [extensionId]: extensionUuid }) },
    });
    report.browser = await browser.version();
    assert.equal(await browser.installExtension(extension), extensionId);
    console.log(`Installed ${project} in ${report.browser}`);
    const watch = page => { page.on('pageerror', error => report.browserErrors.push(String(error))); return page; };
    const openPopup = async () => {
      const page = watch(await browser.newPage());
      // Firefox does not emit BiDi load events for extension pages (Mozilla bug 1959376).
      await page.mainFrame().browsingContext.navigate(`moz-extension://${extensionUuid}/shared/extension/popup.html`, 'none');
      await page.waitForFunction(() => document.readyState === 'complete');
      return page;
    };
    let popup = await openPopup();
    const message = (body) => popup.evaluate((value) => browser.runtime.sendMessage(value), body);
    const status = async () => {
      const response = await message({ channel: config.channel, target: 'background', type: 'status' });
      assert(response?.ok, response?.error || 'No background response');
      return response.result;
    };
    assert.deepEqual(await status(), { state: 'idle', pending: 0 });
    assert.equal(await visualization(popup), null);
    pass('Popup status before use does not initialize the model');
    const pages = await Promise.all([0, 1].map(async (index) => {
      const page = watch(await browser.newPage());
      await page.goto(`http://127.0.0.1:${port}/${index === 0 ? 'en' : 'ko'}`);
      return page;
    }));
    await delay(350);
    for (const page of pages) assert.equal(await page.$('.receipt-reader-panel'), null);
    const question = (page, value) => page.$eval('#rcpt_info', (node, text) => {
      window.requiredAnswerEvents = window.answerEvents + 1;
      node.textContent = text;
    }, value);
    const answer = async (page, expected) => {
      try {
        await page.waitForFunction(() => window.answerEvents >= window.requiredAnswerEvents
          && document.querySelector('.receipt-reader-panel')?.getAttribute('aria-busy') === 'false',
        { timeout: 120_000, polling: 50 });
        const actual = await page.$eval('input[name=captcha]', node => node.value);
        report.predictions.push({ expected, actual, correct: actual === expected });
        assert.equal(actual, expected, 'Public synthetic receipt answer');
      } catch (error) {
        const panel = await page.$eval('.receipt-reader-panel', node => node.textContent).catch(() => 'No panel');
        throw new Error(`Expected ${expected}: ${panel}`, { cause: error });
      }
    };
    if (lifecycle) {
      const audit = () => message({ target: 'test-worker-audit' });
      assert.equal((await audit()).active, 0);
      await question(pages[0], receipts[0].questions[0].question + ' [delayed]');
      await until(async () => (await status()).pending > 0, 'active inference');
      await delay(idleMs + 250);
      assert((await status()).pending > 0, 'An active request must survive the idle deadline');
      assert.equal((await audit()).terminated, 0);
      await answer(pages[0], receipts[0].questions[0].answer);
      const first = await status();
      assert.equal(first.initializations, 1);
      pass('Inference lasting longer than the idle timeout remains active');
      const rejected = await message({ channel: config.channel, target: 'background', type: 'read',
        image: receipts[0].imageUrl, question: receipts[0].questions[0].question });
      assert.equal(rejected.ok, false, 'Extension pages cannot forge a supported-site read');
      pass('Read requests from the popup are rejected');
      await until(async () => (await status()).state === 'idle', 'worker release despite status polling');
      assert.deepEqual(await audit(), { created: 1, terminated: 1, active: 0 });
      assert.equal(await visualization(popup), null, 'Idle release also drops the retained image and attention map');
      pass('Idle calls Worker.terminate; repeated status queries do not retain the WASM worker');
      await question(pages[0], receipts[0].questions[1].question);
      await answer(pages[0], receipts[0].questions[1].answer);
      const second = await status();
      assert.notEqual(second.workerId, first.workerId);
      assert.equal(second.initializations, 1);
      assert.deepEqual(await audit(), { created: 2, terminated: 1, active: 1 });
      pass('The next receipt creates a new worker and loads the model again');
      await until(async () => (await status()).state === 'idle', 'second worker release');
      const instance = await message({ target: 'test-background-instance' });
      await popup.close();
      // No extension views or status calls: allow Firefox's own event-page idle timeout.
      await delay(35_000);
      await question(pages[0], receipts[0].questions[0].question);
      await answer(pages[0], receipts[0].questions[0].answer);
      popup = await openPopup();
      assert.notEqual(await message({ target: 'test-background-instance' }), instance);
      const woken = await status();
      assert.notEqual(woken.workerId, second.workerId);
      assert.equal(woken.initializations, 1);
      pass('Firefox suspends the event page without open extension views and wakes on the next receipt');
      await testPoolSettings({ popup, pages, receipts, status, question, answer, pass });
    } else {
      for (let index = 0; index < Math.max(...receipts.map(receipt => receipt.questions.length)); index++) {
        await Promise.all(pages.map(async (page, i) => {
          const item = receipts[i].questions[index];
          if (!item) return;
          await question(page, item.question);
          await answer(page, item.answer);
        }));
        if (index === 0) {
          const initial = await status();
          assert.equal(initial.initializations, 1);
          assert.equal(initial.forwardPasses, 2);
          pass('Two simultaneous tabs share one model initialization');
        }
      }
      const generated = await status();
      const count = receipts.reduce((sum, receipt) => sum + receipt.questions.length, 0);
      assert.equal(generated.forwardPasses, config.cache_by_question ? count : 2);
      await Promise.all(pages.map(async (page, i) => {
        await question(page, receipts[i].questions[0].question);
        await answer(page, receipts[i].questions[0].answer);
      }));
      const cached = await status();
      assert.equal(cached.workerId, generated.workerId);
      assert.equal(cached.forwardPasses, generated.forwardPasses);
      assert.equal(cached.cacheHits, generated.cacheHits + 2);
      pass(`${count} public fixture questions match expected answers; repeated queries use the cache`);
      await testHeatmap({ popup, pages, receipts, project, status, pass });
      const excluded = watch(await browser.newPage());
      await excluded.goto(`http://127.0.0.2:${port}/en`);
      await question(excluded, receipts[0].questions[0].question);
      await delay(350);
      assert.equal(await excluded.$('.receipt-reader-panel'), null);
      pass('Unlisted sites receive no content script');
      await testContentSettings({ browser, popup, page: pages[0], receipt: receipts[0], pass });
    }
    assert.deepEqual(report.browserErrors, []);
    report.passed = true;
  } catch (error) {
    report.error = String(error.stack || error);
    console.error(report.error, report.browserErrors);
    throw error;
  } finally {
    await fs.writeFile(path.join(results, `${reportName}.json`), JSON.stringify(report, null, 2) + '\n');
    await browser?.close();
    if (server) await new Promise(resolve => server.close(resolve));
    await fs.rm(temporary, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 });
  }
}

for (const project of requested === 'all' ? ['digit', 'vqa'] : [requested]) {
  const config = JSON.parse(await fs.readFile(path.join(root, project, 'extension.json'), 'utf8'));
  const precisions = precisionArgument === 'all'
    ? [...new Set([config.default_precision, 'w8a8'])]
    : [precisionArgument === 'default' ? config.default_precision : precisionArgument];
  for (const precision of precisions) await runProject(project, precision);
}
