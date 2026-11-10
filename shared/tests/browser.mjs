import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
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
const requestedPrecision = argument('--precision', 'default');
const expectedVariant = argument('--variant', 'ptq');
const localRuntimeDir = argument('--runtime-dir', null);
const resultsRoot = path.join(root, 'test-results', ...(localRuntimeDir ? ['local-runtime'] : []));
if (!['all', 'digit', 'vqa'].includes(requested)) throw new Error('Use --model digit|vqa|all');
const frozenProfiles = JSON.parse(await fs.readFile(path.join(root, 'digit/frozen-profiles.json'), 'utf8')).profiles;
const frozenVqaProfiles = JSON.parse(await fs.readFile(path.join(root, 'vqa/frozen-profiles.json'), 'utf8')).profiles;
if (!['default', 'all', 'w8a8', 'w4a8', ...Object.keys(frozenProfiles)].includes(requestedPrecision)) throw new Error('Unknown precision profile');
if (Object.hasOwn(frozenProfiles, requestedPrecision) && requested !== 'digit') throw new Error('Frozen profiles require --model digit');
const runtimeVersion = JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf8')).config.volvoxai_version;
let selectedRuntimeVersion = runtimeVersion;
let runtimeSource = 'published';
if (localRuntimeDir) {
  const javascript = await fs.readFile(path.join(path.resolve(localRuntimeDir), 'volvoxai.lite.min.js'));
  const published = await fs.readFile(path.join(root, 'node_modules/volvoxai-lite/dist', runtimeVersion, 'volvoxai.lite.min.js'));
  runtimeSource = javascript.equals(published) ? 'local-wasm-build' : 'local-dist';
  if (runtimeSource === 'local-dist') {
    const header = javascript.toString().split('\n')[0].match(/^\/\*! volvoxai-browser-provenance-v1 (\{.*\}) \*\/$/);
    assert(header, 'Local runtime requires release provenance');
    selectedRuntimeVersion = JSON.parse(header[1]).packageVersion;
  }
}
async function runProject(project, precision) {
  const config = JSON.parse(await fs.readFile(path.join(root, project, 'extension.json'), 'utf8'));
  const results = path.join(resultsRoot, project, precision);
  const distribution = path.join(root, 'dist', project, precision);
  async function fixtures() {
    const source = path.join(root, '.cache', project, 'huggingface', config.model.revision);
    const data = JSON.parse(await fs.readFile(path.join(source, config.examples), 'utf8'));
    const examples = Array.isArray(data) ? data : data.examples;
    return Promise.all(examples.map(async (example, i) => ({
      ...example, language: example.language || (i === 0 ? 'en' : 'ko'),
      questions: example.questions || [...example.cases, ...(example.additional_cases || [])]
        .map((item) => ({ ...item, answer: item.expected_answer })),
      imageUrl: 'data:image/jpeg;base64,' + (await fs.readFile(path.join(source, 'examples', example.image))).toString('base64'),
    })));
  }

  async function testBrowser() {
    const production = process.argv.includes('--production');
    const reportName = production ? 'chrome' : 'chrome-dev';
    const buildName = reportName + (localRuntimeDir ? '-local' : '');
    const buildRoot = path.join(distribution, buildName);
    const buildInfo = JSON.parse(await fs.readFile(path.join(buildRoot, 'build-info.json'), 'utf8'));
    assert.equal(buildInfo.runtime_version, selectedRuntimeVersion);
    assert.equal(buildInfo.model_precision, precision);
    assert.equal(buildInfo.runtime_profile, 'lite');
    assert.equal(buildInfo.dependencies.javascript.version, selectedRuntimeVersion);
    assert.equal(buildInfo.dependencies.javascript.package, 'volvoxai-lite');
    assert.equal(buildInfo.dependencies.javascript.source, runtimeSource);
    assert.equal(buildInfo.dependencies.python.version, runtimeVersion);
    assert.equal(buildInfo.model_variant, expectedVariant);
    if (expectedVariant === 'ptq') {
      assert.equal(buildInfo.model_quantization.method, frozenProfiles[precision]?.method ?? 'volvoxai-ptq');
      const modelManifest = JSON.parse(await fs.readFile(path.join(buildRoot, 'model/manifest.json'), 'utf8'));
      assert.deepEqual(buildInfo.model_quantization, modelManifest.quantization);
      if (modelManifest.frozen_candidate) {
        const expectedProfile = (project === 'digit' ? frozenProfiles : frozenVqaProfiles)[precision];
        assert(expectedProfile, 'Frozen model profile must be registered');
        assert.deepEqual(modelManifest.frozen_candidate, expectedProfile);
        assert.equal(modelManifest.converter.source, 'frozen-candidate-import');
        assert.equal(modelManifest.quantization.version, modelManifest.converter.version);
        for (const [file, expected] of Object.entries(expectedProfile.sha256)) {
          const bytes = await fs.readFile(path.join(buildRoot, 'model', file));
          assert.equal(createHash('sha256').update(bytes).digest('hex'), expected);
        }
      } else assert.equal(buildInfo.model_quantization.version, runtimeVersion);
      assert.equal(buildInfo.model_quantization.source_variant, 'fp32');
      assert.equal(buildInfo.model_quantization.precision, precision);
      assert.equal(buildInfo.model_quantization.activation_dtype, 'int8');
      const frozen = frozenProfiles[precision];
      assert.equal(buildInfo.model_quantization.packed_arithmetic, frozen || precision === 'w4a8' ? 'dynamic-i8' : 'f32');
      if (config.cache_by_question) {
        for (const role of ['encoder', 'decoder']) assert(buildInfo.model_quantization.roles[role].quantized_nodes > 0);
      } else assert.equal(buildInfo.model_quantization.quantized_convolutions, 11);
      // Verify the packaged graph, so changing only a manifest cannot pass the precision matrix.
      const graphs = config.cache_by_question ? ['encoder/graph.json', 'decoder/graph.json'] : ['graph.json'];
      for (const name of graphs) {
        const graph = JSON.parse(await fs.readFile(path.join(buildRoot, 'model', name), 'utf8'));
        const expectedConvolutions = config.cache_by_question ? (name.startsWith('encoder/') ? 13 : 0) : 11;
        assert.equal(graph.nodes.filter((node) => node.opType === 'QConv2D').length, expectedConvolutions);
        assert.equal(graph.nodes.filter((node) => node.opType === 'Conv2D').length, 0);
        for (const node of graph.nodes.filter((node) => node.opType === 'QConv2D')) {
          assert.equal(node.params.weight_bits ?? 8, frozen?.conv_weight_bits ?? 8,
            `${name}: ${node.id} convolution storage precision`);
          if (frozen?.conv_weight_bits === 4) assert.equal(node.params.kernel_size.length, 2);
        }
        if (frozen || precision === 'w4a8') {
          const packed = graph.nodes.filter((node) => node.opType === 'PackedLinear');
          assert.equal(packed.length, frozen ? 7 : name.startsWith('encoder/') ? 34 : 33);
          for (const node of packed) {
            assert.equal(node.params.weight_bits, frozen?.linear_weight_bits ?? 4, `${name}: ${node.id}`);
            assert.equal(node.params.group_size, buildInfo.model_quantization.group_size, `${name}: ${node.id}`);
            assert(node.params.group_size > 0, 'Packed projections require a positive group size');
          }
        } else {
          assert(graph.nodes.some((node) => ['QConv2D', 'QLinear', 'QMatMul', 'QGemm'].includes(node.opType)),
            `${name} has no static W8A8 operators`);
        }
      }
      assert.equal(buildInfo.model_quantization.calibration.coverage_complete, true);
      const calibration = buildInfo.model_quantization.calibration;
      assert.equal(typeof calibration.evaluation_checked, 'boolean');
      assert.equal(calibration.evaluation_overlap, calibration.evaluation_checked ? 0 : null);
    }
    const runtimeFiles = ['volvoxai.lite.min.js', 'volvoxai.lite.wasm'];
    assert.deepEqual((await fs.readdir(path.join(buildRoot, 'vendor'))).sort(), ['LICENSE.txt', ...runtimeFiles]);
    for (const file of runtimeFiles) {
      const bundled = await fs.readFile(path.join(buildRoot, 'vendor', file));
      assert.equal(createHash('sha256').update(bundled).digest('hex'), buildInfo.runtime_sha256[file],
        `Bundled ${file} must match its build provenance`);
      const runtimeRoot = localRuntimeDir ? path.resolve(localRuntimeDir)
        : path.join(root, 'node_modules/volvoxai-lite', 'dist', runtimeVersion);
      assert.deepEqual(bundled,
        await fs.readFile(path.join(runtimeRoot, file)),
        `Bundled ${file} must match the selected VolvoxAI build`);
    }
    const receipts = await fixtures();
    const escape = (text) => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
    const html = (receipt, question = receipt.questions[0].question) => `<!doctype html><html lang="ko"><meta charset="utf-8"><title>Receipt fixture</title>
    <body><h1 style="font:20px system-ui;color:#172d26">${config.name} — 공개 합성 영수증 데모 / Synthetic receipt demo</h1><main id="rcpt_form"><img id="rcpt_img" class="captcha_img" alt="Captcha Image" src="${receipt.imageUrl}">
    <p id="rcpt_info" class="captcha_message">${escape(question)}</p>
    <input id="rcpt_answer_fixture" name="captcha" aria-label="Answer"><button id="submit">Submit</button></main>
    <script>window.answerEvents=0; window.requiredAnswerEvents=1;
    document.addEventListener('input',event=>{if(event.target.matches('input[name=captcha]')) window.answerEvents++;});</script>
    </body></html>`;
    const server = http.createServer((req, res) => {
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      res.end(html(receipts[req.url?.startsWith('/ko') ? 1 : 0], ''));
    });
    await new Promise((resolve) => server.listen(0, '0.0.0.0', resolve));
    const port = server.address().port;
    const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'receipt-reader-chrome-'));
    await fs.mkdir(results, { recursive: true });
    const pageUrls = production ? ['https://smartstore.naver.com/en', 'https://shopping.naver.com/ns/home']
      : [`http://127.0.0.1:${port}/en`, `http://127.0.0.1:${port}/ko`];
    const report = { project, precision, passed: false, build: buildName, runtime: buildInfo.runtime_version,
      runtimeProfile: buildInfo.runtime_profile, modelVariant: buildInfo.model_variant,
      runtimeSha256: buildInfo.runtime_sha256, modelSha256: buildInfo.model_sha256,
      dependencies: buildInfo.dependencies,
      testCopyChanges: ['Grant existing fixture hosts in an isolated package copy for dynamic registration checks'],
      pageUrls, checks: [], predictions: [], records: [], browser: null };
    let browser;
    const errors = [];
    const watch = (page) => {
      page.setDefaultTimeout(20_000);
      page.on('pageerror', (error) => errors.push(String(error)));
      return page;
    };
    const waitAnswer = async (page, value, record = false) => {
      const started = performance.now();
      try {
        await page.waitForFunction((expected, collect) => window.answerEvents >= window.requiredAnswerEvents
          && document.querySelector('.receipt-reader-panel')?.getAttribute('aria-busy') === 'false'
          && (collect || document.querySelector('input[name=captcha]')?.value === expected),
        { timeout: 120_000, polling: 50 }, value, record);
        const actual = await page.$eval('input[name=captcha]', (node) => node.value);
        if (record) report.predictions.push({
          page: page.url(), question: await page.$eval('#rcpt_info', (node) => node.textContent),
          expected: value, actual, correct: actual === value,
          elapsedMs: Math.round(performance.now() - started),
        });
        if (!record) assert.equal(actual, value, 'Model answer differs from the public fixture');
      } catch (error) {
        const status = await page.$eval('.receipt-reader-panel', (node) => node.textContent).catch(() => 'no panel');
        throw new Error(`Expected answer ${value}; ${status}`, { cause: error });
      }
    };
    const setQuestion = (page, question) => page.$eval('#rcpt_info', (node, value) => {
      window.requiredAnswerEvents = window.answerEvents + 1;
      node.textContent = value;
    }, question);

    try {
      const extension = path.join(profile, 'extension-test');
      await fs.cp(buildRoot, extension, { recursive: true });
      await instrumentHeatmap(extension);
      report.testCopyChanges.push('Audit real attention raster writes and read-response map presence');
      const manifestPath = path.join(extension, 'manifest.json');
      const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
      // Static matches alone do not grant Chrome's scripting API host access.
      // Simulate granting those fixture hosts only in this disposable test copy.
      manifest.host_permissions = [...new Set([...(manifest.host_permissions || []),
        ...manifest.content_scripts.flatMap(script => script.matches)])];
      await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
      browser = await puppeteer.launch({
        executablePath: process.env.RECEIPT_CHROME || await puppeteer.executablePath(),
        headless: true, enableExtensions: true, userDataDir: profile, protocolTimeout: 180_000,
        args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-extension-debugging'],
      });
      report.browser = await browser.version();
      const cdp = await browser.target().createCDPSession();
      const { id } = await cdp.send('Extensions.loadUnpacked', { path: extension });
      assert(id, 'Extension did not load');
      const popup = watch(await browser.newPage());
      await popup.goto(`chrome-extension://${id}/shared/extension/popup.html`);
      const status = async () => {
        const response = await popup.evaluate(() => chrome.runtime.sendMessage({
          channel: globalThis.RECEIPT_READER_SITES.channel, target: 'background', type: 'status',
        }));
        assert(response.ok, response.error);
        return response.result;
      };

      const policy = watch(await browser.newPage());
      await policy.goto(`chrome-extension://${id}/privacy.html`);
      assert((await policy.$eval('body', (node) => node.textContent)).includes(config.name.split(' · ')[0]));
      assert.equal((await popup.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }))).length, 0);
      await policy.close();
      report.checks.push('Bundled privacy policy opens offline without starting inference');

      const pages = [watch(await browser.newPage()), watch(await browser.newPage())];
      await Promise.all(pages.map(async (page, index) => {
        if (production) {
          // Exercise both supported hosts without contacting Naver's servers.
          await page.setRequestInterception(true);
          page.on('request', (request) => {
            if (request.isNavigationRequest()) void request.respond({
              status: 200, contentType: 'text/html; charset=utf-8', body: html(receipts[index], ''),
            });
            else void request.respond({ status: 204, body: '' });
          });
        }
        await page.goto(pageUrls[index]);
      }));
      await new Promise((resolve) => setTimeout(resolve, 500));
      for (const page of pages) assert.equal(await page.$('.receipt-reader-panel'), null);
      assert.deepEqual(await status(), { state: 'idle', pending: 0 });
      report.checks.push('Supported pages without a question do not show a panel or initialize inference');
      await Promise.all(pages.map((page, index) => setQuestion(page, receipts[index].questions[0].question)));
      await Promise.all(pages.map((page, index) => waitAnswer(page, receipts[index].questions[0].answer, true)));
      for (let i = 0; i < pages.length; i++) {
        if (receipts[i].expected_record) {
          const record = await pages[i].evaluate(() => Object.fromEntries(
            ['phone', 'street'].map((key) => [key, document.querySelector(`[data-field=${key}]`).textContent])));
          report.records.push({ page: pages[i].url(), expected: receipts[i].expected_record, actual: record,
            correct: ['phone', 'street'].every((key) => record[key] === receipts[i].expected_record[key]) });
        }
      }
      const initial = await status();
      assert.equal(initial.initializations, 1);
      assert.equal(initial.forwardPasses, 2);
      report.initial = initial;
      report.checks.push('Two simultaneous tabs complete with one shared runtime initialization');
      console.log('PASS simultaneous tabs, one shared runtime');

      for (let q = 1; q < receipts[0].questions.length; q++) {
        await Promise.all(pages.map((page, i) => setQuestion(page, receipts[i].questions[q].question)));
        await Promise.all(pages.map((page, i) => waitAnswer(page, receipts[i].questions[q].answer, true)));
        if (q === 2) {
          for (let i = 0; i < pages.length; i++) {
            await pages[i].bringToFront();
            await pages[i].setViewport({ width: 1280, height: 800 });
            await pages[i].screenshot({ path: path.join(results, `store-${receipts[i].language}.png`) });
          }
        }
      }
      const generated = await status();
      const questionCount = receipts.reduce((sum, receipt) => sum + receipt.questions.length, 0);
      report.accuracy = { correct: report.predictions.filter((item) => item.correct).length, total: questionCount };
      assert.equal(report.accuracy.correct, questionCount, 'Public fixture answers differ; inspect predictions');
      assert(report.records.every((item) => item.correct), 'Public fixture records differ; inspect records');
      assert.equal(generated.forwardPasses, config.cache_by_question ? questionCount : 2);
      assert.equal(generated.cacheHits, config.cache_by_question ? 0 : questionCount - 2);
      await Promise.all(pages.map((page, i) => setQuestion(page, receipts[i].questions[0].question)));
      await Promise.all(pages.map((page, i) => waitAnswer(page, receipts[i].questions[0].answer)));
      const cached = await status();
      assert.equal(cached.forwardPasses, generated.forwardPasses);
      assert.equal(cached.cacheHits, generated.cacheHits + 2);
      report.checks.push(`${questionCount} HF questions correct; the model-specific cache key reuses valid results`);
      console.log(`PASS ${project}: ${questionCount} HF questions and model-specific cache`);
      await testHeatmap({ popup, pages, receipts, project, status,
        pass: message => { report.checks.push(message); console.log('PASS', message); } });
      if (!config.cache_by_question) {
        await setQuestion(pages[0], '상호명이 무엇입니까?');
        await pages[0].waitForFunction(() => document.querySelector('[data-field=answer]')?.textContent === '지원하지 않는 질문', { polling: 50 });
        assert.equal(await pages[0].$eval('input[name=captcha]', (node) => node.value), '');
        report.checks.push('Digit adapter declines unsupported questions and clears the previous automatic answer');
      }

      const targets = await cdp.send('Target.getTargets');
      const serviceWorker = targets.targetInfos.find((target) => target.type === 'service_worker'
        && target.url === `chrome-extension://${id}/shared/extension/background.js`);
      assert(serviceWorker);
      await cdp.send('Target.closeTarget', { targetId: serviceWorker.targetId });
      const restarted = await status();
      assert.equal(restarted.workerId, initial.workerId);
      assert.equal(restarted.initializations, 1);
      await setQuestion(pages[0], receipts[0].questions.at(-1).question);
      await waitAnswer(pages[0], receipts[0].questions.at(-1).answer);
      report.checks.push('Service-worker restart reconnects to the existing inference worker');
      console.log('PASS background restart');

      await popup.evaluate(() => chrome.offscreen.closeDocument());
      await setQuestion(pages[0], receipts[0].questions[1].question);
      await pages[0].waitForFunction(() => document.querySelector('.receipt-reader-panel')?.getAttribute('aria-busy') === 'true', { polling: 20 });
      await pages[0].$eval('input[name=captcha]', (node) => {
        node.value = 'user edit'; node.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await pages[0].waitForFunction((answer) => document.querySelector('[data-field=answer]')?.textContent === answer
        && document.querySelector('.receipt-reader-panel')?.getAttribute('aria-busy') === 'false', { polling: 50 }, receipts[0].questions[1].answer);
      assert.equal(await pages[0].$eval('input[name=captcha]', (node) => node.value), 'user edit');
      const recovered = await status();
      assert.notEqual(recovered.workerId, initial.workerId);
      assert.equal(recovered.initializations, 1);
      await pages[0].bringToFront();
      await pages[0].click('[data-action=fill]');
      await waitAnswer(pages[0], receipts[0].questions[1].answer);
      report.checks.push('Offscreen recovery; user input survives inference; explicit fill works');
      console.log('PASS worker recreation and preservation of user edits');

      // Cold initialization gives a real in-flight request while the page changes.
      await popup.evaluate(() => chrome.offscreen.closeDocument());
      await setQuestion(pages[0], receipts[0].questions[0].question);
      await pages[0].waitForFunction(() => document.querySelector('.receipt-reader-panel')?.getAttribute('aria-busy') === 'true', { polling: 20 });
      await pages[0].evaluate((receipt) => {
        globalThis.answerHistory = [];
        window.requiredAnswerEvents = window.answerEvents + 1;
        document.querySelector('input[name=captcha]').addEventListener('input', (event) => answerHistory.push(event.target.value));
        document.querySelector('#rcpt_img').src = receipt.imageUrl;
        document.querySelector('#rcpt_info').textContent = receipt.questions[2].question;
      }, receipts[1]);
      await waitAnswer(pages[0], receipts[1].questions[2].answer);
      const history = await pages[0].evaluate(() => answerHistory);
      assert(!history.includes(receipts[0].questions[0].answer), `Stale answer was filled: ${history}`);
      report.checks.push('Image/question changes during inference discard stale answers');

      const excluded = watch(await browser.newPage());
      await excluded.goto(`http://${production ? '127.0.0.1' : '127.0.0.2'}:${port}/en`);
      await setQuestion(excluded, receipts[0].questions[0].question);
      await new Promise((resolve) => setTimeout(resolve, 500));
      assert.equal(await excluded.$('.receipt-reader-panel'), null);
      report.checks.push('Unlisted websites receive no content script');
      await testContentSettings({ browser, popup, page: pages[0], receipt: receipts[0],
        pass: message => { report.checks.push(message); console.log('PASS', message); } });
      assert.deepEqual(errors, []);
      await pages[1].bringToFront();
      await pages[1].setViewport({ width: 1280, height: 800 });
      await pages[1].screenshot({ path: path.join(results, `${reportName}.png`) });
      report.final = await status();
      report.passed = true;
      await fs.writeFile(path.join(results, `${reportName}.json`), JSON.stringify(report, null, 2) + '\n');
      console.log('PASS stale answer protection, site allowlist, no page errors');
      console.log(JSON.stringify(report, null, 2));
    } catch (error) {
      if (browser) {
        for (const [index, page] of (await browser.pages()).entries()) {
          if (!page.url().endsWith('/offscreen.html')) {
            await page.bringToFront();
            await page.screenshot({ path: path.join(results, `failure-${index}.png`), timeout: 5000 }).catch(() => {});
          }
        }
      }
      report.error = String(error.stack || error);
      report.browserErrors = errors;
      await fs.writeFile(path.join(results, `${reportName}.json`), JSON.stringify(report, null, 2) + '\n');
      console.error('Browser errors:', errors);
      throw error;
    } finally {
      await browser?.close();
      await new Promise((resolve) => server.close(resolve));
      await fs.rm(profile, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 });
    }
  }

  async function testLifecycle() {
    const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'receipt-lifecycle-'));
    const extension = path.join(temporary, 'extension');
    const idleMs = 2_000;
    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const checks = [];
    const pass = (message) => { checks.push(message); console.log('PASS', message); };
    let browser;
    async function until(check, label, timeout = 15_000) {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        if (await check()) return;
        await delay(75);
      }
      throw new Error(`Timed out: ${label}`);
    }

    try {
      await fs.cp(path.join(distribution, localRuntimeDir ? 'chrome-local' : 'chrome'), extension, { recursive: true });
      await instrumentHeatmap(extension);
      const buildInfo = JSON.parse(await fs.readFile(path.join(extension, 'build-info.json'), 'utf8'));
      assert.equal(buildInfo.dependencies.javascript.source, runtimeSource);
      assert.equal(buildInfo.runtime_version, selectedRuntimeVersion);
      for (const file of ['volvoxai.lite.min.js', 'volvoxai.lite.wasm']) {
        const runtimeRoot = localRuntimeDir ? path.resolve(localRuntimeDir)
          : path.join(root, 'node_modules/volvoxai-lite', 'dist', runtimeVersion);
        const bytes = await fs.readFile(path.join(extension, 'vendor', file));
        assert.equal(createHash('sha256').update(bytes).digest('hex'), buildInfo.runtime_sha256[file]);
        assert.deepEqual(bytes, await fs.readFile(path.join(runtimeRoot, file)));
      }
      const protocol = await fs.readFile(path.join(extension, 'shared/extension/requests.js'), 'utf8');
      assert(protocol.includes('IDLE_TIMEOUT_MS = 60_000'));
      // Exercise the real lifecycle with a short timeout in an isolated copy only.
      await fs.writeFile(path.join(extension, 'shared/extension/requests.js'), protocol.replace(
        'IDLE_TIMEOUT_MS = 60_000', `IDLE_TIMEOUT_MS = ${idleMs}`));
      const settingsFile = path.join(extension, 'shared/extension/settings-core.js');
      const settingsSource = await fs.readFile(settingsFile, 'utf8');
      assert(settingsSource.includes('idle_timeout_ms: 60_000'));
      await fs.writeFile(settingsFile, settingsSource.replace('idle_timeout_ms: 60_000', `idle_timeout_ms: ${idleMs}`)
        .replace("['idle_timeout_ms', 10_000", "['idle_timeout_ms', 1_000"));
      const reader = await fs.readFile(path.join(extension, 'shared/extension/reader.worker.js'), 'utf8');
      assert(reader.includes('async function read(request) {'));
      await fs.writeFile(path.join(extension, 'shared/extension/reader.worker.js'), reader.replace(
        'async function read(request) {', `async function read(request) {
      if (request.question?.endsWith(' [delayed]')) {
        await new Promise((resolve) => setTimeout(resolve, ${idleMs + 1_000}));
        request = { ...request, question: request.question.slice(0, -10) };
      }`));
      await fs.appendFile(path.join(extension, 'shared/extension/background.js'), `
    globalThis.__testLifecycleState = () => ({ activeRequests, lastCloseSender: globalThis.__lastCloseSender });
    chrome.runtime.onMessage.addListener((message, sender) => {
      if (message.type === 'close-idle-offscreen') globalThis.__lastCloseSender = sender;
    });
    const originalCloseDocument = chrome.offscreen.closeDocument.bind(chrome.offscreen);
    chrome.offscreen.closeDocument = async () => {
      if (globalThis.__testDelayClose) {
        globalThis.__testCloseStarted = true;
        await new Promise((resolve) => setTimeout(resolve, 1_000));
      }
      return originalCloseDocument();
    };
    `);
      await fs.appendFile(path.join(extension, 'shared/extension/offscreen.js'), `
    globalThis.__testIdleState = () => ({ idleDeadline, now: Date.now(), idleGeneration,
      pending: pending.size, worker: Boolean(worker), lastCloseResponse: globalThis.__lastCloseResponse });
    const originalSendMessage = chrome.runtime.sendMessage.bind(chrome.runtime);
    chrome.runtime.sendMessage = (...args) => {
      const result = originalSendMessage(...args);
      if (args[0]?.type === 'close-idle-offscreen') result.then(
        response => { globalThis.__lastCloseResponse = response; },
        error => { globalThis.__lastCloseResponse = String(error); });
      return result;
    };
    `);
      const receipt = (await fixtures())[0];
      const image = receipt.imageUrl;
      browser = await puppeteer.launch({
        executablePath: process.env.RECEIPT_CHROME || await puppeteer.executablePath(),
        headless: true, enableExtensions: true, userDataDir: path.join(temporary, 'profile'),
        protocolTimeout: 30_000,
        args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-extension-debugging'],
      });
      const cdp = await browser.target().createCDPSession();
      const { id } = await cdp.send('Extensions.loadUnpacked', { path: extension });
      const popup = await browser.newPage();
      await popup.goto(`chrome-extension://${id}/shared/extension/popup.html`);
      const status = async () => {
        const response = await popup.evaluate(() => chrome.runtime.sendMessage({
          channel: globalThis.RECEIPT_READER_SITES.channel, target: 'background', type: 'status',
        }));
        assert(response.ok, response.error);
        return response.result;
      };
      const contexts = () => popup.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }));
      assert.equal((await contexts()).length, 0);
      assert.equal((await status()).state, 'idle');
      assert.equal((await contexts()).length, 0);
      assert.equal(await visualization(popup), null);
      pass('Status while idle does not create an offscreen document');

      const page = await browser.newPage();
      const pageErrors = [];
      page.on('pageerror', (error) => pageErrors.push(String(error)));
      await page.setRequestInterception(true);
      page.on('request', (request) => void request.respond(request.isNavigationRequest()
        ? { status: 200, contentType: 'text/html; charset=utf-8', body: `<!doctype html><main id="rcpt_form"><img id="rcpt_img" class="captcha_img" src="${image}"><p id="rcpt_info" class="captcha_message"></p><input name="captcha" id="rcpt_answer_fixture"></main>` }
        : { status: 204, body: '' }));
      await page.goto('https://shopping.naver.com/ns/home');
      const question = (value) => page.$eval('#rcpt_info', (node, text) => { node.textContent = text; }, value);
      const answer = (value) => page.waitForFunction((expected) =>
        document.querySelector('input[name=captcha]')?.value === expected
        && document.querySelector('.receipt-reader-panel')?.getAttribute('aria-busy') === 'false',
      { polling: 50, timeout: 60_000 }, value);

      await question(receipt.questions[0].question + ' [delayed]');
      await until(async () => (await status()).pending > 0, 'delayed request accepted');
      await delay(idleMs + 250);
      assert.equal((await contexts()).length, 1);
      assert((await status()).pending > 0, 'Idle shutdown interrupted an active request');
      await answer(receipt.questions[0].answer);
      const first = await status();
      pass('A request lasting longer than the idle timeout is not interrupted');

      const forged = await popup.evaluate(() => chrome.runtime.sendMessage({
        channel: globalThis.RECEIPT_READER_SITES.channel, target: 'background', type: 'close-idle-offscreen',
        documentToken: 'forged', idleGeneration: 1,
      }));
      assert.equal(forged.ok, false);
      assert.equal((await contexts()).length, 1);
      pass('Only the owning offscreen document may request idle shutdown');

      const { targetInfos } = await cdp.send('Target.getTargets');
      const background = targetInfos.find((target) => target.url === `chrome-extension://${id}/shared/extension/background.js`);
      await cdp.send('Target.closeTarget', { targetId: background.targetId });
      // Repeated popup-style status requests must not renew the read idle deadline.
      await until(async () => {
        await status();
        return (await contexts()).length === 0;
      }, 'offscreen closes despite status polling after a background restart');
      assert(!browser.targets().some((target) => target.url() === `chrome-extension://${id}/shared/extension/reader.worker.js`));
      assert.equal((await status()).state, 'idle');
      assert.equal((await contexts()).length, 0);
      assert.equal(await visualization(popup), null, 'Idle release also drops the retained image and attention map');
      pass('Idle closes the document and worker after service-worker restart; status polling does not keep them alive');

      await question(receipt.questions[1].question);
      await answer(receipt.questions[1].answer);
      const second = await status();
      assert.notEqual(second.workerId, first.workerId);
      assert.equal(second.initializations, 1);
      pass('The next real question creates a new document and initializes the PTQ model once');

      const restartedBackground = browser.targets().find((target) => target.url() === `chrome-extension://${id}/shared/extension/background.js`);
      const restartedCdp = await restartedBackground.createCDPSession();
      await restartedCdp.send('Runtime.evaluate', { expression: 'globalThis.__testDelayClose = true' });
      await until(async () => (await restartedCdp.send('Runtime.evaluate', {
        expression: 'globalThis.__testCloseStarted === true', returnByValue: true,
      })).result.value, 'idle close entered');
      await question(receipt.questions[0].question);
      await answer(receipt.questions[0].answer);
      const third = await status();
      assert.notEqual(third.workerId, second.workerId);
      assert.equal(third.initializations, 1);
      pass('A new question arriving during closeDocument waits and succeeds in the recreated document');
      const poolReceipts = await fixtures();
      const secondPage = await browser.newPage();
      secondPage.on('pageerror', error => pageErrors.push(String(error)));
      await secondPage.setRequestInterception(true);
      secondPage.on('request', request => void request.respond(request.isNavigationRequest()
        ? { status: 200, contentType: 'text/html; charset=utf-8', body: `<!doctype html><main id="rcpt_form"><img id="rcpt_img" class="captcha_img" src="${poolReceipts[1].imageUrl}"><p id="rcpt_info" class="captcha_message"></p><input name="captcha" id="rcpt_answer_fixture"></main>` }
        : { status: 204, body: '' }));
      await secondPage.goto('https://smartstore.naver.com/pool');
      await testPoolSettings({ popup, pages: [page, secondPage], receipts: poolReceipts, status, pass,
        question: (page, value) => page.$eval('#rcpt_info', (node, text) => { node.textContent = text; }, value),
        answer: (page, expected) => page.waitForFunction(value => document.querySelector('input[name=captcha]')?.value === value
          && document.querySelector('.receipt-reader-panel')?.getAttribute('aria-busy') === 'false',
          { polling: 50, timeout: 60_000 }, expected),
      });
      assert.deepEqual(pageErrors, []);
      const report = { project, precision, passed: true, runtime: selectedRuntimeVersion, browser: await browser.version(), idleTimeoutMsInTestCopy: idleMs,
        productionIdleTimeoutMs: 60_000, runtimeSha256: buildInfo.runtime_sha256, dependencies: buildInfo.dependencies, checks };
      await fs.mkdir(results, { recursive: true });
      await fs.writeFile(path.join(results, 'lifecycle.json'), JSON.stringify(report, null, 2) + '\n');
      console.log(JSON.stringify(report, null, 2));
    } catch (error) {
      await fs.mkdir(results, { recursive: true });
      await fs.writeFile(path.join(results, 'lifecycle.json'), JSON.stringify({
        project, precision, passed: false, runtime: selectedRuntimeVersion, checks,
        error: String(error.stack || error),
      }, null, 2) + '\n');
      for (const target of browser?.targets() || []) {
        if (!target.url().startsWith('chrome-extension://')) continue;
        const session = await target.createCDPSession().catch(() => null);
        if (session) console.error(target.url(), JSON.stringify(await session.send('Runtime.evaluate', {
          expression: 'globalThis.__testIdleState?.() || globalThis.__testLifecycleState?.()', returnByValue: true,
        }).catch(String)));
      }
      throw error;
    } finally {
      await browser?.close();
      await fs.rm(temporary, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 });
    }
  }

  await (process.argv.includes('--lifecycle') ? testLifecycle() : testBrowser());
}

const mode = process.argv.includes('--lifecycle') ? 'lifecycle'
  : process.argv.includes('--production') ? 'chrome' : 'chrome-dev';
const matrix = { runtime: selectedRuntimeVersion, mode, startedAt: new Date().toISOString(), cases: [] };
for (const project of requested === 'all' ? ['digit', 'vqa'] : [requested]) {
  const config = JSON.parse(await fs.readFile(path.join(root, project, 'extension.json'), 'utf8'));
  for (const precision of requestedPrecision === 'all' ? ['w8a8']
    : [requestedPrecision === 'default' ? config.default_precision : requestedPrecision]) {
    const started = performance.now();
    console.log(`Testing ${project} ${precision} (${mode})`);
    const reportPath = path.join(resultsRoot, project, precision, `${mode}.json`);
    await fs.mkdir(path.dirname(reportPath), { recursive: true });
    await fs.rm(reportPath, { force: true });
    try {
      await runProject(project, precision);
      matrix.cases.push({ project, precision, passed: true,
        report: `${project}/${precision}/${mode}.json`, elapsedMs: Math.round(performance.now() - started) });
    } catch (error) {
      // Preflight failures also replace previous reports, rather than leaving a stale pass.
      if (!(await fs.stat(reportPath).catch(() => null))) {
        await fs.writeFile(reportPath, JSON.stringify({ project, precision, passed: false,
          runtime: selectedRuntimeVersion, error: String(error.stack || error) }, null, 2) + '\n');
      }
      matrix.cases.push({ project, precision, passed: false,
        report: `${project}/${precision}/${mode}.json`,
        error: String(error.stack || error), elapsedMs: Math.round(performance.now() - started) });
      console.error(`FAIL ${project} ${precision}:`, error);
    }
  }
}
matrix.passed = matrix.cases.every((item) => item.passed);
matrix.finishedAt = new Date().toISOString();
await fs.mkdir(resultsRoot, { recursive: true });
const matrixPath = path.join(resultsRoot, `${mode}-matrix-${requested}-${requestedPrecision}.json`);
await fs.writeFile(matrixPath, JSON.stringify(matrix, null, 2) + '\n');
console.log(`Matrix: ${matrixPath}`);
if (!matrix.passed) process.exitCode = 1;
