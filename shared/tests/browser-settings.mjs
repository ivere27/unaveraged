import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, label, timeout = 15_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await check()) return;
    await delay(75);
  }
  throw new Error(`Timed out: ${label}`);
}
const read = popup => popup.evaluate(async () => (await import('./settings.js')).readSettings());
const save = (popup, settings) => popup.evaluate(async value => (await import('./settings.js')).saveSettings(value), settings);

export const visualization = popup => popup.evaluate(async () => {
  const api = globalThis.browser ?? globalThis.chrome;
  const response = await api.runtime.sendMessage({ channel: globalThis.RECEIPT_READER_SITES.channel,
    target: 'background', type: 'visualization' });
  if (!response?.ok) throw new Error(response?.error || 'Visualization request failed');
  return response.result;
});

// Instrument only disposable browser-test copies. Counters observe real canvas
// writes and response presence without substituting attention or inference.
export async function instrumentHeatmap(extension) {
  const view = path.join(extension, 'shared/extension/heatmap-view.js');
  const renderer = await fs.readFile(view, 'utf8');
  const rasterCall = 'context.putImageData(pixels, 0, 0);';
  assert(renderer.includes(rasterCall));
  await fs.writeFile(view, renderer.replace(rasterCall, rasterCall
    + '\n          canvas.dataset.testRasterWrites = String(Number(canvas.dataset.testRasterWrites || 0) + 1);'));
  const content = path.join(extension, 'shared/extension/content.js');
  await fs.writeFile(content, `(() => {
    if (globalThis.__testHeatmapResponseAudit) return;
    globalThis.__testHeatmapResponseAudit = true;
    const api = globalThis.browser ?? globalThis.chrome;
    const send = api.runtime.sendMessage.bind(api.runtime);
    api.runtime.sendMessage = (...args) => {
      const result = send(...args);
      if (args[0]?.target === 'background' && args[0]?.type === 'read') result.then(response => {
        if (response?.ok) document.documentElement.dataset.testResultHasHeatmap = String(Boolean(response.result?.heatmap));
      }, () => {});
      return result;
    };
  })();\n` + (await fs.readFile(content, 'utf8')).replace(
    '} catch {\n          overlay?.dispose();',
    '} catch (error) {\n          document.documentElement.dataset.testHeatmapError = error.stack || error.message;\n          overlay?.dispose();'));
}

export async function testHeatmap({ popup, pages, receipts, project, status, pass }) {
  const original = await read(popup);
  assert.equal(original.heatmap_enabled, true, 'Attention capture defaults on');
  const data = await visualization(popup);
  assert(receipts.some(receipt => receipt.imageUrl === data?.image));
  const map = data.heatmap;
  assert.equal(map.kind, 'attention');
  assert.deepEqual([map.width, map.height], project === 'digit' ? [42, 10] : [21, 10]);
  assert.equal(map.values.length, map.width * map.height);
  assert(map.values.every(value => Number.isFinite(value) && value >= 0));
  assert(Math.abs(map.values.reduce((sum, value) => sum + value, 0) - 1) < 1e-5);
  assert(map.imageMass > 0 && map.imageMass <= 1.001);
  assert.equal(map.tensor, project === 'digit' ? 'v117' : 'v172');
  if (project === 'digit') assert.equal(map.slots.length, 1);
  else assert(map.answerTokens > 0 && map.heads === 8);
  pass('Real model attention is finite, normalized and tied to selected digit slots or VQA answer tokens');

  const baseline = await status();
  for (const page of pages) {
    await page.bringToFront();
    try { await page.waitForSelector('[data-receipt-heatmap] canvas', { visible: true }); }
    catch (error) {
      console.error('Heatmap overlay state', await page.evaluate(() => ({
        caption: document.querySelector('[data-field="heatmap-caption"]')?.textContent,
        layer: document.querySelector('[data-receipt-heatmap]')?.outerHTML,
        resultHasMap: document.documentElement.dataset.testResultHasHeatmap,
        renderError: document.documentElement.dataset.testHeatmapError,
      })));
      throw error;
    }
    const initial = await page.evaluate(() => {
      const image = document.querySelector('#rcpt_img');
      const layer = document.querySelector('[data-receipt-heatmap]');
      const canvas = layer.querySelector('canvas');
      const a = image.getBoundingClientRect(), b = layer.getBoundingClientRect();
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      return { width: canvas.width, height: canvas.height, writes: Number(canvas.dataset.testRasterWrites),
        error: Math.max(Math.abs(a.left - b.left), Math.abs(a.top - b.top), Math.abs(a.width - b.width), Math.abs(a.height - b.height)),
        painted: Array.from(pixels).some((value, index) => index % 4 === 3 && value > 0),
        resultHasMap: document.documentElement.dataset.testResultHasHeatmap };
    });
    assert.deepEqual([initial.width, initial.height], [map.width, map.height]);
    assert(initial.error < 1, 'The overlay covers the actual receipt image');
    assert(initial.painted);
    assert.equal(initial.resultHasMap, 'true');
    assert.equal(initial.writes, 1);
    await page.$eval('[data-action="heatmap"]', input => { input.checked = false; input.dispatchEvent(new Event('change')); });
    assert.equal(await page.$eval('[data-receipt-heatmap]', layer => getComputedStyle(layer).display), 'none');
    await page.$eval('[data-action="heatmap"]', input => { input.checked = true; input.dispatchEvent(new Event('change')); });
    await page.evaluate(() => { window.dispatchEvent(new Event('resize')); window.dispatchEvent(new Event('scroll')); });
    await delay(250);
    assert.equal(await page.$eval('[data-receipt-heatmap] canvas', canvas => Number(canvas.dataset.testRasterWrites)), 1,
      'Visibility, geometry updates, and idle time reuse the original raster');
  }
  await popup.bringToFront();
  await popup.$eval('#refresh-preview', button => button.click());
  await popup.waitForFunction(() => !document.querySelector('#refresh-preview').disabled
    && !document.querySelector('#preview-frame').hidden && document.querySelector('[data-receipt-heatmap] canvas'));
  assert.equal(await popup.$eval('#preview-image', image => image.src), data.image);
  assert.equal(await popup.$eval('#preview-answer', node => node.textContent), data.resultSummary.answer);
  const popupWrites = await popup.$eval('[data-receipt-heatmap] canvas', canvas => Number(canvas.dataset.testRasterWrites));
  await popup.$eval('#show-heatmap', input => { input.checked = false; input.dispatchEvent(new Event('change')); });
  assert.equal(await popup.$eval('[data-receipt-heatmap]', layer => getComputedStyle(layer).display), 'none');
  await popup.$eval('#show-heatmap', input => { input.checked = true; input.dispatchEvent(new Event('change')); });
  await delay(250);
  assert.equal(await popup.$eval('[data-receipt-heatmap] canvas', canvas => Number(canvas.dataset.testRasterWrites)), popupWrites);
  const after = await status();
  assert.equal(after.forwardPasses, baseline.forwardPasses);
  assert.equal(after.cacheHits, baseline.cacheHits, 'Display-only toggles do not issue read requests');
  assert(!JSON.stringify(after).includes('data:image/'), 'Status/diagnostics never export receipt images');
  pass('Content and popup overlays reuse one raster; display toggles and idle time do not run inference or repaint');

  try {
    const before = await Promise.all(pages.map(page => page.evaluate(() => window.answerEvents)));
    await save(popup, { ...original, heatmap_enabled: false });
    await Promise.all(pages.map((page, index) => page.waitForFunction(count => window.answerEvents > count
      && document.querySelector('.receipt-reader-panel').getAttribute('aria-busy') === 'false'
      && document.documentElement.dataset.testResultHasHeatmap === 'false', { polling: 50, timeout: 120_000 }, before[index])));
    for (const page of pages) assert.equal(await page.$('[data-receipt-heatmap]'), null);
    assert.equal(await visualization(popup), null);
    await popup.bringToFront();
    await popup.$eval('#refresh-preview', button => button.click());
    await popup.waitForFunction(() => !document.querySelector('#refresh-preview').disabled && document.querySelector('#preview-frame').hidden);
    pass('Disabling heatmap capture returns no map and clears content overlays and retained popup preview');
  } finally {
    const before = await Promise.all(pages.map(page => page.evaluate(() => window.answerEvents)));
    await save(popup, original);
    await Promise.all(pages.map((page, index) => page.waitForFunction(count => window.answerEvents > count
      && document.querySelector('.receipt-reader-panel').getAttribute('aria-busy') === 'false'
      && document.documentElement.dataset.testResultHasHeatmap === 'true', { polling: 50, timeout: 120_000 }, before[index])));
  }
}

/** Exercise settings through the extension options form and the actual content script. */
export async function testContentSettings({ browser, popup, page, receipt, pass }) {
  const original = await read(popup);
  await popup.evaluate(async (settings) => {
    const api = globalThis.browser ?? globalThis.chrome;
    const { syncContentScripts } = await import('./settings.js');
    const sites = globalThis.RECEIPT_READER_SITES;
    const builtInMatches = sites.matches;
    const pattern = settings.matches.find(value => value.startsWith('https://') || value.startsWith('http://'));
    const narrowed = pattern.replace(/\/\*$/, '/dynamic-test/*');
    if (!await api.permissions.contains({ origins: [pattern] })) throw new Error('Test fixture host permission was not granted');
    // Use an already granted host to exercise the actual registration API without a permission dialog.
    sites.matches = [];
    try {
      await syncContentScripts({ ...settings, matches: [pattern] });
      const [created] = await api.scripting.getRegisteredContentScripts({ ids: ['reader-custom-sites'] });
      if (created?.matches[0] !== pattern) throw new Error('Custom site registration failed');
      await syncContentScripts({ ...settings, matches: [narrowed] });
      const [updated] = await api.scripting.getRegisteredContentScripts({ ids: ['reader-custom-sites'] });
      if (updated?.matches[0] !== narrowed) throw new Error('Custom site registration update failed');
      await syncContentScripts({ ...settings, matches: [] });
      if ((await api.scripting.getRegisteredContentScripts({ ids: ['reader-custom-sites'] })).length) {
        throw new Error('Custom site registration was not removed');
      }
    } finally { sites.matches = builtInMatches; await syncContentScripts(settings); }
  }, original);
  pass('Custom site content-script registrations create, update and unregister through the browser API');
  assert.equal(original.auto_submit, false, 'Automatic submit is opt-in');
  const settings = { ...original, container_selector: '#watched-receipt',
    image_selectors: ['img[data-receipt]'], question_selectors: ['[data-question]'],
    answer_selectors: ['input[data-answer]'], auto_submit: false, submit_selector: '[data-submit]' };
  const options = await browser.newPage();
  const optionsUrl = new URL('options.html', await popup.evaluate(() => location.href)).href;
  if (browser.protocol === 'webDriverBiDi') {
    await options.mainFrame().browsingContext.navigate(optionsUrl, 'none');
    await options.waitForFunction(() => document.readyState === 'complete');
  } else await options.goto(optionsUrl);
  try {
    await options.waitForSelector('#save-settings');
    assert.equal(await options.$('#debug'), null, 'There is no separate debug toggle');
    assert.equal(await options.$eval('#profiling', field => field.type), 'checkbox');
    pass('Options expose one profiling toggle for trace summaries and execution history');
    await page.evaluate(receipt => {
      document.body.innerHTML = `<section id="outside"><img data-receipt><p data-question>Outside watch scope</p><input data-answer></section>
        <form id="watched-receipt"><img data-receipt><p data-question></p><input name="captcha" data-answer>
        <button type="submit" data-submit>Confirm</button></form>`;
      document.querySelector('#watched-receipt img').src = receipt.imageUrl;
      document.querySelector('#watched-receipt [data-question]').textContent = receipt.questions[0].question;
      window.submitClicks = 0;
      document.querySelector('#watched-receipt').addEventListener('submit', event => event.preventDefault());
      document.querySelector('[data-submit]').addEventListener('click', () => { window.submitClicks++; });
    }, receipt);
    // Saving the visible form also exercises parsing/validation and storage persistence.
    await options.evaluate(value => {
      for (const [name, item] of Object.entries(value)) {
        const field = document.getElementById(name);
        if (!field) continue;
        if (field.type === 'checkbox') field.checked = item;
        else field.value = Array.isArray(item) ? item.join('\n') : String(item);
      }
    }, settings);
    await options.click('#save-settings');
    await page.bringToFront();
    await until(async () => (await read(popup)).container_selector === '#watched-receipt', 'options form saved');
    await page.waitForFunction(expected => document.querySelector('#watched-receipt input').value === expected,
      { timeout: 120_000, polling: 50 }, receipt.questions[0].answer);
    assert.equal(await page.$eval('#outside input', node => node.value), '');
    assert.equal(await page.evaluate(() => window.submitClicks), 0);
    pass('Options form persists custom selectors; only the watched container fills and default auto-submit stays off');

    await page.$eval('#watched-receipt [data-question]', node => { node.textContent = ''; });
    await delay(350);
    await save(popup, { ...settings, auto_submit: true });
    await delay(350);
    await page.$eval('#watched-receipt [data-question]', (node, text) => { node.textContent = text; }, receipt.questions[1].question);
    try {
      await page.waitForFunction(expected => document.querySelector('#watched-receipt input').value === expected
        && window.submitClicks === 1, { timeout: 30_000, polling: 50 }, receipt.questions[1].answer);
    } catch (error) {
      console.error('Auto-submit state', await page.evaluate(() => ({
        question: document.querySelector('#watched-receipt [data-question]').textContent,
        value: document.querySelector('#watched-receipt input').value, clicks: window.submitClicks,
        panel: document.querySelector('.receipt-reader-panel')?.textContent,
      })), await read(popup));
      throw error;
    }
    await page.evaluate(() => document.querySelector('#watched-receipt').append(document.createElement('span')));
    await delay(350);
    assert.equal(await page.evaluate(() => window.submitClicks), 1, 'Unrelated mutations do not resubmit');
    const answerEvents = await page.evaluate(() => window.answerEvents);
    await save(popup, { ...settings, auto_submit: true, max_pending: settings.max_pending === 64 ? 63 : settings.max_pending + 1 });
    await page.waitForFunction(count => window.answerEvents > count
      && document.querySelector('.receipt-reader-panel').getAttribute('aria-busy') === 'false',
    { polling: 50 }, answerEvents);
    assert.equal(await page.evaluate(() => window.submitClicks), 1, 'Runtime settings changes do not resubmit');
    await page.bringToFront();
    await page.click('[data-action=fill]');
    assert.equal(await page.evaluate(() => window.submitClicks), 1, 'Manual fill does not automatically submit');
    pass('Explicit auto-submit clicks once; unrelated mutations, runtime setting changes and manual fill do not resubmit');
    await save(popup, { ...settings, container_selector: '#missing-container', auto_submit: false });
    await page.waitForFunction(() => !document.querySelector('.receipt-reader-panel')
      || document.querySelector('.receipt-reader-panel').hidden, { polling: 50 });
    pass('Changing the watch container immediately stops watching the previous form');
    await save(popup, settings);
    await page.waitForSelector('.receipt-reader-panel', { visible: true });
    const url = new URL(page.url());
    await save(popup, { ...settings, matches: [`${url.protocol}//${url.hostname}/unmatched-test-path/*`] });
    await page.waitForFunction(() => !document.querySelector('.receipt-reader-panel')
      || document.querySelector('.receipt-reader-panel').hidden, { polling: 50 });
    pass('Changing URL filters immediately stops watching an excluded page');
  } finally {
    await save(popup, original);
    await options.close();
  }
}

/** Lifecycle test copies delay tagged reads, so real overlapping inference is deterministic. */
export async function testPoolSettings({ popup, pages, receipts, status, question, answer, pass }) {
  const original = await read(popup);
  try {
    await Promise.all(pages.map(page => question(page, '')));
    await until(async () => (await status()).pending === 0, 'previous reads drained');
    await save(popup, { ...original, worker_count: 2, max_pending: 4, profiling: true });
    await delay(350);
    await Promise.all(pages.map((page, i) => question(page, receipts[i].questions[0].question + ' [delayed]')));
    await until(async () => {
      const state = await status();
      return state.workerCount === 2 && state.active === 2;
    }, 'two independently active inference workers');
    await Promise.all(pages.map((page, i) => answer(page, receipts[i].questions[0].answer)));
    const parallel = await status();
    assert.equal(parallel.configuredWorkers, 2);
    assert.equal(parallel.workerCount, 2);
    assert.equal(new Set(parallel.workers.map(worker => worker.workerId)).size, 2);
    pass('Changing worker count to two runs simultaneous receipts in two actual workers');
    assert(parallel.diagnostics.some(entry => entry.event === 'worker-created'));
    const profiles = parallel.diagnostics.filter(entry => entry.event === 'read' && !entry.cached).map(entry => entry.profile);
    assert(profiles.length >= 2);
    assert(profiles.every(profile => profile && !profile.error), 'VolvoxAI operator profiling must produce trace summaries');
    assert(parallel.diagnostics.length <= 32, 'Profiling history stays bounded');
    assert(!JSON.stringify(parallel.diagnostics).includes('data:image/'), 'Diagnostics exclude image input');
    pass('Profiling produces real VolvoxAI trace summaries and execution history is bounded');

    await Promise.all(pages.map(page => question(page, '')));
    await until(async () => (await status()).pending === 0, 'parallel reads drained');
    await save(popup, { ...original, worker_count: 1, max_pending: 1, idle_timeout_ms: 10_000,
      profiling: true });
    await delay(350);
    await question(pages[0], receipts[0].questions[1].question + ' [delayed]');
    await until(async () => (await status()).pending === 1, 'bounded request in flight');
    await question(pages[1], receipts[1].questions[1].question + ' [delayed]');
    await pages[1].waitForFunction(() => document.querySelector('.rdr-status')?.dataset.error === 'true', { polling: 50 });
    assert.match(await pages[1].$eval('.rdr-status', node => node.textContent), /대기|많|queue/i);
    const bounded = await status();
    assert.equal(bounded.pending, 1);
    assert.equal(bounded.workerCount, 1);
    await answer(pages[0], receipts[0].questions[1].answer);
    pass('A queue limit of one rejects excess requests without interrupting admitted inference');
    await until(async () => (await status()).state === 'idle', 'configured 10-second idle release', 20_000);
    assert.equal(await visualization(popup), null, 'Idle pool release clears the retained visualization');
    pass('The configured idle timeout releases the worker despite status polling');
  } finally {
    await save(popup, original);
  }
}
