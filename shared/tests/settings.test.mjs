import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { matchesSite as requestMatchesSite } from '../extension/requests.js';

const sites = JSON.parse(readFileSync(new URL('../sites.json', import.meta.url), 'utf8'));
globalThis.RECEIPT_READER_SITES = sites;
const {
  SETTINGS_KEY, defaultSettings, normalizeSettings, readSettings, saveSettings,
  subscribeSettings, matchesSite, additionalMatches, syncContentScripts,
} = await import('../extension/settings.js');

test('defaults disable submit and diagnostics; corrupt stored values cannot enable them or exhaust memory', () => {
  const defaults = defaultSettings();
  assert.equal(defaults.worker_count, 1);
  assert.equal(defaults.max_pending, 16);
  assert.equal(defaults.idle_timeout_ms, 60_000);
  assert.equal(defaults.heatmap_enabled, true);
  assert.equal(normalizeSettings({ heatmap_enabled: false }, { strict: true }).heatmap_enabled, false);
  for (const flag of ['auto_submit', 'profiling']) assert.equal(defaults[flag], false);
  assert(!Object.hasOwn(defaults, 'debug'));
  defaults.matches.push('https://changed.test/*');
  assert.deepEqual(defaultSettings().matches, sites.matches, 'Defaults do not mutate site configuration');
  const normalized = normalizeSettings({ worker_count: 100, max_pending: -1, idle_timeout_ms: 0,
    auto_submit: true, submit_selector: '', profiling: 'true', matches: ['<all_urls>'],
    private_image: 'not a setting' });
  assert.deepEqual(normalized, defaultSettings());
  assert.deepEqual(normalizeSettings(null), defaultSettings());
  assert.deepEqual(normalizeSettings([]), defaultSettings());
});

test('legacy debug settings are discarded without enabling engine profiling', async () => {
  const normalized = normalizeSettings({ debug: true, profiling: false });
  assert.deepEqual(normalized, defaultSettings());
  assert(!Object.hasOwn(normalized, 'debug'));
  assert.equal(normalizeSettings({ debug: true }).profiling, false);
  assert.equal(normalizeSettings({ debug: false, profiling: true }).profiling, true);
  const api = { storage: { local: {
    async get() { return { [SETTINGS_KEY]: { debug: true, profiling: false, worker_count: 2 } }; },
  } } };
  const stored = await readSettings(api);
  assert.equal(stored.worker_count, 2);
  assert.equal(stored.profiling, false);
  assert(!Object.hasOwn(stored, 'debug'));
});

test('saving rejects invalid limits, unsafe URL patterns and automatic submission without a target', () => {
  for (const [key, value] of [
    ['worker_count', 0], ['worker_count', 5], ['worker_count', 1.5], ['max_pending', 65],
    ['idle_timeout_ms', 9_999], ['idle_timeout_ms', 1_800_001], ['profiling', 'true'],
    ['heatmap_enabled', 'false'],
    ['image_selectors', []], ['image_selectors', ['']], ['question_selectors', Array(65).fill('#x')],
  ]) assert.throws(() => normalizeSettings({ [key]: value }, { strict: true }), key);
  for (const pattern of ['<all_urls>', 'https://*/*', 'https://*.example.test/*', 'javascript:alert(1)',
    'https://user@example.test/*', 'https://example.test:8443/*', 'https://example.test/?#fragment',
    'https://example.test?query/*', 'file:///*']) {
    assert.throws(() => normalizeSettings({ matches: [pattern] }, { strict: true }), pattern);
  }
  assert.throws(() => normalizeSettings({ auto_submit: true }, { strict: true }), /제출/);
  assert.equal(normalizeSettings({ auto_submit: true, submit_selector: '#submit' }, { strict: true }).auto_submit, true);
  assert.deepEqual(normalizeSettings({ matches: [] }, { strict: true }).matches, []);
  assert.deepEqual(normalizeSettings({ matches: ['https://EXAMPLE.test/orders/*', 'https://example.test/orders/*'] },
    { strict: true }).matches, ['https://example.test/orders/*']);
  assert.deepEqual(normalizeSettings({ matches: ['file:///*'] },
    { strict: true, sites: { ...sites, matches: ['file:///*'] } }).matches, ['file:///*']);
});

test('CSS validation rejects malformed selectors in pages and normalizes corrupt persisted selectors', () => {
  globalThis.document = { querySelector(selector) {
    if (selector === '[') throw new Error('Invalid selector');
    return null;
  } };
  try {
    assert.throws(() => normalizeSettings({ container_selector: '[' }, { strict: true }), /CSS/);
    assert.throws(() => normalizeSettings({ image_selectors: ['['] }, { strict: true }), /CSS/);
    assert.throws(() => normalizeSettings({ auto_submit: true, submit_selector: '[' }, { strict: true }), /CSS/);
    const recovered = normalizeSettings({ auto_submit: true, submit_selector: '[', image_selectors: ['['] });
    assert.equal(recovered.auto_submit, false);
    assert.deepEqual(recovered.image_selectors, sites.image_selectors);
  } finally { delete globalThis.document; }
});

test('watch URL matching agrees with request admission and does not match lookalike hosts', () => {
  const patterns = ['https://example.test/orders/*', 'http://localhost/*'];
  for (const [url, expected] of [
    ['https://example.test/orders/12?tab=receipt', true], ['https://example.test/other', false],
    ['https://example.test.evil.test/orders/12', false], ['http://example.test/orders/12', false],
    ['http://localhost:8123/receipt', true], ['garbage', false],
  ]) {
    assert.equal(matchesSite(url, patterns), expected, url);
    assert.equal(matchesSite(url, patterns), requestMatchesSite(url, patterns));
  }
});

test('settings persist only normalized fields and subscriptions ignore other storage changes', async () => {
  const stored = {};
  const listeners = new Set();
  const api = { storage: {
    local: { async get() { return stored; }, async set(update) { Object.assign(stored, update); } },
    onChanged: { addListener(fn) { listeners.add(fn); }, removeListener(fn) { listeners.delete(fn); } },
  } };
  assert.deepEqual(await readSettings(api), defaultSettings());
  const saved = await saveSettings({ worker_count: 2, max_pending: 4, private_image: 'discard' }, api);
  assert.equal(saved.worker_count, 2);
  assert.equal(saved.max_pending, 4);
  assert(!Object.hasOwn(stored[SETTINGS_KEY], 'private_image'));
  assert.deepEqual(await readSettings(api), saved);
  const events = [];
  const unsubscribe = subscribeSettings(settings => events.push(settings), api);
  for (const callback of listeners) {
    callback({ unrelated: { newValue: 1 } }, 'local');
    callback({ [SETTINGS_KEY]: { newValue: { worker_count: 3 } } }, 'sync');
    callback({ [SETTINGS_KEY]: { newValue: { worker_count: 3 } } }, 'local');
    callback({ [SETTINGS_KEY]: { newValue: undefined } }, 'local');
  }
  assert.equal(events.length, 2);
  assert.equal(events[0].worker_count, 3);
  assert.deepEqual(events[1], defaultSettings(), 'Deleting settings restores defaults');
  unsubscribe();
  assert.equal(listeners.size, 0);
  await assert.rejects(saveSettings({ auto_submit: true }, api), /제출/);
  assert.deepEqual(await readSettings(api), saved, 'Invalid save leaves previous settings intact');
});

test('custom registration requires granted origins, updates serially, and removes revoked or disabled sites', async () => {
  let registered = [];
  const granted = new Set(['https://first.test/*', 'https://second.test/*']);
  const operations = [];
  const api = {
    permissions: { async contains({ origins }) { return origins.every(origin => granted.has(origin)); } },
    scripting: {
      async getRegisteredContentScripts() { return registered; },
      async registerContentScripts(scripts) { operations.push('register'); registered = scripts; },
      async updateContentScripts(scripts) { operations.push('update'); registered = scripts; },
      async unregisterContentScripts() { operations.push('unregister'); registered = []; },
    },
  };
  const first = normalizeSettings({ matches: [...sites.matches, 'https://first.test/*', 'https://denied.test/*'] });
  assert.deepEqual(additionalMatches(first), ['https://first.test/*', 'https://denied.test/*']);
  await syncContentScripts(first, api);
  assert.deepEqual(registered[0].matches, ['https://first.test/*']);
  assert.deepEqual(registered[0].js, ['shared/extension/site-config.js', 'shared/extension/settings-core.js', 'shared/extension/heatmap-view.js', 'shared/extension/content.js']);
  assert.equal(registered[0].persistAcrossSessions, true);
  await Promise.all([
    syncContentScripts(normalizeSettings({ matches: ['https://first.test/*'] }), api),
    syncContentScripts(normalizeSettings({ matches: ['https://second.test/*'] }), api),
  ]);
  assert.deepEqual(registered[0].matches, ['https://second.test/*']);
  granted.clear();
  await syncContentScripts(normalizeSettings({ matches: ['https://second.test/*'] }), api);
  assert.deepEqual(registered, []);
  assert.deepEqual(operations, ['register', 'update', 'update', 'unregister']);
  await syncContentScripts(defaultSettings(), api);
  assert.equal(operations.length, 4, 'Built-in scripts need no duplicate registration');
});
