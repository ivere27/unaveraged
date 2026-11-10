import './site-config.js';
import { defaultSettings, normalizeSettings, readSettings, saveSettings, additionalMatches } from './settings.js';

const api = globalThis.browser ?? globalThis.chrome;
const sites = globalThis.RECEIPT_READER_SITES;
const form = document.querySelector('#settings-form');
const status = document.querySelector('#save-status');
const listFields = ['matches', 'image_selectors', 'question_selectors', 'answer_selectors'];
const stringFields = ['container_selector', 'submit_selector'];
const numberFields = ['worker_count', 'max_pending', 'idle_timeout_ms'];
const toggleFields = ['heatmap_enabled', 'auto_submit', 'profiling'];
let diagnostics;

document.title = `${sites.ui.title} · 설정`;
document.querySelector('h1').textContent = `${sites.ui.title} 설정`;

function showSettings(settings) {
  for (const key of listFields) form.elements.namedItem(key).value = settings[key].join('\n');
  for (const key of [...stringFields, ...numberFields]) form.elements.namedItem(key).value = settings[key];
  for (const key of toggleFields) form.elements.namedItem(key).checked = settings[key];
  form.elements.namedItem('submit_selector').required = settings.auto_submit;
}

function collectSettings() {
  const values = {};
  for (const key of listFields) values[key] = form.elements.namedItem(key).value.split('\n').map((line) => line.trim()).filter(Boolean);
  for (const key of stringFields) values[key] = form.elements.namedItem(key).value;
  for (const key of numberFields) values[key] = Number(form.elements.namedItem(key).value);
  for (const key of toggleFields) values[key] = form.elements.namedItem(key).checked;
  return normalizeSettings(values, { strict: true });
}

function report(message, error = false) {
  status.textContent = message;
  status.dataset.error = String(error);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const save = document.querySelector('#save-settings');
  try {
    const settings = collectSettings();
    const origins = [...new Set(additionalMatches(settings).map((pattern) => new URL(pattern).origin + '/*'))];
    // Request directly in the save gesture, before any storage or other awaits.
    if (origins.length && !await api.permissions.request({ origins })) {
      report('사이트 접근 권한이 허용되지 않아 설정을 저장하지 않았습니다.', true);
      return;
    }
    save.disabled = true;
    await saveSettings(settings);
    report('설정을 저장했습니다. 새로 추가한 사이트는 페이지를 새로고침하세요.');
  } catch (error) { report(error.message || String(error), true); }
  finally { save.disabled = false; }
});

document.querySelector('#auto_submit').addEventListener('change', (event) => {
  document.querySelector('#submit_selector').required = event.target.checked;
});
document.querySelector('#reset-settings').addEventListener('click', () => {
  showSettings(defaultSettings());
  report('기본값을 불러왔습니다. 적용하려면 설정 저장을 누르세요.');
});

async function request(type) {
  const response = await api.runtime.sendMessage({ channel: sites.channel, target: 'background', type });
  if (!response?.ok) throw new Error(response?.error || '실행 상태를 확인하지 못했습니다.');
  return response.result;
}

async function refreshDiagnostics() {
  try {
    diagnostics = { capturedAt: new Date().toISOString(), extension: sites.ui.title, status: await request('status') };
    document.querySelector('#diagnostics').textContent = JSON.stringify(diagnostics, null, 2);
    document.querySelector('#export-diagnostics').disabled = false;
  } catch (error) { document.querySelector('#diagnostics').textContent = error.message || String(error); }
}
document.querySelector('#refresh-diagnostics').addEventListener('click', refreshDiagnostics);
document.querySelector('#clear-diagnostics').addEventListener('click', async () => {
  try { await request('clear-diagnostics'); await refreshDiagnostics(); }
  catch (error) { document.querySelector('#diagnostics').textContent = error.message || String(error); }
});
document.querySelector('#export-diagnostics').addEventListener('click', () => {
  if (!diagnostics) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(diagnostics, null, 2) + '\n'], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'unaveraged-diagnostics.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
});

try { showSettings(await readSettings()); }
catch (error) { report(error.message || String(error), true); }
