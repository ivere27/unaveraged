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
  const submitInput = document.querySelector('#submit_selector');
  submitInput.required = event.target.checked;
  if (event.target.checked && !submitInput.value.trim()) {
    submitInput.value = '#cpt_confirm';
  }
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

function formatTime(isoString) {
  try {
    const d = new Date(isoString);
    return d.toTimeString().split(' ')[0] + '.' + String(d.getMilliseconds()).padStart(3, '0');
  } catch { return isoString; }
}

function renderDiagnosticsView(data) {
  const st = data?.status || {};
  const summaryBox = document.querySelector('#diagnostics-summary');
  const eventsBox = document.querySelector('#diagnostics-events');
  if (!summaryBox || !eventsBox) return;

  const stateLabels = {
    ready: '정상 가동 준비됨',
    idle: '유휴 상태 (대기 중)',
    loading: '모델 준비 중…',
    error: '오류 발생',
  };
  const stateDesc = {
    ready: '추론 요청 즉시 처리 가능',
    idle: '새 요청 시 모델 자동 로드',
    loading: 'WASM 런타임 및 가중치 로드 중',
    error: '페이지에서 다시 시도 필요',
  };

  document.querySelector('#diag-state').textContent = stateLabels[st.state] || st.state || '준비됨';
  document.querySelector('#diag-state-desc').textContent = stateDesc[st.state] || '';
  document.querySelector('#diag-inferences').textContent = Number(st.forwardPasses || 0).toLocaleString();
  document.querySelector('#diag-cache').textContent = Number(st.cacheHits || 0).toLocaleString();
  document.querySelector('#diag-workers').textContent = `${st.active || 0} / ${st.configuredWorkers || 1}`;
  document.querySelector('#diag-workers-desc').textContent = `활성 ${st.active || 0}개 · 대기열 ${st.queued || 0}개`;
  summaryBox.hidden = false;

  const events = Array.isArray(st.diagnostics) ? st.diagnostics : [];
  const tbody = document.querySelector('#events-body');
  document.querySelector('#events-count').textContent = `${events.length}건`;
  if (tbody) {
    tbody.replaceChildren();
    if (!events.length) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td colspan="5" style="text-align:center;color:#6b8277;padding:16px;">기록된 이벤트가 없습니다. (설정에서 프로파일링을 켜면 최근 추론이 기록됩니다)</td>';
      tbody.appendChild(tr);
    } else {
      for (const ev of [...events].reverse()) {
        const tr = document.createElement('tr');
        const eventLabel = ev.event === 'read' ? '추론 완료'
          : ev.event === 'read-error' ? '추론 실패'
          : ev.event === 'worker-created' ? '워커 생성'
          : ev.event;
        const inferenceTime = Number.isFinite(ev.inferenceMs) ? `${Math.round(ev.inferenceMs)} ms` : '—';
        const queueTime = Number.isFinite(ev.queueMs) ? `${Math.round(ev.queueMs)} ms` : '—';
        const cached = ev.cached ? '적중 (Hit)' : (ev.event === 'read' ? '신규 (Miss)' : '—');
        tr.innerHTML = `<td><code>${formatTime(ev.at)}</code></td><td><strong>${eventLabel}</strong></td><td>${inferenceTime}</td><td>${queueTime}</td><td>${cached}</td>`;
        tbody.appendChild(tr);
      }
    }
    eventsBox.hidden = false;
  }
}

async function refreshDiagnostics() {
  try {
    diagnostics = { capturedAt: new Date().toISOString(), extension: sites.ui.title, status: await request('status') };
    document.querySelector('#diagnostics').textContent = JSON.stringify(diagnostics, null, 2);
    renderDiagnosticsView(diagnostics);
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
