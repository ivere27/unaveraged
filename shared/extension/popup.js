import './site-config.js';
import { readSettings } from './settings.js';

const api = globalThis.browser ?? globalThis.chrome;
const sites = globalThis.RECEIPT_READER_SITES;
const image = document.querySelector('#preview-image');
const toggle = document.querySelector('#show-heatmap');
let overlay;
let hasHeatmap = false;
let settings;

document.querySelector('h1').textContent = sites.ui.title;
document.title = sites.ui.title;
document.querySelector('#open-options').addEventListener('click', (event) => {
  event.preventDefault();
  api.runtime.openOptionsPage();
});

async function request(type) {
  const response = await api.runtime.sendMessage({ channel: sites.channel, target: 'background', type });
  if (!response?.ok) throw new Error(response?.error || '상태를 확인하지 못했습니다.');
  return response.result;
}

function updateToggle() {
  overlay?.setVisible(toggle.checked);
  document.querySelector('#heatmap-legend').hidden = !hasHeatmap || !toggle.checked;
}
toggle.addEventListener('change', updateToggle);

async function showVisualization(data) {
  overlay?.dispose();
  overlay = null;
  hasHeatmap = false;
  document.querySelector('#result-summary').hidden = true;
  document.querySelector('#preview-empty').hidden = true;
  document.querySelector('#preview-frame').hidden = true;
  if (!data?.image || !/^data:image\/(?:png|jpe?g|webp);base64,/i.test(data.image)) {
    image.removeAttribute('src');
    document.querySelector('#preview-empty').hidden = false;
    document.querySelector('#heatmap-note').textContent = '모델이 참고한 영역을 이미지 위에서 확인하세요.';
    updateToggle();
    return;
  }
  image.src = data.image;
  await image.decode();
  document.querySelector('#preview-frame').hidden = false;
  overlay = globalThis.RECEIPT_HEATMAP.createOverlay(image, { visible: toggle.checked });
  hasHeatmap = overlay.setMap(data.heatmap);
  document.querySelector('#heatmap-note').textContent = hasHeatmap
    ? '실제 attention의 상대 강도입니다. 색이 진할수록 더 많이 참고한 영역입니다.'
    : settings?.heatmap_enabled === false ? '설정에서 히트맵 표시를 켜면 다음 추론부터 표시됩니다.'
      : '이 추론에는 표시할 attention이 없습니다.';
  const summary = data.resultSummary;
  if (typeof summary?.answer === 'string' && summary.answer) {
    document.querySelector('#preview-answer').textContent = summary.answer;
    const duration = summary.inferenceMs;
    document.querySelector('#preview-duration').textContent = Number.isFinite(duration)
      ? duration < 1_000 ? `${Math.round(duration)} ms` : `${(duration / 1_000).toFixed(1)} s` : '';
    document.querySelector('#result-summary').hidden = false;
  }
  updateToggle();
}

async function refresh() {
  const button = document.querySelector('#refresh-preview');
  button.disabled = true;
  try {
    const [status, visualization] = await Promise.allSettled([request('status'), request('visualization')]);
    if (status.status === 'fulfilled') {
      const state = status.value;
      const labels = { idle: '영수증 페이지를 기다리고 있습니다.', loading: '영수증을 읽을 준비를 하고 있습니다.',
        ready: '읽을 준비가 되었습니다.', error: '페이지에서 다시 읽기를 눌러 주세요.' };
      document.querySelector('#status').textContent = state.pending
        ? `영수증 ${state.pending}개를 처리 중입니다.` : labels[state.state] || labels.idle;
    } else document.querySelector('#status').textContent = status.reason.message;
    if (visualization.status === 'fulfilled') await showVisualization(visualization.value);
  } catch {
    document.querySelector('#heatmap-note').textContent = '최근 영수증을 표시하지 못했습니다. 새로고침해 주세요.';
  } finally { button.disabled = false; }
}

document.querySelector('#refresh-preview').addEventListener('click', refresh);
window.addEventListener('pagehide', () => overlay?.dispose(), { once: true });
try {
  settings = await readSettings();
  document.querySelector('#submit-hint').textContent = settings.auto_submit
    ? '자동 제출이 켜져 있습니다. 설정에서 변경할 수 있습니다.'
    : '입력된 답을 확인한 뒤 페이지에서 제출하세요.';
  for (const pattern of settings.matches) {
    const item = document.createElement('li');
    item.textContent = pattern;
    document.querySelector('#sites').append(item);
  }
} catch { /* The status and preview remain available if preferences cannot load. */ }
await refresh();
