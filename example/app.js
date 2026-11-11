import { questionPresets, expectedAnswer } from './questions.js';

const $ = selector => document.querySelector(selector);
const IDLE_TIMEOUT_MS = 60_000;
const form = $('#demo-form');
const image = $('#receipt-image');
const question = $('#question');
const sampleSelect = $('#sample');
const presetSelect = $('#question-preset');
const run = $('#run');
const releaseButton = $('#release-memory');
const heatmapToggle = $('#show-heatmap');
const overlay = globalThis.RECEIPT_HEATMAP.createOverlay(image, { zIndex: 3 });
let catalog, samples, presets = [], worker, workerModel, pending, idleTimer, requestSequence = 0;
let working = false;

const model = () => form.elements.model.value;
const sample = () => samples?.find(item => item.id === sampleSelect.value);
const milliseconds = value => value >= 1_000 ? `${(value / 1_000).toFixed(2)} s` : `${value.toFixed(1)} ms`;
const sameOriginUrl = value => {
  const url = new URL(value, location.href);
  if (url.origin !== location.origin) throw new Error('모델 파일의 배포 경로를 확인하지 못했습니다.');
  return url.href;
};

function status(text, state = 'idle') {
  $('#status').textContent = text;
  $('#status-dot').dataset.state = state;
}

function setWorking(value) {
  working = value;
  run.disabled = value;
  question.disabled = value || !samples;
  sampleSelect.disabled = value || !samples;
  presetSelect.disabled = value || !samples;
  for (const choice of form.elements.model) choice.disabled = value;
  $('#cancel').hidden = !value;
  form.setAttribute('aria-busy', String(value));
  $('#run-label').textContent = value ? '실행 중' : '실행';
  releaseButton.disabled = value || !worker;
}

function resetResult() {
  overlay.setMap(null);
  $('#answer').textContent = '—';
  $('#answer-check').hidden = true;
  $('#inference-time').textContent = '—';
  $('#load-time').textContent = '—';
  $('#map-description').textContent = '실행 후 표시됩니다.';
  document.body.dataset.result = 'empty';
}

function releaseMemory(text = '메모리 해제됨') {
  clearTimeout(idleTimer);
  worker?.terminate();
  worker = null;
  workerModel = null;
  pending = null;
  setWorking(false);
  status(text);
}

const familyLabels = { phone: '전화번호', address: '주소', store: '가게', item_row: '상품',
  item_math: '상품 계산', item_lookup: '상품 찾기', math: '계산', other: '기타', example: '공개 예제' };

function updatePresets(preserve = false) {
  const previousId = presetSelect.value;
  const previousText = question.value;
  presets = questionPresets(sample(), model());
  const custom = document.createElement('option');
  custom.value = '';
  custom.textContent = '직접 입력';
  presetSelect.replaceChildren(custom);
  const groups = new Map();
  for (const preset of presets) {
    const family = preset.family || 'other';
    if (!groups.has(family)) {
      const group = document.createElement('optgroup');
      group.label = familyLabels[family] || family;
      presetSelect.append(group);
      groups.set(family, group);
    }
    const option = document.createElement('option');
    option.value = preset.id;
    option.textContent = preset.label;
    groups.get(family).append(option);
  }
  if (preserve && previousId === '' && previousText.trim()) {
    presetSelect.value = '';
    question.value = previousText;
  } else {
    const selected = (preserve && presets.find(item => item.id === previousId)) || presets[0];
    presetSelect.value = selected?.id || '';
    question.value = selected?.question || '';
  }
  question.disabled = presetSelect.disabled = false;
}

function selectSample(preserveQuestion = true) {
  resetResult();
  const selected = sample();
  if (!selected) return;
  const imageUrl = sameOriginUrl(selected.image);
  if (image.src !== imageUrl) image.src = imageUrl;
  image.alt = selected.label;
  image.width = selected.width;
  image.height = selected.height;
  updatePresets(preserveQuestion);
  status('준비');
}

function selectModel() {
  releaseMemory('준비');
  if (!samples) return;
  const previous = sampleSelect.value;
  const compatible = samples.filter(item => item.models.includes(model()));
  sampleSelect.replaceChildren(...compatible.map(item => {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = item.label;
    return option;
  }));
  sampleSelect.value = compatible.find(item => item.id === previous)?.id
    || compatible.find(item => item.defaultFor === model())?.id || compatible[0]?.id || '';
  sampleSelect.disabled = false;
  selectSample(false);
}

function showSourceLink(manifest) {
  const link = $('#release-link');
  link.hidden = true;
  const candidates = [
    [manifest.source?.url, typeof manifest.source?.commit === 'string'
      ? `소스 ${manifest.source.commit.slice(0, 7)} ↗` : '소스 ↗'],
    [manifest.release?.url, manifest.release?.tag ? `배포본 ${manifest.release.tag} ↗` : '모델 배포본 ↗'],
  ];
  for (const [value, label] of candidates) {
    if (!value) continue;
    try {
      const url = new URL(value);
      if (url.protocol !== 'https:') continue;
      link.href = url.href;
      link.textContent = label;
      link.hidden = false;
      return;
    } catch {
      // Optional provenance must not prevent loading the local model catalog.
    }
  }
}

async function loadCatalog() {
  run.disabled = true;
  status('준비 중');
  const [releaseResponse, samplesResponse] = await Promise.all([
    fetch(new URL('./release.json', import.meta.url)),
    fetch(new URL('./samples.json', import.meta.url)),
  ]);
  if (!releaseResponse.ok || !samplesResponse.ok) throw new Error('데모를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.');
  const release = await releaseResponse.json();
  const collection = await samplesResponse.json();
  if (release.format !== 'unaveraged-demo/v1' || !release.models?.digit?.adapter || !release.models?.vqa?.adapter
      || collection.format !== 'unaveraged-demo-samples/v1' || !Array.isArray(collection.samples)
      || !collection.samples.some(item => item.models?.includes('digit'))
      || !collection.samples.some(item => item.models?.includes('vqa'))) {
    throw new Error('모델 배포 정보를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.');
  }
  for (const id of ['digit', 'vqa']) {
    sameOriginUrl(release.models[id].adapter);
    const bytes = release.models[id].downloadBytes;
    if (Number.isFinite(bytes) && bytes > 0) $(`#${id}-size`).textContent = `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  }
  catalog = release;
  samples = collection.samples;
  showSourceLink(release);
  selectModel();
  document.body.dataset.ready = 'true';
}

function getWorker() {
  if (worker && workerModel === model()) return worker;
  worker?.terminate();
  const owner = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker = owner;
  workerModel = model();
  owner.onmessage = ({ data }) => {
    if (owner !== worker || !pending || data.id !== pending.id) return;
    if (data.type === 'progress') {
      status(data.stage === 'loading' ? '모델 로드 중' : '추론 중', 'busy');
      return;
    }
    const request = pending;
    pending = null;
    if (!data.ok) {
      releaseMemory(data.error || '읽기에 실패했습니다. 다시 실행해 주세요.');
      status(data.error || '읽기에 실패했습니다. 다시 실행해 주세요.', 'error');
      return;
    }
    const answer = data.result.answer || '';
    $('#answer').textContent = answer || '답변 없음';
    const match = answer === request.expected;
    const knownAnswer = typeof request.expected === 'string';
    $('#answer-check').hidden = !knownAnswer;
    $('#answer-check').dataset.match = String(match);
    $('#answer-check').textContent = knownAnswer ? (match ? '✓' : `예제: ${request.expected}`) : '';
    $('#answer-check').title = knownAnswer ? (match ? '예제 정답과 일치' : '예제 정답과 모델 답변이 다릅니다.') : '';
    $('#answer-check').setAttribute('aria-label', $('#answer-check').title);
    $('#inference-time').textContent = milliseconds(data.inferenceMs);
    $('#load-time').textContent = data.loaded ? milliseconds(data.loadMs) : '재사용';
    const map = data.result.heatmap;
    if (overlay.setMap(map)) {
      overlay.setVisible(heatmapToggle.checked);
      const detail = map.imageMass < .999 ? ` · 이미지 비중 ${(map.imageMass * 100).toFixed(1)}%` : '';
      $('#map-description').textContent = `${map.label || '모델의 실제 어텐션'}${detail}`;
    } else $('#map-description').textContent = '어텐션 없음';
    setWorking(false);
    $('#run-label').textContent = '다시 실행';
    status('완료', 'ready');
    document.body.dataset.result = 'ready';
    // One timeout after a real read; UI changes never retain the model indefinitely.
    idleTimer = setTimeout(() => releaseMemory('유휴 1분 · 메모리 해제됨'), IDLE_TIMEOUT_MS);
  };
  owner.onerror = event => {
    if (owner !== worker) return;
    event.preventDefault();
    releaseMemory();
    status('모델을 실행하지 못했습니다. 페이지를 새로고침한 뒤 다시 시도해 주세요.', 'error');
  };
  return owner;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (working) return;
  try {
    if (!catalog) await loadCatalog();
    const text = question.value.trim();
    if (!sample() || !text || text.length > 512) throw new Error('512자 이하의 질문을 입력해 주세요.');
    clearTimeout(idleTimer);
    overlay.setMap(null);
    setWorking(true);
    status('모델 로드 중', 'busy');
    document.body.dataset.result = 'running';
    const id = ++requestSequence;
    pending = { id, expected: expectedAnswer(sample(), model(), text) };
    getWorker().postMessage({ id, adapterUrl: sameOriginUrl(catalog.models[model()].adapter),
      imageUrl: image.src, question: text, heatmap: true });
  } catch (error) {
    releaseMemory();
    status(error.message || '실행을 시작하지 못했습니다.', 'error');
  }
});
for (const choice of form.elements.model) choice.addEventListener('change', selectModel);
sampleSelect.addEventListener('change', () => selectSample(true));
presetSelect.addEventListener('change', () => {
  const selected = presets.find(item => item.id === presetSelect.value);
  if (selected) question.value = selected.question;
  resetResult();
  status('준비');
});
question.addEventListener('input', () => {
  presetSelect.value = '';
  resetResult();
  status('준비');
});
heatmapToggle.addEventListener('change', () => overlay.setVisible(heatmapToggle.checked));
$('#map-opacity').addEventListener('input', event => { overlay.element.style.opacity = Number(event.target.value) / 100; });
overlay.element.style.opacity = Number($('#map-opacity').value) / 100;
releaseButton.addEventListener('click', () => releaseMemory());
$('#cancel').addEventListener('click', () => releaseMemory('중지 · 메모리 해제됨'));
window.addEventListener('pagehide', () => { releaseMemory(); overlay.dispose(); });
loadCatalog().catch(error => { run.disabled = false; $('#run-label').textContent = '다시 연결'; status(error.message, 'error'); });
