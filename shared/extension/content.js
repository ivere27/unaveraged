(async () => {
  'use strict';
  const api = globalThis.browser ?? globalThis.chrome;
  const baseSites = globalThis.RECEIPT_READER_SITES;
  const channel = baseSites.channel;
  // Static and optional-host registrations can overlap on a configured page.
  const installed = globalThis.__unaveragedContent ??= new Set();
  if (installed.has(channel)) return;
  installed.add(channel);
  const settingsApi = globalThis.RECEIPT_READER_SETTINGS;
  let settings = await settingsApi.readSettings(api);
  let sites = { ...baseSites, ...settings };
  let settingsVersion = 0;
  let panel, status, resultView, retryButton, fillButton, heatmapToggle, heatmapCaption, overlay;
  const recordViews = new Map();
  let current, attempted, latest, busy = false, timer;
  let editVersion = 0, filling = false, lastFilled = '', lastFilledInput;
  let submitted;

  function first(root, selectors, predicate) {
    for (const selector of selectors) {
      for (const element of root.querySelectorAll(selector)) {
        if (element.getClientRects().length && predicate(element)) return element;
      }
    }
    return null;
  }

  function snapshot() {
    if (!settingsApi.matchesSite(location.href, sites.matches)) return null;
    const root = sites.container_selector ? document.querySelector(sites.container_selector) : document;
    if (!root) return null;
    const image = first(root, sites.image_selectors, (node) => node instanceof HTMLImageElement);
    const questionNode = first(root, sites.question_selectors, () => true);
    const input = first(root, sites.answer_selectors, (node) =>
      (node instanceof HTMLInputElement || node instanceof HTMLTextAreaElement) && !node.disabled && !node.readOnly);
    const question = questionNode?.textContent.replace(/\s+/g, ' ').trim();
    if (!image || !input || !question) return null;
    return { root, image, input, question, source: image.currentSrc || image.src, settingsVersion };
  }

  function sameReceipt(left, right) {
    return Boolean(left && right && left.root === right.root
      && left.image === right.image && left.input === right.input
      && left.source === right.source && left.question === right.question);
  }

  function same(left, right) {
    return sameReceipt(left, right) && left.settingsVersion === right.settingsVersion;
  }

  function setStatus(text, error = false) {
    status.textContent = text;
    status.dataset.error = String(error);
  }

  function resetResult() {
    latest = null;
    overlay?.dispose();
    overlay = null;
    if (heatmapToggle) heatmapToggle.disabled = !settings.heatmap_enabled;
    if (heatmapCaption) heatmapCaption.textContent = settings.heatmap_enabled
      ? '추론 후 실제 어텐션을 표시합니다.' : '설정에서 히트맵을 켤 수 있습니다.';
    resultView.textContent = '—';
    for (const view of recordViews.values()) view.textContent = '—';
    fillButton.disabled = true;
  }

  function setInput(input, value) {
    const prototype = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    filling = true;
    try {
      Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    } finally { filling = false; }
  }

  function submit(result) {
    if (!settings.auto_submit || !settings.submit_selector || !same(result.snapshot, snapshot())) return;
    // Changing a runtime preference must not submit the same receipt again.
    if (submitted?.answer === result.answer && sameReceipt(submitted.snapshot, result.snapshot)) return;
    const input = result.snapshot.input;
    if (input.value !== result.answer) return;
    const button = first(result.snapshot.root, [settings.submit_selector], (node) =>
      (node instanceof HTMLButtonElement || (node instanceof HTMLInputElement
        && ['submit', 'button', 'image'].includes(node.type)))
      && !node.disabled && node.getAttribute('aria-disabled') !== 'true'
      && (!node.form || !input.form || node.form === input.form));
    if (!button) {
      setStatus('답변을 입력했습니다. 설정한 제출 버튼을 찾을 수 없어 직접 제출해야 합니다.');
      return;
    }
    submitted = { answer: result.answer, snapshot: result.snapshot };
    setStatus('답변을 입력하고 설정한 제출 버튼을 눌렀습니다.');
    button.click();
  }

  function fill(result, automatic = false) {
    if (!result?.answer || !same(result.snapshot, snapshot())) {
      setStatus('영수증이나 질문이 바뀌었습니다. 다시 읽어 주세요.');
      return;
    }
    const input = result.snapshot.input;
    setInput(input, result.answer);
    lastFilled = result.answer;
    lastFilledInput = input;
    setStatus('답변을 입력했습니다. 확인 후 제출하세요.');
    // Give synchronous input handlers and their microtasks time to update the page.
    if (automatic) queueMicrotask(() => submit(result));
  }

  function mount() {
    if (panel) return;
    panel = document.createElement('section');
    panel.id = sites.ui.panel_id;
    panel.className = 'receipt-reader-panel';
    panel.setAttribute('aria-label', sites.ui.title);
    panel.innerHTML = `
      <header><strong></strong><span>기기에서 처리</span></header>
      <p class="rdr-status" role="status" aria-live="polite"></p>
      <dl></dl>
      <div class="rdr-answer"><span>답변</span><output data-field="answer">—</output></div>
      <div class="rdr-heatmap"><label><input type="checkbox" data-action="heatmap" checked> 히트맵 보여주기</label>
      <small data-field="heatmap-caption">추론 후 실제 어텐션을 표시합니다.</small></div>
      <footer><button type="button" data-action="read">다시 읽기</button>
      <button type="button" data-action="fill" disabled>답 입력</button></footer>`;
    status = panel.querySelector('.rdr-status');
    resultView = panel.querySelector('[data-field="answer"]');
    panel.querySelector('header strong').textContent = sites.ui.title;
    const fields = panel.querySelector('dl');
    for (const { key, label } of sites.ui.record_fields) {
      const row = document.createElement('div');
      const term = document.createElement('dt');
      const view = document.createElement('dd');
      term.textContent = label;
      view.dataset.field = key;
      view.textContent = '—';
      row.append(term, view);
      fields.append(row);
      recordViews.set(key, view);
    }
    fields.hidden = recordViews.size === 0;
    retryButton = panel.querySelector('[data-action="read"]');
    fillButton = panel.querySelector('[data-action="fill"]');
    heatmapToggle = panel.querySelector('[data-action="heatmap"]');
    heatmapCaption = panel.querySelector('[data-field="heatmap-caption"]');
    heatmapToggle.disabled = !settings.heatmap_enabled;
    heatmapToggle.addEventListener('change', () => overlay?.setVisible(heatmapToggle.checked));
    retryButton.addEventListener('click', () => { attempted = null; refresh(); });
    fillButton.addEventListener('click', () => fill(latest));
    document.body.append(panel);
  }

  function encodeImage(image, source) {
    if (/^data:image\/(?:png|jpeg|jpg|webp);base64,/.test(source)) return source;
    if (image.naturalWidth * image.naturalHeight > 12 * 1024 * 1024) throw new Error('이미지 해상도가 너무 큽니다.');
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.getContext('2d').drawImage(image, 0, 0);
    try { return canvas.toDataURL('image/png'); }
    catch { throw new Error('페이지에서 이 이미지의 픽셀에 접근할 수 없습니다. 이미지의 출처/CORS 설정을 확인해 주세요.'); }
  }

  async function read(value) {
    busy = true;
    attempted = value;
    retryButton.disabled = true;
    panel.setAttribute('aria-busy', 'true');
    setStatus('영수증을 읽고 있습니다…');
    const originalValue = value.input.value;
    const originalEditVersion = editVersion;
    let timeout;
    try {
      const image = encodeImage(value.image, value.source);
      const response = await Promise.race([
        api.runtime.sendMessage({ channel, target: 'background', type: 'read', image, question: value.question }),
        new Promise((_, reject) => { timeout = setTimeout(() => reject(new Error('응답 시간이 초과되었습니다. 다시 읽어 주세요.')), 125_000); }),
      ]);
      if (!same(value, snapshot())) return;
      if (!response?.ok) throw new Error(response?.error || '읽기 작업이 중단되었습니다. 다시 읽어 주세요.');
      latest = { ...response.result, snapshot: value };
      overlay?.dispose();
      overlay = null;
      if (settings.heatmap_enabled && globalThis.RECEIPT_HEATMAP?.validateMap(latest.heatmap)) {
        try {
          overlay = globalThis.RECEIPT_HEATMAP.createOverlay(value.image, { visible: heatmapToggle.checked });
          overlay.setMap(latest.heatmap);
          heatmapCaption.textContent = latest.heatmap.label || '모델의 실제 어텐션 · 이미지 내 상대 강도';
        } catch {
          overlay?.dispose();
          overlay = null;
          heatmapCaption.textContent = '히트맵을 표시하지 못했습니다. 추론 답변은 사용할 수 있습니다.';
        }
      } else {
        heatmapCaption.textContent = settings.heatmap_enabled ? '이 결과에는 표시할 어텐션이 없습니다.' : '설정에서 히트맵을 켤 수 있습니다.';
      }
      for (const [key, view] of recordViews) view.textContent = latest.record?.[key] || '읽지 못함';
      resultView.textContent = latest.answer || '지원하지 않는 질문';
      fillButton.disabled = !latest.answer;
      if (!latest.answer) {
        setStatus('이 질문은 자동으로 답할 수 없습니다. 영수증을 확인해 주세요.');
      } else if (editVersion !== originalEditVersion || value.input.value !== originalValue
          || (originalValue && (originalValue !== lastFilled || value.input !== lastFilledInput))) {
        setStatus('답을 읽었습니다. 작성 중인 답을 바꾸려면 ‘답 입력’을 누르세요.');
      } else {
        fill(latest, true);
      }
    } catch (error) {
      if (same(value, snapshot())) {
        resetResult();
        setStatus(error.message || '읽기에 실패했습니다. 다시 읽어 주세요.', true);
      }
    } finally {
      clearTimeout(timeout);
      busy = false;
      retryButton.disabled = false;
      panel.setAttribute('aria-busy', 'false');
      schedule();
    }
  }

  function refresh() {
    const value = snapshot();
    if (!value) {
      if (panel) { panel.hidden = true; resetResult(); }
      current = null;
      attempted = null;
      return;
    }
    mount();
    if (!panel.isConnected) document.body.append(panel);
    panel.hidden = false;
    overlay?.refresh();
    if (!same(current, value)) {
      if (lastFilled && value.input === lastFilledInput && value.input.value === lastFilled) {
        setInput(value.input, '');
        lastFilled = '';
      }
      current = value;
      resetResult();
    }
    if (!value.image.complete || !value.image.naturalWidth) {
      setStatus('영수증 이미지가 표시되기를 기다리고 있습니다…');
      return;
    }
    if (!busy && !same(attempted, value)) void read(value);
  }

  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(refresh, 100);
  }

  document.addEventListener('input', (event) => {
    if (!filling && event.target === current?.input) { editVersion++; lastFilled = ''; }
  }, true);
  document.addEventListener('load', (event) => {
    if (event.target === current?.image) attempted = null;
    if (event.target instanceof HTMLImageElement) schedule();
  }, true);
  new MutationObserver((changes) => {
    if (changes.some((change) => !panel?.contains(change.target)
        && !change.target.closest?.('[data-receipt-heatmap]'))) schedule();
  }).observe(document.documentElement, {
    childList: true, subtree: true, characterData: true, attributes: true,
    attributeFilter: ['src', 'srcset', 'hidden', 'class', 'style', 'disabled', 'readonly'],
  });
  settingsApi.subscribeSettings((next) => {
    settings = next;
    sites = { ...baseSites, ...settings };
    settingsVersion++;
    attempted = null;
    if (panel) resetResult();
    schedule();
  }, api);
  window.addEventListener('popstate', schedule);
  window.addEventListener('hashchange', schedule);
  refresh();
})();
