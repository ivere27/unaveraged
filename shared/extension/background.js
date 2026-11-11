import './site-config.js';
import { matchesSite, validateRead } from './requests.js';
import { readSettings, subscribeSettings, initializeContentScripts, syncContentScripts } from './settings.js';
const CHANNEL = globalThis.RECEIPT_READER_SITES.channel;

const offscreenUrl = chrome.runtime.getURL('shared/extension/offscreen.html');
let creating;
let activeRequests = 0;
let lifecycle = Promise.resolve();

function withLifecycle(operation) {
  const result = lifecycle.then(operation);
  lifecycle = result.catch(() => {});
  return result;
}

async function closeIdleOffscreen(message, sender) {
  if (sender.tab || sender.url !== offscreenUrl) {
    throw new Error('허용되지 않은 종료 요청입니다.');
  }
  return withLifecycle(async () => {
    if (activeRequests) return { ok: true, result: { closed: false } };
    if (!await hasOffscreen()) {
      return { ok: true, result: { closed: false } };
    }
    // Offscreen senders may omit documentId. The current document verifies its
    // random token and idle generation, rejecting old documents and late closes.
    const ready = await chrome.runtime.sendMessage({
      channel: CHANNEL, target: 'offscreen', type: 'prepare-idle-close',
      documentToken: message.documentToken, idleGeneration: message.idleGeneration,
    });
    if (activeRequests || !ready?.ok || !ready.result.canClose) {
      return { ok: true, result: { closed: false } };
    }
    await chrome.offscreen.closeDocument();
    return { ok: true, result: { closed: true } };
  });
}

async function hasOffscreen() {
  return (await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'], documentUrls: [offscreenUrl],
  })).length > 0;
}

async function ensureOffscreen() {
  if (!creating) {
    creating = (async () => {
      if (!await hasOffscreen()) {
        await chrome.offscreen.createDocument({
          url: offscreenUrl, reasons: ['WORKERS'],
          justification: 'Share a local receipt inference worker across supported tabs.',
        });
      }
    })().finally(() => { creating = null; });
  }
  await creating;
}

async function handle(message, sender) {
  if (sender.id !== chrome.runtime.id) throw new Error('허용되지 않은 요청입니다.');
  if (message.type === 'heartbeat' && !sender.tab) return { ok: true };
  if (message.type === 'close-idle-offscreen') return closeIdleOffscreen(message, sender);
  const settings = await readSettings();
  if (message.type === 'read') {
    if (!sender.tab || !matchesSite(sender.url, settings.matches)) {
      throw new Error('지원되는 영수증 페이지에서만 읽을 수 있습니다.');
    }
    validateRead(message);
  } else if (['status', 'clear-diagnostics', 'visualization'].includes(message.type) && sender.url?.startsWith(chrome.runtime.getURL(''))) {
    // Status observes the existing document without creating one.
  } else {
    throw new Error('지원되지 않는 요청입니다.');
  }
  activeRequests++;
  try {
    const available = await withLifecycle(async () => {
      if (message.type === 'read') { await ensureOffscreen(); return true; }
      return hasOffscreen();
    });
    if (!available) return { ok: true, result: message.type === 'visualization' ? null : { state: 'idle', pending: 0 } };
    const response = await chrome.runtime.sendMessage({
      channel: CHANNEL, target: 'offscreen', id: crypto.randomUUID(),
      type: message.type, image: message.image, question: message.question, settings,
    });
    if (!response) throw new Error('읽기 작업이 중단되었습니다. 다시 읽어 주세요.');
    return response;
  } finally { activeRequests--; }
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message?.channel !== CHANNEL || message.target !== 'background') return;
  handle(message, sender).then(respond,
    (error) => respond({ ok: false, error: error.message || String(error) }));
  return true;
});

const register = () => initializeContentScripts().catch(() => {});
chrome.runtime.onInstalled.addListener(register);
chrome.runtime.onStartup.addListener(register);
chrome.permissions.onAdded.addListener(register);
chrome.permissions.onRemoved.addListener(register);
subscribeSettings((settings) => {
  void syncContentScripts(settings).catch(() => {});
  void withLifecycle(async () => {
    if (await hasOffscreen()) await chrome.runtime.sendMessage({
      channel: CHANNEL, target: 'offscreen', type: 'configure', settings,
    });
  }).catch(() => {});
});
void register();
