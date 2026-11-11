import './site-config.js';
import { IDLE_TIMEOUT_MS } from './requests.js';
import { InferencePool } from './inference-pool.js';
const CHANNEL = globalThis.RECEIPT_READER_SITES.channel;

let worker;
let heartbeat;
let idleTimer;
let idleSince = 0;
let idleDeadline = 0;
let idleGeneration = 0;
let settings = { idle_timeout_ms: IDLE_TIMEOUT_MS };
const documentToken = crypto.randomUUID();
const pool = new InferencePool({ onActivity: startRequest, onIdle: releaseWhenIdle,
  onChange: () => { worker = pool.slots[0]?.worker || null; } });
const pending = pool.pending;

function scheduleIdleClose(delay = Math.max(0, idleDeadline - Date.now())) {
  clearTimeout(idleTimer);
  if (idleDeadline) idleTimer = setTimeout(closeWhenIdle, delay);
}

async function closeWhenIdle() {
  if (!idleDeadline) return;
  if (Date.now() < idleDeadline) { scheduleIdleClose(); return; }
  if (pending.size) { scheduleIdleClose(1_000); return; }
  pool.terminate();
  const generation = idleGeneration;
  try {
    const response = await chrome.runtime.sendMessage({
      channel: CHANNEL, target: 'background', type: 'close-idle-offscreen',
      documentToken, idleGeneration: generation,
    });
    if (response?.ok && response.result.closed) return;
  } catch { /* A restarted background worker can handle the next attempt. */ }
  if (generation === idleGeneration && idleDeadline) scheduleIdleClose(1_000);
}

function releaseWhenIdle() {
  clearInterval(heartbeat);
  heartbeat = null;
  idleSince = Date.now();
  idleDeadline = idleSince + settings.idle_timeout_ms;
  scheduleIdleClose();
}

function startRequest() {
  idleGeneration++;
  idleDeadline = 0;
  clearTimeout(idleTimer);
  // Keep Chrome's MV3 response channel alive only during admitted reads.
  if (!heartbeat) heartbeat = setInterval(() => {
    chrome.runtime.sendMessage({ channel: CHANNEL, target: 'background', type: 'heartbeat' }).catch(() => {});
  }, 20_000);
}

function configure(value) {
  if (!value) return;
  settings = value;
  pool.configure(settings);
  if (idleDeadline) {
    idleDeadline = idleSince + settings.idle_timeout_ms;
    scheduleIdleClose();
  }
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message?.channel !== CHANNEL || message.target !== 'offscreen'
      || sender.id !== chrome.runtime.id || sender.tab) return;
  if (message.type === 'prepare-idle-close') {
    respond({ ok: true, result: { canClose: message.documentToken === documentToken
      && message.idleGeneration === idleGeneration && pending.size === 0
      && idleDeadline > 0 && Date.now() >= idleDeadline } });
    return;
  }
  configure(message.settings);
  if (message.type === 'configure') { respond({ ok: true }); return; }
  if (message.type === 'visualization') { respond({ ok: true, result: pool.visualization() }); return; }
  if (message.type === 'clear-diagnostics') pool.clearDiagnostics();
  if (message.type === 'status' || message.type === 'clear-diagnostics') {
    respond({ ok: true, result: pool.status() });
    return;
  }
  if (message.type !== 'read') return;
  pool.read(message).then(respond);
  return true;
});
