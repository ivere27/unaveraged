import './site-config.js';
import { matchesSite, validateRead, IDLE_TIMEOUT_MS } from './requests.js';
import { InferencePool } from './inference-pool.js';
import { readSettings, subscribeSettings, initializeContentScripts, syncContentScripts } from './settings.js';

// Firefox event pages host the same pool directly; the returned Promise keeps
// each message response alive until its read completes.
const api = globalThis.browser ?? globalThis.chrome;
const sites = globalThis.RECEIPT_READER_SITES;
let settings = { idle_timeout_ms: IDLE_TIMEOUT_MS };
let idleTimer;
let idleSince = 0;
let idleDeadline = 0;
const pool = new InferencePool({ onActivity() {
  clearTimeout(idleTimer);
  idleDeadline = 0;
}, onIdle() {
  idleSince = Date.now();
  idleDeadline = idleSince + settings.idle_timeout_ms;
  scheduleIdleClose();
} });

function scheduleIdleClose() {
  clearTimeout(idleTimer);
  if (idleDeadline) idleTimer = setTimeout(() => {
    if (pool.pending.size) return;
    if (Date.now() < idleDeadline) { scheduleIdleClose(); return; }
    pool.terminate();
    idleDeadline = 0;
  }, Math.max(0, idleDeadline - Date.now()));
}

function configure(value) {
  settings = value;
  pool.configure(value);
  if (idleDeadline) {
    idleDeadline = idleSince + settings.idle_timeout_ms;
    scheduleIdleClose();
  }
}

async function handle(message, sender) {
  if (sender.id !== api.runtime.id) throw new Error('허용되지 않은 요청입니다.');
  const current = await readSettings();
  configure(current);
  if (message.type === 'read') {
    if (!sender.tab || !matchesSite(sender.url, current.matches)) {
      throw new Error('설정한 영수증 페이지에서만 읽을 수 있습니다.');
    }
    validateRead(message);
    return pool.read(message);
  }
  if (['status', 'clear-diagnostics', 'visualization'].includes(message.type) && sender.url?.startsWith(api.runtime.getURL(''))) {
    if (message.type === 'visualization') return { ok: true, result: pool.visualization() };
    if (message.type === 'clear-diagnostics') pool.clearDiagnostics();
    return { ok: true, result: pool.status() };
  }
  throw new Error('지원되지 않는 요청입니다.');
}

api.runtime.onMessage.addListener((message, sender) => {
  if (message?.channel !== sites.channel || message.target !== 'background') return;
  return handle(message, sender).catch((error) => ({ ok: false, error: error.message || String(error) }));
});

const register = () => initializeContentScripts().catch(() => {});
api.runtime.onInstalled.addListener(register);
api.runtime.onStartup.addListener(register);
api.permissions.onAdded.addListener(register);
api.permissions.onRemoved.addListener(register);
subscribeSettings((value) => {
  configure(value);
  void syncContentScripts(value).catch(() => {});
});
void register();
