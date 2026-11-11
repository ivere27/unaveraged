export const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
export const MAX_PIXELS = 12 * 1024 * 1024;
export const MAX_PENDING = 16;
export const REQUEST_TIMEOUT_MS = 120_000;
export const IDLE_TIMEOUT_MS = 60_000;

export function validateRead(request) {
  if (typeof request?.image !== 'string'
      || request.image.length > Math.ceil(MAX_IMAGE_BYTES / 3) * 4 + 64
      || !/^data:image\/(?:png|jpeg|jpg|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(request.image)) {
    throw new Error('4 MB 이하의 PNG, JPEG, WebP 영수증 이미지가 필요합니다.');
  }
  if (typeof request.question !== 'string' || request.question.length > 2000) {
    throw new Error('질문은 2,000자 이하의 문자열이어야 합니다.');
  }
}

/** The allowlist uses Chrome match patterns with explicit schemes/hosts. */
export function matchesSite(url, patterns) {
  try {
    const parsed = new URL(url);
    return patterns.some((pattern) => {
      const match = pattern.match(/^(https?|file):\/\/([^/]*)(\/.*)$/);
      if (!match || parsed.protocol !== `${match[1]}:`) return false;
      if (parsed.hostname !== match[2]) return false;
      const path = match[3].split('*').map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*');
      return new RegExp(`^${path}$`).test(parsed.pathname + parsed.search);
    });
  } catch {
    return false;
  }
}

/** One WASM owner, bounded FIFO admission, and recovery after failed requests. */
export class SerialQueue {
  #tail = Promise.resolve();
  #pending = 0;
  constructor(limit) { this.limit = limit; }
  get pending() { return this.#pending; }
  run(operation) {
    if (this.#pending >= this.limit) {
      return Promise.reject(new Error('처리 대기 중인 영수증이 많습니다. 잠시 후 다시 읽어 주세요.'));
    }
    this.#pending++;
    const result = this.#tail.then(operation).finally(() => { this.#pending--; });
    this.#tail = result.catch(() => {});
    return result;
  }
}
