// Shared by classic content scripts and extension-page ES modules.
(() => {
  const SETTINGS_KEY = 'readerSettings';
  const api = () => globalThis.browser ?? globalThis.chrome;
  const siteConfig = () => globalThis.RECEIPT_READER_SITES ?? {};

  function defaultSettings(sites = siteConfig()) {
    return {
      matches: [...(sites.matches ?? [])],
      container_selector: '',
      image_selectors: [...(sites.image_selectors ?? ['#rcpt_img'])],
      question_selectors: [...(sites.question_selectors ?? ['#rcpt_info'])],
      answer_selectors: [...(sites.answer_selectors ?? ["input[name='captcha']"])],
      worker_count: 1,
      max_pending: 16,
      idle_timeout_ms: 60_000,
      heatmap_enabled: true,
      auto_submit: false,
      submit_selector: '',
      profiling: false,
    };
  }

  function matchesSite(url, patterns) {
    try {
      const parsed = new URL(url);
      return patterns.some((pattern) => {
        const match = pattern.match(/^(https?|file):\/\/([^/]*)(\/.*)$/);
        if (!match || parsed.protocol !== `${match[1]}:` || parsed.hostname !== match[2]) return false;
        const path = match[3].split('*').map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*');
        return new RegExp(`^${path}$`).test(parsed.pathname + parsed.search);
      });
    } catch { return false; }
  }

  function normalizeSettings(value = {}, { strict = false, sites = siteConfig() } = {}) {
    const defaults = defaultSettings(sites);
    const source = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    if (strict && source !== value) throw new Error('설정은 객체여야 합니다.');
    const settings = { ...defaults };
    const normalize = (key, validate) => {
      if (!Object.hasOwn(source, key)) return;
      try { settings[key] = validate(source[key]); }
      catch (error) { if (strict) throw error; }
    };
    const string = (value, label) => {
      if (typeof value !== 'string' || value.length > 1_000) throw new Error(`${label}: 1,000자 이하의 문자열이 필요합니다.`);
      return value.trim();
    };
    const selector = (value, label) => {
      const result = string(value, label);
      if (result && globalThis.document?.querySelector) {
        try { document.querySelector(result); }
        catch { throw new Error(`${label}: 올바른 CSS 선택자를 입력하세요.`); }
      }
      return result;
    };
    const list = (value, label, convert, allowEmpty = false) => {
      if (!Array.isArray(value) || value.length > 64 || (!allowEmpty && !value.length)) {
        throw new Error(`${label}: ${allowEmpty ? '0' : '1'}~64개의 항목을 입력하세요.`);
      }
      return [...new Set(value.map((item) => {
        const result = convert(item, label);
        if (!result) throw new Error(`${label}: 빈 항목은 사용할 수 없습니다.`);
        return result;
      }))];
    };
    normalize('matches', (value) => list(value, '감시할 URL', (item, label) => {
      const pattern = string(item, label);
      if (pattern === 'file:///*' && defaults.matches.includes(pattern)) return pattern;
      const match = pattern.match(/^(https?):\/\/([^/]+)(\/[^#]*)$/);
      if (!match || /[*@:\s?#\\]/.test(match[2])) {
        throw new Error('URL은 https://example.com/*처럼 정확한 호스트와 경로를 입력하세요.');
      }
      let parsed;
      try { parsed = new URL(pattern); } catch { throw new Error('올바른 URL 패턴을 입력하세요.'); }
      if (!parsed.hostname || parsed.username || parsed.password || parsed.port) throw new Error('URL에 계정이나 포트 번호를 넣을 수 없습니다.');
      return `${match[1]}://${parsed.hostname}${match[3]}`;
    }, true));
    for (const [key, label] of [
      ['image_selectors', '이미지 선택자'], ['question_selectors', '질문 선택자'], ['answer_selectors', '답변 선택자'],
    ]) normalize(key, (value) => list(value, label, selector));
    normalize('container_selector', (value) => selector(value, '감시 영역 선택자'));
    normalize('submit_selector', (value) => selector(value, '제출 버튼 선택자'));
    for (const [key, min, max, label] of [
      ['worker_count', 1, 4, '추론 워커 수'], ['max_pending', 1, 64, '최대 요청 수'],
      ['idle_timeout_ms', 10_000, 1_800_000, '유휴 시간(ms)'],
    ]) normalize(key, (value) => {
      if (!Number.isInteger(value) || value < min || value > max) throw new Error(`${label}: ${min}~${max} 사이의 정수를 입력하세요.`);
      return value;
    });
    for (const key of ['heatmap_enabled', 'auto_submit', 'profiling']) normalize(key, (value) => {
      if (typeof value !== 'boolean') throw new Error(`${key}: 켜짐 또는 꺼짐 값이 필요합니다.`);
      return value;
    });
    if (settings.auto_submit && !settings.submit_selector) {
      if (strict) throw new Error('자동 제출을 켜려면 제출 버튼 선택자를 입력하세요.');
      settings.auto_submit = false;
    }
    return settings;
  }

  async function readSettings(browserApi = api()) {
    if (!browserApi?.storage?.local) return defaultSettings();
    const stored = await browserApi.storage.local.get(SETTINGS_KEY);
    return normalizeSettings(stored[SETTINGS_KEY]);
  }

  async function saveSettings(value, browserApi = api()) {
    const settings = normalizeSettings(value, { strict: true });
    await browserApi.storage.local.set({ [SETTINGS_KEY]: settings });
    return settings;
  }

  function subscribeSettings(listener, browserApi = api()) {
    const changed = (changes, area) => {
      if (area === 'local' && Object.hasOwn(changes, SETTINGS_KEY)) {
        listener(normalizeSettings(changes[SETTINGS_KEY].newValue));
      }
    };
    browserApi?.storage?.onChanged?.addListener(changed);
    return () => browserApi?.storage?.onChanged?.removeListener(changed);
  }

  // Hosts already covered by static scripts need no additional registration.
  function additionalMatches(settings, sites = siteConfig()) {
    return settings.matches.filter((pattern) => !(sites.matches ?? []).some((builtIn) => {
      const origin = builtIn.match(/^(https?):\/\/[^/]+/);
      return builtIn === pattern || (origin && builtIn === `${origin[0]}/*` && pattern.startsWith(`${origin[0]}/`));
    }));
  }

  globalThis.RECEIPT_READER_SETTINGS = Object.freeze({
    SETTINGS_KEY, defaultSettings, normalizeSettings, matchesSite,
    readSettings, saveSettings, subscribeSettings, additionalMatches,
  });
})();
