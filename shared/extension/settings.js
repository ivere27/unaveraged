import './settings-core.js';

export const {
  SETTINGS_KEY, defaultSettings, normalizeSettings, matchesSite,
  readSettings, saveSettings, subscribeSettings, additionalMatches,
} = globalThis.RECEIPT_READER_SETTINGS;
export const loadSettings = readSettings;

const SCRIPT_ID = 'reader-custom-sites';
let registration = Promise.resolve();

// Call from the background, which owns registration updates across option pages.
export function syncContentScripts(settings, api = globalThis.browser ?? globalThis.chrome) {
  const update = async () => {
    if (!api?.scripting || !api?.permissions) return;
    const requested = additionalMatches(normalizeSettings(settings));
    const allowed = await Promise.all(requested.map(async (pattern) => (
      await api.permissions.contains({ origins: [pattern] }) ? pattern : null
    )));
    const matches = allowed.filter(Boolean);
    const existing = await api.scripting.getRegisteredContentScripts({ ids: [SCRIPT_ID] });
    if (!matches.length) {
      if (existing.length) await api.scripting.unregisterContentScripts({ ids: [SCRIPT_ID] });
      return;
    }
    const script = {
      id: SCRIPT_ID, matches,
      js: ['shared/extension/site-config.js', 'shared/extension/settings-core.js', 'shared/extension/heatmap-view.js', 'shared/extension/content.js'],
      css: ['shared/extension/content.css'],
      allFrames: true, runAt: 'document_idle', persistAcrossSessions: true,
    };
    if (existing.length) await api.scripting.updateContentScripts([script]);
    else await api.scripting.registerContentScripts([script]);
  };
  const result = registration.then(update);
  registration = result.catch(() => {});
  return result;
}

export async function initializeContentScripts(api = globalThis.browser ?? globalThis.chrome) {
  await syncContentScripts(await readSettings(api), api);
}
