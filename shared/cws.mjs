#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PROJECTS = ['digit', 'vqa'];
const PUBLISHER_ID_RE = /^[A-Za-z0-9_-]{1,128}$/;
const ITEM_ID_RE = /^[a-z0-9]{32}$/;
const SEMVER_RE = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;
const CHROME_VERSION_RE = /^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*)){0,3}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const PUBLISH_TYPES = new Set(['DEFAULT_PUBLISH', 'STAGED_PUBLISH']);
const ASYNC_IN_PROGRESS_STATES = new Set(['IN_PROGRESS', 'UPLOAD_IN_PROGRESS']);
const ACTIVE_PUBLISHED_STATES = new Set(['PUBLISHED', 'PUBLISHED_TO_TESTERS']);
const ACTIVE_SUBMITTED_STATES = new Set(['PENDING_REVIEW', 'STAGED']);
const DEFAULT_HTTP_TIMEOUT_MS = 30_000;
const DEFAULT_UPLOAD_TIMEOUT_MS = 120_000;
const MIN_CLI_POLL_INTERVAL_MS = 1000;
const DEFAULT_MAX_TRANSIENT_POLL_ERRORS = 3;

const COMMAND_OPTIONS = {
  status: new Set(['keyFile']),
  upload: new Set([
    'keyFile', 'wait', 'publish', 'publishType', 'skipReview',
    'blockOnWarnings', 'deployPercentage', 'pollIntervalMs', 'pollTimeoutMs',
  ]),
  publish: new Set([
    'keyFile', 'publishType', 'skipReview', 'blockOnWarnings', 'deployPercentage',
  ]),
  'publish-ci': new Set([
    'keyFile', 'tag', 'dist', 'project', 'root', 'uploadOnly', 'requireChecksums',
    'dryRun', 'publishType', 'skipReview', 'blockOnWarnings', 'pollIntervalMs', 'pollTimeoutMs',
  ]),
};

export function resolveKeyFile(explicit) {
  if (explicit) {
    return explicit;
  }
  const candidates = [
    process.env.GOOGLE_APPLICATION_CREDENTIALS,
    process.env.CWS_KEY_FILE,
    'cws-service-account.private.json',
    'cws-credentials.private.json',
    'cws-credentials.json',
    'service-account.json',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[0] || 'cws-service-account.private.json';
}

export function loadCredentials(keyFile) {
  const resolved = resolveKeyFile(keyFile);
  let stats;
  try {
    stats = fs.statSync(resolved);
  } catch (error) {
    throw new Error(`Chrome Web Store key file not found: ${resolved}`, { cause: error });
  }
  if (!stats.isFile()) {
    throw new Error(`Chrome Web Store key path must be a regular file: ${resolved}`);
  }
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(resolved, 'utf8'));
  } catch (error) {
    throw new Error(`Invalid JSON in Chrome Web Store key file: ${resolved}`, { cause: error });
  }
  if (!parsed || typeof parsed.client_email !== 'string' || !parsed.client_email
      || typeof parsed.private_key !== 'string' || !parsed.private_key) {
    throw new Error(`Chrome Web Store key file must contain client_email and private_key: ${resolved}`);
  }
  return parsed;
}

function base64UrlEncode(str) {
  return Buffer.from(str)
    .toString('base64')
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');
}

function requireValidIds(publisherId, itemId) {
  if (typeof publisherId !== 'string' || !PUBLISHER_ID_RE.test(publisherId)) {
    throw new Error(`Invalid Chrome Web Store publisherId: ${String(publisherId)}`);
  }
  if (typeof itemId !== 'string' || !ITEM_ID_RE.test(itemId)) {
    throw new Error(`Invalid Chrome Web Store itemId: ${String(itemId)}`);
  }
}

function timeoutOption(timeoutMs) {
  if (Number.isFinite(timeoutMs) && timeoutMs > 0 && typeof AbortSignal?.timeout === 'function') {
    return { signal: AbortSignal.timeout(timeoutMs) };
  }
  return {};
}

async function parseResponseBody(res) {
  const text = await res.text();
  if (!text || !text.trim()) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { rawText: text.slice(0, 500) };
  }
}

export function compareChromeVersions(a, b) {
  if (typeof a !== 'string' || !CHROME_VERSION_RE.test(a)) {
    throw new Error(`Invalid Chrome version string: ${String(a)}`);
  }
  if (typeof b !== 'string' || !CHROME_VERSION_RE.test(b)) {
    throw new Error(`Invalid Chrome version string: ${String(b)}`);
  }
  const partsA = a.split('.').map(Number);
  const partsB = b.split('.').map(Number);
  for (let i = 0; i < 4; i++) {
    const left = partsA[i] ?? 0;
    const right = partsB[i] ?? 0;
    if (left < right) return -1;
    if (left > right) return 1;
  }
  return 0;
}

export function validateZipFile(zipPath) {
  if (!zipPath || typeof zipPath !== 'string') {
    throw new Error('Archive path is required.');
  }
  let lstat;
  try {
    lstat = fs.lstatSync(zipPath);
  } catch (error) {
    throw new Error(`Extension ZIP not found: ${zipPath}`, { cause: error });
  }
  if (lstat.isSymbolicLink() || !lstat.isFile()) {
    throw new Error(`Extension ZIP must be a regular non-symlink file: ${zipPath}`);
  }
  if (lstat.size < 22) {
    throw new Error(`Extension ZIP is empty or truncated: ${zipPath}`);
  }
  const fd = fs.openSync(zipPath, 'r');
  try {
    const header = Buffer.alloc(4);
    const bytesRead = fs.readSync(fd, header, 0, 4, 0);
    if (bytesRead < 4 || header[0] !== 0x50 || header[1] !== 0x4b || header[2] !== 0x03 || header[3] !== 0x04) {
      throw new Error(`File is not a valid non-empty ZIP archive: ${zipPath}`);
    }
  } finally {
    fs.closeSync(fd);
  }
}

export function parseChecksumsText(rawText) {
  const map = new Map();
  for (const rawLine of rawText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const match = line.match(/^([0-9a-f]{64})\s+\*?(.+)$/);
    if (!match) {
      throw new Error(`Invalid SHA256SUMS line: ${line}`);
    }
    const [, digest, filename] = match;
    if (!SHA256_RE.test(digest) || map.has(filename)) {
      throw new Error(`Duplicate or invalid SHA256SUMS entry: ${filename}`);
    }
    map.set(filename, digest);
  }
  return map;
}

export function fileSha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

export async function getAccessToken(
  creds,
  { fetchImpl = globalThis.fetch, now = () => Date.now(), httpTimeoutMs = DEFAULT_HTTP_TIMEOUT_MS } = {},
) {
  const iat = Math.floor(now() / 1000);
  const header = { alg: 'RS256', typ: 'JWT' };
  const claimSet = {
    iss: creds.client_email,
    scope: 'https://www.googleapis.com/auth/chromewebstore',
    aud: 'https://oauth2.googleapis.com/token',
    exp: iat + 3600,
    iat,
  };

  const unsignedJwt = `${base64UrlEncode(JSON.stringify(header))}.${base64UrlEncode(JSON.stringify(claimSet))}`;
  const signer = crypto.createSign('RSA-SHA256');
  signer.update(unsignedJwt);
  const signature = signer.sign(creds.private_key, 'base64')
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');

  const assertion = `${unsignedJwt}.${signature}`;
  const res = await fetchImpl('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
      assertion,
    }).toString(),
    ...timeoutOption(httpTimeoutMs),
  });

  const data = await parseResponseBody(res);
  if (!res.ok || !data.access_token) {
    const detail = data.error_description || data.error || JSON.stringify(data);
    throw new Error(`Failed to obtain Chrome Web Store access token (HTTP ${res.status}): ${detail}`);
  }
  return data.access_token;
}

export async function resolveAuthToken({
  accessToken,
  keyFile,
  fetchImpl = globalThis.fetch,
  now = () => Date.now(),
  httpTimeoutMs = DEFAULT_HTTP_TIMEOUT_MS,
} = {}) {
  if (accessToken) return accessToken;
  const creds = loadCredentials(keyFile);
  return getAccessToken(creds, { fetchImpl, now, httpTimeoutMs });
}

export async function fetchStatus({
  publisherId,
  itemId,
  keyFile,
  accessToken,
  fetchImpl = globalThis.fetch,
  now,
  httpTimeoutMs = DEFAULT_HTTP_TIMEOUT_MS,
}) {
  requireValidIds(publisherId, itemId);
  const token = await resolveAuthToken({ accessToken, keyFile, fetchImpl, now, httpTimeoutMs });

  const url = `https://chromewebstore.googleapis.com/v2/publishers/${publisherId}/items/${itemId}:fetchStatus`;
  const res = await fetchImpl(url, {
    method: 'GET',
    headers: { Authorization: `Bearer ${token}` },
    ...timeoutOption(httpTimeoutMs),
  });

  const body = await parseResponseBody(res);
  return { ok: res.ok, status: res.status, data: body };
}

export async function uploadPackage({
  publisherId,
  itemId,
  zipPath,
  keyFile,
  accessToken,
  fetchImpl = globalThis.fetch,
  now,
  httpTimeoutMs = DEFAULT_UPLOAD_TIMEOUT_MS,
}) {
  requireValidIds(publisherId, itemId);
  validateZipFile(zipPath);
  const token = await resolveAuthToken({ accessToken, keyFile, fetchImpl, now, httpTimeoutMs });

  const fileBytes = fs.readFileSync(zipPath);
  const url = `https://chromewebstore.googleapis.com/upload/v2/publishers/${publisherId}/items/${itemId}:upload`;
  const res = await fetchImpl(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/zip',
    },
    body: fileBytes,
    ...timeoutOption(httpTimeoutMs),
  });

  const body = await parseResponseBody(res);
  return { ok: res.ok, status: res.status, data: body };
}

function validatePublishOptions({ publishType = 'DEFAULT_PUBLISH', deployPercentage } = {}) {
  if (!PUBLISH_TYPES.has(publishType)) {
    throw new Error(`Unsupported Chrome Web Store publishType: ${publishType}`);
  }
  if (deployPercentage !== undefined) {
    if (!Number.isInteger(deployPercentage) || deployPercentage < 0 || deployPercentage > 100) {
      throw new Error(`deployPercentage must be an integer between 0 and 100: ${deployPercentage}`);
    }
  }
}

export async function publishItem({
  publisherId,
  itemId,
  publishType = 'DEFAULT_PUBLISH',
  skipReview = false,
  blockOnWarnings = false,
  deployPercentage,
  keyFile,
  accessToken,
  fetchImpl = globalThis.fetch,
  now,
  httpTimeoutMs = DEFAULT_HTTP_TIMEOUT_MS,
}) {
  requireValidIds(publisherId, itemId);
  validatePublishOptions({ publishType, deployPercentage });
  const payload = { publishType };
  if (skipReview) payload.skipReview = true;
  if (blockOnWarnings) payload.blockOnWarnings = true;
  if (deployPercentage !== undefined) {
    payload.deployInfos = [{ deployPercentage }];
  }

  const token = await resolveAuthToken({ accessToken, keyFile, fetchImpl, now, httpTimeoutMs });
  const url = `https://chromewebstore.googleapis.com/v2/publishers/${publisherId}/items/${itemId}:publish`;
  const res = await fetchImpl(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    ...timeoutOption(httpTimeoutMs),
  });

  const body = await parseResponseBody(res);
  return { ok: res.ok, status: res.status, data: body };
}

function isTransientPollHttpStatus(status) {
  return status === 429 || (Number.isInteger(status) && status >= 500 && status < 600);
}

export async function waitForUpload({
  publisherId,
  itemId,
  initialUploadData,
  keyFile,
  accessToken,
  fetchImpl = globalThis.fetch,
  sleep = (ms) => delay(ms),
  now = () => Date.now(),
  pollIntervalMs = 5000,
  pollTimeoutMs = 120000,
  httpTimeoutMs = DEFAULT_HTTP_TIMEOUT_MS,
  maxTransientPollErrors = DEFAULT_MAX_TRANSIENT_POLL_ERRORS,
}) {
  requireValidIds(publisherId, itemId);
  if (!Number.isFinite(pollIntervalMs) || pollIntervalMs < 0) {
    throw new Error(`Invalid pollIntervalMs: ${pollIntervalMs}`);
  }
  if (!Number.isFinite(pollTimeoutMs) || pollTimeoutMs < 0) {
    throw new Error(`Invalid pollTimeoutMs: ${pollTimeoutMs}`);
  }
  // CWS v2 :upload omits crxVersion while uploadState is IN_PROGRESS, and :fetchStatus
  // reports crxVersion only on submitted/published revisions; ZIP/manifest version checks
  // are already enforced upstream by loadCiTargets() and shared/release.py verify-ci.
  let state = initialUploadData?.uploadState;
  const crxVersion = initialUploadData?.crxVersion;
  if (!state || state === 'SUCCEEDED') {
    return { uploadState: state || 'SUCCEEDED', crxVersion };
  }
  if (state === 'FAILED') {
    throw new Error(`Chrome Web Store package upload failed for item ${itemId}: ${JSON.stringify(initialUploadData)}`);
  }
  if (!ASYNC_IN_PROGRESS_STATES.has(state)) {
    throw new Error(`Unexpected Chrome Web Store uploadState "${state}" for item ${itemId}`);
  }

  const token = await resolveAuthToken({ accessToken, keyFile, fetchImpl, now, httpTimeoutMs });
  const deadline = now() + pollTimeoutMs;
  let transientErrors = 0;

  while (true) {
    if (now() >= deadline) {
      throw new Error(`Timed out waiting for Chrome Web Store async upload for item ${itemId}`);
    }
    await sleep(pollIntervalMs);

    let statusRes;
    try {
      statusRes = await fetchStatus({
        publisherId,
        itemId,
        accessToken: token,
        fetchImpl,
        now,
        httpTimeoutMs,
      });
    } catch (error) {
      transientErrors += 1;
      if (transientErrors <= maxTransientPollErrors && now() < deadline) {
        console.warn(`Transient error polling CWS status for ${itemId} (${transientErrors}/${maxTransientPollErrors}): ${error?.message || String(error)}`);
        continue;
      }
      throw new Error(`Failed polling upload status for ${itemId}: ${error?.message || String(error)}`, { cause: error });
    }

    if (!statusRes.ok) {
      if (isTransientPollHttpStatus(statusRes.status)) {
        transientErrors += 1;
        if (transientErrors <= maxTransientPollErrors && now() < deadline) {
          console.warn(`Transient HTTP ${statusRes.status} polling CWS status for ${itemId} (${transientErrors}/${maxTransientPollErrors})`);
          continue;
        }
      }
      throw new Error(`Failed polling upload status for ${itemId} (HTTP ${statusRes.status}): ${JSON.stringify(statusRes.data)}`);
    }

    transientErrors = 0;
    state = statusRes.data?.lastAsyncUploadState;
    if (state === 'SUCCEEDED') {
      return { uploadState: state, crxVersion };
    }
    if (state === 'FAILED' || state === 'NOT_FOUND') {
      throw new Error(`Chrome Web Store async upload ended with state ${state} for item ${itemId}`);
    }
    if (!ASYNC_IN_PROGRESS_STATES.has(state)) {
      throw new Error(`Unexpected lastAsyncUploadState "${state}" while polling item ${itemId}`);
    }
  }
}

function extractChannelVersions(revisionStatus) {
  const channels = Array.isArray(revisionStatus?.distributionChannels)
    ? revisionStatus.distributionChannels
    : [];
  return channels
    .map((channel) => channel?.crxVersion)
    .filter((value) => typeof value === 'string' && CHROME_VERSION_RE.test(value));
}

export function evaluateItemPreflight(
  statusData,
  expectedVersion,
  { uploadOnly = false, publishType = 'DEFAULT_PUBLISH' } = {},
) {
  if (statusData?.takenDown === true) {
    return {
      decision: 'blocker',
      reason: 'Item is currently taken down in Chrome Web Store for a policy violation.',
    };
  }
  if (ASYNC_IN_PROGRESS_STATES.has(statusData?.lastAsyncUploadState)) {
    return {
      decision: 'blocker',
      reason: 'An async package upload is already in progress for this item.',
    };
  }

  const publishedRev = statusData?.publishedItemRevisionStatus;
  const publishedState = publishedRev?.state;
  const publishedVersions = extractChannelVersions(publishedRev);

  const submittedRev = statusData?.submittedItemRevisionStatus;
  const submittedState = submittedRev?.state;
  const submittedVersions = extractChannelVersions(submittedRev);

  if (publishedVersions.length > 0 && expectedVersion) {
    const newerVersion = publishedVersions.find((v) => compareChromeVersions(v, expectedVersion) > 0);
    if (newerVersion) {
      return {
        decision: 'blocker',
        reason: `Chrome Web Store already has newer published version ${newerVersion} (> ${expectedVersion}).`,
      };
    }
    const publishedMatch = publishedVersions.find((v) => compareChromeVersions(v, expectedVersion) === 0);
    if (publishedMatch && ACTIVE_PUBLISHED_STATES.has(publishedState) && !ACTIVE_SUBMITTED_STATES.has(submittedState)) {
      return {
        decision: 'skip',
        reason: 'already_published',
        version: publishedMatch,
        state: publishedState,
      };
    }
  }

  if (submittedState && ACTIVE_SUBMITTED_STATES.has(submittedState)) {
    const matchedSubmitted = expectedVersion
      ? submittedVersions.find((v) => compareChromeVersions(v, expectedVersion) === 0)
      : undefined;
    const describedSubmitted = submittedVersions[0] || 'unknown';
    if (submittedState === 'PENDING_REVIEW') {
      if (matchedSubmitted && !uploadOnly) {
        return {
          decision: 'skip',
          reason: 'already_submitted',
          version: matchedSubmitted,
          state: submittedState,
        };
      }
      return {
        decision: 'blocker',
        reason: `Item already has a PENDING_REVIEW submission (version ${describedSubmitted}).`,
      };
    }
    if (submittedState === 'STAGED') {
      if (matchedSubmitted) {
        if (uploadOnly || publishType === 'STAGED_PUBLISH') {
          return {
            decision: 'skip',
            reason: 'already_staged',
            version: matchedSubmitted,
            state: submittedState,
          };
        }
        return {
          decision: 'publish_only',
          reason: 'staged_ready',
          version: matchedSubmitted,
          state: submittedState,
        };
      }
      return {
        decision: 'blocker',
        reason: `Item has an active STAGED submission (version ${describedSubmitted}).`,
      };
    }
  }

  return { decision: uploadOnly ? 'upload_only' : 'upload_and_publish' };
}

export function loadCiTargets({
  tag,
  dist,
  project = 'all',
  root = ROOT,
  requireChecksums = false,
}) {
  if (typeof tag !== 'string' || !tag.startsWith('v') || !SEMVER_RE.test(tag.slice(1))) {
    throw new Error(`Stable CWS release requires a strict vMAJOR.MINOR.PATCH tag, got: ${String(tag)}`);
  }
  const version = tag.slice(1);
  if (!dist) {
    throw new Error('--dist directory is required.');
  }
  const distPath = path.isAbsolute(dist) ? dist : path.join(root, dist);
  let distStat;
  try {
    distStat = fs.lstatSync(distPath);
  } catch (error) {
    throw new Error(`Release dist directory not found: ${distPath}`, { cause: error });
  }
  if (distStat.isSymbolicLink() || !distStat.isDirectory()) {
    throw new Error(`Release dist path must be a real directory: ${distPath}`);
  }

  const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const lock = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8'));
  const rootPkgVersion = lock?.packages?.['']?.version;
  if (pkg.version !== version || lock.version !== version || rootPkgVersion !== version) {
    throw new Error(
      `Project version mismatch for tag ${tag}: package.json=${pkg.version}, `
      + `package-lock.json=${lock.version}, root-package=${rootPkgVersion}`
    );
  }

  const checksumPath = path.join(distPath, 'SHA256SUMS');
  let checksums = null;
  if (fs.existsSync(checksumPath)) {
    const csStat = fs.lstatSync(checksumPath);
    if (csStat.isSymbolicLink() || !csStat.isFile()) {
      throw new Error(`SHA256SUMS must be a regular non-symlink file: ${checksumPath}`);
    }
    checksums = parseChecksumsText(fs.readFileSync(checksumPath, 'utf8'));
  } else if (requireChecksums) {
    throw new Error(`Missing required SHA256SUMS in ${distPath}`);
  }

  const selectedProjects = project === 'all' ? PROJECTS : [project];
  for (const selected of selectedProjects) {
    if (!PROJECTS.includes(selected)) {
      throw new Error(`Unknown project "${selected}"; expected all, digit, or vqa`);
    }
  }

  const targets = [];
  for (const name of selectedProjects) {
    const configPath = path.join(root, name, 'extension.json');
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    if (config.id !== name) {
      throw new Error(`${configPath}: extension id ${config.id} differs from ${name}`);
    }
    if (config.version !== version) {
      throw new Error(`${configPath}: version ${config.version} does not match release tag ${tag}`);
    }
    const cws = config.chrome_web_store;
    if (!cws || typeof cws.publisher_id !== 'string' || typeof cws.item_id !== 'string') {
      throw new Error(`${configPath}: missing chrome_web_store.publisher_id / item_id`);
    }
    requireValidIds(cws.publisher_id, cws.item_id);

    const archiveName = `${config.slug}-chrome.zip`;
    const zipPath = path.join(distPath, archiveName);
    validateZipFile(zipPath);
    const sha256 = fileSha256(zipPath);
    if (checksums !== null) {
      const expectedDigest = checksums.get(archiveName);
      if (!expectedDigest) {
        throw new Error(`Missing ${archiveName} in ${checksumPath}`);
      }
      if (sha256 !== expectedDigest) {
        throw new Error(`SHA256SUMS mismatch for ${archiveName}: expected ${expectedDigest}, got ${sha256}`);
      }
    }
    targets.push({
      project: name,
      slug: config.slug,
      version,
      publisherId: cws.publisher_id,
      itemId: cws.item_id,
      archiveName,
      zipPath,
      sha256,
    });
  }
  return targets;
}

function escapeMarkdownCell(value) {
  return String(value ?? '-').replace(/[|\r\n]+/g, ' ').trim() || '-';
}

function formatStepSummaryState(entry) {
  if ((entry.action === 'failed' || entry.state === 'FAILED') && entry.reason) {
    const normalized = String(entry.reason).replace(/\s+/g, ' ').trim();
    const snippet = normalized.length > 160 ? `${normalized.slice(0, 157)}...` : normalized;
    return `FAILED: ${snippet}`;
  }
  return entry.state || entry.reason;
}

function writeStepSummary(results) {
  const summaryPath = process.env.GITHUB_STEP_SUMMARY;
  if (!summaryPath || !results.length) return;
  const rows = results.map((entry) =>
    `| ${escapeMarkdownCell(entry.project)} | \`${escapeMarkdownCell(entry.itemId)}\` | ${escapeMarkdownCell(entry.version)} | \`${escapeMarkdownCell(entry.archiveName)}\` | ${escapeMarkdownCell(entry.action)} | ${escapeMarkdownCell(formatStepSummaryState(entry))} |`
  ).join('\n');
  const markdown = [
    '### Chrome Web Store Publish Summary',
    '',
    '| Extension | Item ID | Version | Archive | Action | Store State |',
    '| --- | --- | --- | --- | --- | --- |',
    rows,
    '',
  ].join('\n');
  fs.appendFileSync(summaryPath, markdown);
}

export async function publishCi({
  tag,
  dist,
  project = 'all',
  root = ROOT,
  keyFile,
  uploadOnly = false,
  requireChecksums = false,
  dryRun = false,
  publishType = 'DEFAULT_PUBLISH',
  skipReview = false,
  blockOnWarnings = false,
  pollIntervalMs = 5000,
  pollTimeoutMs = 120000,
  httpTimeoutMs = DEFAULT_HTTP_TIMEOUT_MS,
  uploadTimeoutMs = DEFAULT_UPLOAD_TIMEOUT_MS,
  maxTransientPollErrors = DEFAULT_MAX_TRANSIENT_POLL_ERRORS,
  fetchImpl = globalThis.fetch,
  sleep = (ms) => delay(ms),
  now = () => Date.now(),
}) {
  if (!PUBLISH_TYPES.has(publishType)) {
    throw new Error(`Unsupported publishType: ${publishType}`);
  }
  const targets = loadCiTargets({ tag, dist, project, root, requireChecksums });

  if (dryRun) {
    return targets.map((target) => ({
      project: target.project,
      itemId: target.itemId,
      publisherId: target.publisherId,
      version: target.version,
      archiveName: target.archiveName,
      sha256: target.sha256,
      action: uploadOnly ? 'dry-run-upload' : 'dry-run-upload-and-publish',
      state: 'DRY_RUN',
    }));
  }

  const creds = loadCredentials(keyFile);
  const accessToken = await getAccessToken(creds, { fetchImpl, now, httpTimeoutMs });

  // Preflight all items before mutating any item so partial conflicts fail early.
  const plans = [];
  for (const target of targets) {
    const statusRes = await fetchStatus({
      publisherId: target.publisherId,
      itemId: target.itemId,
      accessToken,
      fetchImpl,
      now,
      httpTimeoutMs,
    });
    if (!statusRes.ok) {
      throw new Error(
        `[${target.project}] fetchStatus failed for ${target.itemId} `
        + `(HTTP ${statusRes.status}): ${JSON.stringify(statusRes.data)}`
      );
    }
    const plan = evaluateItemPreflight(statusRes.data, target.version, { uploadOnly, publishType });
    if (plan.decision === 'blocker') {
      throw new Error(`[${target.project}] CWS preflight blocked (${target.itemId}): ${plan.reason}`);
    }
    plans.push({ target, plan });
  }

  const results = [];
  try {
    for (const { target, plan } of plans) {
      try {
        if (plan.decision === 'skip') {
          results.push({
            project: target.project,
            itemId: target.itemId,
            publisherId: target.publisherId,
            version: target.version,
            archiveName: target.archiveName,
            sha256: target.sha256,
            action: 'skipped',
            reason: plan.reason,
            state: plan.state,
          });
          continue;
        }

        let crxVersion = plan.version;
        if (plan.decision !== 'publish_only') {
          const uploadRes = await uploadPackage({
            publisherId: target.publisherId,
            itemId: target.itemId,
            zipPath: target.zipPath,
            accessToken,
            fetchImpl,
            now,
            httpTimeoutMs: uploadTimeoutMs,
          });
          if (!uploadRes.ok) {
            throw new Error(
              `[${target.project}] upload failed for ${target.itemId} `
              + `(HTTP ${uploadRes.status}): ${JSON.stringify(uploadRes.data)}`
            );
          }
          const uploadFinal = await waitForUpload({
            publisherId: target.publisherId,
            itemId: target.itemId,
            initialUploadData: uploadRes.data,
            accessToken,
            fetchImpl,
            sleep,
            now,
            pollIntervalMs,
            pollTimeoutMs,
            httpTimeoutMs,
            maxTransientPollErrors,
          });
          crxVersion = uploadFinal.crxVersion;
          if (crxVersion && compareChromeVersions(crxVersion, target.version) !== 0) {
            throw new Error(
              `[${target.project}] uploaded CRX version ${crxVersion} does not match expected ${target.version}`
            );
          }
        }

        if (uploadOnly) {
          results.push({
            project: target.project,
            itemId: target.itemId,
            publisherId: target.publisherId,
            version: target.version,
            archiveName: target.archiveName,
            sha256: target.sha256,
            action: 'uploaded',
            state: 'DRAFT',
          });
          continue;
        }

        const publishRes = await publishItem({
          publisherId: target.publisherId,
          itemId: target.itemId,
          publishType,
          skipReview,
          blockOnWarnings,
          accessToken,
          fetchImpl,
          now,
          httpTimeoutMs,
        });
        if (!publishRes.ok) {
          throw new Error(
            `[${target.project}] publish failed for ${target.itemId} `
            + `(HTTP ${publishRes.status}): ${JSON.stringify(publishRes.data)}`
          );
        }
        results.push({
          project: target.project,
          itemId: target.itemId,
          publisherId: target.publisherId,
          version: target.version,
          archiveName: target.archiveName,
          sha256: target.sha256,
          action: plan.decision === 'publish_only' ? 'published_staged' : 'uploaded_and_submitted',
          state: publishRes.data?.state || 'SUBMITTED',
          ...(publishRes.data?.warningInfo ? { warningInfo: publishRes.data.warningInfo } : {}),
        });
      } catch (error) {
        results.push({
          project: target.project,
          itemId: target.itemId,
          publisherId: target.publisherId,
          version: target.version,
          archiveName: target.archiveName,
          sha256: target.sha256,
          action: 'failed',
          state: 'FAILED',
          reason: error?.message || String(error),
        });
        throw error;
      }
    }
  } finally {
    writeStepSummary(results);
  }
  return results;
}

function parseUrlOrIds(arg1, arg2) {
  // e.g. https://chrome.google.com/webstore/devconsole/<publisherId>/<itemId>/...
  const match = (arg1 || '').match(/devconsole\/([a-zA-Z0-9_-]+)\/([a-z0-9]{32})/);
  if (match) {
    return { publisherId: match[1], itemId: match[2], matchedUrl: true };
  }
  return { publisherId: arg1, itemId: arg2, matchedUrl: false };
}

function takeFlagValue(args, index, flag) {
  const value = args[index + 1];
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`Option ${flag} requires a value`);
  }
  return value;
}

const OPTION_FLAG_NAMES = {
  keyFile: '--key-file',
  tag: '--tag',
  dist: '--dist',
  project: '--project',
  root: '--root',
  publishType: '--publish-type',
  deployPercentage: '--deploy-percentage',
  pollIntervalMs: '--poll-interval-ms',
  pollTimeoutMs: '--poll-timeout-ms',
  publish: '--publish',
  uploadOnly: '--upload-only',
  requireChecksums: '--require-checksums',
  dryRun: '--dry-run',
  wait: '--wait',
  skipReview: '--skip-review',
  blockOnWarnings: '--block-on-warnings',
};

function parseCliArgs(argv) {
  const positionals = [];
  const options = {};
  const providedKeys = new Set();
  const setOpt = (key, value) => {
    options[key] = value;
    providedKeys.add(key);
  };
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token === '--key-file') {
      setOpt('keyFile', takeFlagValue(argv, i++, token));
    } else if (token === '--tag') {
      setOpt('tag', takeFlagValue(argv, i++, token));
    } else if (token === '--dist') {
      setOpt('dist', takeFlagValue(argv, i++, token));
    } else if (token === '--project') {
      setOpt('project', takeFlagValue(argv, i++, token));
    } else if (token === '--root') {
      setOpt('root', path.resolve(takeFlagValue(argv, i++, token)));
    } else if (token === '--publish-type') {
      setOpt('publishType', takeFlagValue(argv, i++, token));
    } else if (token === '--deploy-percentage') {
      const raw = takeFlagValue(argv, i++, token);
      if (!/^(0|[1-9][0-9]*)$/.test(raw)) {
        throw new Error(`Invalid integer for --deploy-percentage: ${raw}`);
      }
      setOpt('deployPercentage', Number(raw));
    } else if (token === '--poll-interval-ms') {
      const raw = takeFlagValue(argv, i++, token);
      if (!/^[1-9][0-9]*$/.test(raw) || Number(raw) < MIN_CLI_POLL_INTERVAL_MS) {
        throw new Error(`--poll-interval-ms must be an integer >= ${MIN_CLI_POLL_INTERVAL_MS}: ${raw}`);
      }
      setOpt('pollIntervalMs', Number(raw));
    } else if (token === '--poll-timeout-ms') {
      const raw = takeFlagValue(argv, i++, token);
      if (!/^[1-9][0-9]*$/.test(raw)) {
        throw new Error(`Invalid positive integer for --poll-timeout-ms: ${raw}`);
      }
      setOpt('pollTimeoutMs', Number(raw));
    } else if (token === '--publish') {
      setOpt('publish', true);
    } else if (token === '--upload-only') {
      setOpt('uploadOnly', true);
    } else if (token === '--require-checksums') {
      setOpt('requireChecksums', true);
    } else if (token === '--dry-run') {
      setOpt('dryRun', true);
    } else if (token === '--wait') {
      setOpt('wait', true);
    } else if (token === '--skip-review') {
      setOpt('skipReview', true);
    } else if (token === '--block-on-warnings') {
      setOpt('blockOnWarnings', true);
    } else if (token.startsWith('--')) {
      throw new Error(`Unknown option: ${token}`);
    } else {
      positionals.push(token);
    }
  }
  return { positionals, options, providedKeys };
}

function validateCommandOptions(command, providedKeys) {
  const allowed = COMMAND_OPTIONS[command];
  if (!allowed) return;
  for (const key of providedKeys) {
    if (!allowed.has(key)) {
      throw new Error(`Unsupported option for "${command}": ${OPTION_FLAG_NAMES[key] || key}`);
    }
  }
}

function usageText() {
  return `Chrome Web Store API CLI
Usage:
  node shared/cws.mjs status <publisherId> <itemId> [--key-file <path>]
  node shared/cws.mjs status <devconsole-url> [--key-file <path>]
  node shared/cws.mjs upload <publisherId> <itemId> <path-to-zip> [--wait] [--publish] [--key-file <path>]
  node shared/cws.mjs upload <devconsole-url> <path-to-zip> [--wait] [--publish] [--key-file <path>]
  node shared/cws.mjs publish <publisherId> <itemId> [--publish-type DEFAULT_PUBLISH|STAGED_PUBLISH] [--skip-review] [--block-on-warnings] [--deploy-percentage <0-100>] [--key-file <path>]
  node shared/cws.mjs publish <devconsole-url> [--publish-type DEFAULT_PUBLISH|STAGED_PUBLISH] [--key-file <path>]
  node shared/cws.mjs publish-ci --tag <vX.Y.Z> --dist <dir> [--project all|digit|vqa] [--upload-only] [--require-checksums] [--dry-run] [--key-file <path>]
`;
}

export async function runCli(argv, { fetchImpl = globalThis.fetch, sleep = (ms) => delay(ms), now = () => Date.now() } = {}) {
  const [command, ...rest] = argv;
  if (!command || command === 'help' || command === '--help' || command === '-h') {
    console.log(usageText());
    return 0;
  }

  if (!Object.hasOwn(COMMAND_OPTIONS, command)) {
    throw new Error(`Unknown command: ${command}`);
  }

  const { positionals, options, providedKeys } = parseCliArgs(rest);
  validateCommandOptions(command, providedKeys);

  if (command === 'status') {
    const { publisherId, itemId } = parseUrlOrIds(positionals[0], positionals[1]);
    if (!publisherId || !itemId) {
      throw new Error('publisherId and itemId are required.');
    }
    const result = await fetchStatus({
      publisherId,
      itemId,
      keyFile: options.keyFile,
      fetchImpl,
      now,
    });
    console.log(JSON.stringify(result, null, 2));
    return result.ok ? 0 : 1;
  }

  if (command === 'upload') {
    const { publisherId, itemId, matchedUrl } = parseUrlOrIds(positionals[0], positionals[1]);
    const zipPath = matchedUrl ? positionals[1] : positionals[2];
    if (!publisherId || !itemId || !zipPath) {
      throw new Error('publisherId, itemId, and zipPath are required.');
    }
    requireValidIds(publisherId, itemId);
    validateZipFile(zipPath);
    if (options.publish || options.publishType !== undefined || options.deployPercentage !== undefined) {
      validatePublishOptions({
        publishType: options.publishType ?? 'DEFAULT_PUBLISH',
        deployPercentage: options.deployPercentage,
      });
    }
    const accessToken = await resolveAuthToken({
      keyFile: options.keyFile,
      fetchImpl,
      now,
    });
    const uploadResult = await uploadPackage({
      publisherId,
      itemId,
      zipPath,
      accessToken,
      fetchImpl,
      now,
    });
    if (!uploadResult.ok) {
      console.log(JSON.stringify(uploadResult, null, 2));
      return 1;
    }
    let settled = uploadResult.data;
    if (options.wait || options.publish) {
      settled = await waitForUpload({
        publisherId,
        itemId,
        initialUploadData: uploadResult.data,
        accessToken,
        fetchImpl,
        sleep,
        now,
        pollIntervalMs: options.pollIntervalMs,
        pollTimeoutMs: options.pollTimeoutMs,
      });
    }
    if (!options.publish) {
      console.log(JSON.stringify({ ...uploadResult, settled }, null, 2));
      return 0;
    }
    const publishResult = await publishItem({
      publisherId,
      itemId,
      publishType: options.publishType,
      skipReview: options.skipReview,
      blockOnWarnings: options.blockOnWarnings,
      deployPercentage: options.deployPercentage,
      accessToken,
      fetchImpl,
      now,
    });
    console.log(JSON.stringify({ upload: uploadResult, settled, publish: publishResult }, null, 2));
    return publishResult.ok ? 0 : 1;
  }

  if (command === 'publish') {
    const { publisherId, itemId } = parseUrlOrIds(positionals[0], positionals[1]);
    if (!publisherId || !itemId) {
      throw new Error('publisherId and itemId are required.');
    }
    const result = await publishItem({
      publisherId,
      itemId,
      publishType: options.publishType,
      skipReview: options.skipReview,
      blockOnWarnings: options.blockOnWarnings,
      deployPercentage: options.deployPercentage,
      keyFile: options.keyFile,
      fetchImpl,
      now,
    });
    console.log(JSON.stringify(result, null, 2));
    return result.ok ? 0 : 1;
  }

  if (command === 'publish-ci') {
    const results = await publishCi({
      ...options,
      fetchImpl,
      sleep,
      now,
    });
    console.log(JSON.stringify(results, null, 2));
    return 0;
  }

  throw new Error(`Unknown command: ${command}`);
}

function checkIsEntrypoint() {
  if (!process.argv[1]) return false;
  try {
    return fs.realpathSync(process.argv[1]) === fs.realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (checkIsEntrypoint()) {
  try {
    const exitCode = await runCli(process.argv.slice(2));
    if (exitCode !== 0) {
      process.exit(exitCode);
    }
  } catch (error) {
    console.error(`Chrome Web Store failed: ${error?.message || String(error)}`);
    process.exit(1);
  }
}
