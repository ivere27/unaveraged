import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  compareChromeVersions,
  evaluateItemPreflight,
  fetchStatus,
  loadCiTargets,
  loadCredentials,
  publishCi,
  publishItem,
  runCli,
  uploadPackage,
  waitForUpload,
} from '../cws.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const CWS_SCRIPT = path.join(ROOT, 'shared/cws.mjs');
const DIGIT_ID = 'aibjkkjaomgeppjjeekimpdoonndhmmb';
const VQA_ID = 'fdbkjhljjjacepioaemjjigaicbammaa';
const PUBLISHER_ID = 'd818c2f3-ad14-4f9d-9c6e-cf99c8fa8dee';

function makeTempFixtureRepo(version = '0.2.0') {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'unaveraged-cws-test-'));
  const distDir = path.join(tmp, 'release-dist');
  fs.mkdirSync(distDir, { recursive: true });
  fs.mkdirSync(path.join(tmp, 'digit'), { recursive: true });
  fs.mkdirSync(path.join(tmp, 'vqa'), { recursive: true });

  fs.writeFileSync(path.join(tmp, 'package.json'), JSON.stringify({ name: 'unaveraged', version }));
  fs.writeFileSync(path.join(tmp, 'package-lock.json'), JSON.stringify({
    name: 'unaveraged',
    version,
    packages: { '': { version } },
  }));

  const digitCfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'digit/extension.json'), 'utf8'));
  digitCfg.version = version;
  fs.writeFileSync(path.join(tmp, 'digit/extension.json'), JSON.stringify(digitCfg));

  const vqaCfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'vqa/extension.json'), 'utf8'));
  vqaCfg.version = version;
  fs.writeFileSync(path.join(tmp, 'vqa/extension.json'), JSON.stringify(vqaCfg));

  // Minimal valid ZIP bytes (PK\x03\x04 local header + PK\x05\x06 EOCD >= 22 bytes).
  const zipPayload = Buffer.concat([
    Buffer.from([0x50, 0x4b, 0x03, 0x04]),
    Buffer.alloc(24, 0),
    Buffer.from([0x50, 0x4b, 0x05, 0x06]),
    Buffer.alloc(18, 0),
  ]);
  const digitZip = path.join(distDir, 'unaveraged-digit-chrome.zip');
  const vqaZip = path.join(distDir, 'unaveraged-vqa-chrome.zip');
  fs.writeFileSync(digitZip, Buffer.concat([zipPayload, Buffer.from('digit')]));
  fs.writeFileSync(vqaZip, Buffer.concat([zipPayload, Buffer.from('vqa')]));

  const sha = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  fs.writeFileSync(
    path.join(distDir, 'SHA256SUMS'),
    `${sha(digitZip)}  unaveraged-digit-chrome.zip\n${sha(vqaZip)}  unaveraged-vqa-chrome.zip\n`,
  );

  const { privateKey } = crypto.generateKeyPairSync('rsa', {
    modulusLength: 2048,
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
    publicKeyEncoding: { type: 'spki', format: 'pem' },
  });
  const keyFile = path.join(tmp, 'sa-key.json');
  fs.writeFileSync(keyFile, JSON.stringify({
    client_email: 'cws-ci@example.iam.gserviceaccount.com',
    private_key: privateKey,
  }));

  return { tmp, distDir, digitZip, vqaZip, keyFile };
}

function mockJsonResponse(status, data) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async text() {
      return JSON.stringify(data);
    },
  };
}

test('digit/vqa extension.json chrome_web_store metadata matches STORE.md identifiers', () => {
  const storeMd = fs.readFileSync(path.join(ROOT, 'STORE.md'), 'utf8');
  const digitCfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'digit/extension.json'), 'utf8'));
  const vqaCfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'vqa/extension.json'), 'utf8'));

  assert.deepEqual(digitCfg.chrome_web_store, { publisher_id: PUBLISHER_ID, item_id: DIGIT_ID });
  assert.deepEqual(vqaCfg.chrome_web_store, { publisher_id: PUBLISHER_ID, item_id: VQA_ID });
  assert(storeMd.includes(PUBLISHER_ID));
  assert(storeMd.includes(DIGIT_ID));
  assert(storeMd.includes(VQA_ID));
});

test('compareChromeVersions compares dotted numeric components cleanly', () => {
  assert.equal(compareChromeVersions('0.1.0', '0.1.0.0'), 0);
  assert.equal(compareChromeVersions('0.1.0', '0.2.0'), -1);
  assert.equal(compareChromeVersions('0.10.0', '0.2.0'), 1);
  assert.throws(() => compareChromeVersions('0.1.0-beta', '0.1.0'), /Invalid Chrome version/);
});

test('evaluateItemPreflight handles idempotency, blockers, multi-channel lists, and STAGED_PUBLISH', () => {
  assert.equal(evaluateItemPreflight({}, '0.1.0').decision, 'upload_and_publish');
  assert.equal(evaluateItemPreflight({
    publishedItemRevisionStatus: { state: 'PUBLISHED', distributionChannels: [{ crxVersion: '0.1.0' }] },
  }, '0.2.0').decision, 'upload_and_publish');

  assert.deepEqual(evaluateItemPreflight({
    publishedItemRevisionStatus: { state: 'PUBLISHED', distributionChannels: [{ crxVersion: '0.2.0' }] },
  }, '0.2.0'), { decision: 'skip', reason: 'already_published', version: '0.2.0', state: 'PUBLISHED' });

  // Non-active submittedState variants never fall through when already published at expectedVersion.
  assert.deepEqual(evaluateItemPreflight({
    publishedItemRevisionStatus: { state: 'PUBLISHED', distributionChannels: [{ crxVersion: '0.2.0' }] },
    submittedItemRevisionStatus: { state: 'CANCELLED_BY_USER', distributionChannels: [{ crxVersion: '0.2.0' }] },
  }, '0.2.0'), { decision: 'skip', reason: 'already_published', version: '0.2.0', state: 'PUBLISHED' });

  // Multi-channel distributionChannels checks all channels, not only index 0.
  assert.equal(evaluateItemPreflight({
    publishedItemRevisionStatus: {
      state: 'PUBLISHED',
      distributionChannels: [{ crxVersion: '0.1.0' }, { crxVersion: '0.3.0' }],
    },
  }, '0.2.0').decision, 'blocker');
  assert.deepEqual(evaluateItemPreflight({
    publishedItemRevisionStatus: {
      state: 'PUBLISHED',
      distributionChannels: [{ crxVersion: '0.1.0' }, { crxVersion: '0.2.0' }],
    },
  }, '0.2.0'), { decision: 'skip', reason: 'already_published', version: '0.2.0', state: 'PUBLISHED' });

  assert.deepEqual(evaluateItemPreflight({
    submittedItemRevisionStatus: { state: 'PENDING_REVIEW', distributionChannels: [{ crxVersion: '0.2.0' }] },
  }, '0.2.0'), { decision: 'skip', reason: 'already_submitted', version: '0.2.0', state: 'PENDING_REVIEW' });

  assert.equal(evaluateItemPreflight({
    publishedItemRevisionStatus: { state: 'PUBLISHED', distributionChannels: [{ crxVersion: '0.1.0' }] },
    submittedItemRevisionStatus: { state: 'REJECTED', distributionChannels: [{ crxVersion: '0.2.0' }] },
  }, '0.2.0').decision, 'upload_and_publish');

  assert.equal(evaluateItemPreflight({
    submittedItemRevisionStatus: { state: 'STAGED', distributionChannels: [{ crxVersion: '0.2.0' }] },
  }, '0.2.0').decision, 'publish_only');

  assert.deepEqual(evaluateItemPreflight({
    submittedItemRevisionStatus: { state: 'STAGED', distributionChannels: [{ crxVersion: '0.2.0' }] },
  }, '0.2.0', { publishType: 'STAGED_PUBLISH' }), {
    decision: 'skip',
    reason: 'already_staged',
    version: '0.2.0',
    state: 'STAGED',
  });

  assert.equal(evaluateItemPreflight({ takenDown: true }, '0.2.0').decision, 'blocker');
  assert.equal(evaluateItemPreflight({
    publishedItemRevisionStatus: { state: 'PUBLISHED', distributionChannels: [{ crxVersion: '0.3.0' }] },
  }, '0.2.0').decision, 'blocker');
  assert.equal(evaluateItemPreflight({
    submittedItemRevisionStatus: { state: 'PENDING_REVIEW', distributionChannels: [{ crxVersion: '0.1.9' }] },
  }, '0.2.0').decision, 'blocker');
});

test('publishCi validates inputs offline in dry-run and rejects tampered archives or symlinks', async (t) => {
  const { tmp, distDir, digitZip } = makeTempFixtureRepo('0.2.0');
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const dryRun = await publishCi({
    tag: 'v0.2.0',
    dist: distDir,
    root: tmp,
    requireChecksums: true,
    dryRun: true,
  });
  assert.equal(dryRun.length, 2);
  assert.deepEqual(dryRun.map((r) => r.action), ['dry-run-upload-and-publish', 'dry-run-upload-and-publish']);

  const savedDigit = fs.readFileSync(digitZip);
  fs.writeFileSync(digitZip, Buffer.concat([savedDigit, Buffer.from('-tampered')]));
  await assert.rejects(
    () => publishCi({ tag: 'v0.2.0', dist: distDir, root: tmp, requireChecksums: true, dryRun: true }),
    /SHA256SUMS mismatch/,
  );

  const outsideZip = path.join(tmp, 'outside.zip');
  fs.writeFileSync(outsideZip, savedDigit);
  fs.unlinkSync(digitZip);
  fs.symlinkSync(outsideZip, digitZip);
  assert.throws(
    () => loadCiTargets({ tag: 'v0.2.0', dist: distDir, root: tmp, requireChecksums: true }),
    /non-symlink/,
  );
});

test('publishCi runs preflight, polls async IN_PROGRESS uploads, and calls :publish for both items', async (t) => {
  const { tmp, distDir, keyFile } = makeTempFixtureRepo('0.2.0');
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  const calls = [];
  let digitPollCount = 0;
  let clock = 1_700_000_000_000;
  const fakeSleep = async (ms) => { clock += ms; };

  const fakeFetch = async (url, init = {}) => {
    calls.push({ url, method: init.method || 'GET', body: init.body, signal: init.signal });
    assert(init.signal instanceof AbortSignal, 'fetch requests must pass an AbortSignal timeout');
    if (url === 'https://oauth2.googleapis.com/token') {
      assert.equal(init.method, 'POST');
      const params = new URLSearchParams(init.body);
      assert.equal(params.get('grant_type'), 'urn:ietf:params:oauth:grant-type:jwt-bearer');
      assert.equal(params.get('assertion').split('.').length, 3);
      return mockJsonResponse(200, { access_token: 'mock-access-token' });
    }
    assert.equal(init.headers?.Authorization, 'Bearer mock-access-token');

    if (url.endsWith(`/${DIGIT_ID}:fetchStatus`)) {
      if (calls.some((c) => c.url.endsWith(`/${DIGIT_ID}:upload`))) {
        digitPollCount += 1;
        return mockJsonResponse(200, {
          itemId: DIGIT_ID,
          lastAsyncUploadState: digitPollCount >= 2 ? 'SUCCEEDED' : 'IN_PROGRESS',
        });
      }
      return mockJsonResponse(200, {
        itemId: DIGIT_ID,
        publishedItemRevisionStatus: { state: 'PUBLISHED', distributionChannels: [{ crxVersion: '0.1.0' }] },
      });
    }
    if (url.endsWith(`/${VQA_ID}:fetchStatus`)) {
      return mockJsonResponse(200, {
        itemId: VQA_ID,
        publishedItemRevisionStatus: { state: 'PUBLISHED', distributionChannels: [{ crxVersion: '0.1.0' }] },
      });
    }
    if (url.endsWith(`/${DIGIT_ID}:upload`)) {
      assert.equal(init.headers['Content-Type'], 'application/zip');
      return mockJsonResponse(200, { itemId: DIGIT_ID, uploadState: 'IN_PROGRESS' });
    }
    if (url.endsWith(`/${VQA_ID}:upload`)) {
      return mockJsonResponse(200, { itemId: VQA_ID, crxVersion: '0.2.0', uploadState: 'SUCCEEDED' });
    }
    if (url.endsWith(`/${DIGIT_ID}:publish`) || url.endsWith(`/${VQA_ID}:publish`)) {
      assert.equal(init.method, 'POST');
      assert.deepEqual(JSON.parse(init.body), { publishType: 'DEFAULT_PUBLISH' });
      return mockJsonResponse(200, { state: 'PENDING_REVIEW' });
    }
    throw new Error(`Unexpected fetch URL: ${url}`);
  };

  const results = await publishCi({
    tag: 'v0.2.0',
    dist: distDir,
    root: tmp,
    keyFile,
    requireChecksums: true,
    fetchImpl: fakeFetch,
    sleep: fakeSleep,
    now: () => clock,
    pollIntervalMs: 100,
    pollTimeoutMs: 5000,
  });

  assert.equal(digitPollCount, 2);
  assert.deepEqual(results.map((r) => [r.project, r.action, r.state]), [
    ['digit', 'uploaded_and_submitted', 'PENDING_REVIEW'],
    ['vqa', 'uploaded_and_submitted', 'PENDING_REVIEW'],
  ]);

  const firstUploadIdx = calls.findIndex((c) => c.url.includes(':upload'));
  assert(calls.slice(0, firstUploadIdx).some((c) => c.url.endsWith(`/${DIGIT_ID}:fetchStatus`)));
  assert(calls.slice(0, firstUploadIdx).some((c) => c.url.endsWith(`/${VQA_ID}:fetchStatus`)));
});

test('publishCi records partial progress in GITHUB_STEP_SUMMARY on mid-run failure and supports retry', async (t) => {
  const { tmp, distDir, keyFile } = makeTempFixtureRepo('0.2.0');
  const prevSummary = process.env.GITHUB_STEP_SUMMARY;
  const summaryPath = path.join(tmp, 'step-summary.md');
  process.env.GITHUB_STEP_SUMMARY = summaryPath;
  t.after(() => {
    if (prevSummary === undefined) {
      delete process.env.GITHUB_STEP_SUMMARY;
    } else {
      process.env.GITHUB_STEP_SUMMARY = prevSummary;
    }
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  await assert.rejects(
    () => publishCi({
      tag: 'v0.2.0',
      dist: distDir,
      root: tmp,
      keyFile,
      requireChecksums: true,
      fetchImpl: async (url) => {
        if (url === 'https://oauth2.googleapis.com/token') {
          return mockJsonResponse(200, { access_token: 'tok' });
        }
        if (url.endsWith(':fetchStatus')) {
          return mockJsonResponse(200, {});
        }
        if (url.endsWith(`/${DIGIT_ID}:upload`)) {
          return mockJsonResponse(200, { crxVersion: '0.2.0', uploadState: 'SUCCEEDED' });
        }
        if (url.endsWith(`/${DIGIT_ID}:publish`)) {
          return mockJsonResponse(200, { state: 'PENDING_REVIEW' });
        }
        if (url.endsWith(`/${VQA_ID}:upload`)) {
          return mockJsonResponse(503, { error: { message: 'Temporary outage' } });
        }
        throw new Error(`Unexpected URL: ${url}`);
      },
    }),
    /upload failed/,
  );

  const summaryText = fs.readFileSync(summaryPath, 'utf8');
  assert(summaryText.includes('| digit |'));
  assert(summaryText.includes('uploaded_and_submitted'));
  assert(summaryText.includes('| vqa |'));
  assert(summaryText.includes('| failed |'));
  assert(summaryText.includes('FAILED: [vqa] upload failed'));
  assert(summaryText.includes('Temporary outage'));

  const conflictCalls = [];
  await assert.rejects(
    () => publishCi({
      tag: 'v0.2.0',
      dist: distDir,
      root: tmp,
      keyFile,
      requireChecksums: true,
      fetchImpl: async (url) => {
        conflictCalls.push(url);
        if (url === 'https://oauth2.googleapis.com/token') {
          return mockJsonResponse(200, { access_token: 'tok' });
        }
        if (url.endsWith(`/${DIGIT_ID}:fetchStatus`)) {
          return mockJsonResponse(200, { itemId: DIGIT_ID });
        }
        if (url.endsWith(`/${VQA_ID}:fetchStatus`)) {
          return mockJsonResponse(200, {
            itemId: VQA_ID,
            submittedItemRevisionStatus: {
              state: 'PENDING_REVIEW',
              distributionChannels: [{ crxVersion: '0.1.9' }],
            },
          });
        }
        throw new Error(`Must not upload or publish when preflight blocks: ${url}`);
      },
    }),
    /PENDING_REVIEW/,
  );
  assert.equal(conflictCalls.some((u) => u.includes(':upload') || u.includes(':publish')), false);

  const retryCalls = [];
  const retryResults = await publishCi({
    tag: 'v0.2.0',
    dist: distDir,
    root: tmp,
    keyFile,
    requireChecksums: true,
    fetchImpl: async (url) => {
      retryCalls.push(url);
      if (url === 'https://oauth2.googleapis.com/token') {
        return mockJsonResponse(200, { access_token: 'tok' });
      }
      if (url.endsWith(`/${DIGIT_ID}:fetchStatus`)) {
        return mockJsonResponse(200, {
          itemId: DIGIT_ID,
          submittedItemRevisionStatus: { state: 'PENDING_REVIEW', distributionChannels: [{ crxVersion: '0.2.0' }] },
        });
      }
      if (url.endsWith(`/${VQA_ID}:fetchStatus`)) {
        return mockJsonResponse(200, { itemId: VQA_ID });
      }
      if (url.endsWith(`/${VQA_ID}:upload`)) {
        return mockJsonResponse(200, { crxVersion: '0.2.0', uploadState: 'SUCCEEDED' });
      }
      if (url.endsWith(`/${VQA_ID}:publish`)) {
        return mockJsonResponse(200, { state: 'PENDING_REVIEW' });
      }
      throw new Error(`Unexpected call: ${url}`);
    },
  });
  assert.deepEqual(retryResults.map((r) => [r.project, r.action]), [
    ['digit', 'skipped'],
    ['vqa', 'uploaded_and_submitted'],
  ]);
  assert.equal(retryCalls.some((u) => u.includes(DIGIT_ID) && (u.includes(':upload') || u.includes(':publish'))), false);
});

test('runCli upload --wait --publish reuses one OAuth token and waitForUpload retries transient poll errors', async (t) => {
  const { tmp, digitZip, keyFile } = makeTempFixtureRepo('0.2.0');
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  let tokenCalls = 0;
  let pollAttempt = 0;
  let clock = 1_700_000_000_000;
  const fakeSleep = async (ms) => { clock += ms; };

  const exitCode = await runCli(
    ['upload', PUBLISHER_ID, DIGIT_ID, digitZip, '--publish', '--poll-interval-ms', '1000', '--key-file', keyFile],
    {
      sleep: fakeSleep,
      now: () => clock,
      fetchImpl: async (url) => {
        if (url === 'https://oauth2.googleapis.com/token') {
          tokenCalls += 1;
          return mockJsonResponse(200, { access_token: 'single-tok' });
        }
        if (url.endsWith(`/${DIGIT_ID}:upload`)) {
          return mockJsonResponse(200, { uploadState: 'IN_PROGRESS' });
        }
        if (url.endsWith(`/${DIGIT_ID}:fetchStatus`)) {
          pollAttempt += 1;
          if (pollAttempt === 1) return mockJsonResponse(503, { error: 'backend unavailable' });
          if (pollAttempt === 2) throw new Error('ECONNRESET');
          return mockJsonResponse(200, { lastAsyncUploadState: 'SUCCEEDED' });
        }
        if (url.endsWith(`/${DIGIT_ID}:publish`)) {
          return mockJsonResponse(200, { state: 'PENDING_REVIEW' });
        }
        throw new Error(`Unexpected URL: ${url}`);
      },
    },
  );
  assert.equal(exitCode, 0);
  assert.equal(tokenCalls, 1, 'OAuth token must be minted once for upload + wait + publish');
  assert.equal(pollAttempt, 3);

  await assert.rejects(
    () => waitForUpload({
      publisherId: PUBLISHER_ID,
      itemId: DIGIT_ID,
      initialUploadData: { uploadState: 'IN_PROGRESS' },
      accessToken: 'tok',
      sleep: fakeSleep,
      now: () => clock,
      pollIntervalMs: 100,
      pollTimeoutMs: 5000,
      fetchImpl: async () => mockJsonResponse(403, { error: 'forbidden' }),
    }),
    /HTTP 403/,
  );

  // Invalid IDs on direct waitForUpload() calls must fail immediately without minting tokens, sleeping, or retrying.
  let badIdFetchCalls = 0;
  let badIdSleepCalls = 0;
  await assert.rejects(
    () => waitForUpload({
      publisherId: 'invalid publisher!',
      itemId: DIGIT_ID,
      initialUploadData: { uploadState: 'IN_PROGRESS' },
      keyFile,
      sleep: async () => { badIdSleepCalls += 1; },
      now: () => clock,
      pollIntervalMs: 1000,
      pollTimeoutMs: 5000,
      fetchImpl: async () => {
        badIdFetchCalls += 1;
        return mockJsonResponse(200, {});
      },
    }),
    /Invalid Chrome Web Store publisherId/,
  );
  assert.equal(badIdFetchCalls, 0);
  assert.equal(badIdSleepCalls, 0);

  // runCli upload must validate IDs, ZIP path, and publish options before requesting an OAuth token or uploading.
  let unexpectedFetchCalls = 0;
  const failIfFetched = async (url) => {
    unexpectedFetchCalls += 1;
    throw new Error(`Must validate locally before network calls: ${url}`);
  };
  await assert.rejects(
    () => runCli(
      ['upload', PUBLISHER_ID, 'bad-item-id', digitZip, '--key-file', keyFile],
      { fetchImpl: failIfFetched },
    ),
    /Invalid Chrome Web Store itemId/,
  );
  await assert.rejects(
    () => runCli(
      ['upload', PUBLISHER_ID, DIGIT_ID, path.join(tmp, 'missing.zip'), '--key-file', keyFile],
      { fetchImpl: failIfFetched },
    ),
    /Extension ZIP not found/,
  );
  await assert.rejects(
    () => runCli(
      ['upload', PUBLISHER_ID, DIGIT_ID, digitZip, '--publish', '--publish-type', 'INVALID_TYPE', '--key-file', keyFile],
      { fetchImpl: failIfFetched },
    ),
    /Unsupported Chrome Web Store publishType/,
  );
  await assert.rejects(
    () => runCli(
      ['upload', PUBLISHER_ID, DIGIT_ID, digitZip, '--publish', '--deploy-percentage', '150', '--key-file', keyFile],
      { fetchImpl: failIfFetched },
    ),
    /deployPercentage must be an integer between 0 and 100/,
  );
  assert.equal(unexpectedFetchCalls, 0);
});

test('CLI validates poll interval, rejects inapplicable command options, and resists non-existent argv[1]', async (t) => {
  const { tmp, distDir, keyFile } = makeTempFixtureRepo('0.2.0');
  t.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  assert.throws(
    () => execFileSync(process.execPath, [
      CWS_SCRIPT, 'publish-ci', '--tag', 'v0.2.0', '--dist', distDir,
      '--root', tmp, '--key-file', path.join(tmp, 'missing.json'),
    ], { encoding: 'utf8', stdio: 'pipe' }),
    (err) => err.status === 1 && /key file not found/.test(err.stderr),
  );

  await assert.rejects(
    () => runCli(['publish-ci', '--tag', 'v0.2.0', '--dist', distDir, '--root', tmp, '--poll-interval-ms', '0']),
    /--poll-interval-ms must be an integer >= 1000/,
  );
  await assert.rejects(
    () => runCli(['publish-ci', '--tag', 'v0.2.0', '--dist', distDir, '--root', tmp, '--deploy-percentage', '50']),
    /Unsupported option for "publish-ci": --deploy-percentage/,
  );
  await assert.rejects(
    () => runCli(['status', PUBLISHER_ID, DIGIT_ID, '--tag', 'v0.2.0']),
    /Unsupported option for "status": --tag/,
  );

  // Importing shared/cws.mjs when process.argv[1] does not exist on disk must not throw.
  const imported = execFileSync(process.execPath, [
    '--input-type=module',
    '-e',
    `process.argv[1] = '/nonexistent/wrapper-path.js'; const m = await import(${JSON.stringify(CWS_SCRIPT)}); console.log(typeof m.publishCi);`,
  ], { encoding: 'utf8' });
  assert.equal(imported.trim(), 'function');

  const preview = execFileSync(process.execPath, [
    CWS_SCRIPT, 'publish-ci', '--tag', 'v0.2.0', '--dist', distDir,
    '--root', tmp, '--require-checksums', '--dry-run',
  ], { encoding: 'utf8' });
  const parsed = JSON.parse(preview);
  assert.equal(parsed.length, 2);
  assert.equal(parsed[0].state, 'DRY_RUN');

  assert.equal(loadCredentials(keyFile).client_email, 'cws-ci@example.iam.gserviceaccount.com');
  const status = await fetchStatus({
    publisherId: PUBLISHER_ID,
    itemId: DIGIT_ID,
    accessToken: 'tok',
    fetchImpl: async () => mockJsonResponse(403, { error: { message: 'Forbidden' } }),
  });
  assert.equal(status.ok, false);
  assert.equal(status.status, 403);

  const uploaded = await uploadPackage({
    publisherId: PUBLISHER_ID,
    itemId: DIGIT_ID,
    zipPath: path.join(distDir, 'unaveraged-digit-chrome.zip'),
    accessToken: 'tok',
    fetchImpl: async () => mockJsonResponse(200, { uploadState: 'SUCCEEDED', crxVersion: '0.2.0' }),
  });
  assert.equal(uploaded.ok, true);

  const published = await publishItem({
    publisherId: PUBLISHER_ID,
    itemId: DIGIT_ID,
    publishType: 'STAGED_PUBLISH',
    deployPercentage: 50,
    accessToken: 'tok',
    fetchImpl: async (_url, init) => {
      assert.deepEqual(JSON.parse(init.body), {
        publishType: 'STAGED_PUBLISH',
        deployInfos: [{ deployPercentage: 50 }],
      });
      return mockJsonResponse(200, { state: 'STAGED' });
    },
  });
  assert.equal(published.ok, true);
  assert.equal(published.data.state, 'STAGED');
});
