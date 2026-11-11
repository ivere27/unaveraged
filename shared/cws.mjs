#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

function resolveKeyFile(explicit) {
  if (explicit && fs.existsSync(explicit)) return explicit;
  const candidates = [
    process.env.GOOGLE_APPLICATION_CREDENTIALS,
    process.env.CWS_KEY_FILE,
    'cws-service-account.private.json',
    'cws-credentials.private.json',
    'cws-credentials.json',
    'service-account.json',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return candidates[0] || 'cws-service-account.private.json';
}

const KEY_FILE = resolveKeyFile();

function base64UrlEncode(str) {
  return Buffer.from(str)
    .toString('base64')
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');
}

async function getAccessToken(creds) {
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: 'RS256', typ: 'JWT' };
  const claimSet = {
    iss: creds.client_email,
    scope: 'https://www.googleapis.com/auth/chromewebstore',
    aud: 'https://oauth2.googleapis.com/token',
    exp: now + 3600,
    iat: now,
  };

  const unsignedJwt = `${base64UrlEncode(JSON.stringify(header))}.${base64UrlEncode(JSON.stringify(claimSet))}`;
  const signer = crypto.createSign('RSA-SHA256');
  signer.update(unsignedJwt);
  const signature = signer.sign(creds.private_key, 'base64')
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');

  const assertion = `${unsignedJwt}.${signature}`;
  const res = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
      assertion,
    }).toString(),
  });

  const data = await res.json();
  if (!res.ok || !data.access_token) {
    throw new Error(`Failed to get access token: ${JSON.stringify(data)}`);
  }
  return data.access_token;
}

export async function fetchStatus({ publisherId, itemId, keyFile = KEY_FILE }) {
  const creds = JSON.parse(fs.readFileSync(keyFile, 'utf8'));
  const token = await getAccessToken(creds);

  const url = `https://chromewebstore.googleapis.com/v2/publishers/${publisherId}/items/${itemId}:fetchStatus`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });

  const body = await res.json();
  return { ok: res.ok, status: res.status, data: body };
}

export async function uploadPackage({ publisherId, itemId, zipPath, keyFile = KEY_FILE }) {
  const creds = JSON.parse(fs.readFileSync(keyFile, 'utf8'));
  const token = await getAccessToken(creds);

  const fileBytes = fs.readFileSync(zipPath);
  const url = `https://chromewebstore.googleapis.com/upload/v2/publishers/${publisherId}/items/${itemId}:upload`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/zip',
    },
    body: fileBytes,
  });

  const body = await res.json();
  return { ok: res.ok, status: res.status, data: body };
}

function parseUrlOrIds(arg1, arg2) {
  // Check if arg1 is devconsole URL
  // e.g. https://chrome.google.com/webstore/devconsole/<publisherId>/<itemId>/...
  const match = arg1.match(/devconsole\/([a-zA-Z0-9_\-]+)\/([a-z0-9]{32})/);
  if (match) {
    return { publisherId: match[1], itemId: match[2] };
  }
  return { publisherId: arg1, itemId: arg2 };
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const [,, command, ...args] = process.argv;

  if (!command || command === 'help' || command === '--help') {
    console.log(`Chrome Web Store API CLI
Usage:
  node shared/cws.mjs status <publisherId> <itemId>
  node shared/cws.mjs status <devconsole-url>
  node shared/cws.mjs upload <publisherId> <itemId> <path-to-zip>
  node shared/cws.mjs upload <devconsole-url> <path-to-zip>
`);
    process.exit(0);
  }

  if (command === 'status') {
    const { publisherId, itemId } = parseUrlOrIds(args[0] || '', args[1] || '');
    if (!publisherId || !itemId) {
      console.error('Error: publisherId and itemId are required.');
      process.exit(1);
    }
    const result = await fetchStatus({ publisherId, itemId });
    console.log(JSON.stringify(result, null, 2));
  } else if (command === 'upload') {
    const { publisherId, itemId } = parseUrlOrIds(args[0] || '', args[1] || '');
    const zipPath = args[1] && fs.existsSync(args[1]) ? args[1] : args[2];
    if (!publisherId || !itemId || !zipPath) {
      console.error('Error: publisherId, itemId, and valid zipPath are required.');
      process.exit(1);
    }
    const result = await uploadPackage({ publisherId, itemId, zipPath });
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.error(`Unknown command: ${command}`);
    process.exit(1);
  }
}
