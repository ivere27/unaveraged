import assert from 'node:assert/strict';
import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { preprocessPixels } from '../extension/preprocess.js';
import { SerialQueue, matchesSite, validateRead, MAX_IMAGE_BYTES } from '../extension/requests.js';
const python = fileURLToPath(new URL('../../.venv/bin/python', import.meta.url));

test('preprocessing agrees with Pillow on upsampling, downsampling, and non-square images', () => {
  for (const [width, height, targetWidth, targetHeight] of [[17, 9, 31, 20], [41, 27, 13, 8], [1, 1, 5, 3], [670, 320, 672, 320]]) {
    const data = Uint8Array.from({ length: width * height * 4 }, (_, i) => (i * 37 + (i >> 3) * 17) % 256);
    const referenceBytes = execFileSync(python, ['-I', '-c', `
import sys
from PIL import Image
import numpy as np
w,h,tw,th = map(int,sys.argv[1:])
im=Image.frombytes('RGBA',(w,h),sys.stdin.buffer.read()).convert('L').resize((tw,th),Image.Resampling.BILINEAR)
x=(np.asarray(im,dtype=np.float32)/255.0-0.5)/0.5
sys.stdout.buffer.write(x.tobytes())`, ...[width, height, targetWidth, targetHeight].map(String)], { input: data, maxBuffer: 4_000_000 });
    const reference = new Float32Array(referenceBytes.buffer.slice(referenceBytes.byteOffset, referenceBytes.byteOffset + referenceBytes.byteLength));
    const result = preprocessPixels({ data, width, height }, targetWidth, targetHeight);
    assert.deepEqual(result, reference, `${width}x${height} -> ${targetWidth}x${targetHeight}`);
  }
});

test('bounded FIFO serializes requests and continues after rejection', async () => {
  const queue = new SerialQueue(2);
  const order = [];
  let release;
  const first = queue.run(async () => { order.push('first'); await new Promise((resolve) => { release = resolve; }); throw new Error('bad image'); });
  const rejected = assert.rejects(first, /bad image/);
  const second = queue.run(async () => { order.push('second'); return 'ok'; });
  await assert.rejects(queue.run(() => {}), /대기/);
  assert.deepEqual(order, ['first']);
  release();
  await rejected;
  assert.equal(await second, 'ok');
  assert.deepEqual(order, ['first', 'second']);
  assert.equal(queue.pending, 0);
  assert.equal(await queue.run(() => 'recovered'), 'recovered');
});

test('only configured URLs and bounded image requests are accepted', () => {
  const sites = JSON.parse(readFileSync(new URL('../sites.json', import.meta.url), 'utf8'));
  const patterns = [...sites.matches, ...sites.development_matches];
  assert(matchesSite('https://smartstore.naver.com/challenge?x=1', patterns));
  assert(matchesSite('https://shopping.naver.com/ns/home', patterns));
  assert(matchesSite('https://shopping.naver.com/ns/home?query=receipt', patterns));
  assert(matchesSite('http://localhost:8123/demo', patterns));
  for (const url of ['https://smartstore.naver.com.evil.test/', 'https://shopping.naver.com.evil.test/',
    'http://smartstore.naver.com/', 'http://shopping.naver.com/ns/home', 'https://example.com/', 'garbage']) {
    assert.equal(matchesSite(url, patterns), false);
  }
  validateRead({ image: 'data:image/png;base64,YQ==', question: 'phone digit 1' });
  assert.throws(() => validateRead({ image: 'https://example.com/private', question: 'phone' }));
  assert.throws(() => validateRead({ image: 'data:image/png;base64,' + 'a'.repeat(MAX_IMAGE_BYTES * 2), question: '' }));
});

test('a question-dependent adapter keeps answers separate in the shared worker cache', async () => {
  const replies = new Map();
  const messages = [];
  const image = 'data:image/png;base64,YQ==';
  const source = readFileSync(new URL('../extension/reader.worker.js', import.meta.url), 'utf8')
    .replace(/^import .*;\n/gm, '');
  let decoded = 0;
  const self = { postMessage(message) { messages.push(message); replies.get(message.id)(message); } };
  vm.runInNewContext(source, {
    self, SerialQueue, validateRead, MAX_IMAGE_BYTES, MAX_PENDING: 16, MAX_PIXELS: 12 * 1024 * 1024,
    crypto: globalThis.crypto, atob, Uint8Array, Blob, performance,
    async createImageBitmap() { decoded++; return { width: 1, height: 1, close() {} }; },
    OffscreenCanvas: class { getContext() {
      return { drawImage() {}, getImageData() { return { data: new Uint8Array(4), width: 1, height: 1 }; } };
    } },
    modelAdapter: {
      cacheKey: ({ imageHash, question }) => JSON.stringify([imageHash, question]),
      async open() { return { async infer({ question, heatmap }) { return question.toUpperCase() + (heatmap ? '_MAP' : ''); }, async close() {} }; },
      answer(value) { return { answer: value }; },
    },
  });
  let id = 0;
  function send(type, question, heatmap = false) {
    const request = { id: ++id, type, image, question, heatmap };
    return new Promise((resolve) => { replies.set(request.id, resolve); self.onmessage({ data: request }); });
  }
  const [first, second, cached] = await Promise.all([send('read', 'one'), send('read', 'two'), send('read', 'one')]);
  assert.deepEqual([first.result.answer, second.result.answer, cached.result.answer], ['ONE', 'TWO', 'ONE']);
  assert.equal(cached.result.cached, true);
  assert.equal(decoded, 2);
  const status = await send('status');
  assert.equal(status.result.initializations, 1);
  assert.equal(status.result.forwardPasses, 2);
  assert.equal(status.result.cacheHits, 1);
  const captured = await send('read', 'one', true);
  assert.equal(captured.result.answer, 'ONE_MAP');
  assert.equal(captured.result.cached, false, 'Turning attention capture on requires a matching inference record');
  const plain = await send('read', 'one', false);
  assert.equal(plain.result.answer, 'ONE');
  assert.equal(plain.result.cached, true, 'The original plain-result cache stays independent');
  assert(messages.every((message) => message.ok));
});


test('public monorepo export contains both adapters and one shared tree, without private files', () => {
  const root = fileURLToPath(new URL('../../', import.meta.url));
  execFileSync(python, ['-I', '-c', `
import importlib.util, os, pathlib, shutil, subprocess, sys, tempfile, zipfile
root=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('extension_build',root/'shared/build.py')
build=importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
with tempfile.TemporaryDirectory(prefix='receipt shared export ') as temporary:
 repo=pathlib.Path(temporary)/'repository with spaces'
 for name in build.SOURCE_FILES:
  target=repo/name; target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(root/name,target)
 for name in ('.env','shared/extension/private.txt','digit/private.png','vqa/private.jpg',
              '.cache/calibration-inputs.json','test-results/private.json','shared/tests/receipt.png'):
  target=repo/name; target.parent.mkdir(parents=True,exist_ok=True);target.write_text('not for publication')
 build.ROOT=repo;build.export_source()
 with zipfile.ZipFile(repo/'dist/unaveraged-source.zip') as archive:
  assert set(archive.namelist())==set(build.SOURCE_FILES)
  for name in ('shared/extension/runtime.js','digit/adapter.js','vqa/adapter.js'):
   assert archive.namelist().count(name)==1
 copied=repo/'dist/source'
 subprocess.run([sys.executable,'-I',str(copied/'shared/build.py'),'--check-inputs'],
                env={k:v for k,v in os.environ.items() if k not in
                     ('PRECISION','MODEL_SOURCE','DIGIT_MODEL_SOURCE','VQA_MODEL_SOURCE')},check=True)
 env=dict(os.environ,CALIBRATION_IMAGES='',CALIBRATION_QUESTIONS='',EVAL_IMAGES='',CALIBRATION_COUNT='1')
 cmd=[sys.executable,'-I',str(copied/'shared/build.py'),'--check-inputs','--precision','w8a8']
 missing=subprocess.run(cmd,env=env,capture_output=True,text=True)
 assert missing.returncode==2 and 'CALIBRATION_IMAGES' in missing.stderr
 images=pathlib.Path(temporary)/'private images';images.mkdir();(images/'one.png').write_bytes(b'preflight only')
 env['CALIBRATION_IMAGES']=str(images)
 for model in ('digit','vqa','all'):subprocess.run(cmd+['--model',model],env=env,check=True)
 env['EVAL_IMAGES']=str(images/'missing')
 invalid=subprocess.run(cmd,env=env,capture_output=True,text=True)
 assert invalid.returncode==2 and 'does not exist' in invalid.stderr
 subprocess.run([sys.executable,'-I',str(copied/'shared/build.py'),'--source'],env=env,check=True)
 assert not (copied/'.venv').exists() and not (copied/'.cache').exists()
`, root], { encoding: 'utf8' });
});
