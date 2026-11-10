import assert from 'node:assert/strict';
import test from 'node:test';
import { copyFile, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Exercise the actual extension bridge with the published pair by default,
// or an explicit release directory for local qualification.
async function runtimeFixture({ graph: customGraph, resources = {} } = {}) {
  const release = process.env.VOLVOXAI_RUNTIME_DIR ? path.resolve(process.env.VOLVOXAI_RUNTIME_DIR)
    : path.dirname(fileURLToPath(import.meta.resolve('volvoxai-lite/min')));
  const root = await mkdtemp(path.join(tmpdir(), 'receipt-runtime-'));
  const server = createServer((request, response) => {
    if (Object.hasOwn(resources, request.url)) {
      response.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      response.end(resources[request.url]);
      return;
    }
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(customGraph ?? {
      format: 'volvox-graph/v1', dimensions: {},
      inputs: { x: { dtype: 'float32', shape: [2] } },
      nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input: 'x' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2] } }, params: {} }],
      outputs: ['y'],
      execution: { stateless: {}, entrypoints: [{ name: 'evaluate', effect: 1,
        allowed_ranges: [1], inputs: ['x'], outputs: ['y'] }] },
    }));
  });
  try {
    // Preserve packaged import paths instead of rewriting the module under test.
    await mkdir(path.join(root, 'shared', 'extension'), { recursive: true });
    await mkdir(path.join(root, 'vendor'));
    await writeFile(path.join(root, 'package.json'), '{"type":"module"}');
    await copyFile(new URL('../extension/runtime.js', import.meta.url), path.join(root, 'shared', 'extension', 'runtime.js'));
    for (const name of ['volvoxai.lite.min.js', 'volvoxai.lite.wasm']) {
      await symlink(path.join(release, name), path.join(root, 'vendor', name));
    }
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    const api = await import(pathToFileURL(path.join(root, 'shared', 'extension', 'runtime.js')));
    assert.equal(typeof api.pb.ExecuteRequest, 'function', 'The runtime must expose the VolvoxAI protobuf API');
    return { ...api, graph: `http://127.0.0.1:${server.address().port}/graph.json`,
      async close() {
        await new Promise((resolve) => server.close(resolve));
        await rm(root, { recursive: true, force: true });
      } };
  } catch (error) {
    await new Promise((resolve) => server.close(resolve));
    await rm(root, { recursive: true, force: true });
    throw error;
  }
}

test('0.8 WASM contexts reuse graphs and preserve named inline outputs for both arithmetic modes', async () => {
  const fixture = await runtimeFixture();
  let sessions;
  try {
    const models = Object.fromEntries(['f32', 'dynamic-i8'].map((packedArithmetic) =>
      [packedArithmetic, { graph: fixture.graph, weights: [], packedArithmetic }]));
    sessions = await fixture.openSessions(models);
    const input = (values) => [{ name: 'x', shape: [2], dtype: fixture.pb.DataType.DATA_TYPE_F32,
      inline: new Uint8Array(Float32Array.from(values).buffer) }];
    for (const name of Object.keys(models)) {
      const first = await sessions.run(name, input([-2, 3]), ['y']);
      const second = await sessions.run(name, input([7, -5]), ['y']);
      assert.equal(first.y.dtype, fixture.pb.DataType.DATA_TYPE_F32);
      assert.deepEqual(first.y.shape, [2n]);
      assert.deepEqual(Array.from(new Float32Array(first.y.inline.buffer)), [0, 3]);
      assert.deepEqual(Array.from(new Float32Array(second.y.inline.buffer)), [7, 0]);
      assert.deepEqual(await sessions.run(name, input([1, 2]), []), {});
      // Feedback reads context outputs even when they were not exported.
      const feedback = await sessions.run(name, [], ['y'], {
        feedback: [{ inputName: 'x', outputName: 'y' }],
      });
      assert.deepEqual(Array.from(new Float32Array(feedback.y.inline.buffer)), [1, 2]);
      const reused = await sessions.run(name, [], ['y'], { reuse: ['x'] });
      assert.deepEqual(Array.from(new Float32Array(reused.y.inline.buffer)), [1, 2]);
      const reset = await sessions.run(name, input([-3, 9]), ['y']);
      assert.deepEqual(Array.from(new Float32Array(reset.y.inline.buffer)), [0, 9]);
    }
    await assert.rejects(sessions.run('missing', input([1, 2]), ['y']), /Unknown model/);
    await assert.rejects(sessions.run('toString', input([1, 2]), ['y']), /Unknown model/);
  } finally {
    await sessions?.close();
    await fixture.close();
  }
});

test('two WASM models load one shared weight URL plus distinct local shards', async () => {
  function safetensors(name, values) {
    const data = Buffer.from(Float32Array.from(values).buffer);
    const descriptor = JSON.stringify({ [name]: { dtype: 'F32', shape: [values.length], data_offsets: [0, data.length] } });
    const header = Buffer.from(descriptor.padEnd(Math.ceil(Buffer.byteLength(descriptor) / 8) * 8, ' '));
    const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
    return Buffer.concat([length, header, data]);
  }
  const graph = { format: 'volvox-graph/v1', dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [2] } },
    nodes: [
      { id: 'shared', opType: 'Add', inputs: { a: 'x', b: 'shared_weight' },
        outputs: { out: { tensor: 'middle', dtype: 'float32', shape: [2] } }, params: {} },
      { id: 'local', opType: 'Add', inputs: { a: 'middle', b: 'local_weight' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2] } }, params: {} },
    ], outputs: ['y'], execution: { stateless: {}, entrypoints: [{ name: 'evaluate', effect: 1,
      allowed_ranges: [1], inputs: ['x'], outputs: ['y'] }] } };
  const fixture = await runtimeFixture({ graph, resources: {
    '/shared.safetensors': safetensors('shared_weight', [2, 3]),
    '/encoder.safetensors': safetensors('local_weight', [1, 1]),
    '/decoder.safetensors': safetensors('local_weight', [-1, -1]),
  } });
  let sessions;
  try {
    const origin = new URL(fixture.graph).origin;
    sessions = await fixture.openSessions(Object.fromEntries(['encoder', 'decoder'].map((role) => [role, {
      graph: fixture.graph, weights: [`${origin}/${role}.safetensors`, `${origin}/shared.safetensors`],
      packedArithmetic: 'dynamic-i8',
    }])));
    const input = [{ name: 'x', shape: [2], dtype: fixture.pb.DataType.DATA_TYPE_F32,
      inline: new Uint8Array(Float32Array.from([10, 20]).buffer) }];
    for (const [role, expected] of [['encoder', [13, 24]], ['decoder', [11, 22]]]) {
      const result = await sessions.run(role, input, ['y']);
      assert.deepEqual(Array.from(new Float32Array(result.y.inline.buffer)), expected);
    }
  } finally {
    await sessions?.close();
    await fixture.close();
  }
});

test('0.8 runtime rejects an unsupported packed arithmetic configuration', async () => {
  const fixture = await runtimeFixture();
  try {
    await assert.rejects(fixture.openSession({ graph: fixture.graph, weights: [], packedArithmetic: 'int4' }),
      /지원하지 않는 모델 연산 형식/);
  } finally {
    await fixture.close();
  }
});

test('optional WASM profiling returns bounded operator timings and releases completed traces', async () => {
  const fixture = await runtimeFixture();
  let session;
  try {
    session = await fixture.openSession({ graph: fixture.graph, weights: [] });
    for (let index = 0; index < 2; index++) {
      const measured = await session.profile(() => session.run([{
        name: 'x', shape: [2], dtype: fixture.pb.DataType.DATA_TYPE_F32,
        inline: new Uint8Array(Float32Array.from([-2, 3]).buffer),
      }], ['y']));
      assert.deepEqual(Array.from(new Float32Array(measured.value.y.inline.buffer)), [0, 3]);
      assert.equal(measured.profile.error, undefined);
      assert(measured.profile.rows.length > 0 && measured.profile.rows.length <= 32);
      assert(measured.profile.rows.some((row) => row.name.includes('ReLU')));
      assert(!JSON.stringify(measured.profile).includes('inline'));
    }
  } finally {
    await session?.close();
    await fixture.close();
  }
});

test('derived inline graphs expose actual attention without changing operators, weights or ordinary outputs', async () => {
  const graph = {
    format: 'volvox-graph/v1', dimensions: {}, inputs: { x: { dtype: 'float32', shape: [1, 3] } },
    nodes: [
      { id: 'attention', opType: 'Softmax', inputs: { input: 'x' },
        outputs: { out: { tensor: 'attention', dtype: 'float32', shape: [1, 3] } }, params: { axis: -1 } },
      { id: 'output', opType: 'Add', inputs: { a: 'attention', b: 'offset' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 3] } }, params: {} },
    ], outputs: ['y'], execution: { stateless: {}, entrypoints: [{ name: 'evaluate', effect: 1,
      allowed_ranges: [1], inputs: ['x'], outputs: ['y'] }] },
  };
  const before = JSON.stringify(graph);
  const header = Buffer.from(JSON.stringify({ offset: { dtype: 'F32', shape: [1, 3], data_offsets: [0, 12] } }));
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
  const weights = Buffer.concat([length, header, Buffer.from(Float32Array.from([1, 2, 3]).buffer)]);
  const fixture = await runtimeFixture({ graph, resources: { '/weights.safetensors': weights } });
  let plain, observed;
  try {
    const options = { graph: fixture.graph, weights: [new URL('/weights.safetensors', fixture.graph)] };
    const exposed = [{ name: 'attention', opType: 'Softmax', dtype: 'float32', shape: [1, 3], axis: -1 }];
    const derived = fixture.exposeGraphOutputs(graph, exposed);
    assert.equal(JSON.stringify(graph), before, 'The original graph is not mutated');
    assert.deepEqual(derived.nodes, graph.nodes);
    assert.deepEqual(derived.outputs, ['y', 'attention']);
    assert.deepEqual(derived.execution.entrypoints[0].outputs, ['y', 'attention']);
    assert.throws(() => fixture.exposeGraphOutputs(graph, [{ ...exposed[0], shape: [3] }]), /어텐션 출력 형식/);
    plain = await fixture.openSession(options);
    observed = await fixture.openSession({ ...options, exposeOutputs: exposed });
    const input = [{ name: 'x', shape: [1, 3], dtype: fixture.pb.DataType.DATA_TYPE_F32,
      inline: new Uint8Array(Float32Array.from([0, 1, 2]).buffer) }];
    const baseline = await plain.run(input, ['y']);
    const result = await observed.run(input, ['y', 'attention']);
    assert.deepEqual(result.y.inline, baseline.y.inline, 'Exposing attention preserves the exact inference output');
    const actual = new Float32Array(result.attention.inline.buffer);
    const denominator = 1 + Math.exp(1) + Math.exp(2);
    actual.forEach((value, index) => assert(Math.abs(value - Math.exp(index) / denominator) < 1e-6));
    assert(Math.abs(actual.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
    assert.deepEqual(Buffer.from(await (await fetch(new URL('/weights.safetensors', fixture.graph))).arrayBuffer()), weights);
  } finally {
    await plain?.close();
    await observed?.close();
    await fixture.close();
  }
});
