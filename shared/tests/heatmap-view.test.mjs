import assert from 'node:assert/strict';
import test from 'node:test';
import '../extension/heatmap-view.js';

const { validateMap, colorize, createOverlay } = globalThis.RECEIPT_HEATMAP;
const map = { kind: 'attention', width: 2, height: 2, values: [0, .1, .3, .6] };

test('renderer accepts real finite grids and rejects malformed or unbounded input', () => {
  assert(validateMap(map));
  assert(validateMap({ ...map, values: new Float32Array(map.values) }));
  for (const malformed of [null, {}, { ...map, kind: 'intensity' }, { ...map, width: 0 },
    { ...map, height: 1.5 }, { ...map, width: 100_000 }, { ...map, values: [0, 1] },
    { ...map, values: [0, .1, NaN, .9] }, { ...map, values: [0, .1, Infinity, .9] },
    { ...map, values: [-.1, .1, .1, .9] }]) {
    assert.equal(validateMap(malformed), false);
    assert.throws(() => colorize(malformed), /attention grid/);
  }
});

test('native-grid raster preserves weights and leaves zero-focus maps transparent', () => {
  const before = structuredClone(map);
  const raster = colorize(map);
  assert.equal(raster.width, 2);
  assert.equal(raster.height, 2);
  assert.equal(raster.pixels.length, 16);
  assert.equal(raster.pixels[3], 0);
  assert(raster.pixels[7] < raster.pixels[11]);
  assert(raster.pixels[11] < raster.pixels[15]);
  assert.equal(raster.pixels[15], 158);
  assert.deepEqual(map, before);
  for (const value of [0, .25]) {
    const uniform = colorize({ ...map, values: Array(4).fill(value) });
    assert([3, 7, 11, 15].every(index => uniform.pixels[index] === 0));
  }
});

test('overlay rasterizes only on new maps, batches layout events, and disposes its observers', () => {
  let paints = 0, observer, disconnected = false, nextFrame = 0;
  const frames = new Map(), listeners = new Map();
  const imageListeners = new Map();
  const rectangle = { left: 20, top: 30, width: 200, height: 100 };
  const body = { append(node) { this.child = node; } };
  const win = {
    ImageData: class { constructor(pixels, width, height) { Object.assign(this, { pixels, width, height }); } },
    ResizeObserver: class { constructor(callback) { observer = callback; } observe() {} disconnect() { disconnected = true; } },
    requestAnimationFrame(fn) { const id = ++nextFrame; frames.set(id, fn); return id; },
    cancelAnimationFrame(id) { frames.delete(id); },
    addEventListener(name, fn) { listeners.set(name, fn); },
    removeEventListener(name) { listeners.delete(name); },
    getComputedStyle() { return { objectFit: 'contain', objectPosition: '50% 50%', opacity: '1' }; },
  };
  const doc = { body, defaultView: win, createElement() {
    return { style: {}, dataset: {}, setAttribute() {}, append(node) { this.child = node; },
      remove() { this.removed = true; }, getContext() { return {
        createImageData(width, height) { return { data: new Uint8ClampedArray(width * height * 4) }; },
        putImageData() { paints++; },
      }; } };
  } };
  const image = { ownerDocument: doc, parentElement: body, naturalWidth: 400, naturalHeight: 200,
    offsetWidth: 200, offsetHeight: 100, isConnected: true,
    getClientRects: () => [rectangle], getBoundingClientRect: () => rectangle,
    addEventListener(name, fn) { imageListeners.set(name, fn); }, removeEventListener(name) { imageListeners.delete(name); } };
  const overlay = createOverlay(image);
  assert(overlay.setMap(map));
  assert.equal(paints, 1);
  assert.equal(body.child.child.width, 2, 'Canvas uses model grid, not screenshot pixels');
  assert.equal(body.child.style.left, '20px');
  assert.equal(body.child.child.style.width, '200px');
  rectangle.left = 50;
  observer(); listeners.get('scroll')(); listeners.get('resize')();
  assert.equal(frames.size, 1, 'A burst of layout events schedules one update');
  const callback = [...frames.values()][0]; frames.clear(); callback();
  assert.equal(frames.size, 0, 'Layout refresh does not start an animation loop');
  assert.equal(body.child.style.left, '50px');
  assert.equal(paints, 1, 'Layout changes never rerasterize attention');
  overlay.setVisible(false);
  assert.equal(body.child.style.display, 'none');
  observer(); assert.equal(frames.size, 0);
  overlay.setVisible(true);
  assert.equal(paints, 1);
  overlay.dispose();
  assert(disconnected);
  assert(body.child.removed);
  assert.equal(listeners.size, 0);
  assert.equal(imageListeners.size, 0);
});
