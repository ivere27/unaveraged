// Pillow's grayscale -> bilinear resize -> float32 normalization contract.
// Downsampling widens the triangle filter; each separable pass rounds to uint8.
const PRECISION = 1 << 22;

function coefficients(source, target) {
  const scale = source / target;
  const support = Math.max(1, scale);
  return Array.from({ length: target }, (_, i) => {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5));
    const end = Math.min(source, Math.trunc(center + support + 0.5));
    const weights = Array.from({ length: end - start }, (_, j) =>
      Math.max(0, 1 - Math.abs((start + j - center + 0.5) / support)));
    const sum = weights.reduce((a, b) => a + b, 0);
    return { start, weights: weights.map((weight) => Math.floor(weight / sum * PRECISION + 0.5)) };
  });
}

export function preprocessPixels({ data, width: sourceWidth, height: sourceHeight }, width = 672, height = 320) {
  if (!Number.isInteger(sourceWidth) || !Number.isInteger(sourceHeight)
      || sourceWidth < 1 || sourceHeight < 1 || data.length !== sourceWidth * sourceHeight * 4) {
    throw new Error('올바른 RGBA 이미지가 필요합니다.');
  }
  const gray = new Uint8Array(sourceWidth * sourceHeight);
  for (let i = 0; i < gray.length; i++) {
    const offset = i * 4;
    gray[i] = (data[offset] * 19595 + data[offset + 1] * 38470
      + data[offset + 2] * 7471 + 32768) >> 16;
  }
  const horizontal = coefficients(sourceWidth, width);
  const intermediate = new Uint8Array(width * sourceHeight);
  for (let y = 0; y < sourceHeight; y++) {
    for (let x = 0; x < width; x++) {
      const { start, weights } = horizontal[x];
      let value = PRECISION / 2;
      for (let k = 0; k < weights.length; k++) value += gray[y * sourceWidth + start + k] * weights[k];
      intermediate[y * width + x] = Math.min(255, Math.floor(value / PRECISION));
    }
  }
  const vertical = coefficients(sourceHeight, height);
  const output = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const { start, weights } = vertical[y];
    for (let x = 0; x < width; x++) {
      let value = PRECISION / 2;
      for (let k = 0; k < weights.length; k++) value += intermediate[(start + k) * width + x] * weights[k];
      const pixel = Math.min(255, Math.floor(value / PRECISION));
      output[y * width + x] = (Math.fround(pixel / 255) - 0.5) / 0.5;
    }
  }
  return output;
}
